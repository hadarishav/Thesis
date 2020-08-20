import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import shutil

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class AbuseDataset(Dataset):

  def __init__(self, reviews, targets, c_list, c_num, tokenizer, max_len, context_window, ids):
    self.reviews = reviews
    self.targets = targets
    self.c_list = c_list
    self.c_num = c_num
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.context_window = context_window
    self.ids = ids
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    c=["[PAD]" for i in range(self.context_window)]
    review = str(self.reviews[item])
    target = self.targets[item]
    c_num = self.c_num[item]
    c_list = self.c_list[item]
    idx = self.ids[item]
    n = c_num if c_num<self.context_window else self.context_window
    for i in range(n):
      c[i] = c_list[i]
    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      truncation=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    context_input_ids = []
    context_attention_mask = []
    for i in range(0,self.context_window):
      encoding_context = self.tokenizer.encode_plus(
      c[i],
      add_special_tokens=True,
      truncation=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt')
      context_input_ids.append(encoding_context['input_ids'].flatten())
      context_attention_mask.append(encoding_context['attention_mask'].flatten())

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.float),
      'context_input_ids': torch.stack(context_input_ids),
      'context_attention_masks': torch.stack(context_attention_mask),
      'context_num': c_num,
      'ids': idx
    }

class GeneralAttention(nn.Module):

  def __init__(self, hidden_size, context_window, sparsemax=False):
    super().__init__()
    self.hidden_size = hidden_size
    self.linear = nn.Linear(self.hidden_size, 1)
    self.context_window = context_window
    # self.normaliser = masked_softmax
    
    self.weights = []

  def masked_softmax(self, vector, mask):
    while mask.dim() < vector.dim():
      mask = mask.unsqueeze(1)
    # To limit numerical errors from large vector elements outside the mask, we zero these out.
    result = torch.nn.functional.softmax(vector * mask, dim=-1)
    result = result * mask
    result = result / (
        result.sum(dim=-1, keepdim=True) + 1e-4
    )
    return result

  def forward(self, context, masks, batch_size, device, word_attention = 1):
    if(word_attention):
      # context = torch.cat(context, dim=0)
      weights = self.linear(context).squeeze(-1)
      weights = weights.to(device)
      masks = masks.to(device)
      weights = self.masked_softmax(weights, masks)
      context = torch.bmm(weights.unsqueeze(dim=1), context)

      return context, weights
    else:
      context = torch.cat(context, dim=1)
      context = context.reshape(-1,self.context_window, self.hidden_size)
      weights = self.linear(context).squeeze(-1)
      weights = weights.to(device)
      masks = masks.to(device)
      weights = self.masked_softmax(weights, masks)
      context = torch.bmm(weights.unsqueeze(dim=1), context)

      return context, weights

class MSLELoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.mse = nn.MSELoss(reduction = 'sum')
        
  def forward(self, pred, actual):
    return self.mse(torch.log(pred + 1.00005), torch.log(actual + 1.00005))

class Abuse_lightning(LightningModule):

  def __init__(self, df_train,df_val,df_test, config):
    
    super(Abuse_lightning, self).__init__()
    self.save_hyperparameters()
    self.df_train = df_train
    self.df_val = df_val
    self.df_test = df_test
    self.config = config

    self.n_classes = config['abuse_classes']
    self.max_len = config['max_len']
    self.batch_size = config['batch_size']
    self.max_epochs = config['num_epochs']

    self.PRE_TRAINED_MODEL_NAME = config['PRE_TRAINED_MODEL_NAME']
    self.bert = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
    bert_dropout = config['bert_dropout']
    for layer in self.bert.encoder.layer:
      layer.attention.self.dropout = torch.nn.Dropout(self.bert.config.attention_probs_dropout_prob + bert_dropout)
      layer.output.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob + bert_dropout)
    self.drop = nn.Dropout(p = config['fc_dropout'])
    self.out = nn.Linear(self.bert.config.hidden_size, self.n_classes)
    self.loss = nn.MSELoss().to(self.device)
    self.attention = GeneralAttention(self.bert.config.hidden_size, self.config['context_window'])
    self.attention = self.attention.to(self.device)
  ################################ DATA PREPARATION ############################################

  def __retrieve_dataset(self, train=True, val=True, test=True):

    """ Retrieves task specific dataset """
    self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)

    if train:
      num = self.df_train.context_num.astype(int)
      context_list = []
      for i, k in enumerate(num):
        context = []
        for j in range(k):
          context.append(self.df_train["context"+str(j+1)][i])
        context_list.append(context)
      ds = AbuseDataset(reviews=self.df_train.comment.to_numpy(), targets=self.df_train.Score.to_numpy(), c_list = context_list,
       c_num = self.df_train.context_num.to_numpy(), tokenizer=self.tokenizer,max_len=self.max_len, context_window = self.config['context_window'], ids = self.df_train.idx)
    if test:
      num = self.df_test.context_num.astype(int)
      context_list = []
      for i, val in enumerate(num):
        context = []
        for j in range(val):
          context.append(self.df_test["context"+str(j+1)][i])
        context_list.append(context)
      ds = AbuseDataset(reviews=self.df_test.comment.to_numpy(), targets=self.df_test.Score.to_numpy(), c_list = context_list,
       c_num = self.df_test.context_num.to_numpy(), tokenizer=self.tokenizer,max_len=self.max_len, context_window = self.config['context_window'], ids = self.df_test.idx)
    return ds

  @pl.data_loader
  def train_dataloader(self):
    self._train_dataset = self.__retrieve_dataset(val=False, test=False)
    return DataLoader(dataset=self._train_dataset, batch_size=self.batch_size,num_workers=4, shuffle = True)
    
  @pl.data_loader
  def test_dataloader(self):
    self._test_dataset = self.__retrieve_dataset(train=False, val=False)
    return DataLoader(dataset=self._test_dataset, batch_size=self.batch_size,num_workers=4)

  
  ################################ MODEL AND TRAINING PREPARATION ############################################  
  
  def forward(self, input_ids, attention_mask):
    outputs = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    pooled_output = outputs[0]
    output = self.drop(pooled_output)
    return output
  
  def training_step(self, d, batch_idx):

    if(self.current_epoch > 5):
      for param in self.bert.encoder.parameters():
        param.requires_grad = False

    input_ids = d["input_ids"].to(self.device)
    attention_mask = d["attention_mask"].to(self.device)
    targets = d["targets"].to(self.device)
    context_input_ids = d["context_input_ids"].to(self.device)
    context_attention_masks = d["context_attention_masks"].to(self.device)
    context_num = d['context_num'].to(self.device)
    outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
    out_encoding = []
    t1,t2,t3 = torch.unbind(context_input_ids, dim = 1)
    a1,a2,a3 = torch.unbind(context_attention_masks, dim = 1)
    c = self.forward(input_ids=t1.to(self.device),attention_mask=a1.to(self.device))
    weighted, _ = self.attention.forward(c, a1, self.batch_size, self.device)
    out_encoding.append(weighted)
    c = self.forward(input_ids=t2.to(self.device),attention_mask=a2.to(self.device))
    weighted, _ = self.attention.forward(c, a2, self.batch_size, self.device)
    out_encoding.append(weighted)
    c = self.forward(input_ids=t3.to(self.device),attention_mask=a3.to(self.device))
    weighted, _ = self.attention.forward(c, a3, self.batch_size, self.device)
    out_encoding.append(weighted)
    mask = torch.zeros([input_ids.shape[0], self.config['context_window']])
    for i in range(len(context_num)):
      arr = np.zeros(self.config['context_window'])
      arr[:context_num[i]] = 1
      mask[i] = torch.tensor(arr)
    mask = mask.to(self.device)
    weighted, _ = self.attention.forward(out_encoding, mask, self.batch_size, self.device, word_attention = 0)
    outputs = outputs.mean(dim = 1)
    main_context = outputs.add(weighted.squeeze(dim=1))
    val = self.out(main_context)
    preds = torch.tanh(val)
    loss = self.loss(preds.squeeze(dim = 1), targets)
    p = preds.squeeze(dim=1).to('cpu').detach().numpy()
    t = targets.to('cpu').detach().numpy()
    loss = loss.type(torch.FloatTensor)
    
    return {'prediction': p, 'target': t, 'loss': loss}

  def test_step(self, d, batch_idx):

    input_ids = d["input_ids"].to(self.device)
    attention_mask = d["attention_mask"].to(self.device)
    targets = d["targets"].to(self.device)
    ids = d['ids']
    context_input_ids = d["context_input_ids"].to(self.device)
    context_attention_masks = d["context_attention_masks"].to(self.device)
    context_num = d['context_num'].to(self.device)
    outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
    out_encoding = []
    t1,t2,t3 = torch.unbind(context_input_ids, dim = 1)
    a1,a2,a3 = torch.unbind(context_attention_masks, dim = 1)
    c = self.forward(input_ids=t1.to(self.device),attention_mask=a1.to(self.device))
    weighted, s1_weights = self.attention.forward(c, a1, self.batch_size, self.device)
    out_encoding.append(weighted)
    c = self.forward(input_ids=t2.to(self.device),attention_mask=a2.to(self.device))
    weighted, s2_weights = self.attention.forward(c, a2, self.batch_size, self.device)
    out_encoding.append(weighted)
    c = self.forward(input_ids=t3.to(self.device),attention_mask=a3.to(self.device))
    weighted, s3_weights = self.attention.forward(c, a3, self.batch_size, self.device)
    out_encoding.append(weighted)
    mask = torch.zeros([input_ids.shape[0], self.config['context_window']])
    for i in range(len(context_num)):
      arr = np.zeros(self.config['context_window'])
      arr[:context_num[i]] = 1
      mask[i] = torch.tensor(arr)
    mask = mask.to(self.device)
    weighted, sen_weights = self.attention.forward(out_encoding, mask, self.batch_size, self.device, word_attention = 0)
    outputs = outputs.mean(dim = 1)
    main_context = outputs.add(weighted.squeeze(dim=1))
    val = self.out(main_context)
    preds = torch.tanh(val)
    loss = self.loss(preds.squeeze(dim = 1), targets)
    p = preds.squeeze(dim=1).to('cpu').detach().numpy()
    t = targets.to('cpu').detach().numpy()
    loss = loss.type(torch.FloatTensor)
    
    return {'prediction': p, 'target': t, 'loss': loss, 'ids': ids, 's1_weights': s1_weights, 's2_weights': s2_weights, 's3_weights': s3_weights, 'sen_weights':sen_weights}

  def configure_optimizers(self):

    # para = [{"params":self.bert.parameters(), "lr":self.config['lr'], "weight_decay" : self.config['wd'], "correct_bias":False},
    # {"params":self.attention.parameters(), "lr":self.config['lr']*10, "weight_decay" : self.config['wd'], "correct_bias":False},
    # {"params":self.out.parameters(), "lr":self.config['lr']*10, "weight_decay" : self.config['wd'], "correct_bias":False}]
    optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay= self.config['wd'], correct_bias=False)
    total_steps = len(self.train_dataloader()) * self.max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0, num_training_steps = total_steps )
    # optimizer_att = torch.optim.Adam(self.attention.parameters(),lr=0.001)
    return [optimizer], [scheduler]
    
  def training_epoch_end(self, outputs):

    # called at the end of the training epoch
    # outputs is an array with what you returned in validation_step for each batch
    # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    p = []
    for x in outputs:
      p.extend(x['prediction'])
    t = []
    for x in outputs:
      t.extend(x['target'])
    pear = pearsonr(t,p)
    spear = spearmanr(t,p)
    tau = kendalltau(t,p)
    tensor_pear = torch.tensor(pear[0], device=self.device)
    logs = {'train_loss': avg_loss.item(), 'train_pearson':pear[0], 'train_spearman':spear[0], 'train_kendall':tau[0]}
    print(" Train Pearson {}.Train Spearman {}.Train Kendall {} Train Loss {}".format(pear[0], spear[0], tau[0], avg_loss))

    return {'pearson':tensor_pear, 'spearman':spear[0], 'kendall':tau[0], 'loss': avg_loss, 'log': logs}

  def test_epoch_end(self, outputs):
    # called at the end of the validation epoch
    # outputs is an array with what you returned in validation_step for each batch
    # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    p = []
    ids = []
    sw = []
    sw1 = []
    sw2 = []
    sw3 = []

    for x in outputs:
      p.extend(x['prediction'])
      # test_preds.extend(x['prediction'])
      ids.extend(x['ids'])
      sw.extend(x['sen_weights'])
      sw1.extend(x['s1_weights'])
      sw2.extend(x['s2_weights'])
      sw3.extend(x['s3_weights'])


    t = []
    for x in outputs:
      t.extend(x['target'])
      # test_targets.extend(x['target'])
    with open('testing_preds_convo_words.csv', 'a', encoding = 'utf-8') as f:
      writer = csv.writer(f)
      # writer.writerow(['ID', 'Prediction', 'Target'])
      row = []
      for i,idx in enumerate(ids):
        row.append(idx.item())
        row.append(p[i])
        row.append(t[i])
        # row.append(sw[i].to('cpu').detach().numpy())
        # row.append(sw1[i].to('cpu').detach().numpy())
        # row.append(sw2[i].to('cpu').detach().numpy())
        # row.append(sw3[i].to('cpu').detach().numpy())

        writer.writerow(row)
        row = []
    f.close()

    pear = pearsonr(t,p)
    spear = spearmanr(t,p)
    tau = kendalltau(t,p)
    print("Test hparams: ",self.config['lr'],self.config['fc_dropout'],self.config['bert_dropout'])
    print(" Test Pearson {}.Test Spearman {}.Test Kendall {} Test Loss {}".format(pear[0], spear[0], tau[0], avg_loss))
    return {'pearson':pear[0], 'spearman':spear[0], 'kendall':tau[0], 'loss': avg_loss}

if __name__ == "__main__":
  ctr = 1
  parser = argparse.ArgumentParser(description="Enter args")
  parser.add_argument('--PRE_TRAINED_MODEL_NAME', default="bert-base-cased", type=str)
  parser.add_argument('--batch_size', default=16, type=int)
  parser.add_argument('--max_len', default=200, type=int)
  parser.add_argument('--abuse_classes', default=1, type=int)
  parser.add_argument('--bert_dropout', default=0.2, type=float)
  parser.add_argument('--fc_dropout', default=0.4, type=float)
  parser.add_argument('--num_epochs', default=12, type=int)
  parser.add_argument('--context_window', default=3, type=int)
  parser.add_argument('--lr', default=2e-5, type=float)
  parser.add_argument('--wd', default=1e-4, type=float)
  parser.add_argument('--csv_index', default=1, type=int)

  args = parser.parse_args()
  config = {
    'PRE_TRAINED_MODEL_NAME': args.PRE_TRAINED_MODEL_NAME,
    'batch_size': args.batch_size,
    'max_len': args.max_len,
    'abuse_classes': args.abuse_classes,
    'bert_dropout': args.bert_dropout,
    'fc_dropout': args.fc_dropout,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    'num_epochs': args.num_epochs,
    'lr': args.lr,
    'wd': args.wd,
    'context_window':args.context_window}

  df_train = pd.read_csv('train' + str(args.csv_index) + '.csv')
  # df_val = pd.read_csv("val.csv")
  df_test = pd.read_csv('test' + str(args.csv_index) + '.csv')
  model = Abuse_lightning(df_train,[],df_test, config)
  path = os.path.join(os.getcwd(), 'runs/lightning_logs/version_'+str(ctr)+'/checkpoints/')
  # model.to(device)
  checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    filepath = path,
    verbose=True,
    monitor='loss',
    mode='min')

  # early_stop_callback = EarlyStopping(
  #  monitor='val_loss',
  #  min_delta=0.00,
  #  patience=3,
  #  verbose=False,
  #  mode='min')

  trainer = pl.Trainer(gpus = 4, progress_bar_refresh_rate=0, max_epochs= config['num_epochs'], checkpoint_callback=checkpoint_callback, distributed_backend="ddp") #, distributed_backend="ddp"
  trainer.fit(model)
  # trainer = pl.Trainer(gpus = 1, progress_bar_refresh_rate=0, max_epochs= config['num_epochs'], checkpoint_callback=checkpoint_callback)
  # path = os.path.join(os.getcwd(), 'runs/lightning_logs/version_'+str(ctr)+'/')
  # path = path + os.listdir(path)[0]
  # print(path)
  # model = Abuse_lightning.load_from_checkpoint(path, df_test=df_test)
  # trainer.model = model
  # trainer.test(model)
  # shutil.rmtree("runs", ignore_errors=True)
  # shutil.rmtree("lightning_logs", ignore_errors=True)
  # os.remove('train' + str(args.csv_index) + '.csv')
  # os.remove('test' + str(args.csv_index) + '.csv')
