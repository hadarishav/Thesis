# Thesis
Code for Conversation modeling in offensive language detection.
There are 4 types of models:
  1. stl_regression: is a simple BERT-base model regression model for offensiveness degree prediction
  2. Conversation based models, where the previous 'n' comments in the conversation are modeled using an attention mechanism. It has following variants:\\
    a. Attention at word level\\
    b. Attention at sentence level\\
    c. Hierarchical attention model\\
    d. An experimental model where subsequent comments are taken as a part of the conversation
