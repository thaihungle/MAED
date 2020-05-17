How to run experiments:
1. Some of prepared data for each experiment can be found on ./busi_task/data/ folder
2. File busi_task/*_run.py contains code for experiments
3. In busi_task/*_run.py: there are train and test funtions for each task. Just call the appropriate one

How to tune hyper parameters:
1. In each function, hyper parameters are hard-coded
2. Just edit directly

Type of hyper parameters: 
1. Method type (edit in constructer's arguments):
- LSTM seq2seq:  use_mem=False
- DNC: use_mem=True, decoder_mode=True/False, dual_controller=False, write_protect=False
- DC-MANN: use_mem=True, decoder_mode=True, dual_controller=True, write_protect=False
- DCw_MANN: use_mem=True, decoder_mode=True/False, dual_controller=True, write_protect=True
2. Model parameters (edit in constructer's arguments):
- use_emb=True/False: use embedding layer or not
- dual_emb=True/False: if use emedding layer, use one share or two embeddings for encoder and decoder
- hidden_controller_dim: dimension of controller hidden state
3. Memory parameters (if use memory):
- words_count: number of memory slots
- word_size: size of each memory slots
- read_heads: number of reading heads
4. Training parameters:
- batch_size: number of sequence sampled per batch
- iterations: max number of training step
- lm_train=True/False: training by the language model's way (edit in prepare_sample_batch function)
- optimizer: in file dnc.py, function build_loss_function_mask (default is adam)

Notes:
1. The current hyper-parameters are picked by experience from other projects
2. Except from different method types, I have not tried with other hyper-parameters



