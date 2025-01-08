# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-enwik8'
eval_interval = 1000
eval_iters = 200
log_interval = 1000

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'enwik8'
wandb_run_name = 'mini-gpt'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 192
block_size = 512

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.
shared_layers = []

learning_rate = 1e-3
max_iters = 10000
lr_decay_iters = max_iters
min_lr = learning_rate / 10

warmup_iters = 100 # not super necessary potentially
