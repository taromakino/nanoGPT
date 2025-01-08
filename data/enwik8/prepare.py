"""
Prepare the enwik8 dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import zipfile
from io import BytesIO

N_TOTAL = 100000000
N_TRAIN = 90000000
N_VAL = 5000000
N_VOCAB = 256
BLOCK_SIZE = 512
PAD_ID = 32

# download the enwik8 dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://mattmahoney.net/dc/enwik8.zip'
    response = requests.get(data_url)
    with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
        f_bytes = zip_ref.read('enwik8')
        with open(input_file_path, 'wb') as f:
            f.write(f_bytes)

with open(input_file_path, 'rb') as f:
    data = list(f.read())  # converts bytes to list of integers
print(f"length of dataset in characters: {len(data):,}")
assert len(data) == N_TOTAL

# get all the unique characters that occur in this text
bytes = sorted(list(set(data)))
vocab_size = len(bytes)
print(f"vocab size: {vocab_size:,}")
assert vocab_size == 205

# create a mapping from nonconsecutive to consecutive integers
byte_to_idx = {byte: idx for idx, byte in enumerate(bytes)}
idx_to_byte = {idx: byte for idx, byte in enumerate(bytes)}
def encode(bytes):
    return [byte_to_idx[c] for c in bytes]
def decode(idxs):
    return ''.join([idx_to_byte[i] for i in idxs])

data = encode(data)

# create the train and test splits
PAD = [PAD_ID] * BLOCK_SIZE
train_ids = data[:N_TRAIN]
val_ids = data[N_TRAIN:N_TRAIN + N_VAL]
test_ids = PAD + data[N_TRAIN + N_VAL:]

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': N_VOCAB,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)