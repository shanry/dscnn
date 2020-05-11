"""
    get word embedding from .bin and generate .npy file

"""
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

import json

DATA_DIR = "D:/WSL/Intra-Bag-and-Inter-Bag-Attentions/NYT_data"
OUT_DIR = "D:/WSL/output"
we_file = DATA_DIR + "/vec.bin"
npy_file = OUT_DIR + "/wv.npy"
w2id_file = OUT_DIR + "/word2id.json"

wv = KeyedVectors.load_word2vec_format(we_file, binary=True)
wv.init_sims(True)  # L2 normalization
print("type:{}".format(type(wv)))
# wv.save_word2vec_format('w.txt', binary=False)
np.save(npy_file, wv.vectors)

word2id = {"UNK": 0}
i = 0
for i, word in enumerate(wv.vocab):
    word2id[word] = i+1

jf = open(w2id_file, 'w')
json.dump(word2id, jf)
