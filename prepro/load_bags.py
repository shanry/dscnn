"""
    load bags from dump file, generate bags feature

"""
from .tool import get_relation2id
from .get_bags import Instance
import numpy as np
import json
import pickle

from collections import Counter

MAX_POS = 50

DATA_DIR = "D:/WSL/Intra-Bag-and-Inter-Bag-Attentions/NYT_data"
OUT_DIR = "D:/WSL/output"
we_filename = DATA_DIR + "/vec.bin"
npy_filename = OUT_DIR + "/wv.npy"
w2id_filename = OUT_DIR + "/word2id.json"

train_filename = DATA_DIR + "/train.txt"
test_filename = DATA_DIR + "/test.txt"

relation2id = get_relation2id(OUT_DIR)
print(relation2id)

def pf(pos, ht, max_pf):
    pe = pos - ht
    if pe > 0:
        pe = min(pe, max_pf)
    if pe < 0:
        pe = max(pe, -max_pf)
    return pe


class Instance_Feature(object):
    def __init__(self, inst: Instance, max_len = 80, max_pos = 50, pad_num=0):
        self.h_id_fb = inst.h_id_fb
        self.t_id_fb = inst.h_id_fb
        self.h = inst.h
        self.s = inst.s
        self.pos_h = inst.pos_h  # + 1
        self.pos_t = inst.pos_t  # + 1
        # self.sent = sentence
        # try:
        #     self.relation = relation2id[inst.relation]
        # except KeyError:
        #     print(inst.relation)
        #     exit(0)
        self.relation = relation2id.get(inst.relation, 0)
        self.max_len = max_len
        self.max_pos = max_pos
        self.pad_num = pad_num
        self.get_feature(inst, self.pad_num)
        assert len(self.sent) == self.max_len + 2
        assert len(self.pf1)  == self.max_len + 2
        assert len(self.pf2)  == self.max_len + 2
        assert len(self.mask) == self.max_len + 2
        assert self.mask[0] == 1
        assert self.mask[-1] == 3, (self.mask, self.pos_h, self.pos_t)
        assert 2 in self.mask

    def get_feature(self, inst: Instance, pad_id):
        self.sent = [pad_id]
        self.pf1 = [0]
        self.pf2 = [0]
        self.mask = [1]
        sentence = inst.v_sent
        if len(sentence) <= self.max_len:
            if self.pos_h == self.pos_t:
                if self.pos_t < self.max_len-1:
                    self.pos_t += 1
                else:
                    self.pos_h += 1
            for i, widx in enumerate(sentence):
                self.sent.append(widx)
                self.pf1.append(pf(i, self.pos_h, self.max_pos)+self.max_pos+1)
                self.pf2.append(pf(i, self.pos_t, self.max_pos)+self.max_pos+1)
                if i <= min(self.pos_h, self.pos_t):
                    self.mask.append(1)
                elif i <= max(self.pos_h, self.pos_t):
                    self.mask.append(2)
                else:
                    self.mask.append(3)
            for i in range(len(sentence), self.max_len):
                self.sent.append(pad_id)
                self.pf1.append(pf(i, self.pos_h, self.max_pos) + self.max_pos+1)
                self.pf2.append(pf(i, self.pos_t, self.max_pos) + self.max_pos+1)
                self.mask.append(3)
        else:
            if abs(self.pos_h - self.pos_t)+1 > self.max_len:
                start = min(self.pos_h, self.pos_t)
                for i in range(start, start+self.max_len):
                    self.sent.append(sentence[i])
                    self.pf1.append(pf(i, self.pos_h, self.max_pos) + self.max_pos+1)
                    self.pf2.append(pf(i, self.pos_t, self.max_pos) + self.max_pos+1)
                    self.mask.append(2)
                self.mask[0] = 1
                self.mask[-1] = 3
            else:
                start = min(self.pos_h, self.pos_t)
                end = max(self.pos_h, self.pos_t)
                while end-start+1 < self.max_len:
                    if start>0:
                        start -= 1
                    if end-start+1 == self.max_len:
                        break
                    if end<len(sentence)-1:
                        end += 1
                    if end-start+1 == self.max_len:
                        break
                if self.pos_h == self.pos_t:
                    if self.pos_t < end:
                        self.pos_t += 1
                    else:
                        self.pos_h += 1
                for i in range(start, end+1):
                    self.sent.append(sentence[i])
                    self.pf1.append(pf(i, self.pos_h, self.max_pos) + self.max_pos+1)
                    self.pf2.append(pf(i, self.pos_t, self.max_pos) + self.max_pos+1)
                    if i <= min(self.pos_h, self.pos_t):
                        self.mask.append(1)
                    elif i <= max(self.pos_h, self.pos_t):
                        self.mask.append(2)
                    else:
                        self.mask.append(3)
        self.sent.append(pad_id)
        self.pf1.append(0)
        self.pf2.append(0)
        self.mask.append(3)


def load_inst(inst_filename):
    print("inst_filename:{}".format(inst_filename))
    inst_file = open(inst_filename, 'rb')
    inst_list = pickle.load(inst_file)
    print("len of inst_list:{}".format(len(inst_list)))
    inst_features = []
    relations = set()
    unknown_relation = 0
    for inst in inst_list:
        relations.add(inst.relation)
        if inst.relation not in relation2id:
            unknown_relation += 1
    print("unknown_relation:{}".format(unknown_relation))
    print("len of relations:{}".format(len(relations)))
    print(relations)
    for rel in relations:
        if rel not in relation2id:
            print(rel)
    for inst in inst_list:
        inst_features.append(Instance_Feature(inst))
    if 'train' in inst_filename:
        feature_filename = OUT_DIR + "/train_feature"
    else:
        feature_filename = OUT_DIR + "/test_feature"
    feature_file = open(feature_filename, 'wb')
    print("len of inst_features:{}".format(len(inst_features)))
    pickle.dump(inst_features, feature_file)
    feature_file.close()


def load_bags(bags_filename):
    print("bags_filename:{}".format(bags_filename))
    bags_file = open(bags_filename)
    bags = json.load(bags_file)
    print("len of bags :{}".format(len(bags)))
    counter = Counter()
    inst_count = 0
    for trip in bags:
        counter[len(bags[trip])] += 1
        inst_count += len(bags[trip])
    print(counter)
    print(inst_count)


if __name__ == "__main__":

    train_inst_filename = OUT_DIR + "/inst_" + "train"
    test_inst_filename = OUT_DIR + "/inst_" + "test"
    train_bags_filename = OUT_DIR + "/bags_" + "train"
    test_bags_filename = OUT_DIR + "/bags_" + "test"

    # train_inst_file = open(train_inst_filename, 'rb')
    # train_inst_list = pickle.load(train_inst_file)
    # print("len of train inst_list:{}".format(len(train_inst_list)))
    # train_inst_features = []
    # for inst in train_inst_list:
    #     train_inst_features.append(Instance_Feature(inst))
    #
    # train_bags_file = open(train_bags_filename)
    # train_bags = json.load(train_bags_file)
    # print("len of train_bags :{}".format(len(train_bags)))
    #
    # test_inst_file = open(test_inst_filename, 'rb')
    # test_inst_list = pickle.load(test_inst_file)
    # print("len of test inst_list:{}".format(len(test_inst_list)))
    # test_inst_features = []
    # for inst in test_inst_list:
    #     test_inst_features.append(Instance_Feature(inst))
    #
    #
    # test_bags_file = open(test_bags_filename)
    # test_bags = json.load(test_bags_file)
    # print("len of test_bags :{}".format(len(test_bags)))

    load_inst(train_inst_filename)
    load_bags(train_bags_filename)

    load_inst(test_inst_filename)
    load_bags(test_bags_filename)












