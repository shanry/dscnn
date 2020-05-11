"""
    get bags from train.txt and test.txt

"""
from .tool import get_relation2id


import numpy as np
import json
import pickle

MAX_POS = 50

DATA_DIR = "D:/WSL/Intra-Bag-and-Inter-Bag-Attentions/NYT_data"
OUT_DIR = "D:/WSL/output"


class Instance(object):
    def __init__(self, h_id, t_id, h_s, t_s, r, sentence, pos_h, pos_t, word2id_map):
        self.h_id_fb = h_id
        self.t_id_fb = t_id
        self.h = h_s
        self.s = t_s
        # self.sent = sentence
        self.relation = r
        self.vecs(sentence, word2id_map)
        self.pos_h = pos_h  # + 1
        self.pos_t = pos_t  # + 1

    def vecs(self, sentence, word2id_map):
        self.v_sent = []  # not padding in the beginning
        for word in sentence:
            word_id = 0
            if word in word2id_map:
                word_id = word2id_map[word]
            self.v_sent.append(word_id)
        # print(sentence)
        # print(self.v_sent)
        # exit(0)
        # self.v_sent.append(-1)  # padding in the ending


def get_bags(filename, outfilename):
    noht_id = []
    noht = set()
    nofinds = []
    max_len = 0
    max_ht_dist = 0
    min_ht_dist = 0
    long_sentlen = 0
    samepos = 0
    inst_list = []

    line_set = set()
    sent_set = set()
    # with open(test_filename) as f:
    with open(filename) as f:
        for i, line in enumerate(f):
            line_set.add(line)
        for i, line in enumerate(line_set):
            hpos = 0  # pseudo posi
            tpos = 0  # pseudo posi
            find_h = False
            find_t = False
            splits = line.strip().split()
            max_len = max(max_len, len(splits) - 5 - 1)
            if len(splits) - 5 > 80:
                long_sentlen += 1
            assert splits[0].startswith("m.")
            assert splits[1].startswith("m.")
            assert splits[4].startswith("/") or splits[4] == "NA", "{}:{}".format(i, line)
            assert splits[-1] == "###END###"
            for i in range(5, len(splits) - 1):
                if splits[i] == splits[2] and not find_h:
                    hpos = i - 5
                    find_h = True
                if splits[i] == splits[3] and not find_t:
                    tpos = i - 5
                    find_t = True
            if not find_h or not find_t:
                noht_id.append((i, line))
                noht.add("  ".join((splits[2], splits[3])))
                nofinds.append((find_h, find_t))
            # global max_ht_dist
            max_ht_dist = max(abs(hpos - tpos), max_ht_dist)
            min_ht_dist = min(abs(hpos - tpos), max_ht_dist)
            if hpos == tpos:
                samepos += 1
                print(hpos, tpos)
            inst = Instance(splits[0], splits[1], splits[2], splits[3], splits[4],
                            splits[5:-1], hpos, tpos, word2id_map=word2id)
            inst_list.append(inst)
            sent_set.add(" ".join(splits[5:-1]))  # + splits[0]+splits[1]

    print("len of hoht_id:{}".format(len(noht_id)))
    # for j in range(10):
    #     print(noht_id[j][0], noht_id[j][1])
    print("len of noht:{}".format(len(noht)))
    # print(noht)
    print("number of distinct lines:{}".format(len(line_set)))
    print("number of distinct sents:{}".format(len(sent_set)))

    print("max_len:{}".format(max_len))
    print("max_ht_dist:{}".format(max_ht_dist))
    print("min_ht_dist:{}".format(min_ht_dist))
    print("same position for h and t: {}".format(samepos))
    print("long:{}".format(long_sentlen))
    print(nofinds)

    bags = {}
    for i, inst in enumerate(inst_list):
        key = " ".join((inst.h_id_fb, inst.t_id_fb, inst.relation))
        if key not in bags:
            bags[key] = [i]
        else:
            bags[key].append(i)

    print("number of sentences:{}".format(len(inst_list)))
    print("number of bags:{}".format(len(bags)))

    inst_list_filename = OUT_DIR + "/inst_" + outfilename
    inst_list_file = open(inst_list_filename, 'wb')
    pickle.dump(inst_list, inst_list_file)
    inst_list_file.close()

    # bags_filename = OUT_DIR + "/bags_" + outfilename
    # bags_file = open(bags_filename, 'w')
    # json.dump(bags, bags_file)
    # bags_file.close()
    return


if __name__ == "__main__":

    npy_filename = OUT_DIR + "/wv.npy"
    w2id_filename = OUT_DIR + "/word2id.json"

    train_filename = DATA_DIR + "/train.txt"
    test_filename = DATA_DIR + "/test.txt"

    w2id_file = open(w2id_filename)
    word2id = json.load(w2id_file)
    print("len of word2id:{}".format(len(word2id)))
    w2id_file.close()
    re2id = get_relation2id(DATA_DIR)
    print(re2id)

    get_bags(train_filename, "train")
    get_bags(test_filename, "test")


