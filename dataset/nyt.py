# -*- coding: utf-8 -*-
from prepro import Instance
from prepro import Instance_Feature


from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import json


class NYTData(Dataset):

    def __init__(self, root_path, train=True):
        if train:
            feature_file = open(root_path+'/train_feature', 'rb')
            print("feature file:{}".format(feature_file))
            features = pickle.load(feature_file)
            bags_file = open(root_path + '/bags_train')
            bags = json.load(bags_file)
            print("type of features:{}".format(type(features)))
            print("len of insts:{}".format(len(features)))


            print('loading train data')
        else:
            feature_file = open(root_path + '/test_feature', 'rb')
            features = pickle.load(feature_file)
            bags_file = open(root_path + '/bags_test')
            bags = json.load(bags_file)
            print('loading test data')

        # self.labels = np.load(path + 'labels.npy')
        self.x = []
        self.labels = []
        for key in bags:
            bag = bags[key]
            x = [[],[],[],[]]
            for indx in bag:
                # feat = Instance_Feature()
                # try:
                #     x.append(features[indx])
                # except IndexError:
                #     print(indx, len(features))
                feat = features[indx]
                x[0].append(feat.sent)
                x[1].append(feat.pf1)
                x[2].append(feat.pf2)
                x[3].append(feat.mask)
            label = features[bag[0]].relation
            self.x.append(x)
            self.labels.append(label)
        print("len of self.x:{}".format(len(self.x)))
        print("len of self.label:{}".format(len(self.labels)))

        self.x = list(zip(self.x, self.labels))

        print('loading finish')

        print('loading finish')

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    # data = NYTLoad('./dataset/NYT/')
    pass