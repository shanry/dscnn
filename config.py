# -*- coding: utf-8 -*-

INPUT_DIR = "dataset/OUT"

data_dic ={
    'NYT': {
        'data_root': INPUT_DIR,
        'w2v_path': INPUT_DIR+"/wv.npy",
        'vocab_size': 114043,
        'rel_num': 53
    },
}


class DefaultConfig(object):

    model = 'PCNN_ATT'  # the name of used model, in  <models/__init__.py>
    data = 'NYT'  # SEM NYT FilterNYT

    result_dir = './out'
    data_root = data_dic[data]['data_root']  # the data dir
    w2v_path = data_dic[data]['w2v_path']
    load_model_path = 'checkpoints/model.pth'  # the trained model

    seed = 3435
    batch_size = 128  # batch size
    use_gpu = True  # user GPU or not
    gpu_id = 0
    num_workers = 0  # how many workers for loading data

    max_len = 80 + 2  # max_len for each sentence + two padding
    limit = 60  # the position range <-limit, limit>

    vocab_size = data_dic[data]['vocab_size'] + 1 # padding # vocab + UNK + BLANK
    rel_num = data_dic[data]['rel_num']
    word_dim = 50
    pos_dim = 5
    pos_size = limit * 2 + 2

    norm_emb=True

    num_epochs = 16  # the number of epochs for training
    drop_out = 0.5
    lr = 0.0003  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.0001  # optimizer parameter

    # Conv
    filters = [3]
    filters_num = 230
    sen_feature_dim = filters_num

    rel_dim = filters_num * len(filters)
    rel_filters_num = 100

    print_opt = 'DEF'
    use_pcnn=True


def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)
        data_list = ['data_root', 'w2v_path', 'rel_num', 'vocab_size', 'p1_2v_path', 'p2_2v_path']
        for r in data_list:
            setattr(self, r, data_dic[self.data][r])

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


DefaultConfig.parse = parse
opt = DefaultConfig()