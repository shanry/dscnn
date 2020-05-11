from prepro import Instance_Feature

from utils import now
from config import opt
import models
import dataset

import torch
from torch.utils.data import DataLoader
from torch import optim

import fire


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def train(**kwargs):
    # kwargs.update({'model': 'PCNN_ATT'})
    # opt.parse(kwargs)
    # opt.use_gpu = False
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = getattr(models, 'PCNN_ATT')(opt)
    print(model.model_name)
    if opt.use_gpu:
        model.cuda()
    # loading data
    DataModel = getattr(dataset, opt.data + 'Data')
    train_data = DataModel(opt.data_root, train=True)
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                                   collate_fn=collate_fn)

    test_data = DataModel(opt.data_root, train=False)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                  collate_fn=collate_fn)
    print('{} train data: {}; test data: {}'.format(now(), len(train_data), len(test_data)))

    # criterion and optimizer
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)

    for epoch in range(opt.num_epochs):
        total_loss = 0
        for idx, (data, label_set) in enumerate(train_data_loader):

            # label = [l[0] for l in label_set]
            label = [l for l in label_set]

            optimizer.zero_grad()
            model.batch_size = opt.batch_size
            loss = model(data, label)
            if opt.use_gpu:
                label = torch.LongTensor(label).cuda()
            else:
                label = torch.LongTensor(label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch > 2:
            # true_y, pred_y, pred_p= predict(model, test_data_loader)
            # all_pre, all_rec = eval_metric(true_y, pred_y, pred_p)
            pred_res, p_num = predict_var(model, test_data_loader)
            all_pre, all_rec = eval_metric_var(pred_res, p_num)

            last_pre, last_rec = all_pre[-1], all_rec[-1]
            print('test: {} Epoch {}/{}: train loss: {}; test precision: {}, test recall {}'.format(now(), epoch + 1,
                                                                                              opt.num_epochs,
                                                                                              total_loss, last_pre,
                                                                                  last_rec))
        else:
            print('train {} Epoch {}/{}: train loss: {};'.format(now(), epoch + 1, opt.num_epochs, total_loss))

    return


def preidct(model, data):
    return


def predict_var(model, test_data_loader):
    '''
    Apply the prediction method in  Lin 2016
    '''
    model.eval()

    res = []
    true_y = []
    for idx, (data, labels) in enumerate(test_data_loader):
        out = model(data)
        true_y.extend(labels)
        if opt.use_gpu:
            #  out = map(lambda o: o.data.cpu().numpy().tolist(), out)
            out = out.data.cpu().numpy().tolist()
        else:
            #  out = map(lambda o: o.data.numpy().tolist(), out)
            out = out.data.numpy().tolist()

        for r in range(1, opt.rel_num):
            for j in range(len(out[0])):
                res.append([labels[j], r, out[r][j]])

        #  if idx % 100 == 99:
            #  print('{} Eval: iter {}'.format(now(), idx))
        exit(0)

    model.train()
    positive_num = len([i for i in true_y if i[0] > 0])
    return res, positive_num


def eval_metric_var(pred_res, p_num):
    '''
    Apply the evalation method in  Lin 2016
    '''

    pred_res_sort = sorted(pred_res, key=lambda x: -x[2])
    correct = 0.0
    all_pre = []
    all_rec = []

    for i in range(2000):
        true_y = pred_res_sort[i][0]
        pred_y = pred_res_sort[i][1]
        for j in true_y:
            if pred_y == j:
                correct += 1
                break
        precision = correct / (i + 1)
        recall = correct / p_num
        all_pre.append(precision)
        all_rec.append(recall)

    print("positive_num: {};  correct: {}".format(p_num, correct))
    return all_pre, all_rec


if __name__ == "__main__":
    # fire.Fire()
    train()