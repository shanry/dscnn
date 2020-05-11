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
        print('{} Epoch {}/{}: train loss: {};'.format(now(), epoch + 1, opt.num_epochs, total_loss))

    return


def preidct(model, data):
    return


if __name__ == "__main__":
    # fire.Fire()
    train()