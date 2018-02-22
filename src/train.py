import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
import datetime
import pdb
import numpy as np
import onmt
import opts
import codecs
from itertools import count
from torchnet import meter

def train_model(train_data, dev_data, model, args):

    args.cuda = args.gpu > -1

    if args.cuda:
        torch.cuda.set_device(args.gpu)

    optimizer = torch.optim.Adam(model.parameters() , lr=args.lr)

    model.train()

    for epoch in range(1, args.epochs+1):

        print("-------------\nEpoch {}:\n".format(epoch))

        train_loss = run_epoch(train_data, True, model, optimizer, args)
        print('Train Cross Entropy loss: {:.6f}'.format( train_loss))
        # print('list parameters', list(model.parameters()))

        dev_loss = run_epoch(dev_data, False, model, optimizer, args)
        print('Dev Cross Entropy loss: {:.6f}'.format( dev_loss))

        # Save model
        torch.save(model, args.save_path)

def run_epoch(data, is_training, model, optimizer, args):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    losses = []
    confusion_matrix = meter.ConfusionMeter(args.num_classes)

    if is_training:
        model.train()
    else:
        model.eval()

    for batch in data_loader:
        # x, y = autograd.Variable(batch['x'], requires_grad=False), autograd.Variable(batch['y'], requires_grad=False)
        x, y = autograd.Variable(batch['x']), autograd.Variable(batch['y'])

        if args.cuda:
            x, y = x.cuda(), y.cuda()

        if is_training:
            optimizer.zero_grad()

        out = model(x[0])
        prediction = torch.max(out.data, 1)[1]
        confusion_matrix.add(prediction, batch['y'].type(torch.LongTensor))

        #Test function compute the accuracy
        #use tensor board
        loss = F.cross_entropy(out, y.long())

        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(loss.cpu().data[0])

    print (confusion_matrix.conf)
    accuracy = np.trace(confusion_matrix.conf)*1.0/np.sum(confusion_matrix.conf)
    print ("Accuracy is, "+str(accuracy))

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss
