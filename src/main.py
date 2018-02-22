import argparse
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import model as model_utils
import train as train_utils
import data as data_utils
import os
import torch
import datetime
import cPickle as pickle
import pdb
import opts
import data

parser = argparse.ArgumentParser(description='classifier arguments')
# homonym
parser.add_argument('--homonym', type=str, default='and', help='the analyzed word')
# learning
parser.add_argument('--num_classes', type=int, default=2, help='number of possible senses')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0.001, help='initial constant for regularization loss [default: 0.001]')
parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training [default: 64]')
parser.add_argument('-beam_size',  type=int, default=5, help='Beam size')
# data loading
parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='num workers for data loader')
# parser.add_argument('--all_data', nargs='?', type=str, default=None, help='path to all data')
# model
parser.add_argument('--model', nargs="?", type=str, default='dan', help="The LSTM encoder model name")
# device
parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
parser.add_argument('--train', action='store_true', default=False, help='enable train')
parser.add_argument('--gpu', type=int, default=-1, help="Device to run on")
# task
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
parser.add_argument('--save_path', type=str, default="model.pt", help='Path where to dump model')


args = parser.parse_args()

if __name__ == '__main__':
    # update args and print

    opt = parser.parse_args()
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    train_data, dev_data, test_data, replength, classes = data_utils.load_dataset(args.homonym, opt, dummy_opt)
    args.replength = replength
    args.num_classes = len(classes)

    # model
    if args.snapshot is None:
        classifier = model_utils.get_model(args)
    else :
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            classifier = torch.load(args.snapshot)
        except :
            print("Sorry, This snapshot doesn't exist."); exit()
    print(classifier)

    # train
    if args.train :
		train_utils.train_model(train_data, dev_data, classifier, args)
