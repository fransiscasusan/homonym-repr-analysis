import torch
import torch.nn.functional as F
import torch.nn as nn

'''
Building a one layer homonym classifier with a linear layer and a softmax layer.
'''

# Depending on arg, build dataset
def get_model(args):
    print("\nBuilding model...")

    return FNN(args)


class FNN(nn.Module):

    def __init__(self, args):
        super(FNN, self).__init__()

        self.args = args
        self.lin = nn.Linear(args.replength, args.num_classes)

    def forward(self, x):
        #x = F.relu(self.lin(x)) #dropout
        x = F.softmax(self.lin(x))

        return x