from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor

import mc_dropout


class BayesianNet(mc_dropout.BayesianModule):
    def __init__(self, num_classes):
        super().__init__(num_classes)

        self.input_size = 512 * 7 * 7
        self.fc1 = nn.Linear(self.input_size, 512)
        self.fc1_drop = mc_dropout.MCDropout()
        self.fc2 = nn.Linear(512, 20)
        self.fc2_drop = mc_dropout.MCDropout()
        self.fc3 = nn.Linear(20, 10)
        self.fc3_drop = mc_dropout.MCDropout()

    def mc_forward_impl(self, input: Tensor):
        input = input.view(-1, self.input_size)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = F.relu(self.fc2_drop(self.fc2(input)))
        input = F.relu(self.fc3_drop(self.fc3(input)))
        input = F.log_softmax(input, dim=1)

        return input
