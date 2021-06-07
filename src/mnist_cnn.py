from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor

import mc_dropout

class BayesianNet(mc_dropout.BayesianModule):
	def __init__(self, num_classes):
		super().__init__(num_classes)

		self.axis_dim = 28
		self.kernel_size = 5
		self.n_channels = 16
		self.conv = nn.Conv2d(1, self.n_channels, self.kernel_size, padding=0)
		newdim = self.axis_dim - self.kernel_size +1

		self.pool_kernel, self.stride = 2, 2
		self.pool = nn.MaxPool2d(self.pool_kernel, stride=self.stride)
		newdim = (newdim - self.pool_kernel) // self.stride + 1

		self.in_size = self.n_channels * (newdim ** 2)

		self.fc1 = nn.Linear(self.in_size, 20)
		self.fc1_drop = mc_dropout.MCDropout()
		self.fc2 = nn.Linear(20, 20)
		self.fc2_drop = mc_dropout.MCDropout()
		self.fc3 = nn.Linear(20, num_classes)
		self.fc3_drop = mc_dropout.MCDropout()

	def mc_forward_impl(self, input: Tensor):
		input = input.view(-1, 1, self.axis_dim, self.axis_dim)
		input = F.relu(self.conv(input))
		input = self.pool(input).view(-1, self.in_size)

		input = F.relu(self.fc1_drop(self.fc1(input)))
		input = F.relu(self.fc2_drop(self.fc2(input)))
		input = F.relu(self.fc3_drop(self.fc3(input)))
		input = F.log_softmax(input, dim=1)

		return input
