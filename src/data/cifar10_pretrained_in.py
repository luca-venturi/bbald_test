import numpy as np
import torch, torchvision
import pickle
from vgg16_pretrained_in import vgg16

transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)

model = vgg16(pretrained=True)
model.eval()

train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=100, num_workers=4)
test_data_loader = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=4)

x_train = np.array([]) 
y_train = np.array([])
for i, data in enumerate(train_data_loader, 0):
	x, y = data	
	x = model.features_transformation(x).detach().cpu().numpy()
	y = y.cpu().numpy()

	x_train = np.concatenate((x_train, x)) if x_train.size else x
	y_train = np.concatenate((y_train, y)) if y_train.size else y
	print('iter {} done'.format(i))

	if (i+1) % 100 == 0:
		print(x_train.shape, y_train.shape)
		with open('./cifar10_vgg16_train_{}'.format(i), 'wb') as wfile:
			pickle.dump([x_train, y_train], wfile)
		x_train = np.array([])
		y_train = np.array([])

x_test = np.array([]) 
y_test = np.array([])
for i, data in enumerate(test_data_loader, 0):
	x, y = data	
	x = model.features_transformation(x).detach().cpu().numpy()
	y = y.cpu().numpy()

	x_test = np.concatenate((x_test, x)) if x_test.size else x
	y_test = np.concatenate((y_test, y)) if y_test.size else y
	print('iter {} done - x {} - y {}'.format(i, x_test.shape, y_test.shape))

with open('./cifar10_vgg16_test', 'wb') as wfile:
	pickle.dump([x_test, y_test], wfile)
