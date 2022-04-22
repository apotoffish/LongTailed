import os

from torchvision.transforms import transforms

from Writer import Writer
from Parameters import Parameters
from torchvision import models
from loss.NpairLoss import NpairLoss
import torch
from torch.utils import data
import numpy as np
import json
import cv2 as cv

f = open('label.json', 'rb')
label_data = json.load(f)
print(label_data.keys())

counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
frames = set()
for i in range(len(label_data['train_dataset'])):
    n = label_data['train_dataset'][i]['label']
    print(str(i)+' '+str(n))
    counts[n] += 1

for i in range(len(label_data['test_dataset'])):
    n = label_data['test_dataset'][i]['label']
    print(str(i) + ' ' + str(n))
    counts[n] += 1

print(counts)
'''a = torch.tensor(np.array([[1.0], [2.0], [3.0]]))
b = torch.tensor(np.array([[1.02], [1.98], [3.05]]))
labels = torch.tensor(np.array([0, 1, 2]))
creterion = NpairLoss()
loss = creterion(a, b, labels)'''
'''
numpy_array= np.array([1,2,3])
torch_tensor1 = torch.from_numpy(numpy_array)
torch_tensor2 = torch.tensor(1.1)
print(torch_tensor2)
print(torch_tensor2.item())
torch_tensor3 = torch.tensor(numpy_array)
'''
'''writer = Writer()

settings = {"a":1, "b":'asb'}
train_accuracy = [1,2,3]
test_accuracy = [4,5,6]
losses = [7,8,9]
writer.write(settings,train_accuracy, test_accuracy, losses)'''

'''parameter = Parameters('settings.yml')
parameter.load()

print(parameter.params.items())'''

'''net = models.resnet34(pretrained=True)
print(net)'''
