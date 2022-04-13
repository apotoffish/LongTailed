from Writer import Writer
from Parameters import Parameters
from torchvision import models
from loss.NpairLoss import NpairLoss
import torch
import numpy as np

a = torch.tensor(np.array([[1.0], [2.0], [3.0]]))
b = torch.tensor(np.array([[1.02], [1.98], [3.05]]))
labels = torch.tensor(np.array([0, 1, 2]))
creterion = NpairLoss()
loss = creterion(a, b, labels)
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

