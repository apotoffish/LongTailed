from Writer import Writer
from Parameters import Parameters
from torchvision import models
import torch
import numpy as np

numpy_array= np.array([1,2,3])
torch_tensor1 = torch.from_numpy(numpy_array)
torch_tensor2 = torch.tensor(1.1)
print(torch_tensor2)
print(torch_tensor2.item())
torch_tensor3 = torch.tensor(numpy_array)

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

