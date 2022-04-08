import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms
import torch.optim as optim

# import LeNet
# import ResNet
# import TripletLoss

import yaml

with open(r'settings.yml','rb') as f:
    y = yaml.safe_load(f)


data_path = y['default']['data_path']
my_batch_size = y['default']['batch_size']
isShuffle = y['default']['shuffle']
out_class = y['default']['out_class']
learning_rate = y['default']['lr']
my_momentum = y['default']['momentum']
my_step_size = y['default']['step_size']
my_gamma = y['default']['gamma']
my_epochs = y['default']['epochs']
my_criterion = y['default']['criterion']
# triplet_margin = y['loss']['triplet_margin']

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

cifar_train = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                           transform=transform)
cifar_test = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                          transform=transform)

trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=my_batch_size, shuffle=isShuffle)
testloader = torch.utils.data.DataLoader(cifar_test, batch_size=my_batch_size, shuffle=isShuffle)
dataloader = [trainloader, testloader]

train_size = len(cifar_train)
test_size = len(cifar_test)
# 有GPU
model_ft = models.resnet34(pretrained=True)
num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, num_ftrs)

# net = ResNet.ResNet()
if my_criterion == 'CrossEntropyLoss':
    model_ft.fc = nn.Linear(num_ftrs, out_class)
    criterion = nn.CrossEntropyLoss()
elif my_criterion == 'TripletLoss':
    #criterion = TripletLoss.TripletLoss(margin=triplet_margin)
    model_ft.fc.classifier = nn.Linear(num_ftrs, out_class)
    # criterion = nn.TripletMarginLoss()
    criterion = TripletLoss.TripletLoss()
else:
    model_ft.fc.classifier = nn.Linear(num_ftrs, out_class)
    criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=my_momentum)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=my_step_size, gamma=my_gamma)


def train_model(model, optimizer, scheduler, num_epochs=1):
    since = time.time()

    log = open('log.txt','w')
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        optimizer.step()
        scheduler.step()
        model.train(True)

        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(testloader):
            inputs, labels = data

            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            if my_criterion == 'CrossEntropyLoss':
                loss = criterion(outputs, labels)
            elif my_criterion == 'TripletLoss':
                loss = criterion.forward(outputs, labels)
            else:
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs.data, 1)  # ？
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #running_corrects += torch.sum(preds == labels)
            running_corrects += preds.eq(labels.data).sum()
            print('running_loss: ', running_loss)
            #print('running_corrects', running_corrects)

        epoch_loss = running_loss / test_size
        epoch_acc = running_corrects / test_size

        print('train Loss:{:.4f} Acc:{:.4f}'.format(epoch_loss, epoch_acc))
        print('\n')
        log.write('train Loss:{:.4f} Acc:{:.4f}\n'.format(epoch_loss, epoch_acc))

        model.eval()
        with torch.no_grad():
            correct=0
            for i,data in enumerate(testloader):
                inputs,labels = data
                inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                correct += preds.eq(labels.data).sum()

            print('{} Test Acc:{:.4f}'.format(epoch, correct/test_size))
            if best_acc < correct/test_size:
                best_acc = correct/test_size
        '''
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in [0, 1]:  # 0:train 1:valid
            if phase == 0:
                optimizer.step()
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloader[phase]:
                inputs, labels = data

                inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 0:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                print('running_loss: ', running_loss)
                #print('running_corrects', running_corrects)

            if phase == 0:
                epoch_loss = running_loss / labels.size(0)
                epoch_acc = running_corrects / labels.size(0)
            else:
                epoch_loss = running_loss / labels.size(0)
                epoch_acc = running_corrects / labels.size(0)

            print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 1 and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print('\n')
    '''
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    log.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    log.write('Best val Acc: {:.4f}\n'.format(best_acc))
    log.close()
    model.load_state_dict(best_model_wts)
    return model


model_ft = train_model(model_ft, optimizer, exp_lr_scheduler, num_epochs=my_epochs)
'''
print("Start Training...")
for epoch in range(30):
    loss100 = 0.0

    for i, data in enumerate(trainloader):
        inputs, labels = data
        # inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss100 += loss.item()
        if i % 100 ==99:
            print('[Epoch %d, Batch %5d]  loss: %.3f' %
                  (epoch+1 ,i+1, loss100/100))
            loss100 = 0.0

print("Training Done.")

dataiter = iter(testloader)

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        # images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' %
      (100*correct/total))
'''