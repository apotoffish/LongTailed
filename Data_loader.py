import time
import numpy as np
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
from dataset.TripletDataset import TripletDataset
from dataset.NpairDataset import NpairDataset
from loss.TripletLoss import TripletLoss
from loss.NpairLoss import NpairLoss
from Parameters import Parameters
from Writer import Writer

parameter = Parameters('settings.yml')
parameter.load()
params = parameter.params

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


class Dataset:
    def __init__(self, params, train, transform):
        self.params = params
        self.train = train
        self.transform = transform

        if self.params['dataset'] == 'cifar-10':
            self.dataset = torchvision.datasets.CIFAR10(root=params['data_path'], train=self.train,
                                                        transform=transform)
            if self.params['long_tailed']:
                self.dataset = self.LongTailed()
            if params['criterion'] == 'TripletLoss':
                self.dataset = TripletDataset(self.dataset, transform)
            elif params['criterion'] == 'NpairLoss':
                self.dataset = NpairDataset(self.dataset, transform)
        elif self.params['dataset'] == 'my-image':
            # 暂时
            self.dataset = torchvision.datasets.CIFAR10(root=self.params['data_path'], train=self.train,
                                                        transform=self.transform)
            if self.train:
                if params['criterion'] == 'TripletLoss':
                    self.dataset = TripletDataset(self.dataset, transform)
                elif params['criterion'] == 'NpairLoss':
                    self.dataset = NpairDataset(self.dataset, transform)

    def LongTailed(self):
        # 这里将cifar数据集长尾化处理，暂时按照线性分布处理，之后具体实现不同分布的处理
        dataset = self.dataset
        class_num = self.params['class_num']
        # dataset.data: ndarray
        # dataset.targets: list
        # 10是 不平衡比率imbalance ratio
        '''
        reserved = np.rint(
            np.linspace(len(dataset.data) / (class_num * 10), len(dataset.data) / class_num, class_num)).astype(int)
        '''
        reserved = np.zeros([class_num, 1],dtype=int)
        reserved[0:int(class_num * 0.5)] = len(dataset.data) / (class_num * 10)
        reserved[int(class_num * 0.5):class_num] = len(dataset.data) / class_num
        reserved = len(dataset.data) / class_num - reserved
        remove = []
        for i in range(len(dataset.targets)):
            if reserved[dataset.targets[i]] != 0:
                remove.append(i)
                reserved[dataset.targets[i]] -= 1
        dataset.data = np.delete(dataset.data, remove, axis=0)
        for counter, index in enumerate(remove):
            index = index - counter
            dataset.targets.pop(index)
        return dataset


# train: read data ->
train = Dataset(params, True, transform)
test = Dataset(params, False, transform)
cifar_train = train.dataset
cifar_test = test.dataset
# torchvision.datasets.CIFAR10(root=params['data_path'], train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=params['batch_size'], shuffle=params['isShuffled'])
testloader = torch.utils.data.DataLoader(cifar_test, batch_size=params['batch_size'], shuffle=params['isShuffled'])
dataloader = [trainloader, testloader]

train_size = len(cifar_train)
test_size = len(cifar_test)

net = models.resnet34(pretrained=True)
num_ftrs = net.fc.in_features

criterion = nn.CrossEntropyLoss()
# net = ResNet.ResNet()
if params['criterion'] == 'CrossEntropyLoss':
    net.fc = nn.Linear(num_ftrs, params['class_num'])
    # criterion = nn.CrossEntropyLoss()
elif params['criterion'] == 'TripletLoss':
    # criterion = TripletLoss.TripletLoss(margin=triplet_margin)
    net.fc = nn.Linear(num_ftrs, params['embedding_size'])
    net.classifier = nn.Linear(params['embedding_size'], params['class_num'])
    criterion_triplet = nn.TripletMarginLoss(margin=params['triplet_margin'], p=2)
    # criterion = TripletLoss(params['triplet_margin'])
elif params['criterion'] == 'NpairLoss':
    net.fc = nn.Linear(num_ftrs, params['embedding_size'])
    net.classifier = nn.Linear(params['embedding_size'], params['class_num'])
    criterion_npair = NpairLoss(l2_reg=params['l2_regression'])

if torch.cuda.is_available():
    net = net.cuda()

optimizer = optim.SGD(net.parameters(), lr=params['learning_rate'], momentum=params['momentum'])
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])


def train_model(model, optimizer, scheduler, num_epochs=1):
    since = time.time()
    train_acc = []
    test_acc = []
    losses = []
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        optimizer.step()
        scheduler.step()
        model.train(True)

        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(trainloader):
            if params['criterion'] == 'CrossEntropyLoss':
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            elif params['criterion'] == 'TripletLoss':
                a_in, p_in, n_in, labels, n_labels = data
                a_in, p_in, n_in, labels, n_labels = Variable(a_in), Variable(p_in), Variable(n_in), Variable(
                    labels), Variable(n_labels)
                if torch.cuda.is_available():
                    a_in, p_in, n_in, labels, n_labels = a_in.cuda(), p_in.cuda(), n_in.cuda(), labels.cuda(), n_labels.cuda()

                a_out = model(a_in)
                p_out = model(p_in)
                n_out = model(n_in)
                losst = criterion_triplet(a_out, p_out, n_out)
                # loss = criterion.forward(outputs, labels)

                outputs = model.classifier(a_out)
                # loss = CE + Triplet
                lossc = criterion(outputs, labels)
                loss = losst + lossc
            elif params['criterion'] == 'NpairLoss':
                a_in, p_in, labels = data
                a_in, p_in, labels = Variable(a_in), Variable(p_in), Variable(labels)
                if torch.cuda.is_available():
                    a_in, p_in, labels = a_in.cuda(), p_in.cuda(), labels.cuda()
                a_out = model(a_in)
                p_out = model(p_in)
                outputs = model.classifier(a_out)
                lossn = criterion_npair(a_out, p_out, labels)
                lossc = criterion(outputs, labels)
                loss = lossn + lossc
            else:
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # running_corrects += torch.sum(preds == labels)
            running_corrects += preds.eq(labels.data).sum()
            print('running_loss: ', running_loss)
            # print('running_corrects', running_corrects)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects / train_size

        print('train Loss:{:.4f} Acc:{:.4f}'.format(epoch_loss, epoch_acc))
        print('\n')
        train_acc.append(epoch_acc)
        losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            correct = 0
            for i, data in enumerate(testloader):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                if params['criterion'] == 'CrossEntropyLoss':
                    _, preds = torch.max(outputs.data, 1)
                elif params['criterion'] == 'TripletLoss':
                    a_pred = model.classifier(outputs)
                    _, preds = torch.max(a_pred, 1)
                elif params['criterion'] == 'NpairLoss':
                    outputs = model.classifier(outputs)
                    _, preds = torch.max(outputs.data, 1)

                correct += preds.eq(labels.data).sum()

            print('{} Test Acc:{:.4f}'.format(epoch, correct / test_size))
            test_acc.append(correct / test_size)
            if best_acc < correct / test_size:
                best_acc = correct / test_size
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    params['time_elapsed'] = '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)

    writer = Writer()
    writer.write(params, train_acc, test_acc, losses)
    model.load_state_dict(best_model_wts)
    t = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    torch.save(best_model_wts, 'log/' + t + '.pth')
    return model


net = train_model(net, optimizer, exp_lr_scheduler, num_epochs=params['epochs'])
