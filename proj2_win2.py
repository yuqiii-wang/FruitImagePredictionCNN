#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:05:49 2018

@author: yuqi
"""

# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use('agg') # to use non-GUI backend that can run on cloud
import matplotlib.pyplot as plt

class Vanilla_net(torch.nn.Module):
    # this is for test
    def __init__(self, num_classes=18):
        torch.nn.Module.__init__(self)
        self.fc = torch.nn.Linear(224*224*3, num_classes)
        
    def forward(self, X):
        X = X.view(-1, 224*224*3)
        X = self.fc(X)
        return X  

class Bilinear_CNN(torch.nn.Module):
    def __init__(self, num_classes=18):
        torch.nn.Module.__init__(self)
        
        # only to have some custimizations on alex
        self.alex = torchvision.models.alexnet(pretrained=True).features
        
        self.added_conv = torch.nn.Sequential(
                torch.nn.Conv2d(384, 306, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(306, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                )
        
        self.dense = torch.nn.Sequential(
            torch.nn.Dropout(),
            # size = [10, 256, 6, 6]
            torch.nn.Linear(256 * 256, 6796),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(6796, 996),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(996, 356),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(356, 86),
            torch.nn.ReLU(inplace=True),
        )
        
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(86, num_classes),
            )
        
    def forward(self, x):
        x = self.alex(x)
        # size = [10, 256, 6, 6]
        x = x.view(x.size(0), 256, 6 * 6)
        x = torch.bmm(x, torch.transpose(x, 1, 2))  # Bilinear
        x = x.view(x.size(0), 256 ** 2)
        x = self.dense(x)
        x = self.clf(x)
        return x
    
class Resnet152(torch.nn.Module):
    def __init__(self, num_classes=18):        
        torch.nn.Module.__init__(self)
        self.resnet152 = torchvision.models.resnet152(pretrained=False, num_classes=num_classes)
        
    def forward(self, x):
        x = self.resnet152(x)
        return x
    
class Bilinear_Resnet152(torch.nn.Module):
    def __init__(self, num_classes=18):        
        torch.nn.Module.__init__(self)
        self.resnet152 = torchvision.models.resnet152(pretrained=False, num_classes=num_classes)[:-1]
        
    def forward(self, x):
        x = self.resnet152(x)
        print(x.size())
        return x

        
class LSTM_CNN(torch.nn.Module):
    def __init__(self, num_classes=18):
        torch.nn.Module.__init__(self)
        self.features = torchvision.models.alexnet(pretrained=True).features
        
        self.added_conv = torch.nn.Sequential(
                torch.nn.Conv2d(256, 196, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                )
        
        self.dense = torch.nn.Sequential(
            torch.nn.Dropout(),
            # size = [10, 256, 6, 6]
            torch.nn.Linear(196 * 2 * 2, 1596),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1596, 796),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(796, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 56),
            torch.nn.ReLU(inplace=True),
        )
        
        self.lstm = torch.nn.Sequential(
            torch.nn.LSTM(196, 196),
            )
        
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(7 * 8, num_classes),
            )
        
    def forward(self, x):
        x = self.features(x)
        x = self.added_conv(x)
        x = x.view(x.size(0), 196, 2 * 2)
        x = torch.transpose(x, 1, 2).contiguous()
        x = torch.transpose(x, 0, 1).contiguous()
        x, _ = self.lstm(x)
        x = torch.transpose(x, 0, 1).contiguous() 
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.contiguous().view(x.size(0), 196 * 2 * 2)
        x = self.dense(x)
        x = self.clf(x)
        return x
    
class Alex_net(torch.nn.Module):

    def __init__(self, num_classes=18):
        super(Alex_net, self).__init__()
        self.features = torchvision.models.alexnet(pretrained=True).features
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            # size = [10, 256, 6, 6]
            torch.nn.Linear(256 * 6 * 6, 1596),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1596, 796),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(796, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 56),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(56, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
         # size = [10, 256, 6, 6]
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
class vgg19_bn(torch.nn.Module):
    
    def __init__(self, num_classes=18):        
        torch.nn.Module.__init__(self)
        self.vgg19_bn_features = torchvision.models.vgg19_bn(pretrained=True).features
        self.vgg19_bn_clf = torchvision.models.vgg19_bn(pretrained=True).classifier
        
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(1000, num_classes),
            )
        
    def forward(self, x):
        x = self.vgg19_bn_features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg19_bn_clf(x)
        x = self.clf(x)

        return x
    
class bilinear_vgg19_bn(torch.nn.Module):
    
    def __init__(self, num_classes=18):        
        torch.nn.Module.__init__(self)
        self.vgg19_bn_features = torchvision.models.vgg19_bn(pretrained=True).features
        self.vgg19_bn_clf = torchvision.models.vgg19_bn(pretrained=True).classifier
        
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(512 ** 2, 512 * 7 * 7),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            )
        
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(1000, num_classes)
            )
        
    def forward(self, x):
        x = self.vgg19_bn_features(x)
        x = x.view(x.size(0), 512, -1)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) # Bilinear Layer
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = self.vgg19_bn_clf(x)
        x = self.clf(x)

        return x
    
class createDataset(Dataset):
    
    def __init__(self, file_list, lbl, transform=None):
        self.file_list = file_list
        self.lbl = lbl
        self.transform = transform
    
    def __len__(self):
        return len(self.lbl)
    
    def __getitem__(self, idx):
        PIL_img = Image.open(self.file_list[idx])
        sample = {"train_data": PIL_img, "train_labels": self.lbl[idx]}
        if not self.transform is None:
            sample['train_data'] = self.transform(sample['train_data'])
        return sample["train_data"], sample["train_labels"]

# to shuffle at each time it loads
def get_dataloader(trainset, valiset, num_workers=2):
    trainloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=num_workers)
    valiloader = DataLoader(valiset, batch_size=10, shuffle=True, num_workers=num_workers)
    return trainloader, valiloader

def plot_performance(tr_loss_epoch, val_loss_epoch, tr_acc_epoch, val_acc_epoch, title_str=None):
    if title_str is None:
        title_str = "Model"
    
    num_epochs = len(tr_loss_epoch)
    
    loss_title_str = "Loss by " + title_str
    accu_title_str = "Accuracy by " + title_str
    
    t = np.arange(1, num_epochs+1, 1).astype(np.int32)
    
    fig, ax = plt.subplots(1, 2, figsize=(9, 6))
    ax[0].plot(t, tr_loss_epoch, 'r', label='Train Loss')
    ax[0].plot(t, val_loss_epoch, 'b', label='Validation Loss')
    ax[0].legend()
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title(loss_title_str)
    
    ax[1].plot(t, tr_acc_epoch, 'r', label='Train Accuracy')
    ax[1].plot(t, val_acc_epoch, 'b', label='Validation Acuuracy')
    ax[1].legend()
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title(accu_title_str)
    
    fig.savefig(title_str + ".jpg")
    
    fig.show() # show() should be put in the end of the function to guarantee scuccess of creation of the figure

def run_file(net_select = 7):
    # Our study has shown that Bilinear_CNN performed the best
    num_epochs = 100
    n_batches_per_print = 20
    
    train_folder = "new_release/new_release/train"
    val_folder = "new_release/new_release/val"
    
    if net_select == 1:
        net = Vanilla_net()
        title_str = "Vanilla_net"
    elif net_select == 2:
        net = Alex_net()
        title_str = "Alex_net"
    elif net_select == 3:
        net = Bilinear_CNN()
        title_str = "Bilinear_CNN"
    elif net_select == 4:
        net = LSTM_CNN()
        title_str = "LSTM_CNN"
    elif net_select == 5:
        net = Resnet152()
        title_str = "Resnet152"
    elif net_select == 6:
        net = Bilinear_Resnet152()
        title_str = "Bilinear_Resnet152"
    elif net_select == 7:
        net = vgg19_bn()
        title_str = "vgg19_bn"
    elif net_select == 8:
        net = bilinear_vgg19_bn()
        title_str = "bilinear_vgg19_bn"
    else:
        return "Please specify a model to run."

    try:
        net = net.cuda()
    except:
        pass
    
    saved_net_file = title_str + ".pth"
    input_crop_size = 224
    # resize then center crop
    transform_comp = transforms.Compose([transforms.CenterCrop((int(input_crop_size*2.05), int(input_crop_size*2.05))), \
                                         transforms.Resize((input_crop_size, input_crop_size)), \
                                         transforms.ToTensor(), \
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = ImageFolder(train_folder, transform_comp)
    valiset = ImageFolder(val_folder, transform_comp)
    
    try:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    except:
        criterion = torch.nn.CrossEntropyLoss()
        
    # 0.0004 proven the best for bilinear cnn
    # 0.001 proven the best for resnet152
    optimizer = optim.SGD(net.parameters(), lr=0.0005 , momentum=0.9)
    lambda_lr = lambda epoch: 0.85 ** epoch \
        if epoch < 20 else 0.8 ** epoch \
        if epoch < 50 else 0.7 ** epoch

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    
    epoch_tr_loss_records = []
    epoch_val_loss_records = []
    epoch_tr_acc_records = []
    epoch_val_acc_records = []
    print("Training starts")
    for epoch in range(num_epochs):
        running_loss = 0.0
        trainloader, valiloader = get_dataloader(trainset, valiset)
        scheduler.step()
        
        tr_loss_records = []
        correct = 0.0
        total = 0.0
        loss_list = np.zeros(18)
        loss_record_list = np.zeros(18)
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            try:
                inputs, labels = inputs.cuda(), labels.cuda()
            except:
                pass
            optimizer.zero_grad()
            
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            for each_pred, each_label in zip(outputs, labels):
                loss_each = criterion(each_pred, each_label)

            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % n_batches_per_print == n_batches_per_print-1:    
                ave_loss = running_loss / n_batches_per_print
                print('Training [%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, ave_loss))
                tr_loss_records.append(ave_loss)
                running_loss = 0.0
                
            # to record accu
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("tr acc: " + str(correct / total))
        epoch_tr_acc_records.append(correct / total)
        epoch_tr_loss_records.append(np.mean(tr_loss_records))
                
        val_loss_records = []
        correct = 0.0
        total = 0.0
        for i, data in enumerate(valiloader, 0):
            # get the inputs
            inputs, labels = data
            try:
                inputs, labels = inputs.cuda(), labels.cuda()
            except:
                pass
            optimizer.zero_grad()
            
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            # print statistics
            running_loss += loss.item()
            if i % n_batches_per_print == n_batches_per_print-1:    
                ave_loss = running_loss / n_batches_per_print
                print('Validation [%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, ave_loss))
                val_loss_records.append(ave_loss)
                running_loss = 0.0
                
            # to record accu
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("val acc: " + str(correct / total))
        epoch_val_acc_records.append(correct / total)
        epoch_val_loss_records.append(np.mean(val_loss_records))
        
        if correct / total > 0.97:
            break
      
    # final accuracy test
    _, valiloader = get_dataloader(trainset, valiset)
    correct = 0.0
    total = 0.0
    for i, data in enumerate(valiloader, 0):  
        images, labels = data
        try:
            images, labels = images.cuda(), labels.cuda()
        except:
            pass        
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images (The first test with shuffled data): %d %%' % (
        100 * correct / total))
    
    torch.save(net.state_dict(), saved_net_file)
    
    # reload the model for test to double confirm the result
    if net_select == 1:
        loaded_net = Vanilla_net()
    elif net_select == 2:
        loaded_net = Alex_net()
    elif net_select == 3:
        loaded_net = Bilinear_CNN()
    elif net_select == 4:
        loaded_net = LSTM_CNN()
    elif net_select == 5:
        loaded_net = Resnet152()
    elif net_select == 6:
        loaded_net = Bilinear_Resnet152()
    elif net_select == 7:
        loaded_net = vgg19_bn()
    elif net_select == 8:
        loaded_net = bilinear_vgg19_bn()

    loaded_net.load_state_dict(torch.load(saved_net_file))
    
    try:
        loaded_net = loaded_net.cuda()
    except:
        pass
    # final accuracy test
    _, valiloader = get_dataloader(trainset, valiset)
    correct = 0.0
    total = 0.0
    for i, data in enumerate(valiloader, 0):  
        images, labels = data        
        try:
            images, labels = images.cuda(), labels.cuda()
        except:
            pass
        outputs = loaded_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images (The second test with data reshuffled): %d %%' % (
        100 * correct / total))
    
    plot_performance(epoch_tr_loss_records, epoch_val_loss_records, epoch_tr_acc_records, epoch_val_acc_records, title_str=title_str)
    
    recorded_results = {
            "epoch_tr_loss_records": epoch_tr_loss_records,
            "epoch_val_loss_records": epoch_val_loss_records,
            "epoch_tr_acc_records": epoch_tr_acc_records,
            "epoch_val_acc_records": epoch_val_acc_records
            }
            
    return net, recorded_results
            
if __name__=="__main__":
    net, recorded_results = run_file()