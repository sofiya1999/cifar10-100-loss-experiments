import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from pytorch_metric_learning import losses
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model_wrappers import ModelWrapper as mw
from modules import margin_loss

device = torch.device("cuda:0")

batch_size_const = 64


def make_data_sets():
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        'test':
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
    }

    data_sets = {
        'train': torchvision.datasets.CIFAR100(root='data', train=True, download=True,
                                               transform=data_transforms['train']),
        'test': torchvision.datasets.CIFAR100(root='data', download=True, transform=data_transforms['test'])
    }

    dataloaders = {
        'train':
            torch.utils.data.DataLoader(data_sets['train'],
                                        batch_size=batch_size_const,
                                        shuffle=True,
                                        num_workers=0),
        'test':
            torch.utils.data.DataLoader(data_sets['test'],
                                        batch_size=batch_size_const,
                                        shuffle=False,
                                        num_workers=0)
    }

    return data_transforms, data_sets, dataloaders


def main_fun():
    dt, ds, dl = make_data_sets()
    model = models.resnet50(weights=None, num_classes=100)
    emb_size = 100
    model.fc = nn.Linear(2048, emb_size)
    resnet50 = mw.ModelWrapper(data_loaders=dl, data_sets=ds, epochs_num=20,
                               model=model,
                               loss_fun=losses.ArcFaceLoss(num_classes=100, embedding_size=emb_size,
                                                           margin=85.8, scale=24).to(torch.device('cuda')),
                               learning_rate=3e-2)
    train_accuracies, test_accuracies, train_losses, test_losses = resnet50.train_it()

    #vgg19 = mw.ModelWrapper(data_loaders=dl, data_sets=ds, epochs_num=20, model=models.vgg19(weights=None))
    #train_accuracies, test_accuracies, train_losses, test_losses = vgg19.train_it()

    #vgg16 = mw.ModelWrapper(data_loaders=dl, data_sets=ds, epochs_num=20, model=models.vgg16(weights=None))
    #train_accuracies, test_accuracies, train_losses, test_losses = vgg16.train_it()

    #mobile_net = mw.ModelWrapper(data_loaders=dl, data_sets=ds, epochs_num=20,
    #                             model=models.mobilenet_v2(weights=None, num_classes=100))
    #train_accuracies, test_accuracies, train_losses, test_losses = mobile_net.train_it()

    plt.title("Train-Test Accuracy")
    plt.plot(train_accuracies, label='Training accuracy')
    plt.plot(test_accuracies, label='Test accuracy')
    plt.xlabel('epochs_num')
    plt.ylabel('accuracy')
    plt.legend(frameon=False)
    plt.savefig('accuracy_stat_resnet50_margin_loss.png')

    plt.clf()

    plt.title("Train-Test Loss")
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.xlabel('epochs_num')
    plt.ylabel('loss')
    plt.legend(frameon=False)
    plt.savefig('losses_stat_resnet50_margin_loss.png')


if __name__ == '__main__':
    print('Cuda:', torch.cuda.is_available())
    #dt, ds, dl = make_data_sets()
    #print(ds['train'])
    #print('=' * 85)
    #print(ds['test'])
    main_fun()

    #model = models.resnet50(weights=None, num_classes=100).cuda()
    #model.fc = nn.Linear(2048, 100).cuda()
    #summary(model, (3, 32, 32))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
