import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

batch_size_const = 64


class ModelWrapper:

    def __init__(self, data_loaders, data_sets, epochs_num=20, model=models.resnet50(weights=None)):
        self.batch_size = batch_size_const
        learning_rate = 3e-3
        self.epoch_num = epochs_num
        self.model = model
        self.model = self.model.cuda()
        self.loss_fun = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.dataloaders = data_loaders
        self.data_sets = data_sets

    def train_it(self):

        train_accuracies = []
        test_accuracies = []
        train_losses = []
        test_losses = []
        total_step = len(self.dataloaders['train'])
        test_loss_min = np.Inf

        for epoch in range(self.epoch_num):
            print(f'Epoch {epoch}\n')
            is_better = False

            for phase in ['train', 'test']:

                loss_value = 0.0
                corrects_count = 0

                if phase == 'train':
                    self.model.train()

                    for batch_index, (inputs, labels) in enumerate(self.dataloaders[phase]):
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())

                        inputs = inputs.float()
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = self.loss_fun(outputs, labels)
                        loss.backward()
                        self.optimizer.step()

                        _, predictions = torch.max(outputs, 1)
                        loss_value += loss.item()
                        corrects_count += torch.sum(predictions == labels.data)
                        if batch_index % 20 == 0:
                            print(
                                'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, self.epoch_num - 1,
                                                                                   batch_index, total_step,
                                                                                   loss.item()))
                    self.scheduler.step()
                else:
                    with torch.no_grad():
                        self.model.eval()
                        for inputs, labels in self.dataloaders[phase]:
                            inputs = Variable(inputs.cuda())
                            labels = Variable(labels.cuda())

                            inputs = inputs.float()
                            outputs = self.model(inputs)
                            loss = self.loss_fun(outputs, labels)

                            _, preds = torch.max(outputs, 1)
                            loss_value += loss.item()
                            corrects_count += torch.sum(preds == labels.data)
                    is_better = loss_value < test_loss_min
                    test_loss_min = loss_value if is_better else test_loss_min

                epoch_loss = loss_value / len(self.data_sets[phase])
                epoch_accuracy = corrects_count.double() / len(self.data_sets[phase])

                if phase == 'train':
                    train_accuracies.append(epoch_accuracy * 100)
                    train_losses.append(epoch_loss)
                else:
                    test_accuracies.append(epoch_accuracy * 100)
                    test_losses.append(epoch_loss)
            print(f'\ntrain-loss: {np.mean(train_losses):.4f}, train-acc: {train_accuracies[-1]:.4f}')
            print(f'test loss: {np.mean(test_losses):.4f}, test acc: {test_accuracies[-1]:.4f}\n')

            if is_better:
                torch.save(self.model.state_dict(), f'models/weights.h5')
                print('Improvement-Detected, save-model')
        train_accuracies_cpu = []
        test_accuracies_cpu = []
        for t in train_accuracies:
            train_accuracies_cpu.append(t.cpu())
        for t in test_accuracies:
            test_accuracies_cpu.append(t.cpu())
        return train_accuracies_cpu, test_accuracies_cpu, train_losses, train_losses