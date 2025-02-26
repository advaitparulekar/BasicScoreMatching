import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import numpy as np


class negativeone_to_one(object):
    def __init__(self):
        pass
    
    def __call__(self, im):
        im = 2*im-1
        return im

def load_mnist():
    train = datasets.MNIST(root="./datasets/mnist", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 negativeone_to_one()
                             ]))

    val = datasets.MNIST(root="./datasets/mnist", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               negativeone_to_one()
                           ]))
    return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(batch_size):
    
    training_data, validation_data = load_mnist()
    training_loader, validation_loader = data_loaders(
        training_data, validation_data, batch_size)
    print(training_data.data.shape)
    x_train_var = torch.var(training_data.data / 255.0)

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, file_name):
    results_to_save = {
        'model': model.state_dict(),
        'results': results,
    }
    torch.save(results_to_save, file_name)