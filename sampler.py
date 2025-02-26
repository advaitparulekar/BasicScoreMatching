
import torch
from abc import ABC, abstractmethod
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageDraw
import cv2
import numpy as np
import random

import matplotlib.pyplot as plt


class Distribution(ABC):
    def __init__(self, latent_dim, ambient_dim):
        self.latent_dim = latent_dim
        self.ambient_dim = ambient_dim

    @abstractmethod
    def sample(self, batch_size):
        pass

class MNISTSampler():
    def __init__(self, samplerConfig):
        print(samplerConfig)
        preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (samplerConfig['image_size'], samplerConfig['image_size'])),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2*(x-0.5)),
            ]
        )
        
        mnist = datasets.MNIST(samplerConfig['image_dir'], train=True, download=True, transform = preprocess)

        self.dataloader = DataLoader(mnist, batch_size=samplerConfig['batch_size'], shuffle=True, generator=torch.Generator(device=samplerConfig['device']))    
        self.length = len(self.dataloader)
        self.image_iterator = iter(self.dataloader)
        self.batch_size = samplerConfig['batch_size']
        self.device = samplerConfig['device']


    def __iter__(self):
        return self

    def __next__(self):
        try:
            current_batch = next(self.image_iterator)
        except StopIteration:
            self.image_iterator = iter(self.dataloader)
            raise StopIteration
        [images, labels] = current_batch
        images = images.to(self.device)
        var = torch.var(images.clone()).detach()
        return images, labels, var
    
