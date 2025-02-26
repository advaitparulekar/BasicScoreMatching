# %%
# Imports

# Pytorch
import torch
import torchvision

# HuggingFace
import datasets
import diffusers
import accelerate
import yaml

# Training and Visualization
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import PIL

# Causal Diffusion Model
import Diffusion
from sampler import MNISTSampler
from dataclasses import dataclass

import sys
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/MNIST.yaml', type=str)
    args = parser.parse_args()
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    torch.set_default_device(config["device"])

    sampler = MNISTSampler(config["score_sampler_params"])
    model = Diffusion.Diffuser(config)

    model.train_loop(sampler)
