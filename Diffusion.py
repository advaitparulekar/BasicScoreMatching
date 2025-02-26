from abc import ABC, abstractmethod
import torch
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
import os
from torchvision import datasets, transforms
import diffusers
from unet_base import Unet
from VQVAE import VQVAE
from datetime import datetime
import dataclasses
import json


NUM_TRAIN_TIMESTEPS = 400
class Diffuser():
    def __init__(self, config):

        # Config #
        pretrain_encoder_config = config["pretrain_encoder_params"]
        self.unet_config = config["unet_params"]
        self.training_config = config["score_training_params"]

        self.device = torch.device(self.training_config["device"])

        self.config = config
        self.noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)

        self.score_network = Unet(im_channels = 1, model_config = self.unet_config).to(self.device)
  
        self.base_encoder = VQVAE(pretrain_encoder_config).to(self.device)
        self.loadVAE(self.training_config["vae_file"])


    def parameters(self):
        parameters = []
        parameters += list(self.score_network.parameters())
        return parameters
    
    def print_num_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("number of parameter: ", params)

    def setup(self, num_training_steps):
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.training_config["learning_rate"])
        self.print_num_parameters()
        self.lr_scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(
            optimizer = self.optimizer,
            num_warmup_steps = self.training_config["learning_rate"],
            num_training_steps = num_training_steps,
        )
    
    def score(self, z, noise_levels):
        return self.score_network(z, noise_levels)

    # TRAIN
    def train_loop(self, train_dataloader):
        loss_func = nn.MSELoss()
        self.setup(train_dataloader.length * 15)


        if self.training_config["from_checkpoint"]:
            self.load_model()
        
        for epoch in range(self.training_config["num_epochs"]):
            progress_bar = tqdm(total=train_dataloader.length)
            progress_bar.set_description(f"Epoch {epoch}")
            
            for _, (clean_images, _, var) in enumerate(train_dataloader):
                clean_images = clean_images.to(self.device)
                batch_size = clean_images.shape[0]
                _, _, embedded_images, _ = self.base_encoder(clean_images)
                timesteps = torch.randint(0, NUM_TRAIN_TIMESTEPS, (batch_size,), device=self.device)
                noise = torch.randn(embedded_images.shape).to(self.device)
                noisy_images = self.noise_scheduler.add_noise(embedded_images, noise, timesteps)
                noise_pred = self.score(noisy_images, timesteps)
                loss = loss_func(noise_pred, noise)
                loss.backward()
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                progress_bar.update(1)
                logs = {
                    "loss" : loss.detach().item(),
                    "lr" : self.lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)
            if epoch % 10 == 0:
                self.save_model(self.training_config["save_file"])
                self.display()


    

    # SAMPLE
    @torch.no_grad()
    def sample(self, num_samples, seed, save_process_dir=None):
        torch.manual_seed(seed)

        if save_process_dir:
            if not os.path.exists(save_process_dir):
                os.mkdir(save_process_dir)
        
        self.noise_scheduler.set_timesteps(NUM_TRAIN_TIMESTEPS)
        image=torch.randn((num_samples, 1, self.unet_config["image_size"], self.unet_config["image_size"])).to(self.device)
        num_steps=max(self.noise_scheduler.timesteps).numpy()
        
        for t in self.noise_scheduler.timesteps:
            model_output=self.score(image,torch.ones((num_samples, )).to(self.device)*t)
            image=self.noise_scheduler.step(model_output,int(t),image,generator=None)['prev_sample']
            if save_process_dir:
                save_image = transforms.ToPILImage()(image.squeeze(0))
                save_image.resize((256,256)).save(
                    os.path.join(save_process_dir,"seed-"+str(seed)+"_"+f"{num_steps-t.numpy():03d}"+".png"),format="png")
            
        image = self.base_encoder.decoder(image)

        return image



    # LOAD AND SAVE

    def save(self, optimizer, it, PATH):
        if os.path.exists(PATH):
            os.remove(PATH)
        state = {}
        state.update({'score_network': self.score_network.state_dict()})
        state['optimizer'] = optimizer.state_dict()
        state['it'] = it
        torch.save(state, PATH)


    def save_model(self, PATH):
        if os.path.exists(PATH):
            os.remove(PATH)
        state = {}
        state.update({'score_network': self.score_network.state_dict()})

        torch.save(state, PATH)

    
    def load(self, optimizer, PATH):
        checkpoint = torch.load(PATH, weights_only=False)
        self.score_network.load_state_dict(checkpoint['score_network'])
        self.score_network.train()

        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['it']
    
    def loadVAE(self, PATH):
        checkpoint = torch.load(PATH, weights_only=False)
        self.base_encoder.load_state_dict(checkpoint['model'])
        self.base_encoder.eval()
    
    def load_model(self):
        PATH = self.training_config['checkpoint']
        checkpoint = torch.load(PATH)
        self.score_network.load_state_dict(checkpoint['score_network'])
        self.score_network.train()
        
    def display(self, save = True):
        def process(im):
            return torch.flatten(im, start_dim=0, end_dim=1).detach().cpu().numpy()
        samples = self.sample(16, 0)
        fig, ax = plt.subplots(4, 4)
        for i in range(4):
            for j in range(4):
                ax[i, j].imshow((process(samples[4*i+j])))
                ax[i, j].axis("off")
        if save:
            plt.savefig(self.training_config["display_file"])


