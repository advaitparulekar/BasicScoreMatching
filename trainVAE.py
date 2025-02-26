import VQVAE
from VQVAE import VQVAE
from torchvision import datasets
import torch
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import utils
import sampler
import argparse
import yaml
import matplotlib.pyplot as plt
from sampler import MNISTSampler
from dataclasses import dataclass



def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    torch.set_default_device(config["device"])
    device = torch.device(config["device"])
    sampler = MNISTSampler(config["pretrain_sampler_params"])
        
    results = {
        'n_updates': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
    }
    pretrain_encoder_config = config["pretrain_encoder_params"]
    pretrain_training_config = config["pretrain_training_params"]

    model = VQVAE(pretrain_encoder_config).to(config["device"])

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=pretrain_training_config["learning_rate"], amsgrad=True)
    

    log_interval = pretrain_training_config["log_interval"]
    loss = torch.nn.MSELoss()
    for epoch in range(pretrain_training_config["num_epochs"]):
        for i, batch in enumerate(sampler):
            x, labels, var = batch
            x = x.to(device)
            optimizer.zero_grad()
            embedding_loss, x_hat, y, perplexity = model(x.clone())
            assert y.shape[2] == pretrain_encoder_config["embedding_size"]
            recon_loss = loss(x_hat, x)/var
            total_loss = embedding_loss + recon_loss + torch.var(y)

            total_loss.backward()
            optimizer.step()

            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(total_loss.cpu().detach().numpy())
            results["n_updates"] = i

            if i % log_interval== 0:
                """
                save model and print values
                """
                
                print('Epoch #', epoch, 'Update #', i, 'Recon Error:',
                    np.mean(results["recon_errors"][-log_interval:]),
                    'Loss', np.mean(results["loss_vals"][-log_interval:]),
                    'Perplexity:', np.mean(results["perplexities"][-log_interval:]))
                fig, ax = plt.subplots(2,4, figsize = (10, 5))
                for i in range(4):
                    ax[0, i].imshow(process(x[i]), cmap = 'gray')
                    _, x_hat, y, _ = model(x)
                    ax[1, i].imshow(process(x_hat[i]), cmap = 'gray')
                plt.savefig('vae.png')
                plt.close()
        utils.save_model_and_results(
            model, results, pretrain_training_config["vae_file"]
        )

def save_path(config):
    return "_pretrained_vae_" + str(config["image_size"])+'_'+str(config["image_channels"]) + "_"+str(config["embedding_size"])+'_'+str(config["embedding_channels"])

def process(im):
    return torch.flatten(im, start_dim=0, end_dim=1).detach().cpu().numpy()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='./config/MNIST.yaml', type=str)
    args = parser.parse_args()
    train(args)

