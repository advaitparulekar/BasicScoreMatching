
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from blocks import UpBlock, MidBlock, DownBlock
# device = torch.device('cuda')


class Decoder(nn.Module):
    def __init__(self, input_width, output_width, in_channels, h_channels, out_channels, n_res_layers=1, layers = 8):
        super(Decoder, self).__init__()
        up_layers = int(math.log2(output_width/input_width))
        self.input_conv = nn.ConvTranspose2d(in_channels, h_channels//(2**up_layers), kernel_size=3, stride=1, padding=1)

        up_stack = [UpBlock(in_channels = h_channels//(2**up_layers)*(2**depth),
                            out_channels = h_channels//(2**up_layers)*2**(depth+1), 
                            t_emb_dim = None,
                            up_sample = True,
                            num_heads = None,
                            num_layers = n_res_layers,
                            attn = None,
                            norm_channels = max(1, h_channels//(2**up_layers)*(2**depth)//4)) for depth in range(up_layers)]
        
        mid_stack = [UpBlock(in_channels = h_channels,
                            out_channels = h_channels, 
                            t_emb_dim = None,
                            num_heads = None,
                            up_sample = False,
                            attn = None,
                            num_layers = n_res_layers,
                            norm_channels = max(1, out_channels//4)) for depth in range(layers - up_layers)]
        
        self.inverse_conv_stack = nn.Sequential(
                # ResidualStack(in_channels, in_channels, res_h_dim, n_res_layers, (h_channels, input_width, input_width)),
                *up_stack,
                *mid_stack
                # ResidualStack(h_channels, h_channels, res_h_dim, n_res_layers, (h_channels, output_width, output_width))
        )
        self.norm_out = nn.GroupNorm(h_channels//4, h_channels)
        self.act_out = nn.SiLU()
        self.output_conv = nn.ConvTranspose2d(h_channels, out_channels, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x):
        x = self.input_conv(x)
        x = self.inverse_conv_stack(x)
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.output_conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_width, output_width, in_channels, h_channels, out_channels, n_res_layers = 1, layers = 8):
        super(Encoder, self).__init__()
        down_layers = int(math.log2(input_width/output_width))
        self.input_conv = nn.Conv2d(in_channels, h_channels//2**down_layers, kernel_size=3, stride=1, padding=1)

        down_stack = [DownBlock(in_channels = h_channels//(2**down_layers)*(2**depth),
                            out_channels = h_channels//(2**down_layers)*(2**(depth+1)), 
                            t_emb_dim = None,
                            down_sample = True,
                            num_heads = None,
                            num_layers = n_res_layers,
                            attn = None,
                            norm_channels = max(1, h_channels//(2**down_layers)*(2**depth)//4)) for depth in range(down_layers)]
        
        mid_stack = [DownBlock(in_channels = h_channels,
                            out_channels = h_channels, 
                            t_emb_dim = None,
                            num_heads = None,
                            down_sample = False,
                            attn = None,
                            num_layers = n_res_layers,
                            norm_channels = max(1, h_channels//4)) for _ in range(layers - down_layers)]
        self.conv_stack = nn.Sequential(
                # ResidualStack(in_channels, in_channels, res_h_dim, n_res_layers, (h_channels, input_width, input_width)),
                *down_stack,
                *mid_stack
                # ResidualStack(h_channels, h_channels, res_h_dim, n_res_layers, (h_channels, output_width, output_width))
        )
        self.norm_out = nn.GroupNorm(h_channels//4, h_channels)
        self.act_out = nn.SiLU()
        self.output_conv = nn.Conv2d(h_channels, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.input_conv(x)
        x = self.conv_stack(x)
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.output_conv(x)
        return x

#=============================#

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta, device):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.device = device

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
    


# #=============================#

# class DownBlock(nn.Module):
#     """
#     One residual layer inputs:
#     - in_dim : the input dimension
#     - h_dim : the hidden layer dimension
#     - res_h_dim : the hidden dimension of the residual block
#     """

#     def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, in_size, out_size):
#         super(DownBlock, self).__init__()
#         self.down_block = nn.Sequential(
#             nn.LayerNorm(in_size),
#             nn.Conv2d(in_dim, res_h_dim, kernel_size=4,
#                       stride=2, padding=1),
#             nn.SiLU(True),
#             nn.Conv2d(res_h_dim, h_dim, 3, 1, 1),
#             nn.SiLU(True)
#         )
#         self.res_stack = nn.ModuleList([ResidualLayer(h_dim, h_dim, res_h_dim, out_size) for _ in range(n_res_layers)])

#     def forward(self, x):
#         x = self.down_block(x)
#         for layer in self.res_stack:
#             x = x + layer(x)
#         return x
    
# #=============================#

# class UpBlock(nn.Module):
#     """
#     One residual layer inputs:
#     - in_dim : the input dimension
#     - h_dim : the hidden layer dimension
#     - res_h_dim : the hidden dimension of the residual block
#     """

#     def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, in_size, out_size):
#         super(UpBlock, self).__init__()
#         self.down_block = nn.Sequential(
#             nn.LayerNorm(in_size),
#             nn.ConvTranspose2d(in_dim, res_h_dim, kernel_size=4,
#                       stride=2, padding=1),
#             nn.SiLU(True),
#             nn.ConvTranspose2d(res_h_dim, h_dim, 3, 1, 1),
#             nn.SiLU(True)
#         )
#         self.res_stack = nn.ModuleList([ResidualLayer(h_dim, h_dim, res_h_dim, out_size) for _ in range(n_res_layers)])

#     def forward(self, x):
#         x = self.down_block(x)
#         for layer in self.res_stack:
#             x = x + layer(x)
#         return x
# #=============================#

class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim, size):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.LayerNorm(size),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.SiLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=3,
                      stride=1, padding = 1, bias=False),
            nn.SiLU(True)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, size):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim, size)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class LinearLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, dim):
        super(LinearLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.linear(x)
        return x

class LinearStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_channels, in_width, n_linear_layers):
        super(LinearStack, self).__init__()
        self.n_linear_layers = n_linear_layers
        self.stack = nn.ModuleList(
            [nn.Flatten()]+
            [LinearLayer(in_channels * in_width * in_width)]*n_linear_layers+
            [nn.Unflatten(1, (in_channels, in_width, in_width))])

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return x
    
if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()
    # test Residual Layer
    res = ResidualLayer(40, 40, 20)
    res_out = res(x)
    print('Res Layer out shape:', res_out.shape)
    # test res stack
    res_stack = ResidualStack(40, 40, 20, 3)
    res_stack_out = res_stack(x)
    print('Res Stack out shape:', res_stack_out.shape)



class VQVAE(nn.Module):
    def __init__(self, encoder_config, save_img_embedding_map=False):
        super(VQVAE, self).__init__()

        input_size=encoder_config["image_size"]
        output_size=encoder_config["embedding_size"]
        ambient_channels=encoder_config["image_channels"]
        latent_channels = encoder_config["latent_channels"]
        embedding_channels=encoder_config["embedding_channels"]
        h_channels = encoder_config["h_channels"]
        n_res_layers = encoder_config["n_res_layers"]
        n_embeddings=encoder_config["code_book_size"]
        layers = encoder_config["layers"]
        
        beta = encoder_config["beta"]
        device = encoder_config["device"]
        # encode image into continuous latent space
        self.encoder = Encoder(input_size, output_size, ambient_channels, h_channels, latent_channels, n_res_layers, layers)
        self.pre_quantization_conv = nn.Conv2d(
            latent_channels, embedding_channels, kernel_size=1, stride=1)
        self.device = device
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_channels, beta, self.device)
        # decode the discrete latent representation
        self.decoder = Decoder(output_size, input_size, embedding_channels, h_channels, ambient_channels, n_res_layers, layers)
        
        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e_q = self.pre_quantization_conv(z_e.clone())
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e_q)
        x_hat = self.decoder(z_q.clone())

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('quantized data shape', z_q.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, z_q, perplexity
    
    def encoder_forward(self, x):
        z_e = self.encoder(x)
        z_e_q = self.pre_quantization_conv(z_e.clone())
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e_q)
        return embedding_loss, z_q, perplexity
    

