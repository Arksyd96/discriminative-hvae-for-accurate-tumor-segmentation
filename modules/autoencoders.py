from typing import Any
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import pytorch_lightning as pl

from .base import (
    TimePositionalEmbedding, EncodingBlock, DecodingBlock,
    ResidualBlock, SelfAttention
)
from .lpips import LPIPSWithDiscriminator

class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels, 
        z_channels=4, 
        pemb_dim=None, 
        num_channels=128, 
        channels_mult=[1, 2, 4, 4], 
        num_res_blocks=2, 
        attn=None
    ) -> None:
        super().__init__()
        if attn is not None:
            assert channels_mult.__len__() == attn.__len__(), 'channels_mult and attn must have the same length'
            self.attn = attn
        else:
            self.attn = [False] * channels_mult.__len__()

        self.z_channels = z_channels
        self.channels_mult = [1, *channels_mult]
        
        # architecture modules
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding='same')
        self.enocoder = nn.ModuleList([
            EncodingBlock(
                in_channels=num_channels * self.channels_mult[idx],
                out_channels=num_channels * self.channels_mult[idx + 1],
                temb_dim=pemb_dim,
                num_blocks=num_res_blocks,
                attn=self.attn[idx],
                downsample=True if idx != self.channels_mult.__len__() - 2 else False
            ) for idx in range(self.channels_mult.__len__() - 1)
        ])
        bottleneck_channels = num_channels * self.channels_mult[-1]
        self.bottleneck_res_a = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, temb_dim=pemb_dim, groups=8)
        self.bottleneck_sa = SelfAttention(in_channels=bottleneck_channels, num_heads=8, head_dim=32, groups=8)
        self.bottleneck_res_b = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, temb_dim=pemb_dim, groups=8)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=bottleneck_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=bottleneck_channels, out_channels=self.z_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, pemb=None):
        x = self.in_conv(x)
        for encoder in self.enocoder:
            x = encoder(x, pemb)
        x = self.bottleneck_res_a(x, pemb)
        x = self.bottleneck_sa(x)
        x = self.bottleneck_res_b(x, pemb)
        x = self.out_conv(x)
        return x
    
class Decoder(nn.Module):
    def __init__(
        self, 
        out_channels, 
        z_channels, 
        pemb_dim=None, 
        num_channels=128, 
        channels_mult=[1, 2, 4, 4],
        num_res_blocks=2, 
        attn=None
    ) -> None:
        super().__init__()
        if attn is not None:
            assert channels_mult.__len__() == attn.__len__(), 'channels_mult and attn must have the same length'
            self.attn = list(reversed(attn))
        else: 
            self.attn = [False] * channels_mult.__len__()

        self.channels_mult = list(reversed([1, *channels_mult]))
        self.z_channels = z_channels
        
        # architecture modules
        bottleneck_channels = num_channels * self.channels_mult[0]
        self.in_conv = nn.Conv2d(self.z_channels, bottleneck_channels, kernel_size=3, padding='same')
        self.bottleneck_res_a = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, temb_dim=pemb_dim, groups=8)
        self.bottleneck_sa = SelfAttention(in_channels=bottleneck_channels, num_heads=8, head_dim=32, groups=8)
        self.bottleneck_res_b = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, temb_dim=pemb_dim, groups=8)

        self.decoder = nn.ModuleList([
            DecodingBlock(
                in_channels=num_channels * self.channels_mult[idx],
                out_channels=num_channels * self.channels_mult[idx + 1],
                temb_dim=pemb_dim,
                num_blocks=num_res_blocks,
                attn=self.attn[idx],
                upsample=True if idx != 0 else False
            ) for idx in range(self.channels_mult.__len__() - 1)
        ])
        
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=num_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, pemb=None):
        x = self.in_conv(x)
        x = self.bottleneck_res_a(x, pemb)
        x = self.bottleneck_sa(x)
        x = self.bottleneck_res_b(x, pemb)
        for decoder in self.decoder:
            x = decoder(x, pemb)
        x = self.out_conv(x)
        return x


##############################################################
###################### VAE MODEL #############################
##############################################################

class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape, # should be only the image shape (C, H, W)
        z_channels,
        pemb_dim=None,
        num_channels=128,
        channels_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn=None
    ) -> None:
        super().__init__()

        self.input_shape = np.array(input_shape)
        self.z_channels = z_channels
        in_channels = out_channels = self.input_shape[0]
        self.latent_shape = self.input_shape[1:] // 2 ** (channels_mult.__len__() - 1)
        self.latent_dim = z_channels * np.prod(self.latent_shape)
        
        # encoder network
        self.encoder = Encoder(
            in_channels, 2 * z_channels, pemb_dim, num_channels, channels_mult, num_res_blocks, attn
        )

        # decoder network
        self.decoder = Decoder(
            out_channels, z_channels, pemb_dim, num_channels, channels_mult, num_res_blocks, attn
        )

        # define a N(0, I) distribution
        self.normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim),
            covariance_matrix=torch.eye(self.latent_dim),
        )

    def forward(self, x, pemb=None):
        moments = self.encode(x, pemb)
        z, mu, logvar, eps = DiagonalGaussianDistribution(moments).sample(return_all=True)
        recon_x = self.decode(z, pemb)
        return recon_x, z, mu, logvar, eps

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def encode(self, x, pemb=None):
        moments = self.encoder(x, pemb).reshape(-1, 2 * self.latent_dim) # flattening for simplicity
        return moments

    def decode(self, z, pemb=None):
        z = z.reshape(-1, self.z_channels, *self.latent_shape) # reshape the latent space for simplicity
        x_prob = self.decoder(z, pemb)
        return torch.sigmoid(x_prob)

    def sample_img(
        self,
        z=None,
        x=None,
        n_samples=1,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """
        Simulate p(x|z) to generate an image
        """
        with torch.no_grad():
            if z is None:
                z = self.normal.sample(sample_shape=(n_samples,)).to(device)

            else:
                n_samples = z.shape[0]

            if x is not None:
                recon_x, z, _, _, _ = self.forward(x)
                return recon_x

            # z.requires_grad_(True)
            recon_x = self.decode(z)
            return recon_x

    def _tempering(self, k, K):
        """Perform tempering step"""
        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k

    ########## Estimate densities ##########

    def log_p_x_given_z(self, recon_x, x, reduction="none"):
        """
        Estimate the decoder's log-density modelled as follows:
            p(x|z)     = \prod_i Bernouilli(x_i|pi_{theta}(z_i))
            p(x = s|z) = \prod_i (pi(z_i))^x_i * (1 - pi(z_i)^(1 - x_i))"""
        return -F.binary_cross_entropy(recon_x, x, reduction=reduction).sum(dim=(1, 2, 3))

    def log_z(self, z):
        """
        Return Normal density function as prior on z
        """
        return self.normal.log_prob(z)

# NOTE: not working
    def log_p_x(self, x, sample_size=10):
        """
        Estimate log(p(x)) using importance sampling with q(z|x)
        """
        moments = self.encode(x)
        z, mu, logvar, eps = DiagonalGaussianDistribution(moments).sample(return_all=True)
        recon_x = self.decode(z)
        bce = F.binary_cross_entropy(
            recon_x, x.repeat(sample_size, 1, 1, 1), reduction="none"
        ).sum(dim=(1, 2, 3))

        # compute densities to recover p(x)
        logpxz = -bce.reshape(sample_size, -1, self.input_shape).sum(dim=2)
        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(z))
        logqzx = self.normal.log_prob(Eps) - 0.5 * log_var.sum(dim=1)

        logpx = (logpxz + logpz - logqzx).logsumexp(dim=0).mean(dim=0) - torch.log(
            torch.Tensor([sample_size]).to(device)
        )

        return logpx
# NOTE: not working
    def log_p_z_given_x(self, z, recon_x, x, sample_size=10):
        """
        Estimate log(p(z|x)) using Bayes rule and Importance Sampling for log(p(x))
        """
        logpx = self.log_p_x(x, sample_size)
        lopgxz = self.log_p_x_given_z(recon_x, x)
        logpz = self.log_z(z)
        return lopgxz + logpz - logpx

    def log_p_xz(self, recon_x, x, z):
        """
        Estimate log(p(x, z)) using Bayes rule
        """
        logpxz = self.log_p_x_given_z(recon_x, x)
        logpz = self.log_z(z)
        return logpxz + logpz

    ########## Kullback-Leiber divergences estimates ##########

    def kl_prior(self, mu, log_var):
        """KL[q(z|y) || p(z)] : exact formula"""
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def kl_cond(self, recon_x, x, z, mu, log_var, sample_size=10):
        """
        KL[p(z|x) || q(z|x)]

        Note:
        -----
        p(z|x) is approximated using IS on log(p(x))
        """
        logpzx = self.log_p_z_given_x(z, recon_x, x, sample_size=sample_size)
        logqzx = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=torch.diag_embed(torch.exp(log_var))
        ).log_prob(z)

        return (logqzx - logpzx).sum()


class HamiltonianAutoencoder(VariationalAutoencoder, pl.LightningModule):
    def __init__(
        self,
        input_shape, # should be only the image shape (C, H, W)
        z_channels,
        pemb_dim=None,
        T = 64,
        num_channels=128,
        channels_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn=None,
        n_lf=3,
        eps_lf=0.001,
        beta_zero=0.3,
        reg_weight=0.05,
        lr=1e-5,
        weight_decay=1e-6,
        lr_d_factor=1,
        precision=32,
        **kwargs
    ) -> None:
        """
        Inputs:
        -------

        n_lf (int): Number of leapfrog steps to perform
        eps_lf (float): Leapfrog step size
        beta_zero (float): Initial tempering
        tempering (str): Tempering type (free, fixed)
        model_type (str): Model type for VAR (mlp, convnet)
        latent_dim (int): Latentn dimension
        """
        pl.LightningModule.__init__(self)
        VariationalAutoencoder.__init__(
            self, input_shape, z_channels, pemb_dim, num_channels, channels_mult, num_res_blocks, attn
        )
        
        print('Device: ', self.device)
        self.positional_encoder = TimePositionalEmbedding(pemb_dim, T, local_device=None)

        self.vae_forward = super().forward
        self.n_lf = n_lf
        self.reg_weight = reg_weight

        self.eps_lf = nn.Parameter(torch.Tensor([eps_lf]), requires_grad=False)

        assert 0 < beta_zero <= 1, "Tempering factor should belong to [0, 1]"

        self.beta_zero_sqrt = nn.Parameter(
            torch.Tensor([beta_zero]), requires_grad=False
        )

        self.regularization = LPIPSWithDiscriminator(**kwargs['loss'])

        self.save_hyperparameters()
        self.automatic_optimization = False

    def forward(self, x, pos):
        """
        The HVAE model
        """
        pemb = self.positional_encoder(pos)
        recon_x, z0, mu, log_var, eps0 = self.vae_forward(x, pemb)
        gamma = torch.randn_like(z0, device=x.device)
        rho = gamma / self.beta_zero_sqrt
        z = z0
        beta_sqrt_old = self.beta_zero_sqrt

        recon_x = self.decode(z, pemb)

        for k in range(self.n_lf):

            # perform leapfrog steps

            # computes potential energy
            U = -self.log_p_xz(recon_x, x, z).sum()

            # Compute its gradient
            g = grad(U, z, create_graph=True)[0]

            # 1st leapfrog step
            rho_ = rho - (self.eps_lf / 2) * g

            # 2nd leapfrog step
            z = z + self.eps_lf * rho_

            recon_x = self.decode(z, pemb)

            U = -self.log_p_xz(recon_x, x, z).sum()
            g = grad(U, z, create_graph=True)[0]

            # 3rd leapfrog step
            rho__ = rho_ - (self.eps_lf / 2) * g

            # tempering steps
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        return recon_x, z, z0, rho, eps0, gamma, mu, log_var

    def loss_function(self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var):

        logpxz = self.log_p_xz(recon_x, x, zK)  # log p(x, z_K)
        logrhoK = self.normal.log_prob(rhoK)  # log p(\rho_K)
        logp = logpxz + logrhoK

        logq = self.normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)  # q(z_0|x)

        return -(logp - logq).sum()

    def log_p_x(self, x, sample_size=10):
        """
        Estimate log(p(x)) using importance sampling on q(z|x)
        """
        mu, log_var = self.encode(x.view(-1, self.input_dim))
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=x.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)

        recon_X = self.decode(Z)

        gamma = torch.randn_like(Z, device=x.device)
        rho = gamma / self.beta_zero_sqrt
        rho0 = rho
        beta_sqrt_old = self.beta_zero_sqrt
        X_rep = x.repeat(sample_size, 1, 1, 1).reshape(-1, self.input_dim)

        for k in range(self.n_lf):

            U = self.hamiltonian(recon_X, X_rep, Z, rho)
            g = grad(U, Z, create_graph=True)[0]

            # step 1
            rho_ = rho - (self.eps_lf / 2) * g

            # step 2
            Z = Z + self.eps_lf * rho_

            recon_X = self.decode(Z)

            U = self.hamiltonian(recon_X, X_rep, Z, rho_)
            g = grad(U, Z, create_graph=True)[0]

            # step 3
            rho__ = rho_ - (self.eps_lf / 2) * g

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        bce = F.binary_cross_entropy(recon_X, X_rep, reduction="none")

        # compute densities to recover p(x)
        logpxz = -bce.reshape(sample_size, -1, self.input_dim).sum(dim=2)  # log(p(X|Z))

        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(Z))

        logrho0 = self.normal.log_prob(rho0 * self.beta_zero_sqrt).reshape(
            sample_size, -1
        )  # log(p(rho0))
        logrho = self.normal.log_prob(rho).reshape(sample_size, -1)  # log(p(rho_K))
        logqzx = self.normal.log_prob(Eps) - 0.5 * log_var.sum(dim=1)  # q(Z_0|X)

        logpx = (logpxz + logpz + logrho - logrho0 - logqzx).logsumexp(dim=0).mean(
            dim=0
        ) - torch.log(torch.Tensor([sample_size]).to(x.device))
        return logpx

    def hamiltonian(self, recon_x, x, z):
        """
        Computes the Hamiltonian function.
        used for HVAE and RHVAE
        """
        return -self.log_p_xz(recon_x, x, z).sum()

    def _tempering(self, k, K):
        """Perform tempering step"""

        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k
    
    def on_train_start(self) -> None:
        # pushing the normal distribution attributes to GPU
        self.normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim).to(self.device),   # at this point local GPU is recognized 
            covariance_matrix=torch.eye(self.latent_dim).to(self.device),
        )

        self.positional_encoder.embedding = self.positional_encoder.embedding.to(self.device)
    
    def training_step(self, batch, batch_idx):
        # optimizers & schedulers
        ae_opt, disc_opt = self.optimizers()

        x, pos = batch
        x, pos = x.type(torch.float16 if self.hparams.precision == 16 else torch.float32), pos.type(torch.long)
        
        # x_hat=x_hat, z=z, z0=z0, rho=rho, eps=eps, gamma=gamma, mean=mean, logvar=logvar
        recon_x, z, z0, rho, eps0, gamma, mu, log_var = self.forward(x, pos)

        ########################
        # Optimize Autoencoder #
        ########################
        hvae_loss = self.loss_function(recon_x, x, z0, z, rho, eps0, gamma, mu, log_var)
        reg_loss, reg_log = self.regularization.autoencoder_loss(
            x, recon_x, self.global_step, last_layer=self.decoder.out_conv[-1].weight
        )
        
        ae_loss = (1 - self.reg_weight) * hvae_loss + self.reg_weight * reg_loss
        ae_opt.zero_grad(set_to_none=True)
        self.manual_backward(ae_loss)
        ae_opt.step()
        # ae_scheduler.step()

        ##########################
        # Optimize Discriminator #
        ##########################
        # sampling an image
        generated = self.sample_img(n_samples=x.shape[0])

        disc_loss, disc_log = self.regularization.discriminator_loss(x, generated, self.global_step)
        disc_opt.zero_grad(set_to_none=True)
        self.manual_backward(disc_loss)
        disc_opt.step()
        # disc_scheduler.step()

        # logging
        self.log('hvae_loss', hvae_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(reg_log, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(disc_log, on_step=True, on_epoch=True, prog_bar=False, logger=True)
    
    def configure_optimizers(self):
        ae_opt = torch.optim.AdamW(list(self.encoder.parameters()) + 
                                   list(self.decoder.parameters()) + 
                                   list(self.positional_encoder.parameters()),
                                   lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=(0.5, 0.9))
        disc_opt = torch.optim.AdamW(list(self.regularization.discriminator.parameters()), 
                                    lr=self.hparams.lr * self.hparams.lr_d_factor, weight_decay=self.hparams.weight_decay, betas=(0.5, 0.9))

        return [ae_opt, disc_opt]
    
    def sample_img(
        self,
        z=None,
        x=None,
        pos=None,
        n_samples=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Simulate p(x|z) to generate an image
        """
        with torch.no_grad():
            if z is None:
                z = self.normal.sample(sample_shape=(n_samples,)).to(device)

            if pos is None:
                pos = torch.randint(0, self.hparams.T, (n_samples,)).to(device)

            else:
                n_samples = z.shape[0]

            if x is not None:
                recon_x, z, _, _, _ = self.forward(x, pos)
                return recon_x

            # z.requires_grad_(True)
            recon_x = self.decode(z, pemb=self.positional_encoder(pos))
            return recon_x


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self, return_all=False):
        eps = torch.randn(self.mean.shape).to(device=self.parameters.device)
        z = self.mean + self.std * eps
        if return_all:
            return z, self.mean, self.logvar, eps
        return z

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
