import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from .vae import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HVAE(VAE):
    def __init__(self, input_shape, out_channels, latent_dim, hidden_channels, n_lf=3, eps_lf=0.01, beta_zero=0.3):
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
        VAE.__init__(self, input_shape, out_channels, latent_dim, hidden_channels)

        self.vae_forward = super().forward
        self.n_lf = n_lf

        self.eps_lf = nn.Parameter(torch.Tensor([eps_lf]), requires_grad=False)

        assert 0 < beta_zero <= 1, "Tempering factor should belong to [0, 1]"

        self.beta_zero_sqrt = nn.Parameter(
            torch.Tensor([beta_zero]), requires_grad=False
        )

    def forward(self, x):
        """
        The HVAE model
        """

        output = self.vae_forward(x)
        recon_x, z0, eps0, mu, logvar = output['x'], output['z'], output['eps'], output['mu'], output['logvar']
        gamma = torch.randn_like(z0, device=device)
        rho = gamma / self.beta_zero_sqrt
        z = z0
        beta_sqrt_old = self.beta_zero_sqrt

        recon_x = self.decode(z)

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

            recon_x = self.decode(z)

            U = -self.log_p_xz(recon_x, x, z).sum()
            g = grad(U, z, create_graph=True)[0]

            # 3rd leapfrog step
            rho__ = rho_ - (self.eps_lf / 2) * g

            # tempering steps
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        return dict(x=recon_x, z=z, z0=z0, rho=rho, eps=eps0, gamma=gamma, mu=mu, logvar=logvar)

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
        mu, logvar = self.encode(x)
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=device)
        Z = (mu + Eps * torch.exp(0.5 * logvar)).reshape(-1, self.latent_dim)

        recon_X = self.decode(Z)

        gamma = torch.randn_like(Z, device=device)
        rho = gamma / self.beta_zero_sqrt
        rho0 = rho
        beta_sqrt_old = self.beta_zero_sqrt
        X_rep = x.repeat(sample_size, 1, 1, 1)

        for k in range(self.n_lf):

            U = self.hamiltonian(recon_X, X_rep, Z, rho, name='HVAE')
            g = grad(U, Z, create_graph=True)[0]

            # step 1
            rho_ = rho - (self.eps_lf / 2) * g

            # step 2
            Z = Z + self.eps_lf * rho_

            recon_X = self.decode(Z)

            U = self.hamiltonian(recon_X, X_rep, Z, rho_, name='HVAE')
            g = grad(U, Z, create_graph=True)[0]

            # step 3
            rho__ = rho_ - (self.eps_lf / 2) * g

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        bce = F.binary_cross_entropy(recon_X, X_rep, reduction="none")

        # compute densities to recover p(x)
        logpxz = -bce.sum(dim=2)  # log(p(X|Z))

        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(Z))

        logrho0 = self.normal.log_prob(rho0 * self.beta_zero_sqrt).reshape(
            sample_size, -1
        )  # log(p(rho0))
        logrho = self.normal.log_prob(rho).reshape(sample_size, -1)  # log(p(rho_K))
        logqzx = self.normal.log_prob(Eps) - 0.5 * logvar.sum(dim=1)  # q(Z_0|X)

        logpx = (logpxz + logpz + logrho - logrho0 - logqzx).logsumexp(dim=0).mean(
            dim=0
        ) - torch.log(torch.Tensor([sample_size]).to(device))
        return logpx

    def hamiltonian(self, recon_x, x, z, rho, G_inv=None, G_log_det=None, name = 'HVAE'):
        """
        Computes the Hamiltonian function.
        used for HVAE and RHVAE
        """
        if name == "HVAE":
            return -self.log_p_xz(recon_x, x, z).sum()
        norm = (
            torch.transpose(rho.unsqueeze(-1), 1, 2) @ G_inv @ rho.unsqueeze(-1)
        ).sum()

        return -self.log_p_xz(recon_x, x, z).sum() + 0.5 * norm + 0.5 * G_log_det.sum()

    def _tempering(self, k, K):
        """Perform tempering step"""

        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k