import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion import *

class GaussianVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim, input_dim=args.input_dim)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=args.input_dim, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
        
    def get_loss(self, x, kl_weight=1.0):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        # log_pz = standard_normal_logprob(z).sum(dim=1)  # (B, ), Independence assumption
        # entropy = gaussian_entropy(logvar=z_sigma)      # (B, )
        # loss_prior = (- log_pz - entropy).mean()
        
        loss_prior = torch.mean(-0.5 * torch.sum(1 + z_sigma - z_mu ** 2 - z_sigma.exp(), dim = 1), dim = 0)

        loss_recons = self.diffusion.get_loss(x, z)

        loss = kl_weight * loss_prior + loss_recons
        
        print("loss_prior:", loss_prior.item(), "loss_recons:", loss_recons.item())

        return loss

    def sample(self, z, num_points, flexibility=0.0, truncate_std=None):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(num_points, context=z, point_dim=self.args.input_dim, flexibility=flexibility)
        return samples
