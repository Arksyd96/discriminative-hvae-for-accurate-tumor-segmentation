import torch
import pytorch_lightning as pl
import wandb

class ImageSampler(pl.Callback):
    def __init__(self, 
        n_samples=5,
        label='Sampling'
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.label = label

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # sample images
            pl_module.eval()
            generated = pl_module.sample_img(n_samples=self.n_samples)
            
            # channel wise grid
            img_grid = torch.cat([ 
                torch.hstack([img for img in generated[:, idx, ...]])
                for idx in range(generated.shape[1])
            ], dim=0)
            
            wandb.log({
                'Reconstruction examples': wandb.Image(
                    img_grid.detach().cpu().numpy(), 
                    caption='{}'.format(self.label)
                )
            })
            