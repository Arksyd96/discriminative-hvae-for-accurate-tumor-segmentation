import numpy as np
import torch
import pytorch_lightning as pl
from nibabel import load
from nibabel.processing import resample_to_output
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IdentityDataset(torch.utils.data.Dataset):
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]

class BRATSDataModule(pl.LightningDataModule):
    def __init__(self,
        target_shape=(64, 128, 128),
        n_samples=500,
        modalities=['t1', 't1ce', 't2', 'flair', 'seg'],
        binarize=True,
        npy_path='../data/brats_preprocessed.npy',
        root_path='../../common_data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021',
        batch_size=32,
        shuffle=True,
        num_workers=4,
        **kwargs
    ) -> None:
        assert all([m in ['t1', 't1ce', 't2', 'flair', 'seg'] for m in modalities]), 'Invalid modality!'
        
        super().__init__()
        self.num_modalities = len(modalities)
        self.prepare_data_per_node = True

        # just for a faster access
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if not os.path.exists(self.hparams.npy_path):
            print('Loading dataset from NiFTI files...')
            placeholder = np.zeros(shape=(
                self.hparams.n_samples, 
                self.num_modalities, 
                self.hparams.target_shape[1], 
                self.hparams.target_shape[2], 
                self.hparams.target_shape[0]
            ))

            for idx, instance in enumerate(tqdm(os.listdir(self.hparams.root_path)[: self.hparams.n_samples], position=0, leave=True)):
                # loading models
                volumes = {}
                for _, m in enumerate(self.hparams.modalities):
                    volumes[m] = load(os.path.join(self.hparams.root_path, instance, instance + f'_{m}.nii.gz'))

                # Compute the scaling factors (output will not be exactly the same as defined in OUTPUT_SHAPE)
                orig_shape = volumes[self.hparams.modalities[0]].shape
                scale_factor = (orig_shape[0] / self.hparams.target_shape[1], # height
                                orig_shape[1] / self.hparams.target_shape[2], # width
                                orig_shape[2] / self.hparams.target_shape[0]) # depth

                # Resample the image using trilinear interpolation
                # Drop the last extra rows/columns/slices to get the exact desired output size
                for _, m in enumerate(self.hparams.modalities):
                    volumes[m] = resample_to_output(volumes[m], voxel_sizes=scale_factor, order=1).get_fdata()
                    volumes[m] = volumes[m][:self.hparams.target_shape[1], :self.hparams.target_shape[2], :self.hparams.target_shape[0]]

                # binarizing the mask (for simplicity), you can comment out this to keep all labels
                if self.hparams.binarize and 'seg' in self.hparams.modalities:
                    volumes['seg'] = (volumes['seg'] > 0).astype(np.float32)

                # saving models
                for idx_m, m in enumerate(self.hparams.modalities):
                    placeholder[idx, idx_m, :, :] = volumes[m]

            print('Saving dataset as npy file...')    
            # saving the dataset as a npy file
            np.save(self.hparams.npy_path, placeholder)
                
            print('Saved!')
            print('Max: {}, Min: {}'.format(placeholder.max(), placeholder.min()))
                
        else:
            print('Dataset already exists at {}'.format(self.hparams.npy_path))
        
    def setup(self, stage='fit'):
        assert os.path.exists(self.hparams.npy_path), 'npy data file does not exist!'
        
        print('Loading dataset from npy file...')
        self.data = torch.from_numpy(np.load(self.hparams.npy_path))
        self.data = self.data[:self.hparams.n_samples]
        
        # normalize the data [0, 1] example by example
        for idx in range(self.data.shape[0]):
            self.data[idx] = (self.data[idx] - self.data[idx].min()) / (self.data[idx].max() - self.data[idx].min())

        self.data = self.data.permute(0, 4, 1, 2, 3) # depth first
            
        # if switching to 2D for autoencoder training
        D, W, H = self.hparams.target_shape
        self.data = self.data.reshape(self.hparams.n_samples * D, -1, W, H)

        # keeping track on slice positions for positional embedding
        self.slice_positions = torch.arange(D)[None, :].repeat(self.hparams.n_samples, 1)
        self.slice_positions = self.slice_positions.flatten()
        
        train_size = int(0.85 * self.data.shape[0])
        
        self.train_x = self.data[:train_size]
        self.train_pos = self.slice_positions[:train_size]
        self.test_x = self.data[train_size:]
        self.test_pos = self.slice_positions[train_size:]

        print('Train shape:', self.train_x.shape) 
        print('Test shape:', self.test_x.shape)
        print('Train slice positions shape:', self.train_pos.shape)
        print('Test slice positions shape:', self.test_pos.shape)
        print('Min: {}, Max: {}'.format(self.data.min(), self.data.max()))
        
        self.train_dataset = IdentityDataset(self.train_x, self.train_pos)
        self.test_dataset = IdentityDataset(self.test_x, self.test_pos)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=self.hparams.shuffle, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True
        )
    