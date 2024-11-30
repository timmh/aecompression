# adapted from: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html

## Standard libraries
import os
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

## Imports for plotting
import matplotlib.pyplot as plt
# %matplotlib inline 
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()
sns.set()

## Progress bar
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Tensorboard extension (for visualization purposes later)
from torch.utils.tensorboard import SummaryWriter
# %load_ext tensorboard

# Setting the seed
pl.seed_everything(42)

# # Ensure that all operations are deterministic on GPU (if used) for reproducibility
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

CHECKPOINT_PATH = "checkpoints"
HEIGHT = 96
WIDTH = 160
PRECISION = 32

class IWildCamDataset(data.Dataset):
    def __init__(self, data_dir, split, width=WIDTH, height=HEIGHT, precision=PRECISION):
        if split == "train":
            annotation_filename = f"iwildcam2022_{split}_annotations.json"
        elif split == "test":
            annotation_filename = f"iwildcam2022_{split}_information.json"
        else:
            raise ValueError(f"Invalid split {split}")
        with open(data_dir / "metadata" / annotation_filename) as f:
            self.data = json.load(f)
        self.raw_image_dir = data_dir / split
        self.width = width
        self.height = height
        self.dtype = getattr(torch, f"float{precision}")
        self.transforms = transforms_v2.Compose([
            transforms_v2.Resize(size=(self.height, self.width), antialias=True),
            transforms_v2.ToDtype(self.dtype, scale=True),
            transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.cache_path = Path("/tmp/aecompression-cache") / "datasets" / f"{split}_{width}_{height}_{precision}.pt"
        self._cache = None

    def __len__(self):
        return len(self.data["images"])

    def cache_on_device_(self, device):
        if self.cache_path.exists():
            print(f"Loading cache {self.cache_path}")
            self._cache = torch.load(self.cache_path, map_location=device, weights_only=True, mmap=True)
            print(f"Finished loading cache {self.cache_path} of shape {self._cache.shape}")
        else:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache = torch.cat(list(tqdm(iter(data.DataLoader(self, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)))))
            torch.save(self._cache, self.cache_path)
            self._cache = self._cache.to(device=device)
            torch.save(self._cache, self.cache_path)

    def unload_cache_(self):
        self._cache = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        else:
            image = self.data["images"][idx]
            image_path = self.raw_image_dir / image["file_name"]
            try:
                with Image.open(str(image_path)) as img:
                    return self.transforms(transforms_v2.functional.pil_to_tensor(img.convert("RGB"))).clone()
            except Exception as e:
                print(f"Failed to read image '{image_path}'. Replacing with zero tensor. Full exception: {e}")
                return torch.zeros((3, self.height, self.width), dtype=self.dtype)

def get_train_images(num):
    return torch.stack([train_dataset[i] for i in range(num)], dim=0)


class Encoder(nn.Module):
    
    def __init__(self, 
                 num_input_channels : int, 
                 base_channel_size : int, 
                 latent_dim : int, 
                 act_fn : object = nn.GELU):
        """
        Inputs: 
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),  # torch.Size([2, 64, 20, 12])
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*c_hid * (WIDTH * HEIGHT) // (4 ** 3), latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    
    def __init__(self, 
                 num_input_channels : int, 
                 base_channel_size : int, 
                 latent_dim : int, 
                 act_fn : object = nn.GELU):
        """
        Inputs: 
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * (HEIGHT // 8) * (WIDTH // 8) * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, HEIGHT // 8, WIDTH // 8)
        x = self.net(x)
        return x


class Autoencoder(pl.LightningModule):
    
    def __init__(self, 
                 base_channel_size: int, 
                 latent_dim: int, 
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3, 
                 width: int = WIDTH, 
                 height: int = HEIGHT):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters() 
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)
        
    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         factor=0.2, 
                                                         patience=20, 
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)                             
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)


def compare_imgs(img1, img2, title_prefix="", i=0):
    os.makedirs('figures', exist_ok=True)
    # Calculate MSE loss between both images
    loss = F.mse_loss(img1, img2, reduction="sum")
    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(torch.stack([img1, img2], dim=0), nrow=2, normalize=True)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4,2))
    plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(f"figures/comparisons_{i}.png", bbox_inches="tight")
    plt.close()


class GenerateCallback(pl.Callback):
    
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
        
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, value_range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


def train_iwildcam(latent_dim):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"iwildcam_{latent_dim}"), 
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         precision=PRECISION,
                         devices=1,
                         max_epochs=100,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(get_train_images(8), every_n_epochs=1),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"iwildcam_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=32, latent_dim=latent_dim).to(dtype=getattr(torch, f"float{PRECISION}"))
        trainer.fit(model, train_loader, val_loader)
        trainer.save_checkpoint(pretrained_filename)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result


def visualize_reconstructions(model, input_imgs, latent_dim=None):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(model.device))
    
    # Plotting
    imgs = torch.stack([input_imgs.detach().cpu(), reconst_imgs.detach().cpu()], dim=1).flatten(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7,4.5))
    plt.title(f"Reconstructed from {model.hparams.latent_dim} latents")
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(f"figures/reconstructions_{latent_dim}.png", bbox_inches="tight")
    plt.close()


def embed_imgs(model, data_loader, n=float("inf")):
    # Encode all images in the data_laoder using model, and return both images and encodings
    img_list, embed_list = [], []
    model.eval()
    for i, imgs in enumerate(tqdm(data_loader, desc="Encoding images", leave=False)):
        if i >= n:
            break
        with torch.no_grad():
            z = model.encoder(imgs.to(model.device))
        img_list.append(imgs)
        embed_list.append(z.detach().cpu())
    return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0))


def find_similar_images(query_img, query_z, key_embeds, K=8, i=0):
    # Find closest K images. We use the euclidean distance here but other like cosine distance can also be used.
    dist = torch.cdist(query_z[None,:], key_embeds[1], p=2)
    dist = dist.squeeze(dim=0)
    dist, indices = torch.sort(dist)
    # Plot K closest images
    imgs_to_display = torch.cat([query_img[None], key_embeds[0][indices[:K]]], dim=0).detach().cpu()
    grid = torchvision.utils.make_grid(imgs_to_display, nrow=K+1, normalize=True, value_range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(12,3))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(f"figures/similar_{i}.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = IWildCamDataset(Path("/data/vision/beery/scratch/data/iwildcam_unzipped"), split="train")
    train_dataset.cache_on_device_(device)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [0.9, 0.1], torch.Generator().manual_seed(42))

    # Loading the test set
    test_set = IWildCamDataset(Path("/data/vision/beery/scratch/data/iwildcam_unzipped"), split="test")
    test_set.cache_on_device_(device)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, num_workers=0)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=0)

    for i in range(2):
        # Load example image
        img = train_dataset[i].cpu().to(dtype=torch.float32)
        img_mean = img.mean(dim=[1,2], keepdims=True)

        # Shift image by one pixel
        SHIFT = 1
        img_shifted = torch.roll(img, shifts=SHIFT, dims=1)
        img_shifted = torch.roll(img_shifted, shifts=SHIFT, dims=2)
        img_shifted[:,:1,:] = img_mean
        img_shifted[:,:,:1] = img_mean
        compare_imgs(img, img_shifted, "Shifted -", i=i)

        # Set half of the image to zero
        img_masked = img.clone()
        img_masked[:,:img_masked.shape[1]//2,:] = img_mean
        compare_imgs(img, img_masked, "Masked -", i=i)

    model_dict = {}
    for latent_dim in [256]: #[64, 128, 256, 384]:
        model_ld, result_ld = train_iwildcam(latent_dim)
        model_dict[latent_dim] = {"model": model_ld, "result": result_ld}


    latent_dims = sorted([k for k in model_dict])
    val_scores = [model_dict[k]["result"]["val"][0]["test_loss"] for k in latent_dims]

    fig = plt.figure(figsize=(6,4))
    plt.plot(latent_dims, val_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
    plt.xscale("log")
    plt.xticks(latent_dims, labels=latent_dims)
    plt.title("Reconstruction error over latent dimensionality", fontsize=14)
    plt.xlabel("Latent dimensionality")
    plt.ylabel("Reconstruction error")
    plt.minorticks_off()
    plt.ylim(0,100)
    plt.savefig("figures/reconstruction_err_vs_dim.png", bbox_inches="tight")
    plt.close()

    input_imgs = get_train_images(4)
    for latent_dim in model_dict:
        visualize_reconstructions(model_dict[latent_dim]["model"], input_imgs, latent_dim=latent_dim)


    rand_imgs = torch.rand(2, 3, HEIGHT, WIDTH) * 2 - 1
    visualize_reconstructions(model_dict[256]["model"], rand_imgs, latent_dim="rand")


    plain_imgs = torch.zeros(3, 3, HEIGHT, WIDTH)

    # Single color channel
    plain_imgs[1,0] = 1 
    # Checkboard pattern
    plain_imgs[2,:,:16,:16] = 1
    plain_imgs[2,:,16:,16:] = -1

    visualize_reconstructions(model_dict[256]["model"], plain_imgs, latent_dim="plain")


    model = model_dict[256]["model"]
    latent_vectors = torch.randn(8, model.hparams.latent_dim, device=model.device)
    with torch.no_grad():
        imgs = model.decoder(latent_vectors)
        imgs = imgs.cpu()

    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1,1), pad_value=0.5)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(8,5))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(f"figures/reconstructions_random_latent.png", bbox_inches="tight")
    plt.close()

    # We use the following model throughout this section. 
    # If you want to try a different latent dimensionality, change it here!
    model = model_dict[128]["model"] 

    n_embeddings = 100
    train_img_embeds = embed_imgs(model, train_loader, n=n_embeddings)
    test_img_embeds = embed_imgs(model, test_loader, n=n_embeddings)

    # Plot the closest images for the first N test images as example
    for i in range(8):
        find_similar_images(test_img_embeds[0][i], test_img_embeds[1][i], key_embeds=train_img_embeds, i=i)



    # We use the following model throughout this section. 
    # If you want to try a different latent dimensionality, change it here!
    model = model_dict[128]["model"]


    # Create a summary writer
    writer = SummaryWriter("tensorboard/")


    writer.add_embedding(test_img_embeds[1][:n_embeddings], # Encodings per image
                        metadata=[test_set[i][1] for i in range(n_embeddings)], # Adding the labels per image to the plot
                        label_img=(test_img_embeds[0][:n_embeddings]+1)/2.0) # Adding the original images to the plot


    # tensorboard --logdir tensorboard/


    # Closing the summary writer
    writer.close()