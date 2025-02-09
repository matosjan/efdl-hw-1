import torch
import hydra
import wandb
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel
from hydra.utils import instantiate
from utils import set_random_seed  
from omegaconf import OmegaConf

def main():
    config = OmegaConf.load("params.yaml")
    #### wandb
    wandb.login()

    wandb.init(
        project=config.logger.project_name,
        name=config.logger.run_name,
        mode='online'
    )

    artifact = wandb.Artifact("config", type="config")
    artifact.add_file("params.yaml")
    wandb.log_artifact(artifact)
    ####

    os.makedirs(config.trainer.path_to_save_samples, exist_ok=True)

    device = config.trainer.device
    set_random_seed(config.trainer.random_seed)

    eps_model = instantiate(config.eps_model)
    ddpm = instantiate(config.model, eps_model=eps_model)
    ddpm.to(device)

    train_transforms = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if config.trainer.random_flip is True:
        train_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms = transforms.Compose(train_transforms)

    dataset = CIFAR10(
        "./data/train",
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = instantiate(config.dataloader, dataset=dataset, shuffle=True)
    optim = instantiate(config.optim, params=ddpm.parameters())

    num_epochs = config.trainer.num_epochs
    for i in range(num_epochs):
        epoch_loss, input_batch = train_epoch(ddpm, dataloader, optim, device)
        wandb.log({'train_loss': epoch_loss}, step=i)
        wandb.log({'lr': optim.param_groups[0]['lr']}, step=i)
        wandb.log({'input_batch': wandb.Image(make_grid(input_batch))}, step=i)
        path = generate_samples(ddpm, device, f"{config.trainer.path_to_save_samples}/{i:02d}.png")
        wandb.log({'samples': wandb.Image(path)}, step=i)
    
    torch.save(ddpm.state_dict(), 'weights.pth')
    print(f"Saved weights saved to weights.pth")


if __name__ == "__main__":
    main()
