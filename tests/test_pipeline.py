import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples
from modeling.unet import UnetModel
from utils import set_random_seed
import os
import shutil
from PIL import Image

@pytest.fixture(scope='module', autouse=True)
def test_helper():
    set_random_seed(seed=1)
    os.makedirs('./tests/tmp/', exist_ok=True)

    yield

    shutil.rmtree('./tests/tmp/')

@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5

@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_training(device, train_dataset):
    # note: implement and test a complete training procedure (including sampling)
    def train(model, optim, samples_path):        
        subset = Subset(train_dataset, list(range(0, 10)))
        dataloader = DataLoader(subset, batch_size=5, num_workers=2, shuffle=True)
        
        losses = []
        for i in range(2):
            loss, _ = train_epoch(model, dataloader, optim, device)
            losses.append(loss)

        generate_samples(model, device, samples_path)

        return sum(losses) / len(losses)
    
    # os.makedirs('./tests/tmp', exist_ok=True)

    ################ 1
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=128),
        betas=(1e-4, 0.02),
        num_timesteps=100,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    loss_1 = train(ddpm, optim, './tests/tmp/test_samples_1.png')

    ################ 2
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=128),
        betas=(1e-4, 0.02),
        num_timesteps=50,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    loss_2 = train(ddpm, optim, './tests/tmp/test_samples_2.png')

    ############## different num_timesteps
    assert loss_1 != loss_2

    samples_1 = Image.open('./tests/tmp/test_samples_1.png').convert('RGB')
    samples_2 = Image.open('./tests/tmp/test_samples_2.png').convert('RGB')

    assert samples_1 != samples_2
    
    ##############

    ################ 3
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=128),
        betas=(1e-4, 0.02),
        num_timesteps=100,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-3)

    loss_3 = train(ddpm, optim, './tests/tmp/test_samples_3.png')

    ################ 4
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=128),
        betas=(1e-4, 0.02),
        num_timesteps=100,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    loss_4 = train(ddpm, optim, './tests/tmp/test_samples_4.png')

    ############## different lr
    assert loss_3 != loss_4

    samples_3 = Image.open('./tests/tmp/test_samples_3.png').convert('RGB')
    samples_4 = Image.open('./tests/tmp/test_samples_4.png').convert('RGB')

    assert samples_3 != samples_4
    ##############


