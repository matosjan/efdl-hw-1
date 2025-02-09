
# Week 2 home assignment
## Bugs and fixes
### `diffusion.py` 
 
* `DiffusionModel.forward(...):` 
	1.  the `timestep` variable was not moved to the right device. Found this bug through testing.
	2.  the `eps` variable was sampled from uniform distribution, however it should be sampled from standard normal distribution, so i used `torch.randn_like()`
	3. wrong formula for the `x_t` variable: `self.sqrt_one_minus_alpha_prod` should be used instead of `self.one_minus_alpha_over_prod`

	Bugs `2` and `3` where found when reading the code and comparing the formulas to the original paper
	
* `DiffusionModel.sample(...):`
	1.  the `x_t` variable was not moved to the right device
	2.  the `z` variable was not moved to the right device
	
	These bugs were found in the process of debugging failing tests

* `get_schedules(...):` corrected the assert to check that betas are positive: `assert 0 < beta1 < beta2 < 1.0`.  The comment in the original code helped to identify the mistake



### `unet.py`

* `UnetModel.forward(...):` the `temb` variable should have two fake dimensions(unsqueezed two times), so that broadcasting works when summing tensors. This bug was found because of failed tests

## Changes in tests

* After fixing all the bugs described above there was still a flapping test. So I added a seed fixing procedure to ensure that the testing is deterministic
* Implemented `test_training` test in `test_pipeline.py` that covers the whole training process

### Tests coverage
| Name                      | Stmts | Miss | Cover |
|---------------------------|-------|------|-------|
| modeling/\__init__.py      | 0     | 0    | 100%  |
| modeling/diffusion.py     | 34    | 0    | 100%  |
| modeling/training.py      | 32    | 0    | 100%  |
| modeling/unet.py          | 68    | 0    | 100%  |
| **TOTAL**                 | 134   | 0    | 100%  |

## Other changes in code
* Added [Hydra](https://hydra.cc/) library support.  Made changes to the training code to work with the new configuration method
* Added `wandb` logging. Made minor changes in `main.py` and `training.py`  so that everything is logged right
* Implemented `set_random_seed` function in `utils.py`, to make the training process reproducible
*  Added `dvc` support and integrated it with `hydra`

## How to run

* Setup the environment:
	 ```bash
	conda create --name photomaker python=3.11
	conda activate photomaker
	pip install -r requirements.txt
	```

* Setup `dvc`:
	```bash
	dvc init
	dvc config hydra.enabled True
	```
* Change the training configs to your liking: you can do that in the `./conf` directory
* Run the training pipeline:
	```bash
	dvc exp run
	```
	
## Weight&Biases 

 The full training run and 3 additional runs:  [wandb project](https://wandb.ai/matos-team/efdl_hw_1?nw=nwusermatosyanalex04)