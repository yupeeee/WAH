![logo](https://github.com/yupeeee/WAH/blob/main/WAH.png?raw=true)

## Install

```commandline
pip install wah
```

### Requirements

You might want to manually install [**PyTorch**](https://pytorch.org/get-started/locally/)
for GPU computation.

```text
lightning
matplotlib
numpy
pandas
pillow
requests
timm
torch
torchmetrics
torchvision
tqdm
yaml
```

## Structure

### `wah`
- `classification`
	- `datasets`
		- `CIFAR10`
		- `CIFAR100`
		- `compute_mean_and_std`
		- `ImageNet`
		- `load_dataloader`
		- `portion_dataset`
		- `STL10`
	- `models`
		- `FeatureExtractor`
		- `load`
		- `load_state_dict`
		- `replace`
			- `gelu_with_relu`
			- `relu_with_gelu`
			- `bn_with_ln`
			- `ln_with_bn`
		- `summary`
	- `test`
		- `brier_score`
		- `ece`
	- `Trainer`
- `dicts`
- `fun`
	- `RecursionWrapper`
- `lists`
- `mods`
- `path`
- `random`
- `tensor`
- `time`
- `utils`
	- `ArgumentParser`
	- `download`
	- `zips`