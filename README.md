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
pyperclip
PyYAML
selenium
tensorboard
timm
torch
torchaudio
torchmetrics
torchvision
webdriver_manager
```

## Examples

- [Model Training](https://github.com/yupeeee/WAH/tree/main/examples/model_training)
- [Model Evaluation](https://github.com/yupeeee/WAH/tree/main/examples/model_evaluation)
- [Geodesic Optimization](https://github.com/yupeeee/WAH/tree/main/examples/geodesic_optimization)


## Structure

### `classification`
- `attacks`
    - fgsm:
    `FGSM`,
    `IFGSM`
- `datasets`
    - base:
    `ClassificationDataset`
    - cifar10:
    `CIFAR10`
    - cifar100:
    `CIFAR100`
    - dataloader
        - \_\_init\_\_:
        `to_dataloader`
        - transforms:
        `CollateFunction`
    - imagenet:
    `ImageNet`
    - stl10:
    `STL10`
    - utils:
    `compute_mean_and_std`,
    `DeNormalize`,
    `Normalize`,
    `portion_dataset`,
    `tensor_to_dataset`
- `models`
    - feature_extraction:
    `FeatureExtractor`
    - load:
    `add_preprocess`,
    `load_model`,
    `load_state_dict`
    - replace:
        - \_\_init\_\_:
        `Replacer`
- `test`
    - accuracy:
    `AccuracyTest`
    - eval:
    `EvalTest`
    - hessian_max_eigval_spectrum:
    `HessianMaxEigValSpectrum`
    - loss:
    `LossTest`
    - pred:
    `PredTest`
    - tid:
    `TIDTest`
- `train`
    - plot:
    `proj_train_path_to_2d`,
    `TrainPathPlot2D`
    - train:
    `Wrapper`,
    `load_trainer`

### `module`
`_getattr`,
`get_attrs`,
`get_module_name`,
`get_module_params`,
`get_named_modules`,
`get_valid_attr`

### `np`

### `path`
`basename`,
`clean`,
`dirname`,
`exists`,
`isdir`,
`join`,
`ls`,
`mkdir`,
`rmdir`,
`rmfile`,
`split`,
`splitext`

### `plot`
- dist:
`DistPlot2D`
- hist:
`HistPlot2D`
- image:
`ImShow`
- mat:
`MatShow2D`
- quiver:
`QuiverPlot2D`,
`TrajPlot2D`
- scatter:
`GridPlot2D`,
`ScatterPlot2D`

### `riemann`
- geodesic:
`optimize_geodesic`
- grad:
`compute_jacobian`,
`compute_hessian`
- jacobian_sigvals:
`JacobianSigVals`

### `tensor`
`broadcasted_elementwise_mul`,
`create_1d_traj`,
`create_2d_grid`,
`flatten_batch`,
`repeat`,
`stretch`

### `torch`

### `utils`
- args:
`ArgumentParser`
- dictionary:
`dict_to_df`,
`dict_to_tensor`,
`load_csv_to_dict`,
`load_yaml_to_dict`,
`save_dict_to_csv`
- download:
`disable_ssl_verification`,
`download_url`,
`md5_check`
- logs:
`disable_lightning_logging`
- lst:
`load_txt_to_list`,
`save_list_to_txt`,
`sort_str_list`
- random:
`seed`,
`unseed`
- time:
`time`
- zip:
`extract`
