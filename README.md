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
PyYAML
tensorboard
torch
torchaudio
torchmetrics
torchvision
```

## Model Training

Let's train [**ResNet50**](https://arxiv.org/abs/1512.03385) [1]
on [**CIFAR-10**](https://www.cs.toronto.edu/~kriz/cifar.html) [2] dataset
(Full example codes
[**here**](https://github.com/yupeeee/WAH/tree/main/examples/train_cifar10)).\
First, import the package.

```python
import wah
```

Second, write your own *config.yaml* file (which will do **everything** for you).

```yaml
--- # config.yaml


#
# Dataset
#
num_classes: 10
batch_size: 64
num_workers: 2
mixup_alpha: 0.
cutmix_alpha: 0.
# use_v2: False


#
# Training hyperparameters
#
epochs: 300
init_lr: 1.e-3
seed: 0
gpu: [ 0, ]

optimizer: AdamW
optimizer_cfg:
  weight_decay: 5.e-2

lr_scheduler: CosineAnnealingLR
lr_scheduler_cfg:
  eta_min: 1.e-5

warmup_lr_scheduler: LinearLR
warmup_lr_scheduler_cfg:
  start_factor: 1.e-2
  total_iters: 20

criterion: CrossEntropyLoss
criterion_cfg:
  label_smoothing: 0.


#
# Metrics
#
metrics:
- "Acc1"
- "Acc5"
- "ECE"
- "sECE"

track:
- "grad_l2"
- "feature_rms"


... # end
```

- **num_classes** (*int*) -
  number of classes in the dataset.

- **batch_size** (*int*) -
  how many samples per batch to load.

- **num_workers** (*int*) -
  how many subprocesses to use for data loading.
  0 means that the data will be loaded in the main process.

- **mixup_alpha** (*float*) -
  hyperparameter of the Beta distribution used for mixup.
  ([*mixup: Beyond Empirical Risk Minimization*](https://arxiv.org/abs/1710.09412))

- **cutmix_alpha** (*float*) -
  hyperparameter of the Beta distribution used for mixup.
  ([*CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*](https://arxiv.org/abs/1905.04899))

- **epochs** (*int*) -
  stop training once this number of epochs is reached.

- **init_lr** (*float*) -
  initial learning rate.

- **seed** (*int*) -
  seed value for random number generation.
  Must be a non-negative integer.
  If a negative integer is provided, no seeding will occur.

- **gpu** (*List[int], optional*) -
  the GPU device(s) to be utilized for computation.
  If not specified, the system automatically detects and selects devices.

- **optimizer** (*str*) -
  specifies which optimizer to use.
  Must be one of the optimizers supported in
  [*torch.optim*](https://pytorch.org/docs/stable/optim.html#algorithms).

- **optimizer_cfg** -
  parameters for the specified optimizer.
  The *params* and *lr* parameters do not need to be explicitly provided (automatically initialized).

- **lr_scheduler** (*str*) -
  specifies which scheduler to use.
  Must be one of the schedulers supported in
  [*torch.optim.lr_scheduler*](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).

- **lr_scheduler_cfg** -
  parameters for the specified scheduler.
  The *optimizer* parameter does not need to be explicitly provided (automatically initialized).

- **warmup_lr_scheduler** (*str, optional*) -
  specifies which scheduler to use for warmup phase.
  Must be one of [*"ConstantLR"*, *"LinearLR"*, ].

- **warmup_lr_scheduler_cfg** (*optional*) -
  parameters for the specified warmup scheduler.
  The *optimizer* parameter does not need to be explicitly provided (automatically initialized).
  Note that the *total_iters* parameter initializes the length of warmup phase.

- **criterion** (*str*) -
  specifies which loss function to use.
  Must be one of the loss functions supported in
  [*torch.nn*](https://pytorch.org/docs/stable/nn.html#loss-functions).

- **criterion_cfg** (*optional*) -
  parameters for the specified loss function.

  - **label_smoothing** (*optional, float*) -
     specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing.
     The targets become a mixture of the original ground truth and a uniform distribution
     as described in [*Rethinking the Inception Architecture for Computer Vision*](https://arxiv.org/abs/1512.00567).

- **metrics** -
  metrics to record during the training and validation stages.
  Must be the metrics supported in
  [*torchmetrics*](https://lightning.ai/docs/torchmetrics/stable/).

Third, load your *config.yaml* file.

```python
config = wah.load_config(PATH_TO_CONFIG)
```

Fourth, load your dataloaders.

```python
train_dataset = wah.CIFAR10(
    root=...,
    train=True,
    transform="auto",
    target_transform="auto",
    download=True,
)
val_dataset = wah.CIFAR10(
    root=...,
    train=False,
    transform="auto",
    target_transform="auto",
    download=True,
)

train_dataloader = wah.load_dataloader(
    dataset=train_dataset,
    config=config,
    shuffle=True,
)
val_dataloader = wah.load_dataloader(
    dataset=val_dataset,
    config=config,
    shuffle=False,
)
```

Fifth, load your model.

```python
from torchvision.models import resnet50

model = resnet50(weights=None, num_classes=10)
model = wah.Wrapper(model, config)
```

Finally, train your model!

```python
trainer = wah.load_trainer(
    config=config,
    save_dir=TRAIN_LOG_ROOT,
    name="cifar10-resnet50",
    every_n_epochs=SAVE_CKPT_PER_THIS_EPOCH,
)
trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
```

You can check your train logs by running the following command:

```commandline
tensorboard --logdir TRAIN_LOG_ROOT
```

### References

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. CVPR, 2016.\
[2] Alex Krizhevsky and Geoffrey Hinton. Learning Multiple Layers of Features from Tiny Images. Tech. Rep., University
of Toronto, Toronto, Ontario, 2009.
