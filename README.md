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
num_classes: 10
batch_size: 128
num_workers: 2

epochs: 200
init_lr: 0.1
seed: 0

optimizer: SGD
optimizer_cfg:
  momentum: 0.9
  weight_decay: 5.e-4

lr_scheduler: MultiStepLR
lr_scheduler_cfg:
  milestones: [ 60, 120, 160, ]
  gamma: 0.2

warmup_lr_scheduler: LinearLR
warmup_lr_scheduler_cfg:
  start_factor: 0.01
  total_iters: 10

criterion: CrossEntropyLoss

metrics:
  acc@1:
    Accuracy:
      task: multiclass
      top_k: 1
  ece:
    CalibrationError:
      task: multiclass
```

- **num_classes** (*int*) -
  number of classes in the dataset.

- **batch_size** (*int*) -
  how many samples per batch to load.

- **num_workers** (*int*) -
  how many subprocesses to use for data loading.
  0 means that the data will be loaded in the main process.

- **epochs** (*int*) -
  stop training once this number of epochs is reached.

- **init_lr** (*float*) -
  initial learning rate.

- **seed** (*int*) -
  seed value for random number generation.
  Must be a non-negative integer.
  If a negative integer is provided, no seeding will occur.

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

- **warmup_lr_scheduler** (*str*) -
  specifies which scheduler to use for warmup phase.
  Must be one of [*"ConstantLR"*, *"LinearLR"*, ].

- **warmup_lr_scheduler_cfg** -
  parameters for the specified warmup scheduler.
  The *optimizer* parameter does not need to be explicitly provided (automatically initialized).
  Note that the *total_iters* parameter initializes the length of warmup phase.

- **criterion** (*str*) -
  specifies which loss function to use.
  Must be one of the loss functions supported in
  [*torch.nn*](https://pytorch.org/docs/stable/nn.html#loss-functions).

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
from torchvision.datasets import CIFAR10

train_dataset = CIFAR10(train=True, ...)
val_dataset = CIFAR10(train=False, ...)

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
