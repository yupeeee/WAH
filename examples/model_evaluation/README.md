# Model Evaluation

## Accuracy

Let's compute the top-1 accuracy of [**ResNet50**](https://arxiv.org/abs/1512.03385) [1]
on [**ImageNet**](https://www.image-net.org/challenges/LSVRC/2012/index.php) [2] validation dataset
(Full example codes
[**here**](https://github.com/yupeeee/WAH/tree/main/examples/model_evaluation/acc1.py)).\
First, import the package.

```python
import wah
```

Second, load your dataset.

```python
dataset = wah.classification.datasets.ImageNet(
    root=...,
    split="val",
    transform="auto",
    target_transform="auto",
    download=True,
)
```

Third, load your model.

```python
model = wah.classification.models.load_model(
    name="resnet50",
    weights="IMAGENET1K_V1",
    load_from="torchvision",
)
```

Fourth, load your test.

```python
test = wah.classification.test.AccuracyTest(
    top_k=1,
    batch_size=32,
    num_workers=4,
    devices="auto",
)
```

Finally, test your model!

```python
acc1 = test(
    model=model,
    dataset=dataset,
)
print(f"Acc@1 of resnet50 on ImageNet: {acc1 * 100:.2f}%")
```

The result will be as follows, which is the same as reported in [**link**](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights).

```
>>> 
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing DataLoader 0: 100%|█████...██████| 1563/1563 [00:00<00:00, 0s/it]
Acc@1 of resnet50 on ImageNet: 76.13%
```
![resnet50_IMAGENET1K_V1_acc1](https://github.com/yupeeee/WAH/blob/main/examples/model_evaluation/resnet50_IMAGENET1K_V1_acc1.PNG?raw=true)


### References

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. CVPR, 2016.\
[2] Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.
