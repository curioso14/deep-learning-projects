# ResNet18 vs PlainNet18 — Residual Connections on CIFAR-100

Implementation and experimental comparison of **ResNet18** and a plain baseline (**PlainNet18**) trained from scratch on CIFAR-100, isolating the effect of skip connections on gradient flow, convergence speed, and final accuracy.

## What was built

- `BasicBlock` with optional shortcut projection (1×1 conv + BN) for dimension mismatches
- `PlainBlock` — identical architecture with the residual addition removed
- Full training pipeline: SGD + momentum, weight decay, cosine annealing LR, data augmentation
- CIFAR-adapted stem: 3×3 conv, stride 1, no maxpool (standard for 32×32 inputs)

## Results

| Model | Best Val Accuracy | Parameters |
|-------|------------------|------------|
| **ResNet18** | **72.27%** | 11,220,132 |
| PlainNet18 | 71.85% | 11,046,308 |

**Convergence:** ResNet18 reached 20.63% validation accuracy after epoch 1 vs. 14.77% for PlainNet18. The gap was most pronounced in early training — at 18 layers, skip connections primarily improve stability and convergence speed rather than final accuracy, consistent with the original ResNet paper's findings.

## Key insights

- Shortcut projections (1×1 conv) are needed at the start of stages 2, 3, and 4 where both spatial resolution and channel count change
- Without residual connections, gradients must pass through all conv layers sequentially — making early layers slow to learn
- At 18 layers the vanishing gradient problem is mild; the benefit of residual connections becomes more pronounced in deeper architectures (50+ layers)

## Training config

```
Optimizer : SGD (momentum=0.9, weight_decay=5e-4)
Scheduler : CosineAnnealingLR (T_max=50)
Batch size: 128
Epochs    : 50
Augment   : RandomCrop(32, padding=4), RandomHorizontalFlip, RandomRotation(15°)
Dataset   : CIFAR-100 (50k train / 10k test, 100 classes)
```

## Files

```
project1_resnet_cifar100/
├── notebook.ipynb   # Full implementation, training, and comparison plots
└── report.pdf       # Written analysis
```

## Run

```bash
pip install torch torchvision matplotlib numpy
jupyter notebook notebook.ipynb
```

> GPU recommended — training both models takes ~30–40 min on a single GPU.
