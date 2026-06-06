# Visual Representations with DINOv2 — Features, Matching & Transfer Learning

Exploration of **DINOv2 (ViT-S/14)** self-supervised visual features across three tasks: token analysis, cross-image feature matching, and transfer learning for image classification.

## What was built

- Feature extraction pipeline using `forward_features` to retrieve both CLS and patch tokens
- Cross-image patch-level feature matching with cosine similarity visualization
- MLP classifiers trained on raw pixels vs. DINOv2 CLS features on CIFAR-10

## Part 1 — CLS token vs. Patch tokens

DINOv2 ViT-S/14 produces:
- **CLS token** — 384-dim global summary of the entire image (what the image is)
- **Patch tokens** — 256 × 384 local representations, one per 14×14 pixel region (where things are)

Patch token norm maps show that highest activations correspond to the most semantically informative image regions, making them well-suited for dense tasks like segmentation and multimodal alignment.

## Part 2 — Cross-image feature matching

Computed patch-level cosine similarity across 10 real-world image pairs (two different views of the same object):

| Object | Avg Score | Matches > 0.6 |
|--------|-----------|--------------|
| Castillo | 0.944 | 256 |
| Park (Parc Güell) | 0.947 | 253 |
| Guitarra | 0.926 | 144 |
| Torre Eiffel | 0.924 | 140 |
| Cerveza | 0.885 | 164 |

**Mean similarity across all pairs: 0.918** — demonstrating strong scale and viewpoint invariance in DINOv2 features. Objects with higher scale variation between views (guitar, Eiffel Tower) show fewer strong matches but maintain high average similarity scores.

## Part 3 — Transfer learning on CIFAR-10

Trained a 3-layer MLP classifier on CIFAR-10 using two input representations:

| Input | Test Accuracy | Convergence |
|-------|--------------|-------------|
| Raw pixels (3072-dim) | 56.0% | Slow, 30 epochs |
| **DINOv2 CLS token (384-dim)** | **74.2%** | Fast — 67.6% at epoch 1 |

DINOv2 features surpass raw pixels by **+18.2%** while using a 8× smaller input dimension. The MLP on top only needs to learn a linear-ish decision boundary in an already structured feature space, rather than learning to detect edges, textures, and objects from scratch.

**Note on CIFAR-10 + DINOv2:** CIFAR-10 images are 32×32 and must be upsampled to 224×224 for DINOv2, introducing interpolation artifacts — yet the features remain highly effective, underscoring the generalization power of large-scale self-supervised pretraining.

## Key insights

- CLS token is the right representation for classification; patch tokens are better for dense/spatial tasks
- DINOv2 patch features show near-perfect scale invariance across diverse object categories
- Self-supervised pretraining on internet-scale data transfers remarkably well even to low-resolution, out-of-distribution inputs like CIFAR

## Files

```
project3_dinov2_visual_representations/
├── notebook.ipynb   # Feature extraction, matching visualization, MLP training
└── report.pdf       # Written analysis
```

## Run

```bash
pip install torch torchvision timm matplotlib numpy scikit-learn requests
jupyter notebook notebook.ipynb
```

> DINOv2 model is loaded via `torch.hub` from `facebookresearch/dinov2`. Internet access required on first run.
