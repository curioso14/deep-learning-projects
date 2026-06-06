# Face Recognition with Contrastive Learning — Siamese Network on LFW

Implementation of a metric learning system for face verification using a Siamese network with contrastive loss, trained on the Labeled Faces in the Wild (LFW) dataset. Instead of a fixed-class softmax classifier, the model learns an embedding space where Euclidean distance reflects identity similarity — enabling generalization to unseen identities.

## What was built

- **Siamese Network** with frozen ResNet18 backbone (ImageNet pretrained) + trainable projection head (512 → 256 → 128 dims, ReLU, Dropout 0.5, L2 normalization)
- **Contrastive Loss** with margin m=1.0: pulls same-identity pairs together, pushes different-identity pairs apart
- **ROC curve analysis** with optimal threshold selection via Youden's index
- **t-SNE visualization** of the 128-dim embedding space
- **Error analysis** — systematic inspection of false positives and false negatives

## Results

| Metric | Value |
|--------|-------|
| AUC | 0.6981 |
| Optimal threshold d* | 0.409 |
| Accuracy (optimal threshold) | 65.40% |
| False positives | 157 / 1000 |
| False negatives | 189 / 1000 |

Only the projection head (≈132K params) was trained; the ResNet18 backbone (≈11.2M params) was frozen throughout.

## Key insights

- **Why not softmax?** Softmax classifiers cannot generalize to new identities — adding a new person requires full retraining. Metric learning separates representation from comparison, enabling open-set recognition
- **Contrastive vs. triplet loss:** Contrastive loss operates on pairs (natural for LFW); triplet loss provides richer relative context but requires careful hard negative mining
- **Error patterns:** False positives (FP) occur when two different people share general appearance traits (age, skin tone, lighting); false negatives (FN) occur when the same person appears under very different lighting or pose — suggesting sensitivity to intra-personal variance
- **t-SNE:** Embeddings show partial but incomplete clustering — the small dataset (2.2K pairs) limits the model's ability to learn fully discriminative representations
- **Improvements:** Unfreeze backbone layers, add pose/lighting augmentation, switch to triplet loss with hard negative mining, or upgrade backbone to ResNet50/ViT

## Architecture

```
Input pair (img1, img2)
    │
    ├── ResNet18 backbone (frozen, ImageNet weights)
    │       └── Projection head: Linear(512→256) → ReLU → Dropout(0.5) → Linear(256→128) → L2 norm
    │
    └── Euclidean distance → Contrastive loss / threshold decision
```

## Training config

```
Dataset     : LFW (2,200 train pairs / 1,000 test pairs)
Embedding   : 128 dimensions, L2-normalized
Margin      : 1.0
Optimizer   : Adam (lr=3e-4, weight_decay=1e-4)
Scheduler   : StepLR (γ=0.5, step every 8 epochs)
Epochs      : 30, batch size 32
```

## Files

```
project5_face_recognition_contrastive_learning/
├── notebook.ipynb   # Siamese network, training, ROC, t-SNE, error analysis
└── report.pdf       # Written analysis
```

## Run

```bash
pip install torch torchvision scikit-learn matplotlib numpy
jupyter notebook notebook.ipynb
```

> LFW pairs are loaded via `sklearn.datasets.fetch_lfw_pairs` — downloaded automatically on first run.
