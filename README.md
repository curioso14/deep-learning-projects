# Deep Learning & Computer Vision Projects

A collection of independent deep learning and computer vision projects covering convolutional neural networks, object detection & segmentation, and self-supervised visual representations.

**Course:** IIC3697 – Deep Learning · Pontificia Universidad Católica de Chile  
**Stack:** Python · PyTorch · torchvision · Ultralytics YOLOv8 · DINOv2 · NumPy · Matplotlib

---

## Projects

| # | Project | Topics | Dataset |
|---|---------|--------|---------|
| 1 | [ResNet18 vs PlainNet18 on CIFAR-100](./project1_resnet_cifar100/) | Residual connections, skip connections, convergence analysis | CIFAR-100 |
| 2 | [Instance Segmentation with YOLOv8](./project2_yolo_instance_segmentation/) | Object detection, instance segmentation, hyperparameter tuning | Custom outdoor dataset |
| 3 | [Visual Representations with DINOv2](./project3_dinov2_visual_representations/) | Self-supervised ViT features, feature matching, transfer learning | COCO, custom pairs, CIFAR-10 |

---

## Skills

- **From-scratch architecture implementation** in PyTorch (ResNet18, PlainNet18, custom MLP)
- **Training pipeline design:** SGD + momentum, cosine annealing LR, data augmentation
- **Fine-tuning & hyperparameter search** on real datasets with YOLOv8
- **Self-supervised feature extraction** with Vision Transformers (DINOv2 ViT-S/14)
- **Quantitative evaluation:** accuracy curves, mAP@50, cosine similarity, parameter counts

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/deep-learning-projects.git
pip install torch torchvision ultralytics timm matplotlib numpy scikit-learn
```

Each project has its own `README.md` with detailed results and instructions.
