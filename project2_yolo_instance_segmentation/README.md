# Instance Segmentation with YOLOv8 — Pretrained Inference & Fine-Tuning

Exploration of **YOLOv8-nano** for instance segmentation: zero-shot inference on real-world urban scenes, followed by fine-tuning on a custom outdoor/action sports dataset with a systematic hyperparameter study.

## What was built

- Zero-shot segmentation pipeline on 3 real-world Chilean scenes (urban traffic, park, metro)
- Fine-tuning setup on a custom dataset (171 train / 49 val images, 10+ classes)
- Controlled hyperparameter experiments varying learning rate, batch size, image resolution, and training duration

## Pretrained model — zero-shot results

Tested on 3 scenes with `conf=0.3`:

| Scene | Objects detected | Notable findings |
|-------|-----------------|-----------------|
| Urban (traffic light) | 21 | Cars up to 91.1% conf; missed occluded vehicles |
| Parque Forestal | 12 | Person masks reasonable; bicycle under person not detected |
| Metro Línea 5 | ~10 | Front-row passengers only; occluded/distant people missed |

**Failure modes identified:** occlusion between objects, small/distant instances, classes with frontal/unusual viewpoints.

## Fine-tuning — hyperparameter study

| Experiment | mAP@50 Mask | mAP@50-95 Mask |
|-----------|------------|---------------|
| Baseline (lr=0.01, batch=16, imgsz=320) | 0.2742 | 0.1580 |
| lr=0.0001 | 0.2742 | 0.1580 |
| batch=32 | 0.2690 | 0.1635 |
| **imgsz=640** | **0.3673** | **0.2284** |
| epochs=60 | 0.2992 | 0.1688 |

**Image resolution had the largest impact (+34% mAP@50)** — higher resolution preserves fine boundary details critical for segmentation masks.

**Per-class breakdown (best model, imgsz=640):**
- Strong classes: person (0.835), canoe (0.694), paddle (0.651)
- Weak classes: dog (0.0), mountain (0.0), dirt (0.069) — due to low instance count or diffuse boundaries

## Key insights

- For segmentation tasks, input resolution dominates over learning rate and batch size at small dataset scales
- Classes with few training examples (<5 val instances) cannot be reliably evaluated, highlighting the importance of balanced datasets
- YOLOv8-nano is competitive for well-separated, texture-rich objects but struggles with diffuse or heavily occluded classes

## Files

```
project2_yolo_instance_segmentation/
├── notebook.ipynb   # Inference pipeline, fine-tuning experiments, metric plots
└── report.pdf       # Written analysis and discussion
```

## Run

```bash
pip install ultralytics roboflow matplotlib numpy pillow
jupyter notebook notebook.ipynb
```

> The fine-tuning section was run on Kaggle with GPU. Inference-only cells run on CPU.
