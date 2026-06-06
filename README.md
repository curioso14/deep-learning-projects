# Deep Learning & Computer Vision Projects

A collection of deep learning projects covering CNNs, object detection & segmentation, self-supervised visual representations, NLP with transformers, and metric learning.

**Stack:** Python · PyTorch · torchvision · HuggingFace Transformers · Ultralytics YOLOv8 · DINOv2 · scikit-learn · NumPy · Matplotlib

---

## Projects

| # | Project | Topics | Dataset |
|---|---------|--------|---------|
| 1 | [ResNet18 vs PlainNet18 on CIFAR-100](./project1_resnet_cifar100/) | Residual connections, skip connections, convergence analysis | CIFAR-100 |
| 2 | [Instance Segmentation with YOLOv8](./project2_yolo_instance_segmentation/) | Object detection, instance segmentation, hyperparameter tuning | Custom outdoor dataset |
| 3 | [Visual Representations with DINOv2](./project3_dinov2_visual_representations/) | Self-supervised ViT features, feature matching, transfer learning | COCO, custom pairs, CIFAR-10 |
| 4 | [News Summarization — LSTM, Attention, BART & T5](./project4_news_summarization_lstm_bart_t5/) | Seq2Seq, Bahdanau attention, transformer fine-tuning, ROUGE | CNN/DailyMail |
| 5 | [Face Recognition with Contrastive Learning](./project5_face_recognition_contrastive_learning/) | Siamese network, contrastive loss, metric learning, t-SNE | LFW |

---

## Skills

- **From-scratch architecture implementation** in PyTorch: ResNet18, PlainNet18, LSTM Enc-Dec, Seq2Seq with attention, Siamese network, MLP
- **Training pipeline design:** SGD/Adam, cosine annealing, teacher forcing, gradient clipping, early stopping
- **Fine-tuning large pretrained models:** BART-base and T5-small via HuggingFace `Seq2SeqTrainer`; YOLOv8-nano via Ultralytics
- **Self-supervised & metric learning:** DINOv2 feature extraction and cross-image matching; contrastive loss for open-set face verification
- **Evaluation:** ROUGE-1/2/L, mAP@50, AUC-ROC, cosine similarity, t-SNE visualization, error analysis

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/deep-learning-projects.git
pip install torch torchvision transformers datasets ultralytics timm scikit-learn matplotlib numpy rouge_score sentencepiece
```

Each project folder contains its own `README.md` with specific setup, results, and instructions.
