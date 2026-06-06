# News Summarization — LSTM, Seq2Seq with Attention, BART & T5

End-to-end exploration of automatic text summarization on the CNN/DailyMail dataset, progressing from a vanilla LSTM encoder-decoder baseline through a Bahdanau attention model to fine-tuned transformer architectures (BART and T5).

## What was built

- **LSTM Encoder-Decoder baseline** trained from scratch with teacher forcing (p=0.5), gradient clipping, and a custom 30K-word vocabulary
- **Seq2Seq + Bahdanau Attention** — encoder returns all hidden states; attention module computes additive energy scores and a dynamic context vector at each decoding step
- **BART fine-tuning** (`facebook/bart-base`, 139M params) via HuggingFace `Seq2SeqTrainer` with FP16, gradient accumulation, and early stopping
- **T5 fine-tuning** (`t5-small`, 60M params) with `"summarize: "` prefix in text-to-text format

## Results

| Model | Params | Pre-trained | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|--------|------------|---------|---------|---------|
| LSTM Enc-Dec | ~10M | No | 4.85 | 0.35 | 4.48 |
| Seq2Seq + Attention | ~12M | No | 11.24 | 1.21 | 9.62 |
| T5-small | 60M | Yes (750 GB) | 32.27 | 12.72 | 23.36 |
| **BART-base** | **139M** | **Yes (160 GB)** | **32.89** | **12.16** | **22.89** |

BART's denoising autoencoding pretraining objective (masked spans, shuffled sentences) is directly aligned with abstractive summarization, explaining its edge over T5 despite T5's larger pretraining corpus.

## Key insights

- The LSTM bottleneck (entire article → single 512-dim vector) causes information loss at scale; attention addresses this by allowing the decoder to selectively read encoder states
- Attention heatmaps revealed degenerate behavior after only 5 epochs: the decoder fixated on 2–3 fixed positions regardless of which word was being generated — more training and regularization needed
- Pretraining matters far more than architecture tuning: BART with 3 fine-tuning epochs achieves ~7× the ROUGE-1 of the LSTM with 5 epochs, despite similar data budgets
- Exposure bias (teacher forcing gap) is a key limitation of LSTM-based generation at inference time

## Training config

```
Dataset     : CNN/DailyMail  (10K train / 1K val / 500 test)
Vocab size  : 30,000 tokens (custom tokenizer)
Article len : 400 tokens max
Summary len : 100 tokens max (LSTM/Seq2Seq), 128 (BART/T5)

LSTM/Seq2Seq:
  Optimizer : Adam (lr=1e-3)
  Epochs    : 5, batch size 32

BART/T5:
  Optimizer : AdamW (lr=3e-5 / 5e-5)
  Epochs    : 3, effective batch 32 (8 × 4 grad accum)
  Early stopping: patience 2
```

## Files

```
project4_news_summarization_lstm_bart_t5/
├── notebook.ipynb   # Full pipeline: data loading, models, training, ROUGE evaluation
└── report.pdf       # Written analysis
```

## Run

```bash
pip install torch transformers datasets evaluate rouge_score sentencepiece
jupyter notebook notebook.ipynb
```

> GPU strongly recommended. BART/T5 fine-tuning was run on Kaggle with a P100.
