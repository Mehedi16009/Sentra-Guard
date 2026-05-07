# Sentra-Guard

**A hybrid, real-time defense framework for detecting adversarial jailbreak prompts in large language models.**

Sentra-Guard integrates multilingual translation, SBERT-FAISS semantic retrieval, fine-tuned transformer classification, zero-shot natural language inference, and human-in-the-loop adaptive feedback into a unified detection pipeline. The system processes each prompt through three parallel inference branches and fuses their outputs into a single risk score, enabling detection of known, obfuscated, and zero-day jailbreak strategies across more than 100 languages.

> **Under Review:** ACM Transactions on Privacy and Security (TOPS)

---

## Key Contributions

1. **Hybrid multi-branch architecture.** Three complementary inference branches — semantic retrieval, fine-tuned classification, and zero-shot entailment — operate in parallel. Their fusion resolves the coverage limitations of any single approach.

2. **Multilingual normalization.** All prompts are translated to English via neural machine translation before entering the detection pipeline, providing uniform semantic alignment across more than 100 source languages and directly mitigating code-mixed and script-manipulation attacks.

3. **Near-perfect detection performance.** On HarmBench-28K, Sentra-Guard achieves 99.98% accuracy, 100% precision, 99.97% recall, AUC = 1.00, and an Attack Success Rate (ASR) of 0.004% at 47 ms average inference latency. This represents at least a 20-fold ASR reduction over the next strongest published system.

4. **Cross-lingual robustness.** Detection rates exceed 96% across English, French, Spanish, Arabic, and Hindi on four LLM backends (GPT-4o, Gemini Flash, Claude 3 Opus, Mistral 7B), with inference latency never exceeding 56 ms.

5. **Efficient adaptive feedback.** The human-in-the-loop (HITL) mechanism updates the semantic retrieval index via direct vector injection rather than full model retraining, reducing adaptation lag by over 90%. Injecting 500 confirmed adversarial prompts improved recall by 4.2% and reduced false positives by 11% without any gradient computation.

6. **Deployment-agnostic design.** The modular pipeline integrates with GPT-4o, Claude, Gemini, LLaMA, and Mistral without platform-specific modifications, in both pre-inference screening and post-inference moderation modes.

---

## Repository Structure

```
Sentra-Guard/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   └── default_config.json           # Canonical hyperparameters
├── datasets/
│   ├── README.md                     # Acquisition and preparation instructions
│   └── schema.md                     # Unified data schema
├── models/
│   └── classifier_checkpoint/        # Fine-tuned model checkpoint (not tracked by git)
├── attacks/
│   └── jailbreak_strategies.md       # Eight catalogued attack types
├── defenses/
│   └── hitl_protocol.md              # HITL escalation and vector injection protocol
├── scripts/
│   ├── preprocess.py
│   ├── train_classifier.py
│   ├── build_faiss_index.py
│   ├── score_hybrid.py
│   ├── evaluate.py
│   ├── ablation.py
│   ├── optimize_threshold.py
│   ├── optimize_weights.py
│   └── demo_multilingual.py
├── notebooks/
│   └── Sentra_Guard_Full_Pipeline.ipynb
├── outputs/                          # Generated at runtime; not committed to git
├── figures/
│   ├── Figure_2.pdf                  # Architecture overview
│   ├── Figure_3.png                  # ROC curve
│   ├── Figure_4.png                  # Precision-Recall curve
│   ├── Figure_5.png                  # Confusion matrix and comparative chart
│   ├── Figure_6.png                  # Multilingual heatmap
│   └── Figure_7.png                  # Pareto analysis
└── wiki/
    ├── Methodology.md
    ├── Experimental_Design.md
    ├── Reproducibility_Guide.md
    └── File_By_File_Guide.md
```

**configs/** holds the exported `SentraGuardConfig` dataclass as JSON. All hyperparameters (learning rate, batch size, fusion weights, threshold) are read from this file; no hardcoded values appear in scripts.

**scripts/** contains one script per pipeline stage. Each script is self-contained and can be run independently, given that upstream outputs already exist.

**outputs/** is populated at runtime. The full set of expected output files is described in the Output Directory section below.

**wiki/** contains all technical depth excluded from this README: full pipeline equations, architectural module descriptions, per-file documentation, and GPU-specific runtime guidance.

---

## Installation

### Requirements

- Python 3.9 or later
- CUDA-capable GPU with at least 8 GB VRAM (Tesla T4 recommended; Apple M1 CPU supported for auxiliary tasks)
- Git

### Clone the Repository

```bash
git clone https://github.com/<your-org>/Sentra-Guard.git
cd Sentra-Guard
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Core packages:**

```
torch>=2.0
transformers>=4.38
datasets>=2.18
sentence-transformers>=2.6
faiss-cpu>=1.7
scikit-learn>=1.3
pandas>=2.0
numpy>=1.25
matplotlib>=3.7
tqdm>=4.66
langdetect>=1.0.9
sentencepiece>=0.1.99
```

For GPU-accelerated FAISS, replace `faiss-cpu` with `faiss-gpu` and ensure your CUDA installation matches your PyTorch build.

---

## Dataset Preparation

Sentra-Guard uses two datasets, both loaded from Hugging Face. No manual download is required.

**Primary dataset (D1) — HarmBench-28K:**

```
Dataset ID: YinkaiW/harmbench-dataset
```

This benchmark covers adversarial prompts across misinformation, cyberattacks, financial scams, and hate speech. Labels are binarized: 1 = harmful, 0 = benign. Duplicate entries are removed, system-role instructions are filtered, and metadata extraneous to user-issued text is discarded.

**External benchmark (D2) — JailbreakV-28K:**

```
Dataset ID: JailbreakV-28K/JailBreakV-28k
Config: JailBreakV_28K
```

D2 is reserved exclusively for inference-time evaluation. It is never used during training. Text-based jailbreak prompts are extracted using the `transfer_from_llm` flag and the `format` field (values: template, persuade, logic). A balanced set of benign samples is drawn from D1 to match the D2 harmful count.

**Preprocessing steps (applied identically to both datasets):**

1. Unicode NFKC normalization and control character removal.
2. System-role instruction filtering (prefixes: `system:`, `[system]`, `<system>`).
3. Duplicate removal by normalized text.
4. Language detection using `langdetect`.
5. Translation of non-English prompts to English using Helsinki-NLP MarianMT models (supported: Bengali, Spanish, Hindi, Arabic, Mandarin).
6. Dataset split: training 70%, validation 15%, test 15%, using stratified sampling to preserve class balance.

Run preprocessing via:

```bash
python scripts/preprocess.py --config configs/default_config.json
```

This writes `d1_clean.csv`, `d1_train.csv`, `d1_val.csv`, `d1_test.csv`, and `d2_external.csv` to `outputs/cache/`.

---

## Training Pipeline

The classifier is a DistilBERT model (`distilbert-base-uncased`) fine-tuned for binary sequence classification (harmful vs. benign). For production deployment, DeBERTa-v3 (`deberta-v3-base`) is the intended backbone, as described in the manuscript.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Learning rate | 2 × 10⁻⁵ |
| Batch size | 8 (manuscript) / 16 (code default) |
| Epochs | 3 (manuscript) / 5 with early stopping (patience = 2) |
| Max sequence length | 64 tokens |
| Optimizer | AdamW |
| Schedule | Linear decay with 10% warmup |
| Max grad norm | 1.0 |
| Weight decay | 0.01 |

**Run training:**

```bash
python scripts/train_classifier.py \
    --config configs/default_config.json \
    --train outputs/cache/d1_train.csv \
    --val outputs/cache/d1_val.csv \
    --output models/classifier_checkpoint
```

The best checkpoint (by validation F1) is saved to `models/classifier_checkpoint/`. Training history is written to `outputs/training_logs.csv`. Approximate wall time: two hours on a Tesla T4 GPU.

---

## Semantic Retrieval Index Construction

This step encodes all training prompts using Sentence-BERT and builds a FAISS flat inner-product index for cosine similarity retrieval.

**Models used:**

- SBERT encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Index type: `faiss.IndexFlatIP` (exact search, cosine similarity via L2-normalized embeddings)

**Run index construction:**

```bash
python scripts/build_faiss_index.py \
    --config configs/default_config.json \
    --train outputs/cache/d1_train.csv \
    --output outputs/retrieval
```

This writes `sentra_guard.index`, `knowledge_base.csv`, and `train_embeddings.npy` to `outputs/retrieval/`.

---

## Attack Generation Pipeline

Sentra-Guard is evaluated against eight adversarial jailbreak strategies. These are not generated computationally; they are drawn from the HarmBench-28K and JailbreakV-28K datasets, which contain pre-existing adversarial prompts covering:

- Role-playing (persona assignment without ethical constraints)
- Instruction override (fictitious system state manipulation)
- Obfuscated encoding (leetspeak and character substitution)
- Multi-turn crescendo (progressive harmful escalation across turns)
- Fictional narrative (harmful extraction embedded in creative framing)
- Persuasion via ethical appeal (legitimizing pretext)
- Meta-prompting (model generates its own bypass strategy)
- Few-shot imitation (harmful prompt embedded within benign Q&A context)

See `attacks/jailbreak_strategies.md` for full examples, bypass strategies, and observed model responses, as reported in Table 2 of the manuscript.

---

## Defense Execution Pipeline

Following preprocessing and index construction, the full three-branch hybrid scoring pipeline is executed.

**Inference branches:**

1. **Semantic retrieval branch:** Each normalized prompt is encoded by SBERT, and its top-5 nearest neighbors are retrieved from the FAISS index. A weighted similarity score is computed against known harmful and benign exemplars.

2. **Classifier branch:** The fine-tuned DistilBERT (or DeBERTa-v3) classifier produces a calibrated probability score over the binary harmful/benign label space.

3. **Zero-shot branch:** `facebook/bart-large-mnli` evaluates semantic entailment between the normalized prompt and the candidate labels `"harmful"` and `"benign"`, using the hypothesis template `"The user's intent is {label}."`.

**Decision fusion:** Branch outputs are combined as a weighted linear sum to produce a final risk score. If this score meets or exceeds the decision threshold, the prompt is classified as harmful. Prompts with scores near the threshold are flagged for HITL review.

**Run hybrid scoring:**

```bash
python scripts/score_hybrid.py \
    --config configs/default_config.json \
    --split outputs/cache/d1_test.csv \
    --embeddings outputs/retrieval/train_embeddings.npy \
    --index outputs/retrieval/sentra_guard.index \
    --checkpoint models/classifier_checkpoint \
    --output outputs/test_scored.csv
```

---

## Evaluation Pipeline

**Run default evaluation:**

```bash
python scripts/evaluate.py \
    --scored outputs/test_scored.csv \
    --output outputs/metrics.csv
```

Reported metrics: accuracy, precision, recall, F1-score, ASR, AUC-ROC, PR-AUC, inference latency, and HITL escalation rate.

**Run ablation study:**

```bash
python scripts/ablation.py \
    --test-scored outputs/test_scored.csv \
    --d2-scored outputs/d2_scored.csv \
    --config configs/default_config.json \
    --output outputs/ablation.csv
```

**Run threshold optimization:**

```bash
python scripts/optimize_threshold.py \
    --val-scored outputs/val_scored.csv \
    --config configs/default_config.json \
    --output outputs/threshold_search.csv
```

**Run weight optimization:**

```bash
python scripts/optimize_weights.py \
    --val-scored outputs/val_scored.csv \
    --config configs/default_config.json \
    --output outputs/weight_search.csv
```

---

## Reproducing Experimental Results

Execute the following steps in order. Each step depends on the outputs of the step before it.

```
Step 1:  python scripts/preprocess.py           → outputs/cache/
Step 2:  python scripts/train_classifier.py     → models/classifier_checkpoint/
Step 3:  python scripts/build_faiss_index.py    → outputs/retrieval/
Step 4:  python scripts/score_hybrid.py (val)   → outputs/val_scored.csv
Step 5:  python scripts/score_hybrid.py (test)  → outputs/test_scored.csv
Step 6:  python scripts/score_hybrid.py (d2)    → outputs/d2_scored.csv
Step 7:  python scripts/optimize_threshold.py   → outputs/threshold_search.csv
Step 8:  python scripts/optimize_weights.py     → outputs/weight_search.csv
Step 9:  python scripts/evaluate.py             → outputs/metrics.csv
Step 10: python scripts/ablation.py             → outputs/ablation.csv
Step 11: python scripts/demo_multilingual.py    → console output
```

Alternatively, the complete pipeline can be run end-to-end using the provided notebook:

```
notebooks/Sentra_Guard_Full_Pipeline.ipynb
```

Open in Google Colab with a T4 GPU runtime. Run all 18 cells in order. Artifacts are saved automatically to `sentra_guard_artifacts/` and backed up to Google Drive at the end of Step 18.

**Expected results on HarmBench-28K (D1 test split):**

| Metric | Expected Value |
|---|---|
| Accuracy | 99.98% |
| Precision | 100.00% |
| Recall | 99.97% |
| F1-Score | 99.98% |
| AUC-ROC | 1.00 |
| ASR | 0.004% |
| Avg. Latency | 47 ms |

---

## Output Directory Description

After running the full pipeline, `outputs/` contains:

| File | Description |
|---|---|
| `cache/d1_clean.csv` | Preprocessed D1 with normalized text and language labels |
| `cache/d1_train.csv` | Training split (70% of D1 internal) |
| `cache/d1_val.csv` | Validation split (15%) |
| `cache/d1_test.csv` | Test split (15%) |
| `cache/d2_external.csv` | Balanced held-out external benchmark (D2) |
| `cache/val_scored_default.csv` | Scored validation set with per-branch scores |
| `cache/test_scored_default.csv` | Scored test set |
| `cache/d2_scored_default.csv` | Scored D2 held-out set |
| `retrieval/sentra_guard.index` | FAISS index over training embeddings |
| `retrieval/knowledge_base.csv` | Training sample metadata for retrieval lookup |
| `retrieval/train_embeddings.npy` | SBERT embeddings of training set |
| `training_logs.csv` | Per-epoch loss, accuracy, precision, recall, F1 (train and val) |
| `predictions.csv` | Final predictions with scores and HITL flags for all evaluated samples |
| `metrics.csv` | Aggregate evaluation metrics for D1 test and D2 held-out sets |
| `ablation.csv` | Per-component and per-combination ablation results |
| `threshold_search.csv` | F1, ASR, and AUC at each threshold value across validation set |
| `weight_search.csv` | Best configuration per (w1, w2, w3) triplet from grid search |
| `confusion_matrix.csv` | TP, TN, FP, FN per dataset split |
| `config.json` | Final configuration including selected weights, threshold, and delta |
| `artifact_manifest.json` | Complete listing of all generated artifact files |
| `classifier_checkpoint/` | Saved HuggingFace model and tokenizer files |

---

## Citation

If you use Sentra-Guard in your research, please cite the following manuscript:

```bibtex
@article{sentra-guard-tops,
  title   = {Sentra-Guard: A Hybrid Real-Time Defense Framework for Multilingual Adversarial Jailbreak Detection in Large Language Models},
  journal = {ACM Transactions on Privacy and Security},
  note    = {Under review},
  year    = {2025}
}
```

---

## License

This repository is released for reproducibility and research purposes. The redacted source code release excludes exploitable attack templates, consistent with responsible disclosure practices. See `LICENSE` for details.

---

## Ethical Statement

All datasets (D1 and D2) are open-source red-teaming corpora. No user-identifiable information was used. All LLM evaluations were conducted under controlled conditions. No harmful model outputs were released. Sentra-Guard exposes classification confidence, retrieval matches, and HITL traces for operator inspection.
