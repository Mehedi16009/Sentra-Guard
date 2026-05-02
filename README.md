# Sentra-Guard: A Real-Time Multilingual Defense Against Adversarial LLM Prompts

Sentra-Guard is a modular defense framework for detecting adversarial jailbreak and prompt-injection attacks against large language models (LLMs). The system combines multilingual normalization, semantic retrieval, transformer-based classification, zero-shot semantic reasoning, weighted risk fusion, and Human-in-the-Loop (HITL) adaptation to provide fast and robust protection across multilingual and obfuscated attack scenarios.

This repository provides the official implementation of the Sentra-Guard pipeline and supports reproducibility for the accompanying ACM TOPS manuscript.

## Core Features <br>
i. Multilingual prompt normalization (100+ languages) <br>
ii. SBERT-based semantic retrieval with FAISS <br>
iii. Fine-tuned DeBERTa-v3 binary harmful prompt classifier <br> 
iv. Zero-shot NLI branch for out-of-distribution attack detection <br>
v. Validation-calibrated fusion scoring <br>
vi. Human-in-the-loop adaptive vector injection <br>
vii. Baseline benchmarking and evaluation pipeline <br>

## Repository Structure

```text
Sentra-Guard/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── manuscript/
│   └── sentra_guard_paper.pdf
│
├── notebooks/
│   └── sentra_guard_reproducibility.ipynb
│
├── src/
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   ├── retrieval.py
│   ├── fusion.py
│   ├── inference.py
│   └── utils.py
│
├── scripts/
│   ├── run_train.sh
│   ├── run_eval.sh
│   ├── run_ablation.sh
│   ├── run_threshold_search.sh
│   ├── run_weight_search.sh
│   └── sanity_test.sh
│
├── artifacts/
│   ├── checkpoints/
│   ├── metrics/
│   │   ├── metrics.csv
│   │   ├── ablation.csv
│   │   ├── threshold_search.csv
│   │   ├── weight_search.csv
│   │   ├── confusion_matrix.csv
│   │   └── predictions.csv
│   │
│   └── figures/
│       ├── roc_curve.png
│       ├── pr_curve.png
│       └── architecture.png
│
├── tests/
│   ├── label_audit.py
│   ├── sanity_test.py
│   └── smoke_test.py
│
└── docs/
    ├── reproduction.md
    └── dataset_notes.md

```


## Installation
``pip install -r requirements.txt``

## Dataset Setup

Place datasets under:
``
data/raw/``

Required schema:

prompt (text)<br>
label (0 = benign, 1 = harmful) <br>

Optional:

language <br>
source <br>
role <br>

## Training
``python scripts/train.py``
## Build Retrieval Index
``python scripts/build_index.py``
## Inference
``python scripts/inference.py --prompt "How can I bypass content moderation?"``

## Google Colab

Open:
``
notebooks/sentra_guard_colab.ipynb
``
Run all cells sequentially.

## Model Pipeline<br>
1. Language detection <br>
2. Translation normalization<br>
3. SBERT embedding generation<br>
4. FAISS nearest-neighbor retrieval<br>
5. DeBERTa-v3 harmful prompt classification<br>
6. Zero-shot entailment scoring<br>
7. Fusion risk computation <br>
8. HITL escalation if uncertain <br>


## Decision Fusion

The final risk score is computed as:

```text
S_final = (w1 * P_C) + (w2 * R_score) + (w3 * P_Z)
```


Fusion weights are tuned on validation data and stored in:
``
artifacts/config/run_config.json
``

## HITL Logic

Escalation triggers: <br>

1. Escalation condition:

```text
abs(S_final - theta) < delta
```

2. branch disagreement > threshold <br>

Confirmed harmful prompts:

i. inserted into FAISS index <br>
ii. appended into training buffer <br>


## Reproducibility

Fixed seeds:

i. Python <br>
ii. NumPy <br>
iii. PyTorch <br>

Saved artifacts:

i. trained classifier <br>
ii. FAISS index <br>
iii. metrics <br>
iv. run configuration <br>


## Citation

Add manuscript citation after publication.
