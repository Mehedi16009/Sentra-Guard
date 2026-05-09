# Sentra-Guard: A Real-Time Multilingual Defense Against Adversarial LLM Prompts

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.22628)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)

Official implementation of **Sentra-Guard**, a modular multilingual defense framework for detecting and mitigating adversarial jailbreak and prompt injection attacks against Large Language Models (LLMs).

Repository: [Sentra-Guard GitHub Repository](https://github.com/Mehedi16009/Sentra-Guard/)

---

## Overview

Sentra-Guard is a hybrid defense framework designed for **real-time adversarial prompt detection** across multilingual environments.

Unlike static moderation pipelines, Sentra-Guard combines:

- Fine-tuned transformer classifier
- Semantic adversarial retrieval memory (SBERT + FAISS)
- Zero-shot NLI reasoning
- Hybrid decision fusion
- Human-in-the-loop adaptive feedback

This architecture enables robust defense against:

- Prompt injection attacks
- Jailbreak attacks
- Obfuscated adversarial prompts
- Cross-lingual attack variants

### Core Pipeline

<img width="700" height="600" alt="Methodology" src="https://github.com/user-attachments/assets/1488f8f1-de80-4399-b756-a990ce631886" />

Figure 1. Sentra-Guard framework overview: An end-to-end multilingual human-AI defense pipeline for real-time adversarial LLM jailbreak detection, integrating input normalization, fine-tuned classifier inference, SBERT-FAISS semantic retrieval, zero-shot NLI reasoning, hybrid risk fusion, threshold-based decision making, and human-in-the-loop adaptive memory updates.





---

## Key Features

- Real-time harmful prompt detection
- Multilingual prompt normalization
- Semantic adversarial memory retrieval
- Hybrid classifier-retrieval fusion
- Zero-shot risk estimation
- Human-in-the-loop adaptive learning
- Modular reproducibility pipeline
- End-to-end benchmark evaluation

---

## Dataset Overview

This project uses the **JailBreakV-28K** benchmark dataset.

Dataset:

[JailBreakV-28K Dataset](JailbreakV-28K/JailBreakV-28k)

Dataset Paper:

[JailBreakV-28K Paper](https://arxiv.org/pdf/2404.03027)

### Dataset Statistics

| Category | Count |
|---|---:|
| Harmful Prompts | 28,000 |
| Benign Counterfactuals | 28,000 |
| Total Samples | 56,000 |

### Dataset Usage

Sentra-Guard uses:

- Harmful samples for adversarial retrieval memory
- Full balanced dataset for classifier training
- Validation split for threshold optimization
- Test split for final evaluation

---

## Running Prerequisites

## 1. Experimental Environment

### Hardware and Operating System

Recommended:

- Ubuntu 22.04+ / macOS / Windows (WSL preferred)
- NVIDIA GPU (recommended)
- CUDA-compatible environment

Minimum:

- 16GB RAM
- 20GB storage

Recommended:

- 32GB RAM
- 50GB storage

---

### Python Environment

Recommended Python version:

```bash
Python 3.10+
```

Virtual environment:

```bash
python -m venv sentra_env
source sentra_env/bin/activate
```

---

### Core Library Versions

| Library | Version |
|---|---:|
| Python | 3.10+ |
| PyTorch | 2.x |
| Transformers | latest |
| Datasets | latest |
| SentenceTransformers | latest |
| FAISS | latest |
| Scikit-learn | latest |
| Pandas | latest |
| NumPy | latest |

---

### Deep Learning Backend

Framework:

- PyTorch

Transformer backend:

- HuggingFace Transformers

Retrieval backend:

- SentenceTransformers

Indexing backend:

- FAISS

---

### GPU Configuration

Recommended:

```bash
CUDA 11.8+
```

GPU memory:

Minimum:

```bash
8GB VRAM
```

Recommended:

```bash
16GB+ VRAM
```

---

## Repository Structure

```text
Sentra-Guard/
│
├── notebooks/
│   ├── sentra_guard_pipeline.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── checkpoints/
│   ├── classifier/
│   ├── retrieval_index/
│
├── outputs/
│   ├── metrics.json
│   ├── predictions.csv
│   ├── ablation.csv
│   ├── baseline_results.csv
│
├── wiki/
│   ├── methodology.md
│   ├── experimental_results.md
│
├── requirements.txt
├── run_sentra_guard.py
├── README.md
└── LICENSE
```

---

## How to Run the Project in GitHub

Clone the repository:

```bash
git clone https://github.com/Mehedi16009/Sentra-Guard.git
cd Sentra-Guard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python run_sentra_guard.py
```

---

## How to Run the Project (Local PC)

### Step 1: Create virtual environment

```bash
python -m venv sentra_guard_env
source sentra_guard_env/bin/activate
```

---

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3: Run notebook pipeline

```bash
jupyter notebook
```

Open:

```text
notebooks/sentra_guard_pipeline.ipynb
```

Run all cells sequentially.

---

### Step 4: Export outputs

Generated outputs:

```text
outputs/
```

---

## Required Python Packages

Install manually:

```bash
pip install torch
pip install transformers
pip install datasets
pip install sentence-transformers
pip install faiss-cpu
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install tqdm
pip install accelerate
pip install evaluate
```

Or:

```bash
pip install -r requirements.txt
```

---

## Pipeline Stages

### Phase 1 — Environment Setup

Dependency initialization.

### Phase 2 — Dataset Preparation

Load and clean dataset.

### Phase 3 — Classifier Training

Fine-tune transformer classifier.

### Phase 4 — Retrieval Memory Construction

Build adversarial semantic memory.

### Phase 5 — Zero-Shot Safety Reasoning

Zero-shot harmfulness inference.

### Phase 6 — Hybrid Decision Fusion

Combine risk signals.

### Phase 7 — Threshold Optimization

Optimize decision threshold.

### Phase 8 — Final Evaluation

Benchmark model performance.

### Phase 9 — Ablation Analysis

Component contribution analysis.

### Phase 10 — Artifact Export

Save reproducibility outputs.

---

## Experimental Results

Sentra-Guard achieves:

| Metric | Score |
|---|---:|
| Detection Rate | 99.96% |
| F1 Score | 1.00 |
| ROC-AUC | 1.00 |
| Attack Success Rate (ASR) | 0.004% |

Benchmark performance follows the paper results. :contentReference[oaicite:3]{index=3}

---

## Reproducibility

To reproduce results:

1. Install dependencies  
2. Run notebook pipeline  
3. Run evaluation pipeline  
4. Export artifacts

Outputs:

- metrics.json
- predictions.csv
- ablation.csv
- baseline_results.csv

---

## Citation

If you use this work, please cite:

```bibtex
@article{hasan2025sentra,
  title={Sentra-Guard: A Multilingual Human-AI Framework for Real-Time Defense Against Adversarial LLM Jailbreaks},
  author={Hasan, Md Mehedi and Mehedi, Sk Tanzir and Rahman, Ziaur and Mostafiz, Rafid and Hossain, Md Abir},
  journal={arXiv preprint arXiv:2510.22628},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License.

See:

```text
LICENSE
```

---

## Ethical Statement

Sentra-Guard is developed strictly for defensive AI safety research.

This repository is intended for:

- harmful prompt detection research
- LLM safety benchmarking
- adversarial robustness evaluation

This repository must not be used for:

- generating harmful prompts
- bypassing model safeguards
- deploying offensive jailbreak attacks

Users are responsible for ethical and lawful use.

---

## Contact

Md Mehedi Hasan <br>
Mawlana Bhashani Science and Technology University <br>
- GitHub: [Sentra-Guard Repository](https://github.com/Mehedi16009/Sentra-Guard/))
- Personal Website: [Portfolio](https://md-mehedi-hasan-resume.vercel.app/)
- Email: [mehedi.hasan.ict@mbstu.ac.bd](mehedi.hasan.ict@mbstu.ac.bd)
