# Mechanistic Analysis of Demonstration Conflicts in In-Context Learning

This repository contains code for investigating how large language models process conflicting demonstrations during in-context rule inference.

## Quick Start

### 1. Evaluate Model Performance Under Corruption

```bash
bash run/1-run_calibrated_corruption_test.sh
```

This evaluates how conflicting demonstrations affect model performance on operator induction tasks.

### 2. Interpretability Analysis

The `interp/` folder contains mechanistic analysis tools:

```
interp/
├── helper.py                        # Common utilities for hook management and token categorization
├── representation_extractor.py      # Extract model internal representations
├── heads/
│   ├── attention_extractor.py       # Extract attention patterns across layers
│   ├── find_susc_heads.py          # Identify susceptible heads in late layers
│   ├── comprehensive_head_ablation.py  # Ablation experiments
│   └── extract_acs_metrics.py      # Extract attention and contribution scores
├── logit_lens/
│   ├── logit_lens.py               # Decode predictions at intermediate layers
│   └── run_logit_lens.py           # Run logit lens analysis
└── probes/
    ├── linear_probe_trainer.py      # Train probes to detect rule encoding
    ├── linear_probe_evaluator.py   # Evaluate trained probes
    └── data_preprocessing.py       # Prepare data for probe training
```

Key findings:
- **Linear probes**: Models encode both correct and corrupted rules in early-middle layers
- **Logit lens**: Prediction confidence emerges sharply in late layers
- **Vulnerability heads**: Early-middle layer heads with positional attention bias
- **Susceptible heads**: Late layer heads that reduce support for correct predictions under corruption