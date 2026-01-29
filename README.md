# Promotion in Multi‑Stakeholder Recommendation Systems

This repository provides an experimental pipeline for the paper **Promotion in Multi‑Stakeholder Recommendation Systems**. The core task is **promotion-aware post‑processing**: given baseline recommender scores, compute adjusted per‑user ranking distributions that satisfy **promoter exposure (uplift) constraints** while minimizing deviation from the original personalized policy.

---

### Solvers implemented here

* **Population-level solver (shared dual variables):** a single dual vector is solved once and applied to all users (fast, stable, scalable).
* **Individual-level solver:** solves a per-user projection.
* **Closed-form solutions:** supported for specific cases (e.g., non-overlapping promoters).

### Baselines included

All methods run post‑training (no retraining of the recommender):

* IPF (iterative proportional fitting)
* Mixed-integer formulation (CVXPY; with fallback on infeasibility)
* Sürer-Dual (subgradient dual relaxation)
* Population Dual-Boost (shared boost learned via projected gradient)

---


It provides:

* Training recommender models using RecBole
* Exporting full ranking score matrices
* Generating aligned item metadata (promoters)
* Running promotion‑aware optimization experiments

The framework allows you to evaluate exposure constraints, fairness, and multi‑stakeholder trade‑offs while minimally deviating from personalized rankings.

---

### Workflow Overview

Step 1 — Train a recommender

Step 2 — Generate metadata

Step 3 — Run experiments

You can either generate everything locally or download preprocessed artifacts.

---

### Environment

```bash
conda env create -f environment.yml
conda activate promotion
pip install -r requirements.txt
```
---

## Step 1 — Train Recommender

### MovieLens

```bash
python promotion_data_ml.py
```

### Steam

```bash
python promotion_data_Steam.py
```

### Other datasets

Any RecBole‑compatible dataset can be used:

[https://github.com/RUCAIBox/RecBole/blob/master/recbole/properties/dataset/url.yaml](https://github.com/RUCAIBox/RecBole/blob/master/recbole/properties/dataset/url.yaml)

Ensure the output files follow the same format as the examples below.

---

### Step 2 — Generate Metadata

Run the notebook:

```
promotion_data_metadata.ipynb
```

This creates:

* aligned item metadata CSV
* promoter group columns
* experiment‑ready format

This file defines which items belong to each promoter (genre / publisher / tag / etc.).

---

### Optional: Download Preprocessed Data

If you want to skip training:

[https://drive.google.com/drive/folders/1x0v9WdOl-2I447Jpgg5Ls33dOPLGoYGN?usp=sharing](https://drive.google.com/drive/folders/1x0v9WdOl-2I447Jpgg5Ls33dOPLGoYGN?usp=sharing)

---

## Step 3 — Run Experiments

```bash
python run_experiments.py
```

Inside the script, choose your dataset configuration by setting:

* `SCORES_PATH` – full score matrix
* `POS_U_PATH` – positive user ids
* `POS_I_PATH` – positive item ids
* `META_PATH` – metadata CSV
* `PROMOTERS` – exposure groups

---

### Example Configurations

#### MovieLens‑100K

```python
SCORES_PATH = "scores_memmap/sasrec_ml-100k_full_scores_float16.npy"
POS_U_PATH  = "ml100k_pos_u.npy"
POS_I_PATH  = "ml100k_pos_i.npy"
META_PATH   = "ml100k_item_metadata_aligned.csv"
PROMOTERS   = ["Action", "Comedy", "Romance"]
```

---

#### MovieLens‑10M

```python
SCORES_PATH = "scores_memmap/sasrec_ml-10m_full_scores_float16.npy"
POS_U_PATH  = "ml10m_pos_u.npy"
POS_I_PATH  = "ml10m_pos_i.npy"
META_PATH   = "ml-10m_item_metadata_aligned.csv"
PROMOTERS   = ["Action", "Comedy", "Romance"]
```

---

#### Steam (Overlapping promoters)

```python
SCORES_PATH = "scores_memmap/SASRec_steam-merged_test_full_scores_f16.npy"
POS_U_PATH  = "scores_memmap/SASRec_steam-merged_test_user_ids_i32.npy"
POS_I_PATH  = "scores_memmap/SASRec_steam-merged_test_pos_item_i32.npy"
META_PATH   = "steam_meta_V2.csv"
PROMOTERS   = ["Indie", "Action", "Casual"]
```

---

#### Steam (Non‑overlapping promoters)

Use these for disjoint groups:

```python
PROMOTERS = [
    "Ubisoft - San Francisco",
    "SmiteWorks USA, LLC",
    "Capcom",
]
```

---

## Evaluation protocol

* Recommendation quality: Recall and nDCG@{1,5,10}
* Ranking similarity: Spearman rank correlation over top-100 (Spearman@100)
* Constraint satisfaction:
  * Average constraint violation per user
  * Fail rate: fraction of users whose violation exceeds **1e-7**
* Runtime: average wall-clock time per user (ms)

---

## Memory Tips

Large datasets may require significant RAM because the score matrix stores:

users × items

So in default we used:

* float16
* numpy memmap

If you want you can:
* run on a subsample, or run in batches (CPU RAM dependent)

---


