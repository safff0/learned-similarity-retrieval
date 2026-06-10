# Retrieval with Learned Similarities (RAILS / MoL) — Reproduction Study

We reproduce and analyze **Retrieval with Learned Similarities** (RAILS), which proposes *Mixture-of-Logits* (MoL) as a drop-in replacement for dot-product similarity in sequential recommendation. Our experiments on ML-1M show that MoL yields substantial gains over SASRec (+20% HR@10, +31% NDCG@10) with no special tuning required. For the stronger HSTU backbone, MoL also improves results after careful hyperparameter tuning, reaching the best overall HR@10 of 0.2338. We additionally find that MoL embeddings show no interpretable semantic structure, suggesting MoL acts primarily as a capacity increase rather than a structured decomposition of user–item relevance.

See the [paper](first_step.pdf) for full details.

## Installation

0. Create and activate a conda environment

    ```
    conda create -n mol python=3.10 -y
    conda activate mol
    ```

1. Clone this repository

    ```
    git clone <repo_url>
    cd <repo_dir>
    ```

2. Install required packages

    ```
    pip install -r requirements.txt
    ```

## Experimental Setup

All experiments are run on the **ML-1M** dataset. We use SASRec and HSTU as sequential backbones and evaluate each with and without the MoL scoring head. All models are trained for 50 epochs and evaluated on the validation set using HR@K, NDCG@K (K ∈ {10, 50, 100, 200}), and MRR.

Configurations for all model variants are in `configs/`.

## Train and Inference

To train a model (for example, SASRec with MoL):

```
python train.py --config configs/similarity_comparison/sasrec_mol.yaml
```

To evaluate a trained checkpoint (for example, HSTU with MLP):

```
python validate.py --config configs/similarity_comparison/hstu_mlp.yaml model.params.init_checkpoint=checkpoints/hstu_mlp.pth
```

## Results on ML-1M

### Main Results

| Model         | HR@10  | NDCG@10 | HR@50  | NDCG@50 | HR@100 | NDCG@100 | HR@200 | NDCG@200 | MRR    |
|---------------|-------:|--------:|-------:|--------:|-------:|---------:|-------:|---------:|-------:|
| HSTU          | 0.1971 | 0.0989  | 0.4694 | 0.1583  | 0.5912 | 0.1793   | 0.7028 | 0.1947   | 0.1214 |
| HSTU + MoL    | 0.2338 | 0.1193  | 0.5103 | 0.1804  | 0.6310 | 0.2000   | 0.7414 | 0.2154   | 0.1358 |
| SASRec        | 0.1906 | 0.0905  | 0.4651 | 0.1507  | 0.5969 | 0.1721   | 0.7073 | 0.1876   | 0.1275 |
| **SASRec + MoL** | **0.2290** | **0.1189** | **0.5118** | **0.1809** | **0.6311** | **0.2003** | **0.7369** | **0.2151** | **0.1370** |
|---------------|-------:|--------:|-------:|--------:|-------:|---------:|-------:|---------:|-------:|

Best values per column are highlighted in **bold**.

### Similarity Function Comparison

| Model              | HR@10  | NDCG@10 | HR@50  | NDCG@50 | MRR    |
|--------------------|-------:|--------:|-------:|--------:|-------:|
| SASRec             | 0.1906 | 0.0905  | 0.4651 | 0.1507  | 0.1275 |
| SASRec + cosine    | 0.2025 | 0.1013  | 0.4853 | 0.1637  | 0.1284 |
| SASRec + bilinear  | 0.2071 | 0.1032  | 0.4912 | 0.1652  | 0.1300 |
| SASRec + MLP       | 0.2036 | 0.1028  | 0.4899 | 0.1658  | 0.1256 |
| SASRec + MoL       | 0.2290 | 0.1189  | **0.5118** | **0.1809** | **0.1370** |
| HSTU               | 0.1726 | 0.0845  | 0.4519 | 0.1466  | 0.0989 |
| HSTU + MLP         | 0.2017 | 0.1031  | 0.4775 | 0.1635  | 0.1268 |
| HSTU + cosine      | 0.2058 | 0.1038  | 0.4967 | 0.1680  | 0.1269 |
| HSTU + bilinear    | 0.2046 | 0.1034  | 0.4896 | 0.1662  | 0.1256 |
| **HSTU + MoL**     | **0.2338** | **0.1193** | 0.5103 | 0.1804  | 0.1358 |

### Embedding Clustering Analysis

| Evaluation Type   | Metric               | Mean Score    |
|-------------------|----------------------|--------------:|
| Semantic Alignment | NMI                 | 0.05 – 0.20   |
| Semantic Alignment | ARI                 | 0.01 – 0.10   |
| Geometric Quality  | Silhouette Score    | -0.10 – 0.20  |
| Geometric Quality  | Davies-Bouldin Index | 1.50 – 3.50  |

Near-zero NMI/ARI and poor Silhouette scores indicate that MoL components do not learn semantically meaningful clusters.

## Key Findings

- MoL improves SASRec immediately with default hyperparameters (+20% HR@10, +31% NDCG@10)
- MoL also improves HSTU, but required careful tuning of learning rate, weight decay, and mixture size to unlock gains
- MoL outperforms all other similarity functions (cosine, bilinear, MLP) on SASRec and HSTU both
- MoL component embeddings show no interpretable semantic structure, consistent with MoL acting as a capacity increase rather than a structured decomposition

## Future Work

- Evaluate on additional datasets (Amazon Reviews, MS MARCO)
- Probe component specialization with controlled query sets
- Study the interaction between mixture size and backbone expressiveness
