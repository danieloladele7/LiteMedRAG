# LiteMedRAG

Reference implementation for **LiteMedRAG: Selective Retrieval-Augmented Compact Medical Visual Question Answering** (MIWAI 2026, Paper 157).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The first real run downloads the frozen BioMedCLIP backbone and the public dataset files from Hugging Face. A GPU is recommended for embedding extraction and head training.

## Run

```bash
./run.sh smoke      # CPU-only synthetic sanity check
./run.sh slake      # SLAKE
./run.sh imageclef  # ImageCLEF VQA-Med 2019
./run.sh all        # both datasets
```

Outputs are written to `artifacts/`:

- `csv/main_metrics.csv`
- `csv/predictions_<dataset>_<variant>.csv`
- `csv/policy_<dataset>_<variant>.json`
- `checkpoints/`
- `tables/main_results.tex`
- `figures/accuracy_support.pdf`

## Paper configuration

The backbone is frozen BioMedCLIP. The compact heads use hidden size 256, dropout 0.15, AdamW (`lr=1e-3`, `weight_decay=1e-4`), batch size 128, at most 60 epochs, patience 10, and seed 42. Text retrieval uses the top 5 question neighbours. Multimodal retrieval uses the top 1 neighbour with equal image/question weighting.

The final `LiteMedRAG-Acc` and `LiteMedRAG-Ground` policies are the validation-selected settings reported in the accepted manuscript and are stored in `src/litemedrag/pipeline.py`; they are frozen before test evaluation. Confidence is the uncalibrated Base maximum-softmax score and is never compared across independently trained heads. Retrieval Support Rate is exact normalized lexical agreement with retrieved support answers and is not a clinical-grounding metric.

## Citation

```bibtex
@inproceedings{Oladele_LiteMedRAG,
  title={LiteMedRag: Selective Retrieval-Augmented Compact Medical Visual Question Answering,
  author={Daniel Ayo, Oladele and Malusi Sibiya},
  booktitle={The 19th International Conference on Multi-disciplinary Trends in Artificial Intelligence (MIWAI)},
  year={2026}
}
```
