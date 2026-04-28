# Compact Retrieval-Augmented Medical Vision-Language Question Answering with Compact Models

This repository is the official repository for the study on compact retrieval-augmented medical VQA. Medical VQA models can benefit from retrieval, but in low-resource settings the trade-off between answer accuracy, grounding quality, latency, and memory footprint remains underexplored. This repository implements a compact retrieval-augmented MedVQA pipeline built around a frozen biomedical encoder, lightweight hybrid routing, answer-aware candidate retrieval, and three evaluation settings: a compact model without retrieval, a text-retrieval variant, and a multimodal-retrieval variant. Experiments were conducted on SLAKE and ImageCLEF VQA-Med 2019. The results, showed that **text retrieval** provides the most reliable accuracy-efficiency trade-off.

## Repository structure

```text
compact_medvqa_codebase/
├── README.md
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
├── configs/
│   └── default.json
├── docs/
│   └── RUN_MODES.md
├── results/
│   ├── csv/
│   └── figures/
├── scripts/
│   ├── full_run.py
│   ├── continue_run.py
│   ├── run_full.sh
│   └── run_continue.sh
└── src/compact_medvqa/
    ├── __init__.py
    └── pipeline.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

## Running the code

### Full run

```bash
python scripts/full_run.py
```

### Continue / replay run using caches and checkpoints

```bash
python scripts/continue_run.py --datasets slake imageclef_vqa_med_2019
```

### Skip heavier stages during iteration

```bash
python scripts/continue_run.py --skip-ablations --skip-robustness --skip-figures
```

<!-- ## Main results -->
<!---->
<!-- | Dataset                |  Variant | Accuracy | Closed Acc. | Open Acc. | Support Rate | Unsupported Rate | Single Model MB | Cached Latency ms | -->
<!-- | ---------------------- | -------: | -------: | ----------: | --------: | -----------: | ---------------: | --------------: | ----------------: | -->
<!-- | imageclef_vqa_med_2019 |     base |    0.556 |       0.704 |     0.112 |        0.548 |            0.326 |            9.52 |             0.708 | -->
<!-- | imageclef_vqa_med_2019 | text_rag |    0.590 |       0.755 |     0.096 |        0.620 |            0.252 |            9.78 |             2.990 | -->
<!-- | imageclef_vqa_med_2019 |   mm_rag |    0.582 |       0.741 |     0.104 |        0.850 |            0.120 |            9.52 |            38.868 | -->
<!-- | slake                  |     base |    0.803 |       0.872 |     0.708 |        0.827 |            0.086 |            9.27 |             0.719 | -->
<!-- | slake                  | text_rag |    0.843 |       0.890 |     0.778 |        0.880 |            0.030 |            9.28 |             1.342 | -->
<!-- | slake                  |   mm_rag |    0.262 |       0.213 |     0.330 |        0.366 |            0.634 |            9.27 |            12.113 | -->
<!---->

## Outputs

By default the pipeline writes to:

- `./artifacts/csv/`
- `./artifacts/figures/`
- `./artifacts/tables/`
- `./artifacts/json/`
- `./artifacts/checkpoints/`
- `./cache/`

## Citation

```bibtex
% TODO: replace with final paper metadata
@inproceedings{TODO_icig2026_LiteMedRAG,
  title={LiteMedRag: Compact Retrieval-Augmented Medical Vision-Language Question Answering with Compact Models,
  author={TODO},
  booktitle={International Conference on Image and Graphics (ICIG)},
  year={2026}
}
```
