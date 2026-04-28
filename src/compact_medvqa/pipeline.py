from __future__ import annotations

import os
import gc
import sys
import math
import json
import time
import random
import shutil
import textwrap
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import StratifiedShuffleSplit

import open_clip

warnings.filterwarnings("ignore")

try:
    import google.colab  # type: ignore
    ROOT = Path("/content")
    IN_COLAB = True
except ImportError:
    ROOT = Path("./")
    IN_COLAB = False

ARTIFACTS = ROOT / "artifacts"
FIG_DIR = ARTIFACTS / "figures"
TAB_DIR = ARTIFACTS / "tables"
CSV_DIR = ARTIFACTS / "csv"
JSON_DIR = ARTIFACTS / "json"
CKPT_DIR = ARTIFACTS / "checkpoints"
CACHE_DIR = ROOT / "cache"

for p in [ROOT, ARTIFACTS, FIG_DIR, TAB_DIR, CSV_DIR, JSON_DIR, CKPT_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
print("DEVICE:", DEVICE, "| IN_COLAB:", IN_COLAB)


# Configuration
CFG = {
    "experiment_version": "v5",
    "seed": 42,
    "backbone_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "datasets": {
        "imageclef_vqa_med_2019": {
            "enabled": True,
            "hf_candidates": [
                "claudioreeves/imageclef-vqa-med-2019",
                "dineshcr7/MED-VQA-2019",
            ],
            "val_from_train": 0.10,
            "use_fast_subset": False,
            "fast_subset_train": 2500,
            "fast_subset_val": 600,
            "fast_subset_test": 500,
        },
        "slake": {
            "enabled": True,
            "hf_candidates": [
                "mdwiratathya/SLAKE-vqa-english",
                "BoKelvin/SLAKE",
            ],
            "use_fast_subset": False,
            "fast_subset_train": 3000,
            "fast_subset_val": 800,
            "fast_subset_test": 800,
        },
    },
    "retrieval": {
        "topk_text": 3,
        "topk_mm": 1,
        "audit_topk": 5,
        "mm_image_weight": 0.50,
        "mm_question_weight": 0.50,
        "use_metadata_tokens": True,
        "rerank_alpha": 0.70,
        "dropout_ratio_eval": 0.0,
        "candidate_answer_mix": 0.85,
        "candidate_support_mix": 0.15,
    },
    "answer_space": {
        "closed_min_freq": 1,
        "closed_topn_frequent": 120,
        "closed_max_tokens": 3,
        "force_closed_answers": ["yes", "no"],
        "force_closed_question_types": ["yes-no", "modality", "plane", "organ", "selection"],
        "closed_count_answers": True,
    },
    "model": {
        "fusion_type": "mlp",
        "token_dim": 512,
        "d_model": 384,
        "num_heads": 6,
        "num_layers": 2,
        "ff_mult": 4,
        "dropout": 0.15,
        "adapter_rank": 32,
        "gate_enabled": True,
        "gate_threshold": 0.50,
        "gate_threshold_grid": [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        "candidate_prior_weight": 0.65,
        "route_loss_weight": 0.25,
        "open_loss_weight": 1.75,
        "closed_loss_weight": 1.00,
    },
    "training": {
        "epochs": 100,
        "batch_size": 96,
        "lr": 1.5e-4,
        "weight_decay": 3e-4,
        "patience": 12,
        "label_smoothing": 0.03,
        "grad_clip_norm": 1.0,
        "use_amp": True,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "min_lr": 1e-5,
        "use_class_weights": True,
        "use_route_pos_weight": True,
    },
    "training_overrides": {
        "imageclef_vqa_med_2019": {
            "epochs": 120,
            "batch_size": 64,
            "lr": 1.1e-4,
            "weight_decay": 1.5e-4,
            "patience": 14,
            "label_smoothing": 0.02,
            "warmup_epochs": 6,
            "min_lr": 5.0e-6,
        },
        "slake": {
            "epochs": 100,
            "batch_size": 96,
            "lr": 1.5e-4,
            "weight_decay": 3.0e-4,
            "patience": 12,
            "label_smoothing": 0.03,
            "warmup_epochs": 5,
            "min_lr": 1.0e-5,
        },
    },
    "selection": {
        "metric_weights": {
            "default": {
                "accuracy": 1.00,
                "closed_accuracy": 0.35,
                "open_accuracy": 0.80,
                "grounding_support_rate": 0.10,
                "unsupported_answer_rate": -0.10,
            },
            "imageclef_vqa_med_2019": {
                "base": {"accuracy": 1.00, "closed_accuracy": 0.40, "open_accuracy": 0.20, "grounding_support_rate": 0.05, "unsupported_answer_rate": -0.05},
                "text_rag": {"accuracy": 1.00, "closed_accuracy": 0.45, "open_accuracy": 0.70, "grounding_support_rate": 0.10, "unsupported_answer_rate": -0.10},
                "mm_rag": {"accuracy": 1.00, "closed_accuracy": 0.45, "open_accuracy": 0.70, "grounding_support_rate": 0.10, "unsupported_answer_rate": -0.10},
            },
            "slake": {
                "base": {"accuracy": 1.00, "closed_accuracy": 0.30, "open_accuracy": 0.50, "grounding_support_rate": 0.05, "unsupported_answer_rate": -0.05},
                "text_rag": {"accuracy": 1.00, "closed_accuracy": 0.35, "open_accuracy": 0.90, "grounding_support_rate": 0.10, "unsupported_answer_rate": -0.08},
                "mm_rag": {"accuracy": 1.00, "closed_accuracy": 0.35, "open_accuracy": 0.90, "grounding_support_rate": 0.10, "unsupported_answer_rate": -0.08},
            },
        }
    },
    "ensemble": {
        "enabled": True,
        "seed_sets": {
            "base": [42],
            "text_rag": [42, 52, 62],
            "mm_rag": [42, 52, 62],
        },
    },
    "tuning": {
        "enabled": True,
        "search_epochs": 18,
        "search_patience": 4,
        "search_batch_size": 64,
        "fusion_types": ["mlp", "transformer"],
        "fusion_types_by_dataset": {
            "imageclef_vqa_med_2019": ["mlp", "transformer"],
            "slake": ["mlp", "transformer"],
        },
        "text_topk_candidates": {
            "imageclef_vqa_med_2019": [1, 3, 5],
            "slake": [3, 5],
        },
        "mm_topk_candidates": {
            "imageclef_vqa_med_2019": [1, 3],
            "slake": [1, 3],
        },
        "mm_image_weight_candidates": [0.50, 0.75],
        "rerank_alpha_candidates": [0.55, 0.70],
        "closed_min_freq_candidates": [1, 2, 3],
        "closed_topn_candidates": [80, 120, 160],
        "closed_max_tokens_candidates": [2, 3],
        "candidate_answer_mix_candidates": [0.70, 0.85, 0.95],
    },
    "latency": {
        "warmup": 20,
        "measure_samples": 128,
        "measure_end_to_end_cached": True,
        "measure_raw_encoder": False,
        "measure_raw_samples": 24,
    },
    "ablation": {
        "dataset": "imageclef_vqa_med_2019",
        "text_topk_list": [1, 3, 5],
        "mm_topk_list": [1, 3, 5],
        "mm_image_weights": [0.25, 0.50, 0.75],
        "fusion_types": ["mlp", "transformer"],
        "gate_options": [False, True],
        "closed_min_freq_list": [1, 2, 3],
        "candidate_answer_mix_list": [0.50, 0.70, 0.85, 0.95],
        "rerank_alpha_list": [0.55, 0.70, 0.85],
    },
    "robustness": {
        "corruption_levels": [0.0, 0.25, 0.50, 0.75, 1.0],
        "dropout_levels": [0.0, 0.25, 0.50],
        "min_topk_for_stability_eval": 3,
    },
    "literature_baselines": {
        "enabled": True,
        "registry_csv": "literature_baselines_v5.csv",
        "setup_json": "literature_setup_notes_v5.json",
        "expected_methods": ["PubMedCLIP", "PTUnifier", "METER", "TCL", "MMQL", "MedVInT"],
        "note": "Fill this registry only with verified literature numbers or your own reproduced external-baseline results.",
    },
    "warmup": {
        "enabled": False,
        "note": "Optional external weak-supervision stage scaffold only. Disabled by default to keep the notebook benchmark-focused and Colab-feasible.",
    },
}

VARIANTS = ["base", "text_rag", "mm_rag"]


# Utility functions

def normalize_answer(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = x.replace("\n", " ").replace("\t", " ")
    x = x.replace("’", "'").replace("`", "'")
    for ch in [".", ",", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", '"']:
        x = x.replace(ch, " ")
    x = " ".join(x.split())
    replacements = {
        "x ray": "xray",
        "x-ray": "xray",
        "ct scan": "ct",
        "mri scan": "mri",
        "magnetic resonance imaging": "mri",
        "computed tomography": "ct",
        "ultrasound image": "ultrasound",
    }
    return replacements.get(x, x)

def infer_answer_type(answer: str) -> str:
    a = normalize_answer(answer)
    if a in {"yes", "no"}:
        return "binary"
    if a.isdigit():
        return "count"
    if len(a.split()) <= 2:
        return "short-open"
    return "open"

def safe_str(x):
    return "" if x is None else str(x)

def size_mb_of_model(model) -> float:
    if isinstance(model, (list, tuple)):
        return float(sum(size_mb_of_model(m) for m in model))
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)

def stratified_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n or len(df) == 0:
        return df.copy().reset_index(drop=True)
    strata = df["answer_type"].fillna("unk").astype(str) + "||" + df["dataset"].astype(str)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(splitter.split(df, strata))
    return df.iloc[idx].reset_index(drop=True)

def question_type_from_question(q: str) -> str:
    ql = normalize_answer(q)
    if ql.startswith(("is ", "are ", "does ", "do ", "can ", "could ", "was ", "were ")):
        return "yes-no"
    if ql.startswith("what modality"):
        return "modality"
    if ql.startswith("where"):
        return "location"
    if ql.startswith("which"):
        return "selection"
    if ql.startswith("what plane"):
        return "plane"
    if ql.startswith("what organ"):
        return "organ"
    return "other"

def record_meta_tokens(row: pd.Series) -> str:
    toks = []
    for k in ["dataset", "modality", "body_part", "question_type", "answer_type"]:
        v = safe_str(row.get(k, "")).strip()
        if v:
            toks.append(f"{k}: {v}")
    return " ; ".join(toks)

def build_support_text(row: pd.Series, use_metadata_tokens: bool = True) -> str:
    parts = []
    if use_metadata_tokens:
        meta = record_meta_tokens(row)
        if meta:
            parts.append(meta)
    parts.append(f"question: {safe_str(row['question'])}")
    parts.append(f"answer: {safe_str(row['answer'])}")
    return " | ".join(parts)

def l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(np.sum(e, axis=axis, keepdims=True), 1e-8, None)

def entropy_from_probs(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return -np.sum(p * np.log(np.clip(p, eps, None)), axis=1)

def summarize_similarity(sim: Optional[np.ndarray], k_default: int = 1) -> np.ndarray:
    if sim is None:
        return np.zeros((1, 6), dtype=np.float32)
    probs = softmax_np(sim, axis=1)
    top1 = sim[:, 0]
    meanv = sim.mean(axis=1)
    stdv = sim.std(axis=1)
    margin = sim[:, 0] - sim[:, 1] if sim.shape[1] > 1 else sim[:, 0]
    ent = entropy_from_probs(probs) / np.log(max(sim.shape[1], 2))
    mass = probs[:, 0]
    return np.stack([top1, meanv, stdv, margin, ent, mass], axis=1).astype(np.float32)

def agg_features_from_idx(idx: np.ndarray, sim: Optional[np.ndarray], bank_feat: np.ndarray) -> np.ndarray:
    if sim is None:
        w = np.full(idx.shape, 1.0 / idx.shape[1], dtype=np.float32)
    else:
        w = softmax_np(sim, axis=1)
    gathered = bank_feat[idx]
    return (w[..., None] * gathered).sum(axis=1)

def support_match(pred_answer: str, support_answers: List[str]) -> bool:
    p = normalize_answer(pred_answer)
    supp = [normalize_answer(x) for x in support_answers]
    if p in supp:
        return True
    p_tokens = set(p.split())
    if 0 < len(p_tokens) <= 3:
        for s in supp:
            s_tokens = set(s.split())
            if p_tokens and (p_tokens.issubset(s_tokens) or s_tokens.issubset(p_tokens)):
                return True
    return False

def answer_hit(gold_answer: str, support_answers: List[str]) -> bool:
    g = normalize_answer(gold_answer)
    return any(normalize_answer(a) == g for a in support_answers)

def reciprocal_rank(gold_answer: str, support_answers: List[str]) -> float:
    g = normalize_answer(gold_answer)
    for rank, ans in enumerate(support_answers, start=1):
        if normalize_answer(ans) == g:
            return 1.0 / rank
    return 0.0

def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boots.append(sample.mean())
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi

def batchify_list(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]

def format_float(x, digits=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.{digits}f}"


def get_dataset_training_cfg(dataset_name: str) -> Dict[str, Any]:
    cfg = dict(CFG["training"])
    cfg.update(CFG.get("training_overrides", {}).get(dataset_name, {}))
    return cfg

def get_search_fusion_types(dataset_name: str) -> List[str]:
    by_ds = CFG["tuning"].get("fusion_types_by_dataset", {})
    return list(by_ds.get(dataset_name, CFG["tuning"]["fusion_types"]))

def selection_score_from_metrics(dataset_name: str, variant: str, metrics: Dict[str, Any]) -> float:
    metric_cfg = CFG.get("selection", {}).get("metric_weights", {})
    default = metric_cfg.get("default", {})
    weights = metric_cfg.get(dataset_name, {}).get(variant, default)
    score = 0.0
    for key, weight in weights.items():
        val = metrics.get(key, np.nan)
        if pd.notna(val):
            score += float(weight) * float(val)
    return float(score)

def literature_baseline_template_rows() -> List[Dict[str, Any]]:
    rows = []
    setup_notes = {
        "PubMedCLIP": "Domain-adapted CLIP-style biomedical encoder; in many MedVQA comparisons it is paired with a transformer fusion head such as METER.",
        "PTUnifier": "Soft-prompt unified medical VLP model spanning dual-encoder and fusion-encoder behavior; compared through reported benchmark numbers unless separately reproduced.",
        "METER": "Transformer-based multimodal fusion baseline often used as a strong discriminative VQA architecture.",
        "TCL": "Cross-modal transformer style baseline used in medical VQA comparison tables.",
        "MMQL": "Strong recent MedVQA baseline with reported benchmark results; compare through verified literature numbers unless you run the official code separately.",
        "MedVInT": "Instruction-tuned/generative MedVQA model trained with larger-scale medical VQA supervision; compare through verified reported numbers unless reproduced externally.",
    }
    for method in CFG["literature_baselines"]["expected_methods"]:
        for dataset_name in [k for k, v in CFG["datasets"].items() if v["enabled"]]:
            rows.append({
                "method": method,
                "dataset": dataset_name,
                "accuracy": np.nan,
                "closed_accuracy": np.nan,
                "open_accuracy": np.nan,
                "params_m": np.nan,
                "latency_ms": np.nan,
                "memory_mb": np.nan,
                "source_title": "",
                "source_url": "",
                "source_year": "",
                "setup_summary": setup_notes.get(method, ""),
                "verified": False,
                "notes": "",
            })
    return rows

def ensure_literature_baseline_registry() -> pd.DataFrame:
    csv_path = CSV_DIR / CFG["literature_baselines"]["registry_csv"]
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(literature_baseline_template_rows())
        df.to_csv(csv_path, index=False)
    setup_json = JSON_DIR / CFG["literature_baselines"]["setup_json"]
    if not setup_json.exists():
        setup_map = {
            row["method"]: row["setup_summary"]
            for row in literature_baseline_template_rows()
        }
        setup_json.write_text(json.dumps(setup_map, indent=2), encoding="utf-8")
    return df


# Dataset loading

def _coerce_generic_vqa_schema(df: pd.DataFrame, dataset_name: str, image_keys=None, question_keys=None, answer_keys=None, extra_keys=None) -> pd.DataFrame:
    image_keys = image_keys or ["image", "img", "Image"]
    question_keys = question_keys or ["question", "Question", "query"]
    answer_keys = answer_keys or ["answer", "Answer"]
    extra_keys = extra_keys or {}

    cols = set(df.columns)
    colmap = {}
    for target, options in {"image": image_keys, "question": question_keys, "answer": answer_keys, **extra_keys}.items():
        for c in options:
            if c in cols:
                colmap[target] = c
                break

    if "image" not in colmap or "question" not in colmap or "answer" not in colmap:
        raise KeyError(f"{dataset_name}: could not map image/question/answer columns from {sorted(cols)}")

    out = pd.DataFrame()
    out["image"] = df[colmap["image"]]
    out["question"] = df[colmap["question"]].map(safe_str)
    out["answer"] = df[colmap["answer"]].map(normalize_answer)
    for target in ["modality", "body_part", "language", "category"]:
        if target in colmap:
            out[target] = df[colmap[target]].map(safe_str)
        else:
            out[target] = ""
    return out


def _coerce_slake_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = _coerce_generic_vqa_schema(
        df,
        dataset_name="slake",
        extra_keys={
            "modality": ["modality", "Modality", "base_type"],
            "body_part": ["body_part", "location", "Location"],
            "language": ["language", "lang"],
        },
    )
    out["language"] = out["language"].map(lambda x: safe_str(x).lower() if safe_str(x) else "en")
    return out


def _infer_imageclef_question_type(question: str, category: str) -> str:
    cat = safe_str(category).strip().lower()
    q = safe_str(question).strip().lower()
    if cat:
        if "modality" in cat:
            return "modality"
        if "plane" in cat:
            return "plane"
        if "organ" in cat or "system" in cat or "body" in cat:
            return "organ"
        if "abnormal" in cat:
            return "abnormality"
    return question_type_from_question(q)


def _coerce_imageclef_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = _coerce_generic_vqa_schema(
        df,
        dataset_name="imageclef_vqa_med_2019",
        extra_keys={"category": ["category", "Category", "question_category", "q_category"]},
    )
    out["language"] = "en"
    return out


def load_slake() -> Dict[str, pd.DataFrame]:
    last_err = None
    ds = None
    for name in CFG["datasets"]["slake"]["hf_candidates"]:
        try:
            ds = load_dataset(name)
            print("Loaded SLAKE from:", name)
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"Failed to load SLAKE from candidates. Last error: {last_err}")

    splits = {}
    for split_name in ds.keys():
        df = pd.DataFrame(ds[split_name])
        df = _coerce_slake_schema(df)
        df = df[df["language"].eq("en")].reset_index(drop=True)
        df["dataset"] = "slake"
        df["answer_type"] = df["answer"].map(infer_answer_type)
        df["question_type"] = df["question"].map(question_type_from_question)
        df["split"] = split_name
        splits[split_name] = df

    train_df = splits["train"].copy().reset_index(drop=True)
    val_df = splits["validation"].copy().reset_index(drop=True) if "validation" in splits else pd.DataFrame(columns=train_df.columns)
    test_df = splits["test"].copy().reset_index(drop=True)

    if CFG["datasets"]["slake"]["use_fast_subset"]:
        train_df = stratified_subsample(train_df, CFG["datasets"]["slake"]["fast_subset_train"], CFG["seed"])
        val_df = stratified_subsample(val_df, CFG["datasets"]["slake"]["fast_subset_val"], CFG["seed"])
        test_df = stratified_subsample(test_df, CFG["datasets"]["slake"]["fast_subset_test"], CFG["seed"])

    train_df["sample_id"] = [f"slake_train_{i}" for i in range(len(train_df))]
    val_df["sample_id"] = [f"slake_val_{i}" for i in range(len(val_df))]
    test_df["sample_id"] = [f"slake_test_{i}" for i in range(len(test_df))]
    return {"train": train_df, "val": val_df, "test": test_df}


def load_imageclef_vqa_med_2019() -> Dict[str, pd.DataFrame]:
    last_err = None
    ds = None
    for name in CFG["datasets"]["imageclef_vqa_med_2019"]["hf_candidates"]:
        try:
            ds = load_dataset(name)
            print("Loaded ImageCLEF VQA-Med 2019 from:", name)
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"Failed to load ImageCLEF VQA-Med 2019 from candidates. Last error: {last_err}")

    splits = {}
    for split_name in ds.keys():
        df = pd.DataFrame(ds[split_name])
        df = _coerce_imageclef_schema(df)
        df["dataset"] = "imageclef_vqa_med_2019"
        df["answer_type"] = df["answer"].map(infer_answer_type)
        df["question_type"] = [_infer_imageclef_question_type(q, c) for q, c in zip(df["question"].tolist(), df["category"].tolist())]
        df["modality"] = np.where(df["question_type"].eq("modality"), df["answer"], "")
        df["body_part"] = np.where(df["question_type"].eq("organ"), df["answer"], "")
        df["split"] = split_name
        splits[split_name] = df.reset_index(drop=True)

    train_df = splits["train"].copy().reset_index(drop=True)
    if "validation" in splits:
        val_df = splits["validation"].copy().reset_index(drop=True)
    else:
        val_ratio = CFG["datasets"]["imageclef_vqa_med_2019"].get("val_from_train", 0.10)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=CFG["seed"])
        strata = train_df["answer_type"].astype(str)
        tr_idx, va_idx = next(splitter.split(train_df, strata))
        val_df = train_df.iloc[va_idx].reset_index(drop=True).copy()
        train_df = train_df.iloc[tr_idx].reset_index(drop=True).copy()
        val_df["split"] = "validation"
        train_df["split"] = "train"
    test_df = splits["test"].copy().reset_index(drop=True)

    if CFG["datasets"]["imageclef_vqa_med_2019"]["use_fast_subset"]:
        train_df = stratified_subsample(train_df, CFG["datasets"]["imageclef_vqa_med_2019"]["fast_subset_train"], CFG["seed"])
        val_df = stratified_subsample(val_df, CFG["datasets"]["imageclef_vqa_med_2019"]["fast_subset_val"], CFG["seed"])
        test_df = stratified_subsample(test_df, CFG["datasets"]["imageclef_vqa_med_2019"]["fast_subset_test"], CFG["seed"])

    train_df["sample_id"] = [f"imageclef_train_{i}" for i in range(len(train_df))]
    val_df["sample_id"] = [f"imageclef_val_{i}" for i in range(len(val_df))]
    test_df["sample_id"] = [f"imageclef_test_{i}" for i in range(len(test_df))]
    return {"train": train_df, "val": val_df, "test": test_df}


bundles = {}
if CFG["datasets"]["imageclef_vqa_med_2019"]["enabled"]:
    bundles["imageclef_vqa_med_2019"] = load_imageclef_vqa_med_2019()
if CFG["datasets"]["slake"]["enabled"]:
    bundles["slake"] = load_slake()

for name, split_dict in bundles.items():
    print("\nDATASET:", name)
    for split_name, df in split_dict.items():
        print(split_name, df.shape, df["answer_type"].value_counts(dropna=False).to_dict())


# Compact biomedical encoder
class FrozenBiomedCLIP:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id)
        self.tokenizer = open_clip.get_tokenizer(model_id)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        feats = []
        for i in tqdm(range(0, len(images), batch_size), desc="Encode images", leave=False):
            batch = [self.preprocess(img.convert("RGB")) for img in images[i:i+batch_size]]
            x = torch.stack(batch).to(self.device)
            emb = self.model.encode_image(x)
            emb = F.normalize(emb, dim=-1)
            feats.append(emb.detach().cpu().numpy())
        return np.concatenate(feats, axis=0)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        feats = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encode texts", leave=False):
            toks = self.tokenizer(texts[i:i+batch_size]).to(self.device)
            emb = self.model.encode_text(toks)
            emb = F.normalize(emb, dim=-1)
            feats.append(emb.detach().cpu().numpy())
        return np.concatenate(feats, axis=0)

encoder = FrozenBiomedCLIP(CFG["backbone_id"], DEVICE)


# Feature extraction and caching

def save_np(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def load_or_build_features(dataset_name: str, split_name: str, df: pd.DataFrame, encoder: FrozenBiomedCLIP):
    prefix = CACHE_DIR / dataset_name / split_name
    img_path = prefix / "image_emb.npy"
    q_path = prefix / "question_emb.npy"
    s_path = prefix / "support_text_emb.npy"
    a_path = prefix / "answer_emb.npy"
    meta_path = prefix / "rows.csv"

    work_df = df.copy().reset_index(drop=True)

    if img_path.exists() and q_path.exists() and s_path.exists() and a_path.exists() and meta_path.exists():
        image_emb = np.load(img_path)
        question_emb = np.load(q_path)
        support_text_emb = np.load(s_path)
        answer_emb = np.load(a_path)
        cached_meta = pd.read_csv(meta_path)
        if len(cached_meta) != len(work_df):
            raise ValueError(
                f"Cached row count mismatch for {dataset_name}/{split_name}: "
                f"cache={len(cached_meta)} current={len(work_df)}"
            )
        return work_df, image_emb, question_emb, support_text_emb, answer_emb

    images = work_df["image"].tolist()
    questions = work_df["question"].astype(str).tolist()
    support_texts = [
        build_support_text(row, use_metadata_tokens=CFG["retrieval"]["use_metadata_tokens"])
        for _, row in work_df.iterrows()
    ]
    answer_texts = [f"answer: {safe_str(a)}" for a in work_df["answer"].astype(str).tolist()]

    image_emb = encoder.encode_images(images)
    question_emb = encoder.encode_texts(questions)
    support_text_emb = encoder.encode_texts(support_texts)
    answer_emb = encoder.encode_texts(answer_texts)

    slim = work_df.drop(columns=["image"]).copy()
    prefix.mkdir(parents=True, exist_ok=True)
    slim.to_csv(meta_path, index=False)
    save_np(img_path, image_emb)
    save_np(q_path, question_emb)
    save_np(s_path, support_text_emb)
    save_np(a_path, answer_emb)
    return work_df, image_emb, question_emb, support_text_emb, answer_emb

feature_store = {}

for dataset_name, split_dict in bundles.items():
    feature_store[dataset_name] = {}
    for split_name, df in split_dict.items():
        work_df, img_emb, q_emb, txt_emb, ans_emb = load_or_build_features(dataset_name, split_name, df, encoder)
        feature_store[dataset_name][split_name] = {
            "df": work_df.reset_index(drop=True),
            "image_emb": img_emb.astype(np.float32),
            "question_emb": q_emb.astype(np.float32),
            "support_text_emb": txt_emb.astype(np.float32),
            "answer_emb": ans_emb.astype(np.float32),
        }
        print(dataset_name, split_name, img_emb.shape, q_emb.shape, txt_emb.shape, ans_emb.shape)


# Retrieval utilities

def build_mm_key(image_emb: np.ndarray, question_emb: np.ndarray, w_img: float, w_q: float) -> np.ndarray:
    key = np.concatenate([w_img * image_emb, w_q * question_emb], axis=1)
    return l2norm(key)

def topk_cosine(query: np.ndarray, bank: np.ndarray, k: int, exclude_diagonal: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    sim = query @ bank.T
    if exclude_diagonal and sim.shape[0] == sim.shape[1]:
        np.fill_diagonal(sim, -1e9)
    k_eff = max(1, min(k, sim.shape[1]))
    idx = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    row = np.arange(sim.shape[0])[:, None]
    val = sim[row, idx]
    order = np.argsort(-val, axis=1)
    idx = idx[row, order]
    val = val[row, order]
    return idx.astype(np.int64), val.astype(np.float32)

def simple_rerank_scores(
    q_emb: np.ndarray,
    img_emb: np.ndarray,
    idx: np.ndarray,
    bank_q: np.ndarray,
    bank_img: np.ndarray,
    base_sim: np.ndarray,
    alpha: float = 0.70,
) -> np.ndarray:
    cand_q = bank_q[idx]
    cand_img = bank_img[idx]
    q_sim = np.sum(q_emb[:, None, :] * cand_q, axis=2)
    img_sim = np.sum(img_emb[:, None, :] * cand_img, axis=2)
    rerank = alpha * base_sim + (1.0 - alpha) * (0.65 * q_sim + 0.35 * img_sim)
    return rerank.astype(np.float32)

def summarize_retrieval_quality(sim: Optional[np.ndarray]) -> np.ndarray:
    if sim is None:
        return np.zeros((1, 6), dtype=np.float32)
    return summarize_similarity(sim)

def gather_answer_lists(bank_df: pd.DataFrame, idx: np.ndarray) -> List[List[str]]:
    answers = bank_df["answer"].astype(str).tolist()
    return [[answers[j] for j in row] for row in idx]

def get_retrieval_banks(dataset_name: str, retrieval_cfg: Optional[Dict] = None):
    retrieval_cfg = CFG["retrieval"] if retrieval_cfg is None else retrieval_cfg
    w_img = retrieval_cfg["mm_image_weight"]
    w_q = retrieval_cfg["mm_question_weight"]

    train_pack = feature_store[dataset_name]["train"]
    bank_df = train_pack["df"].reset_index(drop=True).copy()
    bank_q = l2norm(train_pack["question_emb"])
    bank_img = l2norm(train_pack["image_emb"])
    bank_txt = l2norm(train_pack["support_text_emb"])
    bank_ans = l2norm(train_pack["answer_emb"])
    bank_mm = build_mm_key(bank_img, bank_q, w_img, w_q)
    return bank_df, bank_q, bank_img, bank_txt, bank_ans, bank_mm

def build_retrieval_package_from_banks(
    dataset_name: str,
    split_name: str,
    bank_df: pd.DataFrame,
    bank_q: np.ndarray,
    bank_img: np.ndarray,
    bank_txt: np.ndarray,
    bank_ans: np.ndarray,
    bank_mm: np.ndarray,
    retrieval_cfg: Dict,
):
    pack = feature_store[dataset_name][split_name]
    df = pack["df"].reset_index(drop=True)
    q = l2norm(pack["question_emb"])
    img = l2norm(pack["image_emb"])
    mm = build_mm_key(img, q, retrieval_cfg["mm_image_weight"], retrieval_cfg["mm_question_weight"])

    exclude_diagonal = (split_name == "train" and len(df) == len(bank_df))

    text_idx, text_sim = topk_cosine(q, bank_q, retrieval_cfg["topk_text"], exclude_diagonal=exclude_diagonal)
    mm_idx, mm_sim = topk_cosine(mm, bank_mm, retrieval_cfg["topk_mm"], exclude_diagonal=exclude_diagonal)
    audit_idx, audit_sim = topk_cosine(q, bank_q, retrieval_cfg["audit_topk"], exclude_diagonal=exclude_diagonal)

    text_sim = simple_rerank_scores(q, img, text_idx, bank_q, bank_img, text_sim, alpha=retrieval_cfg["rerank_alpha"])
    mm_sim = simple_rerank_scores(q, img, mm_idx, bank_q, bank_img, mm_sim, alpha=retrieval_cfg["rerank_alpha"])
    audit_sim = simple_rerank_scores(q, img, audit_idx, bank_q, bank_img, audit_sim, alpha=retrieval_cfg["rerank_alpha"])

    text_order = np.argsort(-text_sim, axis=1)
    mm_order = np.argsort(-mm_sim, axis=1)
    audit_order = np.argsort(-audit_sim, axis=1)

    row = np.arange(len(df))[:, None]
    text_idx, text_sim = text_idx[row, text_order], text_sim[row, text_order]
    mm_idx, mm_sim = mm_idx[row, mm_order], mm_sim[row, mm_order]
    audit_idx, audit_sim = audit_idx[row, audit_order], audit_sim[row, audit_order]

    text_sup = agg_features_from_idx(text_idx, text_sim, bank_txt)
    mm_text_sup = agg_features_from_idx(mm_idx, mm_sim, bank_txt)
    mm_img_sup = agg_features_from_idx(mm_idx, mm_sim, bank_img)

    return {
        "df": df,
        "image_emb": img.astype(np.float32),
        "question_emb": q.astype(np.float32),
        "text_support_emb": text_sup.astype(np.float32),
        "mm_text_support_emb": mm_text_sup.astype(np.float32),
        "mm_image_support_emb": mm_img_sup.astype(np.float32),
        "text_idx": text_idx,
        "text_sim": text_sim.astype(np.float32),
        "mm_idx": mm_idx,
        "mm_sim": mm_sim.astype(np.float32),
        "audit_idx": audit_idx,
        "audit_sim": audit_sim.astype(np.float32),
        "text_quality": summarize_retrieval_quality(text_sim),
        "mm_quality": summarize_retrieval_quality(mm_sim),
        "audit_quality": summarize_retrieval_quality(audit_sim),
        "bank_df": bank_df.reset_index(drop=True),
        "bank_q": bank_q.astype(np.float32),
        "bank_img": bank_img.astype(np.float32),
        "bank_txt": bank_txt.astype(np.float32),
        "bank_ans": bank_ans.astype(np.float32),
    }

def build_retrieval_package(dataset_name: str, retrieval_cfg: Optional[Dict] = None):
    retrieval_cfg = dict(CFG["retrieval"]) if retrieval_cfg is None else dict(retrieval_cfg)
    bank_df, bank_q, bank_img, bank_txt, bank_ans, bank_mm = get_retrieval_banks(dataset_name, retrieval_cfg)

    packages = {}
    for split_name in feature_store[dataset_name].keys():
        packages[split_name] = build_retrieval_package_from_banks(
            dataset_name=dataset_name,
            split_name=split_name,
            bank_df=bank_df,
            bank_q=bank_q,
            bank_img=bank_img,
            bank_txt=bank_txt,
            bank_ans=bank_ans,
            bank_mm=bank_mm,
            retrieval_cfg=retrieval_cfg,
        )
    return packages

retrieval_store = {name: build_retrieval_package(name) for name in bundles.keys()}

for name, split_dict in retrieval_store.items():
    for split_name, pack in split_dict.items():
        print(
            name, split_name,
            "text_sup", pack["text_support_emb"].shape,
            "mm_sup", pack["mm_image_support_emb"].shape,
            "topk_text", pack["text_idx"].shape[1],
            "topk_mm", pack["mm_idx"].shape[1],
        )


# Answer-space builders and hybrid tensor packs

CLOSED_UNK = "__closed_unk__"

def build_closed_answer_space(train_df: pd.DataFrame, cfg: Dict) -> Tuple[Dict[str, int], Dict[int, str], set]:
    answers = [normalize_answer(a) for a in train_df["answer"].tolist()]
    freq = Counter(answers)
    force_closed = {normalize_answer(a) for a in cfg["force_closed_answers"]}
    force_qtypes = set(cfg.get("force_closed_question_types", []))
    max_tokens = cfg["closed_max_tokens"]

    closed = set(force_closed)
    for _, row in train_df.iterrows():
        ans = normalize_answer(row["answer"])
        count = freq[ans]
        qtype = safe_str(row.get("question_type", ""))
        a_type = infer_answer_type(ans)
        if cfg["closed_count_answers"] and a_type == "count":
            closed.add(ans)
        if qtype in force_qtypes and len(ans.split()) <= max_tokens:
            closed.add(ans)
        if len(ans.split()) <= max_tokens and count >= cfg["closed_min_freq"]:
            closed.add(ans)

    top_freq = [a for a, _ in freq.most_common(cfg["closed_topn_frequent"]) if len(a.split()) <= max_tokens]
    closed.update(top_freq)

    closed = sorted([a for a in closed if a != ""])
    if CLOSED_UNK not in closed:
        closed.append(CLOSED_UNK)

    ans2id = {a: i for i, a in enumerate(closed)}
    id2ans = {i: a for a, i in ans2id.items()}
    return ans2id, id2ans, set(closed) - {CLOSED_UNK}

def encode_closed_answers(df: pd.DataFrame, ans2id: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    unk_id = ans2id[CLOSED_UNK]
    y = []
    is_closed = []
    for a in df["answer"].tolist():
        key = normalize_answer(a)
        if key in ans2id and key != CLOSED_UNK:
            y.append(ans2id[key])
            is_closed.append(True)
        else:
            y.append(unk_id)
            is_closed.append(False)
    return np.asarray(y, dtype=np.int64), np.asarray(is_closed, dtype=bool)

def make_variant_tokens(pack: Dict, variant: str) -> np.ndarray:
    if variant == "base":
        toks = [pack["image_emb"], pack["question_emb"]]
    elif variant == "text_rag":
        toks = [pack["image_emb"], pack["question_emb"], pack["text_support_emb"]]
    elif variant == "mm_rag":
        toks = [pack["image_emb"], pack["question_emb"], pack["mm_text_support_emb"], pack["mm_image_support_emb"]]
    else:
        raise ValueError(variant)
    toks = [l2norm(t).astype(np.float32) for t in toks]
    return np.stack(toks, axis=1).astype(np.float32)

def locate_open_target_mask(gold_answers: List[str], candidate_answers: List[List[str]]) -> np.ndarray:
    if len(candidate_answers) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    k = max(len(x) for x in candidate_answers)
    mask = np.zeros((len(candidate_answers), k), dtype=np.float32)
    for i, (gold, cands) in enumerate(zip(gold_answers, candidate_answers)):
        g = normalize_answer(gold)
        for j, cand in enumerate(cands):
            if normalize_answer(cand) == g:
                mask[i, j] = 1.0
    return mask

def candidate_embeddings_from_idx(pack: Dict, idx: np.ndarray, retrieval_cfg: Dict) -> np.ndarray:
    bank_txt = pack["bank_txt"][idx]
    bank_ans = pack["bank_ans"][idx]
    mix_ans = retrieval_cfg["candidate_answer_mix"]
    mix_sup = retrieval_cfg["candidate_support_mix"]
    cand = mix_ans * bank_ans + mix_sup * bank_txt
    flat = cand.reshape(-1, cand.shape[-1])
    flat = l2norm(flat).reshape(cand.shape)
    return flat.astype(np.float32)

def get_variant_candidates(pack: Dict, variant: str, retrieval_cfg: Dict) -> Tuple[np.ndarray, np.ndarray, List[List[str]], np.ndarray]:
    n = len(pack["df"])
    d = pack["bank_txt"].shape[1]
    if variant == "base":
        idx = np.full((n, 1), -1, dtype=np.int64)
        emb = np.zeros((n, 1, d), dtype=np.float32)
        answers = [[""] for _ in range(n)]
        prior = np.zeros((n, 1), dtype=np.float32)
        return idx, emb, answers, prior

    if variant == "text_rag":
        idx = pack["text_idx"]
        prior = pack["text_sim"]
    elif variant == "mm_rag":
        idx = pack["mm_idx"]
        prior = pack["mm_sim"]
    else:
        raise ValueError(variant)

    emb = candidate_embeddings_from_idx(pack, idx, retrieval_cfg)
    answers = gather_answer_lists(pack["bank_df"], idx)
    prior = prior.astype(np.float32)
    return idx.astype(np.int64), emb, answers, prior

def get_variant_quality(pack: Dict, variant: str) -> np.ndarray:
    n = len(pack["df"])
    if variant == "base":
        return np.zeros((n, 6), dtype=np.float32)
    if variant == "text_rag":
        return pack["text_quality"].astype(np.float32)
    if variant == "mm_rag":
        return pack["mm_quality"].astype(np.float32)
    raise ValueError(variant)

def get_variant_support_for_analysis(pack: Dict, variant: str) -> Tuple[np.ndarray, List[List[str]]]:
    if variant == "base":
        idx = pack["audit_idx"]
    elif variant == "text_rag":
        idx = pack["text_idx"]
    elif variant == "mm_rag":
        idx = pack["mm_idx"]
    else:
        raise ValueError(variant)
    return idx, gather_answer_lists(pack["bank_df"], idx)

def build_hybrid_variant_pack(
    pack: Dict,
    variant: str,
    closed_ans2id: Dict[str, int],
    retrieval_cfg: Dict,
) -> Dict[str, Any]:
    tokens = make_variant_tokens(pack, variant)
    cand_idx, cand_emb, cand_answers, cand_prior = get_variant_candidates(pack, variant, retrieval_cfg)
    quality = get_variant_quality(pack, variant)
    closed_y, closed_mask = encode_closed_answers(pack["df"], closed_ans2id)
    open_target_mask = locate_open_target_mask(pack["df"]["answer"].tolist(), cand_answers)
    copy_available = open_target_mask.sum(axis=1) > 0 if len(open_target_mask) else np.zeros(len(pack["df"]), dtype=bool)
    support_idx_for_eval, support_answers_for_eval = get_variant_support_for_analysis(pack, variant)

    route_y = ((~closed_mask) & copy_available).astype(np.float32)

    return {
        "tokens": tokens.astype(np.float32),
        "candidate_emb": cand_emb.astype(np.float32),
        "candidate_idx": cand_idx.astype(np.int64),
        "candidate_answers": cand_answers,
        "candidate_prior": cand_prior.astype(np.float32),
        "quality": quality.astype(np.float32),
        "closed_y": closed_y.astype(np.int64),
        "route_y": route_y.astype(np.float32),
        "open_target_mask": open_target_mask.astype(np.float32),
        "gold_answer": pack["df"]["answer"].astype(str).tolist(),
        "df": pack["df"].copy().reset_index(drop=True),
        "support_idx_for_eval": support_idx_for_eval.astype(np.int64),
        "support_answers_for_eval": support_answers_for_eval,
        "bank_df": pack["bank_df"].copy().reset_index(drop=True),
        "oov_mask": (~closed_mask).astype(bool),
        "copy_available_mask": copy_available.astype(bool),
    }

def build_dataset_hybrid_store(
    retrieval_split_dict: Dict[str, Dict[str, Any]],
    answer_space_cfg: Dict,
    retrieval_cfg: Dict,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], pd.DataFrame]:
    closed_ans2id, closed_id2ans, closed_set = build_closed_answer_space(
        retrieval_split_dict["train"]["df"],
        answer_space_cfg,
    )
    answer_space = {
        "closed_ans2id": closed_ans2id,
        "closed_id2ans": closed_id2ans,
        "closed_set": sorted(closed_set),
        "closed_vocab_size": len(closed_ans2id) - 1,
        "unk_id": closed_ans2id[CLOSED_UNK],
    }

    store = {}
    coverage_rows = []
    for variant in VARIANTS:
        store[variant] = {}
        for split_name, pack in retrieval_split_dict.items():
            vp = build_hybrid_variant_pack(pack, variant, closed_ans2id, retrieval_cfg)
            store[variant][split_name] = vp

            coverage_rows.append({
                "variant": variant,
                "split": split_name,
                "num_samples": len(pack["df"]),
                "closed_vocab_size": len(closed_ans2id) - 1,
                "closed_gold_rate": float((~vp["oov_mask"]).mean()) if len(vp["oov_mask"]) else np.nan,
                "open_gold_rate": float(vp["oov_mask"].mean()) if len(vp["oov_mask"]) else np.nan,
                "copy_target_available_rate": float(vp["copy_available_mask"].mean()) if len(vp["copy_available_mask"]) else np.nan,
            })
    return store, answer_space, pd.DataFrame(coverage_rows)

def rebuild_all_hybrid_stores(retrieval_store_dict: Dict[str, Dict[str, Any]], answer_space_cfg: Dict, retrieval_cfg: Dict):
    all_answer_spaces = {}
    all_hybrid_store = {}
    coverage_tables = []
    for dataset_name, split_dict in retrieval_store_dict.items():
        ds_store, ds_answer_space, ds_cov = build_dataset_hybrid_store(split_dict, answer_space_cfg, retrieval_cfg)
        ds_cov.insert(0, "dataset", dataset_name)
        all_answer_spaces[dataset_name] = ds_answer_space
        all_hybrid_store[dataset_name] = ds_store
        coverage_tables.append(ds_cov)
    cov = pd.concat(coverage_tables, ignore_index=True) if coverage_tables else pd.DataFrame()
    return all_hybrid_store, all_answer_spaces, cov

hybrid_store, answer_spaces, coverage_df = rebuild_all_hybrid_stores(
    retrieval_store_dict=retrieval_store,
    answer_space_cfg=CFG["answer_space"],
    retrieval_cfg=CFG["retrieval"],
)

coverage_df.to_csv(CSV_DIR / "answer_space_coverage_v3.csv", index=False)
with open(JSON_DIR / "answer_spaces_v3.json", "w") as f:
    json.dump(
        {
            ds: {
                "closed_vocab_size": v["closed_vocab_size"],
                "unk_id": v["unk_id"],
                "closed_answer_preview": v["closed_set"][:50],
            } for ds, v in answer_spaces.items()
        },
        f,
        indent=2,
    )
coverage_df


# Hybrid compact model: tuned routed model with duplicate-aware copy branch and per-candidate priors

class NumpyHybridDataset(Dataset):
    def __init__(self, tokens, candidate_emb, candidate_prior, quality, closed_y, route_y, open_target_mask):
        self.tokens = torch.from_numpy(tokens).float()
        self.candidate_emb = torch.from_numpy(candidate_emb).float()
        self.candidate_prior = torch.from_numpy(candidate_prior).float()
        self.quality = torch.from_numpy(quality).float()
        self.closed_y = torch.from_numpy(closed_y).long()
        self.route_y = torch.from_numpy(route_y).float()
        self.open_target_mask = torch.from_numpy(open_target_mask).float()

    def __len__(self):
        return len(self.closed_y)

    def __getitem__(self, idx):
        return (
            self.tokens[idx],
            self.candidate_emb[idx],
            self.candidate_prior[idx],
            self.quality[idx],
            self.closed_y[idx],
            self.route_y[idx],
            self.open_target_mask[idx],
        )

class LowRankAdapter(nn.Module):
    def __init__(self, dim: int, rank: int = 32, dropout: float = 0.1):
        super().__init__()
        rank = max(1, min(rank, dim))
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.dropout(self.up(self.down(x)))

class HybridCompactMedVQAModel(nn.Module):
    def __init__(self, token_dim: int, num_closed: int, max_candidates: int, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        d_model = cfg["d_model"]
        self.fusion_type = cfg["fusion_type"]
        self.max_candidates = max_candidates
        self.gate_enabled = cfg["gate_enabled"]

        self.token_proj = nn.Linear(token_dim, d_model)
        self.cand_proj = nn.Linear(token_dim, d_model)
        self.quality_proj = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, d_model // 2),
            nn.GELU(),
        )
        self.adapter = LowRankAdapter(d_model, rank=cfg["adapter_rank"], dropout=cfg["dropout"])

        if self.fusion_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=cfg["num_heads"],
                dim_feedforward=d_model * cfg["ff_mult"],
                dropout=cfg["dropout"],
                batch_first=True,
                activation="gelu",
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg["num_layers"])
            self.fusion_norm = nn.LayerNorm(d_model)
        elif self.fusion_type == "mlp":
            self.flat_fuser = nn.Sequential(
                nn.LayerNorm(d_model * 4),
                nn.Linear(d_model * 4, d_model * 2),
                nn.GELU(),
                nn.Dropout(cfg["dropout"]),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
            )
            self.fusion_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(self.fusion_type)

        self.closed_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(d_model, num_closed),
        )
        self.route_head = nn.Sequential(
            nn.LayerNorm(d_model + d_model // 2),
            nn.Linear(d_model + d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(d_model // 2, 1),
        )
        self.open_query = nn.Sequential(
            nn.LayerNorm(d_model + d_model // 2),
            nn.Linear(d_model + d_model // 2, d_model),
            nn.GELU(),
        )
        self.open_bias = nn.Parameter(torch.zeros(1))
        self.best_gate_threshold = float(cfg.get("gate_threshold", 0.50))
        self.best_val_accuracy = None
        self.best_epoch = None

    def fuse_tokens(self, tokens):
        x = self.adapter(self.token_proj(tokens))
        if self.fusion_type == "transformer":
            b = x.size(0)
            cls = self.cls_token.expand(b, -1, -1)
            z = torch.cat([cls, x], dim=1)
            z = self.encoder(z)
            fused = self.fusion_norm(z[:, 0])
            return fused
        else:
            b, t, d = x.shape
            if t < 4:
                pad = torch.zeros(b, 4 - t, d, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
            elif t > 4:
                x = x[:, :4]
            fused = self.flat_fuser(x.reshape(b, -1))
            return self.fusion_norm(fused)

    def forward(self, tokens, candidate_emb, candidate_prior, quality):
        fused = self.fuse_tokens(tokens)
        qfeat = self.quality_proj(quality)
        joint = torch.cat([fused, qfeat], dim=1)

        closed_logits = self.closed_head(fused)
        gate_logits = self.route_head(joint).squeeze(1)

        cand = self.adapter(self.cand_proj(candidate_emb))
        open_query = self.open_query(joint).unsqueeze(1)
        open_scores = (cand * open_query).sum(dim=-1) / math.sqrt(cand.size(-1))
        open_scores = open_scores + self.cfg["candidate_prior_weight"] * candidate_prior
        open_scores = open_scores + self.open_bias

        return {
            "closed_logits": closed_logits,
            "gate_logits": gate_logits,
            "open_scores": open_scores,
            "fused": fused,
        }

def make_hybrid_loaders(train_pack: Dict, val_pack: Dict, batch_size: int):
    train_ds = NumpyHybridDataset(
        train_pack["tokens"],
        train_pack["candidate_emb"],
        train_pack["candidate_prior"],
        train_pack["quality"],
        train_pack["closed_y"],
        train_pack["route_y"],
        train_pack["open_target_mask"],
    )
    val_ds = NumpyHybridDataset(
        val_pack["tokens"],
        val_pack["candidate_emb"],
        val_pack["candidate_prior"],
        val_pack["quality"],
        val_pack["closed_y"],
        val_pack["route_y"],
        val_pack["open_target_mask"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader

def compute_closed_class_weights(train_pack: Dict, num_closed: int) -> Optional[torch.Tensor]:
    y = train_pack["closed_y"]
    unk_id = answer_spaces_local_unk_id if "answer_spaces_local_unk_id" in globals() else None
    counts = np.bincount(y, minlength=num_closed).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / np.mean(weights)
    if unk_id is not None and 0 <= unk_id < len(weights):
        weights[unk_id] = min(weights[unk_id], 0.5)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def compute_route_pos_weight(train_pack: Dict) -> Optional[torch.Tensor]:
    y = train_pack["route_y"].astype(np.float32)
    pos = float(y.sum())
    neg = float(len(y) - pos)
    if pos <= 0 or neg <= 0:
        return None
    return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)

def multi_positive_open_loss(open_scores: torch.Tensor, open_target_mask: torch.Tensor) -> torch.Tensor:
    valid = open_target_mask.sum(dim=1) > 0
    if not valid.any():
        return torch.zeros((), device=open_scores.device)
    log_probs = F.log_softmax(open_scores[valid], dim=1)
    target = open_target_mask[valid]
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    loss = -(target * log_probs).sum(dim=1).mean()
    return loss

def hybrid_losses(
    batch_out: Dict[str, torch.Tensor],
    closed_y: torch.Tensor,
    route_y: torch.Tensor,
    open_target_mask: torch.Tensor,
    cfg_train: Dict,
    cfg_model: Dict,
    closed_class_weights: Optional[torch.Tensor] = None,
    route_pos_weight: Optional[torch.Tensor] = None,
):
    closed_loss = F.cross_entropy(
        batch_out["closed_logits"],
        closed_y,
        weight=closed_class_weights,
        label_smoothing=cfg_train["label_smoothing"],
    )

    route_loss = F.binary_cross_entropy_with_logits(
        batch_out["gate_logits"],
        route_y,
        pos_weight=route_pos_weight,
    )

    open_loss = multi_positive_open_loss(batch_out["open_scores"], open_target_mask)

    total = (
        cfg_model["closed_loss_weight"] * closed_loss
        + cfg_model["route_loss_weight"] * route_loss
        + cfg_model["open_loss_weight"] * open_loss
    )
    return total, {
        "closed_loss": float(closed_loss.detach().cpu()),
        "route_loss": float(route_loss.detach().cpu()),
        "open_loss": float(open_loss.detach().cpu()),
    }

def aggregate_candidate_answer_scores(scores: np.ndarray, cands: List[str]) -> Tuple[str, float]:
    if len(cands) == 0:
        return "", float("-inf")
    bucket = defaultdict(list)
    for s, cand in zip(scores, cands):
        key = normalize_answer(cand)
        if key != "":
            bucket[key].append(float(s))
    if not bucket:
        return "", float("-inf")
    agg = {}
    for k, vals in bucket.items():
        if len(vals) == 0:
            agg[k] = float("-inf")
            continue
        arr = np.asarray(vals, dtype=np.float32)
        m = float(arr.max())
        agg[k] = m + float(np.log(np.exp(arr - m).sum()))
    best_ans = max(agg.items(), key=lambda kv: kv[1])[0]
    return best_ans, agg[best_ans]

@torch.no_grad()
def decode_hybrid_batch(
    out: Dict[str, torch.Tensor],
    candidate_answers: List[List[str]],
    closed_id2ans: Dict[int, str],
    gate_enabled: bool = True,
    gate_threshold: float = 0.50,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    closed_pred_ids = out["closed_logits"].argmax(dim=1).detach().cpu().numpy()
    closed_pred_answers = [closed_id2ans[int(i)] for i in closed_pred_ids]
    gate_prob = torch.sigmoid(out["gate_logits"]).detach().cpu().numpy()
    open_scores = out["open_scores"].detach().cpu().numpy()

    final_answers = []
    used_open = []
    open_best_scores = []
    for i, (closed_ans, cands) in enumerate(zip(closed_pred_answers, candidate_answers)):
        closed_ans = "" if closed_ans == CLOSED_UNK else closed_ans
        open_ans, open_best = aggregate_candidate_answer_scores(open_scores[i], cands)
        choose_open = gate_enabled and (gate_prob[i] >= gate_threshold) and (open_ans != "")
        final_answers.append(open_ans if choose_open else (closed_ans if closed_ans != "" else open_ans))
        used_open.append(bool(choose_open))
        open_best_scores.append(open_best)
    return final_answers, gate_prob, np.asarray(used_open, dtype=bool), closed_pred_ids, np.asarray(open_best_scores, dtype=np.float32)

@torch.no_grad()
def predict_pack_and_select_threshold(
    model: nn.Module,
    pack: Dict,
    closed_id2ans: Dict[int, str],
    gate_enabled: bool = True,
    threshold_grid: Optional[List[float]] = None,
) -> Dict[str, Any]:
    threshold_grid = [0.50] if threshold_grid is None else threshold_grid
    outputs = predict_hybrid_outputs(model, pack)
    gold = [normalize_answer(x) for x in pack["gold_answer"]]
    best = None
    for thr in threshold_grid:
        final_answers, gate_prob, used_open, closed_pred_ids, open_best_scores = decode_hybrid_batch(
            outputs,
            pack["candidate_answers"],
            closed_id2ans,
            gate_enabled=gate_enabled,
            gate_threshold=float(thr),
        )
        correct = np.asarray([normalize_answer(p) == g for p, g in zip(final_answers, gold)], dtype=np.float32)
        closed_mask = ~pack["oov_mask"]
        open_mask = pack["oov_mask"]
        score = float(correct.mean())
        tiebreak = (
            float(correct[open_mask].mean()) if open_mask.any() else -1.0,
            float(correct[closed_mask].mean()) if closed_mask.any() else -1.0,
            -abs(float(np.mean(gate_prob)) - 0.5),
        )
        cur = {
            "threshold": float(thr),
            "accuracy": score,
            "open_accuracy": float(correct[open_mask].mean()) if open_mask.any() else np.nan,
            "closed_accuracy": float(correct[closed_mask].mean()) if closed_mask.any() else np.nan,
            "outputs": outputs,
            "final_answers": final_answers,
            "gate_prob": gate_prob,
            "used_open": used_open,
            "closed_pred_ids": closed_pred_ids,
            "open_best_scores": open_best_scores,
            "correct": correct,
        }
        if best is None or (cur["accuracy"], tiebreak) > (best["accuracy"], (
            best["open_accuracy"] if not np.isnan(best["open_accuracy"]) else -1.0,
            best["closed_accuracy"] if not np.isnan(best["closed_accuracy"]) else -1.0,
            -abs(float(np.mean(best["gate_prob"])) - 0.5),
        )):
            best = cur
    return best

def get_scheduler(optimizer, cfg_train: Dict):
    if cfg_train.get("scheduler", "none") != "cosine":
        return None

    epochs = max(1, int(cfg_train["epochs"]))
    warmup_epochs = max(0, int(cfg_train.get("warmup_epochs", 0)))
    min_lr = float(cfg_train.get("min_lr", 1e-5))
    base_lr = float(cfg_train["lr"])

    def lr_lambda(epoch_idx: int):
        if epoch_idx < warmup_epochs and warmup_epochs > 0:
            return float(epoch_idx + 1) / float(warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = min_lr / base_lr
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_hybrid_model(
    train_pack: Dict,
    val_pack: Dict,
    num_closed: int,
    cfg_model: Dict,
    cfg_train: Dict,
    closed_id2ans: Optional[Dict[int, str]] = None,
    unk_id: Optional[int] = None,
):
    model = HybridCompactMedVQAModel(
        token_dim=train_pack["tokens"].shape[-1],
        num_closed=num_closed,
        max_candidates=train_pack["candidate_emb"].shape[1],
        cfg=cfg_model,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train["lr"], weight_decay=cfg_train["weight_decay"])
    scheduler = get_scheduler(optimizer, cfg_train)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg_train['use_amp'] and torch.cuda.is_available()))
    train_loader, val_loader = make_hybrid_loaders(train_pack, val_pack, cfg_train["batch_size"])

    closed_class_weights = None
    if cfg_train.get("use_class_weights", False):
        counts = np.bincount(train_pack["closed_y"], minlength=num_closed).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = counts.sum() / counts
        weights = weights / np.mean(weights)
        if unk_id is not None and 0 <= unk_id < len(weights):
            weights[unk_id] = min(weights[unk_id], 0.5)
        closed_class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    route_pos_weight = None
    if cfg_train.get("use_route_pos_weight", False):
        pos = float(train_pack["route_y"].sum())
        neg = float(len(train_pack["route_y"]) - pos)
        if pos > 0 and neg > 0:
            route_pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)

    best_state = None
    best_score = -1.0
    patience_left = cfg_train["patience"]
    history = []
    best_threshold = float(cfg_model.get("gate_threshold", 0.50))

    for epoch in range(cfg_train["epochs"]):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        tr_closed_proxy = 0

        for tokens, candidate_emb, candidate_prior, quality, closed_y, route_y, open_target_mask in train_loader:
            tokens = tokens.to(DEVICE, non_blocking=True)
            candidate_emb = candidate_emb.to(DEVICE, non_blocking=True)
            candidate_prior = candidate_prior.to(DEVICE, non_blocking=True)
            quality = quality.to(DEVICE, non_blocking=True)
            closed_y = closed_y.to(DEVICE, non_blocking=True)
            route_y = route_y.to(DEVICE, non_blocking=True)
            open_target_mask = open_target_mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(cfg_train['use_amp'] and torch.cuda.is_available())):
                out = model(tokens, candidate_emb, candidate_prior, quality)
                loss, parts = hybrid_losses(
                    out,
                    closed_y,
                    route_y,
                    open_target_mask,
                    cfg_train=cfg_train,
                    cfg_model=cfg_model,
                    closed_class_weights=closed_class_weights,
                    route_pos_weight=route_pos_weight,
                )
            scaler.scale(loss).backward()
            if cfg_train["grad_clip_norm"] is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg_train["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * closed_y.size(0)
            train_n += closed_y.size(0)
            tr_closed_proxy += (out["closed_logits"].argmax(dim=1) == closed_y).sum().item()

        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_sum / max(train_n, 1)
        train_closed_proxy = tr_closed_proxy / max(train_n, 1)

        val_eval = predict_pack_and_select_threshold(
            model,
            val_pack,
            closed_id2ans=closed_id2ans,
            gate_enabled=cfg_model["gate_enabled"],
            threshold_grid=cfg_model.get("gate_threshold_grid", [cfg_model.get("gate_threshold", 0.50)]),
        )
        score = val_eval["accuracy"]
        best_epoch_threshold = val_eval["threshold"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_closed_proxy": train_closed_proxy,
            "val_accuracy": val_eval["accuracy"],
            "val_closed_accuracy": val_eval["closed_accuracy"],
            "val_open_accuracy": val_eval["open_accuracy"],
            "val_gate_threshold": best_epoch_threshold,
            "val_open_branch_usage": float(np.mean(val_eval["used_open"])) if len(val_eval["used_open"]) else np.nan,
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(
            f"epoch={epoch+1:02d} "
            f"train_closed_proxy={train_closed_proxy:.4f} "
            f"val_acc={val_eval['accuracy']:.4f} "
            f"val_open={val_eval['open_accuracy'] if not np.isnan(val_eval['open_accuracy']) else float('nan'):.4f} "
            f"thr={best_epoch_threshold:.2f}"
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_threshold = float(best_epoch_threshold)
            model.best_val_accuracy = float(score)
            model.best_epoch = int(epoch + 1)
            patience_left = cfg_train["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_gate_threshold = float(best_threshold)
    return model, pd.DataFrame(history)


# Evaluation utilities for the hybrid routed model

def classify_error(row) -> str:
    if row["correct"]:
        return "supported-correct" if row["supported"] else "unsupported-correct"
    if row.get("gold_closed", False) and not row.get("used_open_branch", False):
        if row["retrieval_hit"] and not row["supported"]:
            return "fusion-error"
        if not row["retrieval_hit"]:
            return "closed-misclassification"
    if not row.get("gold_closed", False):
        if row.get("copy_target_available", False) and not row.get("used_open_branch", False):
            return "routing-miss"
        if row.get("used_open_branch", False) and not row["supported"]:
            return "unsupported-hallucination"
        if not row.get("copy_target_available", False):
            return "gold-oov-copy-miss"
    if not row["supported"] and not row["retrieval_hit"]:
        return "retrieval-miss"
    return "unsupported-hallucination"

@torch.no_grad()
def _predict_hybrid_outputs_single(model: nn.Module, pack: Dict, batch_size: int = 256) -> Dict[str, Any]:
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    for mdl in models:
        mdl.eval()
    token_t = torch.from_numpy(pack["tokens"]).float()
    cand_t = torch.from_numpy(pack["candidate_emb"]).float()
    prior_t = torch.from_numpy(pack["candidate_prior"]).float()
    qual_t = torch.from_numpy(pack["quality"]).float()

    loader = DataLoader(TensorDataset(token_t, cand_t, prior_t, qual_t), batch_size=batch_size, shuffle=False)
    out_closed, out_gate, out_open = [], [], []
    for tokens, candidate_emb, candidate_prior, quality in loader:
        tokens = tokens.to(DEVICE)
        candidate_emb = candidate_emb.to(DEVICE)
        candidate_prior = candidate_prior.to(DEVICE)
        quality = quality.to(DEVICE)
        out = model(tokens, candidate_emb, candidate_prior, quality)
        out_closed.append(out["closed_logits"].detach().cpu())
        out_gate.append(out["gate_logits"].detach().cpu())
        out_open.append(out["open_scores"].detach().cpu())

    return {
        "closed_logits": torch.cat(out_closed, dim=0),
        "gate_logits": torch.cat(out_gate, dim=0),
        "open_scores": torch.cat(out_open, dim=0),
    }

def average_output_dicts(outputs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(outputs_list) == 1:
        return outputs_list[0]
    keys = ["closed_logits", "gate_logits", "open_scores"]
    out = {}
    for k in keys:
        stacked = torch.stack([o[k] for o in outputs_list], dim=0)
        out[k] = stacked.mean(dim=0)
    return out

def predict_hybrid_outputs(model, pack: Dict, batch_size: int = 256) -> Dict[str, Any]:
    if isinstance(model, (list, tuple)):
        outs = [_predict_hybrid_outputs_single(m, pack, batch_size=batch_size) for m in model]
        return average_output_dicts(outs)
    return _predict_hybrid_outputs_single(model, pack, batch_size=batch_size)

def select_threshold_from_outputs(
    outputs: Dict[str, Any],
    pack: Dict,
    closed_id2ans: Dict[int, str],
    gate_enabled: bool = True,
    threshold_grid: Optional[List[float]] = None,
) -> Dict[str, Any]:
    threshold_grid = [0.50] if threshold_grid is None else threshold_grid
    gold = [normalize_answer(x) for x in pack["gold_answer"]]
    best = None
    for thr in threshold_grid:
        final_answers, gate_prob, used_open, closed_pred_ids, open_best_scores = decode_hybrid_batch(
            outputs,
            pack["candidate_answers"],
            closed_id2ans,
            gate_enabled=gate_enabled,
            gate_threshold=float(thr),
        )
        correct = np.asarray([normalize_answer(p) == g for p, g in zip(final_answers, gold)], dtype=np.float32)
        closed_mask = ~pack["oov_mask"]
        open_mask = pack["oov_mask"]
        score = float(correct.mean())
        tiebreak = (
            float(correct[open_mask].mean()) if open_mask.any() else -1.0,
            float(correct[closed_mask].mean()) if closed_mask.any() else -1.0,
            -abs(float(np.mean(gate_prob)) - 0.5),
        )
        cur = {
            "threshold": float(thr),
            "accuracy": score,
            "open_accuracy": float(correct[open_mask].mean()) if open_mask.any() else np.nan,
            "closed_accuracy": float(correct[closed_mask].mean()) if closed_mask.any() else np.nan,
            "outputs": outputs,
            "final_answers": final_answers,
            "gate_prob": gate_prob,
            "used_open": used_open,
            "closed_pred_ids": closed_pred_ids,
            "open_best_scores": open_best_scores,
            "correct": correct,
        }
        if best is None or (cur["accuracy"], tiebreak) > (best["accuracy"], (
            best["open_accuracy"] if not np.isnan(best["open_accuracy"]) else -1.0,
            best["closed_accuracy"] if not np.isnan(best["closed_accuracy"]) else -1.0,
            -abs(float(np.mean(best["gate_prob"])) - 0.5),
        )):
            best = cur
    return best

def measure_head_latency(model: nn.Module, pack: Dict, warmup: int = 20, measure_samples: int = 128) -> float:
    if isinstance(model, (list, tuple)):
        if len(model) == 0:
            return np.nan
        model = model[0]
    model.eval()
    m = min(measure_samples, len(pack["tokens"]))
    tokens = torch.from_numpy(pack["tokens"][:m]).float().to(DEVICE)
    cand = torch.from_numpy(pack["candidate_emb"][:m]).float().to(DEVICE)
    prior = torch.from_numpy(pack["candidate_prior"][:m]).float().to(DEVICE)
    qual = torch.from_numpy(pack["quality"][:m]).float().to(DEVICE)
    times = []
    with torch.no_grad():
        for i in range(min(warmup, m)):
            _ = model(tokens[i:i+1], cand[i:i+1], prior[i:i+1], qual[i:i+1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for i in range(m):
            t0 = time.perf_counter()
            _ = model(tokens[i:i+1], cand[i:i+1], prior[i:i+1], qual[i:i+1])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return float(np.median(times)) if times else np.nan

def measure_cached_retrieval_plus_model_latency(model: nn.Module, dataset_name: str, variant: str, split_name: str = "test", measure_samples: int = 64) -> float:
    if isinstance(model, (list, tuple)):
        if len(model) == 0:
            return np.nan
        model = model[0]
    if "selected_hybrid_store" in globals() and dataset_name in selected_hybrid_store and variant in selected_hybrid_store[dataset_name]:
        if variant == "base":
            return measure_head_latency(model, selected_hybrid_store[dataset_name][variant][split_name], CFG["latency"]["warmup"], measure_samples)

    retr_cfg = CFG["retrieval"]
    retr_source = retrieval_store
    if "selected_configs" in globals() and dataset_name in selected_configs and variant in selected_configs[dataset_name]:
        retr_cfg = selected_configs[dataset_name][variant]["retrieval_cfg"]
    if "selected_retrieval_store" in globals() and dataset_name in selected_retrieval_store and variant in selected_retrieval_store[dataset_name]:
        retr_source = selected_retrieval_store[dataset_name][variant]

    if variant == "base":
        pack_src = selected_hybrid_store[dataset_name][variant][split_name] if "selected_hybrid_store" in globals() and dataset_name in selected_hybrid_store and variant in selected_hybrid_store[dataset_name] else hybrid_store[dataset_name][variant][split_name]
        return measure_head_latency(model, pack_src, CFG["latency"]["warmup"], measure_samples)

    retr_pack = retr_source[split_name]
    q = retr_pack["question_emb"]
    img = retr_pack["image_emb"]
    bank_q = retr_pack["bank_q"]
    bank_img = retr_pack["bank_img"]
    bank_txt = retr_pack["bank_txt"]
    model.eval()

    times = []
    n = min(measure_samples, len(q))
    with torch.no_grad():
        for i in range(n):
            q_i = q[i:i+1]
            img_i = img[i:i+1]
            t0 = time.perf_counter()

            if variant == "text_rag":
                idx, sim = topk_cosine(q_i, bank_q, retr_cfg["topk_text"], exclude_diagonal=False)
                sim = simple_rerank_scores(q_i, img_i, idx, bank_q, bank_img, sim, alpha=retr_cfg["rerank_alpha"])
                order = np.argsort(-sim, axis=1)
                row = np.arange(1)[:, None]
                idx, sim = idx[row, order], sim[row, order]
                toks = np.stack([
                    img_i[0],
                    q_i[0],
                    agg_features_from_idx(idx, sim, bank_txt)[0],
                ], axis=0)[None].astype(np.float32)
            else:
                qk = build_mm_key(img_i, q_i, retr_cfg["mm_image_weight"], retr_cfg["mm_question_weight"])
                bank_mm = build_mm_key(bank_img, bank_q, retr_cfg["mm_image_weight"], retr_cfg["mm_question_weight"])
                idx, sim = topk_cosine(qk, bank_mm, retr_cfg["topk_mm"], exclude_diagonal=False)
                sim = simple_rerank_scores(q_i, img_i, idx, bank_q, bank_img, sim, alpha=retr_cfg["rerank_alpha"])
                order = np.argsort(-sim, axis=1)
                row = np.arange(1)[:, None]
                idx, sim = idx[row, order], sim[row, order]
                toks = np.stack([
                    img_i[0],
                    q_i[0],
                    agg_features_from_idx(idx, sim, bank_txt)[0],
                    agg_features_from_idx(idx, sim, bank_img)[0],
                ], axis=0)[None].astype(np.float32)

            cand_emb = candidate_embeddings_from_idx(retr_pack, idx, retr_cfg).astype(np.float32)
            cand_prior = sim.astype(np.float32)
            quality = summarize_retrieval_quality(sim).astype(np.float32)

            token_t = torch.from_numpy(toks).float().to(DEVICE)
            cand_t = torch.from_numpy(cand_emb).float().to(DEVICE)
            prior_t = torch.from_numpy(cand_prior).float().to(DEVICE)
            qual_t = torch.from_numpy(quality).float().to(DEVICE)
            _ = model(token_t, cand_t, prior_t, qual_t)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return float(np.median(times)) if times else np.nan

def evaluate_hybrid_pack(dataset_name: str, variant: str, model: nn.Module, split_name: str = "test") -> Tuple[pd.DataFrame, Dict]:
    pack = hybrid_store[dataset_name][variant][split_name]
    threshold = float(getattr(model, "best_gate_threshold", CFG["model"]["gate_threshold"]))
    outputs = predict_hybrid_outputs(model, pack)

    final_answers, gate_prob, used_open, closed_pred_ids, open_best_scores = decode_hybrid_batch(
        outputs,
        pack["candidate_answers"],
        answer_spaces[dataset_name]["closed_id2ans"],
        gate_enabled=CFG["model"]["gate_enabled"] if variant != "base" else False,
        gate_threshold=threshold,
    )

    gold_answers = pack["gold_answer"]
    correct = [normalize_answer(p) == normalize_answer(g) for p, g in zip(final_answers, gold_answers)]
    support_answers = pack["support_answers_for_eval"]
    supported = [support_match(p, sa) for p, sa in zip(final_answers, support_answers)]
    retrieval_hit = [answer_hit(g, sa) for g, sa in zip(gold_answers, support_answers)]
    retrieval_rr = [reciprocal_rank(g, sa) for g, sa in zip(gold_answers, support_answers)]
    closed_mask = ~pack["oov_mask"]
    open_mask = pack["oov_mask"]
    copy_available = pack["copy_available_mask"]

    out = pack["df"].copy().reset_index(drop=True)
    out["pred_answer"] = final_answers
    out["gold_answer"] = gold_answers
    out["correct"] = correct
    out["supported"] = supported
    out["retrieval_hit"] = retrieval_hit
    out["retrieval_rr"] = retrieval_rr
    out["support_answers_joined"] = [" || ".join(map(str, x)) for x in support_answers]
    out["top_support_answer"] = [x[0] if len(x) else "" for x in support_answers]
    out["variant"] = variant
    out["dataset"] = dataset_name
    out["split"] = split_name
    out["gold_closed"] = closed_mask.astype(bool)
    out["gold_open"] = open_mask.astype(bool)
    out["copy_target_available"] = copy_available.astype(bool)
    out["gate_prob"] = gate_prob
    out["used_open_branch"] = used_open.astype(bool)
    out["open_best_score"] = open_best_scores
    out["pred_answer_type"] = out["pred_answer"].map(infer_answer_type)
    out["error_tag"] = out.apply(classify_error, axis=1)

    correct_arr = np.asarray(correct, dtype=bool)
    closed_mask_arr = np.asarray(closed_mask, dtype=bool)
    open_mask_arr = np.asarray(open_mask, dtype=bool)
    in_vocab_mask = closed_mask_arr
    acc_ci_lo, acc_ci_hi = bootstrap_ci_mean(correct_arr.astype(float), n_boot=500, seed=CFG["seed"])

    metrics = {
        "dataset": dataset_name,
        "split": split_name,
        "variant": variant,
        "fusion_type": CFG["model"]["fusion_type"],
        "gate_enabled": CFG["model"]["gate_enabled"] if variant != "base" else False,
        "gate_threshold": threshold,
        "accuracy": float(correct_arr.mean()),
        "accuracy_ci_lo": acc_ci_lo,
        "accuracy_ci_hi": acc_ci_hi,
        "closed_accuracy": float(correct_arr[closed_mask_arr].mean()) if closed_mask_arr.any() else np.nan,
        "open_accuracy": float(correct_arr[open_mask_arr].mean()) if open_mask_arr.any() else np.nan,
        "in_vocab_accuracy": float(correct_arr[in_vocab_mask].mean()) if in_vocab_mask.any() else np.nan,
        "answer_oov_rate": float(open_mask_arr.mean()) if len(open_mask_arr) else np.nan,
        "copy_target_available_rate": float(copy_available.mean()) if len(copy_available) else np.nan,
        "open_branch_usage_rate": float(used_open.mean()) if len(used_open) else np.nan,
        "grounding_support_rate": float(np.mean(supported)),
        "unsupported_answer_rate": float(np.mean([((not c) and (not s)) for c, s in zip(correct, supported)])),
        "retrieval_hit_rate": float(np.mean(retrieval_hit)),
        "retrieval_mrr": float(np.mean(retrieval_rr)),
        "head_latency_ms_median": measure_head_latency(model, pack, CFG["latency"]["warmup"], CFG["latency"]["measure_samples"]),
        "cached_pipeline_latency_ms_median": measure_cached_retrieval_plus_model_latency(
            model, dataset_name, variant, split_name=split_name, measure_samples=min(CFG["latency"]["measure_samples"], 64)
        ) if CFG["latency"]["measure_end_to_end_cached"] else np.nan,
        "model_size_mb": size_mb_of_model(model),
        "best_val_accuracy": getattr(model, "best_val_accuracy", np.nan),
        "best_epoch": getattr(model, "best_epoch", np.nan),
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = predict_hybrid_outputs(model, {
            "tokens": pack["tokens"][:min(64, len(pack["tokens"]))],
            "candidate_emb": pack["candidate_emb"][:min(64, len(pack["candidate_emb"]))],
            "candidate_prior": pack["candidate_prior"][:min(64, len(pack["candidate_prior"]))],
            "quality": pack["quality"][:min(64, len(pack["quality"]))],
        })
        metrics["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        metrics["peak_gpu_mem_mb"] = np.nan

    return out, metrics



def _apply_runtime_overrides(datasets=None, fast_subset=None):
    if datasets is not None:
        keep = set(datasets)
        for name in list(CFG["datasets"].keys()):
            CFG["datasets"][name]["enabled"] = name in keep
    if fast_subset is not None:
        for name in CFG["datasets"]:
            CFG["datasets"][name]["use_fast_subset"] = bool(fast_subset)


def main(mode="full", datasets=None, fast_subset=None, skip_ablations=False, skip_robustness=False, skip_figures=False, skip_reports=False, skip_qualitative=False, skip_warmup=False):
    """Run the v5 pipeline as a scriptable code path.

    Parameters
    ----------
    mode: str
        Either 'full' or 'continue'. Continue mode simply reuses caches/checkpoints when they exist.
    datasets: list[str] | None
        Optional subset of datasets to enable.
    fast_subset: bool | None
        Override dataset fast-subset behaviour.
    skip_*: bool
        Stage flags for lighter reruns.
    """
    _apply_runtime_overrides(datasets=datasets, fast_subset=fast_subset)
    enabled = [k for k,v in CFG['datasets'].items() if v.get('enabled', False)]
    print(f"[compact_medvqa] mode={mode} | datasets={enabled} | device={DEVICE}")

    # Run the three mandatory experiments with validation-tuned settings for the strongest accuracy

    def merge_cfg(base: Dict, updates: Dict) -> Dict:
        out = json.loads(json.dumps(base))
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k].update(v)
            else:
                out[k] = v
        return out

    def candidate_answer_space_trials():
        trials = []
        min_freqs = CFG["tuning"].get("closed_min_freq_candidates", [CFG["answer_space"]["closed_min_freq"]])
        topns = CFG["tuning"].get("closed_topn_candidates", [CFG["answer_space"]["closed_topn_frequent"]])
        max_tokens_list = CFG["tuning"].get("closed_max_tokens_candidates", [CFG["answer_space"]["closed_max_tokens"]])
        seen = set()
        for min_freq in min_freqs:
            for topn in topns:
                for max_tokens in max_tokens_list:
                    item = {
                        "closed_min_freq": int(min_freq),
                        "closed_topn_frequent": int(topn),
                        "closed_max_tokens": int(max_tokens),
                    }
                    key = tuple(sorted(item.items()))
                    if key not in seen:
                        trials.append(item)
                        seen.add(key)
        if not trials:
            trials = [{
                "closed_min_freq": int(CFG["answer_space"]["closed_min_freq"]),
                "closed_topn_frequent": int(CFG["answer_space"]["closed_topn_frequent"]),
                "closed_max_tokens": int(CFG["answer_space"]["closed_max_tokens"]),
            }]
        return trials

    def make_trial_configs(dataset_name: str, variant: str) -> List[Dict[str, Any]]:
        fusion_types = get_search_fusion_types(dataset_name)
        retrieval_base = dict(CFG["retrieval"])
        model_base = dict(CFG["model"])

        answer_trials = candidate_answer_space_trials()
        trials = []

        if variant == "base":
            for fusion in fusion_types:
                for ans_cfg in answer_trials:
                    trials.append({
                        "name": f"fusion={fusion}|ans={ans_cfg['closed_min_freq']}-{ans_cfg['closed_topn_frequent']}-{ans_cfg['closed_max_tokens']}",
                        "retrieval_cfg": retrieval_base,
                        "answer_space_cfg": merge_cfg(CFG["answer_space"], ans_cfg),
                        "model_cfg": merge_cfg(model_base, {"fusion_type": fusion}),
                    })
            return trials

        if variant == "text_rag":
            topks = CFG["tuning"]["text_topk_candidates"].get(dataset_name, [CFG["retrieval"]["topk_text"]])
            for fusion in fusion_types:
                for k in topks:
                    for alpha in CFG["tuning"]["rerank_alpha_candidates"]:
                        for ans_mix in CFG["tuning"]["candidate_answer_mix_candidates"]:
                            for ans_cfg in answer_trials:
                                retr_cfg = merge_cfg(retrieval_base, {
                                    "topk_text": int(k),
                                    "rerank_alpha": float(alpha),
                                    "candidate_answer_mix": float(ans_mix),
                                    "candidate_support_mix": float(1.0 - ans_mix),
                                })
                                model_cfg = merge_cfg(model_base, {"fusion_type": fusion})
                                trials.append({
                                    "name": f"fusion={fusion}|k={k}|alpha={alpha}|ansmix={ans_mix}|ans={ans_cfg['closed_min_freq']}-{ans_cfg['closed_topn_frequent']}-{ans_cfg['closed_max_tokens']}",
                                    "retrieval_cfg": retr_cfg,
                                    "answer_space_cfg": merge_cfg(CFG["answer_space"], ans_cfg),
                                    "model_cfg": model_cfg,
                                })
            return trials

        if variant == "mm_rag":
            topks = CFG["tuning"]["mm_topk_candidates"].get(dataset_name, [CFG["retrieval"]["topk_mm"]])
            for fusion in fusion_types:
                for k in topks:
                    for w_img in CFG["tuning"]["mm_image_weight_candidates"]:
                        for alpha in CFG["tuning"]["rerank_alpha_candidates"]:
                            for ans_mix in CFG["tuning"]["candidate_answer_mix_candidates"]:
                                for ans_cfg in answer_trials:
                                    retr_cfg = merge_cfg(retrieval_base, {
                                        "topk_mm": int(k),
                                        "mm_image_weight": float(w_img),
                                        "mm_question_weight": float(1.0 - w_img),
                                        "rerank_alpha": float(alpha),
                                        "candidate_answer_mix": float(ans_mix),
                                        "candidate_support_mix": float(1.0 - ans_mix),
                                    })
                                    model_cfg = merge_cfg(model_base, {"fusion_type": fusion})
                                    trials.append({
                                        "name": f"fusion={fusion}|k={k}|wimg={w_img}|alpha={alpha}|ansmix={ans_mix}|ans={ans_cfg['closed_min_freq']}-{ans_cfg['closed_topn_frequent']}-{ans_cfg['closed_max_tokens']}",
                                        "retrieval_cfg": retr_cfg,
                                        "answer_space_cfg": merge_cfg(CFG["answer_space"], ans_cfg),
                                        "model_cfg": model_cfg,
                                    })
            return trials

        raise ValueError(variant)

    def derive_gold_closed_open(df: pd.DataFrame, fallback_closed_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        closed = np.asarray(fallback_closed_mask, dtype=bool).copy()

        if "question_type" in df.columns:
            q = df["question_type"].fillna("").astype(str).str.strip().str.lower()
            closed_qtypes = {str(x).strip().lower() for x in CFG["answer_space"].get("force_closed_question_types", [])}
            valid = q.ne("").to_numpy()
            if valid.any():
                q_closed = q.isin(closed_qtypes).to_numpy()
                closed = np.where(valid, q_closed, closed)
        elif "answer_type" in df.columns:
            a = df["answer_type"].fillna("").astype(str).str.strip().str.lower()
            valid = a.ne("").to_numpy()
            if valid.any():
                a_closed = a.isin({"closed", "close", "closed-ended", "yes/no", "binary"}).to_numpy()
                closed = np.where(valid, a_closed, closed)

        open_mask = ~closed
        return closed.astype(bool), open_mask.astype(bool)

    def first_or_empty(x):
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return str(x[0])
        return ""

    def evaluate_generic_pack(dataset_name: str, variant: str, model: nn.Module, pack: Dict, answer_space: Dict, gate_enabled: bool, gate_threshold: float) -> Tuple[pd.DataFrame, Dict]:
        outputs = predict_hybrid_outputs(model, pack)
        final_answers, gate_prob, used_open, closed_pred_ids, open_best_scores = decode_hybrid_batch(
            outputs,
            pack["candidate_answers"],
            answer_space["closed_id2ans"],
            gate_enabled=gate_enabled,
            gate_threshold=gate_threshold,
        )

        gold_answers = pack["gold_answer"]
        correct = [normalize_answer(p) == normalize_answer(g) for p, g in zip(final_answers, gold_answers)]
        support_answers = pack["support_answers_for_eval"]
        supported = [support_match(p, sa) for p, sa in zip(final_answers, support_answers)]
        retrieval_hit = [answer_hit(g, sa) for g, sa in zip(gold_answers, support_answers)]
        retrieval_rr = [reciprocal_rank(g, sa) for g, sa in zip(gold_answers, support_answers)]

        out = pack["df"].copy().reset_index(drop=True)
        out["dataset"] = dataset_name
        out["variant"] = variant
        if "sample_id" not in out.columns:
            out["sample_id"] = [f"{dataset_name}_{variant}_{i}" for i in range(len(out))]
        out["pred_answer"] = final_answers
        out["gold_answer"] = gold_answers
        out["correct"] = correct
        out["supported"] = supported
        out["retrieval_hit"] = retrieval_hit
        out["retrieval_rr"] = retrieval_rr
        out["gate_prob"] = gate_prob
        out["used_open_branch"] = used_open.astype(bool)
        out["open_best_score"] = open_best_scores
        out["copy_target_available"] = np.asarray(pack["copy_available_mask"]).astype(bool)
        out["top_support_answer"] = [first_or_empty(x) for x in support_answers]
        out["support_answers"] = [" || ".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x) for x in support_answers]

        correct_arr = np.asarray(correct, dtype=bool)
        fallback_closed_mask = ~np.asarray(pack["oov_mask"], dtype=bool)
        closed_mask_arr, open_mask_arr = derive_gold_closed_open(out, fallback_closed_mask)
        out["gold_closed"] = closed_mask_arr.astype(bool)
        out["gold_open"] = open_mask_arr.astype(bool)
        out["error_tag"] = out.apply(classify_error, axis=1)
        metrics = {
            "dataset": dataset_name,
            "variant": variant,
            "accuracy": float(correct_arr.mean()),
            "closed_accuracy": float(correct_arr[closed_mask_arr].mean()) if closed_mask_arr.any() else np.nan,
            "open_accuracy": float(correct_arr[open_mask_arr].mean()) if open_mask_arr.any() else np.nan,
            "grounding_support_rate": float(np.mean(supported)),
            "unsupported_answer_rate": float(np.mean([((not c) and (not s)) for c, s in zip(correct, supported)])),
            "retrieval_hit_rate": float(np.mean(retrieval_hit)),
            "retrieval_mrr": float(np.mean(retrieval_rr)),
            "answer_oov_rate": float(open_mask_arr.mean()) if len(open_mask_arr) else np.nan,
            "copy_target_available_rate": float(pack["copy_available_mask"].mean()) if len(pack["copy_available_mask"]) else np.nan,
            "open_branch_usage_rate": float(np.mean(used_open)) if len(used_open) else np.nan,
            "gate_threshold": float(gate_threshold),
        }
        return out, metrics

    search_rows = []
    all_metrics = []
    all_logs = []
    trained_models = {}
    histories = {}
    selected_configs = {}
    selected_retrieval_store = {}
    selected_hybrid_store = {}
    selected_answer_spaces = {}

    for dataset_name in bundles.keys():
        trained_models[dataset_name] = {}
        histories[dataset_name] = {}
        selected_configs[dataset_name] = {}
        selected_retrieval_store[dataset_name] = {}
        selected_hybrid_store[dataset_name] = {}
        selected_answer_spaces[dataset_name] = {}

        for variant in VARIANTS:
            print("\n===== DATASET:", dataset_name, "| VARIANT:", variant, "=====")
            trial_list = make_trial_configs(dataset_name, variant) if CFG["tuning"]["enabled"] else [{
                "name": "default",
                "retrieval_cfg": dict(CFG["retrieval"]),
                "answer_space_cfg": dict(CFG["answer_space"]),
                "model_cfg": dict(CFG["model"]),
            }]

            best_trial = None
            best_trial_model = None
            best_trial_store = None
            best_trial_answer_space = None
            best_trial_history = None
            best_trial_val_metrics = None

            for trial_idx, trial in enumerate(trial_list, start=1):
                print(f"[search {trial_idx}/{len(trial_list)}] {trial['name']}")

                custom_retrieval = build_retrieval_package(dataset_name, retrieval_cfg=trial["retrieval_cfg"])
                ds_store, ds_answer_space, _ = build_dataset_hybrid_store(
                    custom_retrieval,
                    answer_space_cfg=trial["answer_space_cfg"],
                    retrieval_cfg=trial["retrieval_cfg"],
                )

                tr = ds_store[variant]["train"]
                va = ds_store[variant]["val"]

                if len(va["tokens"]) == 0:
                    n_fallback = max(32, int(0.1 * len(tr["tokens"])))
                    va = {
                        k: (v[:n_fallback] if isinstance(v, np.ndarray) else v[:n_fallback] if isinstance(v, list) else v)
                        for k, v in tr.items()
                    }

                model_cfg = dict(trial["model_cfg"])
                train_cfg = merge_cfg(CFG["training"], {
                    "epochs": CFG["tuning"]["search_epochs"],
                    "patience": CFG["tuning"]["search_patience"],
                    "batch_size": CFG["tuning"]["search_batch_size"],
                })

                model, history = train_hybrid_model(
                    train_pack=tr,
                    val_pack=va,
                    num_closed=len(ds_answer_space["closed_ans2id"]),
                    cfg_model=model_cfg,
                    cfg_train=train_cfg,
                    closed_id2ans=ds_answer_space["closed_id2ans"],
                    unk_id=ds_answer_space["unk_id"],
                )

                _, val_metrics = evaluate_generic_pack(
                    dataset_name=dataset_name,
                    variant=variant,
                    model=model,
                    pack=va,
                    answer_space=ds_answer_space,
                    gate_enabled=(model_cfg["gate_enabled"] if variant != "base" else False),
                    gate_threshold=float(getattr(model, "best_gate_threshold", model_cfg["gate_threshold"])),
                )

                val_score = selection_score_from_metrics(dataset_name, variant, val_metrics)
                search_row = {
                    "dataset": dataset_name,
                    "variant": variant,
                    "trial_name": trial["name"],
                    "search_rank": trial_idx,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_closed_accuracy": val_metrics["closed_accuracy"],
                    "val_open_accuracy": val_metrics["open_accuracy"],
                    "val_grounding_support_rate": val_metrics["grounding_support_rate"],
                    "val_unsupported_answer_rate": val_metrics["unsupported_answer_rate"],
                    "val_retrieval_hit_rate": val_metrics["retrieval_hit_rate"],
                    "val_selection_score": val_score,
                    "selected_gate_threshold": float(getattr(model, "best_gate_threshold", model_cfg["gate_threshold"])),
                    "fusion_type": model_cfg["fusion_type"],
                    "topk_text": trial["retrieval_cfg"]["topk_text"],
                    "topk_mm": trial["retrieval_cfg"]["topk_mm"],
                    "mm_image_weight": trial["retrieval_cfg"]["mm_image_weight"],
                    "rerank_alpha": trial["retrieval_cfg"]["rerank_alpha"],
                    "candidate_answer_mix": trial["retrieval_cfg"]["candidate_answer_mix"],
                    "closed_min_freq": trial["answer_space_cfg"]["closed_min_freq"],
                    "closed_topn_frequent": trial["answer_space_cfg"]["closed_topn_frequent"],
                    "closed_max_tokens": trial["answer_space_cfg"]["closed_max_tokens"],
                }
                search_rows.append(search_row)

                current_key = (
                    val_score,
                    val_metrics["accuracy"],
                    val_metrics["open_accuracy"] if not np.isnan(val_metrics["open_accuracy"]) else -1.0,
                    val_metrics["closed_accuracy"] if not np.isnan(val_metrics["closed_accuracy"]) else -1.0,
                    val_metrics["grounding_support_rate"],
                )
                best_key = None if best_trial_val_metrics is None else (
                    selection_score_from_metrics(dataset_name, variant, best_trial_val_metrics),
                    best_trial_val_metrics["accuracy"],
                    best_trial_val_metrics["open_accuracy"] if not np.isnan(best_trial_val_metrics["open_accuracy"]) else -1.0,
                    best_trial_val_metrics["closed_accuracy"] if not np.isnan(best_trial_val_metrics["closed_accuracy"]) else -1.0,
                    best_trial_val_metrics["grounding_support_rate"],
                )

                if best_key is None or current_key > best_key:
                    best_trial = trial
                    best_trial_model = model
                    best_trial_store = ds_store
                    best_trial_answer_space = ds_answer_space
                    best_trial_history = history
                    best_trial_val_metrics = val_metrics

            selected_configs[dataset_name][variant] = best_trial
            selected_retrieval_store[dataset_name][variant] = build_retrieval_package(dataset_name, retrieval_cfg=best_trial["retrieval_cfg"])
            selected_hybrid_store[dataset_name][variant] = best_trial_store[variant]
            selected_answer_spaces[dataset_name][variant] = best_trial_answer_space

            # final full run from scratch with the selected configuration
            final_retrieval = selected_retrieval_store[dataset_name][variant]
            final_ds_store, final_answer_space, _ = build_dataset_hybrid_store(
                final_retrieval,
                answer_space_cfg=best_trial["answer_space_cfg"],
                retrieval_cfg=best_trial["retrieval_cfg"],
            )
            tr = final_ds_store[variant]["train"]
            va = final_ds_store[variant]["val"]
            if len(va["tokens"]) == 0:
                n_fallback = max(32, int(0.1 * len(tr["tokens"])))
                va = {
                    k: (v[:n_fallback] if isinstance(v, np.ndarray) else v[:n_fallback] if isinstance(v, list) else v)
                    for k, v in tr.items()
                }

            final_model_cfg = dict(best_trial["model_cfg"])
            final_train_cfg = get_dataset_training_cfg(dataset_name)

            seed_list = [CFG["seed"]]
            if CFG.get("ensemble", {}).get("enabled", False):
                seed_list = list(CFG["ensemble"].get("seed_sets", {}).get(variant, [CFG["seed"]]))

            final_models = []
            final_histories = []
            for seed_idx, seed in enumerate(seed_list, start=1):
                print(f"[final {seed_idx}/{len(seed_list)}] seed={seed}")
                seed_everything(int(seed))
                model_seed, history_seed = train_hybrid_model(
                    train_pack=tr,
                    val_pack=va,
                    num_closed=len(final_answer_space["closed_ans2id"]),
                    cfg_model=final_model_cfg,
                    cfg_train=final_train_cfg,
                    closed_id2ans=final_answer_space["closed_id2ans"],
                    unk_id=final_answer_space["unk_id"],
                )
                final_models.append(model_seed)
                final_histories.append(history_seed)
                history_path = CSV_DIR / f"history_v5_{dataset_name}_{variant}_seed{seed}.csv"
                history_seed.to_csv(history_path, index=False)

                checkpoint = {
                    "dataset": dataset_name,
                    "variant": variant,
                    "seed": int(seed),
                    "state_dict": {k: v.detach().cpu() for k, v in model_seed.state_dict().items()},
                    "model_cfg": final_model_cfg,
                    "training_cfg": final_train_cfg,
                    "selected_trial": best_trial,
                    "answer_space": final_answer_space,
                    "best_gate_threshold": float(getattr(model_seed, "best_gate_threshold", final_model_cfg["gate_threshold"])),
                }
                torch.save(checkpoint, CKPT_DIR / f"v5_{dataset_name}_{variant}_seed{seed}.pt")

            final_model = final_models if len(final_models) > 1 else final_models[0]
            history = final_histories[0] if len(final_histories) == 1 else pd.concat(
                [h.assign(seed=seed_list[idx]) for idx, h in enumerate(final_histories)],
                ignore_index=True,
            )

            trained_models[dataset_name][variant] = final_model
            histories[dataset_name][variant] = history
            selected_hybrid_store[dataset_name][variant] = final_ds_store[variant]
            selected_answer_spaces[dataset_name][variant] = final_answer_space

            pred_log, metrics = evaluate_generic_pack(
                dataset_name=dataset_name,
                variant=variant,
                model=final_model,
                pack=selected_hybrid_store[dataset_name][variant]["test"],
                answer_space=selected_answer_spaces[dataset_name][variant],
                gate_enabled=(final_model_cfg["gate_enabled"] if variant != "base" else False),
                gate_threshold=float(getattr(final_model, "best_gate_threshold", final_model_cfg["gate_threshold"])),
            )
            # --- ensemble-safe metric helpers ---
            ensemble_members = list(final_model) if isinstance(final_model, (list, tuple)) else [final_model]
            latency_model = ensemble_members[0]   # use first member for latency measurement
            ensemble_size = len(ensemble_members)
        
            single_model_size_mb = size_mb_of_model(latency_model)
            total_model_size_mb = single_model_size_mb * ensemble_size
        
            best_val_accuracy = float(np.nanmean([getattr(m, "best_val_accuracy", np.nan) for m in ensemble_members]))
            best_epoch = float(np.nanmean([getattr(m, "best_epoch", np.nan) for m in ensemble_members]))
            metrics.update({
                "split": "test",
                "fusion_type": final_model_cfg["fusion_type"],
                "gate_enabled": final_model_cfg["gate_enabled"] if variant != "base" else False,
                "head_latency_ms_median": measure_head_latency(
                    latency_model,
                    selected_hybrid_store[dataset_name][variant]["test"],
                    CFG["latency"]["warmup"],
                    CFG["latency"]["measure_samples"],
                ),
                "cached_pipeline_latency_ms_median": measure_cached_retrieval_plus_model_latency(
                    latency_model,
                    dataset_name,
                    variant,
                    split_name="test",
                    measure_samples=min(CFG["latency"]["measure_samples"], 64),
                ) if CFG["latency"]["measure_end_to_end_cached"] else np.nan,
                "model_size_mb": total_model_size_mb,
                "single_model_size_mb": single_model_size_mb,
                "ensemble_size": ensemble_size,
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
                "topk_text": best_trial["retrieval_cfg"]["topk_text"],
                "topk_mm": best_trial["retrieval_cfg"]["topk_mm"],
                "mm_image_weight": best_trial["retrieval_cfg"]["mm_image_weight"],
                "mm_question_weight": best_trial["retrieval_cfg"]["mm_question_weight"],
                "rerank_alpha": best_trial["retrieval_cfg"]["rerank_alpha"],
                "candidate_answer_mix": best_trial["retrieval_cfg"]["candidate_answer_mix"],
                "closed_min_freq": best_trial["answer_space_cfg"]["closed_min_freq"],
                "closed_topn_frequent": best_trial["answer_space_cfg"]["closed_topn_frequent"],
                "closed_max_tokens": best_trial["answer_space_cfg"]["closed_max_tokens"],
            })
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                _ = predict_hybrid_outputs(final_model, {
                    "tokens": selected_hybrid_store[dataset_name][variant]["test"]["tokens"][:min(64, len(selected_hybrid_store[dataset_name][variant]["test"]["tokens"]))],
                    "candidate_emb": selected_hybrid_store[dataset_name][variant]["test"]["candidate_emb"][:min(64, len(selected_hybrid_store[dataset_name][variant]["test"]["candidate_emb"]))],
                    "candidate_prior": selected_hybrid_store[dataset_name][variant]["test"]["candidate_prior"][:min(64, len(selected_hybrid_store[dataset_name][variant]["test"]["candidate_prior"]))],
                    "quality": selected_hybrid_store[dataset_name][variant]["test"]["quality"][:min(64, len(selected_hybrid_store[dataset_name][variant]["test"]["quality"]))],
                })
                metrics["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                metrics["peak_gpu_mem_mb"] = np.nan

            pred_log.to_csv(CSV_DIR / f"predictions_v5_{dataset_name}_{variant}.csv", index=False)
            all_logs.append(pred_log)
            all_metrics.append(metrics)

    search_df = pd.DataFrame(search_rows)
    metrics_df = pd.DataFrame(all_metrics)
    logs_df = pd.concat(all_logs, ignore_index=True)

    search_df.to_csv(CSV_DIR / "search_metrics_v5.csv", index=False)
    metrics_df.to_csv(CSV_DIR / "main_metrics_v5.csv", index=False)
    logs_df.to_csv(CSV_DIR / "all_predictions_v3.csv", index=False)

    with open(JSON_DIR / "selected_configs_v5.json", "w") as f:
        json.dump(selected_configs, f, indent=2)

    display_cols = [
        "dataset", "variant", "accuracy", "closed_accuracy", "open_accuracy",
        "grounding_support_rate", "unsupported_answer_rate", "retrieval_hit_rate",
        "head_latency_ms_median", "cached_pipeline_latency_ms_median",
        "peak_gpu_mem_mb", "model_size_mb", "gate_threshold", "fusion_type",
        "topk_text", "topk_mm", "candidate_answer_mix",
    ]
    metrics_df[display_cols]


    if not skip_ablations:
        # Ablation studies around the selected best configuration

        ABLATION_DATASET = CFG["ablation"]["dataset"] if CFG["ablation"]["dataset"] in bundles else list(bundles.keys())[0]
        print("Ablation dataset:", ABLATION_DATASET)

        def deep_update_cfg(base: Dict, update: Dict) -> Dict:
            out = json.loads(json.dumps(base))
            for k, v in update.items():
                out[k] = v
            return out

        def run_ablation_trial(dataset_name: str, variant: str, retrieval_cfg: Dict, answer_space_cfg: Dict, model_cfg: Dict, train_cfg: Optional[Dict] = None):
            custom_retrieval = build_retrieval_package(dataset_name, retrieval_cfg=retrieval_cfg)
            ds_store, ds_answer_space, _ = build_dataset_hybrid_store(
                custom_retrieval,
                answer_space_cfg=answer_space_cfg,
                retrieval_cfg=retrieval_cfg,
            )
            tr = ds_store[variant]["train"]
            va = ds_store[variant]["val"]
            if len(va["tokens"]) == 0:
                n_fallback = max(32, int(0.1 * len(tr["tokens"])))
                va = {
                    k: (v[:n_fallback] if isinstance(v, np.ndarray) else v[:n_fallback] if isinstance(v, list) else v)
                    for k, v in tr.items()
                }

            train_cfg = dict(CFG["training"]) if train_cfg is None else dict(train_cfg)
            train_cfg["epochs"] = min(train_cfg["epochs"], max(20, CFG["tuning"]["search_epochs"]))
            train_cfg["patience"] = min(train_cfg["patience"], CFG["tuning"]["search_patience"])

            model, _ = train_hybrid_model(
                train_pack=tr,
                val_pack=va,
                num_closed=len(ds_answer_space["closed_ans2id"]),
                cfg_model=model_cfg,
                cfg_train=train_cfg,
                closed_id2ans=ds_answer_space["closed_id2ans"],
                unk_id=ds_answer_space["unk_id"],
            )
            _, val_metrics = evaluate_generic_pack(
                dataset_name=dataset_name,
                variant=variant,
                model=model,
                pack=va,
                answer_space=ds_answer_space,
                gate_enabled=(model_cfg["gate_enabled"] if variant != "base" else False),
                gate_threshold=float(getattr(model, "best_gate_threshold", model_cfg["gate_threshold"])),
            )
            return val_metrics

        available_trials = selected_configs.get(ABLATION_DATASET, {})
        variants_for_ablation = [v for v in ["text_rag", "mm_rag"] if v in available_trials]

        if not variants_for_ablation:
            raise ValueError(
                f"No selected ablation configs found for dataset={ABLATION_DATASET}. "
                "Re-run Cell 13 first."
            )

        missing_variants = sorted(set(["text_rag", "mm_rag"]) - set(variants_for_ablation))
        if missing_variants:
            print(f"Warning: missing selected configs for {ABLATION_DATASET}: {missing_variants}. "
                  f"Running ablations only for: {variants_for_ablation}")

        ablation_rows = []
        for variant in variants_for_ablation:
            base_trial = available_trials[variant]
            base_retrieval = dict(base_trial["retrieval_cfg"])
            base_answer_space = dict(base_trial["answer_space_cfg"])
            base_model = dict(base_trial["model_cfg"])

            if variant == "text_rag":
                for k in CFG["ablation"]["text_topk_list"]:
                    rcfg = deep_update_cfg(base_retrieval, {"topk_text": int(k)})
                    metrics = run_ablation_trial(ABLATION_DATASET, variant, rcfg, base_answer_space, base_model)
                    metrics.update({"ablation_group": "text_topk", "setting": f"k={k}"})
                    ablation_rows.append(metrics)

            if variant == "mm_rag":
                for k in CFG["ablation"]["mm_topk_list"]:
                    rcfg = deep_update_cfg(base_retrieval, {"topk_mm": int(k)})
                    metrics = run_ablation_trial(ABLATION_DATASET, variant, rcfg, base_answer_space, base_model)
                    metrics.update({"ablation_group": "mm_topk", "setting": f"k={k}"})
                    ablation_rows.append(metrics)

                for w in CFG["ablation"]["mm_image_weights"]:
                    rcfg = deep_update_cfg(base_retrieval, {"mm_image_weight": float(w), "mm_question_weight": float(1.0 - w)})
                    metrics = run_ablation_trial(ABLATION_DATASET, variant, rcfg, base_answer_space, base_model)
                    metrics.update({"ablation_group": "mm_weight", "setting": f"w_img={w:.2f}, w_q={1.0-w:.2f}"})
                    ablation_rows.append(metrics)

            for alpha in CFG["ablation"]["rerank_alpha_list"]:
                rcfg = deep_update_cfg(base_retrieval, {"rerank_alpha": float(alpha)})
                metrics = run_ablation_trial(ABLATION_DATASET, variant, rcfg, base_answer_space, base_model)
                metrics.update({"ablation_group": "rerank_alpha", "setting": f"alpha={alpha:.2f}"})
                ablation_rows.append(metrics)

            for amix in CFG["ablation"]["candidate_answer_mix_list"]:
                rcfg = deep_update_cfg(base_retrieval, {"candidate_answer_mix": float(amix), "candidate_support_mix": float(1.0 - amix)})
                metrics = run_ablation_trial(ABLATION_DATASET, variant, rcfg, base_answer_space, base_model)
                metrics.update({"ablation_group": "candidate_answer_mix", "setting": f"ans_mix={amix:.2f}"})
                ablation_rows.append(metrics)

            for fusion in CFG["ablation"]["fusion_types"]:
                mcfg = deep_update_cfg(base_model, {"fusion_type": fusion})
                metrics = run_ablation_trial(ABLATION_DATASET, variant, base_retrieval, base_answer_space, mcfg)
                metrics.update({"ablation_group": "fusion_type", "setting": fusion})
                ablation_rows.append(metrics)

            for gate in CFG["ablation"]["gate_options"]:
                mcfg = deep_update_cfg(base_model, {"gate_enabled": bool(gate)})
                metrics = run_ablation_trial(ABLATION_DATASET, variant, base_retrieval, base_answer_space, mcfg)
                metrics.update({"ablation_group": "gate_enabled", "setting": str(bool(gate))})
                ablation_rows.append(metrics)

            for cmf in CFG["ablation"]["closed_min_freq_list"]:
                acfg = deep_update_cfg(base_answer_space, {"closed_min_freq": int(cmf)})
                metrics = run_ablation_trial(ABLATION_DATASET, variant, base_retrieval, acfg, base_model)
                metrics.update({"ablation_group": "closed_min_freq", "setting": f"{cmf}"})
                ablation_rows.append(metrics)

        ablation_df = pd.DataFrame(ablation_rows)
        ablation_df.to_csv(CSV_DIR / "ablation_metrics_v5.csv", index=False)
        ablation_df.sort_values(["ablation_group", "setting"]).reset_index(drop=True)


    if not skip_robustness:
        # Robustness: retrieval corruption and retrieval dropout for the selected multimodal model


        stable_min_topk = int(CFG["robustness"].get("min_topk_for_stability_eval", 3))
        robust_trial = selected_configs[ABLATION_DATASET]["mm_rag"]
        robust_retrieval_cfg = dict(robust_trial["retrieval_cfg"])
        if int(robust_retrieval_cfg.get("topk_mm", 1)) < stable_min_topk:
            robust_retrieval_cfg["topk_mm"] = stable_min_topk
            print(f"Robustness companion config: overriding topk_mm to {stable_min_topk} for non-degenerate robustness curves.")
            robust_retrieval = build_retrieval_package(ABLATION_DATASET, retrieval_cfg=robust_retrieval_cfg)
            robust_store, robust_answer_space, _ = build_dataset_hybrid_store(
                robust_retrieval,
                answer_space_cfg=robust_trial["answer_space_cfg"],
                retrieval_cfg=robust_retrieval_cfg,
            )
            robust_pack = robust_store["mm_rag"]["test"]
            robust_answer_space_ref = robust_answer_space
        else:
            robust_pack = selected_hybrid_store[ABLATION_DATASET]["mm_rag"]["test"]
            robust_answer_space_ref = selected_answer_spaces[ABLATION_DATASET]["mm_rag"]

        def corrupt_indices(original_idx: np.ndarray, bank_df: pd.DataFrame, gold_answers: List[str], ratio: float, seed: int = 42):
            rng = np.random.default_rng(seed)
            idx = original_idx.copy()
            if ratio <= 0:
                return idx
            n, k = idx.shape
            n_replace = int(np.floor(k * ratio))
            if ratio >= 1.0:
                n_replace = k
            if n_replace <= 0:
                return idx
            answer_array = bank_df["answer"].astype(str).tolist()
            for i in range(n):
                wrong_pool = [j for j, a in enumerate(answer_array) if normalize_answer(a) != normalize_answer(gold_answers[i])]
                if len(wrong_pool) == 0:
                    continue
                replace = rng.choice(wrong_pool, size=n_replace, replace=len(wrong_pool) < n_replace)
                idx[i, :n_replace] = replace
            return idx

        def dropout_support_indices(original_idx: np.ndarray, keep_ratio: float, seed: int = 42):
            # When k=1, dropout levels below 1.0 are effectively no-ops; keep that interpretation explicit.
            rng = np.random.default_rng(seed)
            idx = original_idx.copy()
            if keep_ratio >= 1.0:
                return idx
            n, k = idx.shape
            keep_k = max(1, int(round(k * keep_ratio)))
            out = []
            for i in range(n):
                chosen = np.sort(rng.choice(np.arange(k), size=keep_k, replace=False))
                row = idx[i, chosen]
                if keep_k < k:
                    pad = np.repeat(row[:1], k - keep_k)
                    row = np.concatenate([row, pad], axis=0)
                out.append(row)
            return np.stack(out, axis=0)

        def build_mm_pack_from_support_indices(dataset_name: str, support_idx: np.ndarray, support_sim: Optional[np.ndarray], retrieval_cfg: Dict, answer_space: Dict):
            query_pack = build_retrieval_package(dataset_name, retrieval_cfg=retrieval_cfg)["test"]
            bank_df = query_pack["bank_df"]
            bank_txt = query_pack["bank_txt"]
            bank_img = query_pack["bank_img"]
            df = query_pack["df"].copy().reset_index(drop=True)

            mm_text_sup = agg_features_from_idx(support_idx, support_sim, bank_txt)
            mm_img_sup = agg_features_from_idx(support_idx, support_sim, bank_img)

            tokens = np.stack([
                query_pack["image_emb"],
                query_pack["question_emb"],
                mm_text_sup,
                mm_img_sup,
            ], axis=1).astype(np.float32)
            cand_emb = candidate_embeddings_from_idx(query_pack, support_idx, retrieval_cfg).astype(np.float32)
            cand_answers = gather_answer_lists(bank_df, support_idx)
            candidate_prior = support_sim.astype(np.float32)
            quality = summarize_retrieval_quality(support_sim).astype(np.float32)
            closed_y, closed_mask = encode_closed_answers(df, answer_space["closed_ans2id"])
            open_target_mask = locate_open_target_mask(df["answer"].tolist(), cand_answers)
            copy_available = open_target_mask.sum(axis=1) > 0 if len(open_target_mask) else np.zeros(len(df), dtype=bool)

            return {
                "tokens": tokens,
                "candidate_emb": cand_emb,
                "candidate_idx": support_idx.astype(np.int64),
                "candidate_answers": cand_answers,
                "candidate_prior": candidate_prior,
                "quality": quality,
                "closed_y": closed_y.astype(np.int64),
                "route_y": ((~closed_mask) & copy_available).astype(np.float32),
                "open_target_mask": open_target_mask.astype(np.float32),
                "gold_answer": df["answer"].astype(str).tolist(),
                "df": df,
                "support_idx_for_eval": support_idx.astype(np.int64),
                "support_answers_for_eval": cand_answers,
                "bank_df": bank_df.copy().reset_index(drop=True),
                "oov_mask": (~closed_mask).astype(bool),
                "copy_available_mask": copy_available.astype(bool),
            }

        robust_rows = []

        for dataset_name in bundles.keys():
            variant = "mm_rag"
            mm_model = trained_models[dataset_name][variant]
            mm_trial = selected_configs[dataset_name][variant]
            mm_retrieval_cfg = mm_trial["retrieval_cfg"]
            mm_answer_space = selected_answer_spaces[dataset_name][variant]

            test_retr = selected_retrieval_store[dataset_name][variant]["test"]
            base_support_idx = test_retr["mm_idx"]
            base_support_sim = test_retr["mm_sim"]
            gold_answers = test_retr["df"]["answer"].tolist()
            bank_df = test_retr["bank_df"]

            for ratio in CFG["robustness"]["corruption_levels"]:
                bad_idx = corrupt_indices(base_support_idx, bank_df, gold_answers, ratio, seed=CFG["seed"])
                bad_pack = build_mm_pack_from_support_indices(dataset_name, bad_idx, base_support_sim, mm_retrieval_cfg, mm_answer_space)
                _, result = evaluate_generic_pack(
                    dataset_name=dataset_name,
                    variant=variant,
                    model=mm_model,
                    pack=bad_pack,
                    answer_space=mm_answer_space,
                    gate_enabled=(mm_trial["model_cfg"]["gate_enabled"]),
                    gate_threshold=float(getattr(mm_model, "best_gate_threshold", mm_trial["model_cfg"]["gate_threshold"])),
                )
                result.update({"robustness_type": "corruption", "level": ratio})
                robust_rows.append(result)

            for drop in CFG["robustness"]["dropout_levels"]:
                keep_ratio = max(1e-6, 1.0 - drop)
                dropped_idx = dropout_support_indices(base_support_idx, keep_ratio=keep_ratio, seed=CFG["seed"])
                keep_k = max(1, int(round(base_support_sim.shape[1] * keep_ratio)))
                dropped_sim = np.concatenate(
                    [base_support_sim[:, :keep_k], np.repeat(base_support_sim[:, :1], base_support_sim.shape[1] - keep_k, axis=1)],
                    axis=1,
                ).astype(np.float32)
                dropped_pack = build_mm_pack_from_support_indices(dataset_name, dropped_idx, dropped_sim, mm_retrieval_cfg, mm_answer_space)
                _, result = evaluate_generic_pack(
                    dataset_name=dataset_name,
                    variant=variant,
                    model=mm_model,
                    pack=dropped_pack,
                    answer_space=mm_answer_space,
                    gate_enabled=(mm_trial["model_cfg"]["gate_enabled"]),
                    gate_threshold=float(getattr(mm_model, "best_gate_threshold", mm_trial["model_cfg"]["gate_threshold"])),
                )
                result.update({"robustness_type": "dropout", "level": drop})
                robust_rows.append(result)

        robust_df = pd.DataFrame(robust_rows)
        robust_df.to_csv(CSV_DIR / "robustness_metrics_v5.csv", index=False)
        robust_df


    if not skip_warmup:
        # Optional warm-up scaffold (disabled by default)
        # This cell is intentionally lightweight: it documents how to add an external weak-supervision
        # stage without making the benchmark notebook dependent on extra datasets.

        def maybe_run_optional_warmup():
            if not CFG["warmup"]["enabled"]:
                print("Warm-up stage skipped. CFG['warmup']['enabled'] = False")
                print(CFG["warmup"]["note"])
                return None

            print("Warm-up stage enabled.")
            print("Recommended sources for weak supervision:")
            print("- PMC-VQA")
            print("- ROCO captions converted to pseudo-QA")
            print("- MedPix-style case text converted to pseudo-QA")
            print("")
            print("Practical recipe:")
            print("1. Build pseudo QA pairs with answer text and optional rationale.")
            print("2. Reuse the same frozen encoder.")
            print("3. Warm up only the fusion / routing / copy heads for 1-3 epochs.")
            print("4. Fine-tune on ImageCLEF VQA-Med 2019 and SLAKE as in the main training cell.")
            print("")
            print("This scaffold is a placeholder because external-data licensing and preprocessing differ by source.")

        maybe_run_optional_warmup()


    if not skip_qualitative:
        # Qualitative example mining and error taxonomy

        required_cols = ["dataset", "variant", "error_tag", "correct", "supported", "used_open_branch"]
        missing = [c for c in required_cols if c not in logs_df.columns]
        if missing:
            raise ValueError(
                f"logs_df is missing required columns for qualitative analysis: {missing}. "
                "Re-run the main experiment cell after applying the patched evaluate_generic_pack."
            )

        taxonomy = (
            logs_df.groupby(["dataset", "variant", "error_tag"])
            .size()
            .reset_index(name="count")
            .sort_values(["dataset", "variant", "count"], ascending=[True, True, False])
        )
        taxonomy.to_csv(CSV_DIR / "error_taxonomy_v5.csv", index=False)

        def select_qual_examples(df: pd.DataFrame, n_each: int = 2) -> pd.DataFrame:
            frames = []
            mask_supported = (df["variant"].isin(["text_rag", "mm_rag"])) & (df["correct"]) & (df["supported"])
            mask_open_good = df.get("gold_open", pd.Series(False, index=df.index)).fillna(False).astype(bool) & df["correct"]
            mask_route_miss = df["error_tag"].eq("routing-miss")
            mask_hall = df["error_tag"].isin(["unsupported-hallucination", "gold-oov-copy-miss", "retrieval-miss"])
            for mask in [mask_supported, mask_open_good, mask_route_miss, mask_hall]:
                picked = df[mask].head(n_each)
                if len(picked):
                    frames.append(picked)
            if not frames:
                return df.head(min(len(df), max(6, n_each * 4))).copy()
            out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["sample_id", "variant"])
            return out.head(max(6, n_each * 4))

        qual_examples = []
        for dataset_name in bundles.keys():
            sub = logs_df[logs_df["dataset"] == dataset_name].copy()
            if len(sub):
                qual_examples.append(select_qual_examples(sub, n_each=2))
        qual_df = pd.concat(qual_examples, ignore_index=True) if qual_examples else pd.DataFrame()
        qual_df.to_json(JSON_DIR / "qualitative_examples_v5.json", orient="records", indent=2)

        taxonomy_summary = (
            taxonomy.groupby(["dataset", "variant"])
            .apply(lambda x: dict(zip(x["error_tag"], x["count"])), include_groups=False)
            .reset_index(name="error_breakdown")
        )
        taxonomy_summary.to_json(JSON_DIR / "error_taxonomy_summary_v5.json", orient="records", indent=2)

        show_cols = [c for c in [
            "dataset", "variant", "question", "gold_answer", "pred_answer",
            "top_support_answer", "gate_prob", "used_open_branch", "error_tag"
        ] if c in qual_df.columns]
        qual_df[show_cols].head(16)


    if not skip_figures:
        # Figure generation

        def save_rag_workflow_figure(path: Path):
            fig, ax = plt.subplots(figsize=(11.8, 4.8))
            ax.axis("off")
            boxes = [
                (0.02, 0.35, 0.14, 0.28, "Input\nimage + question"),
                (0.20, 0.35, 0.16, 0.28, "Frozen\nBiomedCLIP encoders"),
                (0.40, 0.60, 0.18, 0.18, "Text retrieval\n+ reranking"),
                (0.40, 0.34, 0.18, 0.18, "Answer-aware\ncandidate bank"),
                (0.40, 0.08, 0.18, 0.18, "Multimodal retrieval\n+ reranking"),
                (0.62, 0.35, 0.16, 0.28, "Fusion encoder\n(MLP / Transformer)"),
                (0.82, 0.60, 0.14, 0.14, "Closed\nanswer head"),
                (0.82, 0.38, 0.14, 0.14, "Open copy\nanswer head"),
                (0.82, 0.16, 0.14, 0.14, "Gate +\nthreshold tuning"),
            ]
            for x, y, w, h, txt in boxes:
                rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
                ax.add_patch(rect)
                ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=10.5)
            arrows = [
                ((0.16, 0.49), (0.20, 0.49)),
                ((0.36, 0.49), (0.40, 0.69)),
                ((0.36, 0.49), (0.40, 0.43)),
                ((0.36, 0.49), (0.40, 0.17)),
                ((0.58, 0.69), (0.62, 0.49)),
                ((0.58, 0.43), (0.62, 0.49)),
                ((0.58, 0.17), (0.62, 0.49)),
                ((0.78, 0.49), (0.82, 0.67)),
                ((0.78, 0.49), (0.82, 0.45)),
                ((0.78, 0.49), (0.82, 0.23)),
            ]
            for (x1, y1), (x2, y2) in arrows:
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2))
            ax.text(0.96, 0.49, "Final\nanswer", ha="left", va="center", fontsize=11)
            fig.tight_layout()
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

        def save_latency_quality_tradeoff(metrics_df: pd.DataFrame, path: Path):
            fig, ax = plt.subplots(figsize=(7.3, 4.7))
            for dataset_name in metrics_df["dataset"].unique():
                sub = metrics_df[metrics_df["dataset"] == dataset_name].copy()
                x = sub["cached_pipeline_latency_ms_median"].fillna(sub["head_latency_ms_median"])
                y = sub["accuracy"]
                ax.plot(x, y, marker="o", label=dataset_name)
                for _, row in sub.iterrows():
                    xx = row["cached_pipeline_latency_ms_median"] if pd.notna(row["cached_pipeline_latency_ms_median"]) else row["head_latency_ms_median"]
                    ax.annotate(row["variant"], (xx, row["accuracy"]), fontsize=8)
            ax.set_xlabel("Cached retrieval + model latency (ms / sample)")
            ax.set_ylabel("Answer accuracy")
            ax.set_title("Latency-quality trade-off")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

        def save_open_closed_plot(metrics_df: pd.DataFrame, path: Path):
            fig, ax = plt.subplots(figsize=(7.3, 4.7))
            width = 0.35
            variants = list(metrics_df["variant"].unique())
            x = np.arange(len(variants))
            closed_vals = [metrics_df[metrics_df["variant"] == v]["closed_accuracy"].mean() for v in variants]
            open_vals = [metrics_df[metrics_df["variant"] == v]["open_accuracy"].mean() for v in variants]
            ax.bar(x - width/2, closed_vals, width=width, label="Closed accuracy")
            ax.bar(x + width/2, open_vals, width=width, label="Open accuracy")
            ax.set_xticks(x)
            ax.set_xticklabels(variants)
            ax.set_ylabel("Accuracy")
            ax.set_title("Closed vs open-answer performance")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

        def save_taxonomy_bar(taxonomy_df: pd.DataFrame, path: Path):
            pivot = taxonomy_df.pivot_table(index="error_tag", columns="variant", values="count", aggfunc="sum", fill_value=0)
            ax = pivot.plot(kind="bar", figsize=(10, 4.7))
            ax.set_ylabel("Count")
            ax.set_title("Error taxonomy")
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        def save_ablation_plot(ablation_df: pd.DataFrame, path: Path):
            fig, ax = plt.subplots(figsize=(10, 4.7))
            for group in ablation_df["ablation_group"].unique():
                sub = ablation_df[ablation_df["ablation_group"] == group].copy()
                sub = sub.sort_values("setting")
                ax.plot(np.arange(len(sub)), sub["accuracy"], marker="o", label=group)
            ax.set_xlabel("Setting index within group")
            ax.set_ylabel("Validation accuracy")
            ax.set_title("Ablation trends")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

        def save_qualitative_sheet(dataset_name: str, qual_subset: pd.DataFrame, path: Path):
            n = max(1, len(qual_subset))
            fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(11.5, 2.1 * n))
            if n == 1:
                axes = [axes]
            for ax, (_, row) in zip(axes, qual_subset.iterrows()):
                ax.axis("off")
                txt = (
                    f"Q: {row.get('question', '')}\n"
                    f"Gold: {row.get('gold_answer', '')} | Pred: {row.get('pred_answer', '')}\n"
                    f"Top support: {row.get('top_support_answer', '')}\n"
                    f"Gate: {row.get('gate_prob', np.nan):.3f} | Open used: {row.get('used_open_branch', False)} | Tag: {row.get('error_tag', '')}"
                )
                ax.text(0.01, 0.95, txt, va="top", ha="left", fontsize=9, family="monospace")
                ax.axhline(0.02, color="black", lw=0.8)
            fig.suptitle(f"Qualitative groundedness examples: {dataset_name}", y=0.995)
            fig.tight_layout()
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

        for ext in ["pdf", "png"]:
            save_rag_workflow_figure(FIG_DIR / f"rag_workflow_v5.{ext}")
            save_latency_quality_tradeoff(metrics_df, FIG_DIR / f"latency_quality_tradeoff_v5.{ext}")
            save_open_closed_plot(metrics_df, FIG_DIR / f"open_closed_accuracy_v5.{ext}")
            save_taxonomy_bar(taxonomy, FIG_DIR / f"error_taxonomy_v5.{ext}")
            save_ablation_plot(ablation_df, FIG_DIR / f"ablation_plot_v5.{ext}")
            for dataset_name in bundles.keys():
                ds_qual = qual_df[qual_df["dataset"] == dataset_name].head(6)
                save_qualitative_sheet(dataset_name, ds_qual, FIG_DIR / f"qualitative_{dataset_name}_v5.{ext}")

        Image.open(FIG_DIR / "rag_workflow_v5.png")


    if not skip_reports:
        # LaTeX table writers, reproducibility manifest, and runtime notes

        def latex_escape(s: str) -> str:
            s = str(s)
            repl = {
                "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
            }
            for k, v in repl.items():
                s = s.replace(k, v)
            return s

        def pretty_variant_name(v: str) -> str:
            return {
                "base": "CompactVLM (no retrieval)",
                "text_rag": "CompactVLM + text retrieval",
                "mm_rag": "CompactVLM + multimodal retrieval",
            }.get(v, v)


        def pretty_dataset_name(d: str) -> str:
            return {
                "imageclef_vqa_med_2019": "ImageCLEF VQA-Med 2019",
                "slake": "SLAKE (English)",
            }.get(d, d)

        def write_main_table(metrics_df: pd.DataFrame, path: Path):
            dataset_order = [d for d in ["imageclef_vqa_med_2019", "slake"] if d in metrics_df["dataset"].unique().tolist()]
            if len(dataset_order) < 2:
                dataset_order = list(metrics_df["dataset"].drop_duplicates().tolist())[:2]
            if len(dataset_order) != 2:
                raise ValueError(f"write_main_table expects two datasets, got {dataset_order}")

            d1, d2 = dataset_order
            c1, c2 = pretty_dataset_name(d1), pretty_dataset_name(d2)

            lines = []
            lines.append(r"\begin{table*}[t]")
            lines.append(r"\centering")
            lines.append(r"\caption{Main comparison on " + latex_escape(c1) + r" and " + latex_escape(c2) + r". GSR: grounding support rate; UAR: unsupported answer rate.}")
            lines.append(r"\label{tab:main_results}")
            lines.append(r"\resizebox{\textwidth}{!}{")
            lines.append(r"\begin{tabular}{lcccccc|cccccc}")
            lines.append(r"\toprule")
            lines.append("& \\multicolumn{6}{c|}{" + latex_escape(c1) + "} & \\multicolumn{6}{c}{" + latex_escape(c2) + "} \\")
            lines.append(r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}")
            lines.append("Method & Acc. & Closed & Open & GSR & UAR & Lat. & Acc. & Closed & Open & GSR & UAR & Lat. \\")
            lines.append(r"\midrule")
            for variant in ["base", "text_rag", "mm_rag"]:
                row_1 = metrics_df[(metrics_df["dataset"] == d1) & (metrics_df["variant"] == variant)]
                row_2 = metrics_df[(metrics_df["dataset"] == d2) & (metrics_df["variant"] == variant)]
                if len(row_1) == 0 or len(row_2) == 0:
                    continue
                a = row_1.iloc[0]
                b = row_2.iloc[0]
                lat_a = a["cached_pipeline_latency_ms_median"] if pd.notna(a["cached_pipeline_latency_ms_median"]) else a["head_latency_ms_median"]
                lat_b = b["cached_pipeline_latency_ms_median"] if pd.notna(b["cached_pipeline_latency_ms_median"]) else b["head_latency_ms_median"]
                lines.append(
                    f"{latex_escape(pretty_variant_name(variant))} & "
                    f"{a['accuracy']:.3f} & {a['closed_accuracy']:.3f} & {a['open_accuracy']:.3f} & {a['grounding_support_rate']:.3f} & {a['unsupported_answer_rate']:.3f} & {lat_a:.1f} & "
                    f"{b['accuracy']:.3f} & {b['closed_accuracy']:.3f} & {b['open_accuracy']:.3f} & {b['grounding_support_rate']:.3f} & {b['unsupported_answer_rate']:.3f} & {lat_b:.1f} \\"
                )
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}}")
            lines.append(r"\end{table*}")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")




    return {
        "selected_configs": selected_configs if 'selected_configs' in locals() else None,
        "main_df": main_df if 'main_df' in locals() else None,
        "ablation_df": ablation_df if 'ablation_df' in locals() else None,
        "robustness_df": robustness_df if 'robustness_df' in locals() else None,
    }


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Compact MedVQA RAG v5 pipeline")
    parser.add_argument("--mode", choices=["full", "continue"], default="full")
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of datasets to run, e.g. slake imageclef_vqa_med_2019")
    parser.add_argument("--fast-subset", action="store_true", help="Enable fast-subset mode for all enabled datasets")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-reports", action="store_true")
    parser.add_argument("--skip-qualitative", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        mode=args.mode,
        datasets=args.datasets,
        fast_subset=args.fast_subset if args.fast_subset else None,
        skip_ablations=args.skip_ablations,
        skip_robustness=args.skip_robustness,
        skip_figures=args.skip_figures,
        skip_reports=args.skip_reports,
        skip_qualitative=args.skip_qualitative,
        skip_warmup=args.skip_warmup,
    )
