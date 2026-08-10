"""LiteMedRAG reference pipeline for MIWAI 2026 paper #157.

The BioMedCLIP backbone is frozen; only compact answer heads are trained.
The final LiteMedRAG-Acc and LiteMedRAG-Ground policies are the validation-selected
settings reported in the accepted manuscript and are frozen for test evaluation.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as e:  # pragma: no cover
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None


VERSION = "final"
UNK = "__unk__"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if torch is not None:
        try:
            torch.set_num_threads(int(os.environ.get("LITEMEDRAG_TORCH_THREADS", "1")))
        except Exception:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_answer(x: Any) -> str:
    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", "-")
    s = " ".join(s.split())
    aliases = {
        "yes.": "yes", "no.": "no", "true": "yes", "false": "no",
        "x ray": "x-ray", "xray": "x-ray", "ct scan": "ct", "mri scan": "mri",
    }
    return aliases.get(s, s)


def safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    if isinstance(x, float) and math.isnan(x):
        return default
    return str(x)


def infer_answer_type(answer: Any) -> str:
    a = normalize_answer(answer)
    if a in {"yes", "no"}:
        return "binary"
    if a.isdigit():
        return "count"
    n = len(a.split())
    if n <= 3:
        return "short-open"
    return "open"


def infer_question_type(question: Any, category: Any = "") -> str:
    c = normalize_answer(category)
    q = normalize_answer(question)
    if c:
        if "modality" in c:
            return "modality"
        if "plane" in c:
            return "plane"
        if "organ" in c or "system" in c or "body" in c:
            return "organ"
        if "abnormal" in c:
            return "abnormality"
    if q.startswith(("is ", "are ", "does ", "do ", "can ", "was ", "were ")):
        return "yes-no"
    if "modality" in q:
        return "modality"
    if "plane" in q or "view" in q:
        return "plane"
    if "organ" in q or "body" in q or "where" in q:
        return "organ"
    if "how many" in q or q.startswith("number"):
        return "count"
    return "other"


def route_preference_flags(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Question-derived routing preferences; no gold answer fields are used."""
    n = len(df)
    if n == 0 or not cfg.get("enabled", True):
        z = np.zeros(n, dtype=bool)
        return z, z

    def colstr(col: str, default: str = "") -> pd.Series:
        s = df[col] if col in df.columns else pd.Series([default] * n, index=df.index)
        return s.fillna(default).astype(str).str.strip().str.lower()

    dataset = colstr("dataset", "").iat[0]
    qtypes = colstr("question_type", "")
    prefer_open_set = {str(x).strip().lower() for x in cfg.get("prefer_open_by_dataset", {}).get(dataset, [])}
    prefer_closed_set = {str(x).strip().lower() for x in cfg.get("prefer_closed_by_dataset", {}).get(dataset, [])}
    return qtypes.isin(prefer_open_set).to_numpy(dtype=bool), qtypes.isin(prefer_closed_set).to_numpy(dtype=bool)


def build_support_text(row: pd.Series) -> str:
    parts = []
    for key in ["dataset", "modality", "body_part", "question_type", "answer_type"]:
        val = safe_str(row.get(key, "")).strip()
        if val:
            parts.append(f"{key}: {val}")
    parts.append(f"question: {safe_str(row.get('question', ''))}")
    parts.append(f"answer: {safe_str(row.get('answer', ''))}")
    return " | ".join(parts)


def l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-8, None)


def topk_similarity_chunked(
    query: np.ndarray,
    bank: np.ndarray,
    k: int,
    chunk: int = 1024,
    exclude_self: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine top-k retrieval with optional diagonal exclusion for train->train retrieval."""
    query = l2norm(np.asarray(query, dtype=np.float32))
    bank = l2norm(np.asarray(bank, dtype=np.float32))
    n = query.shape[0]
    max_k = bank.shape[0] - (1 if exclude_self and n == bank.shape[0] else 0)
    if max_k < 1:
        raise ValueError("Retrieval bank must contain at least two items when self-exclusion is enabled.")
    k = max(1, min(k, max_k))
    all_idx = np.empty((n, k), dtype=np.int64)
    all_sim = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        q = query[start:start + chunk]
        sim = q @ bank.T
        if exclude_self and n == bank.shape[0]:
            local = np.arange(len(q))
            sim[local, start + local] = -np.inf
        part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        part_sim = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_sim, axis=1)
        idx = np.take_along_axis(part, order, axis=1)
        vals = np.take_along_axis(part_sim, order, axis=1)
        all_idx[start:start + len(q)] = idx
        all_sim[start:start + len(q)] = vals.astype(np.float32)
    return all_idx, all_sim


def aggregate_support(bank_feat: np.ndarray, idx: np.ndarray, sim: np.ndarray) -> np.ndarray:
    weights = softmax_np(sim, axis=1).astype(np.float32)
    out = np.empty((idx.shape[0], bank_feat.shape[1]), dtype=np.float32)
    for start in range(0, idx.shape[0], 2048):
        ii = idx[start:start + 2048]
        ww = weights[start:start + 2048]
        out[start:start + len(ii)] = (bank_feat[ii] * ww[..., None]).sum(axis=1)
    return out


def answer_hit(gold: str, support_answers: Sequence[str]) -> bool:
    g = normalize_answer(gold)
    return any(normalize_answer(x) == g for x in support_answers)


def reciprocal_rank(gold: str, support_answers: Sequence[str]) -> float:
    g = normalize_answer(gold)
    for i, a in enumerate(support_answers, 1):
        if normalize_answer(a) == g:
            return 1.0 / i
    return 0.0


def support_match(pred: str, support_answers: Sequence[str]) -> bool:
    """Exact normalized lexical support used for Retrieval Support Rate (RSR)."""
    p = normalize_answer(pred)
    return bool(p) and any(normalize_answer(s) == p for s in support_answers)


DEFAULT_CFG: Dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "backbone_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "embedding_batch_size_image": 12,
    "embedding_batch_size_text": 64,
    "retrieval_chunk": 256,
    "variants": ["base", "text_rag", "mm_rag", "litemedrag_acc", "litemedrag_ground"],
    "datasets": {
        "imageclef_vqa_med_2019": {
            "hf_candidates": ["claudioreeves/imageclef-vqa-med-2019"],
            "image_keys": ["image", "img", "Image"],
            "question_keys": ["question", "Question"],
            "answer_keys": ["answer", "Answer"],
            "category_keys": ["category", "Category", "question_category", "q_category"],
            "enabled": True,
        },
        "slake": {
            "hf_candidates": ["mdwiratathya/SLAKE-vqa-english"],
            "image_keys": ["image", "img", "Image"],
            "question_keys": ["question", "Question"],
            "answer_keys": ["answer", "Answer"],
            "enabled": True,
        },
    },
    "answer_space": {"closed_min_freq": 1, "closed_topn": 220, "add_unk": True},
    "retrieval": {
        "text_topk": 5,
        "mm_topk": 1,
        "mm_image_weight": 0.5,
        "mm_question_weight": 0.5,
        "copy_conf_threshold": 0.35,
        "exclude_self_on_train": True
    },
    "model": {"hidden": 256, "dropout": 0.15},
    "training": {"epochs": 60, "patience": 10, "batch_size": 128, "lr": 1e-3, "weight_decay": 1e-4},
    "question_routing": {
        "enabled": True,
        "prefer_open_by_dataset": {
            "imageclef_vqa_med_2019": ["abnormality"],
            "slake": ["other"]
        },
        "prefer_closed_by_dataset": {
            "imageclef_vqa_med_2019": ["modality", "plane", "organ", "yes-no", "count"],
            "slake": ["yes-no", "count", "modality", "plane", "organ"]
        }
    },
    "paper_policies": {
        "imageclef_vqa_med_2019": {
            "litemedrag_acc": {
                "branch_mode": "mm_rag",
                "candidate_branches": ["mm_rag"],
                "sim_threshold": -1.0,
                "base_conf_threshold": 0.35,
                "require_supported": True,
                "prefer_open_always": True
            },
            "litemedrag_ground": {
                "branch_mode": "best_supported",
                "candidate_branches": ["text_rag", "mm_rag"],
                "sim_threshold": -1.0,
                "base_conf_threshold": 1.01,
                "require_supported": True,
                "prefer_open_always": False
            }
        },
        "slake": {
            "litemedrag_acc": {
                "branch_mode": "mm_rag",
                "candidate_branches": ["mm_rag"],
                "sim_threshold": -1.0,
                "base_conf_threshold": 0.65,
                "require_supported": True,
                "prefer_open_always": False
            },
            "litemedrag_ground": {
                "branch_mode": "mm_rag",
                "candidate_branches": ["mm_rag"],
                "sim_threshold": -1.0,
                "base_conf_threshold": 1.01,
                "require_supported": False,
                "prefer_open_always": False
            }
        }
    }
}


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CFG)
    if config_path:
        user = json.loads(Path(config_path).read_text(encoding="utf-8"))
        cfg = deep_update(cfg, user)
    return cfg


class EncoderBase:
    def encode_images(self, images: Sequence[Any], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def encode_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class RandomEncoder(EncoderBase):
    """Deterministic encoder used only for smoke tests."""
    def __init__(self, dim: int = 512, seed: int = 42):
        self.dim = dim
        self.seed = seed

    def _vec(self, s: str) -> np.ndarray:
        digest = hashlib.sha256(s.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32))
        v = rng.normal(size=self.dim).astype(np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-8)

    def encode_images(self, images: Sequence[Any], batch_size: int) -> np.ndarray:
        return np.stack([self._vec(f"image-{i}-{safe_str(type(img))}") for i, img in enumerate(images)], axis=0)

    def encode_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts], axis=0)


class BiomedCLIPEncoder(EncoderBase):
    def __init__(self, model_id: str, device: str):
        if torch is None:
            raise RuntimeError(f"PyTorch import failed: {_TORCH_IMPORT_ERROR}")
        import open_clip  # lazy import prevents import-time failure during smoke tests
        from PIL import Image
        self.Image = Image
        self.open_clip = open_clip
        self.device = "cuda" if (device in {"auto", "cuda"} and torch.cuda.is_available()) else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id)
        self.tokenizer = open_clip.get_tokenizer(model_id)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_images(self, images: Sequence[Any], batch_size: int) -> np.ndarray:
        feats: List[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch_imgs = []
            for img in images[start:start + batch_size]:
                if not hasattr(img, "convert"):
                    img = self.Image.open(img)
                batch_imgs.append(self.preprocess(img.convert("RGB")))
            x = torch.stack(batch_imgs).to(self.device, non_blocking=False)
            y = self.model.encode_image(x)
            y = F.normalize(y, dim=-1)
            feats.append(y.detach().cpu().numpy().astype(np.float32))
            del x, y, batch_imgs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.concatenate(feats, axis=0)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        feats: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            toks = self.tokenizer(list(texts[start:start + batch_size])).to(self.device)
            y = self.model.encode_text(toks)
            y = F.normalize(y, dim=-1)
            feats.append(y.detach().cpu().numpy().astype(np.float32))
            del toks, y
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.concatenate(feats, axis=0)

    def close(self) -> None:
        try:
            self.model.cpu()
        except Exception:
            pass
        del self.model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_encoder(cfg: Dict[str, Any], encoder_mode: str = "biomedclip") -> EncoderBase:
    if encoder_mode == "random":
        return RandomEncoder(dim=512, seed=int(cfg["seed"]))
    return BiomedCLIPEncoder(cfg["backbone_id"], cfg.get("device", "cuda"))


# ---------------------------- dataset loading -----------------------------

def _find_col(columns: Sequence[str], candidates: Sequence[str]) -> str:
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"None of {candidates} found in columns {list(columns)}")


def _load_hf_dataset(name_candidates: Sequence[str]):
    from datasets import load_dataset  # lazy import
    last = None
    for name in name_candidates:
        try:
            ds = load_dataset(name)
            print(f"Loaded dataset from: {name}", flush=True)
            return ds
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not load dataset candidates {name_candidates}. Last error: {last}")


def _split_name_map(ds) -> Dict[str, str]:
    keys = set(ds.keys())
    if "train" not in keys or "test" not in keys:
        raise RuntimeError(f"Dataset must provide train and test splits; found {sorted(keys)}")
    if "validation" in keys:
        val = "validation"
    elif "val" in keys:
        val = "val"
    else:
        raise RuntimeError(f"Dataset must provide a validation split; found {sorted(keys)}")
    return {"train": "train", "val": val, "test": "test"}

def build_metadata_from_hf_split(split, dataset_name: str, cfg: Dict[str, Any], split_name: str, limit: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    cols = list(split.column_names)
    dcfg = cfg["datasets"][dataset_name]
    img_col = _find_col(cols, dcfg.get("image_keys", ["image"]))
    q_col = _find_col(cols, dcfg.get("question_keys", ["question"]))
    a_col = _find_col(cols, dcfg.get("answer_keys", ["answer"]))
    cat_col = None
    if dataset_name == "imageclef_vqa_med_2019":
        for cand in dcfg.get("category_keys", []):
            if cand in cols:
                cat_col = cand
                break
    n = len(split) if limit is None else min(len(split), int(limit))
    rows = []
    for i in range(n):
        row = split[i]
        q = safe_str(row.get(q_col, ""))
        a = normalize_answer(row.get(a_col, ""))
        category = safe_str(row.get(cat_col, "")) if cat_col else ""
        qtype = infer_question_type(q, category)
        rows.append({
            "sample_id": f"{dataset_name}_{split_name}_{i}",
            "dataset": dataset_name,
            "split": split_name,
            "question": q,
            "answer": a,
            "category": category,
            "question_type": qtype,
            "answer_type": infer_answer_type(a),
            "modality": a if qtype == "modality" else "",
            "body_part": a if qtype == "organ" else "",
        })
    return pd.DataFrame(rows), img_col


def load_dataset_metadata(dataset_name: str, cfg: Dict[str, Any], split_limit: Optional[int] = None, synthetic: bool = False) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame], Dict[str, str]]:
    if synthetic:
        return make_synthetic_dataset(dataset_name, split_limit or 128)
    dcfg = cfg["datasets"][dataset_name]
    ds = _load_hf_dataset(dcfg["hf_candidates"])
    smap = _split_name_map(ds)
    split_objs: Dict[str, Any] = {}
    dfs: Dict[str, pd.DataFrame] = {}
    img_cols: Dict[str, str] = {}
    for canonical in ["train", "val", "test"]:
        raw_name = smap[canonical]
        df, img_col = build_metadata_from_hf_split(ds[raw_name], dataset_name, cfg, canonical, split_limit)
        split_objs[canonical] = ds[raw_name]
        dfs[canonical] = df
        img_cols[canonical] = img_col
    print(f"\nDATASET: {dataset_name}", flush=True)
    for s, df in dfs.items():
        print(s, df.shape, df["answer_type"].value_counts().to_dict(), flush=True)
    return split_objs, dfs, img_cols


def make_synthetic_dataset(dataset_name: str, n: int) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame], Dict[str, str]]:
    from PIL import Image
    def make_df(split: str, m: int) -> pd.DataFrame:
        qs = ["is there abnormality", "what modality is shown", "what organ is shown", "what is the finding"]
        ans = ["yes", "ct", "lung", "opacity"]
        rows = []
        for i in range(m):
            q = qs[i % len(qs)]
            a = ans[i % len(ans)]
            rows.append({
                "sample_id": f"{dataset_name}_{split}_{i}", "dataset": dataset_name, "split": split,
                "question": q, "answer": a, "category": "", "question_type": infer_question_type(q),
                "answer_type": infer_answer_type(a), "modality": a if "modality" in q else "", "body_part": a if "organ" in q else "",
            })
        return pd.DataFrame(rows)
    class Split:
        column_names = ["image"]
        def __init__(self, m): self.m = m
        def __len__(self): return self.m
        def __getitem__(self, i):
            return {"image": Image.new("RGB", (224, 224), color=(i % 255, 0, 0))}
    sizes = {"train": n, "val": max(16, n // 4), "test": max(16, n // 4)}
    split_objs = {k: Split(v) for k, v in sizes.items()}
    dfs = {k: make_df(k, v) for k, v in sizes.items()}
    return split_objs, dfs, {k: "image" for k in sizes}


def extract_images_from_split(split_obj: Any, img_col: str, indices: Sequence[int]) -> List[Any]:
    imgs = []
    for i in indices:
        row = split_obj[int(i)]
        imgs.append(row[img_col])
    return imgs


# ---------------------------- feature caching -----------------------------

def dataset_cache_dir(root: Path, dataset_name: str) -> Path:
    return ensure_dir(root / "cache" / dataset_name)


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


def extract_or_load_embeddings(dataset_name: str, split_objs: Dict[str, Any], dfs: Dict[str, pd.DataFrame], img_cols: Dict[str, str], cfg: Dict[str, Any], out_root: Path, encoder_mode: str, force: bool = False) -> Dict[str, Dict[str, Path]]:
    ddir = dataset_cache_dir(out_root, dataset_name)
    paths: Dict[str, Dict[str, Path]] = {}
    need_encoder = False
    for split, df in dfs.items():
        paths[split] = {
            "meta": ddir / f"{split}_meta.pkl",
            "image": ddir / f"{split}_image.npy",
            "question": ddir / f"{split}_question.npy",
            "support_text": ddir / f"{split}_support_text.npy",
        }
        if force or not all(p.exists() for p in paths[split].values()):
            need_encoder = True
    encoder: Optional[EncoderBase] = make_encoder(cfg, encoder_mode) if need_encoder else None
    for split, df in dfs.items():
        save_df(df.drop(columns=[c for c in ["image"] if c in df.columns], errors="ignore"), paths[split]["meta"])
        if not force and paths[split]["image"].exists() and paths[split]["question"].exists() and paths[split]["support_text"].exists():
            print(f"{dataset_name} {split}: embeddings cached", flush=True)
            continue
        assert encoder is not None
        print(f"{dataset_name} {split}: encoding {len(df)} samples", flush=True)
        img_feats = []
        bs = int(cfg["embedding_batch_size_image"])
        for start in range(0, len(df), bs):
            idx = range(start, min(start + bs, len(df)))
            imgs = extract_images_from_split(split_objs[split], img_cols[split], idx)
            img_feats.append(encoder.encode_images(imgs, batch_size=bs))
            del imgs
            gc.collect()
        image_emb = np.concatenate(img_feats, axis=0).astype(np.float32)
        q_emb = encoder.encode_texts(df["question"].fillna("").astype(str).tolist(), int(cfg["embedding_batch_size_text"])).astype(np.float32)
        s_texts = [build_support_text(row) for _, row in df.iterrows()]
        s_emb = encoder.encode_texts(s_texts, int(cfg["embedding_batch_size_text"])).astype(np.float32)
        np.save(paths[split]["image"], image_emb)
        np.save(paths[split]["question"], q_emb)
        np.save(paths[split]["support_text"], s_emb)
        print(f"{dataset_name} {split}: saved image {image_emb.shape}, question {q_emb.shape}, support {s_emb.shape}", flush=True)
        del image_emb, q_emb, s_emb, img_feats, s_texts
        gc.collect()
    if encoder is not None:
        encoder.close()
    return paths


def load_cached_split(paths: Dict[str, Path], mmap: bool = True) -> Dict[str, Any]:
    mode = "r" if mmap else None
    return {
        "df": load_df(paths["meta"]),
        "image_emb": np.load(paths["image"], mmap_mode=mode),
        "question_emb": np.load(paths["question"], mmap_mode=mode),
        "support_text_emb": np.load(paths["support_text"], mmap_mode=mode),
    }


def build_answer_space(train_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[int, str]]:
    freq = train_df["answer"].map(normalize_answer).value_counts()
    freq = freq[freq >= int(cfg.get("closed_min_freq", 1))]
    answers = freq.head(int(cfg.get("closed_topn", 220))).index.tolist()
    if cfg.get("add_unk", True) and UNK not in answers:
        answers.append(UNK)
    ans2id = {a: i for i, a in enumerate(answers)}
    id2ans = {i: a for a, i in ans2id.items()}
    return ans2id, id2ans


def encode_closed_y(df: pd.DataFrame, ans2id: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    unk = ans2id.get(UNK, len(ans2id) - 1)
    ys, oov = [], []
    for a in df["answer"].tolist():
        aa = normalize_answer(a)
        ys.append(ans2id.get(aa, unk))
        oov.append(aa not in ans2id or aa == UNK)
    return np.asarray(ys, dtype=np.int64), np.asarray(oov, dtype=bool)


def build_retrieval_for_dataset(dataset_name: str, cache_paths: Dict[str, Dict[str, Path]], cfg: Dict[str, Any], out_root: Path, force: bool = False) -> Dict[str, Dict[str, Path]]:
    ddir = dataset_cache_dir(out_root, dataset_name)
    retr_paths: Dict[str, Dict[str, Path]] = {}
    train = load_cached_split(cache_paths["train"], mmap=True)
    train_q = np.asarray(train["question_emb"], dtype=np.float32)
    train_img = np.asarray(train["image_emb"], dtype=np.float32)
    train_supp = np.asarray(train["support_text_emb"], dtype=np.float32)
    topk_text = int(cfg["retrieval"]["text_topk"])
    topk_mm = int(cfg["retrieval"]["mm_topk"])
    w_img = float(cfg["retrieval"]["mm_image_weight"])
    w_q = float(cfg["retrieval"]["mm_question_weight"])
    for split, paths in cache_paths.items():
        out = {
            "text_idx": ddir / f"{split}_text_idx.npy",
            "text_sim": ddir / f"{split}_text_sim.npy",
            "mm_idx": ddir / f"{split}_mm_idx.npy",
            "mm_sim": ddir / f"{split}_mm_sim.npy",
            "text_support": ddir / f"{split}_text_support.npy",
            "mm_text_support": ddir / f"{split}_mm_text_support.npy",
            "mm_image_support": ddir / f"{split}_mm_image_support.npy",
        }
        retr_paths[split] = out
        if not force and all(p.exists() for p in out.values()):
            print(f"{dataset_name} {split}: retrieval cached", flush=True)
            continue
        split_pack = load_cached_split(paths, mmap=True)
        q = np.asarray(split_pack["question_emb"], dtype=np.float32)
        img = np.asarray(split_pack["image_emb"], dtype=np.float32)
        text_idx, text_sim = topk_similarity_chunked(q, train_q, topk_text, int(cfg["retrieval_chunk"]), exclude_self=(split == "train" and bool(cfg["retrieval"].get("exclude_self_on_train", True))))
        # mixed multimodal similarity by concatenating weighted normalised vectors
        mm_query = np.concatenate([w_q * l2norm(q), w_img * l2norm(img)], axis=1)
        mm_bank = np.concatenate([w_q * l2norm(train_q), w_img * l2norm(train_img)], axis=1)
        mm_idx, mm_sim = topk_similarity_chunked(mm_query, mm_bank, topk_mm, int(cfg["retrieval_chunk"]), exclude_self=(split == "train" and bool(cfg["retrieval"].get("exclude_self_on_train", True))))
        text_support = aggregate_support(train_supp, text_idx, text_sim)
        mm_text_support = aggregate_support(train_supp, mm_idx, mm_sim)
        mm_image_support = aggregate_support(train_img, mm_idx, mm_sim)
        for key, arr in [("text_idx", text_idx), ("text_sim", text_sim), ("mm_idx", mm_idx), ("mm_sim", mm_sim), ("text_support", text_support), ("mm_text_support", mm_text_support), ("mm_image_support", mm_image_support)]:
            np.save(out[key], arr.astype(np.int64 if key.endswith("idx") else np.float32))
        print(f"{dataset_name} {split}: built retrieval text {text_idx.shape}, mm {mm_idx.shape}", flush=True)
        del split_pack, q, img, text_idx, text_sim, mm_query, mm_bank, mm_idx, mm_sim, text_support, mm_text_support, mm_image_support
        gc.collect()
    del train, train_q, train_img, train_supp
    gc.collect()
    return retr_paths


# ---------------------------- modelling ----------------------------------

class FeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
    def __len__(self) -> int:
        return len(self.y)
    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, n_cls: int, hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_cls),
        )
    def forward(self, x):
        return self.net(x)


def feature_matrix(split_pack: Dict[str, Any], retr_paths: Dict[str, Path], variant: str) -> np.ndarray:
    img = np.asarray(split_pack["image_emb"], dtype=np.float32)
    q = np.asarray(split_pack["question_emb"], dtype=np.float32)
    if variant == "base":
        return np.concatenate([img, q], axis=1).astype(np.float32)
    if variant == "text_rag":
        txt = np.load(retr_paths["text_support"], mmap_mode="r")
        return np.concatenate([img, q, np.asarray(txt, dtype=np.float32)], axis=1).astype(np.float32)
    if variant == "mm_rag":
        mt = np.load(retr_paths["mm_text_support"], mmap_mode="r")
        mi = np.load(retr_paths["mm_image_support"], mmap_mode="r")
        return np.concatenate([img, q, np.asarray(mt, dtype=np.float32), np.asarray(mi, dtype=np.float32)], axis=1).astype(np.float32)
    raise ValueError(variant)


def train_classifier(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, n_cls: int, cfg: Dict[str, Any], device: str) -> Tuple[nn.Module, Dict[str, Any]]:
    if torch is None:
        raise RuntimeError(f"PyTorch unavailable: {_TORCH_IMPORT_ERROR}")
    device = "cuda" if device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    model = MLPClassifier(Xtr.shape[1], int(n_cls), int(cfg["model"]["hidden"]), float(cfg["model"]["dropout"])).to(device)
    train_cfg = cfg["training"]
    loader = DataLoader(FeatureDataset(Xtr, ytr), batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=0, pin_memory=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    best_state = None
    best_acc = -1.0
    best_epoch = 0
    stale = 0
    hist = []
    xv = torch.from_numpy(np.asarray(Xva, dtype=np.float32)).to(device)
    yv = np.asarray(yva, dtype=np.int64)
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred = model(xv).argmax(1).detach().cpu().numpy()
        acc = float((pred == yv).mean()) if len(yv) else 0.0
        hist.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_acc": acc})
        print(f"epoch={epoch:03d} loss={np.mean(losses):.4f} val_acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(train_cfg["patience"]):
            print("Early stopping.", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_accuracy": best_acc, "best_epoch": best_epoch, "history": hist}


def predict_classifier(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    device = "cuda" if device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    preds, confs = [], []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.asarray(X[start:start + batch_size], dtype=np.float32)).to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)
            conf, pred = prob.max(1)
            preds.append(pred.cpu().numpy())
            confs.append(conf.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


def support_answers_from_idx(train_answers: Sequence[str], idx: np.ndarray) -> List[List[str]]:
    return [[train_answers[int(j)] for j in row] for row in idx]



def _support_indices_for_variant(variant: str, n: int, retr_paths: Dict[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Return retrieval indices and similarities for a branch."""
    if variant == "base":
        # Base does not consume retrieval features; text-neighbour answers are used only
        # by the common lexical support audit reported for every operating point.
        return np.load(retr_paths["text_idx"], mmap_mode="r"), np.load(retr_paths["text_sim"], mmap_mode="r")
    if variant == "text_rag":
        return np.load(retr_paths["text_idx"], mmap_mode="r"), np.load(retr_paths["text_sim"], mmap_mode="r")
    if variant == "mm_rag":
        return np.load(retr_paths["mm_idx"], mmap_mode="r"), np.load(retr_paths["mm_sim"], mmap_mode="r")
    raise ValueError(f"Unsupported retrieval branch: {variant}")


def branch_prediction_frame(
    dataset_name: str,
    variant: str,
    split_name: str,
    model: nn.Module,
    cache_paths: Dict[str, Dict[str, Path]],
    retr_paths: Dict[str, Dict[str, Path]],
    ans2id: Dict[str, int],
    id2ans: Dict[int, str],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Predict one branch on one split and return inference-time fields.

    The branch prediction is intentionally saved as a compact DataFrame rather than
    keeping all branch outputs in memory. This keeps policy application lightweight.
    """
    train_df = load_df(cache_paths["train"]["meta"])
    split_pack = load_cached_split(cache_paths[split_name], mmap=True)
    df = split_pack["df"].copy().reset_index(drop=True)
    X = feature_matrix(split_pack, retr_paths[split_name], variant)
    _, oov = encode_closed_y(df, ans2id)
    pred_ids, conf = predict_classifier(model, X, cfg.get("device", "cuda"), int(cfg["training"]["batch_size"]) * 4)
    cls_answers = [normalize_answer(id2ans.get(int(i), UNK)) for i in pred_ids]

    supp_idx, supp_sim = _support_indices_for_variant(variant, len(df), retr_paths[split_name])
    supp_idx_arr = np.asarray(supp_idx)
    supp_sim_arr = np.asarray(supp_sim, dtype=np.float32)
    support_answers = support_answers_from_idx(train_df["answer"].tolist(), supp_idx_arr)
    top_sim = supp_sim_arr[:, 0] if supp_sim_arr.ndim == 2 and supp_sim_arr.shape[1] else np.zeros(len(df), dtype=np.float32)

    prefer_open, prefer_closed = route_preference_flags(df, cfg.get("question_routing", {}))
    final_answers: List[str] = []
    used_copy: List[bool] = []
    for pa, cands, po, pc, cf in zip(cls_answers, support_answers, prefer_open, prefer_closed, conf):
        if variant != "base" and po and len(cands) > 0:
            final_answers.append(normalize_answer(cands[0]))
            used_copy.append(True)
        elif variant != "base" and (not pc) and cf < float(cfg["retrieval"].get("copy_conf_threshold", 0.35)) and len(cands) > 0:
            final_answers.append(normalize_answer(cands[0]))
            used_copy.append(True)
        else:
            final_answers.append(normalize_answer(pa))
            used_copy.append(False)

    gold = df["answer"].map(normalize_answer).tolist()
    supported = np.asarray([support_match(p, sa) for p, sa in zip(final_answers, support_answers)], dtype=bool)
    hit = np.asarray([answer_hit(g, sa) for g, sa in zip(gold, support_answers)], dtype=bool)
    rr = np.asarray([reciprocal_rank(g, sa) for g, sa in zip(gold, support_answers)], dtype=float)

    out = df[[c for c in df.columns if c not in {"image"}]].copy()
    out["branch_variant"] = variant
    out["gold_answer"] = gold
    out["pred_answer"] = final_answers
    out["classifier_answer"] = cls_answers
    out["classifier_confidence"] = np.asarray(conf, dtype=np.float32)
    out["top_retrieval_sim"] = np.asarray(top_sim, dtype=np.float32)
    out["supported"] = supported
    out["retrieval_hit"] = hit
    out["retrieval_rr"] = rr
    out["used_copy"] = np.asarray(used_copy, dtype=bool)
    out["answer_oov"] = np.asarray(oov, dtype=bool)
    out["prefer_open"] = prefer_open.astype(bool)
    out["prefer_closed"] = prefer_closed.astype(bool)
    out["correct"] = out["pred_answer"].map(normalize_answer).to_numpy() == np.asarray(gold)
    out["support_answers"] = [" || ".join(map(normalize_answer, sa[:5])) for sa in support_answers]

    del split_pack, X, supp_idx, supp_sim, supp_idx_arr, supp_sim_arr, support_answers
    gc.collect()
    return out


def metrics_from_prediction_log(dataset_name: str, variant: str, pred_log: pd.DataFrame) -> Dict[str, Any]:
    correct = pred_log["correct"].astype(bool).to_numpy()
    oov = pred_log.get("answer_oov", pd.Series([False] * len(pred_log))).astype(bool).to_numpy()
    prefer_open = pred_log.get("prefer_open", pd.Series([False] * len(pred_log))).astype(bool).to_numpy()
    closed_mask = ~oov
    open_mask = oov | prefer_open
    supported = pred_log["supported"].astype(bool).to_numpy()
    hit = pred_log.get("retrieval_hit", pd.Series([False] * len(pred_log))).astype(bool).to_numpy()
    rr = pred_log.get("retrieval_rr", pd.Series([0.0] * len(pred_log))).astype(float).to_numpy()
    used_copy = pred_log.get("used_copy", pd.Series([False] * len(pred_log))).astype(bool).to_numpy()
    return {
        "dataset": dataset_name,
        "variant": variant,
        "accuracy": float(correct.mean()) if len(correct) else np.nan,
        "closed_accuracy": float(correct[closed_mask].mean()) if closed_mask.any() else np.nan,
        "open_accuracy": float(correct[open_mask].mean()) if open_mask.any() else np.nan,
        "retrieval_support_rate": float(supported.mean()) if len(supported) else np.nan,
        "unsupported_answer_rate": float((~supported).mean()) if len(supported) else np.nan,
        "retrieval_hit_rate": float(hit.mean()) if len(hit) else np.nan,
        "retrieval_mrr": float(rr.mean()) if len(rr) else np.nan,
        "answer_oov_rate": float(oov.mean()) if len(oov) else np.nan,
        "copy_usage_rate": float(used_copy.mean()) if len(used_copy) else np.nan,
    }


def evaluate_variant(dataset_name: str, variant: str, model: nn.Module, cache_paths: Dict[str, Dict[str, Path]], retr_paths: Dict[str, Dict[str, Path]], ans2id: Dict[str, int], id2ans: Dict[int, str], cfg: Dict[str, Any], out_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    pred_log = branch_prediction_frame(dataset_name, variant, "test", model, cache_paths, retr_paths, ans2id, id2ans, cfg)
    metrics = metrics_from_prediction_log(dataset_name, variant, pred_log)
    return pred_log, metrics


def _apply_select_policy(base_df: pd.DataFrame, branch_dfs: Dict[str, pd.DataFrame], policy: Dict[str, Any], dataset_name: str, split_name: str, variant_name: str) -> pd.DataFrame:
    mode = policy["branch_mode"]
    sim_thr = float(policy["sim_threshold"])
    base_conf_thr = float(policy["base_conf_threshold"])
    require_supported = bool(policy["require_supported"])
    prefer_open_always = bool(policy["prefer_open_always"])
    candidates = list(policy.get("candidate_branches", branch_dfs.keys()))

    rows = []
    for i in range(len(base_df)):
        base = base_df.iloc[i]
        chosen = base
        chosen_branch = "base"
        eligible = []
        for branch in candidates:
            if branch not in branch_dfs:
                continue
            cand = branch_dfs[branch].iloc[i]
            base_conf = float(base.get("classifier_confidence", 0.0))
            top_sim = float(cand.get("top_retrieval_sim", 0.0))
            is_supported = bool(cand.get("supported", False))
            prefer_open = bool(base.get("prefer_open", False))
            ok = top_sim >= sim_thr
            if require_supported:
                ok = ok and is_supported
            ok = ok and ((base_conf <= base_conf_thr) or (prefer_open_always and prefer_open))
            if ok:
                eligible.append(cand)

        if eligible:
            if mode in branch_dfs:
                for cand in eligible:
                    if str(cand.get("branch_variant")) == mode:
                        chosen = cand
                        chosen_branch = mode
                        break
            elif mode == "best_supported":
                chosen = max(eligible, key=lambda r: (bool(r.get("supported", False)), float(r.get("top_retrieval_sim", 0.0))))
                chosen_branch = str(chosen.get("branch_variant"))
            elif mode == "best_sim":
                chosen = max(eligible, key=lambda r: float(r.get("top_retrieval_sim", 0.0)))
                chosen_branch = str(chosen.get("branch_variant"))

        out = base.to_dict()
        for col in ["pred_answer", "classifier_answer", "classifier_confidence", "top_retrieval_sim", "supported", "retrieval_hit", "retrieval_rr", "used_copy", "support_answers"]:
            out[col] = chosen.get(col, out.get(col))
        out["selected_branch"] = chosen_branch
        out["branch_variant"] = variant_name
        out["correct"] = normalize_answer(out.get("pred_answer", "")) == normalize_answer(out.get("gold_answer", out.get("answer", "")))
        rows.append(out)
    selected = pd.DataFrame(rows)
    selected["dataset"] = dataset_name
    selected["split"] = split_name
    return selected


def evaluate_paper_policy(
    dataset_name: str,
    variant_name: str,
    branch_models: Dict[str, nn.Module],
    cache_paths: Dict[str, Dict[str, Path]],
    retr_paths: Dict[str, Dict[str, Path]],
    ans2id: Dict[str, int],
    id2ans: Dict[int, str],
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    policy = dict(cfg["paper_policies"][dataset_name][variant_name])
    candidate_branches = [b for b in policy["candidate_branches"] if b in branch_models]
    base_df = branch_prediction_frame(dataset_name, "base", "test", branch_models["base"], cache_paths, retr_paths, ans2id, id2ans, cfg)
    branch_dfs = {b: branch_prediction_frame(dataset_name, b, "test", branch_models[b], cache_paths, retr_paths, ans2id, id2ans, cfg) for b in candidate_branches}
    pred_log = _apply_select_policy(base_df, branch_dfs, policy, dataset_name, "test", variant_name)
    metrics = metrics_from_prediction_log(dataset_name, variant_name, pred_log)
    metrics["selected_rag_rate"] = float((pred_log["selected_branch"] != "base").mean())
    metrics["policy"] = json.dumps(policy, sort_keys=True)
    return pred_log, metrics, policy

def model_size_mb(model: nn.Module) -> float:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / (1024 ** 2)



def _checkpoint_path(out_root: Path, dataset_name: str, variant: str) -> Path:
    return ensure_dir(out_root / "checkpoints") / f"{dataset_name}_{variant}.pt"


def _load_model_for_variant(dataset_name: str, variant: str, cache_paths: Dict[str, Dict[str, Path]], retr_paths: Dict[str, Dict[str, Path]], ans2id: Dict[str, int], cfg: Dict[str, Any], out_root: Path) -> nn.Module:
    try:
        ckpt = torch.load(_checkpoint_path(out_root, dataset_name, variant), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(_checkpoint_path(out_root, dataset_name, variant), map_location="cpu")
    val_pack = load_cached_split(cache_paths["val"], mmap=True)
    Xva = feature_matrix(val_pack, retr_paths["val"], variant)
    state = ckpt["state_dict"]
    # The training head may omit an unused UNK class if no training item maps to it.
    # Reconstruct the classifier with the checkpoint's actual output dimension.
    out_dim = None
    for key, val in state.items():
        if key.endswith("bias") and getattr(val, "ndim", 0) == 1:
            out_dim = int(val.shape[0])
    if out_dim is None:
        out_dim = len(ans2id)
    model = MLPClassifier(Xva.shape[1], out_dim, int(cfg["model"]["hidden"]), float(cfg["model"]["dropout"]))
    model.load_state_dict(state)
    del val_pack, Xva, ckpt, state
    gc.collect()
    return model


def train_and_evaluate_dataset(dataset_name: str, cache_paths: Dict[str, Dict[str, Path]], retr_paths: Dict[str, Dict[str, Path]], cfg: Dict[str, Any], out_root: Path, variants: Sequence[str]) -> pd.DataFrame:
    csv_dir = ensure_dir(out_root / "csv")
    ckpt_dir = ensure_dir(out_root / "checkpoints")
    train_df = load_df(cache_paths["train"]["meta"])
    ans2id, id2ans = build_answer_space(train_df, cfg["answer_space"])
    requested = list(dict.fromkeys(variants))
    selective = [v for v in requested if v in {"litemedrag_acc", "litemedrag_ground"}]
    train_variants = [v for v in requested if v in {"base", "text_rag", "mm_rag"}]
    for sel in selective:
        policy = cfg["paper_policies"][dataset_name][sel]
        for prereq in ["base"] + list(policy["candidate_branches"]):
            if prereq not in train_variants:
                train_variants.append(prereq)

    rows: List[Dict[str, Any]] = []
    train_seconds_by_variant: Dict[str, float] = {}
    for variant in train_variants:
        print(f"\n===== {dataset_name} | {variant} =====", flush=True)
        set_seed(int(cfg["seed"]))
        train_pack = load_cached_split(cache_paths["train"], mmap=True)
        val_pack = load_cached_split(cache_paths["val"], mmap=True)
        Xtr = feature_matrix(train_pack, retr_paths["train"], variant)
        Xva = feature_matrix(val_pack, retr_paths["val"], variant)
        ytr, _ = encode_closed_y(train_pack["df"], ans2id)
        yva, _ = encode_closed_y(val_pack["df"], ans2id)
        t0 = time.perf_counter()
        model, hist = train_classifier(Xtr, ytr, Xva, yva, len(ans2id), cfg, cfg.get("device", "auto"))
        train_seconds = time.perf_counter() - t0
        train_seconds_by_variant[variant] = float(train_seconds)
        pred_log, metrics = evaluate_variant(dataset_name, variant, model, cache_paths, retr_paths, ans2id, id2ans, cfg, out_root)
        metrics.update({
            "train_seconds": float(train_seconds),
            "model_size_mb": model_size_mb(model),
            "parameter_count_m": sum(p.numel() for p in model.parameters()) / 1e6,
            "best_val_accuracy": hist["best_val_accuracy"],
            "best_epoch": hist["best_epoch"],
            "selected_rag_rate": 0.0 if variant == "base" else 1.0,
            "policy": "",
        })
        if variant in requested:
            rows.append(metrics)
        pred_log.to_csv(csv_dir / f"predictions_{dataset_name}_{variant}.csv", index=False)
        torch.save({"state_dict": model.state_dict(), "ans2id": ans2id, "id2ans": id2ans, "metrics": metrics, "history": hist}, ckpt_dir / f"{dataset_name}_{variant}.pt")
        del train_pack, val_pack, Xtr, Xva, ytr, yva, model, pred_log
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if selective:
        branch_models: Dict[str, nn.Module] = {}
        try:
            needed = {"base"}
            for sel in selective:
                needed.update(cfg["paper_policies"][dataset_name][sel]["candidate_branches"])
            for branch in sorted(needed):
                branch_models[branch] = _load_model_for_variant(dataset_name, branch, cache_paths, retr_paths, ans2id, cfg, out_root)

            branch_sizes = {v: model_size_mb(m) for v, m in branch_models.items()}
            branch_params = {v: sum(p.numel() for p in m.parameters()) / 1e6 for v, m in branch_models.items()}
            for sel in selective:
                print(f"\n===== {dataset_name} | {sel} =====", flush=True)
                pred_log, metrics, policy = evaluate_paper_policy(dataset_name, sel, branch_models, cache_paths, retr_paths, ans2id, id2ans, cfg)
                used = {"base", *policy["candidate_branches"]}
                metrics.update({
                    "train_seconds": float(sum(train_seconds_by_variant.get(v, 0.0) for v in used)),
                    "model_size_mb": float(sum(branch_sizes[v] for v in used)),
                    "parameter_count_m": float(sum(branch_params[v] for v in used)),
                    "best_val_accuracy": np.nan,
                    "best_epoch": np.nan,
                })
                rows.append(metrics)
                pred_log.to_csv(csv_dir / f"predictions_{dataset_name}_{sel}.csv", index=False)
                (csv_dir / f"policy_{dataset_name}_{sel}.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
        finally:
            for m in branch_models.values():
                try: m.cpu()
                except Exception: pass
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.DataFrame(rows)

def write_reports(main_df: pd.DataFrame, out_root: Path, make_figures: bool = True) -> None:
    csv_dir = ensure_dir(out_root / "csv")
    table_dir = ensure_dir(out_root / "tables")
    fig_dir = ensure_dir(out_root / "figures")
    main_path = csv_dir / "main_metrics.csv"
    if main_path.exists():
        old = pd.read_csv(main_path)
        if {"dataset", "variant"}.issubset(old.columns):
            keys = set(zip(main_df["dataset"].astype(str), main_df["variant"].astype(str)))
            keep = [tuple(x) not in keys for x in zip(old["dataset"].astype(str), old["variant"].astype(str))]
            main_df = pd.concat([old.loc[keep], main_df], ignore_index=True)

    order = {"base": 0, "text_rag": 1, "mm_rag": 2, "litemedrag_acc": 3, "litemedrag_ground": 4}
    main_df["_order"] = main_df["variant"].map(order).fillna(99)
    main_df = main_df.sort_values(["dataset", "_order"]).drop(columns="_order").reset_index(drop=True)
    main_df.to_csv(main_path, index=False)

    deploy_cols = [c for c in ["dataset", "variant", "accuracy", "model_size_mb", "parameter_count_m", "train_seconds", "retrieval_support_rate", "unsupported_answer_rate", "selected_rag_rate"] if c in main_df.columns]
    main_df[deploy_cols].to_csv(csv_dir / "deployment_metrics.csv", index=False)

    if make_figures:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6.2, 4.2))
            for _, r in main_df.iterrows():
                x = 100.0 * float(r["retrieval_support_rate"])
                y = 100.0 * float(r["accuracy"])
                size = 45.0 + 55.0 * float(r["parameter_count_m"])
                plt.scatter(x, y, s=size)
                plt.text(x, y, str(r["variant"]), fontsize=7)
            plt.xlabel("Retrieval support rate (%)")
            plt.ylabel("Answer accuracy (%)")
            plt.tight_layout()
            plt.savefig(fig_dir / "accuracy_support.pdf")
            plt.close()
        except Exception as e:
            print(f"Figure generation skipped: {e}", flush=True)

    display = {
        "base": "Base", "text_rag": "Text-RAG", "mm_rag": "MM-RAG",
        "litemedrag_acc": "LiteMedRAG-Acc", "litemedrag_ground": "LiteMedRAG-Ground",
    }
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{LiteMedRAG internal compact MedVQA results.}",
        r"\label{tab:litemedrag_main}",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Dataset & Variant & Acc. & Closed & Open & RSR & UAR & Sel. RAG \\",
        r"\midrule",
    ]
    for _, r in main_df.iterrows():
        lines.append(
            f"{r['dataset']} & {display.get(r['variant'], r['variant'])} & "
            f"{100*r['accuracy']:.2f} & {100*r['closed_accuracy']:.2f} & {100*r['open_accuracy']:.2f} & "
            f"{100*r['retrieval_support_rate']:.2f} & {100*r['unsupported_answer_rate']:.2f} & {100*r['selected_rag_rate']:.2f} \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (table_dir / "main_results.tex").write_text("\n".join(lines), encoding="utf-8")

def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if getattr(args, "max_epochs", None) is not None:
        cfg["training"]["epochs"] = int(args.max_epochs)
    if getattr(args, "patience", None) is not None:
        cfg["training"]["patience"] = int(args.patience)
    set_seed(int(cfg["seed"]))
    out_root = ensure_dir(Path(args.artifact_root))
    ensure_dir(out_root / "csv")
    (out_root / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if args.variants:
        variants = args.variants
    else:
        variants = cfg.get("variants", ["base", "text_rag", "mm_rag", "litemedrag_acc", "litemedrag_ground"])
    all_rows = []
    for dataset_name in args.datasets:
        print(f"\n######## PROCESSING DATASET: {dataset_name} ########", flush=True)
        split_objs, dfs, img_cols = load_dataset_metadata(dataset_name, cfg, split_limit=args.limit_per_split, synthetic=args.synthetic)
        cache_paths = extract_or_load_embeddings(dataset_name, split_objs, dfs, img_cols, cfg, out_root, args.encoder, force=args.force_embeddings)
        # Free HF dataset image references before retrieval/training where possible.
        del split_objs, dfs, img_cols
        gc.collect()
        retr_paths = build_retrieval_for_dataset(dataset_name, cache_paths, cfg, out_root, force=args.force_retrieval)
        if args.skip_train:
            continue
        df_metrics = train_and_evaluate_dataset(dataset_name, cache_paths, retr_paths, cfg, out_root, variants)
        all_rows.append(df_metrics)
        del cache_paths, retr_paths, df_metrics
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if all_rows:
        main_df = pd.concat(all_rows, ignore_index=True)
        write_reports(main_df, out_root, make_figures=not bool(getattr(args, "skip_figures", False)))
    return {"ok": True, "artifact_root": str(out_root)}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LiteMedRAG MIWAI 2026 reference runner")
    p.add_argument("--datasets", nargs="+", default=["slake", "imageclef_vqa_med_2019"])
    p.add_argument("--variants", nargs="*", default=None, choices=["base", "text_rag", "mm_rag", "litemedrag_acc", "litemedrag_ground"])
    p.add_argument("--config", default=None)
    p.add_argument("--artifact-root", default="artifacts")
    p.add_argument("--device", default=None)
    p.add_argument("--encoder", choices=["biomedclip", "random"], default="biomedclip")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data and random encoder for smoke tests")
    p.add_argument("--limit-per-split", type=int, default=None, help="Limit samples per split for smoke/debug runs")
    p.add_argument("--force-embeddings", action="store_true")
    p.add_argument("--force-retrieval", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--skip-figures", action="store_true")
    return p


def main(*, datasets: Optional[Sequence[str]] = None, **kwargs) -> Dict[str, Any]:
    """Programmatic entry point."""
    parser = build_argparser()
    ns = parser.parse_args([])
    if datasets is not None:
        ns.datasets = list(datasets)
    for k, v in kwargs.items():
        if hasattr(ns, k.replace("-", "_")) and v is not None:
            setattr(ns, k.replace("-", "_"), v)
    return run_pipeline(ns)


def cli() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    cli()
