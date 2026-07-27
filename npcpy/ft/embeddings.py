"""Embedding fine-tuning: classical (real-valued) and quantum (Hilbert-space).

Functional API matching npcpy/ft patterns.

Usage:
    from npcpy.ft.embeddings import run_embedding_sft, EmbeddingConfig
    config = EmbeddingConfig(base_model_name="sentence-transformers/all-MiniLM-L6-v2")
    run_embedding_sft(anchors, positives, config=config)

    from npcpy.ft.embeddings import run_hilbert_embedding_sft, HilbertConfig
    run_hilbert_embedding_sft(anchors, positives, config=config)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import os
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None

import numpy as np

@dataclass
class EmbeddingConfig:
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    output_model_path: str = "models/embedding"
    device: str = "cpu"
    embedding_dim: int = 384
    num_train_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    temperature: float = 0.07
    margin: float = 0.5
    loss_type: str = "infonce"
    max_length: int = 256
    logging_steps: int = 50
    save_steps: int = 500
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    from_scratch: bool = False
    vocab_size: int = 30522
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536

@dataclass
class HilbertConfig:
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    output_model_path: str = "models/hilbert_embedding"
    device: str = "cpu"
    embedding_dim: int = 384
    num_train_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    temperature: float = 0.07
    margin: float = 0.5
    loss_type: str = "hilbert_infonce"
    max_length: int = 256
    logging_steps: int = 50
    save_steps: int = 500
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    use_phase: bool = True
    phase_init_scale: float = 0.1
    lambda_phase: float = 0.5
    from_scratch: bool = False
    vocab_size: int = 30522
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536

def _mean_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)

def _infonce_loss(anchor_emb, positive_emb, temperature=0.07):
    anchor_emb = F.normalize(anchor_emb, p=2, dim=-1)
    positive_emb = F.normalize(positive_emb, p=2, dim=-1)
    logits = anchor_emb @ positive_emb.T / temperature
    labels = torch.arange(len(anchor_emb), device=anchor_emb.device)
    return F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)

def _triplet_loss(anchor_emb, positive_emb, negative_emb, margin=0.5):
    d_pos = 1 - F.cosine_similarity(anchor_emb, positive_emb, dim=-1)
    d_neg = 1 - F.cosine_similarity(anchor_emb, negative_emb, dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()

def _mnr_loss(anchor_emb, positive_emb, temperature=1.0):
    anchor_emb = F.normalize(anchor_emb, p=2, dim=-1)
    positive_emb = F.normalize(positive_emb, p=2, dim=-1)
    scores = anchor_emb @ positive_emb.T / temperature
    labels = torch.arange(len(anchor_emb), device=anchor_emb.device)
    return F.cross_entropy(scores, labels)

class ComplexTensor:
    """ψ = |ψ| · e^(iθ)"""
    def __init__(self, magnitude, angle):
        self.magnitude = magnitude
        self.angle = angle

def _complex_linear(x, w_real, w_imag, b_real, b_imag):
    """Linear layer with complex weights."""
    real = x @ w_real.T + b_real
    imag = x @ w_imag.T + b_imag
    mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
    ang = torch.atan2(imag, real)
    return ComplexTensor(mag, ang)

def _hilbert_similarity(ct1, ct2):
    """Re(⟨ψ₁|ψ₂⟩) / (||ψ₁|| ||ψ₂||)"""
    mag_prod = ct1.magnitude * ct2.magnitude
    phase_diff = ct1.angle - ct2.angle
    real_part = (mag_prod * torch.cos(phase_diff)).sum(dim=-1)
    norm1 = (ct1.magnitude ** 2).sum(dim=-1).sqrt()
    norm2 = (ct2.magnitude ** 2).sum(dim=-1).sqrt()
    return real_part / (norm1 * norm2 + 1e-8)

def _hilbert_similarity_matrix(batch1, batch2):
    """Pairwise Hilbert similarities."""
    mag1 = batch1.magnitude
    ang1 = batch1.angle
    mag2 = batch2.magnitude
    ang2 = batch2.angle

    mag_prod = mag1.unsqueeze(1) * mag2.unsqueeze(0)
    phase_diff = ang1.unsqueeze(1) - ang2.unsqueeze(0)
    real_part = (mag_prod * torch.cos(phase_diff)).sum(dim=-1)

    norm1 = (mag1 ** 2).sum(dim=-1).sqrt().unsqueeze(1)
    norm2 = (mag2 ** 2).sum(dim=-1).sqrt().unsqueeze(0)
    return real_part / (norm1 * norm2 + 1e-8)

def _hilbert_infonce(anchor_ct, positive_ct, temperature=0.07):
    sim_matrix = _hilbert_similarity_matrix(anchor_ct, positive_ct)
    labels = torch.arange(len(anchor_ct.magnitude), device=anchor_ct.magnitude.device)
    loss_a2p = F.cross_entropy(sim_matrix / temperature, labels)
    loss_p2a = F.cross_entropy(sim_matrix.T / temperature, labels)
    return (loss_a2p + loss_p2a) / 2.0

def _phase_triplet_loss(anchor_ct, positive_ct, negative_ct, margin=0.5, lambda_phase=0.5):
    d_pos_mag = 1 - _hilbert_similarity(anchor_ct, positive_ct)
    d_neg_mag = 1 - _hilbert_similarity(anchor_ct, negative_ct)
    triplet = F.relu(d_pos_mag - d_neg_mag + margin).mean()

    phase_diff_pos = 1 - torch.cos(anchor_ct.angle - positive_ct.angle).mean(dim=-1)
    phase_diff_neg = 1 - torch.cos(anchor_ct.angle - negative_ct.angle).mean(dim=-1)
    phase_penalty = F.relu(phase_diff_pos - phase_diff_neg).mean()

    return triplet + lambda_phase * phase_penalty

def _normalize_hilbert(ct):
    n = (ct.magnitude ** 2).sum(dim=-1, keepdim=True).sqrt()
    return ComplexTensor(ct.magnitude / (n + 1e-8), ct.angle)

def _create_foundation_model(config):
    """Create a small transformer from scratch for embedding training."""
    from transformers import BertConfig, BertModel

    bert_config = BertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=512,
        type_vocab_size=2,
    )
    model = BertModel(bert_config)
    return model

def _load_base_model(config, tokenizer):
    """Load base model: either from pretrained or from scratch."""
    if getattr(config, 'from_scratch', False):
        print(f"Training from scratch: vocab={config.vocab_size}, hidden={config.hidden_size}, layers={config.num_hidden_layers}")
        base = _create_foundation_model(config)
        base.resize_token_embeddings(len(tokenizer))
    else:
        base = AutoModel.from_pretrained(config.base_model_name)
    return base

def run_embedding_sft_torch(
    anchors: List[str],
    positives: List[str],
    negatives: Optional[List[str]] = None,
    config: Optional[EmbeddingConfig] = None,
) -> str:
    if config is None:
        config = EmbeddingConfig()

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    os.makedirs(config.output_model_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    base = _load_base_model(config, tokenizer)

    device = torch.device(config.device if config.device != "mlx" else "cpu")
    base = base.to(device)
    projector = nn.Linear(base.config.hidden_size, config.embedding_dim).to(device)

    def encode(texts):
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=config.max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = base(**enc).last_hidden_state
        pooled = _mean_pooling(out, enc["attention_mask"])
        emb = projector(pooled)
        return F.normalize(emb, p=2, dim=-1)

    optimizer = torch.optim.AdamW(
        list(base.parameters()) + list(projector.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = (len(anchors) // config.batch_size) * config.num_train_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )

    for epoch in range(config.num_train_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(anchors), config.batch_size):
            batch_a = anchors[i:i + config.batch_size]
            batch_p = positives[i:i + config.batch_size]

            a_emb = encode(batch_a)
            p_emb = encode(batch_p)

            if config.loss_type == "infonce":
                loss = _infonce_loss(a_emb, p_emb, config.temperature)
            elif config.loss_type == "mnr":
                loss = _mnr_loss(a_emb, p_emb, config.temperature)
            elif config.loss_type == "triplet" and negatives is not None:
                batch_n = negatives[i:i + config.batch_size]
                n_emb = encode(batch_n)
                loss = _triplet_loss(a_emb, p_emb, n_emb, config.margin)
            else:
                loss = _infonce_loss(a_emb, p_emb, config.temperature)

            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            if (i // config.batch_size + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(base.parameters()) + list(projector.parameters()), 1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                if scheduler and scheduler.last_epoch < warmup_steps:
                    scheduler.step()

        print(f"Epoch {epoch+1}/{config.num_train_epochs} avg loss: {epoch_loss / max(num_batches, 1):.4f}")

    torch.save({
        "base": base.state_dict(),
        "projector": projector.state_dict(),
        "config": config,
    }, os.path.join(config.output_model_path, "model.pt"))
    tokenizer.save_pretrained(config.output_model_path)

    print(f"Model saved to {config.output_model_path}")
    return config.output_model_path

def run_embedding_sft_mlx(
    anchors: List[str],
    positives: List[str],
    negatives: Optional[List[str]] = None,
    config: Optional[EmbeddingConfig] = None,
) -> str:
    if config is None:
        config = EmbeddingConfig()

    if not MLX_AVAILABLE:
        raise ImportError("MLX backend requires mlx. pip install mlx")

    os.makedirs(config.output_model_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    base = _load_base_model(config, tokenizer)
    base.eval()

    hidden = base.config.hidden_size
    import numpy as np
    proj_w = mx.array(np.random.randn(hidden, config.embedding_dim).astype(np.float32) * 0.02)
    proj_b = mx.array(np.zeros(config.embedding_dim, dtype=np.float32))

    def encode_mlx(texts):
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=config.max_length, return_tensors="pt")
        with torch.no_grad():
            out = base(**enc).last_hidden_state.numpy()
        mask = enc["attention_mask"].numpy().astype(np.float32)[:, :, None]
        pooled = (out * mask).sum(axis=1) / mask.sum(axis=1)
        pooled_mx = mx.array(pooled)
        emb = pooled_mx @ proj_w + proj_b
        norm = mx.sqrt(mx.sum(emb ** 2, axis=-1, keepdims=True) + 1e-8)
        return emb / norm

    def mlx_infonce(a, p, temperature=0.07):
        logits = (a @ p.T) / temperature
        labels = mx.arange(a.shape[0])
        loss_a = mx.mean(mx.logsumexp(logits, axis=1) - logits[labels, labels])
        loss_p = mx.mean(mx.logsumexp(logits.T, axis=1) - logits[labels, labels])
        return (loss_a + loss_p) / 2.0

    for epoch in range(config.num_train_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(anchors), config.batch_size):
            batch_a = anchors[i:i + config.batch_size]
            batch_p = positives[i:i + config.batch_size]

            a_emb = encode_mlx(batch_a)
            p_emb = encode_mlx(batch_p)

            loss = mlx_infonce(a_emb, p_emb, config.temperature)
            epoch_loss += float(loss)
            num_batches += 1

        print(f"Epoch {epoch+1}/{config.num_train_epochs} avg loss: {epoch_loss / max(num_batches, 1):.4f}")

    np.savez(os.path.join(config.output_model_path, "projector.npz"),
             w=np.array(proj_w), b=np.array(proj_b))
    tokenizer.save_pretrained(config.output_model_path)

    print(f"MLX model saved to {config.output_model_path}")
    return config.output_model_path

def run_embedding_sft(
    anchors: List[str],
    positives: List[str],
    negatives: Optional[List[str]] = None,
    config: Optional[EmbeddingConfig] = None,
) -> str:
    if config is None:
        config = EmbeddingConfig()

    if config.device == "mlx":
        return run_embedding_sft_mlx(anchors, positives, negatives, config)
    return run_embedding_sft_torch(anchors, positives, negatives, config)

def run_hilbert_embedding_sft_torch(
    anchors: List[str],
    positives: List[str],
    negatives: Optional[List[str]] = None,
    config: Optional[HilbertConfig] = None,
) -> str:
    if config is None:
        config = HilbertConfig()

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    os.makedirs(config.output_model_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    base = _load_base_model(config, tokenizer)

    device = torch.device(config.device if config.device != "mlx" else "cpu")
    base = base.to(device)

    hidden = base.config.hidden_size
    dim = config.embedding_dim

    proj_w_real = nn.Parameter(torch.randn(dim, hidden) * 0.02).to(device)
    proj_w_imag = nn.Parameter(torch.randn(dim, hidden) * 0.02).to(device)
    proj_b_real = nn.Parameter(torch.zeros(dim)).to(device)
    proj_b_imag = nn.Parameter(torch.zeros(dim)).to(device)
    phase_init = nn.Parameter(torch.randn(dim) * config.phase_init_scale).to(device)

    params = list(base.parameters()) + [proj_w_real, proj_w_imag, proj_b_real, proj_b_imag, phase_init]

    def encode_hilbert(texts):
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=config.max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = base(**enc).last_hidden_state
        pooled = _mean_pooling(out, enc["attention_mask"])
        ct = _complex_linear(pooled, proj_w_real, proj_w_imag, proj_b_real, proj_b_imag)
        ct = ComplexTensor(ct.magnitude, ct.angle + phase_init)
        return _normalize_hilbert(ct)

    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = (len(anchors) // config.batch_size) * config.num_train_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )

    for epoch in range(config.num_train_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(anchors), config.batch_size):
            batch_a = anchors[i:i + config.batch_size]
            batch_p = positives[i:i + config.batch_size]

            a_ct = encode_hilbert(batch_a)
            p_ct = encode_hilbert(batch_p)

            if config.loss_type == "hilbert_infonce":
                loss = _hilbert_infonce(a_ct, p_ct, config.temperature)
            elif config.loss_type == "phase_triplet" and negatives is not None:
                batch_n = negatives[i:i + config.batch_size]
                n_ct = encode_hilbert(batch_n)
                loss = _phase_triplet_loss(a_ct, p_ct, n_ct, config.margin, config.lambda_phase)
            else:
                loss = _hilbert_infonce(a_ct, p_ct, config.temperature)

            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            if (i // config.batch_size + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler and scheduler.last_epoch < warmup_steps:
                    scheduler.step()

        print(f"Epoch {epoch+1}/{config.num_train_epochs} avg loss: {epoch_loss / max(num_batches, 1):.4f}")

    torch.save({
        "base": base.state_dict(),
        "proj_w_real": proj_w_real,
        "proj_w_imag": proj_w_imag,
        "proj_b_real": proj_b_real,
        "proj_b_imag": proj_b_imag,
        "phase_init": phase_init,
        "config": config,
    }, os.path.join(config.output_model_path, "model.pt"))
    tokenizer.save_pretrained(config.output_model_path)

    print(f"Hilbert model saved to {config.output_model_path}")
    return config.output_model_path

def run_hilbert_embedding_sft_mlx(
    anchors: List[str],
    positives: List[str],
    negatives: Optional[List[str]] = None,
    config: Optional[HilbertConfig] = None,
) -> str:
    if config is None:
        config = HilbertConfig()

    if not MLX_AVAILABLE:
        raise ImportError("MLX backend requires mlx. pip install mlx")

    os.makedirs(config.output_model_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    base = _load_base_model(config, tokenizer)
    base.eval()

    hidden = base.config.hidden_size
    dim = config.embedding_dim
    import numpy as np

    proj_w_real = mx.array(np.random.randn(hidden, dim).astype(np.float32) * 0.02)
    proj_w_imag = mx.array(np.random.randn(hidden, dim).astype(np.float32) * 0.02)
    proj_b_real = mx.array(np.zeros(dim, dtype=np.float32))
    proj_b_imag = mx.array(np.zeros(dim, dtype=np.float32))
    phase_init = mx.array(np.random.randn(dim).astype(np.float32) * config.phase_init_scale)

    def encode_hilbert_mlx(texts):
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=config.max_length, return_tensors="pt")
        with torch.no_grad():
            out = base(**enc).last_hidden_state.numpy()
        mask = enc["attention_mask"].numpy().astype(np.float32)[:, :, None]
        pooled = (out * mask).sum(axis=1) / mask.sum(axis=1)
        pooled_mx = mx.array(pooled)

        real = pooled_mx @ proj_w_real + proj_b_real
        imag = pooled_mx @ proj_w_imag + proj_b_imag
        mag = mx.sqrt(real ** 2 + imag ** 2 + 1e-8)
        ang = mx.arctan2(imag, real) + phase_init
        norm = mx.sqrt(mx.sum(mag ** 2, axis=-1, keepdims=True) + 1e-8)
        return mag / norm, ang

    def mlx_hilbert_infonce(a_mag, a_ang, p_mag, p_ang, temperature=0.07):
        mag_prod = a_mag[:, None, :] * p_mag[None, :, :]
        phase_diff = a_ang[:, None, :] - p_ang[None, :, :]
        real_part = mx.sum(mag_prod * mx.cos(phase_diff), axis=-1)
        norm1 = mx.sqrt(mx.sum(a_mag ** 2, axis=-1))
        norm2 = mx.sqrt(mx.sum(p_mag ** 2, axis=-1))
        sim = real_part / (norm1[:, None] * norm2[None, :] + 1e-8)

        labels = mx.arange(a_mag.shape[0])
        loss_a = mx.mean(mx.logsumexp(sim / temperature, axis=1) - (sim / temperature)[labels, labels])
        loss_p = mx.mean(mx.logsumexp(sim.T / temperature, axis=1) - (sim / temperature)[labels, labels])
        return (loss_a + loss_p) / 2.0

    for epoch in range(config.num_train_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(anchors), config.batch_size):
            batch_a = anchors[i:i + config.batch_size]
            batch_p = positives[i:i + config.batch_size]

            a_mag, a_ang = encode_hilbert_mlx(batch_a)
            p_mag, p_ang = encode_hilbert_mlx(batch_p)

            loss = mlx_hilbert_infonce(a_mag, a_ang, p_mag, p_ang, config.temperature)
            epoch_loss += float(loss)
            num_batches += 1

        print(f"Epoch {epoch+1}/{config.num_train_epochs} avg loss: {epoch_loss / max(num_batches, 1):.4f}")

    np.savez(os.path.join(config.output_model_path, "projector.npz"),
             w_real=np.array(proj_w_real), w_imag=np.array(proj_w_imag),
             b_real=np.array(proj_b_real), b_imag=np.array(proj_b_imag),
             phase_init=np.array(phase_init))
    tokenizer.save_pretrained(config.output_model_path)

    print(f"MLX Hilbert model saved to {config.output_model_path}")
    return config.output_model_path

def run_hilbert_embedding_sft(
    anchors: List[str],
    positives: List[str],
    negatives: Optional[List[str]] = None,
    config: Optional[HilbertConfig] = None,
) -> str:
    if config is None:
        config = HilbertConfig()

    if config.device == "mlx":
        return run_hilbert_embedding_sft_mlx(anchors, positives, negatives, config)
    return run_hilbert_embedding_sft_torch(anchors, positives, negatives, config)

def load_embedding_model(model_path: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ckpt = torch.load(os.path.join(model_path, "model.pt"), map_location="cpu", weights_only=False)
    config = ckpt["config"]

    base = _load_base_model(config, tokenizer)
    projector = nn.Linear(base.config.hidden_size, config.embedding_dim)

    base.load_state_dict(ckpt["base"])
    projector.load_state_dict(ckpt["projector"])

    dev = torch.device(device if device != "mlx" else "cpu")
    base = base.to(dev)
    projector = projector.to(dev)

    return base, projector, tokenizer, config

def encode_texts(
    texts: List[str],
    model,
    projector,
    tokenizer,
    device: str = "cpu",
    max_length: int = 256,
    batch_size: int = 32,
) -> List[List[float]]:
    """Encode texts to normalized embeddings, one ``batch_size`` chunk at a time.

    The whole list used to be tokenized and forwarded in a single pass, so peak
    memory scaled with ``len(texts)`` and large inputs went OOM. Chunking keeps
    peak memory bounded by ``batch_size`` instead.

    Results are unchanged by chunking: padding is per-batch, but ``_mean_pooling``
    divides by the attention mask, so pad tokens never contribute to the mean.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    model.eval()
    dev = torch.device(device)

    embeddings: List[List[float]] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}

            out = model(**enc).last_hidden_state
            pooled = _mean_pooling(out, enc["attention_mask"])
            emb = F.normalize(projector(pooled), p=2, dim=-1)

            embeddings.extend(emb.cpu().numpy().tolist())

    return embeddings

def evaluate_embeddings(
    anchors: List[str],
    positives: List[str],
    negatives: List[str],
    model,
    projector,
    tokenizer,
    device: str = "cpu",
    max_length: int = 256,
    batch_size: int = 32,
) -> Dict[str, float]:
    a_emb = np.array(encode_texts(anchors, model, projector, tokenizer, device, max_length, batch_size))
    p_emb = np.array(encode_texts(positives, model, projector, tokenizer, device, max_length, batch_size))
    n_emb = np.array(encode_texts(negatives, model, projector, tokenizer, device, max_length, batch_size))

    combined = np.concatenate([p_emb, n_emb], axis=0)
    sim = a_emb @ combined.T
    labels = np.arange(len(a_emb))

    ranks = (sim > sim[np.arange(len(a_emb)), labels][:, None]).sum(axis=1) + 1
    mrr = (1.0 / ranks).mean()
    r1 = (ranks == 1).mean()
    r5 = (ranks <= 5).mean()

    return {"mrr": float(mrr), "recall@1": float(r1), "recall@5": float(r5)}
