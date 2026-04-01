"""
Qwen3-8B (frozen) + Graph Memory Module (trainable)

Backbone LM is frozen. Only the graph memory pathway is trained with NTP loss.
This tests whether a pretrained LM benefits from an inspectable graph memory
at a scale where parametric memory alone is insufficient.

Modes:
  off          — Qwen with restricted local window, no graph
  nodes_only   — graph pathway with uniform retrieval (all spans averaged)
  learned      — graph pathway with learned retrieval routing

Usage:
  python qwen_graph_ntp.py --mode off --device cuda --max-docs 5000 --epochs 3
  python qwen_graph_ntp.py --mode nodes_only --device cuda --max-docs 5000 --epochs 3
  python qwen_graph_ntp.py --mode learned --device cuda --max-docs 5000 --epochs 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(msg: str):
    print(msg, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_documents(data_dir: str, max_docs: int = 0) -> list[str]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.txt"))
    if max_docs > 0:
        files = files[:max_docs]
    texts = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        if len(text) > 500:
            texts.append(text)
    return texts


def make_examples(
    texts: list[str],
    tokenizer,
    local_window: int,
    span_size: int,
    max_doc_tokens: int,
) -> list[dict]:
    examples = []
    for doc_idx, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)[:max_doc_tokens]
        if len(token_ids) < local_window + span_size:
            continue

        spans = []
        for i in range(0, len(token_ids) - span_size + 1, span_size):
            spans.append(token_ids[i : i + span_size])

        for window_start_span in range(1, len(spans) - 1):
            past_span_ids = [spans[j] for j in range(window_start_span)]
            current_and_next = []
            for j in range(window_start_span, min(window_start_span + (local_window // span_size) + 1, len(spans))):
                current_and_next.extend(spans[j])

            if len(current_and_next) < 2:
                continue

            examples.append({
                "doc_idx": doc_idx,
                "past_spans": past_span_ids,
                "tokens": current_and_next[:local_window + span_size],
                "window_start_span": window_start_span,
            })

    return examples


# ---------------------------------------------------------------------------
# Graph Memory Module (trainable, fp32 weights for stable optimizer)
# ---------------------------------------------------------------------------

class GraphMemory(nn.Module):
    def __init__(self, d_model: int, d_graph: int = 512):
        super().__init__()
        self.d_model = d_model
        self.d_graph = d_graph
        self.span_proj = nn.Linear(d_model, d_graph)
        self.query_proj = nn.Linear(d_model, d_graph)
        self.graph_to_model = nn.Linear(d_graph, d_model)

    def build_nodes(self, span_embeds: torch.Tensor) -> torch.Tensor:
        """(*, d_model) → (*, d_graph). Works for any leading dims."""
        return self.span_proj(span_embeds)

    # -- single-example methods (kept for trace/demo code) --

    def retrieve_uniform(self, queries: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        if nodes.shape[0] == 0:
            return torch.zeros(queries.shape[0], self.d_graph, device=queries.device, dtype=queries.dtype)
        avg = nodes.mean(dim=0, keepdim=True)
        return avg.expand(queries.shape[0], -1)

    def retrieve_learned(self, queries: torch.Tensor, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = queries.shape[0]
        if nodes.shape[0] == 0:
            return (
                torch.zeros(seq_len, self.d_graph, device=queries.device, dtype=queries.dtype),
                torch.zeros(seq_len, 0, device=queries.device),
            )
        q = self.query_proj(queries)
        scores = torch.matmul(q, nodes.T)
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, nodes)
        return context, weights

    # -- batched methods (used in compute_loss_batch) --

    def retrieve_learned_batched(
        self, queries: torch.Tensor, nodes: torch.Tensor, node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,L,d_model), (B,S,d_graph), (B,S) bool → (B,L,d_graph), (B,L,S)"""
        q = self.query_proj(queries)
        scores = torch.bmm(q, nodes.transpose(1, 2))
        scores = scores.masked_fill(~node_mask.unsqueeze(1), float("-inf"))
        weights = F.softmax(scores, dim=-1).nan_to_num(0.0)
        return torch.bmm(weights, nodes), weights

    def retrieve_uniform_batched(
        self, queries: torch.Tensor, nodes: torch.Tensor, node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """(B,L,d_model), (B,S,d_graph), (B,S) bool → (B,L,d_graph)"""
        counts = node_mask.float().sum(-1, keepdim=True).clamp(min=1)
        avg = (nodes * node_mask.unsqueeze(-1).float()).sum(1) / counts
        return avg.unsqueeze(1).expand(-1, queries.shape[1], -1)

    def fuse(self, hidden: torch.Tensor, graph_context: torch.Tensor) -> torch.Tensor:
        return hidden + self.graph_to_model(graph_context)


# ---------------------------------------------------------------------------
# Span Encoder (uses frozen Qwen base model, outputs fp16 on GPU)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_spans_batch(
    base_model: nn.Module,
    span_ids_list: list[list[int]],
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode spans through frozen base model, mean-pool. Returns (N, d_model) fp16 on GPU."""
    all_embeds = []
    for i in range(0, len(span_ids_list), batch_size):
        batch_spans = span_ids_list[i : i + batch_size]
        max_len = max(len(s) for s in batch_spans)

        input_ids = torch.zeros(len(batch_spans), max_len, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(batch_spans), max_len, dtype=torch.long, device=device)
        for j, sp in enumerate(batch_spans):
            input_ids[j, :len(sp)] = torch.tensor(sp, dtype=torch.long)
            attention_mask[j, :len(sp)] = 1

        hidden = base_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        ).last_hidden_state  # (B, seq_len, d_model) fp16

        mask = attention_mask.unsqueeze(-1).half()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_embeds.append(pooled)  # GPU, fp16

    return torch.cat(all_embeds, dim=0)


@torch.no_grad()
def precompute_span_embeddings(
    base_model: nn.Module,
    examples: list[dict],
    device: str,
    batch_size: int = 64,
) -> dict[tuple, torch.Tensor]:
    """Encode all unique spans once. Returns {tuple(token_ids): embedding} on GPU fp16."""
    unique_spans: dict[tuple, list[int]] = {}
    for ex in examples:
        for sp in ex["past_spans"]:
            key = tuple(sp)
            if key not in unique_spans:
                unique_spans[key] = sp

    keys = list(unique_spans.keys())
    spans = [unique_spans[k] for k in keys]
    log(f"  {len(spans)} unique spans to encode")

    if not spans:
        return {}

    all_embeds = []
    for i in range(0, len(spans), batch_size):
        batch = spans[i : i + batch_size]
        embeds = encode_spans_batch(base_model, batch, device, batch_size=len(batch))
        all_embeds.append(embeds)
        done = i + len(batch)
        if (i // batch_size + 1) % 100 == 0 or done == len(spans):
            log(f"    encoded {done}/{len(spans)}")

    all_embeds = torch.cat(all_embeds, dim=0)
    cache = {k: all_embeds[i] for i, k in enumerate(keys)}
    log(f"  done: {len(cache)} spans cached on GPU fp16")
    return cache


# ---------------------------------------------------------------------------
# Microbatched forward
# ---------------------------------------------------------------------------

def compute_loss_batch(
    base_model: nn.Module,
    graph: GraphMemory,
    examples: list[dict],
    mode: str,
    device: str,
    local_window: int,
    span_cache: dict[tuple, torch.Tensor],
    lm_head_weight: torch.Tensor,
) -> tuple[torch.Tensor | None, int, int, float, float]:
    """
    Fully batched: backbone forward + graph retrieval + loss all via bmm.
    Returns: (loss, n_tokens, n_tail_tokens, weighted_full_nll, weighted_tail_nll)
    """
    B = len(examples)
    seq_lens = [len(ex["tokens"]) - 1 for ex in examples]
    max_seq = max(seq_lens)

    # 1. Padded batch → single backbone forward
    input_ids = torch.zeros(B, max_seq, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, max_seq, dtype=torch.long, device=device)
    labels = torch.full((B, max_seq), -100, dtype=torch.long, device=device)
    for i, ex in enumerate(examples):
        L = seq_lens[i]
        input_ids[i, :L] = torch.tensor(ex["tokens"][:-1], dtype=torch.long)
        attn_mask[i, :L] = 1
        labels[i, :L] = torch.tensor(ex["tokens"][1:], dtype=torch.long)

    with torch.no_grad():
        hidden_all = base_model(
            input_ids=input_ids, attention_mask=attn_mask, use_cache=False,
        ).last_hidden_state  # (B, max_seq, d_model) fp16

    # 2. Batched graph memory
    with torch.amp.autocast("cuda", dtype=torch.float16):
        if mode != "off":
            span_counts = [len(ex["past_spans"]) for ex in examples]
            max_spans = max(span_counts) if span_counts else 0

            if max_spans > 0:
                d_m = hidden_all.shape[-1]
                padded_spans = torch.zeros(B, max_spans, d_m, device=device, dtype=torch.float16)
                node_mask = torch.zeros(B, max_spans, dtype=torch.bool, device=device)
                for i, ex in enumerate(examples):
                    n = len(ex["past_spans"])
                    if n > 0:
                        padded_spans[i, :n] = torch.stack(
                            [span_cache[tuple(sp)] for sp in ex["past_spans"]])
                        node_mask[i, :n] = True

                nodes = graph.build_nodes(padded_spans)
                if mode == "nodes_only":
                    ctx = graph.retrieve_uniform_batched(hidden_all, nodes, node_mask)
                else:
                    ctx, _ = graph.retrieve_learned_batched(hidden_all, nodes, node_mask)
                fused = graph.fuse(hidden_all, ctx)
            else:
                fused = hidden_all
        else:
            fused = hidden_all

        logits = F.linear(fused, lm_head_weight)  # (B, max_seq, vocab)

    # 3. Batched loss
    flat_logits = logits.float().view(-1, logits.shape[-1])
    flat_labels = labels.view(-1)
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

    if loss.isnan():
        return None, 0, 0, 0.0, 0.0

    # 4. Logging metrics
    total_tokens = sum(seq_lens)
    with torch.no_grad():
        per_token = F.cross_entropy(flat_logits, flat_labels,
                                    ignore_index=-100, reduction="none").view(B, max_seq)
        total_full_nll = per_token.sum().item()
        total_tail_nll = 0.0
        total_tail_tokens = 0
        for i in range(B):
            L = seq_lens[i]
            tail_start = local_window if local_window > 0 else max(L // 3, 1)
            if tail_start < L:
                total_tail_nll += per_token[i, tail_start:L].sum().item()
                total_tail_tokens += L - tail_start
            else:
                total_tail_nll += per_token[i, :L].sum().item()
                total_tail_tokens += L

    return loss, total_tokens, total_tail_tokens, total_full_nll, total_tail_nll


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    base_model: nn.Module,
    graph: GraphMemory,
    train_examples: list[dict],
    mode: str,
    epochs: int,
    lr: float,
    device: str,
    local_window: int,
    span_cache: dict[tuple, torch.Tensor],
    lm_head_weight: torch.Tensor,
    micro_batch: int = 16,
    accum_steps: int = 2,
    save_dir: str = "train_qwen",
):
    if mode == "off":
        log("  mode=off: nothing to train, skipping to eval")
        return

    optimizer = torch.optim.AdamW(graph.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    n_total = len(train_examples)
    n_microbatches = (n_total + micro_batch - 1) // micro_batch

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    log(f"\n>>> Training graph module [{mode}] — {n_total} examples, {epochs} epochs")
    log(f"    micro_batch={micro_batch}, accum_steps={accum_steps}, "
        f"effective_batch={micro_batch * accum_steps}")
    log(f"    trainable params: {sum(p.numel() for p in graph.parameters()):,}")

    for epoch in range(epochs):
        t0 = time.time()
        graph.train()
        random.shuffle(train_examples)

        epoch_full = 0.0
        epoch_tail = 0.0
        epoch_tokens = 0
        epoch_tail_tokens = 0

        optimizer.zero_grad()

        for mb_idx in range(n_microbatches):
            start = mb_idx * micro_batch
            batch_exs = train_examples[start : start + micro_batch]

            loss, n_tok, n_tail, w_full, w_tail = compute_loss_batch(
                base_model, graph, batch_exs, mode, device,
                local_window, span_cache, lm_head_weight,
            )

            if loss is not None:
                scaler.scale(loss / accum_steps).backward()

            epoch_full += w_full
            epoch_tail += w_tail
            epoch_tokens += n_tok
            epoch_tail_tokens += n_tail

            if (mb_idx + 1) % accum_steps == 0 or mb_idx == n_microbatches - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(graph.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (mb_idx + 1) % 50 == 0:
                done_ex = min((mb_idx + 1) * micro_batch, n_total)
                r_nll = epoch_full / epoch_tokens if epoch_tokens > 0 else 0
                r_tail = epoch_tail / epoch_tail_tokens if epoch_tail_tokens > 0 else 0
                elapsed = time.time() - t0
                rate = done_ex / elapsed
                eta = (n_total - done_ex) / rate if rate > 0 else 0
                log(f"  [{mode:12s}] ep {epoch+1:2d} | {done_ex:6d}/{n_total} | "
                    f"nll={r_nll:.4f} tail={r_tail:.4f} | "
                    f"{rate:.0f} ex/s | eta {eta:.0f}s")

        avg_nll = epoch_full / epoch_tokens if epoch_tokens > 0 else 0
        avg_tail = epoch_tail / epoch_tail_tokens if epoch_tail_tokens > 0 else 0
        elapsed = time.time() - t0
        log(f"  [{mode:12s}] ep {epoch+1:2d} DONE | nll={avg_nll:.4f} tail={avg_tail:.4f} | "
            f"{n_total} ex | {elapsed:.1f}s")

    ckpt_path = Path(save_dir) / f"graph_{mode}.pt"
    torch.save(graph.state_dict(), ckpt_path)
    log(f"  saved → {ckpt_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    base_model: nn.Module,
    graph: GraphMemory,
    test_examples: list[dict],
    mode: str,
    device: str,
    local_window: int,
    span_cache: dict[tuple, torch.Tensor],
    lm_head_weight: torch.Tensor,
    micro_batch: int = 16,
) -> dict:
    graph.eval()
    total_full = 0.0
    total_tail = 0.0
    total_tokens = 0
    total_tail_tokens = 0

    n = len(test_examples)
    log(f"  eval [{mode}] — {n} examples")

    for start in range(0, n, micro_batch):
        batch_exs = test_examples[start : start + micro_batch]
        _, n_tok, n_tail, w_full, w_tail = compute_loss_batch(
            base_model, graph, batch_exs, mode, device,
            local_window, span_cache, lm_head_weight,
        )
        total_full += w_full
        total_tail += w_tail
        total_tokens += n_tok
        total_tail_tokens += n_tail

    if total_tokens == 0:
        return {"nll": 0.0, "tail_nll": 0.0, "ppl": 1.0, "tail_ppl": 1.0}

    mean_nll = total_full / total_tokens
    mean_tail = total_tail / total_tail_tokens
    result = {
        "nll": mean_nll,
        "tail_nll": mean_tail,
        "ppl": math.exp(min(mean_nll, 20.0)),
        "tail_ppl": math.exp(min(mean_tail, 20.0)),
    }
    log(f"  eval [{mode}] → nll={mean_nll:.4f} tail={mean_tail:.4f} "
        f"ppl={result['ppl']:.2f} tail_ppl={result['tail_ppl']:.2f}")
    return result


# ---------------------------------------------------------------------------
# Uniform vs Learned comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_uniform_vs_learned(
    base_model: nn.Module,
    graph: GraphMemory,
    test_examples: list[dict],
    device: str,
    local_window: int,
    span_cache: dict[tuple, torch.Tensor],
    lm_head_weight: torch.Tensor,
    micro_batch: int = 16,
    max_examples: int = 500,
) -> dict:
    graph.eval()
    examples = test_examples[:max_examples]
    log(f"  uniform vs learned — {len(examples)} examples")

    r_u = evaluate(base_model, graph, examples, "nodes_only", device,
                   local_window, span_cache, lm_head_weight, micro_batch)
    r_l = evaluate(base_model, graph, examples, "learned", device,
                   local_window, span_cache, lm_head_weight, micro_batch)

    delta = r_l["tail_nll"] - r_u["tail_nll"]
    return {
        "uniform_tail_nll": r_u["tail_nll"],
        "learned_tail_nll": r_l["tail_nll"],
        "delta": delta,
        "n": len(examples),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="learned",
                        choices=["off", "nodes_only", "learned"])
    parser.add_argument("--data-dir", type=str, default="data/arxiv_text/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--local-window", type=int, default=512)
    parser.add_argument("--span-size", type=int, default=256)
    parser.add_argument("--max-doc-tokens", type=int, default=8000)
    parser.add_argument("--d-graph", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=2,
                        help="optimizer step every N microbatches")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default="cache",
                        help="directory for cached examples + span embeddings")
    args = parser.parse_args()

    set_seed(args.seed)

    # ── Load backbone ──
    log(f"Loading backbone: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone,
        dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    d_model = backbone.config.hidden_size
    base_model = backbone.model        # Qwen3Model (no lm_head overhead)
    lm_head_weight = backbone.lm_head.weight  # fp16 on GPU, no copy

    log(f"  d_model={d_model}, vocab={backbone.config.vocab_size}")
    log(f"  backbone params: {sum(p.numel() for p in backbone.parameters()):,} (frozen)")

    # ── Cache key: same data params → same cache ──
    cache_key = hashlib.md5(
        f"{args.data_dir}|{args.max_docs}|{args.local_window}|{args.span_size}"
        f"|{args.max_doc_tokens}|{args.backbone}".encode()
    ).hexdigest()[:12]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    examples_path = cache_dir / f"examples_{cache_key}.pt"
    spans_path = cache_dir / f"spans_{cache_key}.pt"

    # ── Load or build examples ──
    if examples_path.exists():
        log(f"\nLoading cached examples from {examples_path}")
        cached = torch.load(examples_path, weights_only=False)
        train_examples = cached["train"]
        test_examples = cached["test"]
        log(f"  train: {len(train_examples)}, test: {len(test_examples)}")
    else:
        log(f"\nLoading documents from {args.data_dir}")
        all_texts = load_documents(args.data_dir, args.max_docs)
        log(f"  total docs: {len(all_texts)}")

        split_idx = int(len(all_texts) * 0.8)
        train_texts = all_texts[:split_idx]
        test_texts = all_texts[split_idx:]
        log(f"  train docs: {len(train_texts)}, test docs: {len(test_texts)}")

        log(f"Making examples (local_window={args.local_window}, span_size={args.span_size})")
        train_examples = make_examples(train_texts, tokenizer, args.local_window, args.span_size, args.max_doc_tokens)
        test_examples = make_examples(test_texts, tokenizer, args.local_window, args.span_size, args.max_doc_tokens)

        train_examples = [e for e in train_examples if len(e["past_spans"]) >= 2]
        test_examples = [e for e in test_examples if len(e["past_spans"]) >= 2]

        log(f"  train: {len(train_examples)}, test: {len(test_examples)}")
        torch.save({"train": train_examples, "test": test_examples}, examples_path)
        log(f"  saved → {examples_path}")

    if args.max_train > 0:
        train_examples = train_examples[:args.max_train]
    test_examples = test_examples[:args.max_test]

    log(f"  using train: {len(train_examples)}, test: {len(test_examples)}")

    # ── Graph module (fp32 for stable optimizer) ──
    graph = GraphMemory(d_model=d_model, d_graph=args.d_graph).to(args.device).float()
    log(f"\nGraph module: d_model={d_model}, d_graph={args.d_graph}")
    log(f"  trainable params: {sum(p.numel() for p in graph.parameters()):,}")

    # ── Load or build span cache (GPU, fp16) ──
    if spans_path.exists():
        log(f"\nLoading cached span embeddings from {spans_path}")
        span_data = torch.load(spans_path, weights_only=True)
        span_cache = {k: v.to(args.device) for k, v in span_data.items()}
        log(f"  loaded {len(span_cache)} spans → GPU")
    else:
        log("\nPre-encoding all spans...")
        span_cache = precompute_span_embeddings(
            base_model, train_examples + test_examples, args.device, batch_size=64,
        )
        log(f"  cached {len(span_cache)} unique spans")
        # Save to CPU for disk (smaller, portable)
        torch.save({k: v.cpu() for k, v in span_cache.items()}, spans_path)
        log(f"  saved → {spans_path}")

    # ── Train ──
    if args.mode != "off":
        train(
            base_model, graph, train_examples, args.mode,
            args.epochs, args.lr, args.device, args.local_window,
            span_cache, lm_head_weight, args.micro_batch, args.accum_steps,
        )

    # ── Evaluate ──
    log(f"\n{'='*60}")
    log(f"  Evaluation")
    log(f"{'='*60}")

    result = evaluate(
        base_model, graph, test_examples, args.mode, args.device,
        args.local_window, span_cache, lm_head_weight, args.micro_batch,
    )
    log(f"\n  [{args.mode}] nll={result['nll']:.4f} tail_nll={result['tail_nll']:.4f} "
        f"ppl={result['ppl']:.2f} tail_ppl={result['tail_ppl']:.2f}")

    # ── Uniform vs Learned ──
    if args.mode == "learned":
        log(f"\n{'='*60}")
        log(f"  Uniform vs Learned comparison")
        log(f"{'='*60}")
        ul_result = eval_uniform_vs_learned(
            base_model, graph, test_examples, args.device, args.local_window,
            span_cache, lm_head_weight, args.micro_batch,
        )
        log(f"  uniform_tail_nll:  {ul_result['uniform_tail_nll']:.4f}")
        log(f"  learned_tail_nll:  {ul_result['learned_tail_nll']:.4f}")
        log(f"  delta (L-U):       {ul_result['delta']:+.4f}")
        log(f"  {'LEARNED WINS' if ul_result['delta'] < 0 else 'UNIFORM WINS'}")
        result["uniform_comparison"] = ul_result

    # ── Save results ──
    out_dir = Path("results/qwen_graph")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.mode}_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "backbone": args.backbone,
            "max_docs": args.max_docs,
            "epochs": args.epochs,
            "local_window": args.local_window,
            "span_size": args.span_size,
            "d_graph": args.d_graph,
            "micro_batch": args.micro_batch,
            **result,
        }, f, indent=2)
    log(f"\n  results → {out_path}")


if __name__ == "__main__":
    main()
