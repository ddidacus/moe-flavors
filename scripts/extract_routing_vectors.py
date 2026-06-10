"""Extract per-sample, per-layer routing distributions from an MoE checkpoint.

Mode 'routing' (default):
    Stores sequence-averaged softmax routing distributions per layer.
    Output: {layer_idx: tensor(num_samples, num_experts)}

Mode 'eci':
    Builds per-token domain segmentation masks from conversation markers,
    computes per-domain averaged routing probs (ECI ingredients), and stores
    compact results plus full per-token data for a visualization subset.
"""

import argparse
import copy
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.temporal_moe import MoEConfig as TemporalMoEConfig, MoEMixin as TemporalMoEMixin
from src.vanilla_moe import MoEConfig as VanillaMoEConfig, MoEMixin as VanillaMoEMixin

DOMAINS = ["chat", "code", "math"]
DOMAIN_TO_IDX = {d: i for i, d in enumerate(DOMAINS)}


def messages_to_text(messages):
    return "\n".join(m["content"] for m in messages)


def build_domain_char_spans(messages):
    """Return (domain_idx, char_start, char_end) spans for the concatenated text."""
    spans = []
    char_offset = 0
    i = 0
    while i < len(messages):
        content = messages[i]["content"]
        msg_start = char_offset
        char_offset += len(content) + 1  # +1 for "\n" from join

        domain_found = None
        for domain in DOMAINS:
            prefix = f"Here is a question from the {domain} domain."
            if content.startswith(prefix):
                domain_found = domain
                break

        if domain_found is not None and i + 1 < len(messages):
            assistant_content = messages[i + 1]["content"]
            span_end = char_offset + len(assistant_content)
            char_offset += len(assistant_content) + 1
            spans.append((DOMAIN_TO_IDX[domain_found], msg_start, span_end))
            i += 2
        else:
            i += 1
    return spans


def tokens_to_domain_mask(offset_mapping, domain_spans, seq_len):
    """Map each token to a domain index (-1 for unmapped/padding)."""
    mask = torch.full((seq_len,), -1, dtype=torch.long)
    for tok_idx in range(seq_len):
        tok_start, tok_end = offset_mapping[tok_idx]
        if tok_start == tok_end:
            continue
        tok_mid = (tok_start + tok_end) / 2
        for domain_idx, span_start, span_end in domain_spans:
            if span_start <= tok_mid < span_end:
                mask[tok_idx] = domain_idx
                break
    return mask


class _HFMoELayerAdapter:
    """Makes an HF native MoE block (gate + experts) look like our custom layers."""
    def __init__(self, gate, num_experts, hook_target, logits_extractor):
        self.router = type("Router", (), {"gate": hook_target})()
        self.config = type("Config", (), {"num_experts": num_experts})()
        self.logits_extractor = logits_extractor


def _find_hf_moe_layers(model):
    layers = []
    for module in model.modules():
        if not (hasattr(module, "gate") and hasattr(module, "experts")):
            continue
        gate = module.gate
        if hasattr(gate, "out_features"):
            # nn.Linear gate (Qwen, Mixtral): hook output is logits directly
            layers.append(_HFMoELayerAdapter(
                gate, gate.out_features,
                hook_target=gate,
                logits_extractor=lambda inp, out: out,
            ))
        elif hasattr(gate, "weight"):
            # Parameter-based gate (DeepSeek): recompute logits from input
            n_experts = gate.weight.shape[0]
            def _make_extractor(g):
                def _extract(inp, out):
                    h = inp[0]
                    if h.dim() == 3:
                        h = h.reshape(-1, h.shape[-1])
                    return F.linear(h, g.weight)
                return _extract
            layers.append(_HFMoELayerAdapter(
                gate, n_experts,
                hook_target=gate,
                logits_extractor=_make_extractor(gate),
            ))
        else:
            raise ValueError(f"Cannot determine gate structure for {type(gate).__name__}")
    return layers


def load_moe_model(checkpoint_dir, device="cuda"):
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "moe_meta.json"

    if meta_path.exists():
        return _load_custom_moe(checkpoint_dir, device)
    return _load_hf_moe(str(checkpoint_dir), device)


def _load_hf_moe(model_id, device="cuda"):
    print(f"Loading HF native MoE model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    moe_layers = _find_hf_moe_layers(model)
    if not moe_layers:
        raise ValueError(f"No MoE layers found in {model_id}")
    model._moe_layers = moe_layers
    print(f"Found {len(moe_layers)} MoE layers, "
          f"{moe_layers[0].config.num_experts} experts each")
    model.to(device).eval()
    return model, tokenizer, "hf_native"


def _load_custom_moe(checkpoint_dir, device="cuda"):
    meta = json.loads((checkpoint_dir / "moe_meta.json").read_text())

    print(f"Base model: {meta['base_model']}")
    print(f"MoE type:   {meta['moe_type']}, config: {meta['moe_config']}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        meta["base_model"], torch_dtype=torch.bfloat16,
    )

    if meta["moe_type"] == "temporal":
        moe_config = TemporalMoEConfig(**meta["moe_config"])
        TemporalMoEMixin.apply(model, moe_config)
    elif meta["moe_type"] == "vanilla":
        moe_config = VanillaMoEConfig(**meta["moe_config"])
        VanillaMoEMixin.apply(model, moe_config)

    state_dict = {}
    safetensors_files = sorted(checkpoint_dir.glob("*.safetensors"))
    bin_files = sorted(checkpoint_dir.glob("*.bin"))
    if safetensors_files:
        from safetensors.torch import load_file
        for f in safetensors_files:
            state_dict.update(load_file(f, device="cpu"))
    elif bin_files:
        for f in bin_files:
            state_dict.update(torch.load(f, map_location="cpu", weights_only=True))
    else:
        raise FileNotFoundError(f"No model weights in {checkpoint_dir}")

    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, tokenizer, meta["moe_type"]


@torch.no_grad()
def extract_routing_vectors(model, tokenizer, texts, max_len, batch_size,
                            device, moe_type):
    num_layers = len(model._moe_layers)
    num_experts = model._moe_layers[0].config.num_experts

    accumulators = {i: [] for i in range(num_layers)}

    gate_outputs = {}
    hooks = []
    for layer_idx, moe_layer in enumerate(model._moe_layers):
        extractor = getattr(moe_layer, "logits_extractor", None)
        def _make_hook(idx, ext=extractor):
            def _hook(module, inp, out):
                gate_outputs[idx] = ext(inp, out) if ext else out
            return _hook
        hooks.append(
            moe_layer.router.gate.register_forward_hook(_make_hook(layer_idx))
        )

    for start in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[start:start + batch_size]
        enc = tokenizer(
            batch_texts, truncation=True, max_length=max_len,
            padding=True, return_tensors="pt",
        ).to(device)

        model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

        mask = enc["attention_mask"]  # (B, L)
        B, L = mask.shape

        for layer_idx in range(num_layers):
            logits = gate_outputs[layer_idx]  # (B, L, E) temporal | (B*L, E) vanilla
            if logits.dim() == 2:
                logits = logits.reshape(B, L, -1)

            probs = F.softmax(logits.float(), dim=-1)  # (B, L, E)

            mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
            seq_lens = mask.sum(dim=1).float()    # (B,)
            avg = (probs * mask_f).sum(dim=1) / seq_lens.unsqueeze(-1)  # (B, E)
            accumulators[layer_idx].append(avg.cpu())

    for h in hooks:
        h.remove()

    return {i: torch.cat(vecs, dim=0) for i, vecs in accumulators.items()}


def _register_gate_hooks(model):
    gate_outputs = {}
    hooks = []
    for layer_idx, moe_layer in enumerate(model._moe_layers):
        extractor = getattr(moe_layer, "logits_extractor", None)
        def _make_hook(idx, ext=extractor):
            def _hook(module, inp, out):
                gate_outputs[idx] = ext(inp, out) if ext else out
            return _hook
        hooks.append(
            moe_layer.router.gate.register_forward_hook(_make_hook(layer_idx))
        )
    return gate_outputs, hooks


@torch.no_grad()
def extract_eci_data(model, tokenizer, dataset_rows, max_len, batch_size,
                     device, num_viz_samples=20):
    num_layers = len(model._moe_layers)
    num_experts = model._moe_layers[0].config.num_experts
    num_domains = len(DOMAINS)
    num_samples = len(dataset_rows)

    eci_data = torch.zeros(num_samples, num_domains, num_layers, num_experts)
    domain_token_counts = torch.zeros(num_samples, num_domains, dtype=torch.long)

    actual_viz = min(num_viz_samples, num_samples)
    viz_routing_probs = torch.zeros(actual_viz, max_len, num_layers, num_experts)
    viz_domain_masks = torch.full((actual_viz, max_len), -1, dtype=torch.long)
    viz_attention_masks = torch.zeros(actual_viz, max_len, dtype=torch.long)

    gate_outputs, hooks = _register_gate_hooks(model)

    texts = [messages_to_text(row["messages"]) for row in dataset_rows]
    all_messages = [row["messages"] for row in dataset_rows]

    for start in tqdm(range(0, num_samples, batch_size), desc="Extracting ECI"):
        end = min(start + batch_size, num_samples)
        batch_texts = texts[start:end]
        batch_messages = all_messages[start:end]
        B_actual = end - start

        enc = tokenizer(
            batch_texts, truncation=True, max_length=max_len,
            padding="max_length", return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = enc.pop("offset_mapping").tolist()  # (B, L, 2)
        enc = enc.to(device)

        batch_domain_masks = []
        for b in range(B_actual):
            spans = build_domain_char_spans(batch_messages[b])
            dm = tokens_to_domain_mask(offset_mapping[b], spans, max_len)
            batch_domain_masks.append(dm)
        batch_domain_masks = torch.stack(batch_domain_masks)

        model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

        attn_mask = enc["attention_mask"].cpu()

        for layer_idx in range(num_layers):
            logits = gate_outputs[layer_idx]
            if logits.dim() == 2:
                logits = logits.reshape(B_actual, max_len, num_experts)
            probs = F.softmax(logits.float(), dim=-1).cpu()

            for b in range(B_actual):
                sample_idx = start + b

                if sample_idx < actual_viz:
                    viz_routing_probs[sample_idx, :, layer_idx, :] = probs[b]

                for d in range(num_domains):
                    domain_mask = (batch_domain_masks[b] == d) & attn_mask[b].bool()
                    count = domain_mask.sum().item()
                    if count > 0:
                        eci_data[sample_idx, d, layer_idx, :] = probs[b, domain_mask].mean(dim=0)
                        if layer_idx == 0:
                            domain_token_counts[sample_idx, d] = count

        for b in range(B_actual):
            sample_idx = start + b
            if sample_idx < actual_viz:
                viz_domain_masks[sample_idx] = batch_domain_masks[b]
                viz_attention_masks[sample_idx] = attn_mask[b]

    for h in hooks:
        h.remove()

    return {
        "meta": {
            "num_layers": num_layers,
            "num_experts": num_experts,
            "num_domains": num_domains,
            "domain_names": DOMAINS,
            "num_samples": num_samples,
            "max_len": max_len,
            "num_viz_samples": actual_viz,
        },
        "eci_data": eci_data,
        "domain_token_counts": domain_token_counts,
        "sample_indices": torch.arange(num_samples),
        "viz_routing_probs": viz_routing_probs,
        "viz_domain_masks": viz_domain_masks,
        "viz_attention_masks": viz_attention_masks,
        "viz_sample_indices": torch.arange(actual_viz),
    }


def _parallel_extract(num_gpus, args, data):
    model, tokenizer, moe_type = load_moe_model(args.checkpoint_dir, "cpu")

    models = []
    for i in range(num_gpus):
        m = copy.deepcopy(model).to(f"cuda:{i}")
        if moe_type == "hf_native":
            m._moe_layers = _find_hf_moe_layers(m)
        models.append(m)
    del model

    shard_size = math.ceil(len(data) / num_gpus)
    shards = [
        data[i * shard_size : min((i + 1) * shard_size, len(data))]
        for i in range(num_gpus)
    ]

    def _worker(rank):
        device = f"cuda:{rank}"
        shard = shards[rank]
        if not shard:
            return None
        if args.mode == "routing":
            return extract_routing_vectors(
                models[rank], tokenizer, shard, args.max_len,
                args.batch_size, device, moe_type,
            )
        viz = args.num_viz_samples if rank == 0 else 0
        return extract_eci_data(
            models[rank], tokenizer, shard, args.max_len,
            args.batch_size, device, num_viz_samples=viz,
        )

    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = {pool.submit(_worker, i): i for i in range(num_gpus)}
        partials = [None] * num_gpus
        for fut in as_completed(futures):
            partials[futures[fut]] = fut.result()

    partials = [p for p in partials if p is not None]

    if args.mode == "routing":
        merged = {}
        for p in partials:
            for layer_idx, vecs in p.items():
                merged.setdefault(layer_idx, []).append(vecs)
        return {k: torch.cat(v, dim=0) for k, v in merged.items()}

    eci_data = torch.cat([p["eci_data"] for p in partials], dim=0)
    domain_token_counts = torch.cat(
        [p["domain_token_counts"] for p in partials], dim=0,
    )
    total_samples = eci_data.shape[0]
    meta = partials[0]["meta"].copy()
    meta["num_samples"] = total_samples
    return {
        "meta": meta,
        "eci_data": eci_data,
        "domain_token_counts": domain_token_counts,
        "sample_indices": torch.arange(total_samples),
        "viz_routing_probs": partials[0]["viz_routing_probs"],
        "viz_domain_masks": partials[0]["viz_domain_masks"],
        "viz_attention_masks": partials[0]["viz_attention_masks"],
        "viz_sample_indices": partials[0]["viz_sample_indices"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-sample routing vectors from an MoE checkpoint",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-dir", default="data/nemotron-moe-exam")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None,
                        help="Output .pt path (default: <output-dir>/<mode>_vectors.pt)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output file (default: checkpoint dir or cwd)")
    parser.add_argument("--mode", choices=["routing", "eci"], default="routing",
                        help="'routing': sequence-averaged vectors; "
                             "'eci': per-domain routing for ECI analysis")
    parser.add_argument("--num-viz-samples", type=int, default=20,
                        help="(eci mode) number of samples to store full "
                             "per-token routing probs for visualization")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs for parallel extraction "
                             "(default: all available)")
    args = parser.parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()

    ds = load_from_disk(args.dataset_dir)["test"]
    print(f"Test set: {len(ds)} samples")

    if args.mode == "routing":
        data = [messages_to_text(row["messages"]) for row in ds]
    else:
        data = list(ds)

    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for parallel extraction")
        result = _parallel_extract(num_gpus, args, data)
    else:
        model, tokenizer, moe_type = load_moe_model(
            args.checkpoint_dir, args.device,
        )
        if args.mode == "routing":
            result = extract_routing_vectors(
                model, tokenizer, data, args.max_len, args.batch_size,
                args.device, moe_type,
            )
        else:
            result = extract_eci_data(
                model, tokenizer, data, args.max_len, args.batch_size,
                args.device, num_viz_samples=args.num_viz_samples,
            )

    if args.mode == "routing":
        num_layers = len(result)
        num_experts = result[0].shape[1]
        num_samples = result[0].shape[0]
        print(f"\nCollected: {num_layers} layers x {num_samples} samples x "
              f"{num_experts} experts")
        default_name = "routing_vectors.pt"
    else:
        meta = result["meta"]
        print(f"\nCollected ECI data: {meta['num_layers']} layers x "
              f"{meta['num_samples']} samples x {meta['num_domains']} domains x "
              f"{meta['num_experts']} experts")
        print(f"Viz samples: {meta['num_viz_samples']}")
        default_name = "eci_routing_data.pt"

    if args.output:
        out_path = args.output
    else:
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            ckpt_dir = Path(args.checkpoint_dir)
            out_dir = ckpt_dir if ckpt_dir.is_dir() else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / default_name)
    torch.save(result, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
