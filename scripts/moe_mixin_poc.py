import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.temporal_moe import MoEConfig as TemporalMoEConfig, MoEMixin as TemporalMoEMixin
from src.vanilla_moe import MoEConfig as VanillaMoEConfig, MoEMixin as VanillaMoEMixin


# ====================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def get_nemotron_loader(tokenizer, seq_len: int, batch_size: int, num_samples: int,
                        split: str = "code"):
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset

    ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2",
                       split=f"{split}[:{num_samples}]", streaming=False)

    def messages_to_text(messages):
        return "\n".join(m["content"] for m in messages)

    class TokenizedDataset(Dataset):
        def __init__(self, rows, tokenizer, seq_len):
            self.tokenizer = tokenizer
            self.seq_len = seq_len
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            text = messages_to_text(self.rows[idx]["messages"])
            tokens = self.tokenizer(
                text, truncation=True,
                max_length=self.seq_len + 1, padding="max_length",
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].squeeze(0) # seq_len+1
            return input_ids[:-1], input_ids[1:]       # x, y

    dataset = TokenizedDataset(ds, tokenizer, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

@torch.no_grad()
def eval_boundaries(model, tokenizer, x, step, layer_idx=0):
    """Pick a random sequence from the batch, run layer-0 router, log boundaries to wandb."""
    model.eval()
    seq_idx = random.randint(0, x.shape[0] - 1)
    input_ids = x[seq_idx] # L

    # get hidden states at the target layer
    embeds = model.model.embed_tokens(input_ids.unsqueeze(0)) # 1, L, D
    router = model._moe_layers[layer_idx].router
    _, top_k_indices, pt, bt = router(embeds)

    bt = bt[0].cpu().tolist()           # L bools
    pt_vals = pt[0].cpu().tolist()      # L floats
    expert_ids = top_k_indices[0].cpu().tolist() # L, top_k

    tokens = [tokenizer.decode(tid) for tid in input_ids.cpu().tolist()]

    # build annotated string: | marks boundaries
    segments = []
    current_segment = []
    for i, tok in enumerate(tokens):
        if bt[i] and current_segment:
            segments.append("".join(current_segment))
            current_segment = []
        current_segment.append(tok)
    if current_segment:
        segments.append("".join(current_segment))
    annotated = " | ".join(segments)

    # wandb table with per-token detail
    table = wandb.Table(columns=["pos", "token", "boundary", "p_t", "experts"])
    for i, tok in enumerate(tokens):
        table.add_data(i, tok, int(bt[i]), round(pt_vals[i], 4), str(expert_ids[i]))

    # plot: expert assignments over token positions with boundary lines
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    num_experts = router.num_experts
    top_k = router.top_k
    L = len(tokens)

    fig, (ax_exp, ax_pt) = plt.subplots(2, 1, figsize=(max(14, L * 0.08), 5),
                                         sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # top panel: expert assignment per token, one row per top-k slot
    colors = plt.cm.Set1(np.linspace(0, 1, num_experts))
    for k_idx in range(top_k):
        experts_k = [expert_ids[i][k_idx] for i in range(L)]
        ax_exp.scatter(range(L), [k_idx] * L, c=[colors[e] for e in experts_k],
                       s=12, marker="s")

    # boundary vertical lines
    boundary_positions = [i for i in range(L) if bt[i]]
    for bp in boundary_positions:
        ax_exp.axvline(x=bp, color="red", alpha=0.4, linewidth=0.8, linestyle="--")
        ax_pt.axvline(x=bp, color="red", alpha=0.4, linewidth=0.8, linestyle="--")

    ax_exp.set_yticks(range(top_k))
    ax_exp.set_yticklabels([f"top-{k+1}" for k in range(top_k)])
    ax_exp.set_ylabel("expert slot")
    ax_exp.set_title(f"Expert assignments over tokens (step {step}, layer {layer_idx})")

    # legend for expert colors
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors[e], label=f"expert {e}") for e in range(num_experts)]
    ax_exp.legend(handles=legend_patches, loc="upper right", fontsize=7, ncol=num_experts)

    # bottom panel: boundary probability p_t
    ax_pt.plot(range(L), pt_vals, color="black", linewidth=0.8)
    ax_pt.axhline(y=router._boundary_threshold, color="red", alpha=0.5, linewidth=0.8, linestyle=":")
    ax_pt.set_ylabel("p_t")
    ax_pt.set_xlabel("token position")
    ax_pt.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    wandb.log({
        "eval/segmented_text": wandb.Html(f"<pre>{annotated}</pre>"),
        "eval/boundary_table": table,
        "eval/expert_boundaries": wandb.Image(fig),
        "eval/num_segments": len(segments),
        "eval/boundary_frac": sum(bt) / len(bt),
    }, step=step)
    plt.close(fig)

    print(f"\n--- eval (step {step}, layer {layer_idx}) ---")
    print(annotated[:500])
    print(f"segments: {len(segments)}, boundary_frac: {sum(bt)/len(bt):.3f}")

    model.train()

def _unwrap(model):
    """Get the underlying model from DDP/accelerate wrappers."""
    return model.module if hasattr(model, "module") else model

# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="MoE Mixin PoC")
    parser.add_argument("--moe-type", choices=["vanilla", "temporal"], default="temporal",
                        help="MoE variant: vanilla (standard top-k) or temporal (chunking)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model id")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--ratio-loss-N", type=int, default=3,
                        help="N parameter for temporal ratio loss (target segment length)")
    parser.add_argument("--ratio-loss-alpha", type=float, default=0.3,
                        help="Weight for temporal ratio loss")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dataset-split", default="code",
                        choices=["code", "math", "stem", "chat",
                                 "multilingual_ja", "multilingual_de",
                                 "multilingual_it", "multilingual_es",
                                 "multilingual_fr"],
                        help="Nemotron SFT split to use")
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--num-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="moe-chunking-poc")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    accelerator = Accelerator(log_with="wandb")

    # logging with wandb

    accelerator.init_trackers(args.wandb_project, config=vars(args))
    if accelerator.is_main_process:
        wandb.config.update({"num_gpus": accelerator.num_processes})

    # loading dense LLM model

    accelerator.print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    )
    dense_params = count_params(model)
    accelerator.print(f"Dense model parameters: {dense_params:,}")

    # dense --> MoE

    if args.moe_type == "temporal":
        moe_config = TemporalMoEConfig(
            num_experts=args.num_experts,
            top_k=args.top_k,
            ratio_loss_N=args.ratio_loss_N,
            ratio_loss_alpha=args.ratio_loss_alpha,
        )
        TemporalMoEMixin.apply(model, moe_config)
    else:
        moe_config = VanillaMoEConfig(
            num_experts=args.num_experts,
            top_k=args.top_k,
        )
        VanillaMoEMixin.apply(model, moe_config)
    moe_params = count_params(model)
    accelerator.print(f"MoE model parameters:   {moe_params:,}  "
                      f"({moe_params / dense_params:.1f}x dense)")

    # load dataset
    accelerator.print(f"Loading Nemotron SFT ({args.dataset_split}, {args.num_samples} docs) ...")
    loader = get_nemotron_loader(tokenizer, args.seq_len, args.batch_size, args.num_samples,
                                 split=args.dataset_split)

    # prepare for distributed training

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if accelerator.is_main_process:
        wandb.log({"dense_params": dense_params, "moe_params": moe_params}, step=0)

    # train loop

    model.train()
    raw_model = _unwrap(model)
    num_moe_layers = len(raw_model._moe_layers)
    if args.moe_type == "temporal":
        ratio_loss_alpha = raw_model._moe_layers[0]._ratio_loss_alpha
    else:
        ratio_loss_alpha = None

    step = 0
    for epoch in range(100): # enough epochs to hit num_steps
        pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False,
                    disable=not accelerator.is_main_process)
        for x, y in pbar:
            outputs = model(input_ids=x, labels=y)
            aux_loss = raw_model.get_moe_loss()
            if args.moe_type == "temporal":
                total_loss = outputs.loss
                lm_loss = (total_loss - aux_loss).detach()
            else:
                lm_loss = outputs.loss
                total_loss = lm_loss + aux_loss
                lm_loss = lm_loss.detach()
            aux_loss = aux_loss.detach()

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if accelerator.is_main_process:
                pbar.set_postfix(lm=f"{lm_loss.item():.4f}", aux=f"{aux_loss.item():.4f}",
                                 step=step)
                if step % args.log_every == 0:
                    metrics = {
                        "train/lm_loss": lm_loss.item(),
                        "train/aux_loss": aux_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "train/epoch": epoch,
                    }
                    if ratio_loss_alpha is not None:
                        metrics["train/ratio_loss_per_layer_unscaled"] = (
                            aux_loss.item() / (num_moe_layers * ratio_loss_alpha)
                        )
                    wandb.log(metrics, step=step)

                if step % args.eval_every == 0 and args.moe_type == "temporal":
                    eval_boundaries(raw_model, tokenizer, x, step)

            if step >= args.num_steps:
                break
        if step >= args.num_steps:
            break

    accelerator.print(f"\nTraining done ({step} steps).")

    # final inspection
    if accelerator.is_main_process:
        raw_model.eval()
        prompt = "The mixture of experts architecture"
        inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
        with torch.no_grad():
            out = raw_model.generate(**inputs, max_new_tokens=50, do_sample=False)
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"Prompt:     {prompt!r}")
        print(f"Generation: {gen_text!r}")
        wandb.log({"final/generation": wandb.Html(f"<pre>{gen_text}</pre>")}, step=step)

    accelerator.end_training()


if __name__ == "__main__":
    main()
