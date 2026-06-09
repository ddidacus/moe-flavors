import argparse
import random

from datasets import Dataset, DatasetDict, load_dataset


def shuffle_and_split(ds, test_size=0.1, seed=42):
    ds = ds.shuffle(seed=seed)
    test_len = int(len(ds) * test_size)
    return {
        "test": ds.select(range(test_len)),
        "train": ds.select(range(test_len, len(ds))),
    }


def extract_turn(messages):
    """Return the first (user, assistant) content pair from a message list."""
    user = assistant = None
    for m in messages:
        if m["role"] == "user" and user is None:
            user = m["content"]
        elif m["role"] == "assistant" and user is not None:
            assistant = m["content"]
            break
    return user, assistant


def build_multi_domain_conversations(domain_splits, split, seed=42):
    """Stack one sample per domain into a single 3-turn conversation.

    Each user turn is prefixed with the domain name so the model can
    distinguish tasks.  Domain order is shuffled per conversation.

    Returns a HF Dataset with a single ``messages`` column.
    """
    rng = random.Random(seed)
    domains = sorted(domain_splits)
    n = min(len(domain_splits[d][split]) for d in domains)

    conversations = []
    for i in range(n):
        order = list(domains)
        rng.shuffle(order)

        conv, valid = [], True
        for domain in order:
            user, assistant = extract_turn(
                domain_splits[domain][split][i]["messages"]
            )
            if user is None or assistant is None:
                valid = False
                break
            conv.append({
                "role": "user",
                "content": f"Here is a question from the {domain} domain.\n\n{user}",
            })
            conv.append({"role": "assistant", "content": assistant})

        if valid:
            conversations.append({"messages": conv})

    return Dataset.from_list(conversations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a multi-domain conversation dataset from Nemotron v2"
    )
    parser.add_argument(
        "--domains", nargs="+", default=["code", "math", "chat"],
        help="Nemotron splits to use as domains",
    )
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-dir", type=str, default="data/multi_domain_conv",
        help="Directory to save the resulting DatasetDict",
    )
    args = parser.parse_args()

    domain_splits = {}
    for domain in args.domains:
        print(f"Loading {domain}...")
        ds = load_dataset(
            "nvidia/Nemotron-Post-Training-Dataset-v2", split=domain
        )
        domain_splits[domain] = shuffle_and_split(
            ds, test_size=args.test_size, seed=args.seed
        )

    result = DatasetDict({
        split: build_multi_domain_conversations(domain_splits, split, seed=args.seed)
        for split in ("train", "test")
    })

    for split in result:
        print(f"{split}: {len(result[split])} conversations")

    result.save_to_disk(args.save_dir)
    print(f"Saved to {args.save_dir}")
