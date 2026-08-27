import os
import sys
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.tokenizer import CodeTokenizer
from transformer.model import RosettaTransformer
from preprocessing.dataset_loader import load_humaneval_x_directory


class ParallelCodeDataset(Dataset):
    """Dataset of (source_code, source_lang, target_code, target_lang) pairs."""

    def __init__(
        self,
        pairs: List[Tuple[str, str, str, str]],
        tokenizer: CodeTokenizer,
        max_length: int = 256
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_code, src_lang, tgt_code, tgt_lang = self.pairs[idx]

        # Source input includes source language token
        src_ids = self.tokenizer.encode(
            src_code,
            language=src_lang,
            max_length=self.max_length
        )

        # Target input starts with target language token
        tgt_ids = self.tokenizer.encode(
            tgt_code,
            language=tgt_lang,
            max_length=self.max_length
        )

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long)
        }


def collate_fn(batch, pad_id: int):
    """Pad sequences to longest length in batch."""
    src_list = [item["src"] for item in batch]
    tgt_list = [item["tgt"] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_list,
        batch_first=True,
        padding_value=pad_id
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_list,
        batch_first=True,
        padding_value=pad_id
    )

    return src_padded, tgt_padded


def load_parallel_pairs(dataset_paths: List[str]) -> List[Tuple[str, str, str, str]]:
    """Extract all cross-lingual parallel pairs from JSON files and HumanEval-X."""
    pairs = []
    languages = ["python", "java", "cpp", "javascript"]

    # 1. Load from JSON files
    for path in dataset_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                for src_lang in languages:
                    if src_lang not in item or not item[src_lang]:
                        continue
                    for tgt_lang in languages:
                        if src_lang == tgt_lang or tgt_lang not in item or not item[tgt_lang]:
                            continue
                        pairs.append((
                            item[src_lang].strip(),
                            src_lang,
                            item[tgt_lang].strip(),
                            tgt_lang
                        ))
        except Exception as e:
            print(f"Notice loading {path}: {e}")

    # 2. Load from HumanEval-X directory
    humaneval_dir = "datasets/humaneval_x"
    if os.path.exists(humaneval_dir):
        hx_items = load_humaneval_x_directory(humaneval_dir)
        for item in hx_items:
            for src_lang in languages:
                if src_lang not in item or not item[src_lang]:
                    continue
                for tgt_lang in languages:
                    if src_lang == tgt_lang or tgt_lang not in item or not item[tgt_lang]:
                        continue
                    pairs.append((
                        item[src_lang].strip(),
                        src_lang,
                        item[tgt_lang].strip(),
                        tgt_lang
                    ))

    return pairs


def train_transformer(
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 5e-4,
    save_path: str = "transformer/rosetta_transformer.pth",
    vocab_path: str = "transformer/vocab.json"
):
    print("=" * 60)
    print("STARTING ROSETTA TRANSFORMER TRAINING (HUMANEVAL-X DATASET)")
    print("=" * 60)

    dataset_paths = [
        "datasets/parallel/algorithms.json",
        "datasets/algorithms/multilingual_algorithms.json"
    ]

    pairs = load_parallel_pairs(dataset_paths)
    print(f"Total parallel cross-lingual pairs: {len(pairs)}")

    if not pairs:
        raise ValueError("No parallel code pairs found in datasets!")

    # Build tokenizer vocabulary
    all_code_samples = []
    for src, _, tgt, _ in pairs:
        all_code_samples.extend([src, tgt])

    tokenizer = CodeTokenizer()
    tokenizer.build_vocab(all_code_samples, min_freq=1, max_vocab_size=12000)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Save vocab
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.vocab, f, indent=2)

    # Train / Validation Split
    random.seed(42)
    random.shuffle(pairs)
    split_idx = max(1, int(0.85 * len(pairs)))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:] if split_idx < len(pairs) else pairs

    train_dataset = ParallelCodeDataset(train_pairs, tokenizer)
    val_dataset = ParallelCodeDataset(val_pairs, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RosettaTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        pad_idx=tokenizer.pad_id
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Target input is tgt[:, :-1], expected target output is tgt[:, 1:]
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(
                output.reshape(-1, output.shape[-1]),
                tgt_expected.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]

                output = model(src, tgt_input)
                loss = criterion(
                    output.reshape(-1, output.shape[-1]),
                    tgt_expected.reshape(-1)
                )
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch {epoch + 1:2d}/{epochs:2d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Best Val: {best_val_loss:.4f}",
            flush=True
        )

    print()
    print("Transformer training finished successfully.", flush=True)
    print(f"Model saved to: {save_path}", flush=True)
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    train_transformer(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
