"""
Trains the CTC + speaker-id model on data/manifest.jsonl (built with
collect_data.py). Jointly learns to transcribe text (CTC loss) and to tell
speakers apart (cross-entropy over z / q / e).

Holds out a validation slice (never trained on, stratified per speaker) so
you can see a real train-vs-validation gap each epoch, instead of guessing
whether the model is generalizing or just memorizing from spot checks.

Regularization: training clips get SpecAugment (random time/frequency
masking of the spectrogram) so the model can't just memorize exact
acoustic detail; weight decay (AdamW) and dropout add further pressure
against memorizing. Validation clips are never augmented, so validation
loss stays a trustworthy generalization signal. If your train loss keeps
dropping while validation loss stalls or rises, that gap -- not epoch
count -- is what these knobs (and more data) are meant to close.

GPU usage: mixed precision (fp16 autocast + gradient scaling) when a CUDA
GPU is available, pinned memory, and parallel data-loading workers so the
GPU stays fed instead of idling between batches.

Run:
    python train.py
    python train.py --batch_size 64 --epochs 200
    python train.py --weight_decay 0.01 --dropout 0.3
"""
import argparse
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import ManifestDataset, collate_fn
from model import CTCSpeechModel, CharVocab, SpeakerVocab

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE.parent / "data" / "manifest.jsonl"
DEFAULT_CHECKPOINT = HERE.parent / "checkpoints" / "voice_model.pt"
DEFAULT_BEST_CHECKPOINT = HERE.parent / "checkpoints" / "voice_model_best.pt"

MIN_GROUP_SIZE_FOR_VAL = 4  # below this many clips for a speaker, all go to train


def split_dataset(dataset, val_fraction=0.15, seed=42):
    """
    Stratified (per-speaker) train/val split, so a speaker with little data
    doesn't get entirely swallowed by one side, and so validation loss stays
    meaningful once more than one speaker has data. Deterministic for a
    given manifest + seed.
    """
    by_speaker = {}
    for i, entry in enumerate(dataset.entries):
        speaker = entry.get("speaker", "z")
        by_speaker.setdefault(speaker, []).append(i)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []

    for speaker, indices in by_speaker.items():
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)

        if n < MIN_GROUP_SIZE_FOR_VAL:
            train_indices.extend(indices)
            continue

        n_val = max(1, round(n * val_fraction))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    return train_indices, val_indices


def run_validation(model, val_loader, ctc_loss_fn, speaker_loss_fn, device, use_amp):
    model.eval()
    total_text_loss = 0.0
    total_speaker_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for feats, input_lengths, targets, target_lengths, speaker_ids in val_loader:
            feats = feats.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)
            speaker_ids = speaker_ids.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                log_probs, out_lengths, speaker_logits = model(feats, input_lengths)
                text_loss = ctc_loss_fn(log_probs.float(), targets, out_lengths, target_lengths)
                speaker_loss = speaker_loss_fn(speaker_logits, speaker_ids)

            total_text_loss += text_loss.item()
            total_speaker_loss += speaker_loss.item()
            n_batches += 1

    model.train()
    return total_text_loss / max(n_batches, 1), total_speaker_loss / max(n_batches, 1)


def print_gpu_status(prefix=""):
    if not torch.cuda.is_available():
        return
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    print(f"{prefix}GPU: {name} -- {reserved:.2f} GiB / {total:.2f} GiB reserved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--best_checkpoint", type=str, default=str(DEFAULT_BEST_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32,
                         help="Raise this if your GPU has memory headroom -- bigger batches "
                                "keep it busier. Lower it if you hit an out-of-memory error.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01,
                         help="AdamW weight decay -- regularization against overfitting.")
    parser.add_argument("--dropout", type=float, default=0.2,
                         help="Dropout inside the transformer -- higher fights overfitting harder.")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.15,
                         help="Fraction of each speaker's clips to hold out for validation.")
    parser.add_argument("--num_workers", type=int, default=4,
                         help="Parallel data-loading processes. Set to 0 if this causes issues on Windows.")
    parser.add_argument("--no_amp", action="store_true",
                         help="Disable mixed-precision (fp16) training on GPU.")
    parser.add_argument("--no_augment", action="store_true",
                         help="Disable SpecAugment on training clips (on by default).")
    parser.add_argument("--lr_patience", type=int, default=6,
                         help="Epochs with no validation improvement before the LR is halved.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and not args.no_amp
    print(f"Using device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print_gpu_status()
        print(f"Mixed precision (fp16): {'on' if use_amp else 'off'}")
    else:
        print("No CUDA GPU detected -- training on CPU will be much slower.")

    vocab = CharVocab()
    speaker_vocab = SpeakerVocab()

    # Two dataset views of the SAME manifest: the training split gets
    # SpecAugment applied per-clip, the validation split never does, so
    # validation loss stays an honest measure of generalization.
    train_dataset_full = ManifestDataset(
        args.manifest, vocab=vocab, speaker_vocab=speaker_vocab, augment=not args.no_augment
    )
    val_dataset_full = ManifestDataset(
        args.manifest, vocab=vocab, speaker_vocab=speaker_vocab, augment=False
    )

    if len(train_dataset_full) == 0:
        print(
            f"No training data found at {args.manifest}.\n"
            "Run collect_data.py first to record some clips of your voice."
        )
        return

    print(f"Loaded {len(train_dataset_full)} clips from {args.manifest}")
    speaker_counts = {}
    for entry in train_dataset_full.entries:
        s = entry.get("speaker", "z")
        speaker_counts[s] = speaker_counts.get(s, 0) + 1
    print(f"Speaker breakdown: {speaker_counts}")
    print(f"SpecAugment on training clips: {'off' if args.no_augment else 'on'}")

    train_indices, val_indices = split_dataset(train_dataset_full, val_fraction=args.val_split)

    num_workers = max(0, min(args.num_workers, os.cpu_count() or 0))
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(
        Subset(train_dataset_full, train_indices),
        batch_size=min(args.batch_size, len(train_indices)),
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    if val_indices:
        val_loader = DataLoader(
            Subset(val_dataset_full, val_indices),
            batch_size=min(args.batch_size, len(val_indices)),
            shuffle=False,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        print(f"Train/val split: {len(train_indices)} train, {len(val_indices)} val clips")
    else:
        val_loader = None
        print(
            f"Not enough clips per speaker yet for a validation split "
            f"(need >= {MIN_GROUP_SIZE_FOR_VAL} per speaker) -- training on all {len(train_indices)}."
        )
    print(f"Batch size: {train_loader.batch_size}, data-loader workers: {num_workers}\n")

    model = CTCSpeechModel(
        n_mels=40,
        vocab_size=vocab.vocab_size(),
        num_speakers=speaker_vocab.num_speakers(),
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    scheduler = None
    if val_loader is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.lr_patience
        )

    ctc_loss_fn = torch.nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    speaker_loss_fn = torch.nn.CrossEntropyLoss()

    print("Starting training...")
    model.train()
    best_val_text_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        total_text_loss = 0.0
        total_speaker_loss = 0.0
        n_batches = 0

        for feats, input_lengths, targets, target_lengths, speaker_ids in train_loader:
            feats = feats.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)
            speaker_ids = speaker_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                log_probs, out_lengths, speaker_logits = model(feats, input_lengths)
                # CTCLoss wants fp32 log-probs even under autocast, for numerical stability
                text_loss = ctc_loss_fn(log_probs.float(), targets, out_lengths, target_lengths)
                speaker_loss = speaker_loss_fn(speaker_logits, speaker_ids)
                loss = text_loss + speaker_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_text_loss += text_loss.item()
            total_speaker_loss += speaker_loss.item()
            n_batches += 1

        avg_text_loss = total_text_loss / max(n_batches, 1)
        avg_speaker_loss = total_speaker_loss / max(n_batches, 1)

        if val_loader is not None:
            val_text_loss, val_speaker_loss = run_validation(
                model, val_loader, ctc_loss_fn, speaker_loss_fn, device, use_amp
            )
            lr_before = optimizer.param_groups[0]["lr"]
            scheduler.step(val_text_loss)
            lr_after = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{args.epochs} - "
                f"train text: {avg_text_loss:.4f} val text: {val_text_loss:.4f} | "
                f"train speaker: {avg_speaker_loss:.4f} val speaker: {val_speaker_loss:.4f} | "
                f"lr: {lr_after:.2e}"
            )
            if lr_after < lr_before:
                print(f"  -> validation loss plateaued, lr reduced {lr_before:.2e} -> {lr_after:.2e}")

            if val_text_loss < best_val_text_loss:
                best_val_text_loss = val_text_loss
                best_path = Path(args.best_checkpoint)
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "chars": vocab.chars,
                    "speakers": speaker_vocab.speakers,
                    "embed_dim": args.embed_dim,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "n_mels": 40,
                    "val_text_loss": val_text_loss,
                    "epoch": epoch,
                }, best_path)
                print(f"  -> new best val loss, saved {best_path}")
        else:
            print(
                f"Epoch {epoch}/{args.epochs} - "
                f"text loss: {avg_text_loss:.4f} - speaker loss: {avg_speaker_loss:.4f}"
            )

        if device == "cuda" and epoch % 10 == 0:
            print_gpu_status(prefix="  ")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "chars": vocab.chars,
        "speakers": speaker_vocab.speakers,
        "embed_dim": args.embed_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_mels": 40,
    }, checkpoint_path)

    print(f"\nFinal model saved to {checkpoint_path}")
    if val_loader is not None:
        print(f"Best model (by validation loss) saved to {args.best_checkpoint}")
        print("Use --checkpoint on transcribe.py to pick whichever you want to actually use.")


if __name__ == "__main__":
    main()
