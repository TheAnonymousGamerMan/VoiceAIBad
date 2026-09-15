"""
Manifest-based dataset for training the CTC + speaker-id model.

Manifest format: data/manifest.jsonl, one JSON object per line:
    {"audio": "clips/clip_0001.wav", "text": "hello there", "speaker": "z"}

"speaker" defaults to "z" if a line doesn't have one (keeps older manifest
entries, recorded before speaker tagging existed, working unchanged).

Audio paths are relative to the manifest file's own directory.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from features import compute_log_mel, spec_augment
from record import load_audio_file
from model import CharVocab, SpeakerVocab


class ManifestDataset(Dataset):
    def __init__(self, manifest_path, vocab=None, speaker_vocab=None, augment=False):
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.vocab = vocab or CharVocab()
        self.speaker_vocab = speaker_vocab or SpeakerVocab()
        self.augment = augment  # SpecAugment -- only ever True for a training split

        self.entries = []
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.entries.append(json.loads(line))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        audio_path = self.base_dir / entry["audio"]

        waveform, sr = load_audio_file(str(audio_path))
        feat = compute_log_mel(waveform, sample_rate=sr)

        if self.augment:
            feat = spec_augment(feat)

        feat = torch.from_numpy(feat).float()

        target = torch.tensor(self.vocab.encode(entry["text"]), dtype=torch.long)
        speaker_id = torch.tensor(
            self.speaker_vocab.encode(entry.get("speaker", "z")), dtype=torch.long
        )
        return feat, target, speaker_id


def collate_fn(batch):
    feats, targets, speaker_ids = zip(*batch)

    input_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    max_t = int(input_lengths.max().item())
    n_mels = feats[0].shape[1]

    padded = torch.zeros(len(feats), max_t, n_mels, dtype=torch.float32)
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f

    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_cat = (
        torch.cat(targets) if any(len(t) > 0 for t in targets) else torch.zeros(0, dtype=torch.long)
    )

    speaker_ids = torch.stack(speaker_ids)

    return padded, input_lengths, targets_cat, target_lengths, speaker_ids
