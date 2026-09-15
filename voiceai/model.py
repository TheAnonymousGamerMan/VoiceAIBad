"""
A from-scratch CTC speech recognizer, with a speaker-identification head.

No pretrained weights, no Whisper, no cloud API -- just matrix multiplications:
a linear projection of mel features, a stack of self-attention + feedforward
transformer blocks (all matmuls), then two output heads sharing that same
encoder:
  - a per-frame character head, trained with CTC loss (the transcription)
  - a whole-clip speaker head, trained with cross-entropy (who's talking)

Speakers are a fixed, known set (z / q / e) so the checkpoint's shape never
has to change as more speakers' data gets added later -- untrained speakers
just start out with a weak, low-confidence signal until they have data too.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BLANK_TOKEN = "<blank>"
CHARS = [BLANK_TOKEN] + list(" abcdefghijklmnopqrstuvwxyz'")
MAX_FRAMES = 3000  # ~30s of audio at a 10ms hop

# Known speakers. 'z' is you; 'q' and 'e' are reserved for the voices you'll
# add later. Fixed on purpose so the model's shape doesn't need to change
# when their data shows up -- just retrain.
SPEAKERS = ["z", "q", "e"]


class CharVocab:
    def __init__(self, chars=CHARS):
        self.chars = chars
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.blank_id = self.char_to_id[BLANK_TOKEN]

    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        text = text.lower()
        ids = []
        for ch in text:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
            # silently drop characters outside the vocab (punctuation etc.)
        return ids

    def decode_greedy(self, ids):
        """Collapse repeats and drop blanks -- standard CTC greedy decoding."""
        chars = []
        prev = None
        for i in ids:
            if i != prev:
                if i != self.blank_id:
                    chars.append(self.chars[i])
            prev = i
        return "".join(chars)


class SpeakerVocab:
    def __init__(self, speakers=SPEAKERS):
        self.speakers = list(speakers)
        self.speaker_to_id = {s: i for i, s in enumerate(self.speakers)}

    def num_speakers(self):
        return len(self.speakers)

    def encode(self, speaker):
        speaker = str(speaker).strip().lower()
        if speaker not in self.speaker_to_id:
            raise ValueError(
                f"Unknown speaker '{speaker}'. Known speakers: {self.speakers} "
                "(edit SPEAKERS in model.py to add more)."
            )
        return self.speaker_to_id[speaker]

    def decode(self, speaker_id):
        return self.speakers[speaker_id]


class CTCSpeechModel(nn.Module):
    def __init__(self, n_mels=40, vocab_size=len(CHARS), num_speakers=len(SPEAKERS),
                 embed_dim=256, n_heads=4, n_layers=4, ff_dim=512, dropout=0.1,
                 max_frames=MAX_FRAMES):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, embed_dim)
        self.pos_embedding = nn.Embedding(max_frames, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, vocab_size)             # per-frame chars (CTC)
        self.speaker_head = nn.Linear(embed_dim, num_speakers)     # whole-clip speaker id

        self.max_frames = max_frames
        self.num_speakers = num_speakers
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, lengths):
        """
        x: [B, T, n_mels] padded log-mel features
        lengths: [B] real (unpadded) frame counts

        returns:
          log_probs      [T, B, vocab_size]   (time-major, for CTCLoss)
          out_lengths     [B]                  (unchanged -- no subsampling)
          speaker_logits   [B, num_speakers]     (whole-clip speaker classification)
        """
        B, T, _ = x.shape
        if T > self.max_frames:
            raise ValueError(f"Clip too long: {T} frames > max_frames={self.max_frames}")

        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.input_proj(x) + self.pos_embedding(positions)

        time_idx = torch.arange(T, device=x.device).unsqueeze(0)
        lengths = lengths.to(x.device)
        key_padding_mask = time_idx >= lengths.unsqueeze(1)  # True = pad

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.ln_f(h)

        logits = self.head(h)  # [B, T, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, B, vocab_size]

        # Mean-pool the encoder output over real (non-padded) frames, then
        # classify who's speaking from that pooled representation.
        valid_mask = (~key_padding_mask).unsqueeze(-1).float()  # [B, T, 1]
        pooled = (h * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)  # [B, E]
        speaker_logits = self.speaker_head(pooled)  # [B, num_speakers]

        return log_probs, lengths, speaker_logits
