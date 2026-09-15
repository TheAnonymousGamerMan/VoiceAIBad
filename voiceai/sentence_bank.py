"""
Loads reference sentences from data/reference_sentences.txt and tracks,
per speaker, which one to read next -- so collect_data.py can show you a
sentence instead of you typing one out every time.

Progress is persisted in data/sentence_progress.json so it picks up where
you left off across runs, and wraps back to the start once a speaker has
gone through the whole list (extra repetitions of the same sentences are
still useful training data).
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
SENTENCES_PATH = DATA_DIR / "reference_sentences.txt"
PROGRESS_PATH = DATA_DIR / "sentence_progress.json"

_LINE_RE = re.compile(r"^\d+\.\s+(.*\S)\s*$")


def load_sentences():
    """Parses the numbered sentence lines out of reference_sentences.txt,
    ignoring headers/blank lines. Returns [] if the file doesn't exist."""
    if not SENTENCES_PATH.exists():
        return []

    sentences = []
    with open(SENTENCES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            match = _LINE_RE.match(line.strip())
            if match:
                sentences.append(match.group(1))
    return sentences


def _load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_progress(progress):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def peek_sentence(speaker, sentences):
    """
    Returns (sentence_text, index, wrapped) for the next sentence this
    speaker should read, WITHOUT marking it as used yet -- call advance()
    once it's actually been recorded (or intentionally skipped).
    """
    if not sentences:
        return None, None, False

    progress = _load_progress()
    idx = progress.get(speaker, 0)
    wrapped = idx >= len(sentences)
    if wrapped:
        idx = 0

    return sentences[idx], idx, wrapped


def advance(speaker, idx):
    """Marks that this speaker has now gotten past sentence `idx`."""
    progress = _load_progress()
    progress[speaker] = idx + 1
    _save_progress(progress)


def set_progress(speaker, idx):
    """Explicitly jumps a speaker to a given sentence index (e.g. to
    prioritize reading a newly-added batch of sentences right away instead
    of reaching it naturally after everything before it). Persists
    immediately, same as advance()."""
    progress = _load_progress()
    progress[speaker] = idx
    _save_progress(progress)
