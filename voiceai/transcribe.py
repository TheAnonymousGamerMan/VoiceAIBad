"""
Transcribe speech to text -- from your live microphone by default, or from
an existing audio file -- tagging each result with who the model thinks was
speaking (z / q / e), and saving the result to a .txt file.

By default this loads the model ONCE and then keeps running: press a key
to record, press a key to stop, get your transcription, repeat -- no need
to re-run the script for every transcription. Press 'q' at the start
prompt to quit.

Run:
    python transcribe.py                      # persistent session, keypress start/stop
    python transcribe.py --once                # record + transcribe once, then exit
    python transcribe.py --seconds 6            # fixed 6s recordings instead of keypress
    python transcribe.py --file clip.wav         # transcribe one file, then exit
    python transcribe.py --output note.txt        # always overwrite this file with the latest result
    python transcribe.py --list-devices           # show every mic sounddevice can see
    python transcribe.py --input-device 3          # force mic index 3 instead of the system default
    python transcribe.py --input-device "USB Mic"   # or match by name/substring

Every transcription (from any mode) is also appended, timestamped, to
transcripts.txt in the project root, so history is never lost.

Defaults to checkpoints/voice_model_best.pt (lowest validation loss during
training) rather than the final epoch's checkpoint, since the final one is
usually more overfit. Pass --checkpoint to use a different one.

Note: speaker identification is only as good as the data it was trained on
-- until clips for 'q' and 'e' exist and the model has been retrained on
them, it will mostly just predict 'z' (or guess weakly).

Note on --input-device: sounddevice normally records from whichever device
Windows currently calls the "default" input. Software like
VoiceMeeter/OBS/Discord can silently change that default to a virtual
cable device, which may not carry your real voice at all -- if live
transcriptions come out far worse than file transcriptions of the same
sentence, run --list-devices and check what the default actually is.
"""
import argparse
from datetime import datetime
from pathlib import Path

import torch

from features import compute_log_mel
from model import CTCSpeechModel, CharVocab, SpeakerVocab
from keypress import read_key
from record import record_from_mic, record_until_keypress, load_audio_file, list_input_devices, parse_device_arg

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE.parent / "checkpoints" / "voice_model_best.pt"
FALLBACK_CHECKPOINT = HERE.parent / "checkpoints" / "voice_model.pt"  # used if no _best.pt exists yet
TRANSCRIPT_LOG = HERE.parent / "transcripts.txt"


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = CharVocab(chars=checkpoint["chars"])
    speaker_vocab = SpeakerVocab(speakers=checkpoint.get("speakers", SpeakerVocab().speakers))

    model = CTCSpeechModel(
        n_mels=checkpoint["n_mels"],
        vocab_size=vocab.vocab_size(),
        num_speakers=speaker_vocab.num_speakers(),
        embed_dim=checkpoint["embed_dim"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)
    return model, vocab, speaker_vocab


def transcribe_waveform(model, vocab, speaker_vocab, waveform, sample_rate, device):
    feat = compute_log_mel(waveform, sample_rate=sample_rate)
    feat = torch.from_numpy(feat).float().unsqueeze(0).to(device)  # [1, T, n_mels]
    lengths = torch.tensor([feat.shape[1]], dtype=torch.long)

    with torch.no_grad():
        log_probs, _, speaker_logits = model(feat, lengths)  # [T, 1, vocab], [1, num_speakers]

    ids = log_probs.squeeze(1).argmax(dim=-1).tolist()  # [T]
    text = vocab.decode_greedy(ids)

    speaker_probs = torch.softmax(speaker_logits.squeeze(0), dim=-1)
    speaker_id = int(speaker_probs.argmax().item())
    speaker_label = speaker_vocab.decode(speaker_id)
    confidence = float(speaker_probs[speaker_id].item())

    return text, speaker_label, confidence


def log_and_save(text, speaker_label, confidence, source, output_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TRANSCRIPT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"[{timestamp}] ({source}) speaker={speaker_label} ({confidence:.0%}) {text}\n"
        )

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(f"[{speaker_label}] {text}", encoding="utf-8")  # latest only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--file", type=str, default=None,
                         help="Transcribe this one audio file, then exit.")
    parser.add_argument("--seconds", type=float, default=None,
                         help="Record fixed-length clips instead of press-a-key start/stop.")
    parser.add_argument("--mic", type=float, default=None,
                         help="Deprecated alias for --seconds; still works the same way.")
    parser.add_argument("--once", action="store_true",
                         help="Record/transcribe a single time then exit, instead of looping.")
    parser.add_argument("--output", "-o", type=str, default=None,
                         help="Always overwrite this .txt file with the latest transcription.")
    parser.add_argument("--input-device", type=str, default=None,
                         help="Mic to record from: an index or a name/substring from --list-devices. "
                                "Defaults to whatever Windows currently calls the default input device.")
    parser.add_argument("--list-devices", action="store_true",
                         help="Print every audio device sounddevice can see, then exit.")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    input_device = parse_device_arg(args.input_device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists() and checkpoint_path == DEFAULT_CHECKPOINT and FALLBACK_CHECKPOINT.exists():
        print(f"No {DEFAULT_CHECKPOINT.name} yet (older train.py run?) -- falling back to {FALLBACK_CHECKPOINT.name}.")
        checkpoint_path = FALLBACK_CHECKPOINT
    if not checkpoint_path.exists():
        print(
            f"No trained model found at {checkpoint_path}.\n"
            "Run collect_data.py to record training clips, then train.py."
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")
    model, vocab, speaker_vocab = load_model(checkpoint_path, device)
    print("Model loaded.\n")

    fixed_duration = args.mic if args.mic is not None else args.seconds

    # --file: one-shot, no live loop.
    if args.file:
        waveform, sr = load_audio_file(args.file)
        text, speaker_label, confidence = transcribe_waveform(
            model, vocab, speaker_vocab, waveform, sr, device
        )
        print(f"Transcription: [{speaker_label}] {text}  ({confidence:.0%} confident)")
        log_and_save(text, speaker_label, confidence, source=args.file, output_path=args.output)
        return

    # Live mic: loads the model once, then keeps transcribing until you quit.
    while True:
        if fixed_duration is not None:
            key = read_key(
                f"Press any key to record {fixed_duration:.1f}s (or 'q' to quit): "
            )
            if key.lower() == "q":
                break
            waveform = record_from_mic(duration_sec=fixed_duration, device=input_device)
            source = f"microphone ({fixed_duration:.1f}s)"
        else:
            key = read_key("Press any key to start recording (or 'q' to quit): ")
            if key.lower() == "q":
                break
            waveform = record_until_keypress(stop_prompt="Recording... press any key to stop.", device=input_device)
            source = "microphone (keypress)"

        if waveform.size == 0:
            print("No audio captured.\n")
            continue

        text, speaker_label, confidence = transcribe_waveform(
            model, vocab, speaker_vocab, waveform, 16000, device
        )
        print(f"Transcription: [{speaker_label}] {text}  ({confidence:.0%} confident)\n")
        log_and_save(text, speaker_label, confidence, source=source, output_path=args.output)

        if args.once:
            break

    print("Session ended.")


if __name__ == "__main__":
    main()
