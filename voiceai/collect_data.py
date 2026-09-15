"""
Record your own training data: clips of your voice + who's talking.

Prompts you to pick a speaker (z / q / e) with a single keypress before you
start reading. Shows you a sentence from data/reference_sentences.txt to
read out loud, and uses that exact text as the transcript -- you never have
to type anything. Press any key to start recording, press any key again to
stop.

At the start-of-turn prompt: any key = start recording, 's' = skip this
sentence without recording it, 'q' = quit.

Run:
    python collect_data.py                 # asks you which speaker before starting
    python collect_data.py --speaker q       # skips the prompt, tags clips as 'q'
    python collect_data.py --list-devices     # show every mic sounddevice can see
    python collect_data.py --input-device 3    # force mic index 3 instead of the system default

Known speakers are z / q / e (see SPEAKERS in model.py). Run this once per
speaker/session -- e.g. pick z for your own voice, then run it again later
and pick q once that voice is ready to record.

If data/reference_sentences.txt is missing, it falls back to asking you to
type what you said.

Note on --input-device: sounddevice normally records from whichever device
Windows currently calls the "default" input. Software like
VoiceMeeter/OBS/Discord can silently change that default to a virtual
cable device, which may not carry your real voice at all -- if you're not
sure, run --list-devices and check what the default actually is before
collecting a bunch of clips through the wrong device.
"""
import argparse
import json
from pathlib import Path

from keypress import read_key
from model import SpeakerVocab
from record import record_until_keypress, save_wav, list_input_devices, parse_device_arg
from sentence_bank import load_sentences, peek_sentence, advance, set_progress

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
CLIPS_DIR = DATA_DIR / "clips"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"


def next_clip_index():
    existing = list(CLIPS_DIR.glob("clip_*.wav"))
    if not existing:
        return 1
    nums = [int(p.stem.split("_")[1]) for p in existing]
    return max(nums) + 1


def choose_speaker():
    """Lets you pick a speaker with a single keypress -- either the number
    shown next to it or the letter itself."""
    speakers = SpeakerVocab().speakers  # stays in sync with SPEAKERS in model.py

    print("Who's talking?")
    for i, s in enumerate(speakers, start=1):
        print(f"  [{i}] {s}")

    while True:
        key = read_key("Press a number (or the letter) to choose: ").strip().lower()
        if key in speakers:
            return key
        if key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(speakers):
                return speakers[idx]
        print(f"'{key}' isn't a valid choice -- try again.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", type=str, default=None,
                         help="Speaker label for every clip recorded this run (z / q / e). "
                                "If omitted, you'll be prompted to choose one at startup.")
    parser.add_argument("--input-device", type=str, default=None,
                         help="Mic to record from: an index or a name/substring from --list-devices. "
                                "Defaults to whatever Windows currently calls the default input device.")
    parser.add_argument("--list-devices", action="store_true",
                         help="Print every audio device sounddevice can see, then exit.")
    parser.add_argument("--start-at", type=int, default=None,
                         help="Jump this speaker's reading position to sentence number N "
                                "(1-indexed, matching the numbers in reference_sentences.txt) "
                                "before starting, instead of continuing from where you left off. "
                                "Useful for prioritizing a newly-added batch of sentences.")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    input_device = parse_device_arg(args.input_device)

    if args.speaker:
        speaker = args.speaker.strip().lower()
        SpeakerVocab().encode(speaker)  # raises a clear error if it's not a known speaker
    else:
        speaker = choose_speaker()

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences()

    if args.start_at is not None:
        target_idx = args.start_at - 1  # sentence numbers are 1-indexed, progress is 0-indexed
        if sentences and not (0 <= target_idx < len(sentences)):
            print(f"--start-at {args.start_at} is out of range (1-{len(sentences)}); ignoring it.")
        else:
            set_progress(speaker, target_idx)
            print(f"Jumped speaker '{speaker}' to sentence {args.start_at}.")

    print(f"\nVoice data collector -- tagging every clip this run as speaker '{speaker}'.")
    print(f"Saving clips to {DATA_DIR}\n")
    if sentences:
        print(f"Reading from {len(sentences)} reference sentences -- just read what's shown, no typing needed.\n")
    else:
        print("No data/reference_sentences.txt found -- you'll type each transcript manually.\n")

    idx = next_clip_index()
    collected = 0

    with open(MANIFEST_PATH, "a", encoding="utf-8") as manifest_file:
        while True:
            transcript = None
            sentence, sent_idx, wrapped = (None, None, False)

            if sentences:
                sentence, sent_idx, wrapped = peek_sentence(speaker, sentences)
                if wrapped:
                    print("(Back to the start of the list -- extra reps of the same lines help too.)")
                print(f'\nRead this out loud:\n    "{sentence}"\n')

                key = read_key(
                    f"[{collected} collected] Press any key to start recording "
                    "('s' to skip, 'q' to quit): "
                )
                if key.lower() == "q":
                    break
                if key.lower() == "s":
                    advance(speaker, sent_idx)
                    print("Skipped.\n")
                    continue
            else:
                key = read_key(
                    f"[{collected} collected] Press any key to start recording (or 'q' to quit): "
                )
                if key.lower() == "q":
                    break

            waveform = record_until_keypress(stop_prompt="Recording... press any key to stop.", device=input_device)

            if waveform.size == 0:
                print("No audio captured -- try that one again.\n")
                continue  # sentence pointer not advanced -- same sentence comes up again

            if sentences:
                transcript = sentence
            else:
                transcript = input("What did you say? (blank to discard): ").strip()
                if not transcript:
                    print("Discarded.\n")
                    continue

            clip_name = f"clip_{idx:04d}.wav"
            save_wav(CLIPS_DIR / clip_name, waveform)

            manifest_file.write(json.dumps({
                "audio": f"clips/{clip_name}",
                "text": transcript,
                "speaker": speaker,
            }) + "\n")
            manifest_file.flush()

            if sentences:
                advance(speaker, sent_idx)

            idx += 1
            collected += 1
            print(f"Saved {clip_name}: \"{transcript}\"\n")

    print(f"\nDone. Collected {collected} new clip(s) for speaker '{speaker}'. Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
