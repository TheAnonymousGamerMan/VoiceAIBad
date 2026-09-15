# VoiceAI

A voice-to-text generator built completely from scratch, in the same spirit
as ManiacAI: no Whisper, no cloud speech APIs. Just hand-rolled feature
extraction and a small transformer trained with CTC loss on your own
recorded voice clips -- and it also learns to tell speakers apart.

## How it works

1. `voiceai/features.py` turns raw audio into a log-mel spectrogram: frame
   the waveform, FFT each frame, then one matrix multiplication against a
   hand-built mel filterbank.
2. `voiceai/model.py` is a small transformer (linear layers + self-attention
   -- all matrix multiplication) with two heads sharing the same encoder:
   a per-frame character head trained with CTC loss (the transcription),
   and a whole-clip speaker head trained with cross-entropy (who's talking).
3. You provide the training data yourself, the same way you built
   `training_data.txt` for ManiacAI's text model -- except here it's your
   own voice (and, later, other people's).

## Speakers: z / q / e

Every clip gets tagged with who's talking. Right now only `z` (you) has
any data; `q` and `e` are reserved for the two other voices you'll add
later. The set is fixed in `SPEAKERS` at the top of `model.py` so the
model's shape doesn't need to change once q/e have real data -- just
record clips for them with `--speaker q` / `--speaker e` and retrain.
Until a speaker has real data, the model will mostly guess `z` (or guess
weakly) for them -- that's expected, not a bug.

## Setup

```
cd VoiceAI
pip install -r requirements.txt
```

## 1. Record training data

```
cd voiceai
python collect_data.py
python collect_data.py --list-devices     # show every mic sounddevice can see
python collect_data.py --input-device 3    # force mic index 3 instead of the system default
python collect_data.py --start-at 3762      # jump straight to sentence #3762 (e.g. the new casual batch)
```

It asks you to pick a speaker first -- press `1`/`2`/`3` or the letter
itself (`z`/`q`/`e`) -- then every clip that run gets tagged with your
choice. Skip the prompt with `--speaker q` (or `z` / `e`) if you already
know which one you want, e.g. for scripting.

It shows you a sentence from `data/reference_sentences.txt` (4,020 unique
sentences across many different sentence shapes, all guaranteed different from each other, including a batch of casual everyday first-person sentences -- "I like...", "I don't really like...", "I've been meaning to..." -- added specifically because template-generated sentences like "the musician cleaned the report" were under-representing that register) to read out loud --
press any key to start recording, press any key again to stop, and that
sentence is used as the transcript automatically. You never type anything.
At the prompt: any key starts recording, `s` skips the current sentence,
`q` quits. Each speaker works through the list independently and your
place is remembered across runs (`data/sentence_progress.json`), so at
2,100 sentences you won't hit a repeat for a very long time -- and if a
speaker ever does reach the end, it wraps back to the start rather than
stopping, since extra repetitions of the same lines are still useful
training data. Add your own lines to `reference_sentences.txt` any time
you want more. If that file is missing, it falls back to asking you to
type what you said instead.

By default it records from whatever Windows currently calls the default input device -- if that's ever a virtual/routed device (VoiceMeeter, OBS, a Discord virtual cable, etc.) instead of your actual mic, your clips can end up not capturing your real voice at all. Run `python collect_data.py --list-devices` to see everything sounddevice can see and pick the right one with `--input-device`.

Clips are saved to `data/clips/` and logged to `data/manifest.jsonl` along
with the speaker tag. Run it as many times as you like across multiple
sessions -- it appends, it doesn't overwrite. The more (and more varied)
clips you record, the better the model will get; expect to need at least a
few hundred short clips before it starts producing recognizable text, same
as the text model needed a sizeable `training_data.txt`.

## 2. Train

```
python train.py
```

Uses your GPU properly if you have an NVIDIA one: mixed-precision (fp16)
training, pinned memory, and parallel data-loading workers so the GPU
stays fed instead of idling between batches -- it'll print your GPU name
and how much of its memory is reserved so you can confirm it's actually
being used (Windows Task Manager's Performance > GPU tab, or `nvidia-smi`
in another terminal, will also show utilization climb while it runs).
Default batch size is 32 now that you have real data volume -- raise it
with `--batch_size` if your GPU has memory headroom, lower it if you hit
an out-of-memory error. `--no_amp` turns off mixed precision if it ever
causes trouble, `--num_workers 0` turns off parallel data loading if
that's the culprit instead.

For the loss itself: the learning rate now automatically gets cut in half
whenever validation loss stops improving for a stretch (`--lr_patience` to
adjust how long it waits), instead of training at one fixed rate the
whole time. You'll see a line printed whenever it kicks in.

If you see train loss keep dropping toward zero while validation loss
stalls or creeps back up, that's overfitting -- the model is memorizing
training clips instead of learning to generalize, and more epochs alone
won't fix it (it'll usually make the gap worse). Three things now fight
that directly: SpecAugment randomly blanks out chunks of each training
clip's spectrogram so it can't rely on exact acoustic detail (on by
default, `--no_augment` to disable), AdamW weight decay
(`--weight_decay`, default 0.01), and dropout inside the transformer
(`--dropout`, default 0.2, up from the old fixed 0.1). None of these are
a substitute for more data long-term, but they get more generalization
out of whatever data you already have.

Also: `transcribe.py` now defaults to `checkpoints/voice_model_best.pt`
(lowest validation loss during training) instead of the final epoch's
checkpoint, since the final one is usually more overfit. If `_best.pt`
doesn't exist yet (an older training run), it automatically falls back to
`voice_model.pt` instead. Pass `--checkpoint` to point at a specific one.

Prints a per-speaker clip count, then automatically holds out ~15% of each
speaker's clips as a validation set it never trains on (`--val_split` to
change the fraction; a speaker with fewer than 4 clips is kept entirely in
training, since there's not enough to usefully hold any out yet). Every
epoch prints both train and validation loss for text and speaker, e.g.:

```
Epoch 40/200 - train text: 0.0512 val text: 0.9831 | train speaker: 0.0021 val speaker: 0.0098
```

That gap between train and val text loss is your real generalization
signal -- train loss dropping while val loss stays flat or rises means
it's memorizing, not learning, and more epochs won't fix that (more data
will). Two checkpoints get saved: `checkpoints/voice_model.pt` (final,
after all epochs) and `checkpoints/voice_model_best.pt` (whichever epoch
had the lowest validation loss -- often a better pick to actually use than
the final one, since training tends to keep memorizing past that point).
Point `transcribe.py --checkpoint ...` at either one. Re-run
`collect_data.py` to add more data (for any speaker) and `train.py` again
to keep improving it.

> The model's architecture changed to add speaker identification, so any
> checkpoint trained before this update needs to be retrained from scratch
> -- just run `train.py` again.

## 3. Transcribe

```
python transcribe.py
```

This loads the model once and keeps running: press any key to start
recording, press any key again to stop, and it prints the transcription
along with who it thinks was talking and how confident it is, e.g.
`[z] hello there (97% confident)` -- then immediately lets you do it
again, with no need to re-run the script each time. Press 'q' at the
start prompt to end the session.

Other options:
```
python transcribe.py --once                # record/transcribe a single time then exit
python transcribe.py --seconds 6            # fixed 6s recordings instead of keypress start/stop
python transcribe.py --file path/to/clip.wav  # transcribe one existing file, then exit
python transcribe.py --output note.txt        # always overwrite this file with [speaker] text
python transcribe.py --checkpoint checkpoints/voice_model.pt  # use the final epoch instead of the best one
python transcribe.py --list-devices           # show every mic sounddevice can see
python transcribe.py --input-device 3          # force mic index 3 instead of the system default
python transcribe.py --input-device "USB Mic"   # or match by name/substring
```

By default it records from whatever Windows currently calls the default input device -- if live transcriptions are much worse than file transcriptions of the same sentence, that's a strong sign the default input isn't actually your mic (VoiceMeeter/OBS/Discord can silently switch it to a virtual cable device). Check with `--list-devices` and pin the right one with `--input-device`.

Every transcription (in any mode) is also appended, timestamped, with its
speaker and confidence, to `transcripts.txt` in the project root -- so
nothing is ever lost even if you don't use `--output`.

## Notes

- Everything expects/resamples to 16kHz mono audio. Live mic recording captures at your mic's own native rate (whatever Windows reports) and resamples down to 16kHz afterward, with the same resampler used for loaded files -- this avoids PortAudio doing real-time sample-rate conversion during capture, which can subtly distort speech (this was the cause of noticeably worse live-mic transcriptions vs. file transcriptions of the same sentence).
- Keypress start/stop (`keypress.py`) works without needing to hit Enter --
  it uses `msvcrt` on Windows and raw terminal mode on macOS/Linux.
- This is a personal, from-scratch model -- its accuracy (for both text and
  speaker id) is entirely a function of how much data you feed it. It will
  not be as accurate as Whisper or a commercial API out of the box; that
  trade-off is the point of building it yourself.
- `--epochs`, `--batch_size`, `embed_dim`, `n_heads`, `n_layers` in
  `train.py` are all worth tuning once you have more data.
