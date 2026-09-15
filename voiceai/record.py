"""
Audio capture: live microphone recording (fixed-duration or press-a-key
start/stop), and loading existing audio files. Everything gets converted to
mono float32 @ 16kHz, which is what features.compute_log_mel expects.

IMPORTANT: live mic capture records at the input device's own native rate
(whatever Windows reports for it) and then resamples down to 16000 Hz
ourselves with the same linear-interpolation resampler used for loaded
files. Earlier versions asked sounddevice/PortAudio to capture directly at
16000 Hz, which forces the OS to do real-time sample-rate conversion during
recording -- on Windows this often goes through a lower-quality resampler
than a simple post-hoc conversion, and can subtly warp the audio (distorted
vowels, consonants blurring together). Doing the resample ourselves, after
capture, keeps the live-mic path consistent with the file-loading path.

Also: every capture function accepts an optional `device` (an index or a
name/substring, same as sounddevice.query_devices() rows) so you can pin a
specific input instead of trusting whatever Windows currently calls the
"default" device. This matters more than it sounds like it should --
software like VoiceMeeter/OBS/Discord can silently change your system
default input to a virtual cable device, which may not carry your actual
voice at all. Run `python transcribe.py --list-devices` (or
`collect_data.py --list-devices`) to see every device sounddevice can see
and figure out which index/name is your real microphone.
"""
import numpy as np
import soundfile as sf

TARGET_SR = 16000


def list_input_devices():
    """Prints every audio device sounddevice can see, with the current
    default input device marked, so you can find your real mic's
    index/name to pass as --input-device."""
    import sounddevice as sd

    print(sd.query_devices())
    try:
        default_in = sd.default.device[0]
        print(f"\nCurrent default input device index: {default_in}")
    except Exception:
        pass


def parse_device_arg(value):
    """Turns a CLI --input-device string into what sounddevice expects:
    an int index if it looks numeric, otherwise a name/substring string.
    None passes through unchanged (meaning "use the system default")."""
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _print_capture_stats(waveform, sample_rate):
    """Quick sanity check so you can tell 'no real mic audio' apart from
    'model just needs more training data' -- prints duration and peak
    volume of whatever was actually captured."""
    if waveform.size == 0:
        print("Captured 0.00s -- nothing recorded.")
        return
    duration = len(waveform) / sample_rate
    peak = float(np.abs(waveform).max())
    warning = ""
    if peak < 0.01:
        warning = "  <-- near-silent! check your mic / default input device in Windows sound settings"
    elif peak > 0.98:
        warning = "  <-- clipping! you may be too close to the mic or input gain is too high"
    print(f"Captured {duration:.2f}s, peak volume {peak:.3f} (0=silence, 1=max){warning}")


def _resample_linear(waveform, orig_sr, target_sr):
    if orig_sr == target_sr:
        return waveform
    duration = len(waveform) / orig_sr
    n_target = int(round(duration * target_sr))
    orig_times = np.linspace(0.0, duration, num=len(waveform), endpoint=False)
    target_times = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(target_times, orig_times, waveform).astype(np.float32)


def _native_input_samplerate(fallback=TARGET_SR, device=None):
    """Ask the OS what sample rate the given input device (or the default
    one, if device is None) actually runs at natively, so we can record at
    that rate instead of forcing PortAudio to resample on the fly during
    capture."""
    import sounddevice as sd

    try:
        if device is not None:
            info = sd.query_devices(device)
        else:
            info = sd.query_devices(kind="input")
        sr = int(round(float(info["default_samplerate"])))
        name = info.get("name", "default input device")
        if sr <= 0:
            return fallback, name
        return sr, name
    except Exception as exc:
        print(f"(could not query native mic sample rate, falling back to {fallback}Hz: {exc})")
        return fallback, "unknown device"


def record_from_mic(duration_sec=3.0, sample_rate=TARGET_SR, device=None):
    """Records a fixed `duration_sec` seconds from the given input device
    (or the system default, if device is None), capturing at the device's
    native rate and resampling to `sample_rate` afterward for consistency
    with the file-loading path."""
    import sounddevice as sd

    native_sr, device_name = _native_input_samplerate(fallback=sample_rate, device=device)
    print(f"Recording for {duration_sec:.1f}s (mic: {device_name} @ {native_sr}Hz native)...")
    audio = sd.rec(
        int(duration_sec * native_sr),
        samplerate=native_sr,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    waveform = audio.reshape(-1)
    if native_sr != sample_rate:
        waveform = _resample_linear(waveform, native_sr, sample_rate)
        print(f"Resampled {native_sr}Hz -> {sample_rate}Hz.")
    print("Done.")
    _print_capture_stats(waveform, sample_rate)
    return waveform


def record_until_keypress(sample_rate=TARGET_SR, stop_prompt="Recording... press any key to stop.", device=None):
    """
    Starts recording immediately and keeps going until any key is pressed,
    so you control the length yourself instead of a fixed duration.
    Captures at the input device's native rate and resamples to
    `sample_rate` afterward, same as record_from_mic. Pass `device` to pin
    a specific mic instead of the system default.
    """
    import sounddevice as sd
    from keypress import read_key

    native_sr, device_name = _native_input_samplerate(fallback=sample_rate, device=device)
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=native_sr,
        channels=1,
        dtype="float32",
        callback=callback,
        device=device,
    )
    print(f"(mic: {device_name} @ {native_sr}Hz native)")
    with stream:
        read_key(stop_prompt)

    print("Done.")

    if not frames:
        waveform = np.zeros(0, dtype=np.float32)
    else:
        waveform = np.concatenate(frames, axis=0).reshape(-1)
        if native_sr != sample_rate:
            waveform = _resample_linear(waveform, native_sr, sample_rate)
            print(f"Resampled {native_sr}Hz -> {sample_rate}Hz.")

    _print_capture_stats(waveform, sample_rate)
    return waveform


def save_wav(path, waveform, sample_rate=TARGET_SR):
    sf.write(str(path), waveform, sample_rate)


def load_audio_file(path, target_sr=TARGET_SR):
    """Loads an audio file, downmixes to mono, resamples to target_sr."""
    waveform, sr = sf.read(str(path), always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)  # downmix to mono

    if sr != target_sr:
        waveform = _resample_linear(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr
