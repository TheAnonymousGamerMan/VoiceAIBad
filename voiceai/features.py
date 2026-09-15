"""
Hand-built log-mel spectrogram feature extraction.

No librosa / torchaudio. Everything here boils down to:
  1. slicing the waveform into overlapping frames (a matrix of frames),
  2. an FFT per frame to get a power spectrum,
  3. one matrix multiplication against a triangular mel filterbank matrix,
  4. a log.

That matrix multiplication (power spectrum @ mel_filterbank.T) is the same
kind of operation the model itself is built from -- it's matrices all the
way down.
"""
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400          # 25ms window at 16kHz
HOP_LENGTH = 160      # 10ms hop at 16kHz
N_MELS = 40
EPS = 1e-6


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    """Returns a [n_mels, n_fft//2 + 1] matrix of triangular mel filters."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    mel_min = hz_to_mel(0.0)
    mel_max = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = hz_points[m - 1], hz_points[m], hz_points[m + 1]

        left_slope = (fft_freqs - f_left) / max(f_center - f_left, 1e-10)
        right_slope = (f_right - fft_freqs) / max(f_right - f_center, 1e-10)

        filterbank[m - 1] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    return filterbank


_MEL_FILTERBANK = build_mel_filterbank()


def frame_signal(signal, frame_length=N_FFT, hop_length=HOP_LENGTH):
    """Slice a 1D waveform into overlapping frames -> [num_frames, frame_length]."""
    if len(signal) < frame_length:
        signal = np.pad(signal, (0, frame_length - len(signal)))

    num_frames = 1 + (len(signal) - frame_length) // hop_length
    indices = (
        np.arange(frame_length)[None, :]
        + hop_length * np.arange(num_frames)[:, None]
    )
    return signal[indices]


def compute_log_mel(waveform, sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                     hop_length=HOP_LENGTH, n_mels=N_MELS, normalize=True):
    """waveform: 1D float32 numpy array -> returns [num_frames, n_mels] float32."""
    waveform = np.asarray(waveform, dtype=np.float32)

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE}Hz audio, got {sample_rate}Hz. "
            "Resample first (record.load_audio_file does this automatically)."
        )

    frames = frame_signal(waveform, n_fft, hop_length)
    window = np.hamming(n_fft).astype(np.float32)
    frames = frames * window

    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power_spectrum = (np.abs(spectrum) ** 2) / n_fft

    filterbank = _MEL_FILTERBANK if n_mels == N_MELS else build_mel_filterbank(
        sample_rate, n_fft, n_mels
    )

    mel_energies = power_spectrum @ filterbank.T
    log_mel = np.log(mel_energies + EPS).astype(np.float32)

    if normalize:
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + EPS)

    return log_mel


def spec_augment(log_mel, time_mask_param=30, freq_mask_param=8,
                  num_time_masks=2, num_freq_masks=2):
    """
    SpecAugment-style regularization: randomly blanks out a few contiguous
    time and frequency bands in a log-mel spectrogram. The model can't rely
    on exact acoustic detail being there every time, so it's forced to learn
    more robust, generalizable patterns instead of memorizing training
    clips outright.

    Only ever apply this to TRAINING examples -- never at inference, and
    never on validation data (that would make val loss meaningless as a
    generalization signal).
    """
    log_mel = log_mel.copy()
    num_frames, n_mels = log_mel.shape
    mask_value = float(log_mel.mean())

    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_param + 1)
        if f == 0 or f >= n_mels:
            continue
        f0 = np.random.randint(0, n_mels - f)
        log_mel[:, f0:f0 + f] = mask_value

    for _ in range(num_time_masks):
        t = np.random.randint(0, min(time_mask_param, num_frames) + 1)
        if t == 0 or t >= num_frames:
            continue
        t0 = np.random.randint(0, num_frames - t)
        log_mel[t0:t0 + t, :] = mask_value

    return log_mel
