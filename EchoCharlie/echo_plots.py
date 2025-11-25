from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip


def load_audio(audio_path: Path, sample_rate: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Load an audio file as mono using librosa."""
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    return audio, sr


def compute_spectrogram(audio: np.ndarray,sample_rate: int,
                        n_fft: int = 2048,
                        hop_length: int = 512) -> np.ndarray:
    """Compute a log-scaled spectrogram."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    return librosa.amplitude_to_db(np.abs(stft), ref=np.max)


def plot_spectrogram(ax: plt.Axes, spec: np.ndarray, sample_rate: int, title: str) -> None:
    """Plot a precomputed spectrogram."""
    img = librosa.display.specshow(spec, sr=sample_rate, x_axis="time", y_axis="log", ax=ax)
    ax.set(title=title, xlabel="Time (s)", ylabel="Frequency (Hz)")
    ax.figure.colorbar(img, ax=ax, format="%+2.0f dB")


def compute_mel_spectrogram(audio: np.ndarray, sample_rate: int, n_mels: int = 128) -> np.ndarray:
    """Compute a mel spectrogram in dB scale."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, fmax=sample_rate / 2)
    return librosa.power_to_db(mel_spec, ref=np.max)


def plot_mel_spectrogram(ax: plt.Axes, mel_spec: np.ndarray, sample_rate: int, title: str) -> None:
    """Plot a mel spectrogram."""
    img = librosa.display.specshow(mel_spec, sr=sample_rate, x_axis="time", y_axis="mel", ax=ax)
    ax.set(title=title, xlabel="Time (s)", ylabel="Mel bins")
    ax.figure.colorbar(img, ax=ax, format="%+2.0f dB")


def plot_frequency_overlay(ax: plt.Axes,audio_a: np.ndarray,audio_b: np.ndarray,
                           sample_rate: int,label_a: str,label_b: str) -> None:
    """Plot both clips' FFT magnitudes on one set of axes."""
    fft_a = np.abs(np.fft.rfft(audio_a))
    fft_b = np.abs(np.fft.rfft(audio_b))
    freqs_a = np.fft.rfftfreq(len(audio_a), d=1.0 / sample_rate)
    freqs_b = np.fft.rfftfreq(len(audio_b), d=1.0 / sample_rate)

    ax.plot(freqs_a, fft_a, label=label_a, alpha=0.8)
    ax.plot(freqs_b, fft_b, label=label_b, alpha=0.8)
    ax.set(title="Frequency Spectrum Comparison", xlabel="Frequency (Hz)", ylabel="Magnitude")
    ax.set_xlim(0, sample_rate / 2)
    ax.grid(alpha=0.3)
    ax.legend()


def plot_mel_difference_heatmap(ax: plt.Axes, mel_a: np.ndarray, mel_b: np.ndarray) -> None:
    """Plot difference between two mel spectrograms using Matplotlib's default colormap."""
    min_rows = min(mel_a.shape[0], mel_b.shape[0])
    min_cols = min(mel_a.shape[1], mel_b.shape[1])
    diff = mel_a[:min_rows, :min_cols] - mel_b[:min_rows, :min_cols]
    img = librosa.display.specshow(diff, x_axis="time", y_axis="mel", ax=ax, cmap=plt.rcParams["image.cmap"])
    ax.set(title="Mel Spectrogram Difference", xlabel="Time (s)", ylabel="Mel bins")
    ax.figure.colorbar(img, ax=ax, format="%+2.0f dB")


def compute_dtw(mfcc_a: np.ndarray, mfcc_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute DTW cumulative cost matrix, optimal path, and average distance."""
    cost_matrix, wp = librosa.sequence.dtw(X=mfcc_a, Y=mfcc_b, metric="euclidean")
    wp = np.asarray(wp)
    avg_distance = float(cost_matrix[-1, -1] / len(wp))
    return cost_matrix, wp, avg_distance


def plot_dtw(ax: plt.Axes, cost_matrix: np.ndarray, path: np.ndarray, distance: float) -> None:
    """Plot DTW cost matrix with alignment path."""
    img = ax.imshow(cost_matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.plot(path[:, 1], path[:, 0], color="w", linewidth=1.5)
    ax.set(
        title=f"DTW Path (avg distance={distance:.2f})",
        xlabel="Audio B frames",
        ylabel="Audio A frames",
    )
    ax.figure.colorbar(img, ax=ax)


def extract_audio_from_video(video_path: Path, sample_rate: Optional[int] = None) -> Path:
    """Extract and persist audio from a video into a deterministic temp wav file."""
    tmp_path = Path(tempfile.gettempdir()) / "reference_audio.wav"
    tmp_path.unlink(missing_ok=True)

    clip = VideoFileClip(str(video_path))
    audio_clip = clip.audio
    try:
        if audio_clip is None:
            raise ValueError(f"Video {video_path} does not contain an audio track.")
        fps = sample_rate or getattr(audio_clip, "fps", None) or 16000
        audio_clip.write_audiofile(tmp_path.as_posix(), fps=fps, logger=None)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        if audio_clip:
            audio_clip.close()
        clip.close()

    return tmp_path


def compare_audio_clips(audio_a: Path,audio_b: Path,output_dir: Optional[Path] = None,
                        sample_rate: Optional[int] = None,show: bool = True,) -> Path | None:
    
    """Render comparison plots for two audio clips."""
    clip_a, sr_a = load_audio(audio_a, sample_rate=sample_rate)
    clip_b, sr_b = load_audio(audio_b, sample_rate=sample_rate if sample_rate is not None else sr_a)
    if sr_a != sr_b:
        raise ValueError(
            f"Sample rates do not match ({sr_a} vs {sr_b}). Use --sample-rate to enforce a shared rate."
        )

    spec_a = compute_spectrogram(clip_a, sr_a)
    spec_b = compute_spectrogram(clip_b, sr_b)
    mel_a = compute_mel_spectrogram(clip_a, sr_a)
    mel_b = compute_mel_spectrogram(clip_b, sr_b)
    mfcc_a = librosa.feature.mfcc(y=clip_a, sr=sr_a, n_mfcc=20)
    mfcc_b = librosa.feature.mfcc(y=clip_b, sr=sr_b, n_mfcc=20)
    cost_matrix, path, dtw_distance = compute_dtw(mfcc_a, mfcc_b)

    fig = plt.figure(figsize=(16, 16), constrained_layout=True)
    fig.suptitle(f"Audio Comparison: {audio_a.name} vs {audio_b.name}", fontsize=14)
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.6])

    ax_spec_a = fig.add_subplot(grid[0, 0])
    plot_spectrogram(ax_spec_a, spec_a, sr_a, f"Spectrogram - {audio_a.name}")
    ax_spec_b = fig.add_subplot(grid[0, 1])
    plot_spectrogram(ax_spec_b, spec_b, sr_b, f"Spectrogram - {audio_b.name}")

    ax_mel_a = fig.add_subplot(grid[1, 0])
    plot_mel_spectrogram(ax_mel_a, mel_a, sr_a, f"Mel Spectrogram - {audio_a.name}")
    ax_mel_b = fig.add_subplot(grid[1, 1])
    plot_mel_spectrogram(ax_mel_b, mel_b, sr_b, f"Mel Spectrogram - {audio_b.name}")

    ax_diff = fig.add_subplot(grid[2, 0])
    plot_mel_difference_heatmap(ax_diff, mel_a, mel_b)

    ax_dtw = fig.add_subplot(grid[2, 1])
    plot_dtw(ax_dtw, cost_matrix, path, dtw_distance)

    ax_freq = fig.add_subplot(grid[3, :])
    plot_frequency_overlay(ax_freq, clip_a, clip_b, sr_a, audio_a.name, audio_b.name)

    saved_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_path = output_dir / f"{audio_a.stem}_vs_{audio_b.stem}_comparison.png"
        fig.savefig(saved_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[echo_plots] DTW average distance ({audio_a.name} vs {audio_b.name}): {dtw_distance:.4f}")
    return saved_path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two audio clips")
    parser.add_argument(
        "--a1",
        type=Path,
        required=True,
        help="Reference clip (audio/video), combine with --a1-is-video",
    )
    parser.add_argument(
        "--a2",
        type=Path,
        required=True,
        help="Comparison audio clip",
    )
    parser.add_argument(
        "--a1-is-video",
        action="store_true",
        help="Treat --a1 as a video file and extract its audio track",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/plots/",
        help="Directory to save comparison plots",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Resample both clips to this rate before processing.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots interactively.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    audio_a = args.a1.expanduser()
    audio_b = args.a2.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else None

    temp_audio: Optional[Path] = None
    try:
        if args.a1_is_video:
            temp_audio = extract_audio_from_video(audio_a, sample_rate=args.sample_rate)
            audio_a = temp_audio

        saved_path = compare_audio_clips(
            audio_a=audio_a,
            audio_b=audio_b,
            output_dir=output_dir,
            sample_rate=args.sample_rate,
            show=not args.no_show,
        )
    except FileNotFoundError as exc:
        print(f"[echo_plots] File not found: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive guardrail
        print(f"[echo_plots] Failed to process {args.a1} vs {args.a2}: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_audio and temp_audio.exists():
            temp_audio.unlink(missing_ok=True)

    if saved_path:
        print(f"[echo_plots] Saved comparison plots to {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
