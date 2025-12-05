"""
Signal coregistration module for temporal synchronization.

This module provides functionality for aligning two electrophysiology recordings
from independent sources with different clocks, sampling rates, and durations.

The coregistration proceeds in two stages:
1. Stage I - Coarse Global Alignment: Computes global time offset using envelope correlation
2. Stage II - Fine Local Alignment: Refines alignment using chunk-based approach

The output is a sample-level timestamp mapping that transforms each sample from
the source signal to the corresponding time in the reference signal's time frame.
This handles both constant offsets and time-varying clock drift (floating clock).

Classes:
    AlignmentMap: Stores and serializes synchronization transformation

Functions:
    compute_alignment: Main entry point for computing signal alignment
    coarse_alignment: Stage I global alignment
    fine_alignment: Stage II local refinement
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any
import json
import numpy as np


@dataclass
class AlignmentMap:
    """
    Stores the synchronization transformation between two signal sources.

    This class holds the results of coregistration including:
    - Global time offset (Stage I)
    - Local drift corrections (Stage II)
    - Sample-level timestamp mapping for the source signal
    - Metadata about the alignment process

    The alignment map provides timestamps for each sample of the source signal
    that map to the reference signal's time frame. This handles both constant
    clock offsets and time-varying clock drift (floating clock).

    Attributes:
        global_offset_s: Global time offset in seconds (t0 from Stage I)
        chunk_offsets_s: List of (time_center_s, offset_s) tuples for local corrections
        source_timestamps_s: Array of timestamps (one per source sample) in reference time frame
        fs_source: Sampling rate of source signal (Source_B)
        fs_reference: Sampling rate of reference signal (Source_A)
        chunk_size_s: Chunk size used for Stage II alignment
        correlation_score: Quality metric from global alignment
        metadata: Additional alignment metadata
    """
    global_offset_s: float = 0.0
    chunk_offsets_s: List[Tuple[float, float]] = field(default_factory=list)
    source_timestamps_s: np.ndarray = field(default_factory=lambda: np.array([]))
    fs_source: float = 0.0
    fs_reference: float = 0.0
    chunk_size_s: float = 300.0  # 5 minutes default
    correlation_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_offset_at_time(self, time_s: float) -> float:
        """
        Get the total time offset at a given time point.

        Combines global offset with interpolated local drift correction.

        Args:
            time_s: Time in seconds in the source signal's time frame

        Returns:
            Total offset in seconds to apply for alignment
        """
        if not self.chunk_offsets_s:
            return self.global_offset_s

        # Extract chunk centers and offsets
        chunk_centers = np.array([c[0] for c in self.chunk_offsets_s])
        chunk_offsets = np.array([c[1] for c in self.chunk_offsets_s])

        # Interpolate offset at requested time
        local_offset = np.interp(time_s, chunk_centers, chunk_offsets)
        return self.global_offset_s + local_offset

    def transform_time(self, source_time_s: float) -> float:
        """
        Transform time from source signal frame to reference signal frame.

        Args:
            source_time_s: Time in seconds in source signal's time frame

        Returns:
            Corresponding time in reference signal's time frame
        """
        offset = self.get_offset_at_time(source_time_s)
        return source_time_s + offset

    def get_reference_timestamps(self, num_samples: Optional[int] = None) -> np.ndarray:
        """
        Get timestamps for each sample of the source signal in the reference time frame.

        This is the primary output for mapping source signal samples to reference time.
        Each element i gives the time in the reference signal where source sample i belongs.

        Args:
            num_samples: Number of samples to generate timestamps for.
                        If None, uses the stored source_timestamps_s.

        Returns:
            Array of timestamps in seconds, one per source sample.
            These timestamps are in the reference signal's time frame.
        """
        if num_samples is None:
            if len(self.source_timestamps_s) > 0:
                return self.source_timestamps_s
            else:
                raise ValueError("No timestamps available. Either provide num_samples or compute alignment first.")

        # Generate timestamps for each sample
        source_times = np.arange(num_samples) / self.fs_source

        if not self.chunk_offsets_s:
            # Simple case: constant offset only
            return source_times + self.global_offset_s
        else:
            # Complex case: interpolate local offsets for each sample
            chunk_centers = np.array([c[0] for c in self.chunk_offsets_s])
            chunk_offsets = np.array([c[1] for c in self.chunk_offsets_s])
            local_offsets = np.interp(source_times, chunk_centers, chunk_offsets)
            return source_times + self.global_offset_s + local_offsets

    def to_dict(self) -> Dict[str, Any]:
        """Convert alignment map to dictionary for serialization."""
        d = asdict(self)
        # Convert numpy array to list for JSON serialization
        if isinstance(d['source_timestamps_s'], np.ndarray):
            d['source_timestamps_s'] = d['source_timestamps_s'].tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlignmentMap':
        """Create alignment map from dictionary."""
        # Convert chunk_offsets_s from list of lists to list of tuples
        if 'chunk_offsets_s' in data:
            data['chunk_offsets_s'] = [tuple(c) for c in data['chunk_offsets_s']]
        # Convert source_timestamps_s from list to numpy array
        if 'source_timestamps_s' in data:
            data['source_timestamps_s'] = np.array(data['source_timestamps_s'])
        return cls(**data)

    def save(self, filepath: str) -> None:
        """
        Save alignment map to JSON file.

        Args:
            filepath: Path to save the alignment map
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'AlignmentMap':
        """
        Load alignment map from JSON file.

        Args:
            filepath: Path to load the alignment map from

        Returns:
            Loaded AlignmentMap instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def _compute_signal_envelope(
    signal: np.ndarray,
    sampling_rate: float,
    envelope_freq_hz: float = 1.0
) -> np.ndarray:
    """
    Compute signal envelope using low-pass filtering of absolute signal.

    Uses a simple moving average for computational efficiency with large signals.

    Args:
        signal: 1D signal array
        sampling_rate: Sampling rate in Hz
        envelope_freq_hz: Target envelope frequency in Hz (determines smoothing)

    Returns:
        Envelope signal (downsampled to approximately envelope_freq_hz)
    """
    # Window size for envelope computation (in samples)
    window_samples = max(1, int(sampling_rate / envelope_freq_hz))

    # Use absolute value
    abs_signal = np.abs(signal)

    # Compute running mean using cumsum for efficiency
    cumsum = np.cumsum(abs_signal)
    cumsum = np.insert(cumsum, 0, 0)

    # Compute envelope with strided view
    envelope_len = len(abs_signal) // window_samples
    envelope = np.zeros(envelope_len)
    for i in range(envelope_len):
        start = i * window_samples
        end = min((i + 1) * window_samples, len(abs_signal))
        envelope[i] = (cumsum[end] - cumsum[start]) / (end - start)

    return envelope


def _resample_to_common_rate(
    signal: np.ndarray,
    from_rate: float,
    to_rate: float
) -> np.ndarray:
    """
    Resample signal from one rate to another using linear interpolation.

    Args:
        signal: 1D signal array
        from_rate: Original sampling rate
        to_rate: Target sampling rate

    Returns:
        Resampled signal
    """
    if from_rate == to_rate:
        return signal.copy()

    duration_s = len(signal) / from_rate
    num_samples_target = int(duration_s * to_rate)

    t_original = np.arange(len(signal)) / from_rate
    t_target = np.arange(num_samples_target) / to_rate

    return np.interp(t_target, t_original, signal)


def coarse_alignment(
    signal_reference: np.ndarray,
    signal_source: np.ndarray,
    fs_reference: float,
    fs_source: float,
    search_range_s: Optional[Tuple[float, float]] = None,
    envelope_freq_hz: float = 1.0,
) -> Tuple[float, float]:
    """
    Stage I: Compute coarse global alignment using envelope cross-correlation.

    This function computes the global time offset (t0) by correlating the signal
    envelopes. Using envelopes is robust against periodic patterns (like stimulation)
    that could cause phase-shifted misalignments.

    The offset represents where the source signal starts relative to the reference.
    If source_time = reference_time + offset, then offset > 0 means source
    starts later in real time.

    Args:
        signal_reference: Reference signal (Source_A, typically longer recording)
        signal_source: Source signal (Source_B, typically shorter recording to align)
        fs_reference: Sampling rate of reference signal in Hz
        fs_source: Sampling rate of source signal in Hz
        search_range_s: Optional (min, max) offset range in seconds to search.
                       If None, searches full range.
        envelope_freq_hz: Envelope computation frequency (affects smoothing)

    Returns:
        Tuple of (offset_s, correlation_score):
        - offset_s: Time offset in seconds. This is the time at which source starts
                   in the reference signal's time frame.
        - correlation_score: Normalized correlation score (0-1, higher is better)
    """
    # Compute envelopes at common low rate for efficiency
    env_ref = _compute_signal_envelope(signal_reference, fs_reference, envelope_freq_hz)
    env_src = _compute_signal_envelope(signal_source, fs_source, envelope_freq_hz)

    # Effective sampling rate of envelopes
    env_fs = envelope_freq_hz

    n_ref = len(env_ref)
    n_src = len(env_src)

    # Normalize signals for correlation
    env_ref_norm = env_ref - np.mean(env_ref)
    env_src_norm = env_src - np.mean(env_src)

    ref_std = np.std(env_ref_norm)
    src_std = np.std(env_src_norm)

    if ref_std < 1e-10 or src_std < 1e-10:
        # Signals are constant, can't compute meaningful correlation
        return 0.0, 0.0

    env_ref_norm = env_ref_norm / ref_std
    env_src_norm = env_src_norm / src_std

    # Use numpy correlate in 'full' mode
    # correlate(ref, src, 'full') gives correlation at all possible lags
    # Result length is n_ref + n_src - 1
    # Index k corresponds to ref aligned with src shifted by (k - (n_src - 1)) samples
    # i.e., lag = k - (n_src - 1)
    cross_corr = np.correlate(env_ref_norm, env_src_norm, mode='full')

    # Normalize by overlap length for each lag
    n_corr = len(cross_corr)
    lags = np.arange(n_corr) - (n_src - 1)

    # Calculate overlap for each lag
    # When lag is 0, full overlap (min of n_ref, n_src)
    # When lag > 0, src is shifted right in ref, overlap decreases
    # When lag < 0, src is shifted left (before ref start), overlap decreases
    overlap = np.zeros(n_corr)
    for i, lag in enumerate(lags):
        if lag >= 0:
            overlap[i] = min(n_src, n_ref - lag)
        else:
            overlap[i] = min(n_src + lag, n_ref)
    overlap = np.maximum(overlap, 1)  # Avoid division by zero

    cross_corr = cross_corr / overlap

    # Convert lag indices to time offsets
    # lag > 0 means source starts at lag samples into the reference
    # This is the time shift we want to detect
    lag_times_s = lags / env_fs

    # Apply search range if specified
    if search_range_s is not None:
        min_offset, max_offset = search_range_s
        valid_mask = (lag_times_s >= min_offset) & (lag_times_s <= max_offset)
        if not np.any(valid_mask):
            # No valid lags in range
            return 0.0, 0.0
        cross_corr = np.where(valid_mask, cross_corr, -np.inf)

    # Find best alignment
    best_idx = np.argmax(cross_corr)
    best_lag = lags[best_idx]
    best_offset_s = best_lag / env_fs
    correlation_score = cross_corr[best_idx]

    # Normalize score to 0-1 range
    correlation_score = float(min(1.0, max(0.0, correlation_score)))

    return best_offset_s, correlation_score


def fine_alignment(
    signal_reference: np.ndarray,
    signal_source: np.ndarray,
    fs_reference: float,
    fs_source: float,
    global_offset_s: float,
    chunk_size_s: float = 300.0,
    overlap_ratio: float = 0.5,
    search_window_s: float = 30.0,
) -> List[Tuple[float, float]]:
    """
    Stage II: Compute fine local alignment using sliding window approach.

    This function refines the global alignment by computing local offsets for
    overlapping chunks. This handles time-varying clock drift between devices.

    Args:
        signal_reference: Reference signal (Source_A)
        signal_source: Source signal (Source_B)
        fs_reference: Sampling rate of reference signal in Hz
        fs_source: Sampling rate of source signal in Hz
        global_offset_s: Global offset from Stage I in seconds. This represents
                        where the source signal starts in the reference signal.
        chunk_size_s: Size of each chunk in seconds (default: 5 minutes)
        overlap_ratio: Overlap ratio between consecutive chunks (default: 0.5)
        search_window_s: Search window around global offset for local refinement

    Returns:
        List of (chunk_center_time_s, local_offset_s) tuples.
        Local offset is relative to global offset.
    """
    # Calculate signal durations
    src_duration_s = len(signal_source) / fs_source

    # Calculate chunk parameters
    step_s = chunk_size_s * (1 - overlap_ratio)
    chunk_centers = []
    chunk_offsets = []

    # Process each chunk
    t = chunk_size_s / 2  # Start at first chunk center in source signal
    while t < src_duration_s - chunk_size_s / 2:
        # Extract chunk from source signal
        src_start_s = t - chunk_size_s / 2
        src_end_s = t + chunk_size_s / 2

        src_start_idx = int(src_start_s * fs_source)
        src_end_idx = int(src_end_s * fs_source)
        src_chunk = signal_source[src_start_idx:src_end_idx]

        # Calculate expected position in reference signal
        # global_offset_s is where source starts in reference
        # So source time t corresponds to reference time (global_offset_s + t)
        ref_center_s = global_offset_s + t
        ref_start_s = ref_center_s - chunk_size_s / 2 - search_window_s
        ref_end_s = ref_center_s + chunk_size_s / 2 + search_window_s

        # Clip to valid range
        ref_start_idx = max(0, int(ref_start_s * fs_reference))
        ref_end_idx = min(len(signal_reference), int(ref_end_s * fs_reference))

        if ref_end_idx <= ref_start_idx:
            # Not enough reference signal
            t += step_s
            continue

        ref_chunk = signal_reference[ref_start_idx:ref_end_idx]

        # Resample source chunk to reference rate for comparison
        if fs_source != fs_reference:
            src_chunk_resampled = _resample_to_common_rate(
                src_chunk, fs_source, fs_reference
            )
        else:
            src_chunk_resampled = src_chunk

        # Compute local cross-correlation
        local_offset_s, _ = _chunk_cross_correlation(
            ref_chunk, src_chunk_resampled, fs_reference, search_window_s
        )

        # Convert to offset relative to global offset
        # The local offset corrects for drift from the global alignment
        chunk_centers.append(t)
        chunk_offsets.append(local_offset_s)

        t += step_s

    return list(zip(chunk_centers, chunk_offsets))


def _chunk_cross_correlation(
    ref_chunk: np.ndarray,
    src_chunk: np.ndarray,
    fs: float,
    search_window_s: float
) -> Tuple[float, float]:
    """
    Compute cross-correlation between two chunks to find local offset.

    Args:
        ref_chunk: Reference chunk (potentially larger with search window)
        src_chunk: Source chunk to align
        fs: Common sampling rate
        search_window_s: Search window in seconds

    Returns:
        Tuple of (offset_s, correlation_score)
    """
    if len(ref_chunk) < len(src_chunk):
        return 0.0, 0.0

    # Normalize signals
    ref_norm = ref_chunk - np.mean(ref_chunk)
    src_norm = src_chunk - np.mean(src_chunk)

    ref_std = np.std(ref_norm)
    src_std = np.std(src_norm)

    if ref_std < 1e-10 or src_std < 1e-10:
        return 0.0, 0.0

    ref_norm = ref_norm / ref_std
    src_norm = src_norm / src_std

    # Use numpy correlate in 'valid' mode for sliding dot product
    correlation = np.correlate(ref_norm, src_norm, mode='valid')

    # Normalize by chunk length
    correlation = correlation / len(src_norm)

    # Find peak
    peak_idx = np.argmax(correlation)
    peak_score = correlation[peak_idx]

    # Convert to time offset
    # Peak at center means zero offset
    center_idx = len(correlation) // 2
    offset_samples = peak_idx - center_idx
    offset_s = offset_samples / fs

    return offset_s, peak_score


def compute_alignment(
    signal_reference: np.ndarray,
    signal_source: np.ndarray,
    fs_reference: float,
    fs_source: float,
    search_range_s: Optional[Tuple[float, float]] = None,
    chunk_size_s: float = 300.0,
    perform_fine_alignment: bool = True,
    envelope_freq_hz: float = 1.0,
    generate_sample_timestamps: bool = True,
) -> AlignmentMap:
    """
    Compute full alignment between two signals.

    This is the main entry point for signal coregistration. It performs:
    1. Stage I - Coarse global alignment using envelope correlation
    2. Stage II - Fine local alignment using chunk-based refinement (optional)
    3. Generate sample-level timestamp mapping (optional)

    The output includes timestamps for each sample of the source signal
    that map to the reference signal's time frame. This handles both
    constant clock offsets and time-varying clock drift (floating clock).

    Args:
        signal_reference: Reference signal (Source_A, typically 24h recording)
        signal_source: Source signal (Source_B, typically shorter recording)
        fs_reference: Sampling rate of reference signal in Hz
        fs_source: Sampling rate of source signal in Hz
        search_range_s: Optional (min, max) offset range in seconds to search
        chunk_size_s: Chunk size for Stage II alignment (default: 5 minutes)
        perform_fine_alignment: Whether to perform Stage II fine alignment
        envelope_freq_hz: Envelope frequency for Stage I (affects smoothing)
        generate_sample_timestamps: Whether to generate per-sample timestamps

    Returns:
        AlignmentMap containing:
        - global_offset_s: Global time offset
        - chunk_offsets_s: Local drift corrections (if fine alignment enabled)
        - source_timestamps_s: Timestamps for each source sample in reference time frame
        - Metadata about the alignment
    """
    # Stage I: Coarse global alignment
    global_offset_s, correlation_score = coarse_alignment(
        signal_reference=signal_reference,
        signal_source=signal_source,
        fs_reference=fs_reference,
        fs_source=fs_source,
        search_range_s=search_range_s,
        envelope_freq_hz=envelope_freq_hz,
    )

    # Stage II: Fine local alignment (optional)
    chunk_offsets = []
    if perform_fine_alignment:
        chunk_offsets = fine_alignment(
            signal_reference=signal_reference,
            signal_source=signal_source,
            fs_reference=fs_reference,
            fs_source=fs_source,
            global_offset_s=global_offset_s,
            chunk_size_s=chunk_size_s,
        )

    # Generate sample-level timestamps
    num_source_samples = len(signal_source)
    source_times = np.arange(num_source_samples) / fs_source

    if generate_sample_timestamps:
        if not chunk_offsets:
            # Simple case: constant offset only
            source_timestamps = source_times + global_offset_s
        else:
            # Complex case: interpolate local offsets for each sample
            chunk_centers = np.array([c[0] for c in chunk_offsets])
            chunk_offsets_arr = np.array([c[1] for c in chunk_offsets])
            local_offsets = np.interp(source_times, chunk_centers, chunk_offsets_arr)
            source_timestamps = source_times + global_offset_s + local_offsets
    else:
        source_timestamps = np.array([])

    # Create alignment map
    alignment_map = AlignmentMap(
        global_offset_s=global_offset_s,
        chunk_offsets_s=chunk_offsets,
        source_timestamps_s=source_timestamps,
        fs_source=fs_source,
        fs_reference=fs_reference,
        chunk_size_s=chunk_size_s,
        correlation_score=correlation_score,
        metadata={
            'search_range_s': search_range_s,
            'envelope_freq_hz': envelope_freq_hz,
            'perform_fine_alignment': perform_fine_alignment,
            'generate_sample_timestamps': generate_sample_timestamps,
            'source_duration_s': len(signal_source) / fs_source,
            'reference_duration_s': len(signal_reference) / fs_reference,
            'num_source_samples': num_source_samples,
        }
    )

    return alignment_map
