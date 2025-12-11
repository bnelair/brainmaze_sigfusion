import pytest
import numpy as np
import tempfile
import os
import time
import pandas as pd
from mef_tools.io import MefWriter, MefReader


# Helper: generate pink (1/f) noise using spectral shaping of white noise
def _generate_pink_noise(num_samples: int, random_state=None, exponent: float = 1.0):
    """
    Generate pink noise (1/f^exponent) using frequency-domain shaping.
    Uses rfft/irfft for efficiency for real-valued signals.

    Args:
        num_samples: number of samples to generate
        random_state: optional integer seed or np.random.Generator
        exponent: spectral exponent (1.0 => pink noise)

    Returns:
        numpy array of length num_samples with zero mean.
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    # generate white noise in time domain
    white = rng.normal(size=num_samples)
    # transform to frequency domain (real FFT)
    freqs = np.fft.rfftfreq(num_samples, d=1.0)
    # compute spectral scaling: avoid division by zero at DC
    # scaling = 1 / (freqs ** (exponent/2)) for rfft amplitude
    # set DC component to 0 (no DC offset)
    scaling = np.ones_like(freqs)
    nonzero = freqs > 0
    scaling[nonzero] = 1.0 / (freqs[nonzero] ** (exponent / 2.0))
    scaling[~nonzero] = 0.0

    spectrum = np.fft.rfft(white) * scaling
    pink = np.fft.irfft(spectrum, n=num_samples)
    # normalize to unit std and zero mean
    pink = pink - np.mean(pink)
    std = np.std(pink)
    if std > 0:
        pink = pink / std
    return pink



def _generate_test_signal_a(
        fs: int = 256,
        duration_s: float = 24 * 3600,
        stim_start_s: float = 6 * 3600,
        stim_end_s: float = 12 * 3600,
        stim_freq_hz: float = 50.0,
        duty_on_s: float = 60,
        duty_off_s: float = 3 * 60,
        pink_noise_seed: int = 42,
        pink_noise_amplitude: float = 20.0,
        stim_amplitude: float = 200.0,
):
    """
    Generate Signal A: 24-hour pink noise with superimposed stimulation artifact.

    This is the "reference" signal recorded by Device A with correct clock.

    Args:
        fs: Sampling rate in Hz.
        duration_s: Total duration in seconds.
        stim_start_s: Stimulation start time in seconds (relative to signal start).
        stim_end_s: Stimulation end time in seconds (relative to signal start).
        stim_freq_hz: Stimulation artifact frequency in Hz.
        duty_on_s: Duration of ON phase in duty cycle (seconds).
        duty_off_s: Duration of OFF phase in duty cycle (seconds).
        pink_noise_seed: Random seed for reproducible pink noise.
        pink_noise_amplitude: Scaling factor for pink noise amplitude (default: 20.0).
        stim_amplitude: Amplitude scaling for stimulation artifact (default: 200.0 = 10x pink noise).

    Returns:
        A dict with keys:
            - 'signal': the generated signal (1D array)
            - 't': time array (1D array)
            - 'stim_mask': boolean mask indicating where stimulation is ON
            - 'metadata': dict with generation parameters for future file I/O
    """
    # Build time array
    num_samples = int(duration_s * fs)
    t = np.arange(num_samples) / fs

    # Generate pink noise background
    pink_noise = _generate_pink_noise(num_samples, random_state=pink_noise_seed, exponent=1.0)
    signal = pink_noise_amplitude * pink_noise

    # Build stimulation mask: True where stim artifact is ON
    duty_cycle_s = duty_on_s + duty_off_s
    stim_mask = np.zeros(num_samples, dtype=bool)

    start_idx = int(stim_start_s * fs)
    end_idx = int(stim_end_s * fs)

    if start_idx < end_idx:
        t_segment = t[start_idx:end_idx]
        # Compute on/off cycles relative to stim_start
        rel_t = t_segment - stim_start_s
        cycle_pos = np.mod(rel_t, duty_cycle_s)
        on_mask_segment = cycle_pos < duty_on_s
        stim_mask[start_idx:end_idx] = on_mask_segment

    # Generate and apply stimulation artifact
    artifact = np.zeros_like(signal)
    if stim_freq_hz < fs / 2:
        # Artifact is representable at this sampling rate
        artifact[stim_mask] = stim_amplitude * np.sin(2 * np.pi * stim_freq_hz * t[stim_mask])
    else:
        # Fallback: use aliased high-frequency representation
        artifact[stim_mask] = stim_amplitude * np.sign(
            np.sin(2 * np.pi * (fs / 4) * t[stim_mask])
        )

    signal = signal + artifact

    # Store metadata for future file I/O operations
    metadata = {
        'fs': fs,
        'duration_s': duration_s,
        'stim_start_s': stim_start_s,
        'stim_end_s': stim_end_s,
        'stim_freq_hz': stim_freq_hz,
        'duty_on_s': duty_on_s,
        'duty_off_s': duty_off_s,
        'pink_noise_seed': pink_noise_seed,
        'pink_noise_amplitude': pink_noise_amplitude,
        'stim_amplitude': stim_amplitude,
    }

    return {
        'signal': signal,
        't': t,
        'stim_mask': stim_mask,
        'metadata': metadata,
    }


def _generate_test_signal_b(
        signal_a_data: dict,
        shift_s: float = 3600,
        buffer_before_s: float = 5 * 60,
        buffer_after_s: float = 5 * 60,
        measurement_noise_seed: int = 12345,
        measurement_noise_std: float = 2.0,
):
    """
    Generate Signal B: a segment of Signal A, temporally shifted to simulate
    clock skew between two recording devices.

    Signal B is recorded from 5 minutes before the stimulation starts to 5 minutes
    after the stimulation ends (in the reference frame of Signal A), but the data
    is time-shifted to account for device clock desynchronization.

    Args:
        signal_a_data: Output dict from _generate_test_signal_a.
        shift_s: Time shift between Device A and Device B in seconds.
        buffer_before_s: Duration before stim_start to include in Signal B.
        buffer_after_s: Duration after stim_end to include in Signal B.
        measurement_noise_seed: Random seed for measurement noise.
        measurement_noise_std: Standard deviation of Gaussian measurement noise
                              (smaller than pink noise amplitude for realistic simulation).

    Returns:
        A dict with keys:
            - 'signal': the extracted and shifted signal (1D array)
            - 't': time array for Signal B
            - 't_shift': the time shift applied in seconds
            - 't_shift_samples': the time shift in samples
            - 'segment_info': dict with extraction details
            - 'metadata': dict with Signal B generation parameters
    """
    signal_a = signal_a_data['signal']
    t_a = signal_a_data['t']
    fs = signal_a_data['metadata']['fs']
    stim_start_s = signal_a_data['metadata']['stim_start_s']
    stim_end_s = signal_a_data['metadata']['stim_end_s']

    # Define the segment boundaries in Signal A's timeline
    segment_start_s = stim_start_s - buffer_before_s
    segment_end_s = stim_end_s + buffer_after_s

    # Convert to sample indices (clipped to valid range)
    start_idx = int(np.clip(segment_start_s * fs, 0, len(signal_a) - 1))
    end_idx = int(np.clip(segment_end_s * fs, 0, len(signal_a)))

    # Extract segment
    signal_b = signal_a[start_idx:end_idx].copy()

    # Add measurement noise to Signal B
    rng = np.random.default_rng(measurement_noise_seed)
    extra_noise = rng.normal(scale=measurement_noise_std, size=signal_b.shape)
    signal_b = signal_b + extra_noise

    # Create time array for Signal B (adjusted by shift)
    num_samples_b = len(signal_b)
    t_b = np.arange(num_samples_b) / fs + shift_s

    # Calculate shift in samples
    shift_samples = int(shift_s * fs)

    # Store extraction details for reference
    segment_info = {
        'segment_start_s': segment_start_s,
        'segment_end_s': segment_end_s,
        'start_idx_in_signal_a': start_idx,
        'end_idx_in_signal_a': end_idx,
        'buffer_before_s': buffer_before_s,
        'buffer_after_s': buffer_after_s,
    }

    metadata = {
        'fs': fs,
        'shift_s': shift_s,
        'shift_samples': shift_samples,
        'measurement_noise_seed': measurement_noise_seed,
        'measurement_noise_std': measurement_noise_std,
    }

    return {
        'signal': signal_b,
        't': t_b,
        't_shift': shift_s,
        't_shift_samples': shift_samples,
        'segment_info': segment_info,
        'metadata': metadata,
    }


@pytest.fixture
def generated_signals():
    """
    Generates Signal A and Signal B for coregistration tests.

    Signal A (Device A - reference):
      - 24 hours of pink noise (1/f), sampled at 256 Hz
      - Superimposed 50 Hz stimulation artifact between hour 6 and hour 12
        with a duty cycle of 1 minute ON / 3 minutes OFF.
      - Stimulation amplitude is 10x the pink noise amplitude.

    Signal B (Device B - shifted):
      - Extracted from 5 minutes before stim_start to 5 minutes after stim_end
        (in Signal A's timeline)
      - Time-shifted by +1 hour to simulate clock skew between devices
      - Additional small Gaussian noise added (2.0 std, smaller than pink noise)
        to simulate measurement noise

    Returns:
        A dict with keys:
            - 'signal_a': Signal A data (dict from _generate_test_signal_a)
            - 'signal_b': Signal B data (dict from _generate_test_signal_b)
            - 't_shift_s': Time shift in seconds (for convenience)
            - 't_shift_samples': Time shift in samples (for convenience)
    """
    # Generate Signal A with default parameters
    signal_a_data = _generate_test_signal_a(
        fs=256,
        duration_s=24 * 3600,
        stim_start_s=6 * 3600,
        stim_end_s=12 * 3600,
        stim_freq_hz=50.0,  # Changed from 100 Hz to 50 Hz
        duty_on_s=60,
        duty_off_s=3 * 60,
    )

    # Generate Signal B with 1-hour shift
    signal_b_data = _generate_test_signal_b(
        signal_a_data=signal_a_data,
        shift_s=3600,  # 1 hour shift
        buffer_before_s=5 * 60,
        buffer_after_s=5 * 60,
    )

    return {
        'signal_a': signal_a_data,
        'signal_b': signal_b_data,
        't_shift_s': signal_b_data['t_shift'],
        't_shift_samples': signal_b_data['t_shift_samples'],
    }


def _resample_signal(signal: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
    """
    Resample a signal from original_fs to target_fs using linear interpolation.

    Args:
        signal: 1D signal array to resample
        original_fs: Original sampling rate in Hz
        target_fs: Target sampling rate in Hz

    Returns:
        Resampled signal array
    """
    if original_fs == target_fs:
        return signal.copy()

    num_samples_original = len(signal)
    duration_s = num_samples_original / original_fs
    num_samples_target = int(duration_s * target_fs)

    # Create original and target time arrays
    t_original = np.arange(num_samples_original) / original_fs
    t_target = np.arange(num_samples_target) / target_fs

    # Linear interpolation
    resampled = np.interp(t_target, t_original, signal)
    return resampled


@pytest.fixture
def generated_signals_different_fs():
    """
    Generates Signal A and Signal B for coregistration tests with different sampling rates.

    Signal A (Device A - reference):
      - 24 hours of pink noise (1/f), sampled at 256 Hz
      - Superimposed 50 Hz stimulation artifact between hour 6 and hour 12
        with a duty cycle of 1 minute ON / 3 minutes OFF.
      - Stimulation amplitude is 10x the pink noise amplitude.

    Signal B (Device B - shifted, higher sampling rate):
      - Extracted from 5 minutes before stim_start to 5 minutes after stim_end
        (in Signal A's timeline)
      - Time-shifted by +1 hour to simulate clock skew between devices
      - Resampled to 500 Hz (different from Device A's 256 Hz)
      - Additional small Gaussian noise added (2.0 std, smaller than pink noise)
        to simulate measurement noise

    Returns:
        A dict with keys:
            - 'signal_a': Signal A data (dict from _generate_test_signal_a)
            - 'signal_b': Signal B data (dict with signal resampled to 500 Hz)
            - 't_shift_s': Time shift in seconds (for convenience)
            - 't_shift_samples_a': Time shift in samples for Signal A (256 Hz)
            - 't_shift_samples_b': Time shift in samples for Signal B (500 Hz)
            - 'fs_a': Sampling rate of Signal A (256 Hz)
            - 'fs_b': Sampling rate of Signal B (500 Hz)
    """
    # Generate Signal A with 256 Hz
    signal_a_data = _generate_test_signal_a(
        fs=256,
        duration_s=24 * 3600,
        stim_start_s=6 * 3600,
        stim_end_s=12 * 3600,
        stim_freq_hz=50.0,
        duty_on_s=60,
        duty_off_s=3 * 60,
    )

    # Generate Signal B with 256 Hz first
    signal_b_data = _generate_test_signal_b(
        signal_a_data=signal_a_data,
        shift_s=3600,  # 1 hour shift
        buffer_before_s=5 * 60,
        buffer_after_s=5 * 60,
    )

    # Resample Signal B from 256 Hz to 500 Hz
    fs_b_target = 500
    signal_b_resampled = _resample_signal(signal_b_data['signal'], 256, fs_b_target)

    # Create new time array for resampled Signal B
    num_samples_b_resampled = len(signal_b_resampled)
    t_b_resampled = np.arange(num_samples_b_resampled) / fs_b_target + signal_b_data['t_shift']

    # Calculate shift in samples for Signal B at 500 Hz
    shift_samples_b = int(signal_b_data['t_shift'] * fs_b_target)

    # Update Signal B metadata to reflect the new sampling rate
    signal_b_data_updated = signal_b_data.copy()
    signal_b_data_updated['signal'] = signal_b_resampled
    signal_b_data_updated['t'] = t_b_resampled
    signal_b_data_updated['metadata']['fs'] = fs_b_target
    signal_b_data_updated['metadata']['shift_samples'] = shift_samples_b

    # Calculate shift in samples for Signal A at 256 Hz
    shift_samples_a = int(signal_b_data['t_shift'] * 256)

    return {
        'signal_a': signal_a_data,
        'signal_b': signal_b_data_updated,
        't_shift_s': signal_b_data['t_shift'],
        't_shift_samples_a': shift_samples_a,
        't_shift_samples_b': shift_samples_b,
        'fs_a': 256,
        'fs_b': fs_b_target,
    }


@pytest.fixture
def generated_signals_mef_files(tmp_path):
    """
    Generates Signal A and Signal B as separate temporary MEF files.

    This fixture creates realistic test data scenario where:
    - Signal A and Signal B are stored in separate MEF files
    - Each file represents recording from different devices
    - Devices have different sampling rates (256 Hz vs 500 Hz)
    - Devices have clock skew (1 hour offset)

    Uses temporary directory for file storage (auto-cleaned after test).

    Returns:
        A dict with keys:
            - 'signal_a': Signal A data dict (from _generate_test_signal_a)
            - 'signal_b': Signal B data dict (from _generate_test_signal_b_different_fs)
            - 'file_path_a': Path to Signal A MEF file
            - 'file_path_b': Path to Signal B MEF file
            - 'tmp_dir': Temporary directory path
            - 't_shift_s': Time shift in seconds
            - 't_shift_samples_a': Time shift in samples for Signal A (256 Hz)
            - 't_shift_samples_b': Time shift in samples for Signal B (500 Hz)
            - 'fs_a': Sampling rate of Signal A (256 Hz)
            - 'fs_b': Sampling rate of Signal B (500 Hz)
    """
    # Generate signals with different sampling rates
    signal_a_data = _generate_test_signal_a(
        fs=256,
        duration_s=24 * 3600,
        stim_start_s=6 * 3600,
        stim_end_s=12 * 3600,
        stim_freq_hz=50.0,
        duty_on_s=60,
        duty_off_s=3 * 60,
    )

    # Generate Signal B and resample to 500 Hz
    signal_b_data = _generate_test_signal_b(
        signal_a_data=signal_a_data,
        shift_s=3600,  # 1 hour shift
        buffer_before_s=5 * 60,
        buffer_after_s=5 * 60,
    )

    # Resample Signal B from 256 Hz to 500 Hz
    fs_b_target = 500
    signal_b_resampled = _resample_signal(signal_b_data['signal'], 256, fs_b_target)

    # Create new time array for resampled Signal B
    num_samples_b_resampled = len(signal_b_resampled)
    t_b_resampled = np.arange(num_samples_b_resampled) / fs_b_target + signal_b_data['t_shift']

    # Calculate shift in samples for Signal B at 500 Hz
    shift_samples_b = int(signal_b_data['t_shift'] * fs_b_target)

    # Update Signal B data with resampled values
    signal_b_data_updated = signal_b_data.copy()
    signal_b_data_updated['signal'] = signal_b_resampled
    signal_b_data_updated['t'] = t_b_resampled
    signal_b_data_updated['metadata']['fs'] = fs_b_target
    signal_b_data_updated['metadata']['shift_samples'] = shift_samples_b

    # Calculate shift in samples for Signal A at 256 Hz
    shift_samples_a = int(signal_b_data['t_shift'] * 256)

    # Create file paths in temporary directory
    # MEF format requires .mefd suffix for session directories
    file_path_a = os.path.join(str(tmp_path), 'signal_a.mefd')
    file_path_b = os.path.join(str(tmp_path), 'signal_b.mefd')

    # Write signals to MEF files
    # Use the same channel name 'ECG' in both files to enable coregistration
    _write_signal_to_mef_file(
        signal=signal_a_data['signal'],
        sampling_rate=256,
        channel_name='ECG',
        output_path=file_path_a,
        metadata=signal_a_data['metadata']
    )

    _write_signal_to_mef_file(
        signal=signal_b_data_updated['signal'],
        sampling_rate=fs_b_target,
        channel_name='ECG',
        output_path=file_path_b,
        metadata=signal_b_data_updated['metadata']
    )

    return {
        'signal_a': signal_a_data,
        'signal_b': signal_b_data_updated,
        'file_path_a': file_path_a,
        'file_path_b': file_path_b,
        'tmp_dir': str(tmp_path),
        't_shift_s': signal_b_data['t_shift'],
        't_shift_samples_a': shift_samples_a,
        't_shift_samples_b': shift_samples_b,
        'fs_a': 256,
        'fs_b': fs_b_target,
    }


def _apply_clock_drift(
        signal: np.ndarray,
        sampling_rate: int,
        base_shift_s: float = 3600,
        max_drift_s: float = 10.0,
        drift_seed: int = 54321,
) -> tuple:
    """
    Apply time-varying clock drift to a signal to simulate floating clock.

    Simulates crystal drift in recording device: average 0 drift with ±max_drift_s
    over 24 hours. The drift is applied as a smooth, continuous time-varying shift.

    Args:
        signal: 1D signal array (original sampled data)
        sampling_rate: Sampling rate in Hz
        base_shift_s: Base constant time shift in seconds (e.g., 1 hour)
        max_drift_s: Maximum drift in seconds (e.g., ±10s over 24 hours)
        drift_seed: Random seed for reproducibility

    Returns:
        Tuple containing:
        - resampled_signal: Signal with time-varying clock applied
        - drift_function: Function drift(t) returning drift in seconds at time t
        - metadata: Dict with drift parameters
    """
    num_samples = len(signal)
    duration_s = num_samples / sampling_rate

    # Create smooth drift function using sine wave for continuous drift
    # Zero mean, amplitude = max_drift_s, period = duration_s (full cycle over record)
    rng = np.random.default_rng(drift_seed)

    # Use sum of sinusoids for more realistic drift pattern
    # Primary frequency: one full cycle over 24 hours
    primary_freq = 1.0 / duration_s  # Hz
    secondary_freq = 3.0 / duration_s  # 3 cycles (faster variations)
    tertiary_freq = 0.1 / duration_s  # Slower underlying trend

    # Generate time array in seconds
    t = np.arange(num_samples) / sampling_rate

    # Create smooth drift with multiple frequency components
    drift_primary = np.sin(2 * np.pi * primary_freq * t)
    drift_secondary = 0.3 * np.sin(2 * np.pi * secondary_freq * t + np.pi/4)
    drift_tertiary = 0.2 * np.sin(2 * np.pi * tertiary_freq * t + np.pi/3)

    # Combine and normalize to max_drift_s
    drift_combined = drift_primary + drift_secondary + drift_tertiary
    drift_combined = drift_combined / np.max(np.abs(drift_combined))  # Normalize to [-1, 1]
    drift_combined = drift_combined * max_drift_s  # Scale to ±max_drift_s

    # Add small random perturbations (Brownian motion-like)
    noise_std = max_drift_s * 0.1  # 10% of max drift
    drift_noise = rng.normal(0, noise_std, num_samples)
    drift_noise = np.cumsum(drift_noise) / num_samples  # Smooth it out

    # Combine smooth drift with noise, ensure zero mean
    total_drift = drift_combined + drift_noise
    total_drift = total_drift - np.mean(total_drift)  # Zero mean drift

    # Clip to ensure max drift limit
    total_drift = np.clip(total_drift, -max_drift_s, max_drift_s)

    # Create time array with base shift and time-varying drift
    # Total time shift at each sample: base_shift_s + total_drift
    total_shift = base_shift_s + total_drift

    # Create output time array
    t_original = np.arange(num_samples) / sampling_rate
    t_shifted = t_original + total_shift

    # Resample signal using the shifted time array
    # Extend original time array slightly to handle edge cases
    t_extended = np.concatenate([[-1/sampling_rate], t_original, [t_original[-1] + 1/sampling_rate]])
    signal_extended = np.concatenate([[signal[0]], signal, [signal[-1]]])

    # Linear interpolation to get resampled signal
    resampled_signal = np.interp(t_shifted, t_extended, signal_extended)

    # Define drift function for reference
    def drift_function(time_s):
        """Return drift in seconds at given time."""
        idx = int(np.clip(time_s * sampling_rate, 0, num_samples - 1))
        return total_drift[idx]

    metadata = {
        'base_shift_s': base_shift_s,
        'max_drift_s': max_drift_s,
        'drift_type': 'floating_clock',
        'drift_seed': drift_seed,
        'primary_freq_hz': primary_freq,
        'secondary_freq_hz': secondary_freq,
        'drift_mean': np.mean(total_drift),
        'drift_std': np.std(total_drift),
        'drift_max': np.max(np.abs(total_drift)),
    }

    return resampled_signal, drift_function, metadata


@pytest.fixture
def generated_signals_floating_clock(tmp_path):
    """
    Generates Signal A and Signal B with floating clock drift simulation.

    This fixture creates a realistic multi-device scenario where:
    - Signal A: Reference device (256 Hz, perfect clock)
    - Signal B: Device with floating clock
      - Base time shift: +1 hour
      - Clock drift: ±10 seconds over 24 hours (simulates crystal drift)
      - Sampling rate: 500 Hz (different from Device A)
      - Time-varying shift simulates realistic device desynchronization

    Uses temporary directory for MEF file storage (auto-cleaned after test).

    Returns:
        A dict with keys:
            - 'signal_a': Signal A data dict (256 Hz, no drift)
            - 'signal_b': Signal B data dict (500 Hz, with clock drift)
            - 'file_path_a': Path to Signal A MEF file
            - 'file_path_b': Path to Signal B MEF file
            - 'tmp_dir': Temporary directory path
            - 't_shift_s': Base time shift in seconds (3600)
            - 't_shift_samples_a': Base shift in samples for Signal A (256 Hz)
            - 't_shift_samples_b': Base shift in samples for Signal B (500 Hz)
            - 'max_drift_s': Maximum drift in seconds (±10)
            - 'drift_function': Function to get drift at any time
            - 'fs_a': Signal A sampling rate (256 Hz)
            - 'fs_b': Signal B sampling rate (500 Hz)
    """
    # Generate Signal A with 256 Hz (24 hours, no drift)
    signal_a_data = _generate_test_signal_a(
        fs=256,
        duration_s=24 * 3600,
        stim_start_s=6 * 3600,
        stim_end_s=12 * 3600,
        stim_freq_hz=50.0,
        duty_on_s=60,
        duty_off_s=3 * 60,
    )

    # Generate Signal B without drift first (base version at 256 Hz)
    signal_b_data_base = _generate_test_signal_b(
        signal_a_data=signal_a_data,
        shift_s=3600,  # 1 hour base shift
        buffer_before_s=5 * 60,
        buffer_after_s=5 * 60,
    )

    # Apply floating clock drift to Signal B
    # Maximum drift: ±10 seconds over 24 hours
    signal_b_with_drift, drift_fn, drift_metadata = _apply_clock_drift(
        signal=signal_b_data_base['signal'],
        sampling_rate=256,
        base_shift_s=3600,  # 1 hour base shift
        max_drift_s=10.0,   # ±10 seconds drift
        drift_seed=54321,
    )

    # Resample Signal B with drift to 500 Hz
    fs_b_target = 500
    signal_b_resampled = _resample_signal(signal_b_with_drift, 256, fs_b_target)

    # Create new time array for resampled Signal B
    num_samples_b_resampled = len(signal_b_resampled)
    # Time array includes base shift only (drift is already in the signal)
    t_b_resampled = np.arange(num_samples_b_resampled) / fs_b_target + 3600

    # Calculate shifts in samples
    shift_samples_a = int(3600 * 256)
    shift_samples_b = int(3600 * fs_b_target)

    # Update Signal B data with drift-applied signal
    signal_b_data_updated = signal_b_data_base.copy()
    signal_b_data_updated['signal'] = signal_b_resampled
    signal_b_data_updated['t'] = t_b_resampled
    signal_b_data_updated['metadata']['fs'] = fs_b_target
    signal_b_data_updated['metadata'].update(drift_metadata)
    signal_b_data_updated['drift_function'] = drift_fn

    # Create MEF file paths
    file_path_a = os.path.join(str(tmp_path), 'signal_a_floating_clock.mefd')
    file_path_b = os.path.join(str(tmp_path), 'signal_b_floating_clock.mefd')

    # Write to MEF files
    # Signal A: Reference signal with perfect clock (no drift)
    _write_signal_to_mef_file(
        signal=signal_a_data['signal'],
        sampling_rate=256,
        channel_name='Device_A_Reference',
        output_path=file_path_a,
        metadata=signal_a_data['metadata']
    )

    # Signal B: Contains the drifted and resampled signal
    # signal_b_resampled = resampled version of signal_b_with_drift
    # signal_b_with_drift = result of _apply_clock_drift() with time-varying clock
    # The MEF file contains the signal with floating clock effects applied
    _write_signal_to_mef_file(
        signal=signal_b_resampled,  # Drifted signal, resampled to 500 Hz
        sampling_rate=fs_b_target,
        channel_name='Device_B_FloatingClock',
        output_path=file_path_b,
        metadata=signal_b_data_updated['metadata']
    )

    return {
        'signal_a': signal_a_data,
        'signal_b': signal_b_data_updated,
        'file_path_a': file_path_a,
        'file_path_b': file_path_b,
        'tmp_dir': str(tmp_path),
        't_shift_s': 3600,
        't_shift_samples_a': shift_samples_a,
        't_shift_samples_b': shift_samples_b,
        'max_drift_s': 10.0,
        'drift_function': drift_fn,
        'fs_a': 256,
        'fs_b': fs_b_target,
    }


def _write_signal_to_mef_file(
        signal: np.ndarray,
        sampling_rate: int,
        channel_name: str,
        output_path: str,
        metadata: dict = None,
) -> None:
    """
    Write a signal to a MEF (Medical Data Stream Format) file.

    This function serves as a placeholder for MEF file writing using mef-tools.
    The actual implementation should:
    - Create MEF3 format compatible files
    - Include channel metadata (sampling rate, channel name, etc.)
    - Store signal data with appropriate data type (float32 or float64)
    - Include session metadata for reproducibility

    Args:
        signal: 1D numpy array of signal samples
        sampling_rate: Sampling rate in Hz
        channel_name: Name of the recording channel/device
        output_path: Output file path (e.g., '/tmp/signal.mef')
        metadata: Optional dict with generation parameters to include in file

    Returns:
        None

    Example structure (to be completed):
        - from meftools import MEFFile, Channel
        - Create MEF session
        - Create channel with metadata
        - Write signal data
        - Close file properly

    Current implementation:
        - PLACEHOLDER: File writing logic to be implemented
        - For now, creates empty file as marker
    """
    from mef_tools.io import MefWriter
    import time

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Get start time in uUTC (microseconds since epoch)
    # Use metadata start time if available, otherwise use current time
    if metadata and 'start_time_utc' in metadata:
        start_time_utc = metadata['start_time_utc']
    else:
        start_time_utc = int(time.time() * 1e6)

    # Password for MEF file (test passwords - can be empty or custom)
    password_write = 'write_password'
    password_read = 'read_password'

    # Create MEF writer with overwrite enabled
    writer = MefWriter(
        output_path,
        overwrite=True,
        password1=password_write,
        password2=password_read
    )

    # Set MEF block length to 1 second (in samples)
    writer.mef_block_len = int(sampling_rate)

    # Set maximum NaNs to write (0 means allow all)
    writer.max_nans_written = 0

    # Write signal data to MEF file
    # The write_data method handles the MEF file format and compression
    # Signature: write_data(data, channel, start_uutc, sampling_freq, reload_metadata=False)
    writer.write_data(
        signal,
        channel_name,
        start_uutc=start_time_utc,
        sampling_freq=sampling_rate,
        precision=3,
        reload_metadata=False,
    )
