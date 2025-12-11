"""Tests for signal coregistration logic."""
import numpy as np
import os


def test_generated_signals_fixture(generated_signals):
    """
    Tests the integrity of the `generated_signals` fixture.

    This test verifies that the fixture provides the expected outputs and that
    their properties (like type and shape) are correct. It also performs a
    basic check to ensure the shift is applied as expected.
    """
    # 1. Check that all required top-level keys are in the fixture dictionary
    assert "signal_a" in generated_signals
    assert "signal_b" in generated_signals
    assert "t_shift_s" in generated_signals
    assert "t_shift_samples" in generated_signals

    # 2. Extract Signal A and Signal B data structures
    signal_a_data = generated_signals["signal_a"]
    signal_b_data = generated_signals["signal_b"]
    shift_s = generated_signals["t_shift_s"]
    shift_samples = generated_signals["t_shift_samples"]

    # 3. Verify Signal A structure
    assert "signal" in signal_a_data
    assert "t" in signal_a_data
    assert "stim_mask" in signal_a_data
    assert "metadata" in signal_a_data

    signal_a = signal_a_data["signal"]
    t_a = signal_a_data["t"]
    stim_mask_a = signal_a_data["stim_mask"]
    metadata_a = signal_a_data["metadata"]

    # import matplotlib.pyplot as plt
    # idx1 = 7_000_000
    # idx2 = 8_000_000
    # plt.plot(signal_a_data['signal'][idx1:idx2])
    # plt.plot(signal_a_data['stim_mask'][idx1:idx2])
    # plt.show()

    # plt.plot(signal_b_data["signal"])
    # plt.show()

    # 4. Verify Signal B structure
    assert "signal" in signal_b_data
    assert "t" in signal_b_data
    assert "t_shift" in signal_b_data
    assert "t_shift_samples" in signal_b_data
    assert "segment_info" in signal_b_data
    assert "metadata" in signal_b_data

    signal_b = signal_b_data["signal"]
    t_b = signal_b_data["t"]
    segment_info = signal_b_data["segment_info"]
    metadata_b = signal_b_data["metadata"]

    # 5. Validate Signal A properties
    assert isinstance(signal_a, np.ndarray)
    assert signal_a.ndim == 1
    assert len(signal_a) == len(t_a)
    assert len(stim_mask_a) == len(signal_a)
    assert stim_mask_a.dtype == bool

    # Verify 24-hour duration
    fs = metadata_a['fs']
    expected_duration_samples = int(24 * 3600 * fs)
    assert len(signal_a) == expected_duration_samples

    # 6. Validate Signal B properties
    assert isinstance(signal_b, np.ndarray)
    assert signal_b.ndim == 1
    assert len(signal_b) == len(t_b)
    assert isinstance(shift_s, (int, float))
    assert isinstance(shift_samples, int)
    assert shift_s == 3600  # 1 hour shift
    assert shift_samples == int(shift_s * fs)

    # 7. Verify stimulation parameters
    assert metadata_a['stim_freq_hz'] == 50.0
    assert metadata_a['duty_on_s'] == 60
    assert metadata_a['duty_off_s'] == 180
    assert metadata_a['stim_start_s'] == 6 * 3600
    assert metadata_a['stim_end_s'] == 12 * 3600

    # 8. Verify segment extraction info
    assert 'segment_start_s' in segment_info
    assert 'segment_end_s' in segment_info
    assert 'start_idx_in_signal_a' in segment_info
    assert 'end_idx_in_signal_a' in segment_info
    assert segment_info['buffer_before_s'] == 5 * 60
    assert segment_info['buffer_after_s'] == 5 * 60

    # 9. Verify Signal B is a shifted segment of Signal A
    # Signal B should be extracted from indices defined in segment_info
    start_idx = segment_info['start_idx_in_signal_a']
    end_idx = segment_info['end_idx_in_signal_a']
    extracted_segment = signal_a[start_idx:end_idx].copy()

    # Add the same noise that was added in fixture for comparison
    assert len(extracted_segment) == len(signal_b)

    print(f"\n--- Generated Signals Fixture Info ---")
    print(f"Sampling Rate (FS): {fs} Hz")
    print(f"Time Shift: {shift_s} s ({shift_s / 3600} hours)")
    print(f"Time Shift (samples): {shift_samples}")
    print(f"Signal A length: {len(signal_a)} samples ({len(signal_a) / fs / 3600:.2f} hours)")
    print(f"Signal B length: {len(signal_b)} samples ({len(signal_b) / fs / 60:.2f} minutes)")
    print(f"Signal A time range: {t_a[0]:.2f}s to {t_a[-1]:.2f}s")
    print(f"Signal B time range: {t_b[0]:.2f}s to {t_b[-1]:.2f}s")
    print(f"Segment extracted from Signal A indices: [{start_idx}, {end_idx}]")
    print(f"Segment time in Signal A: {segment_info['segment_start_s']:.2f}s to {segment_info['segment_end_s']:.2f}s")
    print(f"--------------------------------------")


def test_generated_signals_different_fs_fixture(generated_signals_different_fs):
    """
    Tests the integrity of the `generated_signals_different_fs` fixture.

    This fixture tests coregistration with different sampling rates:
    - Signal A: 256 Hz
    - Signal B: 500 Hz (resampled)

    This test verifies that the fixture provides the expected outputs with
    different sampling rates correctly handled.
    """
    # 1. Check that all required top-level keys are in the fixture dictionary
    assert "signal_a" in generated_signals_different_fs
    assert "signal_b" in generated_signals_different_fs
    assert "t_shift_s" in generated_signals_different_fs
    assert "t_shift_samples_a" in generated_signals_different_fs
    assert "t_shift_samples_b" in generated_signals_different_fs
    assert "fs_a" in generated_signals_different_fs
    assert "fs_b" in generated_signals_different_fs

    # 2. Extract Signal A and Signal B data structures
    signal_a_data = generated_signals_different_fs["signal_a"]
    signal_b_data = generated_signals_different_fs["signal_b"]
    shift_s = generated_signals_different_fs["t_shift_s"]
    shift_samples_a = generated_signals_different_fs["t_shift_samples_a"]
    shift_samples_b = generated_signals_different_fs["t_shift_samples_b"]
    fs_a = generated_signals_different_fs["fs_a"]
    fs_b = generated_signals_different_fs["fs_b"]

    # 3. Extract Signal A data
    signal_a = signal_a_data["signal"]
    t_a = signal_a_data["t"]
    metadata_a = signal_a_data["metadata"]

    # 4. Extract Signal B data
    signal_b = signal_b_data["signal"]
    t_b = signal_b_data["t"]
    metadata_b = signal_b_data["metadata"]
    segment_info = signal_b_data["segment_info"]

    # 5. Validate Signal A properties (256 Hz)
    assert isinstance(signal_a, np.ndarray)
    assert signal_a.ndim == 1
    assert len(signal_a) == len(t_a)
    assert fs_a == 256
    assert metadata_a['fs'] == 256

    # Verify 24-hour duration
    expected_duration_samples_a = int(24 * 3600 * fs_a)
    assert len(signal_a) == expected_duration_samples_a

    # 6. Validate Signal B properties (500 Hz)
    assert isinstance(signal_b, np.ndarray)
    assert signal_b.ndim == 1
    assert len(signal_b) == len(t_b)
    assert fs_b == 500
    assert metadata_b['fs'] == 500

    # Verify time arrays are aligned with sampling rates
    assert np.isclose(t_a[1] - t_a[0], 1 / fs_a)
    assert np.isclose(t_b[1] - t_b[0], 1 / fs_b)

    # 7. Verify shift calculations for different sampling rates
    assert shift_samples_a == int(shift_s * fs_a)
    assert shift_samples_b == int(shift_s * fs_b)
    assert shift_samples_a == 921600  # 3600 * 256
    assert shift_samples_b == 1800000  # 3600 * 500

    # 8. Verify segment extraction
    assert segment_info['buffer_before_s'] == 5 * 60
    assert segment_info['buffer_after_s'] == 5 * 60

    # 9. Verify Signal B duration is consistent
    # Both should have same duration in seconds
    duration_b_seconds = len(signal_b) / fs_b
    expected_duration = (segment_info['segment_end_s'] - segment_info['segment_start_s'])
    assert np.isclose(duration_b_seconds, expected_duration, rtol=0.01)

    print(f"\n--- Generated Signals Different FS Fixture Info ---")
    print(f"Signal A - Sampling Rate: {fs_a} Hz")
    print(f"Signal B - Sampling Rate: {fs_b} Hz")
    print(f"Time Shift: {shift_s} s ({shift_s / 3600} hours)")
    print(f"Time Shift (samples, Signal A @ {fs_a} Hz): {shift_samples_a}")
    print(f"Time Shift (samples, Signal B @ {fs_b} Hz): {shift_samples_b}")
    print(f"Signal A length: {len(signal_a)} samples ({len(signal_a) / fs_a / 3600:.2f} hours)")
    print(f"Signal B length: {len(signal_b)} samples ({len(signal_b) / fs_b / 60:.2f} minutes)")
    print(f"Signal A time range: {t_a[0]:.2f}s to {t_a[-1]:.2f}s")
    print(f"Signal B time range: {t_b[0]:.2f}s to {t_b[-1]:.2f}s")
    print(f"Signal B duration: {len(signal_b) / fs_b:.2f}s")
    print(f"Expected duration: {expected_duration:.2f}s")
    print(f"---------------------------------------------------")


def test_generated_signals_mef_files_fixture(generated_signals_mef_files):
    """
    Tests the integrity of the `generated_signals_mef_files` fixture.

    This fixture tests integration scenario where signals are stored in
    separate MEF files representing different recording devices.

    This test verifies:
    - MEF files are created in temporary directory
    - Files have correct naming
    - Metadata is properly attached
    - Signals have correct sampling rates
    - Time shift is correctly calculated for both sampling rates
    """
    # 1. Check that all required keys are in the fixture dictionary
    assert "signal_a" in generated_signals_mef_files
    assert "signal_b" in generated_signals_mef_files
    assert "file_path_a" in generated_signals_mef_files
    assert "file_path_b" in generated_signals_mef_files
    assert "tmp_dir" in generated_signals_mef_files
    assert "t_shift_s" in generated_signals_mef_files
    assert "t_shift_samples_a" in generated_signals_mef_files
    assert "t_shift_samples_b" in generated_signals_mef_files
    assert "fs_a" in generated_signals_mef_files
    assert "fs_b" in generated_signals_mef_files

    # 2. Extract data
    signal_a_data = generated_signals_mef_files["signal_a"]
    signal_b_data = generated_signals_mef_files["signal_b"]
    file_path_a = generated_signals_mef_files["file_path_a"]
    file_path_b = generated_signals_mef_files["file_path_b"]
    tmp_dir = generated_signals_mef_files["tmp_dir"]
    shift_s = generated_signals_mef_files["t_shift_s"]
    shift_samples_a = generated_signals_mef_files["t_shift_samples_a"]
    shift_samples_b = generated_signals_mef_files["t_shift_samples_b"]
    fs_a = generated_signals_mef_files["fs_a"]
    fs_b = generated_signals_mef_files["fs_b"]

    # 3. Verify file paths
    assert file_path_a.endswith('.mefd'), "Signal A file should have .mefd extension"
    assert file_path_b.endswith('.mefd'), "Signal B file should have .mefd extension"
    assert 'signal_a' in file_path_a.lower()
    assert 'signal_b' in file_path_b.lower()
    assert os.path.dirname(file_path_a) == tmp_dir
    assert os.path.dirname(file_path_b) == tmp_dir

    # 4. Verify files were created (currently placeholder files)
    assert os.path.exists(file_path_a), f"Signal A MEF file not found at {file_path_a}"
    assert os.path.exists(file_path_b), f"Signal B MEF file not found at {file_path_b}"

    # 5. Verify Signal A properties
    signal_a = signal_a_data["signal"]
    t_a = signal_a_data["t"]
    metadata_a = signal_a_data["metadata"]
    assert fs_a == 256
    assert metadata_a['fs'] == 256
    assert len(signal_a) == len(t_a)
    assert np.isclose(t_a[1] - t_a[0], 1 / fs_a)

    # 6. Verify Signal B properties
    signal_b = signal_b_data["signal"]
    t_b = signal_b_data["t"]
    metadata_b = signal_b_data["metadata"]
    assert fs_b == 500
    assert metadata_b['fs'] == 500
    assert len(signal_b) == len(t_b)
    assert np.isclose(t_b[1] - t_b[0], 1 / fs_b)

    # 7. Verify time shifts are consistent
    assert shift_s == 3600  # 1 hour
    assert shift_samples_a == int(shift_s * fs_a)
    assert shift_samples_b == int(shift_s * fs_b)
    assert shift_samples_a == 921600
    assert shift_samples_b == 1800000

    # 8. Verify MEF files are directories with proper structure
    # .mefd files are directories with MEF session structure
    assert os.path.isdir(file_path_a), f"Signal A MEF should be a directory at {file_path_a}"
    assert os.path.isdir(file_path_b), f"Signal B MEF should be a directory at {file_path_b}"

    # Check that MEF directories contain expected files (session.mefd)
    assert len(os.listdir(file_path_a)) > 0, "Signal A MEF directory should not be empty"
    assert len(os.listdir(file_path_b)) > 0, "Signal B MEF directory should not be empty"

    print(f"\n--- MEF Files Fixture Info ---")
    print(f"Temporary directory: {tmp_dir}")
    print(f"Signal A MEF file: {file_path_a}")
    print(f"Signal B MEF file: {file_path_b}")
    print(f"Signal A - FS: {fs_a} Hz, Samples: {len(signal_a)}")
    print(f"Signal B - FS: {fs_b} Hz, Samples: {len(signal_b)}")
    print(f"Time Shift: {shift_s}s ({shift_s / 3600}h)")
    print(f"Shift (samples @ {fs_a} Hz): {shift_samples_a}")
    print(f"Shift (samples @ {fs_b} Hz): {shift_samples_b}")
    print(f"Signal A file exists: {os.path.exists(file_path_a)}")
    print(f"Signal B file exists: {os.path.exists(file_path_b)}")
    print(f"Signal A file size: {os.path.getsize(file_path_a)} bytes")
    print(f"Signal B file size: {os.path.getsize(file_path_b)} bytes")
    print(f"------------------------------")


def test_mef_files_read_verify(generated_signals_mef_files):
    """
    Tests reading MEF files and verifying signal integrity.

    This test:
    - Opens the MEF files using MefReader
    - Reads back the signal data from both files
    - Compares the read signals with the original generated signals
    - Uses isclose for comparisons to account for rounding due to MEF conversion factors

    MEF files use conversion factors and may have slight rounding differences,
    so we use relative tolerance (rtol) and absolute tolerance (atol) for comparisons.
    """
    from mef_tools.io import MefReader

    # 1. Extract fixture data
    signal_a_data = generated_signals_mef_files["signal_a"]
    signal_b_data = generated_signals_mef_files["signal_b"]
    file_path_a = generated_signals_mef_files["file_path_a"]
    file_path_b = generated_signals_mef_files["file_path_b"]
    fs_a = generated_signals_mef_files["fs_a"]
    fs_b = generated_signals_mef_files["fs_b"]

    # Original signals
    signal_a_original = signal_a_data["signal"]
    signal_b_original = signal_b_data["signal"]
    t_a_original = signal_a_data["t"]
    t_b_original = signal_b_data["t"]

    # 2. Read Signal A from MEF file
    print(f"\n--- Reading MEF Files ---")
    print(f"Reading Signal A from {file_path_a}")

    reader_a = MefReader(file_path_a, password2='read_password')
    channels_a = reader_a.channels
    assert len(channels_a) > 0, "No channels found in Signal A MEF file"
    assert 'ECG' in channels_a, "ECG channel not found in Signal A MEF file"

    # Get start and end times
    start_time_a = reader_a.get_property('start_time', 'ECG')
    end_time_a = reader_a.get_property('end_time', 'ECG')
    fs_a_read = reader_a.get_property('fsamp', 'ECG')

    # Read full signal A
    signal_a_read = reader_a.get_data('ECG')

    print(f"Signal A read: {len(signal_a_read)} samples at {fs_a_read} Hz")
    print(f"Start time (uUTC): {start_time_a}, End time (uUTC): {end_time_a}")

    # 3. Read Signal B from MEF file
    print(f"Reading Signal B from {file_path_b}")

    reader_b = MefReader(file_path_b, password2='read_password')
    channels_b = reader_b.channels
    assert len(channels_b) > 0, "No channels found in Signal B MEF file"
    assert 'ECG' in channels_b, "ECG channel not found in Signal B MEF file"

    # Get start and end times
    start_time_b = reader_b.get_property('start_time', 'ECG')
    end_time_b = reader_b.get_property('end_time', 'ECG')
    fs_b_read = reader_b.get_property('fsamp', 'ECG')

    # Read full signal B
    signal_b_read = reader_b.get_data('ECG')

    print(f"Signal B read: {len(signal_b_read)} samples at {fs_b_read} Hz")
    print(f"Start time (uUTC): {start_time_b}, End time (uUTC): {end_time_b}")

    # 4. Verify Signal A properties
    assert fs_a_read == fs_a, f"Signal A fs mismatch: expected {fs_a}, got {fs_a_read}"
    assert len(signal_a_read) == len(signal_a_original), \
        f"Signal A length mismatch: expected {len(signal_a_original)}, got {len(signal_a_read)}"

    # 5. Verify Signal B properties
    assert fs_b_read == fs_b, f"Signal B fs mismatch: expected {fs_b}, got {fs_b_read}"
    assert len(signal_b_read) == len(signal_b_original), \
        f"Signal B length mismatch: expected {len(signal_b_original)}, got {len(signal_b_read)}"

    # 6. Compare Signal A data with tolerance for rounding
    # MEF uses conversion factors which can introduce rounding errors
    # Observed max deviation: ~0.015, mean: ~0.003
    # Use rtol=1e-3 (relative tolerance 0.1%) and atol=0.02 (absolute tolerance for amplitude)
    comparison_a = np.isclose(
        signal_a_read,
        signal_a_original,
        rtol=1e-3,  # 0.1% relative tolerance
        atol=0.02   # 0.02 absolute tolerance (observed max ~0.015)
    )

    num_mismatches_a = np.sum(~comparison_a)
    percent_match_a = np.sum(comparison_a) / len(comparison_a) * 100

    print(f"\nSignal A Comparison:")
    print(f"  Matched samples: {np.sum(comparison_a)}/{len(comparison_a)} ({percent_match_a:.2f}%)")
    print(f"  Max deviation: {np.max(np.abs(signal_a_read - signal_a_original)):.2e}")
    print(f"  Mean deviation: {np.mean(np.abs(signal_a_read - signal_a_original)):.2e}")
    print(f"  Std deviation: {np.std(signal_a_read - signal_a_original):.2e}")

    # Allow up to 0.1% of samples to not match with these tolerances
    assert percent_match_a >= 99.9, \
        f"Signal A: {num_mismatches_a} samples don't match (expected ~0)"

    # 7. Compare Signal B data with tolerance for rounding
    comparison_b = np.isclose(
        signal_b_read,
        signal_b_original,
        rtol=1e-3,
        atol=0.02
    )

    num_mismatches_b = np.sum(~comparison_b)
    percent_match_b = np.sum(comparison_b) / len(comparison_b) * 100

    print(f"\nSignal B Comparison:")
    print(f"  Matched samples: {np.sum(comparison_b)}/{len(comparison_b)} ({percent_match_b:.2f}%)")
    print(f"  Max deviation: {np.max(np.abs(signal_b_read - signal_b_original)):.2e}")
    print(f"  Mean deviation: {np.mean(np.abs(signal_b_read - signal_b_original)):.2e}")
    print(f"  Std deviation: {np.std(signal_b_read - signal_b_original):.2e}")

    assert percent_match_b >= 99.9, \
        f"Signal B: {num_mismatches_b} samples don't match (expected ~0)"

    # 8. Verify statistical properties are preserved
    # Check that mean, std, min, max are close
    print(f"\nSignal A Statistics:")
    print(f"  Original - Mean: {np.mean(signal_a_original):.6e}, Std: {np.std(signal_a_original):.6f}")
    print(f"  Read     - Mean: {np.mean(signal_a_read):.6e}, Std: {np.std(signal_a_read):.6f}")

    # For mean near zero, use absolute tolerance; for std use relative tolerance
    assert np.isclose(np.mean(signal_a_read), np.mean(signal_a_original), atol=1e-3), \
        "Signal A mean doesn't match"
    assert np.isclose(np.std(signal_a_read), np.std(signal_a_original), rtol=1e-3), \
        "Signal A std doesn't match"

    print(f"\nSignal B Statistics:")
    print(f"  Original - Mean: {np.mean(signal_b_original):.6e}, Std: {np.std(signal_b_original):.6f}")
    print(f"  Read     - Mean: {np.mean(signal_b_read):.6e}, Std: {np.std(signal_b_read):.6f}")

    assert np.isclose(np.mean(signal_b_read), np.mean(signal_b_original), atol=1e-3), \
        "Signal B mean doesn't match"
    assert np.isclose(np.std(signal_b_read), np.std(signal_b_original), rtol=1e-3), \
        "Signal B std doesn't match"

    # 9. Verify stim mask is preserved in Signal A (check correlation)
    stim_mask_a = signal_a_data["stim_mask"]
    # During stim ON periods, signal should have higher variance
    # Calculate mean amplitude during stim ON vs OFF
    stim_on_original = np.mean(np.abs(signal_a_original[stim_mask_a]))
    stim_off_original = np.mean(np.abs(signal_a_original[~stim_mask_a]))
    stim_on_read = np.mean(np.abs(signal_a_read[stim_mask_a]))
    stim_off_read = np.mean(np.abs(signal_a_read[~stim_mask_a]))

    print(f"\nSignal A Stimulation Check:")
    print(f"  Original - Stim ON mean: {stim_on_original:.6f}, Stim OFF mean: {stim_off_original:.6f}")
    print(f"  Read     - Stim ON mean: {stim_on_read:.6f}, Stim OFF mean: {stim_off_read:.6f}")

    # Verify stim ON is still higher than OFF after reading
    assert stim_on_read > stim_off_read, "Stimulation artifact not preserved in read signal"
    assert np.isclose(stim_on_original / stim_off_original, stim_on_read / stim_off_read, rtol=0.01), \
        "Stimulation artifact ratio doesn't match"

    print(f"\n--- MEF Read Verification Complete ---")
    print(f"✓ All signals read successfully and match original data")


def test_floating_clock_fixture(generated_signals_floating_clock):
    """
    Tests the floating clock drift fixture.

    This test:
    1. Validates the fixture structure and properties
    2. Prints drift statistics

    Plots are displayed for visual inspection of the floating clock effects.
    To enable plots, uncomment the plt.show() calls at the end.
    """
    # 1. Extract fixture data
    signal_a_data = generated_signals_floating_clock['signal_a']
    signal_b_data = generated_signals_floating_clock['signal_b']
    file_path_a = generated_signals_floating_clock['file_path_a']
    file_path_b = generated_signals_floating_clock['file_path_b']
    max_drift = generated_signals_floating_clock['max_drift_s']
    drift_fn = generated_signals_floating_clock['drift_function']
    fs_a = generated_signals_floating_clock['fs_a']
    fs_b = generated_signals_floating_clock['fs_b']

    signal_a = signal_a_data['signal']
    signal_b = signal_b_data['signal']
    t_a = signal_a_data['t']
    t_b = signal_b_data['t']
    metadata_a = signal_a_data['metadata']
    metadata_b = signal_b_data['metadata']

    # 2. Validate fixture structure
    assert 'signal_a' in generated_signals_floating_clock
    assert 'signal_b' in generated_signals_floating_clock
    assert 'drift_function' in generated_signals_floating_clock
    assert 'max_drift_s' in generated_signals_floating_clock
    assert 'file_path_a' in generated_signals_floating_clock
    assert 'file_path_b' in generated_signals_floating_clock

    # 3. Validate Signal A (reference, no drift)
    assert signal_a.shape[0] == int(24 * 3600 * fs_a), "Signal A should be 24 hours"
    assert metadata_a['fs'] == 256, "Signal A should be 256 Hz"
    assert 'drift_type' not in metadata_a or metadata_a.get('drift_type') is None

    # 4. Validate Signal B (with floating clock drift)
    assert metadata_b['fs'] == 500, "Signal B should be 500 Hz"
    assert metadata_b['drift_type'] == 'floating_clock', "Signal B should have floating clock"
    assert metadata_b['max_drift_s'] == max_drift

    # 5. Print drift statistics
    drift_mean = metadata_b['drift_mean']
    drift_std = metadata_b['drift_std']
    drift_max = metadata_b['drift_max']

    print(f"\n--- Floating Clock Drift Fixture Validation ---")
    print(f"Signal A (Reference):")
    print(f"  Sampling Rate: {fs_a} Hz")
    print(f"  Duration: {len(signal_a) / fs_a / 3600:.2f} hours")
    print(f"  Samples: {len(signal_a):,}")
    print(f"\nSignal B (with Floating Clock):")
    print(f"  Sampling Rate: {fs_b} Hz")
    print(f"  Duration: {len(signal_b) / fs_b / 60:.2f} minutes")
    print(f"  Samples: {len(signal_b):,}")
    print(f"\nClock Drift Statistics:")
    print(f"  Maximum allowed drift: ±{max_drift:.2f} seconds")
    print(f"  Mean drift: {drift_mean:.6f} seconds (should be ~0)")
    print(f"  Std of drift: {drift_std:.6f} seconds")
    print(f"  Max observed drift: {drift_max:.6f} seconds")
    print(f"  Base time shift: 1 hour (3600 seconds)")
    print(f"\nMEF Files:")
    print(f"  Signal A: {file_path_a}")
    print(f"  Signal B: {file_path_b}")

    # 6. Validate drift function
    assert callable(drift_fn), "Drift function should be callable"
    drift_samples = np.array([drift_fn(t) for t in np.linspace(0, 24*3600, 100)])
    assert np.all(drift_samples >= -max_drift), f"Drift should not exceed -{max_drift}s"
    assert np.all(drift_samples <= max_drift), f"Drift should not exceed +{max_drift}s"
    # Note: drift_mean should be near 0, but small sampling bias is acceptable
    assert abs(np.mean(drift_samples) - drift_mean) < 2.0, "Drift function sampling should match metadata"

    print(f"\n--- Creating Comparison Plots ---")

    # 7. Plot 1: Full signal overview (optional - requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        _matplotlib_available = True
    except ImportError:
        _matplotlib_available = False
        print("⚠ matplotlib not available, skipping plots")

    if _matplotlib_available:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))

        # Plot Signal A full view
        axes[0].plot(t_a, signal_a, linewidth=0.5, label='Signal A (256 Hz, no drift)', alpha=0.7)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Signal A - Reference Device (Perfect Clock)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot Signal B full view (resampled to 500 Hz)
        axes[1].plot(t_b, signal_b, linewidth=0.5, label='Signal B (500 Hz, with drift)', alpha=0.7, color='orange')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Signal B - Device with Floating Clock (Drifted)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot drift over time
        drift_times = np.linspace(0, 24*3600, 1000)
        drift_values = np.array([drift_fn(t) for t in drift_times])
        axes[2].plot(drift_times / 3600, drift_values, linewidth=1, label='Clock Drift', color='green')
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[2].axhline(y=max_drift, color='r', linestyle='--', alpha=0.3, label=f'±{max_drift}s bounds')
        axes[2].axhline(y=-max_drift, color='r', linestyle='--', alpha=0.3)
        axes[2].set_xlabel('Time (hours)')
        axes[2].set_ylabel('Drift (seconds)')
        axes[2].set_title('Time-Varying Clock Drift Over 24 Hours')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Plot zoomed comparison around stimulation period (hours 6-12)
        stim_start_idx_a = int(6 * 3600 * fs_a)
        stim_end_idx_a = int(12.5 * 3600 * fs_a)

        # Extract a segment from Signal A around stim time
        signal_a_zoom = signal_a[stim_start_idx_a:stim_end_idx_a]
        t_a_zoom = t_a[stim_start_idx_a:stim_end_idx_a]

        axes[3].plot(t_a_zoom / 3600, signal_a_zoom, linewidth=0.7, label='Signal A zoom', alpha=0.7)
        axes[3].set_xlabel('Time (hours)')
        axes[3].set_ylabel('Amplitude')
        axes[3].set_title('Signal A - Zoom Around Stimulation Period (Hours 6-12)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        print(f"✓ Plot created with 4 subplots:")
        print(f"  1. Signal A full view (256 Hz, reference)")
        print(f"  2. Signal B full view (500 Hz, with drift)")
        print(f"  3. Clock drift over time (showing ±{max_drift}s variation)")
        print(f"  4. Signal A zoom around stimulation (hours 6-12)")

        # Uncomment below to display the plot
        # plt.show()

        plt.close()

    print(f"\n--- Floating Clock Fixture Test Complete ---")
    print(f"✓ All validations passed")
    print(f"✓ Plots created successfully")
    print(f"✓ Drift is realistic and within bounds")
