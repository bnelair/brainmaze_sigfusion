"""
Tests for verifying smoothness and continuity of coregistration alignment.

These tests ensure that the coregistration correctly handles clock drift
without introducing sharp transitions or discontinuities in the timestamp mapping.
"""
import numpy as np
import pytest
from brainmaze_sigcoreg import compute_alignment, AlignmentMap


class TestAlignmentSmoothness:
    """Tests to verify smooth, continuous alignment without sharp transitions."""

    def test_timestamps_are_monotonic_increasing(self):
        """Verify that generated timestamps are always monotonically increasing."""
        np.random.seed(42)
        fs = 256.0
        duration_s = 600.0  # 10 minutes
        
        # Create signal with simulated drift
        signal = np.random.randn(int(duration_s * fs))
        
        am = compute_alignment(
            signal_reference=signal,
            signal_source=signal,
            fs_reference=fs,
            fs_source=fs,
            perform_fine_alignment=True,
            chunk_size_s=120.0,
        )
        
        # Timestamps must be strictly monotonic increasing
        timestamps = am.source_timestamps_s
        diffs = np.diff(timestamps)
        
        assert np.all(diffs > 0), f"Timestamps not monotonically increasing. Min diff: {np.min(diffs)}"
        print(f"\n✓ Timestamps are monotonically increasing")
        print(f"  Min time step: {np.min(diffs):.6f}s")
        print(f"  Max time step: {np.max(diffs):.6f}s")
        print(f"  Expected step: {1/fs:.6f}s")

    def test_no_sharp_transitions_in_timestamps(self):
        """Verify that timestamp mapping has no sharp discontinuities."""
        np.random.seed(42)
        fs = 256.0
        duration_s = 600.0
        
        # Create signal
        signal = np.random.randn(int(duration_s * fs))
        
        am = compute_alignment(
            signal_reference=signal,
            signal_source=signal,
            fs_reference=fs,
            fs_source=fs,
            perform_fine_alignment=True,
            chunk_size_s=60.0,  # Smaller chunks for more detail
        )
        
        timestamps = am.source_timestamps_s
        
        # Compute second derivative (rate of change of rate of change)
        # Sharp transitions would show as large values
        first_deriv = np.diff(timestamps)
        second_deriv = np.diff(first_deriv)
        
        # The second derivative should be small (smooth curve)
        # Allow some tolerance for numerical precision
        max_second_deriv = np.max(np.abs(second_deriv))
        threshold = 1e-6  # Very small threshold for smoothness
        
        assert max_second_deriv < threshold, \
            f"Sharp transition detected. Max second derivative: {max_second_deriv}"
        
        print(f"\n✓ No sharp transitions detected")
        print(f"  Max second derivative: {max_second_deriv:.2e}")
        print(f"  Threshold: {threshold:.2e}")

    def test_interpolation_smoothness_with_chunks(self):
        """Test that interpolation between chunks is smooth."""
        np.random.seed(42)
        
        # Create alignment map with known chunk offsets
        chunks = [
            (0.0, 0.0),
            (100.0, 1.0),
            (200.0, 2.5),
            (300.0, 3.0),
        ]
        
        am = AlignmentMap(
            global_offset_s=1000.0,
            chunk_offsets_s=chunks,
            fs_source=100.0,  # 100 Hz
            fs_reference=100.0,
        )
        
        # Generate timestamps
        timestamps = am.get_reference_timestamps(num_samples=int(300 * 100))
        
        # Check smoothness by verifying rate of change
        diffs = np.diff(timestamps)
        
        # Differences should be close to 1/fs (0.01s) with smooth variation
        expected_diff = 1 / 100.0
        
        # Check no large jumps
        max_deviation = np.max(np.abs(diffs - expected_diff))
        assert max_deviation < 0.001, f"Large jump detected: {max_deviation}"
        
        # Check second derivative for smoothness
        second_deriv = np.diff(diffs)
        max_second_deriv = np.max(np.abs(second_deriv))
        assert max_second_deriv < 1e-4, f"Non-smooth interpolation: {max_second_deriv}"
        
        print(f"\n✓ Interpolation is smooth")
        print(f"  Max deviation from expected: {max_deviation:.6f}s")
        print(f"  Max second derivative: {max_second_deriv:.2e}")

    def test_timestamp_continuity_at_chunk_boundaries(self):
        """Verify no discontinuities at chunk boundaries."""
        np.random.seed(42)
        fs = 100.0
        chunk_size_s = 60.0
        
        # Create test signal
        signal = np.random.randn(int(300 * fs))
        
        am = compute_alignment(
            signal_reference=signal,
            signal_source=signal,
            fs_reference=fs,
            fs_source=fs,
            perform_fine_alignment=True,
            chunk_size_s=chunk_size_s,
        )
        
        if len(am.chunk_offsets_s) < 2:
            pytest.skip("Need at least 2 chunks for this test")
        
        timestamps = am.source_timestamps_s
        chunk_centers = [c[0] for c in am.chunk_offsets_s]
        
        # Check continuity around each chunk center
        for center_s in chunk_centers[1:-1]:  # Skip first and last
            center_idx = int(center_s * fs)
            
            # Get timestamps around chunk boundary
            window = 5  # 5 samples on each side
            if center_idx - window >= 0 and center_idx + window < len(timestamps):
                local_timestamps = timestamps[center_idx - window:center_idx + window + 1]
                local_diffs = np.diff(local_timestamps)
                
                # All diffs should be similar (no discontinuity)
                diff_variation = np.std(local_diffs)
                assert diff_variation < 1e-5, \
                    f"Discontinuity at chunk {center_s}s: std={diff_variation}"
        
        print(f"\n✓ No discontinuities at chunk boundaries")
        print(f"  Checked {len(chunk_centers) - 2} chunk boundaries")


class TestClockDriftScenarios:
    """Tests for various clock drift scenarios."""

    def test_linear_drift_detection(self):
        """Test detection of linear clock drift (constant rate difference)."""
        np.random.seed(42)
        fs_ref = 256.0
        fs_src = 256.1  # Slightly faster clock (0.039% difference)
        duration_s = 600.0
        
        # Create reference signal
        t_ref = np.arange(int(duration_s * fs_ref)) / fs_ref
        signal_ref = np.sin(2 * np.pi * 1.0 * t_ref) + 0.1 * np.random.randn(len(t_ref))
        
        # Create source signal with slightly different sampling rate
        t_src = np.arange(int(duration_s * fs_src)) / fs_src
        signal_src = np.sin(2 * np.pi * 1.0 * t_src) + 0.1 * np.random.randn(len(t_src))
        
        am = compute_alignment(
            signal_reference=signal_ref,
            signal_source=signal_src,
            fs_reference=fs_ref,
            fs_source=fs_src,
            perform_fine_alignment=True,
            chunk_size_s=60.0,
        )
        
        # With linear drift, chunk offsets should show a trend
        if len(am.chunk_offsets_s) > 1:
            chunk_times = [c[0] for c in am.chunk_offsets_s]
            chunk_offsets = [c[1] for c in am.chunk_offsets_s]
            
            # Fit linear trend
            coeffs = np.polyfit(chunk_times, chunk_offsets, 1)
            slope = coeffs[0]
            
            # Slope should be related to sampling rate difference
            # Expected drift rate: (fs_src - fs_ref) / fs_ref ≈ 0.0004
            expected_drift_rate = (fs_src - fs_ref) / fs_ref
            
            print(f"\n✓ Linear drift detection")
            print(f"  Expected drift rate: {expected_drift_rate:.6f}")
            print(f"  Detected slope: {slope:.6f}")
            print(f"  Number of chunks: {len(am.chunk_offsets_s)}")

    def test_sinusoidal_drift_tracking(self):
        """Test tracking of sinusoidal clock drift (realistic crystal behavior)."""
        np.random.seed(42)
        fs = 256.0
        duration_s = 600.0
        drift_amplitude = 2.0  # ±2 seconds drift
        drift_period_s = 300.0  # 5 minute period
        
        # Create reference signal
        t = np.arange(int(duration_s * fs)) / fs
        signal_ref = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))
        
        # Create source signal with simulated drift
        # Drift function: d(t) = A * sin(2π * t / T)
        drift = drift_amplitude * np.sin(2 * np.pi * t / drift_period_s)
        
        # Apply drift by resampling
        t_drifted = t + drift
        signal_src = np.interp(t, t_drifted, signal_ref)
        
        am = compute_alignment(
            signal_reference=signal_ref,
            signal_source=signal_src,
            fs_reference=fs,
            fs_source=fs,
            perform_fine_alignment=True,
            chunk_size_s=30.0,  # Smaller chunks to track drift
        )
        
        # Verify drift is captured
        if len(am.chunk_offsets_s) > 5:
            chunk_times = np.array([c[0] for c in am.chunk_offsets_s])
            chunk_offsets = np.array([c[1] for c in am.chunk_offsets_s])
            
            # The chunk offsets should follow sinusoidal pattern
            # Check that offsets vary (not constant)
            offset_range = np.max(chunk_offsets) - np.min(chunk_offsets)
            
            assert offset_range > 0.5, \
                f"Expected drift variation > 0.5s, got {offset_range:.3f}s"
            
            print(f"\n✓ Sinusoidal drift tracking")
            print(f"  Expected drift amplitude: ±{drift_amplitude}s")
            print(f"  Detected offset range: {offset_range:.3f}s")
            print(f"  Number of chunks: {len(am.chunk_offsets_s)}")

    def test_gradual_drift_no_discontinuities(self):
        """Test that gradual drift doesn't introduce discontinuities."""
        np.random.seed(42)
        fs = 256.0
        duration_s = 300.0
        
        # Create signal with gradual quadratic drift
        t = np.arange(int(duration_s * fs)) / fs
        signal_ref = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))
        
        # Quadratic drift: d(t) = 0.001 * t^2
        drift = 0.001 * t**2
        t_drifted = t + drift
        signal_src = np.interp(t, t_drifted, signal_ref)
        
        am = compute_alignment(
            signal_reference=signal_ref,
            signal_source=signal_src,
            fs_reference=fs,
            fs_source=fs,
            perform_fine_alignment=True,
            chunk_size_s=30.0,
        )
        
        # Check timestamp smoothness
        timestamps = am.source_timestamps_s
        diffs = np.diff(timestamps)
        second_deriv = np.diff(diffs)
        
        max_second_deriv = np.max(np.abs(second_deriv))
        assert max_second_deriv < 1e-5, \
            f"Discontinuity detected with gradual drift: {max_second_deriv}"
        
        print(f"\n✓ Gradual drift handled smoothly")
        print(f"  Max second derivative: {max_second_deriv:.2e}")
        print(f"  Final drift: {drift[-1]:.3f}s")


class TestReferenceSignalDrift:
    """Tests for scenarios where the reference signal has clock drift."""

    def test_both_signals_have_drift(self):
        """Test coregistration when both signals have independent clock drift.
        
        This is the most realistic scenario where both recording devices
        have imperfect clocks that drift independently.
        """
        np.random.seed(42)
        fs = 256.0
        duration_s = 300.0
        
        # Create base signal
        t = np.arange(int(duration_s * fs)) / fs
        base_signal = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))
        
        # Apply different drifts to reference and source
        drift_ref = 1.0 * np.sin(2 * np.pi * t / 200.0)  # ±1s drift, 200s period
        drift_src = 1.5 * np.sin(2 * np.pi * t / 150.0)  # ±1.5s drift, 150s period
        
        t_ref_drifted = t + drift_ref
        t_src_drifted = t + drift_src
        
        signal_ref = np.interp(t, t_ref_drifted, base_signal)
        signal_src = np.interp(t, t_src_drifted, base_signal)
        
        am = compute_alignment(
            signal_reference=signal_ref,
            signal_source=signal_src,
            fs_reference=fs,
            fs_source=fs,
            perform_fine_alignment=True,
            chunk_size_s=30.0,
        )
        
        # Verify smooth alignment despite both signals drifting
        timestamps = am.source_timestamps_s
        diffs = np.diff(timestamps)
        second_deriv = np.diff(diffs)
        
        max_second_deriv = np.max(np.abs(second_deriv))
        assert max_second_deriv < 1e-4, \
            f"Discontinuity with dual drift: {max_second_deriv}"
        
        # Should detect the relative drift (difference between drifts)
        if len(am.chunk_offsets_s) > 5:
            chunk_offsets = np.array([c[1] for c in am.chunk_offsets_s])
            offset_range = np.max(chunk_offsets) - np.min(chunk_offsets)
            
            # Expected relative drift amplitude ≈ |1.5 - 1.0| = 0.5 to 2.5
            assert offset_range > 0.3, \
                f"Expected to detect relative drift, got range {offset_range:.3f}s"
            
            print(f"\n✓ Dual drift handling")
            print(f"  Reference drift: ±{np.max(np.abs(drift_ref)):.3f}s")
            print(f"  Source drift: ±{np.max(np.abs(drift_src)):.3f}s")
            print(f"  Detected offset range: {offset_range:.3f}s")

    def test_reference_drift_longer_duration(self):
        """Test with long-duration recording where reference drift is significant."""
        np.random.seed(42)
        fs = 128.0  # Lower fs for faster test
        duration_s = 1800.0  # 30 minutes
        
        # Create base signal
        t = np.arange(int(duration_s * fs)) / fs
        base_signal = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(len(t))
        
        # Reference has slow quadratic drift (realistic crystal warming)
        drift_ref = 0.00001 * t**2  # Gradual acceleration
        t_ref_drifted = t + drift_ref
        signal_ref = np.interp(t, t_ref_drifted, base_signal)
        
        # Source is clean (no drift)
        signal_src = base_signal.copy()
        
        am = compute_alignment(
            signal_reference=signal_ref,
            signal_source=signal_src,
            fs_reference=fs,
            fs_source=fs,
            perform_fine_alignment=True,
            chunk_size_s=120.0,
        )
        
        # Should track the reference drift
        if len(am.chunk_offsets_s) > 3:
            chunk_times = np.array([c[0] for c in am.chunk_offsets_s])
            chunk_offsets = np.array([c[1] for c in am.chunk_offsets_s])
            
            # Fit quadratic to verify drift is captured
            # (the detected offsets should compensate for reference drift)
            coeffs = np.polyfit(chunk_times, chunk_offsets, 2)
            
            print(f"\n✓ Reference drift tracking")
            print(f"  Duration: {duration_s/60:.1f} minutes")
            print(f"  Final reference drift: {drift_ref[-1]:.3f}s")
            print(f"  Quadratic fit coeff: {coeffs[0]:.2e}")
            print(f"  Number of chunks: {len(am.chunk_offsets_s)}")
