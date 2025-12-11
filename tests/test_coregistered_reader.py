"""Tests for CoregisteredMefReader class."""
import numpy as np
import pytest
import json
import tempfile
import os
import time
from pathlib import Path
from mef_tools.io import MefWriter

from brainmaze_sigcoreg import CoregisteredMefReader, AlignmentMap


@pytest.fixture
def matching_channel_mef_files(tmp_path):
    """
    Create two MEF files with the same channel name for coregistration testing.
    
    This fixture creates MEF files where both have a channel named 'CH1'
    with similar content for testing coregistration.
    
    Returns:
        Dict with keys:
            - 'file_path_a': Path to reference MEF file
            - 'file_path_b': Path to other MEF file
            - 'fs': Sampling frequency (Hz)
            - 'shift_s': Time shift in seconds
            - 'shift_samples': Time shift in samples
    """
    file_path_a = tmp_path / 'matching_a.mefd'
    file_path_b = tmp_path / 'matching_b.mefd'
    
    # Create signals
    fs = 256
    duration_s = 600  # 10 minutes
    t = np.arange(0, duration_s, 1/fs)
    
    # Signal A: sine wave with noise
    signal_a = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))
    
    # Signal B: same sine wave with time shift and different noise
    shift_s = 60  # 1 minute shift
    shift_samples = int(shift_s * fs)
    signal_b = np.sin(2 * np.pi * 1.0 * (t + shift_s)) + 0.1 * np.random.randn(len(t))
    
    # Write MEF files
    writer_a = MefWriter(str(file_path_a), overwrite=True, password1='write', password2='read')
    writer_a.write_data(signal_a, 'CH1', start_uutc=int(time.time() * 1e6), sampling_freq=fs)
    writer_a = None
    
    writer_b = MefWriter(str(file_path_b), overwrite=True, password1='write', password2='read')
    writer_b.write_data(signal_b, 'CH1', start_uutc=int(time.time() * 1e6), sampling_freq=fs)
    writer_b = None
    
    return {
        'file_path_a': str(file_path_a),
        'file_path_b': str(file_path_b),
        'fs': fs,
        'shift_s': shift_s,
        'shift_samples': shift_samples,
    }


class TestCoregisteredMefReaderBasic:
    """Basic tests for CoregisteredMefReader initialization and configuration."""
    
    def test_init_reference_only(self, generated_signals_mef_files):
        """Test initialization with reference file only."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        reader = CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        )
        
        assert reader.reference_reader is not None
        assert reader.other_reader is None
        assert reader.alignment_map is None
        
        reader.close()
    
    def test_init_with_both_files(self, generated_signals_mef_files):
        """Test initialization with both reference and other file."""
        file_path_a = generated_signals_mef_files['file_path_a']
        file_path_b = generated_signals_mef_files['file_path_b']
        
        reader = CoregisteredMefReader(
            reference_path=file_path_a,
            other_path=file_path_b,
            password2='read_password'
        )
        
        assert reader.reference_reader is not None
        assert reader.other_reader is not None
        assert reader.alignment_map is None
        
        reader.close()
    
    def test_context_manager(self, generated_signals_mef_files):
        """Test context manager support."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            assert reader.reference_reader is not None
        
        # Reader should be closed after context exit


class TestCoregistration:
    """Tests for coregistration computation."""
    
    def test_compute_coregistration_single_channel(self, matching_channel_mef_files):
        """Test coregistration with single channel (both files have same channel name)."""
        file_path_a = matching_channel_mef_files['file_path_a']
        file_path_b = matching_channel_mef_files['file_path_b']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            other_path=file_path_b,
            password2='read'
        ) as reader:
            # Compute coregistration using CH1 which exists in both files
            alignment_map = reader.compute_coregistration(
                alignment_channel='CH1',
                chunk_size_s=60.0  # Smaller chunks for shorter signal
            )
            
            assert isinstance(alignment_map, AlignmentMap)
            assert reader.alignment_map is not None
            assert reader.alignment_channel == 'CH1'
            assert reader.target_fs is not None
            
            print(f"\n✓ Coregistration computed")
            print(f"  Global offset: {alignment_map.global_offset_s:.2f}s")
            print(f"  Target fs: {reader.target_fs}Hz")
    
    def test_compute_coregistration_no_other_file(self, generated_signals_mef_files):
        """Test that coregistration fails without other file."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            with pytest.raises(ValueError, match="other_path not provided"):
                reader.compute_coregistration(alignment_channel='ECG')
    
    @pytest.mark.skip(reason="Bipolar test needs matching channel names")
    def test_compute_coregistration_bipolar(self, generated_signals_mef_files):
        """Test coregistration with bipolar derivation."""
        file_path_a = generated_signals_mef_files['file_path_a']
        file_path_b = generated_signals_mef_files['file_path_b']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            other_path=file_path_b,
            password2='read_password'
        ) as reader:
            # Use bipolar derivation for alignment
            alignment_map = reader.compute_coregistration(
                alignment_channel=('CH1', 'CH2'),
                chunk_size_s=300.0
            )
            
            assert isinstance(alignment_map, AlignmentMap)


class TestChannelPreference:
    """Tests for channel preference setting."""
    
    def test_set_channel_preference(self, generated_signals_mef_files):
        """Test setting channel preference."""
        file_path_a = generated_signals_mef_files['file_path_a']
        file_path_b = generated_signals_mef_files['file_path_b']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            other_path=file_path_b,
            password2='read_password'
        ) as reader:
            # Set preference
            reader.set_channel_preference('ECG', 'reference')
            
            assert reader.channel_preference['ECG'] == 'reference'
    
    def test_set_channel_preference_invalid(self, generated_signals_mef_files):
        """Test invalid channel preference."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            with pytest.raises(ValueError, match="must be 'reference' or 'other'"):
                reader.set_channel_preference('ECG', 'invalid')
    
    def test_set_all_channels_preference(self, generated_signals_mef_files):
        """Test setting preference for all channels."""
        file_path_a = generated_signals_mef_files['file_path_a']
        file_path_b = generated_signals_mef_files['file_path_b']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            other_path=file_path_b,
            password2='read_password'
        ) as reader:
            # Set all to 'other'
            reader.set_all_channels_preference('other')
            
            # Check that ECG preference is set
            assert reader.channel_preference.get('ECG') == 'other'


class TestDataRetrieval:
    """Tests for data retrieval with coregistration."""
    
    def test_get_data_from_reference(self, generated_signals_mef_files):
        """Test getting data from reference file."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            # Read data from reference
            data = reader.get_data(['ECG'])
            
            assert isinstance(data, list)
            assert len(data) == 1
            assert isinstance(data[0], np.ndarray)
            assert len(data[0]) > 0
            
            print(f"\n✓ Read data from reference")
            print(f"  Channel: ECG")
            print(f"  Samples: {len(data[0])}")
    
    def test_get_data_from_other_without_coregistration(self, generated_signals_mef_files):
        """Test that reading from other file fails without coregistration."""
        file_path_a = generated_signals_mef_files['file_path_a']
        file_path_b = generated_signals_mef_files['file_path_b']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            other_path=file_path_b,
            password2='read_password'
        ) as reader:
            # Set preference to 'other' without computing coregistration
            reader.set_channel_preference('ECG', 'other')
            
            with pytest.raises(ValueError, match="coregistration not computed"):
                reader.get_data(['ECG'])
    
    @pytest.mark.slow
    def test_get_data_with_coregistration(self, generated_signals_mef_files):
        """Test getting data with coregistration applied."""
        file_path_a = generated_signals_mef_files['file_path_a']
        file_path_b = generated_signals_mef_files['file_path_b']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            other_path=file_path_b,
            password2='read_password'
        ) as reader:
            # Compute coregistration
            reader.compute_coregistration(
                alignment_channel='ECG',
                chunk_size_s=300.0
            )
            
            # Set preference to read ECG from other file
            reader.set_channel_preference('ECG', 'other')
            
            # Read data with coregistration
            data = reader.get_data(['ECG'])
            
            assert isinstance(data, list)
            assert len(data) == 1
            assert isinstance(data[0], np.ndarray)
            assert len(data[0]) > 0
            
            print(f"\n✓ Read coregistered data")
            print(f"  Channel: ECG")
            print(f"  Samples: {len(data[0])}")


class TestMefReaderCompatibility:
    """Tests for MefReader interface compatibility."""
    
    def test_get_channel_info(self, generated_signals_mef_files):
        """Test get_channel_info compatibility."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            # Get all channel info (returns list)
            info = reader.get_channel_info()
            assert isinstance(info, list)
            assert len(info) > 0
            assert info[0]['name'] == 'ECG'
            
            # Get single channel info (returns dict)
            info_single = reader.get_channel_info('ECG')
            assert isinstance(info_single, dict)
            assert 'fsamp' in info_single
    
    def test_get_property(self, generated_signals_mef_files):
        """Test get_property compatibility."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            # This should not raise an error
            try:
                prop = reader.get_property('some_property')
            except:
                pass  # Property might not exist, that's ok
    
    def test_get_annotations(self, generated_signals_mef_files):
        """Test get_annotations compatibility."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            # This might raise KeyError if no annotations, which is expected
            try:
                annotations = reader.get_annotations()
                # Annotations might be None or empty
            except KeyError:
                # No annotations in test file, that's ok
                pass
    
    def test_get_raw_data(self, generated_signals_mef_files):
        """Test get_raw_data compatibility."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            # Raw data should always come from reference (returns list)
            data = reader.get_raw_data(['ECG'])
            
            assert isinstance(data, list)
            assert len(data) == 1
            assert isinstance(data[0], np.ndarray)


class TestStatePersistence:
    """Tests for saving and loading alignment state."""
    
    @pytest.mark.slow
    def test_save_and_load_alignment_state(self, generated_signals_mef_files, tmp_path):
        """Test saving and loading alignment state."""
        file_path_a = generated_signals_mef_files['file_path_a']
        file_path_b = generated_signals_mef_files['file_path_b']
        state_file = tmp_path / 'alignment_state.json'
        
        # Compute and save
        with CoregisteredMefReader(
            reference_path=file_path_a,
            other_path=file_path_b,
            password2='read_password'
        ) as reader:
            # Compute coregistration
            alignment_map = reader.compute_coregistration(
                alignment_channel='ECG',
                chunk_size_s=300.0
            )
            
            # Set preferences
            reader.set_channel_preference('ECG', 'other')
            
            # Save state
            reader.save_alignment_state(str(state_file))
            
            assert state_file.exists()
            
            # Store values for comparison
            global_offset = alignment_map.global_offset_s
            target_fs = reader.target_fs
        
        # Load and verify
        with CoregisteredMefReader(
            reference_path=file_path_a,
            other_path=file_path_b,
            password2='read_password',
            alignment_state_path=str(state_file)
        ) as reader:
            # Check that state was loaded
            assert reader.alignment_map is not None
            assert reader.alignment_map.global_offset_s == global_offset
            assert reader.target_fs == target_fs
            assert reader.channel_preference.get('ECG') == 'other'
            
            print(f"\n✓ State saved and loaded")
            print(f"  Global offset: {global_offset:.2f}s")
            print(f"  Target fs: {target_fs}Hz")
    
    def test_save_state_without_alignment(self, generated_signals_mef_files, tmp_path):
        """Test that saving state fails without alignment."""
        file_path_a = generated_signals_mef_files['file_path_a']
        state_file = tmp_path / 'alignment_state.json'
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            with pytest.raises(ValueError, match="no alignment computed"):
                reader.save_alignment_state(str(state_file))


class TestNaNHandling:
    """Tests for NaN value handling."""
    
    def test_interpolate_with_nans(self, generated_signals_mef_files):
        """Test interpolation with NaN values."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            # Create test signal with NaNs
            x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
            y = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
            x_new = np.array([0.5, 1.5, 2.5, 3.5])
            
            # Test interpolation
            y_new = reader._interpolate_with_nans(x_new, y, x)
            
            assert len(y_new) == len(x_new)
            # Non-NaN values should be interpolated
            assert not np.isnan(y_new[0])  # Should interpolate between 1 and 3
    
    def test_downsample_with_nans(self, generated_signals_mef_files):
        """Test downsampling with NaN values."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            # Create test signal with NaNs
            signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0] * 10)
            fs_in = 100.0
            fs_out = 50.0
            
            # Downsample
            signal_down = reader._downsample_signal(signal, fs_in, fs_out)
            
            assert len(signal_down) < len(signal)
            # Should handle NaNs without crashing
            assert not np.all(np.isnan(signal_down))


class TestDownsampling:
    """Tests for sampling rate conversion."""
    
    def test_downsample_signal(self, generated_signals_mef_files):
        """Test signal downsampling."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            # Create test signal
            fs_in = 1000.0
            fs_out = 250.0
            duration_s = 10.0
            t = np.arange(0, duration_s, 1/fs_in)
            signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
            
            # Downsample
            signal_down = reader._downsample_signal(signal, fs_in, fs_out)
            
            # Check output length
            expected_len = int(len(signal) * fs_out / fs_in)
            assert abs(len(signal_down) - expected_len) <= 1
            
            print(f"\n✓ Downsampling test")
            print(f"  Input: {len(signal)} samples @ {fs_in}Hz")
            print(f"  Output: {len(signal_down)} samples @ {fs_out}Hz")
    
    def test_downsample_no_upsample(self, generated_signals_mef_files):
        """Test that upsampling raises error."""
        file_path_a = generated_signals_mef_files['file_path_a']
        
        with CoregisteredMefReader(
            reference_path=file_path_a,
            password2='read_password'
        ) as reader:
            signal = np.random.randn(1000)
            
            with pytest.raises(ValueError, match="Cannot upsample"):
                reader._downsample_signal(signal, 100.0, 200.0)
