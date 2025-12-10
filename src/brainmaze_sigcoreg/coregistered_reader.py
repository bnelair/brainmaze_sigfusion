"""
Coregistered MEF Reader for temporal synchronization of multi-device recordings.

This module provides a MefReader-compatible interface for reading from two MEF files
where one has been coregistered to the other. It handles:
- Temporal alignment between recordings from independent devices
- Channel preference (which file to read each channel from)
- Different sampling rates with anti-aliasing downsampling
- NaN values in signals without breaking filtering
"""

import os
import json
import numpy as np
from typing import Optional, Union, List, Dict, Any, Tuple
from pathlib import Path
from scipy import signal as scipy_signal

from mef_tools.io import MefReader
from .coregistration import compute_alignment, AlignmentMap


class CoregisteredMefReader:
    """
    MEF Reader with coregistration support for multi-device recordings.
    
    This class provides a MefReader-compatible interface for reading from two MEF files
    where the second file has been temporally aligned (coregistered) to the first.
    
    Key features:
    - Automatic coregistration based on specified alignment channel(s)
    - Channel preference: choose which file to read each channel from
    - Handles different sampling rates with anti-aliasing downsampling
    - Supports bipolar derivations for alignment
    - Persistent state: save/load coregistration to avoid recomputation
    - NaN-aware signal processing
    
    Attributes:
        reference_path: Path to reference MEF file
        other_path: Path to other MEF file to be coregistered
        alignment_map: Temporal alignment transformation
        channel_preference: Dict mapping channel names to preferred file ('reference' or 'other')
        target_fs: Target sampling frequency for mixed-rate channels (uses lower of the two)
    """
    
    def __init__(
        self,
        reference_path: str,
        other_path: Optional[str] = None,
        password2: Optional[str] = None,
        alignment_state_path: Optional[str] = None,
    ):
        """
        Initialize CoregisteredMefReader.
        
        Args:
            reference_path: Path to reference MEF file (primary recording)
            other_path: Optional path to other MEF file to be coregistered
            password2: MEF file password (level 2)
            alignment_state_path: Optional path to load/save alignment state
        """
        self.reference_path = Path(reference_path)
        self.other_path = Path(other_path) if other_path else None
        self.password2 = password2
        self.alignment_state_path = Path(alignment_state_path) if alignment_state_path else None
        
        # Open reference reader
        self.reference_reader = MefReader(
            str(self.reference_path),
            password2=password2
        )
        
        # Open other reader if provided
        self.other_reader = None
        if self.other_path:
            self.other_reader = MefReader(
                str(self.other_path),
                password2=password2
            )
        
        # Alignment state
        self.alignment_map: Optional[AlignmentMap] = None
        self.alignment_channel: Optional[Union[str, Tuple[str, str]]] = None
        self.channel_preference: Dict[str, str] = {}  # channel -> 'reference' or 'other'
        self.target_fs: Optional[float] = None
        
        # Load alignment state if provided
        if self.alignment_state_path and self.alignment_state_path.exists():
            self.load_alignment_state(str(self.alignment_state_path))
    
    def compute_coregistration(
        self,
        alignment_channel: Union[str, Tuple[str, str]],
        chunk_size_s: float = 300.0,
        search_range_s: Optional[Tuple[float, float]] = None,
    ) -> AlignmentMap:
        """
        Compute coregistration between reference and other file.
        
        Args:
            alignment_channel: Channel name for alignment, or tuple of (ch1, ch2) for bipolar
            chunk_size_s: Chunk size for fine alignment (default: 5 minutes)
            search_range_s: Optional search range (min_offset, max_offset) in seconds
            
        Returns:
            AlignmentMap containing the computed synchronization transformation
            
        Raises:
            ValueError: If other_path not provided or channel not found
        """
        if not self.other_reader:
            raise ValueError("Cannot compute coregistration: other_path not provided")
        
        self.alignment_channel = alignment_channel
        
        # Extract signals for alignment
        ref_signal, ref_fs = self._extract_alignment_signal(
            self.reference_reader, alignment_channel
        )
        other_signal, other_fs = self._extract_alignment_signal(
            self.other_reader, alignment_channel
        )
        
        # Compute alignment
        self.alignment_map = compute_alignment(
            signal_reference=ref_signal,
            signal_source=other_signal,
            fs_reference=ref_fs,
            fs_source=other_fs,
            search_range_s=search_range_s,
            chunk_size_s=chunk_size_s,
            perform_fine_alignment=True,
        )
        
        # Determine target sampling frequency (use lower of the two)
        self.target_fs = min(ref_fs, other_fs)
        
        return self.alignment_map
    
    def _extract_alignment_signal(
        self,
        reader: MefReader,
        channel: Union[str, Tuple[str, str]],
    ) -> Tuple[np.ndarray, float]:
        """
        Extract signal for alignment from MEF reader.
        
        Handles both single channel and bipolar derivation.
        
        Args:
            reader: MefReader instance
            channel: Channel name or tuple of (ch1, ch2) for bipolar
            
        Returns:
            Tuple of (signal, sampling_rate)
        """
        if isinstance(channel, tuple):
            # Bipolar derivation
            ch1, ch2 = channel
            data1 = reader.get_data([ch1])  # Returns list of arrays
            data2 = reader.get_data([ch2])
            
            # Get sampling rates
            info1 = reader.get_channel_info(ch1)
            info2 = reader.get_channel_info(ch2)
            fs1 = float(info1['fsamp'][0])
            fs2 = float(info2['fsamp'][0])
            
            if fs1 != fs2:
                raise ValueError(
                    f"Bipolar channels have different sampling rates: "
                    f"{ch1}={fs1}Hz, {ch2}={fs2}Hz"
                )
            
            # Compute bipolar derivation
            signal = data1[0] - data2[0]
            fs = fs1
        else:
            # Single channel
            data = reader.get_data([channel])  # Returns list of arrays
            signal = data[0]  # Get first element
            
            info = reader.get_channel_info(channel)
            fs = float(info['fsamp'][0])
        
        return signal, fs
    
    def set_channel_preference(self, channel: str, source: str):
        """
        Set which file to read a channel from.
        
        Args:
            channel: Channel name
            source: Either 'reference' or 'other'
            
        Raises:
            ValueError: If source is not 'reference' or 'other'
        """
        if source not in ['reference', 'other']:
            raise ValueError(f"Source must be 'reference' or 'other', got '{source}'")
        
        self.channel_preference[channel] = source
    
    def set_all_channels_preference(self, source: str):
        """
        Set preference for all available channels.
        
        Args:
            source: Either 'reference' or 'other'
        """
        # Get all channel names from reference
        # get_channel_info() returns a list of dicts
        ref_info = self.reference_reader.get_channel_info()
        if isinstance(ref_info, list):
            for ch_info in ref_info:
                channel_name = ch_info['name']
                self.set_channel_preference(channel_name, source)
        else:
            # Single channel info
            channel_name = ref_info['name']
            self.set_channel_preference(channel_name, source)
    
    def get_data(
        self,
        channels: List[str],
        t_stamp1: Optional[int] = None,
        t_stamp2: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Get data for specified channels with coregistration applied.
        
        Compatible with MefReader.get_data() interface. Reads from preferred file
        for each channel, applying temporal alignment and sampling rate conversion.
        
        Args:
            channels: List of channel names
            t_stamp1: Start timestamp in uUTC (microseconds since epoch)
            t_stamp2: End timestamp in uUTC (microseconds since epoch)
            
        Returns:
            List of signal arrays (compatible with MefReader)
        """
        result = []
        
        for channel in channels:
            # Determine which reader to use
            preference = self.channel_preference.get(channel, 'reference')
            
            if preference == 'reference' or not self.other_reader:
                # Read from reference
                data = self.reference_reader.get_data([channel], t_stamp1, t_stamp2)
                result.append(data[0])  # MefReader returns list
            else:
                # Read from other with coregistration
                if not self.alignment_map:
                    raise ValueError(
                        "Cannot read from other file: coregistration not computed. "
                        "Call compute_coregistration() first."
                    )
                
                # Get data from other file in its own time frame
                data_other = self.other_reader.get_data([channel], t_stamp1, t_stamp2)
                signal_other = data_other[0]  # MefReader returns list
                
                # Get sampling rates
                info_other = self.other_reader.get_channel_info(channel)
                fs_other = float(info_other['fsamp'][0])
                
                # Apply temporal alignment to map to reference time frame
                signal_aligned = self._apply_coregistration(
                    signal_other, fs_other, t_stamp1, t_stamp2
                )
                
                # Downsample if needed
                if self.target_fs and fs_other > self.target_fs:
                    signal_aligned = self._downsample_signal(
                        signal_aligned, fs_other, self.target_fs
                    )
                
                result.append(signal_aligned)
        
        return result
    
    def _apply_coregistration(
        self,
        signal: np.ndarray,
        fs: float,
        t_stamp1: Optional[int],
        t_stamp2: Optional[int],
    ) -> np.ndarray:
        """
        Apply temporal alignment to signal from other file.
        
        Maps signal from other file's time frame to reference file's time frame.
        
        Args:
            signal: Signal from other file
            fs: Sampling frequency of signal
            t_stamp1: Start timestamp in reference time frame (uUTC)
            t_stamp2: End timestamp in reference time frame (uUTC)
            
        Returns:
            Signal mapped to reference time frame
        """
        # Get timestamps for each sample in reference time frame
        num_samples = len(signal)
        timestamps_ref = self.alignment_map.get_reference_timestamps(num_samples)
        
        # Convert to uUTC if timestamps provided
        if t_stamp1 is not None:
            # Create time array for interpolation
            t_ref_array = np.arange(num_samples) / fs + (t_stamp1 / 1e6)
            
            # Interpolate signal to reference time points
            # Handle NaN values properly
            signal_aligned = self._interpolate_with_nans(
                timestamps_ref, signal, t_ref_array
            )
        else:
            signal_aligned = signal
        
        return signal_aligned
    
    def _interpolate_with_nans(
        self,
        x_new: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate signal handling NaN values properly.
        
        Args:
            x_new: New x coordinates
            y: Signal values (may contain NaNs)
            x: Original x coordinates
            
        Returns:
            Interpolated signal
        """
        # Find NaN locations
        nan_mask = np.isnan(y)
        
        if np.all(nan_mask):
            # All NaN, return NaN array
            return np.full_like(x_new, np.nan)
        
        if not np.any(nan_mask):
            # No NaNs, regular interpolation
            return np.interp(x_new, x, y)
        
        # Interpolate non-NaN values
        valid_mask = ~nan_mask
        y_interp = np.interp(x_new, x[valid_mask], y[valid_mask])
        
        return y_interp
    
    def _downsample_signal(
        self,
        signal: np.ndarray,
        fs_in: float,
        fs_out: float,
    ) -> np.ndarray:
        """
        Downsample signal with anti-aliasing filter.
        
        Handles NaN values by preserving them through the downsampling process.
        
        Args:
            signal: Input signal
            fs_in: Input sampling frequency
            fs_out: Output sampling frequency
            
        Returns:
            Downsampled signal
        """
        if fs_in == fs_out:
            return signal
        
        if fs_out > fs_in:
            raise ValueError(f"Cannot upsample: fs_out ({fs_out}) > fs_in ({fs_in})")
        
        # Handle NaN values
        nan_mask = np.isnan(signal)
        has_nans = np.any(nan_mask)
        
        if has_nans:
            # Replace NaNs temporarily for filtering
            signal_clean = signal.copy()
            signal_clean[nan_mask] = 0.0
        else:
            signal_clean = signal
        
        # Compute decimation factor
        decimation_factor = int(fs_in / fs_out)
        
        if decimation_factor >= 2:
            # Use scipy.signal.decimate for anti-aliasing downsampling
            # This applies a Chebyshev Type I lowpass filter before downsampling
            signal_down = scipy_signal.decimate(
                signal_clean,
                decimation_factor,
                ftype='iir',  # IIR filter for efficiency
                zero_phase=True  # Forward-backward filtering for zero phase
            )
        else:
            # Fractional downsampling - use resample
            num_samples_out = int(len(signal) * fs_out / fs_in)
            signal_down = scipy_signal.resample(signal_clean, num_samples_out)
        
        if has_nans:
            # Restore NaN pattern (approximate)
            nan_indices_down = (np.where(nan_mask)[0] * fs_out / fs_in).astype(int)
            nan_indices_down = nan_indices_down[nan_indices_down < len(signal_down)]
            signal_down[nan_indices_down] = np.nan
        
        return signal_down
    
    def get_channel_info(self, channel: Optional[str] = None) -> Union[List[Dict], Dict]:
        """
        Get channel information.
        
        Compatible with MefReader.get_channel_info() interface.
        
        Args:
            channel: Optional channel name. If None, returns info for all channels.
            
        Returns:
            Channel information: list of dicts if no channel specified, dict if channel specified
        """
        # Always return reference channel info
        return self.reference_reader.get_channel_info(channel)
    
    def get_property(self, property_name: str, channel: Optional[str] = None) -> Any:
        """
        Get property value.
        
        Compatible with MefReader.get_property() interface.
        
        Args:
            property_name: Name of property to get
            channel: Optional channel name
            
        Returns:
            Property value
        """
        return self.reference_reader.get_property(property_name, channel)
    
    def get_annotations(self, channel: Optional[str] = None) -> Any:
        """
        Get annotations.
        
        Compatible with MefReader.get_annotations() interface.
        
        Args:
            channel: Optional channel name
            
        Returns:
            Annotations
        """
        return self.reference_reader.get_annotations(channel)
    
    def get_raw_data(
        self,
        channels: List[str],
        t_stamp1: Optional[int] = None,
        t_stamp2: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Get raw data (without coregistration).
        
        Always reads from reference file.
        
        Args:
            channels: List of channel names
            t_stamp1: Start timestamp in uUTC
            t_stamp2: End timestamp in uUTC
            
        Returns:
            List of signal arrays (compatible with MefReader)
        """
        return self.reference_reader.get_raw_data(channels, t_stamp1, t_stamp2)
    
    def save_alignment_state(self, filepath: str):
        """
        Save coregistration state to file.
        
        Saves alignment map and configuration for later use without recomputation.
        
        Args:
            filepath: Path to save state
        """
        if not self.alignment_map:
            raise ValueError("Cannot save state: no alignment computed")
        
        state = {
            'reference_path': str(self.reference_path),
            'other_path': str(self.other_path) if self.other_path else None,
            'alignment_channel': self.alignment_channel,
            'channel_preference': self.channel_preference,
            'target_fs': self.target_fs,
            'alignment_map': self.alignment_map.to_dict(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_alignment_state(self, filepath: str):
        """
        Load coregistration state from file.
        
        Restores alignment map and configuration.
        
        Args:
            filepath: Path to load state from
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore configuration
        self.alignment_channel = state.get('alignment_channel')
        self.channel_preference = state.get('channel_preference', {})
        self.target_fs = state.get('target_fs')
        
        # Restore alignment map
        if 'alignment_map' in state:
            self.alignment_map = AlignmentMap.from_dict(state['alignment_map'])
    
    def close(self):
        """Close all open MEF readers."""
        if self.reference_reader:
            self.reference_reader.close()
        if self.other_reader:
            self.other_reader.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
