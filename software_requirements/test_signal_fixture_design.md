# Test Fixture Design Documentation

## Overview

The test signal generation system has been refactored to support modular signal generation with built-in support for future file I/O operations. The new architecture separates data generation logic into independent, reusable helper functions that can be easily extended for integration testing with file storage.

## Architecture

### Core Components

#### 1. `_generate_test_signal_a()` - Device A Reference Signal
Generates a 24-hour reference signal representing the primary recording device with synchronized clock.

**Parameters:**
- `fs`: Sampling rate (default: 256 Hz)
- `duration_s`: Total signal duration (default: 24 * 3600 seconds)
- `stim_start_s`: Stimulation start time in seconds (default: 6 * 3600 = hour 6)
- `stim_end_s`: Stimulation end time in seconds (default: 12 * 3600 = hour 12)
- `stim_freq_hz`: Stimulation frequency in Hz (default: 50.0 Hz, **configurable**)
- `duty_on_s`: ON phase duration (default: 60 seconds = 1 minute)
- `duty_off_s`: OFF phase duration (default: 180 seconds = 3 minutes)
- `pink_noise_seed`: Random seed for reproducibility (default: 42)
- `pink_noise_amplitude`: Pink noise scaling factor (default: 20.0)
- `stim_amplitude`: Stimulation artifact amplitude (default: 2.0)

**Returns:** Dictionary with keys:
- `'signal'`: 1D numpy array of the generated signal
- `'t'`: Time array corresponding to the signal
- `'stim_mask'`: Boolean mask indicating stimulation ON periods
- `'metadata'`: Dictionary containing all generation parameters for future file I/O

**Signal Composition:**
- **Pink Noise (1/f):** Base signal covering full 24-hour duration
- **Stimulation Artifact:** 50 Hz sinusoidal artifact superimposed between hours 6-12
  - Duty cycle: 1 minute ON, 3 minutes OFF (repeating)
  - Only applied where `stim_mask == True`

#### 2. `_generate_test_signal_b()` - Device B Shifted Signal
Generates a time-shifted segment of Signal A to simulate clock desynchronization between recording devices.

**Parameters:**
- `signal_a_data`: Output dictionary from `_generate_test_signal_a()`
- `shift_s`: Time shift in seconds (default: 3600 = 1 hour)
- `buffer_before_s`: Duration before stimulation start to include (default: 5 * 60 = 5 minutes)
- `buffer_after_s`: Duration after stimulation end to include (default: 5 * 60 = 5 minutes)
- `measurement_noise_seed`: Random seed for measurement noise (default: 12345)
- `measurement_noise_std`: Measurement noise standard deviation (default: 1.0)

**Returns:** Dictionary with keys:
- `'signal'`: Extracted and noise-augmented signal segment
- `'t'`: Time array adjusted by the shift value
- `'t_shift'`: Time shift applied in seconds
- `'t_shift_samples'`: Time shift converted to number of samples
- `'segment_info'`: Dictionary with extraction metadata:
  - `'segment_start_s'`: Start time in Signal A timeline
  - `'segment_end_s'`: End time in Signal A timeline
  - `'start_idx_in_signal_a'`: Start sample index in Signal A
  - `'end_idx_in_signal_a'`: End sample index in Signal A
  - `'buffer_before_s'`: Buffer duration before stim
  - `'buffer_after_s'`: Buffer duration after stim
- `'metadata'`: Dictionary with Signal B parameters

**Signal Composition:**
- **Extraction Window:** From 5 minutes before stim start to 5 minutes after stim end (in Signal A's timeline)
  - Stim start: 6 hours = 21,600 seconds
  - Stim end: 12 hours = 43,200 seconds
  - Window: 21,300s to 43,500s (370 minutes total)
- **Time Shift:** +1 hour (3,600 seconds) applied to time array to simulate clock offset
- **Measurement Noise:** Gaussian noise added to simulate sensor noise differences between devices

#### 3. `@pytest.fixture generated_signals()` - Main Test Fixture
Orchestrates the creation of both Signal A and Signal B with pre-configured default parameters.

**Returns:** Dictionary with keys:
- `'signal_a'`: Output from `_generate_test_signal_a()`
- `'signal_b'`: Output from `_generate_test_signal_b()`
- `'t_shift_s'`: Time shift in seconds (convenience accessor)
- `'t_shift_samples'`: Time shift in samples (convenience accessor)

## Key Features

### 1. **Configurability**
- Stimulation frequency is configurable (default: 50 Hz)
- Duty cycle parameters can be adjusted
- Sampling rate, duration, and noise levels are all parameterizable
- Seeds are fixed for reproducibility in tests

### 2. **Metadata Preservation**
- All generation parameters are stored in `metadata` dictionaries
- Segment extraction information is preserved in `segment_info`
- Ready for future file I/O operations without code changes

### 3. **Designed for File I/O Extension**
The modular design and metadata preservation make it straightforward to add file writing functionality:

```python
# Future extension - write signals to files
def save_test_signals(signal_a_data, signal_b_data, output_dir):
    # Use metadata to recreate signal generation settings
    # Use segment_info to understand data relationships
    # Write to HDF5, NumPy, or other formats
    pass
```

## Data Characteristics

### Signal A (Reference Signal)
- **Duration:** 24 hours (86,400 seconds)
- **Sampling Rate:** 256 Hz
- **Total Samples:** 22,118,400
- **Background:** Pink noise (1/f spectrum)
- **Stimulation Window:** Hours 6-12 (6 hours)
- **Stimulation Pattern:** 50 Hz with 1-min ON / 3-min OFF duty cycle

### Signal B (Shifted Signal)
- **Duration:** 370 minutes (6 hours 10 minutes)
- **Sampling Rate:** 256 Hz (same as Signal A)
- **Total Samples:** 5,683,200
- **Time Shift:** +1 hour from Signal A reference
- **Extraction Window:** 5 minutes before + stim period + 5 minutes after (in Signal A timeline)
- **Additional Noise:** Gaussian white noise (std = 1.0)

## Time Shift Calculation

The time shift is returned in two forms for flexibility:

```python
shift_s = 3600          # seconds (for human-readable interpretation)
shift_samples = 921600  # samples at 256 Hz (for array indexing)
```

Relationship: `shift_samples = int(shift_s * fs)`

## Usage Example

```python
def test_example(generated_signals):
    # Access Signal A data
    signal_a = generated_signals['signal_a']['signal']
    t_a = generated_signals['signal_a']['t']
    stim_mask_a = generated_signals['signal_a']['stim_mask']
    metadata_a = generated_signals['signal_a']['metadata']
    
    # Access Signal B data
    signal_b = generated_signals['signal_b']['signal']
    t_b = generated_signals['signal_b']['t']
    shift_s = generated_signals['t_shift_s']
    shift_samples = generated_signals['t_shift_samples']
    
    # Use data for testing coregistration algorithm
    assert signal_a.shape[0] == int(24 * 3600 * 256)
    assert shift_s == 3600
```
