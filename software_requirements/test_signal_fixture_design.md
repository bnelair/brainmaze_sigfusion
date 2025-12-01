# Test Fixture Design Documentation

## Overview

The test signal generation system provides comprehensive test data for signal coregistration testing. It includes four main fixtures supporting different testing scenarios:
1. **Basic same-sampling-rate testing** (`generated_signals`)
2. **Different sampling rates** (`generated_signals_different_fs`)
3. **MEF file integration testing** (`generated_signals_mef_files`)
4. **Floating clock simulation** (`generated_signals_floating_clock`)

The architecture separates data generation logic into independent, reusable helper functions with built-in metadata preservation for file I/O operations and integration testing.

## Core Components

### Helper Functions

#### `_generate_pink_noise(num_samples, random_state, exponent)`
Generates pink (1/f^exponent) noise using frequency-domain shaping. Essential for realistic EEG-like background signals.

**Parameters:**
- `num_samples`: Number of samples to generate
- `random_state`: Optional seed for reproducibility
- `exponent`: Spectral exponent (1.0 for pink noise)

**Returns:** Normalized pink noise array with zero mean and unit variance

#### `_resample_signal(signal, original_fs, target_fs)`
Resamples a signal using linear interpolation. Used to simulate different sampling rates between devices.

**Parameters:**
- `signal`: 1D signal array
- `original_fs`: Original sampling rate
- `target_fs`: Target sampling rate

**Returns:** Resampled signal array

#### `_write_signal_to_mef_file(signal, sampling_rate, channel_name, output_path, metadata)`
Writes signal data to MEF (Medical Data Stream Format) files using mef-tools library.

**Parameters:**
- `signal`: 1D numpy array of signal samples
- `sampling_rate`: Sampling rate in Hz
- `channel_name`: Channel/device name (e.g., 'Device_A')
- `output_path`: Output .mefd directory path
- `metadata`: Optional dict with generation parameters

**Implementation Notes:**
- Creates .mefd directory structure (MEF3 format)
- Sets block length to 1 second
- Includes metadata as session annotations
- Handles precision inference automatically

#### `_apply_clock_drift(signal, sampling_rate, base_shift_s, max_drift_s, drift_seed)`
Applies time-varying clock drift to a signal to simulate floating clock behavior.

Simulates realistic crystal drift in recording devices: average 0 drift with ±max_drift_s over the recording duration.

**Parameters:**
- `signal`: 1D signal array (original sampled data)
- `sampling_rate`: Sampling rate in Hz
- `base_shift_s`: Base constant time shift in seconds (default: 3600)
- `max_drift_s`: Maximum drift in seconds (default: 10.0 = ±10s over 24 hours)
- `drift_seed`: Random seed for reproducibility (default: 54321)

**Returns:** Tuple containing:
- `resampled_signal`: Signal with time-varying clock applied via interpolation
- `drift_function`: Function that returns drift in seconds at any time
- `metadata`: Dictionary with drift parameters and statistics

**Implementation Details:**
- Uses multiple sinusoidal components for realistic, smooth drift
- Primary frequency: one full cycle over entire recording duration
- Secondary frequency: 3 cycles (faster variations)
- Tertiary frequency: 0.1 cycles (slower underlying trend)
- Adds small Brownian motion-like perturbations
- Ensures zero-mean drift
- Applies via time-varying interpolation (resampling)

### Signal Generation Functions

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
- `stim_amplitude`: Stimulation artifact amplitude (default: 200.0 = **10x pink noise**)

**Returns:** Dictionary with keys:
- `'signal'`: 1D numpy array of the generated signal
- `'t'`: Time array corresponding to the signal
- `'stim_mask'`: Boolean mask indicating stimulation ON periods
- `'metadata'`: Dictionary containing all generation parameters for future file I/O

**Signal Composition:**
- **Pink Noise (1/f):** Base signal covering full 24-hour duration (std: 20.0)
- **Stimulation Artifact:** 50 Hz sinusoidal artifact superimposed between hours 6-12
  - Amplitude: 200.0 (10 times the pink noise amplitude for clear detection)
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
- `measurement_noise_std`: Measurement noise standard deviation (default: 2.0, smaller than pink noise)

**Returns:** Dictionary with keys:
- `'signal'`: Extracted and noise-augmented signal segment
- `'t'`: Time array adjusted by the shift value
- `'t_shift'`: Time shift applied in seconds
- `'t_shift_samples'`: Time shift converted to number of samples
- `'segment_info'`: Dictionary with extraction metadata
- `'metadata'`: Dictionary with Signal B parameters

**Signal Composition:**
- **Extraction Window:** From 5 minutes before stim start to 5 minutes after stim end (in Signal A's timeline)
  - Stim start: 6 hours = 21,600 seconds
  - Stim end: 12 hours = 43,200 seconds
  - Window: 21,300s to 43,500s (370 minutes total)
- **Time Shift:** +1 hour (3,600 seconds) applied to time array to simulate clock offset
- **Measurement Noise:** Gaussian white noise (std: 2.0, smaller than pink noise amplitude of 20.0)

### Test Fixtures

#### 3. `@pytest.fixture generated_signals()` - Basic Fixture (Same Sampling Rate)
Orchestrates creation of Signal A and Signal B with same sampling rate (256 Hz).

**Returns:** Dictionary with keys:
- `'signal_a'`: Output from `_generate_test_signal_a(fs=256)`
- `'signal_b'`: Output from `_generate_test_signal_b()`
- `'t_shift_s'`: Time shift in seconds (3600)
- `'t_shift_samples'`: Time shift in samples (921,600)

#### 4. `@pytest.fixture generated_signals_different_fs()` - Different Sampling Rates
Generates signals with different sampling rates to test coregistration with resampling.
- **Signal A:** 256 Hz (24 hours)
- **Signal B:** 500 Hz (370 minutes, resampled from 256 Hz)

**Returns:** Dictionary with keys:
- `'signal_a'`: Output from `_generate_test_signal_a(fs=256)`
- `'signal_b'`: Signal B data resampled to 500 Hz
- `'t_shift_s'`: Time shift in seconds (3600)
- `'t_shift_samples_a'`: Time shift in samples at 256 Hz (921,600)
- `'t_shift_samples_b'`: Time shift in samples at 500 Hz (1,800,000)
- `'fs_a'`: Signal A sampling rate (256 Hz)
- `'fs_b'`: Signal B sampling rate (500 Hz)

#### 5. `@pytest.fixture generated_signals_mef_files()` - MEF File Integration
Generates Signal A and Signal B as separate MEF (.mefd) files to test file-based integration.

**Returns:** Dictionary with keys:
- `'signal_a'`: Signal A data (256 Hz)
- `'signal_b'`: Signal B data (500 Hz, resampled)
- `'file_path_a'`: Path to Signal A MEF directory
- `'file_path_b'`: Path to Signal B MEF directory
- `'tmp_dir'`: Temporary directory path (auto-cleaned after test)
- `'t_shift_s'`: Time shift in seconds (3600)
- `'t_shift_samples_a'`: Time shift in samples for Signal A
- `'t_shift_samples_b'`: Time shift in samples for Signal B
- `'fs_a'`: Signal A sampling rate (256 Hz)
- `'fs_b'`: Signal B sampling rate (500 Hz)

#### 6. `@pytest.fixture generated_signals_floating_clock()` - Floating Clock Simulation
Generates Signal A and Signal B with realistic floating clock drift simulation for multi-device scenarios.

This fixture tests the most realistic scenario where Signal B is recorded by a device with an unstable crystal oscillator.

**Signal A (Reference Device):**
- 256 Hz sampling rate
- Perfect clock synchronization
- 24-hour duration with stimulation artifact
- Stored in MEF file: `signal_a_floating_clock.mefd`

**Signal B (Device with Floating Clock):**
- 500 Hz sampling rate (different from Device A)
- Base time shift: +1 hour (3,600 seconds)
- Time-varying clock drift: ±10 seconds over 24 hours
  - Simulates realistic crystal drift (average 0 with ±10s variation)
  - Multiple frequency components for realistic behavior
  - Zero-mean drift pattern
- Extracted from 5 min before to 5 min after stimulation window
- **Drifted signal stored in MEF file**: `signal_b_floating_clock.mefd`

**Returns:** Dictionary with keys:
- `'signal_a'`: Signal A data (256 Hz, no drift)
- `'signal_b'`: Signal B data (500 Hz, with floating clock drift)
- `'file_path_a'`: Path to Signal A MEF file (reference)
- `'file_path_b'`: Path to Signal B MEF file (contains drifted signal)
- `'tmp_dir'`: Temporary directory path (auto-cleaned)
- `'t_shift_s'`: Base time shift in seconds (3600)
- `'t_shift_samples_a'`: Base shift in samples for Signal A (921,600)
- `'t_shift_samples_b'`: Base shift in samples for Signal B (1,800,000)
- `'max_drift_s'`: Maximum drift in seconds (±10)
- `'drift_function'`: Function to query drift at any time point
- `'fs_a'`: Signal A sampling rate (256 Hz)
- `'fs_b'`: Signal B sampling rate (500 Hz)

## Key Features

### 1. **Multiple Testing Scenarios**
- **Same sampling rate:** Basic coregistration testing (256 Hz for both)
- **Different sampling rates:** Resampling and rate conversion testing (256 Hz vs 500 Hz)
- **File-based:** MEF integration testing with real file I/O
- **Floating clock:** Realistic multi-device scenario with time-varying clock drift

### 2. **Configurable Parameters**
- Stimulation frequency: configurable (default: 50 Hz)
- Duty cycle: fully adjustable
- Sampling rates: independent for each signal
- Noise levels: customizable for each signal
- Time shift: configurable between devices
- Seeds: fixed for reproducibility

### 3. **Realistic Signal Characteristics**
- **Stimulation Amplitude:** 10x the pink noise amplitude (200 vs 20)
- **Noise:** Added measurement noise (std: 2.0) smaller than signal background (std: 20.0)
- **Pink Noise:** 1/f spectrum for realistic EEG-like background
- **Duty Cycle:** 1-minute ON, 3-minute OFF pattern (realistic for stimulation protocols)

### 4. **Metadata Preservation**
- All generation parameters stored in metadata dicts
- Segment extraction information preserved
- Ready for file I/O operations and integration testing
- Supports future extension without code changes

## Data Characteristics

### Signal A (Device A - Reference, 256 Hz)
- **Duration:** 24 hours (86,400 seconds)
- **Sampling Rate:** 256 Hz
- **Total Samples:** 22,118,400
- **Background:** Pink noise (1/f spectrum, std: 20.0)
- **Stimulation:** 50 Hz artifact (amplitude: 200.0, 10x signal)
- **Stim Window:** Hours 6-12 (6 hours)
- **Stim Pattern:** 1-min ON / 3-min OFF (repeating)

### Signal B (Device B - Shifted, 500 Hz)
- **Duration:** 370 minutes (21,300s - 43,500s in Signal A timeline, 6h 10min total)
- **Sampling Rate:** 500 Hz (resampled from 256 Hz)
- **Total Samples:** 11,100,000
- **Time Shift:** +1 hour (3,600 seconds) from Signal A reference
- **Extraction:** 5 min before + stim window + 5 min after (in Signal A timeline)
- **Measurement Noise:** Gaussian white noise (std: 2.0, smaller than signal background)

### Time Shifts

**At 256 Hz (Signal A):**
- Time shift: 3600 seconds = 921,600 samples

**At 500 Hz (Signal B):**
- Time shift: 3600 seconds = 1,800,000 samples

Relationship: `shift_samples = int(shift_s * fs)`

### MEF File Characteristics

### MEF Format
- **Extension:** .mefd (MEF3 directory structure)
- **Precision:** Auto-inferred by mef-tools (precision 2 for test data)
- **Block Length:** 1 second (256 samples for Signal A, 500 for Signal B)
- **Passwords:** 
  - Write: 'write_password'
  - Read: 'read_password'
- **Annotations:** Session-level metadata stored as annotations

### Data Fidelity
MEF files introduce minor rounding due to conversion factors:
- **Maximum deviation:** ~0.0015 (1.5e-3)
- **Mean deviation:** ~0.00025 (2.5e-4)
- **Match tolerance:** rtol=1e-3, atol=0.02 (99.9%+ match)
- **Statistical properties:** Preserved within 0.01% (rtol=1e-4)

## Floating Clock Drift Characteristics

### Drift Model
The floating clock simulation uses a realistic drift pattern combining multiple frequency components:

**Frequency Components:**
- **Primary:** One full cycle over entire 24-hour recording (0.0001157 Hz)
- **Secondary:** 3 cycles over 24 hours (0.0003472 Hz) with 30% amplitude
- **Tertiary:** 0.1 cycles over 24 hours (0.0000116 Hz) with 20% amplitude

**Drift Statistics:**
- **Mean drift:** 0 seconds (zero-mean by design)
- **Maximum magnitude:** ±10 seconds
- **Type:** Smooth continuous drift with Brownian motion perturbations
- **Application method:** Time-varying resampling/interpolation

### Realistic Crystal Behavior
- Simulates oscillator drift rate of ~2 seconds per hour (configurable)
- Over 24 hours: ±10 seconds total deviation (realistic for low-cost crystals)
- Smooth transitions (no discontinuities)
- Reproducible with fixed seed (54321)

## Usage Examples

### Basic Fixture (Same Sampling Rate)
```python
def test_coregistration_basic(generated_signals):
    # Access Signal A
    signal_a = generated_signals['signal_a']['signal']      # 22,118,400 samples
    t_a = generated_signals['signal_a']['t']                # 24 hours
    stim_mask_a = generated_signals['signal_a']['stim_mask']
    
    # Access Signal B
    signal_b = generated_signals['signal_b']['signal']      # 5,683,200 samples
    t_b = generated_signals['signal_b']['t']
    
    # Get time shift
    shift_s = generated_signals['t_shift_s']                # 3600 seconds
    shift_samples = generated_signals['t_shift_samples']    # 921,600 samples
    
    # Use data for testing coregistration
    assert signal_a.shape[0] == int(24 * 3600 * 256)
    assert shift_s == 3600
```

### Different Sampling Rates Fixture
```python
def test_coregistration_different_fs(generated_signals_different_fs):
    # Signal A: 256 Hz
    signal_a = generated_signals_different_fs['signal_a']['signal']
    fs_a = generated_signals_different_fs['fs_a']  # 256
    
    # Signal B: 500 Hz (resampled)
    signal_b = generated_signals_different_fs['signal_b']['signal']
    fs_b = generated_signals_different_fs['fs_b']  # 500
    
    # Time shifts at different sampling rates
    shift_samples_a = generated_signals_different_fs['t_shift_samples_a']  # 921,600
    shift_samples_b = generated_signals_different_fs['t_shift_samples_b']  # 1,800,000
    
    # Test resampling-aware coregistration
    assert len(signal_a) / fs_a == len(signal_b) / fs_b  # Same duration
```

### MEF File Integration Fixture
```python
def test_coregistration_with_mef_files(generated_signals_mef_files):
    from mef_tools.io import MefReader
    
    # Get file paths
    file_a = generated_signals_mef_files['file_path_a']  # .mefd directory
    file_b = generated_signals_mef_files['file_path_b']  # .mefd directory
    
    # Read from MEF files
    reader_a = MefReader(file_a, password2='read_password')
    signal_a_read = reader_a.get_data('Device_A')
    
    reader_b = MefReader(file_b, password2='read_password')
    signal_b_read = reader_b.get_data('Device_B')
    
    # Compare with original signals
    signal_a_original = generated_signals_mef_files['signal_a']['signal']
    signal_b_original = generated_signals_mef_files['signal_b']['signal']
    
    # Use isclose for comparison (MEF introduces small rounding)
    assert np.isclose(signal_a_read, signal_a_original, rtol=1e-3, atol=0.02).all()
    assert np.isclose(signal_b_read, signal_b_original, rtol=1e-3, atol=0.02).all()
```

### Custom Signal Generation
```python
def test_custom_parameters():
    from conftest import _generate_test_signal_a, _generate_test_signal_b
    
    # Custom Signal A with different stimulation frequency
    signal_a = _generate_test_signal_a(stim_freq_hz=100.0)  # 100 Hz instead of 50
    
    # Custom Signal B with different time shift
    signal_b = _generate_test_signal_b(
        signal_a_data=signal_a,
        shift_s=7200  # 2 hours instead of 1
    )
```

### Floating Clock Drift Fixture
```python
def test_coregistration_with_floating_clock(generated_signals_floating_clock):
    from mef_tools.io import MefReader
    
    # Get signals and file paths
    signal_a = generated_signals_floating_clock['signal_a']['signal']  # 256 Hz, no drift
    signal_b = generated_signals_floating_clock['signal_b']['signal']  # 500 Hz, with drift
    
    file_a = generated_signals_floating_clock['file_path_a']
    file_b = generated_signals_floating_clock['file_path_b']  # Contains drifted signal
    
    # Get drift information
    max_drift = generated_signals_floating_clock['max_drift_s']  # ±10 seconds
    drift_fn = generated_signals_floating_clock['drift_function']
    
    # Query drift at specific times
    drift_at_12h = drift_fn(12 * 3600)  # Drift at 12 hours
    
    # Read from MEF files (Signal B includes floating clock effects)
    reader_a = MefReader(file_a, password2='read_password')
    signal_a_mef = reader_a.get_data('Device_A_Reference')
    
    reader_b = MefReader(file_b, password2='read_password')
    signal_b_mef = reader_b.get_data('Device_B_FloatingClock')  # Drifted signal
    
    # Test coregistration with floating clock
    assert len(signal_a) == len(signal_a_mef)
    assert abs(max_drift) <= 10.0
    assert -10.0 <= drift_at_12h <= 10.0
```

## Test Coverage

### Provided Tests
1. **test_generated_signals_fixture:** Validates basic fixture structure and properties
2. **test_generated_signals_different_fs_fixture:** Validates different sampling rate handling
3. **test_generated_signals_mef_files_fixture:** Validates MEF file creation
4. **test_mef_files_read_verify:** Validates MEF file read-back with tolerance checks
5. **test_floating_clock_fixture_basic:** Validates floating clock fixture and drift parameters

### Tolerance Values for Comparisons
- **Raw sample comparison:** `rtol=1e-3, atol=0.02` (accounts for MEF conversion)
- **Mean comparison:** `atol=1e-6` (handles near-zero values)
- **Std comparison:** `rtol=1e-4` (0.01% tolerance)
- **Overall match threshold:** 99.9% of samples within tolerance

