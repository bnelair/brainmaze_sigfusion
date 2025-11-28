# Software Requirements Specification: brainmaze_sigfusion

## 1. Overview and Scope

The objective of **brainmaze_sigfusion** is to provide a Python-based utility for the temporal synchronization and fusion of electrophysiological time-series data from two independent recording sources (for example, recordings with differing hardware clocks, sampling rates, and durations). The tool shall utilize the `mef_tools` library for I/O operations and provide a unified interface for accessing merged data streams.

This document describes functional and non-functional requirements, persistence behavior, and a proposed Python API.

---

## 2. Functional Requirements

### 2.1. Input and Data Handling

- **REQ-2.1.1 (File Ingestion)**: The system shall accept file paths for two distinct Multiscale Electrophysiology Format (MEF) datasets, referred to herein as **Source_A** (typically the long-duration recording, e.g., 24 h) and **Source_B** (the subset recording, e.g., 8 h).

- **REQ-2.1.2 (Channel Selection)**: The system shall allow the user to specify the channel(s) used for calculating the synchronization alignment.

- **REQ-2.1.3 (Derivation Support)**: The alignment logic shall support input signals as either:
  - Single-ended: a single channel index from both sources (e.g., Channel 1 from A vs. Channel 1 from B).
  - Bipolar: a differential signal computed between two specified channels (e.g., Ch1 - Ch2 from A vs. Ch1 - Ch2 from B).

### 2.2. Synchronization Algorithm

To account for non-linear clock drift and periodic signal artifacts (for example, stimulation cycles), the synchronization shall proceed in two stages.

- **REQ-2.2.1 (Stage I — Coarse Global Alignment)**: The system shall compute a global time offset (t0) based on the signal envelope over the entire duration of the intersection between Source_A and Source_B.
  - Rationale: This stage must be robust against local minima caused by periodic patterns (for example, stimulation occurring 1 minute ON, 2-5 minutes OFF) to prevent phase-shifted misalignment.

- **REQ-2.2.2 (Stage II — Fine Local Alignment)**: The system shall refine alignment using a sliding-window (chunk-based) approach.
  - The chunk size shall be a configurable parameter (default: 5 minutes).
  - For each chunk, the system shall estimate local clock drift and produce a time-varying transformation map T(t) that relates the time basis of Source_B to Source_A.

### 2.3. Persistence and State Management

- **REQ-2.3.1 (Serialization)**: The computed synchronization transformation (the "Alignment Map") shall be serializable. The system must allow saving this map to disk.

- **REQ-2.3.2 (Loading)**: The system must allow loading a pre-computed Alignment Map to initialize the reader without re-running the computationally expensive coregistration process.

### 2.4. Unified Reader Interface

The core deliverable is a class that abstracts the complexity of reading two MEF files into a single data provider.

- **REQ-2.4.1 (API Consistency)**: The class shall expose an API consistent with a standard `MefReader`, allowing calls such as `read_channel()` or `get_data()`.

- **REQ-2.4.2 (Prioritization Logic)**: The user shall be able to designate a **Preferred Source** (for example, Source_B, the 8 h recording that contains stimulation data).

- **REQ-2.4.3 (Dynamic Data Retrieval)**: When a data request is made for a specific time window and channel:
  1. If the requested data exists in the Preferred Source (after applying the Alignment Map), return data from the Preferred Source.
  2. If the requested data falls outside the Preferred Source (for example, outside the 8 h window), fall back to the Secondary Source (Source_A) and return aligned data from that source.

- **REQ-2.4.4 (Seamless Transition)**: The system shall stitch segments from different sources to ensure continuity, filling gaps in the shorter recording with aligned data from the longer recording.

---

## 3. Non-Functional Requirements

- **REQ-3.1 (Performance)**: The coarse alignment stage must use downsampled envelopes or efficient vectorized operations (for example, NumPy) to ensure processing feasibility for 24-hour recordings.

- **REQ-3.2 (Memory Efficiency)**: The alignment process must not load the entirety of Source_A (24 h) into RAM simultaneously. It must use lazy loading or memory mapping provided by `mef_tools`.

- **REQ-3.3 (Sampling Rate Independence)**: The system must handle inputs with differing sampling rates. The output stream shall correspond to the sampling rate of the requested Preferred Source or to a user-defined target rate.

- **REQ-3.4 (Cross-platform Compatibility & Architecture Independence)**: The system shall be cross-platform and architecture-independent. Specifically:
  - The software must run on major operating systems: Linux, macOS, and Windows.
  - The implementation must support common CPU architectures (x86_64, ARM64) where Python is available.
  - Avoid OS-specific system calls, path assumptions, or shell-dependent behavior in the core library. Use portable libraries (`pathlib`, `os`, cross-platform packaging) and provide platform-specific adapters only when strictly necessary.
  - Prefer pure-Python implementations or provide pre-built binary wheels for any native extensions so users on all supported platforms can install the package without building from source.
  - The CI pipeline shall exercise at least one runner per supported OS/architecture to validate cross-platform behavior (see `.github/workflows/pytest.yml`).

---
