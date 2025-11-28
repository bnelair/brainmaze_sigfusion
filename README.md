# BrainMaze: Signal Synchronization and Fusion Toolbox

A repository for the signal synchronization and fusion project.

## Project status

This is the initial repo skeleton. It contains a Software Requirements Specification (SRS) describing the goals, functional and non-functional requirements, and a proposed Python API for aligning and merging electrophysiology time-series from two MEF recordings.

## High-level description

brainmaze_sigfusion is a Python utility intended to:

- Compute a time-varying synchronization map between two MEF3 recordings with different duration and sampling rates.
- Allow saving/loading of the computed alignment map to avoid re-computation.
- Provide a unified reader that presents a single, temporally-aligned data stream to downstream analysis code while preferring a designated source when overlapping data exists.

Key design points:
- Two-stage alignment: coarse global alignment (envelope-based) and fine, chunk-based local alignment to handle non-linear clock drift and periodic artifacts.
- Memory- and performance-conscious processing (streaming, downsampling, and vectorized operations where possible).
- Sampling-rate independent: the system will handle differing input sample rates and present data at a chosen target or preferred-source rate.

## Where to find the Software Requirements

The full Software Requirements Specification (SRS) is in:

`software_requirements/sw_req.md`

Open that file for the detailed requirements and a minimal proposed Python API.

## Repository layout (current)

- `documents/`
  - `sw_req.md` — Software Requirements Specification (SRS).

