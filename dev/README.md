<!-- markdownlint-disable MD033 -->

# How to make dummy data and train?
 
## Easiest way:
```bash
make run
```
It will run 
  make pull-> 
  make log_dir->
  make clear_data-> 
  make make_data-> 
  make train 
subsequently.

## Otherwise:
If you don't have data yet
```bash
make make_data
make log_dir
make train
```

If you have one
```bash
make clear_data
make make_data
make log_dir
make train
```

If you want to update the branch
```bash
make pull
```

# StormCast Python Pipeline (`run_stormcast.py`)

This script provides a unified, configurable Python pipeline for managing data preparation, training, inference, and cleanup tasks for the StormCast project. It is a flexible and robust alternative to the Makefile workflow.

## Quick Start

To run the full pipeline (pull, log directory setup, cache creation, data cleanup, data creation, and training):

```bash
python run_stormcast.py run
```

## Basic Commands

You can run individual steps using the following commands:

- **Pull latest changes from git:**
  ```bash
  python run_stormcast.py pull
  ```

- **Create dummy data:**
  ```bash
  python run_stormcast.py make_dummy
  ```

- **Generate training/validation data:**
  ```bash
  python run_stormcast.py make_data
  ```

- **Ensure log directory exists:**
  ```bash
  python run_stormcast.py log_dir
  ```

- **Train a model (regression or diffusion):**
  ```bash
  python run_stormcast.py train --model-type regression
  python run_stormcast.py train --model-type diffusion
  ```

- **Train regression or diffusion model directly:**
  ```bash
  python run_stormcast.py train_regression
  python run_stormcast.py train_diffusion
  ```

- **Run inference using latest checkpoints:**
  ```bash
  python run_stormcast.py inference
  ```

- **Remove generated data directories:**
  ```bash
  python run_stormcast.py clear_data
  ```

- **Remove run directory:**
  ```bash
  python run_stormcast.py clear_rundir
  ```

- **Create cache for the specified variable:**
  ```bash
  python run_stormcast.py make_cache
  ```

## Configuration

All paths, variables, and overrides are managed in `master_config.yaml`.  
You can override the config file or specific variables using command-line arguments:

```bash
python run_stormcast.py train --config custom_config.yaml --variable t2m --experiment-name my_exp
```

## Features

- Modular pipeline steps with robust error handling and logging.
- Dynamic updates to dataset, training, and model config files (YAML) with comment preservation.
- Automatic checkpoint detection for training and inference.
- Flexible command-line interface for full or partial pipeline execution.
- All paths and settings are managed via a single YAML config file.

## Help

To see all available commands and options:

```bash
python run_stormcast.py --help
```

---
**Note:**  
This script is intended to replace the Makefile workflow and provide a more maintainable, extensible, and user-friendly interface for StormCast experiments.