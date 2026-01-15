# Architecture (WIP)

This project is mid-refactor into a modular structure. The current focus is a minimal bootstrap with shared utilities; inference modules will be added later.

## Current modules

- `main.py`: bootstrap entry point; ensures runtime directories, sets up logging, and (optionally) validates required environment variables.
- `config/`: environment-backed settings and flags
  - `config/settings.py`: loads `.env` and exposes runtime settings
- `core/`: core runtime modules
  - `core/camera.py`: camera input handling and reconnection
  - `core/fps.py`: FPS tracking with a rolling time window
- `detection/`: detection helpers
  - `detection/parser.py`: parse model outputs
  - `detection/filter.py`: filter labels of interest
  - `detection/visualizer.py`: draw annotations
- `upload/`: upload pipeline
  - `upload/pipeline.py`: upload orchestration
  - `upload/batcher.py`: crack debounce logic
  - `upload/s3_uploader.py`: S3 upload helper
  - `upload/api_client.py`: API posting with retries
  - `upload/deduplicator.py`: track deduplication
- `utils/`: small shared helpers
  - `utils/logger.py`: logging configuration
  - `utils/directories.py`: runtime directory creation
  - `utils/validators.py`: environment validation

## Intended direction

The refactor is incremental. Planned modules include:

- `core/`: camera handling, device selection, inference loop
- `detection/`: parsing and visualization
- `upload/`: batching, de-duplication, and upload pipeline

## Principles

- Small, composable modules with single responsibilities
- Keep startup fast and observable (logs + environment checks)
- Only add modules when their performance impact is understood
