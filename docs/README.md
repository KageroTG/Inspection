# Inspection Bootstrap

This repo is being migrated to a modular structure. For now it includes a minimal bootstrap (`main.py`), a `config/` package for settings, and a small `utils/` package for logging, directory setup, and environment validation.

## Entry point

`main.py` performs a lightweight startup sequence and runs the camera + detection loop:

- Loads environment variables from `.env` if present.
- Ensures runtime directories exist (`logs/`, `images/`, `videos/`).
- Configures logging to both console and a timestamped log file.
- Optionally validates required environment variables when `STRICT_ENV_VALIDATION=1`.
- Opens the configured camera source and (optionally) shows frames when `SHOW_WINDOW=1`.
- Tracks FPS with a rolling window size defined by `FPS_WINDOW`.
- Runs YOLO inference, parses detections, filters labels, and draws annotations.
- Uploads annotated frames for immediate labels and debounces crack uploads.

Run the bootstrap:

```bash
python main.py
```

Enable strict validation:

```bash
STRICT_ENV_VALIDATION=1 python main.py
```


## Utilities

### `utils/logger.py`

- `setup_logging(log_dir, logger_name="edge_inference")`
- Configures `logging` to write a timestamped file under `log_dir` and stream to stdout.
- Returns the named logger for use across modules.

### `utils/directories.py`

- `ensure_directories(*paths)`
- Creates one or more directories if they do not exist.
- Used by `main.py` to set up `logs/` and `images/`.

### `utils/validators.py`

- `validate_environment(api_url, api_key, aws_access_key, aws_secret_key, logger)`
- Verifies required environment variables and raises `SystemExit(1)` with a clear error log when any are missing.

### `utils/__init__.py`

- Marks `utils/` as a Python package.
- Contains a short module docstring only.

## Configuration

### `config/settings.py`

- Centralizes runtime flags and environment-derived settings.
- Calls `load_dotenv()` so `.env` values are available across modules.

### Recording settings

Set this in `.env`:

- `RECORD=1` to enable recording (default off). When enabled, all frames are recorded.

## Core

### `core/camera.py`

- Wraps OpenCV capture for RTSP, USB indexes, and local video files.
- Includes basic reconnect behavior and end-of-file handling.

## Detection

### `detection/parser.py`

- Parses YOLO results into normalized detection dictionaries.

### `detection/filter.py`

- Filters detections by `INTERESTING_LABELS` (or a provided allow-list).

### `detection/visualizer.py`

- Draws bounding boxes and labels on frames (same style as the legacy script).

## Project scaffolding

- `pyproject.toml`: pytest defaults (`-vv -ra`, `tests/` discovery).
- `Makefile`: common commands (`make run`, `make test`, `make lint`).
- `docs/architecture.md`: refactor direction and module layout.

## Notes

- `inferencedashboard.py` and `inference_latest.py` are legacy/experimental scripts and are not invoked by `main.py`.
- Add or move modules incrementally as you continue the refactor (e.g., `core/`, `detection/`, `upload/`).
