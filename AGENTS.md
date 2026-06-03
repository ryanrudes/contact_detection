## Learned User Preferences

- Classify each foot separately as `air`, `ground`, or `skateboard`; do not collapse skate diagnostics into one global contact mask.
- The floor is not assumed to be world `z=0`. Calibrate from the data, using either the scalar lower-foot-height model or robust plane fitting.
- Board contact must require horizontal proximity to the skateboard rigid body, not only vertical distance to a plane.
- When the board is moving, use world-frame `|v_foot - v_board|` as primary board-contact evidence. Use static geometry/speed checks mainly when the board is nearly stationary.
- Contact diagnostic plots should shade per-foot state masks.
- Runtime settings live in YAML configs. Prefer editing `configs/config.yaml` or adding another YAML config over new hard-coded CLI defaults.
- Local trial data and generated plots belong under ignored directories such as `data/`, `outputs/`, and `outputs_*`; do not commit recording data or PNG diagnostics unless the user explicitly asks for fixtures/artifacts.

## Learned Workspace Facts

- NumPy-first contact-detection package plus `python main.py --config configs/config.yaml` for per-foot support-state diagnostic plots.
- Trial input is motion-sync `synced.npz` (file or directory tree). Prefer `SyncClip.load` + `register_contacts` + `clip.detect(SKATE_FOOT_SUPPORT)` + `clip.save`, or `motion-sync detect foot-support <demo>`. `main.py` loads via `SyncClip` and writes diagnostic PNGs only.
- Public imports should come from `contact_detection`. Core modules: `geometry.py` (body-to-contact-frame surfaces), `foot_support.py` (per-foot classifier), `contact.py` (support surfaces and intervals), `quiet.py` (scalar/vector/quaternion quiet detection).
- Run `python -m unittest discover -s tests -v` and `ruff check .` (or `uv run --with ruff ruff check .`).
