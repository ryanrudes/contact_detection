## Repository Guidance

- This repo is a NumPy-first contact-detection package plus a small CLI for
  Vicon trial diagnostics.
- Use `python main.py --config configs/config.yaml` as the default workflow.
- Runtime settings live in YAML configs. Prefer editing `configs/config.yaml`
  or adding another YAML config over adding new hard-coded CLI defaults.
- The supported trial input for `main.py` is unified NPZ data with `t`,
  `vicon__body_names`, and `vicon__body_pos`.
- Local trial data and generated plots belong under ignored directories such as
  `data/`, `outputs/`, and `outputs_*`; do not commit recording data or PNG
  diagnostics unless the user explicitly asks for fixtures/artifacts.

## Skate/Vicon Classification Preferences

- Classify each foot separately as `air`, `ground`, or `skateboard`; do not
  collapse skate diagnostics into one global contact mask.
- The floor is not assumed to be world `z=0`. Calibrate from the data, using
  either the scalar lower-foot-height model or robust plane fitting.
- Board contact must require horizontal proximity to the skateboard rigid body,
  not only vertical distance to a plane.
- When the board is moving, use world-frame `|v_foot - v_board|` as primary
  board-contact evidence. Use static geometry/speed checks mainly when the
  board is nearly stationary.
- Contact diagnostic plots should shade per-foot state masks.

## Package Structure

- Public imports should come from `contact_detection`.
- Compatibility imports through `src.contact_detection` and
  `src.silence_detection` should continue to work.
- `src/contact_detection/foot_support.py` contains the per-foot
  air/ground/skateboard classifier used by `main.py`.
- `src/contact_detection/contact.py` contains the generic support-surface and
  contact interval APIs.
- `src/contact_detection/quiet.py` contains scalar, vector, and quaternion quiet
  detection.

## Verification

- Run `python -m unittest discover -s tests -v`.
- Run `ruff check .`.
- If using the local uv environment, `uv run --with ruff ruff check .` is fine.
