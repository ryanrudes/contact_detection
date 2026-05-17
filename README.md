# event-detection

NumPy-first utilities for detecting quiet intervals and likely body/support
contacts from mocap-style time series.

## APIs

- `src.silence_detection.detect_quiet_intervals(t, X, config=...)`
  detects local quietness for scalar, vector position, generic vector, and
  quaternion signals.
- `src.silence_detection.detect_z_quiet_intervals(t, z)`
  is the backward-compatible z-axis wrapper.
- `src.contact_detection.detect_contact_intervals(t, points, config=...)`
  detects likely support contacts from points shaped `(N, 3)` or `(N, K, 3)`.

The contact detector bootstraps from quiet keypoint intervals, fits a support
surface, scores clearance/slip/speed/quietness relative to that support, and
cleans the final mask temporally. The v1 support defaults are robust plane
fitting with heightmap fallback for non-coplanar support candidates.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
```
