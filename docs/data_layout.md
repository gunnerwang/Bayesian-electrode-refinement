# Expected data layout

```
data/
├── Full_Instances/
│   ├── instance_info/
│   │   └── <image>_info.json        # {"image_height": int, "image_width": int,
│   │                                #  "instances": [{"id": int,
│   │                                #   "bbox": [x, y, w, h], ...}, ...]}
│   ├── repaired_masks/
│   │   └── <image>_instance_<id>.png  # per-instance input mask, bbox-sized,
│   │                                  # foreground > 127
│   └── masks/
│       └── <image>_instance_<id>.png  # per-instance ground truth
│                                      # (evaluation / calibration only)
└── Origin/
    ├── images_enhanced/<image>.png    # enhanced grayscale/BGR image
    ├── coarse_masks/<image>.png       # original coarse full mask (optional)
    └── masks/<image>.png              # full-image ground truth
                                       # (evaluation / tuning only)
```

Per-instance ground truth (`Full_Instances/masks/`) follows the same naming
as `repaired_masks` and is the full-image ground truth (`Origin/masks/`)
cropped at the instance's `bbox` from `instance_info` — i.e. each
`<image>_instance_<id>.png` must have the same shape as the corresponding
input mask in `repaired_masks/`. It is used only by
`evaluation/evaluate_instance_masks.py` and the calibration recipe
([calibration.md](calibration.md)), never by the refiner at inference.

Notes
- `<image>` is the file basename without extension; it must match across all
  directories.
- The refiner consumes `instance_info` + `repaired_masks` + the image; ground
  truth is used only for evaluation and for tuning-set calibration/selection.
