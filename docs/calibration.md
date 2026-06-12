# Advisor calibration block

When a calibration JSON is passed to the LLM advisor
(`refiner_llm_guided.py --calibration <file>`), the advisor uses the
calibrated prompt: per-image statistics plus dataset calibration evidence,
with all parameter decisions left to the model. Without it, the legacy
anchored prompt is used.

## Schema (see example_calibration.json)

```json
{
  "input_vs_gt": {
    "n_instances": 1003,
    "iou_mean": 0.79,          // raw input masks vs per-instance GT
    "precision_mean": 0.93,
    "recall_mean": 0.83,
    "gt_over_input_width_ratio": 1.15
  },
  "probes": [
    {
      "params": {"thickness_factor": 1.3, "transform_ratio": 0.3,
                  "transform_intensity": 0.4, "min_separation": 8},
      "refined_iou_mean": 0.85,      // instance-level result of this fixed
      "refined_precision_mean": 0.86, // config on the tuning images
      "refined_recall_mean": 0.98
    }
  ]
}
```

## How to compute it for a new dataset

Both blocks are derived programmatically from a small labeled tuning set
(we use 10 images) with `scripts/make_calibration.py`:

1. **input_vs_gt** — compare the raw input instance masks against the
   per-instance ground truth of the tuning images
   (`Full_Instances/masks/`, see
   [data_layout.md](data_layout.md)). `scripts/make_calibration.py` computes
   every field of this block — including `gt_over_input_width_ratio`, the
   mean GT instance width divided by the mean input instance width over the
   tuning instances — directly from the two mask directories:

   ```bash
   python scripts/make_calibration.py \
       --baseline-dir data/Full_Instances/repaired_masks \
       --gt-dir data/Full_Instances/masks \
       --output calibration.json
   ```

2. **probes** — run the refiner on the tuning images with a handful of fixed
   configurations spanning the parameter space (we use 5; fewer evaluations
   than a 15-trial random/Bayesian search). Each probe configuration is set
   via the refiner CLI flags `--thickness-factor`, `--transform-ratio`,
   `--transform-intensity` and `--min-separation`; its instance-level
   metrics (`refined_iou_mean`, `refined_precision_mean`,
   `refined_recall_mean`) come from an `evaluation/evaluate_instance_masks.py`
   run on that probe's output directory. Pass each probe's parameters and
   metrics JSON to `scripts/make_calibration.py` to fill the `probes` list
   (see `python scripts/make_calibration.py --help` for the probe-metrics
   inputs).

The calibration is computed once per dataset; inference remains fully
GT-free and per-image.
