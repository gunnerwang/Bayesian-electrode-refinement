#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the LLM-advisor calibration JSON block (schema: docs/calibration.md).

The output feeds `refinement/refiner_llm_guided.py --calibration <file>`.

Worked 2-step recipe (small labeled tuning set, e.g. 10 images)
----------------------------------------------------------------

Step 1 — input_vs_gt block (raw input masks vs per-instance GT, including
`gt_over_input_width_ratio`, which no other shipped tool computes):

    python scripts/make_calibration.py \\
        --baseline-dir data/Full_Instances/repaired_masks \\
        --gt-dir data/Full_Instances/masks \\
        --output calibration.json

Step 2 — probes block. For each fixed probe configuration, run a refiner
entry point with that configuration, evaluate its output, then re-run this
script with one --probe per configuration:

    python refinement/refiner_llm_guided.py --no-use-llm \\
        --thickness-factor 1.3 --transform-ratio 0.3 \\
        --transform-intensity 0.4 --min-separation 8 \\
        --output-dir out/probe1
    python evaluation/evaluate_instance_masks.py \\
        --refined-dir out/probe1 \\
        --baseline-dir data/Full_Instances/repaired_masks \\
        --gt-dir data/Full_Instances/masks \\
        --output out/probe1_metrics.json
    python scripts/make_calibration.py \\
        --baseline-dir data/Full_Instances/repaired_masks \\
        --gt-dir data/Full_Instances/masks \\
        --probe "thickness_factor=1.3,transform_ratio=0.3,transform_intensity=0.4,min_separation=8,metrics=out/probe1_metrics.json" \\
        --output calibration.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics_system import MaskMetricsCalculator  # noqa: E402

PROBE_PARAM_TYPES = {
    'thickness_factor': float,
    'transform_ratio': float,
    'transform_intensity': float,
    'min_separation': int,
}


def compute_input_vs_gt(baseline_dir, gt_dir):
    """Per-instance precision/recall/IoU means of the raw input masks vs GT,
    plus gt_over_input_width_ratio (mean over instances of GT x-extent /
    input x-extent of the binarized masks > 127)."""
    calculator = MaskMetricsCalculator()
    ious, precisions, recalls, width_ratios = [], [], [], []
    tuning_images = set()
    names = sorted(f for f in os.listdir(baseline_dir)
                   if f.endswith('.png') and '_instance_' in f)
    for name in names:
        gt_path = os.path.join(gt_dir, name)
        if not os.path.exists(gt_path):
            continue
        inp = cv2.imread(os.path.join(baseline_dir, name), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if inp is None or gt is None:
            continue
        if inp.shape != gt.shape:
            print(f"[make_calibration] WARNING: shape mismatch for {name}, resizing input to GT")
            inp = cv2.resize(inp, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        ious.append(calculator.calculate_iou(inp, gt))
        precision, recall, _ = calculator.calculate_precision_recall_f1(inp, gt)
        precisions.append(precision)
        recalls.append(recall)
        inp_x = np.where(inp > 127)[1]
        gt_x = np.where(gt > 127)[1]
        if inp_x.size > 0 and gt_x.size > 0:
            width_ratios.append((gt_x.max() - gt_x.min() + 1.0) /
                                (inp_x.max() - inp_x.min() + 1.0))
        tuning_images.add(name.split('_instance_')[0])
    if not ious:
        sys.exit(f"No matching '<image>_instance_<id>.png' pairs found between "
                 f"{baseline_dir} and {gt_dir}")
    block = {
        'n_instances': len(ious),
        'iou_mean': round(float(np.mean(ious)), 4),
        'precision_mean': round(float(np.mean(precisions)), 4),
        'recall_mean': round(float(np.mean(recalls)), 4),
        'gt_over_input_width_ratio': round(float(np.mean(width_ratios)), 4),
    }
    return block, sorted(tuning_images)


def parse_probe(spec):
    """Parse one --probe spec into a calibration 'probes' entry.

    Format: thickness_factor=F,transform_ratio=F,transform_intensity=F,
            min_separation=I,metrics=<evaluate_instance_masks.py output JSON>
    """
    fields = dict(part.split('=', 1) for part in spec.split(',') if '=' in part)
    missing = [k for k in list(PROBE_PARAM_TYPES) + ['metrics'] if k not in fields]
    if missing:
        sys.exit(f"--probe '{spec}' is missing key(s): {', '.join(missing)}")
    params = {key: cast(fields[key]) for key, cast in PROBE_PARAM_TYPES.items()}
    with open(fields['metrics']) as f:
        data = json.load(f)
    stats = data.get('overall_statistics', data)
    entry = {'params': params}
    if 'total_instances' in stats:
        entry['n_instances'] = stats['total_instances']
    for key in ('refined_iou_mean', 'refined_precision_mean', 'refined_recall_mean'):
        if key not in stats:
            sys.exit(f"{fields['metrics']} has no '{key}' — expected the JSON written "
                     f"by evaluation/evaluate_instance_masks.py")
        entry[key] = round(float(stats[key]), 4)
    return entry


def main():
    parser = argparse.ArgumentParser(
        description='Assemble the LLM-advisor calibration JSON (docs/calibration.md)',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--baseline-dir', required=True,
                        help='Directory of raw input instance masks (<image>_instance_<id>.png)')
    parser.add_argument('--gt-dir', required=True,
                        help='Directory of per-instance ground-truth masks, same naming')
    parser.add_argument('--probe', action='append', default=[], metavar='SPEC',
                        help='Repeatable. One probe configuration plus its metrics JSON from '
                             'evaluation/evaluate_instance_masks.py, e.g. '
                             '"thickness_factor=1.3,transform_ratio=0.3,'
                             'transform_intensity=0.4,min_separation=8,'
                             'metrics=out/probe1_metrics.json"')
    parser.add_argument('--output', required=True, help='Output calibration JSON path')
    args = parser.parse_args()

    input_vs_gt, tuning_images = compute_input_vs_gt(args.baseline_dir, args.gt_dir)
    calibration = {
        'tuning_images': tuning_images,
        'input_vs_gt': input_vs_gt,
        'probes': [parse_probe(spec) for spec in args.probe],
    }
    output_parent = os.path.dirname(args.output)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f"[make_calibration] {input_vs_gt['n_instances']} instances, "
          f"{len(calibration['probes'])} probe(s) -> {args.output}")
    if not calibration['probes']:
        print("[make_calibration] NOTE: no --probe given; add probe entries "
              "(step 2 of the recipe in this script's --help) before using "
              "the calibration with the advisor.")


if __name__ == '__main__':
    main()
