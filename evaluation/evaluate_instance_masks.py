#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instance-level mask evaluation module.

This module handles evaluation of instance-level masks, grouping them by base name.
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

try:
    from .metrics_system import (
        MaskMetricsCalculator,
        MetricsAggregator,
        ComparisonMetrics
    )
except ImportError:  # also allow flat (non-package) imports
    from metrics_system import (
        MaskMetricsCalculator,
        MetricsAggregator,
        ComparisonMetrics
    )


def extract_base_name(filename: str) -> str:
    """Extract base name from instance mask filename.
    
    Example: 'sample_image_instance_0.png'
    Returns: 'sample_image'
    """
    if '_instance_' in filename:
        return filename.split('_instance_')[0]
    return filename.replace('.png', '')


def group_instance_masks(mask_dir: str) -> Dict[str, List[str]]:
    """Group instance masks by their base name.
    
    Returns:
        Dictionary mapping base names to lists of instance mask filenames
    """
    grouped_masks = defaultdict(list)
    
    for filename in os.listdir(mask_dir):
        if filename.endswith('.png') and '_instance_' in filename:
            base_name = extract_base_name(filename)
            grouped_masks[base_name].append(filename)
    
    # Sort instance masks within each group
    for base_name in grouped_masks:
        grouped_masks[base_name].sort(key=lambda x: int(x.split('_instance_')[1].replace('.png', '')))
    
    return dict(grouped_masks)


def evaluate_instance_group(base_name: str,
                          instance_masks: List[str],
                          refined_dir: str,
                          baseline_dir: str,
                          gt_dir: str,
                          calculator: MaskMetricsCalculator) -> Tuple[Dict, List[Dict]]:
    """Evaluate all instances for a single base image.
    
    Returns:
        Tuple of (aggregated metrics dict, list of detailed metrics dicts)
    """
    group_metrics = []
    detailed_metrics = []
    
    for mask_filename in instance_masks:
        refined_path = os.path.join(refined_dir, mask_filename)
        baseline_path = os.path.join(baseline_dir, mask_filename)
        gt_path = os.path.join(gt_dir, mask_filename)
        
        # Check if all files exist
        if not all(os.path.exists(p) for p in [refined_path, baseline_path, gt_path]):
            continue
        
        # Load masks
        refined = cv2.imread(refined_path, cv2.IMREAD_GRAYSCALE)
        baseline = cv2.imread(baseline_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if any(img is None for img in [refined, baseline, gt]):
            continue
        
        # Ensure same dimensions
        if refined.shape != gt.shape:
            refined = cv2.resize(refined, (gt.shape[1], gt.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        if baseline.shape != gt.shape:
            baseline = cv2.resize(baseline, (gt.shape[1], gt.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Calculate metrics for this instance
        instance_metrics = calculator.compare_masks(baseline, refined, gt, mask_filename)
        group_metrics.append(instance_metrics)
        
        # Store detailed metrics (aligned with assembled metrics, without consistency)
        detailed_metrics.append({
            'base_name': base_name,
            'instance_mask': mask_filename,
            'baseline_iou': instance_metrics.baseline_metrics.iou,
            'refined_iou': instance_metrics.refined_metrics.iou,
            'iou_improvement': instance_metrics.iou_improvement,
            'baseline_dice': instance_metrics.baseline_metrics.dice,
            'refined_dice': instance_metrics.refined_metrics.dice,
            'dice_improvement': instance_metrics.dice_improvement,
            'baseline_precision': instance_metrics.baseline_metrics.precision,
            'refined_precision': instance_metrics.refined_metrics.precision,
            'baseline_recall': instance_metrics.baseline_metrics.recall,
            'refined_recall': instance_metrics.refined_metrics.recall,
            'baseline_f1': instance_metrics.baseline_metrics.f1_score,
            'refined_f1': instance_metrics.refined_metrics.f1_score,
            'baseline_pixel_accuracy': instance_metrics.baseline_metrics.pixel_accuracy,
            'refined_pixel_accuracy': instance_metrics.refined_metrics.pixel_accuracy,
            'baseline_boundary_iou': instance_metrics.baseline_metrics.boundary_iou,
            'refined_boundary_iou': instance_metrics.refined_metrics.boundary_iou,
            'baseline_hausdorff_distance': instance_metrics.baseline_metrics.hausdorff_distance,
            'refined_hausdorff_distance': instance_metrics.refined_metrics.hausdorff_distance,
            'pixel_change_rate': instance_metrics.pixel_change_rate
        })
    
    if not group_metrics:
        return None, []
    
    # Aggregate metrics for this group (aligned with assembled metrics)
    aggregated = {
        'base_name': base_name,
        'num_instances': len(group_metrics),
        'avg_baseline_iou': np.mean([m.baseline_metrics.iou for m in group_metrics]),
        'avg_refined_iou': np.mean([m.refined_metrics.iou for m in group_metrics]),
        'avg_iou_improvement': np.mean([m.iou_improvement for m in group_metrics]),
        'avg_baseline_dice': np.mean([m.baseline_metrics.dice for m in group_metrics]),
        'avg_refined_dice': np.mean([m.refined_metrics.dice for m in group_metrics]),
        'avg_dice_improvement': np.mean([m.dice_improvement for m in group_metrics]),
        'avg_baseline_precision': np.mean([m.baseline_metrics.precision for m in group_metrics]),
        'avg_refined_precision': np.mean([m.refined_metrics.precision for m in group_metrics]),
        'avg_baseline_recall': np.mean([m.baseline_metrics.recall for m in group_metrics]),
        'avg_refined_recall': np.mean([m.refined_metrics.recall for m in group_metrics]),
        'avg_baseline_f1': np.mean([m.baseline_metrics.f1_score for m in group_metrics]),
        'avg_refined_f1': np.mean([m.refined_metrics.f1_score for m in group_metrics]),
        'avg_baseline_pixel_accuracy': np.mean([m.baseline_metrics.pixel_accuracy for m in group_metrics]),
        'avg_refined_pixel_accuracy': np.mean([m.refined_metrics.pixel_accuracy for m in group_metrics]),
        'avg_baseline_boundary_iou': np.mean([m.baseline_metrics.boundary_iou for m in group_metrics]),
        'avg_refined_boundary_iou': np.mean([m.refined_metrics.boundary_iou for m in group_metrics]),
        'avg_baseline_hausdorff_distance': np.mean([m.baseline_metrics.hausdorff_distance for m in group_metrics if not np.isinf(m.baseline_metrics.hausdorff_distance)]) if any(not np.isinf(m.baseline_metrics.hausdorff_distance) for m in group_metrics) else float('inf'),
        'avg_refined_hausdorff_distance': np.mean([m.refined_metrics.hausdorff_distance for m in group_metrics if not np.isinf(m.refined_metrics.hausdorff_distance)]) if any(not np.isinf(m.refined_metrics.hausdorff_distance) for m in group_metrics) else float('inf'),
        'avg_pixel_change_rate': np.mean([m.pixel_change_rate for m in group_metrics]),
        'total_iou_improvement': sum([m.iou_improvement for m in group_metrics]),
        'instances_improved': sum([1 for m in group_metrics if m.iou_improvement > 0]),
        'instances_degraded': sum([1 for m in group_metrics if m.iou_improvement < 0]),
        'instances_unchanged': sum([1 for m in group_metrics if abs(m.iou_improvement) < 0.001]),
    }
    
    return aggregated, detailed_metrics


def evaluate_instance_masks(refined_dir: str,
                          baseline_dir: str,
                          gt_dir: str,
                          output_path: str) -> Dict:
    """Evaluate instance-level masks grouped by base name.
    
    Args:
        refined_dir: Directory containing refined instance masks
        baseline_dir: Directory containing baseline instance masks
        gt_dir: Directory containing ground truth instance masks
        output_path: Path to save evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    calculator = MaskMetricsCalculator()
    
    # Group masks by base name
    print("Grouping instance masks by base name...")
    grouped_masks = group_instance_masks(refined_dir)
    
    if not grouped_masks:
        print("No instance masks found in refined directory")
        return {}
    
    print(f"Found {len(grouped_masks)} base images with instance masks")
    
    # Evaluate each group
    all_results = []
    all_detailed_metrics = []
    
    for base_name, instance_masks in tqdm(grouped_masks.items(), desc="Evaluating image groups"):
        group_result, detailed_metrics = evaluate_instance_group(
            base_name, instance_masks, refined_dir, baseline_dir, gt_dir, calculator
        )
        
        if group_result:
            all_results.append(group_result)
            all_detailed_metrics.extend(detailed_metrics)
    
    if not all_results:
        print("No valid instance groups found for evaluation")
        return {}
    
    # Calculate overall statistics (aligned with assembled evaluation naming)
    overall_stats = {
        'num_images': len(all_results),
        'total_instances': sum(r['num_instances'] for r in all_results),
        # Mean metrics (using same naming as assembled)
        'baseline_iou_mean': np.mean([r['avg_baseline_iou'] for r in all_results]),
        'refined_iou_mean': np.mean([r['avg_refined_iou'] for r in all_results]),
        'iou_improvement_mean': np.mean([r['avg_iou_improvement'] for r in all_results]),
        'baseline_dice_mean': np.mean([r['avg_baseline_dice'] for r in all_results]),
        'refined_dice_mean': np.mean([r['avg_refined_dice'] for r in all_results]),
        'dice_improvement_mean': np.mean([r['avg_dice_improvement'] for r in all_results]),
        'baseline_precision_mean': np.mean([r['avg_baseline_precision'] for r in all_results]),
        'refined_precision_mean': np.mean([r['avg_refined_precision'] for r in all_results]),
        'baseline_recall_mean': np.mean([r['avg_baseline_recall'] for r in all_results]),
        'refined_recall_mean': np.mean([r['avg_refined_recall'] for r in all_results]),
        'baseline_f1_score_mean': np.mean([r['avg_baseline_f1'] for r in all_results]),
        'refined_f1_score_mean': np.mean([r['avg_refined_f1'] for r in all_results]),
        'baseline_pixel_accuracy_mean': np.mean([r['avg_baseline_pixel_accuracy'] for r in all_results]),
        'refined_pixel_accuracy_mean': np.mean([r['avg_refined_pixel_accuracy'] for r in all_results]),
        'baseline_boundary_iou_mean': np.mean([r['avg_baseline_boundary_iou'] for r in all_results]),
        'refined_boundary_iou_mean': np.mean([r['avg_refined_boundary_iou'] for r in all_results]),
        'baseline_hausdorff_distance_mean': np.mean([r['avg_baseline_hausdorff_distance'] for r in all_results if not np.isinf(r['avg_baseline_hausdorff_distance'])]) if any(not np.isinf(r['avg_baseline_hausdorff_distance']) for r in all_results) else float('inf'),
        'refined_hausdorff_distance_mean': np.mean([r['avg_refined_hausdorff_distance'] for r in all_results if not np.isinf(r['avg_refined_hausdorff_distance'])]) if any(not np.isinf(r['avg_refined_hausdorff_distance']) for r in all_results) else float('inf'),
        'pixel_change_rate_mean': np.mean([r['avg_pixel_change_rate'] for r in all_results]),
        # Standard deviations for all metrics
        'baseline_iou_std': np.std([r['avg_baseline_iou'] for r in all_results]),
        'refined_iou_std': np.std([r['avg_refined_iou'] for r in all_results]),
        'iou_improvement_std': np.std([r['avg_iou_improvement'] for r in all_results]),
        'baseline_dice_std': np.std([r['avg_baseline_dice'] for r in all_results]),
        'refined_dice_std': np.std([r['avg_refined_dice'] for r in all_results]),
        'dice_improvement_std': np.std([r['avg_dice_improvement'] for r in all_results]),
        'baseline_precision_std': np.std([r['avg_baseline_precision'] for r in all_results]),
        'refined_precision_std': np.std([r['avg_refined_precision'] for r in all_results]),
        'baseline_recall_std': np.std([r['avg_baseline_recall'] for r in all_results]),
        'refined_recall_std': np.std([r['avg_refined_recall'] for r in all_results]),
        'baseline_f1_score_std': np.std([r['avg_baseline_f1'] for r in all_results]),
        'refined_f1_score_std': np.std([r['avg_refined_f1'] for r in all_results]),
        'baseline_pixel_accuracy_std': np.std([r['avg_baseline_pixel_accuracy'] for r in all_results]),
        'refined_pixel_accuracy_std': np.std([r['avg_refined_pixel_accuracy'] for r in all_results]),
        'baseline_boundary_iou_std': np.std([r['avg_baseline_boundary_iou'] for r in all_results]),
        'refined_boundary_iou_std': np.std([r['avg_refined_boundary_iou'] for r in all_results]),
        'baseline_hausdorff_distance_std': np.std([r['avg_baseline_hausdorff_distance'] for r in all_results if not np.isinf(r['avg_baseline_hausdorff_distance'])]) if any(not np.isinf(r['avg_baseline_hausdorff_distance']) for r in all_results) else 0.0,
        'refined_hausdorff_distance_std': np.std([r['avg_refined_hausdorff_distance'] for r in all_results if not np.isinf(r['avg_refined_hausdorff_distance'])]) if any(not np.isinf(r['avg_refined_hausdorff_distance']) for r in all_results) else 0.0,
        'pixel_change_rate_std': np.std([r['avg_pixel_change_rate'] for r in all_results]),
        # Instance-specific stats
        'total_instances_improved': sum(r['instances_improved'] for r in all_results),
        'total_instances_degraded': sum(r['instances_degraded'] for r in all_results),
        'total_instances_unchanged': sum(r['instances_unchanged'] for r in all_results),
    }
    
    # Save results
    results = {
        'overall_statistics': overall_stats,
        'per_image_results': all_results
    }
    
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    # Save JSON report
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed CSV
    if all_detailed_metrics:
        df = pd.DataFrame(all_detailed_metrics)
        csv_path = output_path.replace('.json', '_detailed.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved detailed CSV to: {csv_path}")
    
    # Save summary JSON
    summary_path = output_path.replace('.json', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    print(f"Saved summary to: {summary_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Instance-Level Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total images evaluated: {overall_stats['num_images']}")
    print(f"Total instances evaluated: {overall_stats['total_instances']}")
    print(f"Average baseline IoU: {overall_stats['baseline_iou_mean']:.4f} (±{overall_stats['baseline_iou_std']:.4f})")
    print(f"Average refined IoU: {overall_stats['refined_iou_mean']:.4f} (±{overall_stats['refined_iou_std']:.4f})")
    print(f"Average IoU improvement: {overall_stats['iou_improvement_mean']:.4f} (±{overall_stats['iou_improvement_std']:.4f})")
    print(f"Average Dice improvement: {overall_stats['dice_improvement_mean']:.4f} (±{overall_stats['dice_improvement_std']:.4f})")
    print(f"Average pixel change rate: {overall_stats['pixel_change_rate_mean']:.4f} (±{overall_stats['pixel_change_rate_std']:.4f})")
    print(f"Instances improved: {overall_stats['total_instances_improved']}")
    print(f"Instances degraded: {overall_stats['total_instances_degraded']}")
    print(f"Instances unchanged: {overall_stats['total_instances_unchanged']}")
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Per-instance evaluation: refined vs baseline vs ground-truth instance masks')
    parser.add_argument('--refined-dir', required=True,
                        help='Directory of refined instance masks (<image>_instance_<id>.png)')
    parser.add_argument('--baseline-dir', required=True,
                        help='Directory of baseline (input) instance masks, same naming')
    parser.add_argument('--gt-dir', required=True,
                        help='Directory of per-instance ground-truth masks, same naming')
    parser.add_argument('--output', required=True,
                        help='Output JSON path for detailed + aggregated metrics')
    args = parser.parse_args()
    evaluate_instance_masks(args.refined_dir, args.baseline_dir, args.gt_dir, args.output)
