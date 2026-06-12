#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent Metrics System for Mask Refinement Evaluation

This module provides a decoupled metrics system for evaluating mask refinement results.
It can work with any mask data, regardless of how the masks were generated or assembled.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pandas as pd


@dataclass
class MaskMetrics:
    """Container for mask evaluation metrics."""
    image_name: str
    iou: float
    dice: float
    precision: float
    recall: float
    f1_score: float
    pixel_accuracy: float
    boundary_iou: float
    hausdorff_distance: float
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class ComparisonMetrics:
    """Container for comparison between two mask sets."""
    image_name: str
    baseline_metrics: MaskMetrics
    refined_metrics: MaskMetrics
    iou_improvement: float
    dice_improvement: float
    consistency: float
    pixel_change_rate: float
    
    def to_dict(self) -> Dict:
        """Convert comparison metrics to dictionary."""
        result = {
            'image_name': self.image_name,
            'iou_improvement': self.iou_improvement,
            'dice_improvement': self.dice_improvement,
            'consistency': self.consistency,
            'pixel_change_rate': self.pixel_change_rate
        }
        # Add baseline and refined metrics with prefixes
        for key, value in self.baseline_metrics.to_dict().items():
            if key != 'image_name':
                result[f'baseline_{key}'] = value
        for key, value in self.refined_metrics.to_dict().items():
            if key != 'image_name':
                result[f'refined_{key}'] = value
        return result


class MaskMetricsCalculator:
    """Calculate various metrics for mask evaluation."""
    
    def __init__(self, boundary_thickness: int = 3):
        """
        Initialize the metrics calculator.
        
        Args:
            boundary_thickness: Thickness for boundary IoU calculation
        """
        self.boundary_thickness = boundary_thickness
    
    def calculate_iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate Intersection over Union."""
        pred_binary = pred > 127
        gt_binary = gt > 127
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        return intersection / (union + 1e-6)
    
    def calculate_dice(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate Dice coefficient."""
        pred_binary = pred > 127
        gt_binary = gt > 127
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-6)
        
        return dice
    
    def calculate_precision_recall_f1(self, pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        pred_binary = pred > 127
        gt_binary = gt > 127
        
        true_positive = np.logical_and(pred_binary, gt_binary).sum()
        false_positive = np.logical_and(pred_binary, ~gt_binary).sum()
        false_negative = np.logical_and(~pred_binary, gt_binary).sum()
        
        precision = true_positive / (true_positive + false_positive + 1e-6)
        recall = true_positive / (true_positive + false_negative + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return precision, recall, f1
    
    def calculate_pixel_accuracy(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate pixel-wise accuracy."""
        pred_binary = pred > 127
        gt_binary = gt > 127
        
        correct = (pred_binary == gt_binary).sum()
        total = pred_binary.size
        
        return correct / total
    
    def calculate_boundary_iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate boundary IoU."""
        # Get boundaries
        pred_boundary = self._get_boundary(pred)
        gt_boundary = self._get_boundary(gt)
        
        # Dilate boundaries for thickness
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.boundary_thickness, self.boundary_thickness))
        pred_boundary_dilated = cv2.dilate(pred_boundary.astype(np.uint8), kernel)
        gt_boundary_dilated = cv2.dilate(gt_boundary.astype(np.uint8), kernel)
        
        # Calculate IoU of dilated boundaries
        intersection = np.logical_and(pred_boundary_dilated, gt_boundary_dilated).sum()
        union = np.logical_or(pred_boundary_dilated, gt_boundary_dilated).sum()
        
        return intersection / (union + 1e-6)
    
    def calculate_hausdorff_distance(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate Hausdorff distance between boundaries."""
        from scipy.spatial.distance import directed_hausdorff
        
        pred_boundary = self._get_boundary(pred)
        gt_boundary = self._get_boundary(gt)
        
        # Get boundary points
        pred_points = np.column_stack(np.where(pred_boundary))
        gt_points = np.column_stack(np.where(gt_boundary))
        
        # Handle edge cases
        if len(pred_points) == 0 and len(gt_points) == 0:
            # Both masks have no boundary (both empty or both full)
            return 0.0
        elif len(pred_points) == 0:
            # Prediction has no boundary but ground truth does
            # Return the diagonal of the image as max possible distance
            return np.sqrt(gt.shape[0]**2 + gt.shape[1]**2)
        elif len(gt_points) == 0:
            # Ground truth has no boundary but prediction does
            # Return the diagonal of the image as max possible distance
            return np.sqrt(pred.shape[0]**2 + pred.shape[1]**2)
        
        # Calculate bidirectional Hausdorff distance
        d1 = directed_hausdorff(pred_points, gt_points)[0]
        d2 = directed_hausdorff(gt_points, pred_points)[0]
        
        return max(d1, d2)
    
    def _get_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary from binary mask."""
        binary = (mask > 127).astype(np.uint8)
        
        # Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilated = cv2.dilate(binary, kernel)
        eroded = cv2.erode(binary, kernel)
        boundary = dilated - eroded
        
        return boundary > 0
    
    def calculate_all_metrics(self, pred: np.ndarray, gt: np.ndarray, 
                            image_name: str) -> MaskMetrics:
        """Calculate all metrics for a single mask pair."""
        iou = self.calculate_iou(pred, gt)
        dice = self.calculate_dice(pred, gt)
        precision, recall, f1 = self.calculate_precision_recall_f1(pred, gt)
        pixel_accuracy = self.calculate_pixel_accuracy(pred, gt)
        boundary_iou = self.calculate_boundary_iou(pred, gt)
        hausdorff = self.calculate_hausdorff_distance(pred, gt)
        
        return MaskMetrics(
            image_name=image_name,
            iou=iou,
            dice=dice,
            precision=precision,
            recall=recall,
            f1_score=f1,
            pixel_accuracy=pixel_accuracy,
            boundary_iou=boundary_iou,
            hausdorff_distance=hausdorff
        )
    
    def compare_masks(self, baseline: np.ndarray, refined: np.ndarray, 
                     gt: np.ndarray, image_name: str) -> ComparisonMetrics:
        """Compare baseline and refined masks against ground truth."""
        # Calculate metrics for both masks
        baseline_metrics = self.calculate_all_metrics(baseline, gt, image_name)
        refined_metrics = self.calculate_all_metrics(refined, gt, image_name)
        
        # Calculate improvements
        iou_improvement = refined_metrics.iou - baseline_metrics.iou
        dice_improvement = refined_metrics.dice - baseline_metrics.dice
        
        # Calculate consistency between baseline and refined
        consistency = self.calculate_iou(baseline, refined)
        
        # Calculate pixel change rate
        pixel_change_rate = np.sum(baseline != refined) / baseline.size
        
        return ComparisonMetrics(
            image_name=image_name,
            baseline_metrics=baseline_metrics,
            refined_metrics=refined_metrics,
            iou_improvement=iou_improvement,
            dice_improvement=dice_improvement,
            consistency=consistency,
            pixel_change_rate=pixel_change_rate
        )


class MetricsAggregator:
    """Aggregate and analyze metrics across multiple images."""
    
    def __init__(self):
        self.metrics: List[Union[MaskMetrics, ComparisonMetrics]] = []
    
    def add_metrics(self, metrics: Union[MaskMetrics, ComparisonMetrics]):
        """Add metrics for a single image."""
        self.metrics.append(metrics)
    
    def get_summary_statistics(self) -> Dict:
        """Calculate summary statistics across all images."""
        if not self.metrics:
            return {}
        
        # Convert to DataFrame for easy aggregation
        df = pd.DataFrame([m.to_dict() for m in self.metrics])
        
        # Calculate statistics
        summary = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            summary[f'{col}_mean'] = df[col].mean()
            summary[f'{col}_std'] = df[col].std()
            summary[f'{col}_min'] = df[col].min()
            summary[f'{col}_max'] = df[col].max()
            summary[f'{col}_median'] = df[col].median()
        
        summary['num_images'] = len(self.metrics)
        
        return summary
    
    def get_detailed_dataframe(self) -> pd.DataFrame:
        """Get detailed metrics as a DataFrame."""
        if not self.metrics:
            return pd.DataFrame()
        
        return pd.DataFrame([m.to_dict() for m in self.metrics])
    
    def save_report(self, output_path: str):
        """Save comprehensive metrics report."""
        # Create output directory (only when the path has a parent component)
        output_parent = os.path.dirname(output_path)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        
        # Get summary and detailed data
        summary = self.get_summary_statistics()
        detailed_df = self.get_detailed_dataframe()
        
        # Save JSON report
        report = {
            'summary': summary,
            'detailed_metrics': [m.to_dict() for m in self.metrics]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save CSV for detailed metrics
        csv_path = output_path.replace('.json', '_detailed.csv')
        detailed_df.to_csv(csv_path, index=False)
        
        # Save summary as separate file
        summary_path = output_path.replace('.json', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved metrics report to: {output_path}")
        print(f"Saved detailed CSV to: {csv_path}")
        print(f"Saved summary to: {summary_path}")
    
    def print_summary(self, metric_type: str = "comparison"):
        """Print summary statistics to console."""
        summary = self.get_summary_statistics()
        
        if not summary:
            print("No metrics to summarize")
            return
        
        print(f"\n{'='*60}")
        print(f"Metrics Summary ({summary['num_images']} images)")
        print(f"{'='*60}")
        
        if metric_type == "comparison":
            # Key comparison metrics
            key_metrics = [
                ('IoU Improvement', 'iou_improvement'),
                ('Dice Improvement', 'dice_improvement'),
                ('Refined IoU', 'refined_iou'),
                ('Baseline IoU', 'baseline_iou'),
                ('Consistency', 'consistency'),
                ('Pixel Change Rate', 'pixel_change_rate')
            ]
        else:
            # Single mask metrics
            key_metrics = [
                ('IoU', 'iou'),
                ('Dice', 'dice'),
                ('F1 Score', 'f1_score'),
                ('Pixel Accuracy', 'pixel_accuracy'),
                ('Boundary IoU', 'boundary_iou')
            ]
        
        for display_name, metric_name in key_metrics:
            mean_key = f'{metric_name}_mean'
            std_key = f'{metric_name}_std'
            
            if mean_key in summary:
                mean_val = summary[mean_key]
                std_val = summary.get(std_key, 0)
                
                if 'rate' in metric_name or 'improvement' in metric_name:
                    print(f"{display_name:20s}: {mean_val:6.4f} ± {std_val:6.4f}")
                else:
                    print(f"{display_name:20s}: {mean_val:6.4f} ± {std_val:6.4f}")


def evaluate_masks(pred_dir: str, gt_dir: str, output_dir: str, 
                  image_names: Optional[List[str]] = None) -> MetricsAggregator:
    """
    Evaluate a set of predicted masks against ground truth.
    
    Args:
        pred_dir: Directory containing predicted masks
        gt_dir: Directory containing ground truth masks
        output_dir: Directory to save evaluation results
        image_names: Optional list of specific images to evaluate
        
    Returns:
        MetricsAggregator with all computed metrics
    """
    calculator = MaskMetricsCalculator()
    aggregator = MetricsAggregator()
    
    # Get list of images to evaluate
    if image_names is None:
        pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
        image_names = [f.replace('.png', '') for f in pred_files]
    
    print(f"Evaluating {len(image_names)} images...")
    
    for image_name in tqdm(image_names, desc="Calculating metrics"):
        pred_path = os.path.join(pred_dir, f"{image_name}.png")
        gt_path = os.path.join(gt_dir, f"{image_name}.png")
        
        # Check if files exist
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            print(f"Skipping {image_name}: files not found")
            continue
        
        # Load masks
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if pred is None or gt is None:
            print(f"Skipping {image_name}: failed to load images")
            continue
        
        # Resize if needed
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Calculate metrics
        metrics = calculator.calculate_all_metrics(pred, gt, image_name)
        aggregator.add_metrics(metrics)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation_metrics.json")
    aggregator.save_report(output_path)
    aggregator.print_summary(metric_type="single")
    
    return aggregator


def compare_refinement_methods(baseline_dir: str, refined_dir: str, gt_dir: str, 
                              output_dir: str, method_name: str,
                              image_names: Optional[List[str]] = None) -> MetricsAggregator:
    """
    Compare baseline and refined masks against ground truth.
    
    Args:
        baseline_dir: Directory containing baseline masks
        refined_dir: Directory containing refined masks
        gt_dir: Directory containing ground truth masks
        output_dir: Directory to save comparison results
        method_name: Name of the refinement method
        image_names: Optional list of specific images to compare
        
    Returns:
        MetricsAggregator with all comparison metrics
    """
    calculator = MaskMetricsCalculator()
    aggregator = MetricsAggregator()
    
    # Get list of images to compare
    if image_names is None:
        refined_files = sorted([f for f in os.listdir(refined_dir) if f.endswith('.png')])
        image_names = [f.replace('.png', '') for f in refined_files]
    
    print(f"Comparing {len(image_names)} images for method: {method_name}")
    
    for image_name in tqdm(image_names, desc="Comparing masks"):
        baseline_path = os.path.join(baseline_dir, f"{image_name}.png")
        refined_path = os.path.join(refined_dir, f"{image_name}.png")
        gt_path = os.path.join(gt_dir, f"{image_name}.png")
        
        # Check if files exist
        if not all(os.path.exists(p) for p in [baseline_path, refined_path, gt_path]):
            print(f"Skipping {image_name}: files not found")
            continue
        
        # Load masks
        baseline = cv2.imread(baseline_path, cv2.IMREAD_GRAYSCALE)
        refined = cv2.imread(refined_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if any(img is None for img in [baseline, refined, gt]):
            print(f"Skipping {image_name}: failed to load images")
            continue
        
        # Resize if needed
        if baseline.shape != gt.shape:
            baseline = cv2.resize(baseline, (gt.shape[1], gt.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        if refined.shape != gt.shape:
            refined = cv2.resize(refined, (gt.shape[1], gt.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        
        # Calculate comparison metrics
        metrics = calculator.compare_masks(baseline, refined, gt, image_name)
        aggregator.add_metrics(metrics)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"comparison_metrics_{method_name}.json")
    aggregator.save_report(output_path)
    aggregator.print_summary(metric_type="comparison")
    
    return aggregator


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate mask refinement results')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'compare'], 
                       default='compare', help='Evaluation mode')
    parser.add_argument('--pred-dir', type=str, help='Directory with predicted masks')
    parser.add_argument('--baseline-dir', type=str, help='Directory with baseline masks')
    parser.add_argument('--refined-dir', type=str, help='Directory with refined masks')
    parser.add_argument('--gt-dir', type=str, help='Directory with ground truth masks')
    parser.add_argument('--output-dir', type=str, default='results/metrics',
                       help='Output directory for metrics')
    parser.add_argument('--method', type=str, default='unknown',
                       help='Name of the refinement method')
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        if not args.pred_dir or not args.gt_dir:
            parser.error("--pred-dir and --gt-dir are required for evaluate mode")
        evaluate_masks(args.pred_dir, args.gt_dir, args.output_dir)
    
    elif args.mode == 'compare':
        if not all([args.baseline_dir, args.refined_dir, args.gt_dir]):
            parser.error("--baseline-dir, --refined-dir, and --gt-dir are required for compare mode")
        compare_refinement_methods(args.baseline_dir, args.refined_dir, args.gt_dir,
                                 args.output_dir, args.method)