#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assemble refined instance masks back into full images according to instance_info JSON files.

This script takes the refined individual instance masks and reconstructs the full
segmentation mask using the bounding box information from the instance_info JSON files.

Direct mode (--refined-dir) is the supported path and works with any refiner's
output directory; a legacy method table is kept for the default refiner output
locations.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


def load_instance_info(json_path: str) -> Dict:
    """Load instance information from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_instance_mask(mask_path: str) -> np.ndarray:
    """Load an individual instance mask."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")
    return mask


def assemble_full_mask(instance_info: Dict, instance_masks: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Assemble individual instance masks into a full image.
    
    Args:
        instance_info: Dictionary containing image dimensions and instance information
        instance_masks: Dictionary mapping instance IDs to their masks
        
    Returns:
        Full assembled mask
    """
    # Create empty canvas
    # Handle both instance_info key conventions (image_height/image_width and height/width)
    height = instance_info.get('image_height', instance_info.get('height'))
    width = instance_info.get('image_width', instance_info.get('width'))
    full_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Place each instance mask in its correct position
    for instance in instance_info['instances']:
        instance_id = instance['id']
        
        if instance_id not in instance_masks:
            print(f"Warning: Instance {instance_id} mask not found")
            continue
            
        # Get bounding box
        x, y, w, h = instance['bbox']
        
        # Get the instance mask
        mask = instance_masks[instance_id]
        
        # Verify mask dimensions match bbox
        if mask.shape != (h, w):
            print(f"Warning: Instance {instance_id} mask shape {mask.shape} doesn't match bbox ({h}, {w})")
            # Resize if needed
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Ensure bbox is within image bounds
        x_end = min(x + w, width)
        y_end = min(y + h, height)
        
        # Handle the placement with correct dimensions
        mask_h, mask_w = mask.shape
        place_h = min(mask_h, y_end - y)
        place_w = min(mask_w, x_end - x)
        
        # Place mask in full image
        # Handle overlapping instances by taking maximum value
        mask_region = full_mask[y:y+place_h, x:x+place_w]
        full_mask[y:y+place_h, x:x+place_w] = np.maximum(mask_region, mask[:place_h, :place_w])
    
    return full_mask


def process_image_group(base_name: str, instance_info_path: str, 
                       refined_masks_dir: str, output_dir: str):
    """
    Process all instances for a single image group.
    
    Args:
        base_name: Base name of the image (without _instance_X suffix)
        instance_info_path: Path to the instance info JSON
        refined_masks_dir: Directory containing refined instance masks
        output_dir: Directory to save assembled masks
    """
    # Load instance info
    instance_info = load_instance_info(instance_info_path)
    
    # Load all refined masks for this image
    instance_masks = {}
    missing_instances = []
    
    for instance in instance_info['instances']:
        instance_id = instance['id']
        mask_filename = f"{base_name}_instance_{instance_id}.png"
        mask_path = os.path.join(refined_masks_dir, mask_filename)
        
        if os.path.exists(mask_path):
            try:
                instance_masks[instance_id] = load_instance_mask(mask_path)
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
                missing_instances.append(instance_id)
        else:
            missing_instances.append(instance_id)
    
    if not instance_masks:
        print(f"No instance masks found for {base_name}")
        return
        
    if missing_instances:
        print(f"Warning: {base_name} missing instances: {missing_instances}")
    
    # Assemble full mask
    full_mask = assemble_full_mask(instance_info, instance_masks)
    
    # Save assembled mask
    output_path = os.path.join(output_dir, f"{base_name}.png")
    cv2.imwrite(output_path, full_mask)
    print(f"Saved assembled mask: {output_path}")


def assemble_all_refined_masks(data_dir: str, refined_masks_dir: str, output_dir: str, test_image: str = None) -> int:
    """
    Assemble all refined masks in the dataset.
    
    Args:
        data_dir: Base data directory containing instance_info subdirectory
        refined_masks_dir: Directory containing refined instance masks
        output_dir: Directory to save assembled masks
        test_image: If provided, only process images matching this name
        
    Returns:
        Number of successfully assembled masks
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all instance info JSON files
    instance_info_dir = os.path.join(data_dir, 'instance_info')
    
    if not os.path.exists(instance_info_dir):
        print(f"Instance info directory not found: {instance_info_dir}")
        return 0
    
    json_files = sorted([f for f in os.listdir(instance_info_dir) if f.endswith('_info.json')])
    
    # Filter by test_image if provided
    if test_image:
        json_files = [f for f in json_files if test_image in f]
        if not json_files:
            print(f"No instance info files found matching '{test_image}'")
            return 0
    
    print(f"Found {len(json_files)} instance info files{' (filtered)' if test_image else ''}")
    
    successful_assemblies = 0
    
    # Process each image group
    for json_file in tqdm(json_files, desc="Assembling masks"):
        # Extract base name (remove _info.json suffix)
        base_name = json_file[:-10]  # Remove '_info.json'
        
        instance_info_path = os.path.join(instance_info_dir, json_file)
        
        try:
            process_image_group(base_name, instance_info_path, refined_masks_dir, output_dir)
            successful_assemblies += 1
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            
    return successful_assemblies


def compute_metrics_and_visualize(method_name: str, data_dir: str, assembled_dir: str, 
                                 origin_dir: str, use_repaired: bool = True, 
                                 create_visualizations: bool = True,
                                 max_vis: int = 3, test_image: str = None) -> Dict[str, float]:
    """
    Compute metrics and optionally create visualizations.
    
    Args:
        method_name: Name of the refinement method
        data_dir: Base data directory
        assembled_dir: Directory containing assembled refined masks
        use_repaired: If True, use repaired masks. If False, use original coarse masks.
        create_visualizations: If True, create visualization images
        max_vis: Maximum number of visualizations to create
        test_image: If provided, only process images matching this name
        
    Returns:
        Dictionary with average metrics
    """
    # Setup directories
    if use_repaired:
        # For repaired masks, we need the full assembled coarse masks
        coarse_dir = os.path.join(origin_dir, 'coarse_masks')
    else:
        coarse_dir = os.path.join(origin_dir, 'coarse_masks')
    
    images_dir = os.path.join(origin_dir, 'images')
    gt_dir = os.path.join(origin_dir, 'masks')
    output_dir = f"results/assembly_visualizations_{method_name}"
    
    if create_visualizations:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get list of assembled masks
    assembled_files = sorted([f for f in os.listdir(assembled_dir) if f.endswith('.png')])
    
    # Filter by test_image if provided
    if test_image:
        assembled_files = [f for f in assembled_files if test_image in f]
        if not assembled_files:
            print(f"No assembled masks found matching '{test_image}' in {assembled_dir}")
            return {}
    
    if not assembled_files:
        print(f"No assembled masks found in {assembled_dir}")
        return {}
    
    print(f"Processing {len(assembled_files)} masks{' (filtered)' if test_image else ''}...")
    if create_visualizations:
        print(f"Creating visualizations for first {min(max_vis, len(assembled_files))} images...")
    
    # Track metrics for all images
    all_metrics = []
    total_iou_improvement = 0
    
    # Process all images for metrics
    for idx, assembled_file in enumerate(tqdm(assembled_files, desc="Computing metrics")):
        # Extract base name
        base_name = assembled_file.replace('.png', '')
        
        # Paths
        coarse_path = os.path.join(coarse_dir, f"{base_name}.png")
        assembled_path = os.path.join(assembled_dir, assembled_file)
        gt_path = os.path.join(gt_dir, f"{base_name}.png")
        
        # Check if files exist
        if not all(os.path.exists(p) for p in [coarse_path, assembled_path, gt_path]):
            continue
        
        # Load masks
        coarse_mask = cv2.imread(coarse_path, cv2.IMREAD_GRAYSCALE)
        assembled_mask = cv2.imread(assembled_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if any(img is None for img in [coarse_mask, assembled_mask, gt_mask]):
            continue
        
        # Resize masks to match if needed
        if gt_mask.shape != assembled_mask.shape:
            gt_mask = cv2.resize(gt_mask, (assembled_mask.shape[1], assembled_mask.shape[0]))
        if coarse_mask.shape != assembled_mask.shape:
            coarse_mask = cv2.resize(coarse_mask, (assembled_mask.shape[1], assembled_mask.shape[0]))
        
        # Calculate metrics
        def calculate_iou(mask1, mask2):
            intersection = np.logical_and(mask1 > 127, mask2 > 127).sum()
            union = np.logical_or(mask1 > 127, mask2 > 127).sum()
            return intersection / (union + 1e-6)
        
        coarse_gt_iou = calculate_iou(coarse_mask, gt_mask)
        refined_gt_iou = calculate_iou(assembled_mask, gt_mask)
        iou_improvement = refined_gt_iou - coarse_gt_iou
        consistency = calculate_iou(coarse_mask, assembled_mask)
        pixel_change = np.sum(coarse_mask != assembled_mask) / coarse_mask.size * 100
        
        # Store metrics
        all_metrics.append({
            'image': base_name,
            'coarse_gt_iou': coarse_gt_iou,
            'refined_gt_iou': refined_gt_iou,
            'iou_improvement': iou_improvement,
            'consistency': consistency,
            'pixel_change': pixel_change
        })
        
        total_iou_improvement += iou_improvement
        
        # Create visualization for first few images only
        if create_visualizations and idx < max_vis:
            create_detailed_visualization(
                base_name, method_name, 
                os.path.join(images_dir, f"{base_name}.png"),
                coarse_mask, assembled_mask, gt_mask,
                coarse_gt_iou, refined_gt_iou, iou_improvement,
                consistency, pixel_change, output_dir
            )
    
    # Calculate summary statistics
    if all_metrics:
        summary = {
            'num_images': len(all_metrics),
            'average_coarse_gt_iou': np.mean([m['coarse_gt_iou'] for m in all_metrics]),
            'average_refined_gt_iou': np.mean([m['refined_gt_iou'] for m in all_metrics]),
            'average_iou_improvement': np.mean([m['iou_improvement'] for m in all_metrics]),
            'average_consistency': np.mean([m['consistency'] for m in all_metrics]),
            'average_pixel_change': np.mean([m['pixel_change'] for m in all_metrics])
        }
        
        if create_visualizations:
            # Save detailed metrics - append to existing file if it exists
            metrics_file_path = os.path.join(output_dir, 'assembly_metrics_summary.json')
            
            # Load existing metrics if file exists
            existing_data = {'detailed_metrics': []}
            if os.path.exists(metrics_file_path):
                try:
                    with open(metrics_file_path, 'r') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    print(f"Warning: Could not load existing metrics from {metrics_file_path}")
                    existing_data = {'detailed_metrics': []}
            
            # Append new metrics
            if 'detailed_metrics' not in existing_data:
                existing_data['detailed_metrics'] = []
            
            # Add new metrics, avoiding duplicates
            existing_images = {m['image'] for m in existing_data.get('detailed_metrics', [])}
            new_metrics = [m for m in all_metrics if m['image'] not in existing_images]
            existing_data['detailed_metrics'].extend(new_metrics)
            
            # Update existing metrics for images that were reprocessed
            for new_metric in all_metrics:
                if new_metric['image'] in existing_images:
                    # Find and update the existing metric
                    for i, existing_metric in enumerate(existing_data['detailed_metrics']):
                        if existing_metric['image'] == new_metric['image']:
                            existing_data['detailed_metrics'][i] = new_metric
                            break
            
            # Recalculate summary statistics based on all metrics
            all_metrics_combined = existing_data['detailed_metrics']
            if all_metrics_combined:
                updated_summary = {
                    'num_images': len(all_metrics_combined),
                    'average_coarse_gt_iou': np.mean([m['coarse_gt_iou'] for m in all_metrics_combined]),
                    'average_refined_gt_iou': np.mean([m['refined_gt_iou'] for m in all_metrics_combined]),
                    'average_iou_improvement': np.mean([m['iou_improvement'] for m in all_metrics_combined]),
                    'average_consistency': np.mean([m['consistency'] for m in all_metrics_combined]),
                    'average_pixel_change': np.mean([m['pixel_change'] for m in all_metrics_combined])
                }
                
                # Update the summary in the data
                for key, value in updated_summary.items():
                    existing_data[key] = value
            
            # Save the updated metrics
            with open(metrics_file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
        
        print(f"\nSummary Statistics:")
        print(f"Processed images: {summary['num_images']}")
        print(f"Average Coarse→GT IoU: {summary['average_coarse_gt_iou']:.4f}")
        print(f"Average Refined→GT IoU: {summary['average_refined_gt_iou']:.4f}")
        print(f"Average IoU Improvement: {summary['average_iou_improvement']:.4f}")
        print(f"Average Consistency: {summary['average_consistency']:.4f}")
        print(f"Average Pixel Change: {summary['average_pixel_change']:.1f}%")
        
        return summary
    
    return {}


def create_detailed_visualization(base_name: str, method_name: str, image_path: str,
                                 coarse_mask: np.ndarray, assembled_mask: np.ndarray, 
                                 gt_mask: np.ndarray, coarse_gt_iou: float,
                                 refined_gt_iou: float, iou_improvement: float,
                                 consistency: float, pixel_change: float,
                                 output_dir: str):
    """Create detailed visualization comparing masks."""
    
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # Row 1: Full views
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(coarse_mask, cmap='gray')
    axes[0, 1].set_title('Original Coarse Mask', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(assembled_mask, cmap='gray')
    axes[0, 2].set_title(f'Assembled Refined Mask ({method_name})', fontsize=14)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(gt_mask, cmap='gray')
    axes[0, 3].set_title('Ground Truth Mask', fontsize=14)
    axes[0, 3].axis('off')
    
    # Row 2: Detailed comparisons
    # Create overlay
    overlay = np.zeros_like(image_rgb)
    overlay[coarse_mask > 127, 0] = 255  # Red for coarse
    overlay[assembled_mask > 127, 1] = 255  # Green for refined
    overlap = np.logical_and(coarse_mask > 127, assembled_mask > 127)
    overlay[overlap, :] = [255, 255, 0]  # Yellow for overlap
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Overlay (Red=Coarse, Green=Refined, Yellow=Overlap)', fontsize=14)
    axes[1, 0].axis('off')
    
    # Difference map
    diff = cv2.absdiff(coarse_mask, assembled_mask)
    axes[1, 1].imshow(diff, cmap='hot')
    axes[1, 1].set_title('Absolute Difference', fontsize=14)
    axes[1, 1].axis('off')
    
    # GT comparison overlay
    gt_overlay = np.zeros_like(image_rgb)
    gt_overlay[gt_mask > 127, 0] = 255  # Red for GT
    gt_overlay[assembled_mask > 127, 1] = 255  # Green for refined
    gt_overlap = np.logical_and(gt_mask > 127, assembled_mask > 127)
    gt_overlay[gt_overlap, :] = [255, 255, 0]  # Yellow for overlap
    
    axes[1, 2].imshow(gt_overlay)
    axes[1, 2].set_title('GT Comparison (Red=GT, Green=Refined)', fontsize=14)
    axes[1, 2].axis('off')
    
    # Metrics display
    axes[1, 3].text(0.1, 0.8, f'Metrics ({method_name}):', fontsize=16, weight='bold')
    axes[1, 3].text(0.1, 0.6, f'Coarse→GT IoU: {coarse_gt_iou:.3f}', fontsize=14)
    axes[1, 3].text(0.1, 0.5, f'Refined→GT IoU: {refined_gt_iou:.3f}', fontsize=14)
    axes[1, 3].text(0.1, 0.4, f'IoU Improvement: {iou_improvement:+.3f}', fontsize=14, 
                   color='green' if iou_improvement > 0 else 'red')
    axes[1, 3].text(0.1, 0.3, f'Consistency: {consistency:.3f}', fontsize=14)
    axes[1, 3].text(0.1, 0.2, f'Pixel Change: {pixel_change:.1f}%', fontsize=14)
    axes[1, 3].axis('off')
    axes[1, 3].set_xlim(0, 1)
    axes[1, 3].set_ylim(0, 1)
    
    # Add overall IoU improvement
    if iou_improvement > 0:
        improvement_pct = (iou_improvement / coarse_gt_iou) * 100 if coarse_gt_iou > 0 else 0
        fig.text(0.5, 0.01, f'IoU Improvement: {improvement_pct:.1f}%', 
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f"assembly_comparison_{base_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {output_path}")


def main(data_dir: str, origin_dir: str, use_repaired: bool = True, test_image: str = None):
    """Main function to run the assembly process.
    
    Args:
        use_repaired: If True, use repaired masks. If False, use original coarse masks.
        test_image: If provided, only process images matching this name
    """
    # Configuration is now passed as parameter
    
    # Default output directories of the released refiners. Direct mode
    # (--refined-dir) is the supported path; this legacy sweep only checks
    # these conventional locations and skips any that do not exist.
    refined_masks_dirs = {
        'base': "results/refined_masks_base",
        'lattice': "results/refined_masks_lattice",
        'prior': "results/refined_masks_prior",
        'pattern': "results/refined_masks_pattern",
        'guided': "results/refined_masks_guided",
        'llm_guided': "results/refined_masks_llm_guided",
        'shape_transfer': "results/refined_masks_shape_transfer",
        'assembly': "results/refined_masks_assembly",
    }
    
    mask_type = "repaired" if use_repaired else "original"
    print(f"\nUsing {mask_type} masks for visualization comparison")
    
    for method_name, refined_dir in refined_masks_dirs.items():
        print(f"\n{'='*60}")
        print(f"Processing {method_name} refined masks...")
        print(f"{'='*60}")
        
        # Check if refined masks directory exists
        if not os.path.exists(refined_dir):
            print(f"Refined masks directory not found: {refined_dir}")
            continue
        
        # Output directory for assembled masks
        output_dir = f"results/assembled_masks_{method_name}"
        
        # Assemble all masks
        successful = assemble_all_refined_masks(data_dir, refined_dir, output_dir, test_image)
        
        if successful > 0:
            # Compute metrics and create visualizations
            compute_metrics_and_visualize(
                method_name, data_dir, output_dir, origin_dir, use_repaired,
                create_visualizations=True,
                max_vis=3,  # Always create only 3 visualizations
                test_image=test_image
            )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Assemble refined instance masks')
    parser.add_argument('--use-original', action='store_true', default=False,
                        help='Use original coarse masks instead of repaired masks for visualization')
    parser.add_argument('--method', type=str, default=None,
                        help='Process only a specific method by name (e.g., assembly; '
                             'reads results/refined_masks_<method>)')
    parser.add_argument('--refined-dir', type=str, default=None,
                        help='Directory of refined instance masks (direct mode; overrides --method)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for assembled masks (direct mode)')
    parser.add_argument('--data-dir', type=str, default='data/Full_Instances',
                        help='Directory containing instance data')
    parser.add_argument('--origin-dir', type=str, default='data/Origin',
                        help='Directory containing original full images and masks')
    parser.add_argument('--skip-assembly', action='store_true', default=False,
                        help='Skip assembly step and only compute metrics (for refiners that assemble internally, e.g. the assembly refiner)')
    parser.add_argument('--test-image', type=str, default=None,
                        help='Process only specific image matching this name')
    
    args = parser.parse_args()
    
    # If use_original is True, then use_repaired should be False
    use_repaired = not args.use_original
    
    if args.refined_dir:
        # Direct mode: explicit input/output directories
        out = args.output_dir or (args.refined_dir.rstrip('/') + '_assembled')
        n = assemble_all_refined_masks(args.data_dir, args.refined_dir, out, args.test_image)
        print(f"Assembled {n} masks -> {out}")
        raise SystemExit(0)
    if args.method:
        # Process only specific method
        refined_dir = f"results/refined_masks_{args.method}"
        output_dir = f"results/assembled_masks_{args.method}"
        
        if args.skip_assembly:
            # Skip assembly and only compute metrics
            print(f"\nSkipping assembly step for {args.method}")
            if not os.path.exists(output_dir):
                print(f"Assembled masks directory not found: {output_dir}")
                print("Please ensure the method has already assembled the masks.")
            else:
                # Check if assembled masks exist
                assembled_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
                if assembled_files:
                    print(f"Found {len(assembled_files)} pre-assembled masks in {output_dir}")
                    compute_metrics_and_visualize(
                        args.method, args.data_dir, output_dir, args.origin_dir, use_repaired,
                        create_visualizations=True, max_vis=20, test_image=args.test_image
                    )
                else:
                    print(f"No assembled masks found in {output_dir}")
        else:
            # Normal flow: assemble then compute metrics
            if not os.path.exists(refined_dir):
                print(f"Refined masks directory not found: {refined_dir}")
            else:
                successful = assemble_all_refined_masks(args.data_dir, refined_dir, output_dir, args.test_image)
                if successful > 0:
                    compute_metrics_and_visualize(
                        args.method, args.data_dir, output_dir, args.origin_dir, use_repaired,
                        create_visualizations=True, max_vis=20, test_image=args.test_image
                    )
    else:
        main(args.data_dir, args.origin_dir, use_repaired, args.test_image)