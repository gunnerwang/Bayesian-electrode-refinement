#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern Refiner
Enhanced LLM-Bayesian integration with advanced electrode protection

Building on the prior-integration refiner, this version adds:
1. More sophisticated electrode type detection
2. Advanced protection strategies for different electrode types
3. Improved separation algorithms
4. Better handling of complex electrode arrangements
"""

import numpy as np
import cv2
from typing import Dict, Optional, Any, List, Tuple
import os
from tqdm import tqdm
import json
import warnings
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter1d, gaussian_filter
warnings.filterwarnings('ignore')

# Import the prior-integration refiner to inherit from it
import sys
sys.path.append(os.path.dirname(__file__))
from prior_integration import (
    PriorIntegrationRefiner, 
    LLMBayesianPriorGenerator
)
from lattice_refiner import ElectrodeLatticeModel


class CurvaturePreservingLatticeModel(ElectrodeLatticeModel):
    """Lattice model that preserves natural curvature of electrodes."""
    
    def __init__(self, **kwargs):
        # Override vertical continuity to minimal value
        kwargs['vertical_continuity_weight'] = 0.0  # No forced straightening
        kwargs['edge_smoothing_iterations'] = 1  # Minimal smoothing
        kwargs['use_shape_regularization'] = False  # Disable shape regularization
        super().__init__(**kwargs)
    
    def _apply_vertical_continuity(self, mask: np.ndarray, electrode_idx: int) -> np.ndarray:
        """Override to skip vertical continuity enforcement."""
        return mask  # Return mask unchanged


class EnhancedLLMBayesianPriorGenerator(LLMBayesianPriorGenerator):
    """Enhanced LLM prior generator with curvature awareness."""
    
    def _get_bayesian_prior_prompt(self) -> str:
        """Enhanced prompt for Bayesian prior generation with curvature awareness."""
        return """You are an expert in Bayesian image analysis for electrode refinement with advanced shape understanding.
Analyze the unified region containing multiple electrode masks to generate probability priors for Bayesian inference.

The image shows:
- Row 1: Instance masks (red borders indicate boundaries, green dots show centroids)
- Row 2: Ground truth comparison if available
- Row 3: Gray image region around each instance

For each electrode, analyze:
1. SHAPE TYPE:
   - Is the electrode straight, single-bend, or multi-bend?
   - Are there natural curvatures that should be preserved?
   - Look for smooth curves vs sharp angles

2. ELECTRODE TYPE (based on position pattern):
   - Long electrodes (typically alternating positions)
   - Short electrodes (typically alternating with long)
   - Consider the one-skip-one pairing pattern

3. REFINEMENT STRATEGY:
   - For curved electrodes: Use conservative refinement to preserve natural shape
   - For straight electrodes: Standard refinement is acceptable
   - Long electrodes should be fully preserved (protection_level: 1.0)

4. BAYESIAN PARAMETERS:
   - boundary_confidence: How certain are the edges? (0.0-1.0)
   - smoothness_prior: Should boundaries be smooth? (0.0-1.0)
   - protection_level: How much to preserve original shape? (0.0-1.0)
   - curvature_preservation: Should curvatures be maintained? (true/false)

Return a JSON object mapping instance IDs to their Bayesian priors:
{
    "instance_id": {
        "boundary_confidence": 0.0-1.0,
        "smoothness_prior": 0.0-1.0,
        "protection_level": 0.0-1.0,
        "shape_type": "straight|single_bend|multi_bend",
        "electrode_type": "long|short",
        "curvature_preservation": true|false,
        "notes": "Brief explanation"
    }
}"""


class PatternRefiner(PriorIntegrationRefiner):
    """Pattern refiner: extends the prior-integration refiner with advanced electrode protection strategies."""
    
    def __init__(self, **kwargs):
        """Initialize the pattern refiner by extending the prior-integration refiner."""
        # Initialize the prior-integration refiner with all parameters
        super().__init__(**kwargs)
        
        # Override with enhanced LLM generator if LLM is enabled
        if self.enable_llm_priors:
            self.llm_generator = EnhancedLLMBayesianPriorGenerator(
                model=kwargs.get('model'),
                debug_mode=self.debug_mode
            )
        
        # the pattern refiner specific initialization
        print("\n[Pattern] Additional enhancements:")
        print("  - Advanced electrode type detection")
        print("  - Improved separation algorithms")
        print("  - Better handling of complex arrangements")
        print("  - Curvature-aware processing")
        
        # Use curvature-preserving lattice model
        self.lattice_model = CurvaturePreservingLatticeModel()
    
    # the pattern refiner specific methods for curvature-aware processing
    def _analyze_electrode_curvature(self, mask: np.ndarray) -> Dict[str, Any]:
        """Analyze electrode curvature using skeleton analysis."""
        mask_binary = (mask > 127).astype(np.uint8)
        
        # Extract skeleton
        skeleton = skeletonize(mask_binary.astype(bool))
        
        # Get skeleton points
        skel_points = np.argwhere(skeleton)
        
        if len(skel_points) < 5:
            return {
                'is_curved': False,
                'curvature_points': [],
                'max_curvature': 0.0,
                'shape_type': 'straight'
            }
        
        # Sort skeleton points by y-coordinate (top to bottom)
        skel_points = skel_points[skel_points[:, 0].argsort()]
        
        # Calculate local curvatures using finite differences
        curvatures = []
        curvature_threshold = 0.05  # Lower threshold for more sensitive detection
        
        for i in range(2, len(skel_points) - 2):
            # Use 5-point window for curvature estimation
            p1 = skel_points[i-2]
            p2 = skel_points[i-1]
            p3 = skel_points[i]
            p4 = skel_points[i+1]
            p5 = skel_points[i+2]
            
            # Calculate angles
            angle1 = np.arctan2(p2[0] - p1[0], p2[1] - p1[1])
            angle2 = np.arctan2(p3[0] - p2[0], p3[1] - p2[1])
            angle3 = np.arctan2(p4[0] - p3[0], p4[1] - p3[1])
            angle4 = np.arctan2(p5[0] - p4[0], p5[1] - p4[1])
            
            # Calculate angle changes (curvature)
            curvature = abs(angle2 - angle1) + abs(angle3 - angle2) + abs(angle4 - angle3)
            curvatures.append(curvature)
        
        # Identify significant curvature points
        curvature_points = []
        if curvatures:
            curvatures = np.array(curvatures)
            # Smooth curvatures to reduce noise
            if len(curvatures) > 5:
                curvatures = gaussian_filter1d(curvatures, sigma=1.5)
            
            max_curvature = np.max(curvatures)
            mean_curvature = np.mean(curvatures)
            
            # Find peaks in curvature (more sensitive)
            for i, curv in enumerate(curvatures):
                if curv > mean_curvature + 0.3 * (max_curvature - mean_curvature):
                    curvature_points.append(skel_points[i+2])  # Adjust index
        
        # Determine shape type
        if not curvature_points or max(curvatures) < curvature_threshold:
            shape_type = 'straight'
            is_curved = False
        elif len(curvature_points) == 1:
            shape_type = 'single_bend'
            is_curved = True
        else:
            shape_type = 'multi_bend'
            is_curved = True
        
        return {
            'is_curved': is_curved,
            'curvature_points': curvature_points,
            'max_curvature': float(max_curvature) if len(curvatures) > 0 else 0.0,
            'shape_type': shape_type,
            'skeleton': skeleton
        }
    
    def _get_adaptive_kernel(self, mask: np.ndarray, electrode_info: Dict[str, Any], 
                           kernel_type: str = 'separation') -> np.ndarray:
        """Get adaptive kernel based on electrode curvature."""
        # Long electrodes always get minimal kernel for maximum protection
        if electrode_info['type'] == 'long':
            return np.array([[1, 1, 1]], dtype=np.uint8)  # Minimal horizontal kernel
        
        if 'curvature_info' not in electrode_info:
            # Fallback for short electrodes
            return np.ones((3, 3), dtype=np.uint8)
        
        curvature_info = electrode_info['curvature_info']
        
        if not curvature_info['is_curved']:
            # Straight short electrode - standard kernel
            return np.ones((3, 3), dtype=np.uint8)
        
        # Curved short electrode - use adaptive kernel
        if curvature_info['shape_type'] == 'single_bend':
            # For single bend, use cross-shaped kernel to preserve curvature
            kernel = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.uint8)
        else:
            # For multi-bend short electrodes, use slightly larger kernel
            kernel = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.uint8)
        
        return kernel
    
    def _restore_natural_curvature(self, refined_mask: np.ndarray, original_mask: np.ndarray, 
                                  curvature_info: Dict[str, Any]) -> np.ndarray:
        """Restore natural curvature to refined mask based on original shape."""
        if not curvature_info['is_curved']:
            return refined_mask
        
        # More aggressive approach: use original shape as base
        # and only apply refinement where needed
        
        # Find the difference between original and refined
        original_binary = (original_mask > 127).astype(np.uint8)
        refined_binary = (refined_mask > 127).astype(np.uint8)
        
        # Areas that were removed during refinement
        removed_areas = original_binary & (~refined_binary)
        
        # Areas that were added during refinement  
        added_areas = (~original_binary) & refined_binary
        
        # Start with original mask to preserve curvature
        result = original_mask.copy()
        
        # Remove only the areas that clearly need removal
        # (e.g., overlaps with other electrodes)
        # But keep most of the original shape
        
        # Create a mask for areas to definitely remove
        # This should be minimal to preserve curvature
        removal_mask = removed_areas.copy()
        
        # Erode the removal mask to be more conservative
        kernel = np.ones((3, 3), dtype=np.uint8)
        removal_mask = cv2.erode(removal_mask, kernel, iterations=2)
        
        # Apply removal
        result[removal_mask > 0] = 0
        
        # Add back any areas that were added during refinement
        # (these are usually important corrections)
        result[added_areas > 0] = 255
        
        # Smooth the boundaries slightly
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, 
                                 np.ones((3, 3), dtype=np.uint8))
        
        return result
        
    
    def _enforce_minimum_separation(self, masks: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Pattern-based separation using alternating electrode pairs."""
        print(f"\n[Pattern] _enforce_minimum_separation called with {len(masks)} masks")
        
        
        # Analyze electrode characteristics and positions
        electrode_info = {}
        
        for inst_id, mask in masks.items():
            y_coords, x_coords = np.where(mask > 127)
            if len(x_coords) > 0 and len(y_coords) > 0:
                width = np.max(x_coords) - np.min(x_coords) + 1
                height = np.max(y_coords) - np.min(y_coords) + 1
                aspect_ratio = height / max(width, 1)
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                
                # Analyze curvature
                curvature_info = self._analyze_electrode_curvature(mask)
                
                electrode_info[inst_id] = {
                    "aspect_ratio": aspect_ratio,
                    "width": width,
                    "height": height,
                    "center_x": center_x,
                    "center_y": center_y,
                    "area": len(x_coords),
                    "curvature_info": curvature_info
                }
            else:
                electrode_info[inst_id] = {
                    "aspect_ratio": 1.0,
                    "width": 1,
                    "height": 1,
                    "center_x": 0,
                    "center_y": 0,
                    "area": 0,
                    "curvature_info": {
                        'is_curved': False,
                        'curvature_points': [],
                        'max_curvature': 0.0,
                        'shape_type': 'straight'
                    }
                }
        
        # Sort electrodes by x position to identify pattern
        sorted_ids = sorted(electrode_info.keys(), 
                           key=lambda id: electrode_info[id]["center_x"])
        
        # Identify alternating pattern: one-skip-one pairing
        # Usually electrodes alternate between long and short
        for i, inst_id in enumerate(sorted_ids):
            # Use alternating pattern: even indices = one type, odd = another
            if i % 2 == 0:
                # Check aspect ratio to determine if this is the "long" group
                if i == 0 and len(sorted_ids) > 1:
                    # Compare first two electrodes to determine pattern
                    ar0 = electrode_info[sorted_ids[0]]["aspect_ratio"]
                    ar1 = electrode_info[sorted_ids[1]]["aspect_ratio"]
                    if ar0 > ar1 * 1.2:  # First is longer
                        electrode_info[inst_id]["type"] = "long"
                    else:
                        electrode_info[inst_id]["type"] = "short"
                else:
                    # Follow established pattern
                    electrode_info[inst_id]["type"] = electrode_info[sorted_ids[0]]["type"]
            else:
                # Opposite of even indices
                first_type = electrode_info[sorted_ids[0]]["type"]
                electrode_info[inst_id]["type"] = "short" if first_type == "long" else "long"
        
        if self.debug_mode:
            # Show pattern analysis
            print(f"[Pattern] Pattern-based classification with curvature analysis:")
            curved_count = 0
            for i in range(min(10, len(sorted_ids))):  # Show first 10
                id = sorted_ids[i]
                info = electrode_info[id]
                curv = info['curvature_info']
                print(f"  Position {i}: Instance {id} - type={info['type']}, AR={info['aspect_ratio']:.2f}, "
                      f"shape={curv['shape_type']}, max_curv={curv['max_curvature']:.3f}")
                if curv['is_curved']:
                    curved_count += 1
            
            type_counts = {}
            shape_counts = {}
            for info in electrode_info.values():
                t = info.get("type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
                s = info['curvature_info']['shape_type']
                shape_counts[s] = shape_counts.get(s, 0) + 1
            print(f"[Pattern] Type distribution: {type_counts}")
            print(f"[Pattern] Shape distribution: {shape_counts}")
        
        # Sort by x position only - process in spatial order for consistency
        sorted_items = sorted(masks.items(), 
                            key=lambda item: electrode_info[item[0]]["center_x"])
        
        refined = {}
        accumulated_forbidden = None
        
        # First pass: process all electrodes with minimal intervention
        for inst_id, mask in sorted_items:
            mask_binary = (mask > 127).astype(np.uint8)
            
            if accumulated_forbidden is not None:
                # Always remove direct overlaps
                direct_overlap = mask_binary & accumulated_forbidden
                if np.any(direct_overlap):
                    mask_binary = mask_binary & (~accumulated_forbidden)
                
                # Use adaptive kernel based on curvature
                kernel = self._get_adaptive_kernel(mask, electrode_info[inst_id], 'separation')
                forbidden_dilated = cv2.dilate(accumulated_forbidden.astype(np.uint8), kernel, iterations=1)
                
                # Apply separation more gently
                separation_mask = forbidden_dilated & mask_binary
                if np.sum(separation_mask) < np.sum(mask_binary) * 0.3:  # Only apply if removing < 30%
                    mask_binary = mask_binary & (~forbidden_dilated.astype(bool))
            
            # Update accumulated forbidden region
            if accumulated_forbidden is None:
                accumulated_forbidden = mask_binary > 0
            else:
                accumulated_forbidden = accumulated_forbidden | (mask_binary > 0)
            
            refined[inst_id] = mask_binary.astype(np.uint8) * 255
        
        # Second pass: check if any electrode lost too much area
        for inst_id, original_mask in masks.items():
            original_area = np.sum(original_mask > 127)
            refined_area = np.sum(refined[inst_id] > 127)
            
            if refined_area < original_area * 0.7:  # Lost more than 30%
                if self.debug_mode:
                    print(f"[Pattern] Instance {inst_id} lost too much area ({refined_area}/{original_area}), reducing separation")
                
                # Re-process with less aggressive separation
                mask_binary = (original_mask > 127).astype(np.uint8)
                
                # Only remove direct overlaps with other masks
                for other_id, other_mask in refined.items():
                    if other_id != inst_id:
                        mask_binary = mask_binary & (~(other_mask > 127))
                
                refined[inst_id] = mask_binary.astype(np.uint8) * 255
        
        # Third pass: restore natural curvature for curved electrodes
        if self.debug_mode:
            print(f"[Pattern] Applying curvature restoration...")
        
        for inst_id, mask in masks.items():
            if electrode_info[inst_id]['curvature_info']['is_curved']:
                # Apply curvature restoration
                refined[inst_id] = self._restore_natural_curvature(
                    refined[inst_id], 
                    mask,
                    electrode_info[inst_id]['curvature_info']
                )
                if self.debug_mode:
                    shape = electrode_info[inst_id]['curvature_info']['shape_type']
                    print(f"  Restored curvature for instance {inst_id} ({shape})")
        
        if self.debug_mode:
            print(f"\n[Pattern] Final separation summary:")
            for inst_id in electrode_info:
                electrode_type = electrode_info[inst_id]["type"]
                shape_type = electrode_info[inst_id]["curvature_info"]["shape_type"]
                original = np.sum(masks[inst_id] > 127)
                final = np.sum(refined[inst_id] > 127)
                print(f"  Instance {inst_id} ({electrode_type}/{shape_type}): {original} -> {final} "
                      f"({final/max(original,1)*100:.1f}% retained)")
        
        return refined


def main():
    """Test the pattern refiner with enhanced protection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pattern refiner: enhanced LLM-Bayesian refinement')
    parser.add_argument('--data-dir', type=str, 
                       default='data/Full_Instances',
                       help='Base directory containing instance data')
    parser.add_argument('--output-dir', type=str,
                       default='results/refined_masks_pattern',
                       help='Output directory')
    parser.add_argument('--test-image', type=str, default=None,
                       help='Specific image to test')
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--enable-llm-priors", action="store_true",
                       help="Enable the legacy multimodal LLM prior generator (diagnostic only; "
                            "off by default, sends region crops to the configured OpenAI endpoint)")
    parser.add_argument("--model", type=str, default=None,
                       help="OpenAI model to use (e.g., gpt-4o, gpt-4o-mini, gpt-4-turbo)")
    parser.add_argument("--thickness-factor", type=float, default=1.0,
                       help="Thickness control factor (0.1-2.0, lower=thinner)")
    parser.add_argument('--origin-dir', type=str,
                       default='data/Origin',
                       help='Directory containing original full images and masks')
    
    args = parser.parse_args()
    
    # Initialize the pattern refiner with enhanced parameters
    refiner = PatternRefiner(
        instances_per_region=50,  # the lattice refiner main uses 10
        pyramid_levels=1,  # the lattice refiner main uses 1
        enable_joint_refinement=True,
        enforce_separation=True,
        min_separation_pixels=5,  # the lattice refiner main uses 5
        thickness_factor=args.thickness_factor,  # Pass thickness factor
        enable_llm_priors=args.enable_llm_priors,
        model=args.model,  # Pass model parameter
        debug_mode=args.debug
    )
    
    # Process instances (exact copy from the lattice refiner main)
    os.makedirs(args.output_dir, exist_ok=True)
    
    instance_info_dir = os.path.join(args.data_dir, 'instance_info')
    enhanced_image_dir = os.path.join(args.origin_dir, 'images_enhanced')
    mask_dir = os.path.join(args.data_dir, 'repaired_masks')
    
    # Get instance info files
    info_files = sorted([f for f in os.listdir(instance_info_dir) if f.endswith('_info.json')])
    
    if args.test_image:
        info_files = [f for f in info_files if args.test_image in f]
    
    print(f"Processing {len(info_files)} images with the pattern refiner...")
    
    for info_file in tqdm(info_files):
        base_name = info_file.replace('_info.json', '')
        

        # Load instance info
        info_path = os.path.join(instance_info_dir, info_file)
        with open(info_path, 'r') as f:
            info_data = json.load(f)
        
        # Load full image
        full_image_path = os.path.join(enhanced_image_dir, f"{base_name}.png")
        if not os.path.exists(full_image_path):
            print(f"Image not found: {full_image_path}")
            continue
            
        full_image = cv2.imread(full_image_path)
        
        # Process instances in groups
        instances = info_data['instances']
        
        for i in range(0, len(instances), refiner.instances_per_region):
            group = instances[i:i+refiner.instances_per_region]
            
            # Load masks for this group
            instance_data = {}
            
            for inst in group:
                inst_id = inst['id']
                mask_path = os.path.join(mask_dir, f"{base_name}_instance_{inst_id}.png")
                
                if os.path.exists(mask_path):
                    instance_data[inst_id] = {
                        'bbox': inst['bbox'],
                        'mask_path': mask_path
                    }
            
            if not instance_data:
                continue
                
            print(f"Processing group with {len(instance_data)} instances...")
            
            # Create unified region
            region_data = refiner.create_unified_region(instance_data, full_image)
            
            # Refine
            refined_masks = refiner.refine_unified_region(region_data)
            
            # Save results  
            for inst_id, refined_mask in refined_masks.items():
                if inst_id not in region_data['instance_masks']:
                    continue
                
                rel_pos = region_data['instance_masks'][inst_id]['relative_pos']
                bbox = instance_data[inst_id]['bbox']
                
                # Extract instance portion (same as the lattice refiner)
                inst_refined = refined_mask[rel_pos[1]:rel_pos[1]+bbox[3], 
                                           rel_pos[0]:rel_pos[0]+bbox[2]]
                
                output_path = os.path.join(args.output_dir, f"{base_name}_instance_{inst_id}.png")
                cv2.imwrite(output_path, inst_refined)
    
    print("Pattern refiner run complete!")


if __name__ == '__main__':
    main()