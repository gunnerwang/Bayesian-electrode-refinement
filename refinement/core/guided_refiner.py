#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guided Refiner (morphology-based)
Direct morphology-based curvature analysis and geometric transformation

Simplified approach based on GT analysis:
1. Analyze electrode morphology to detect curvature patterns
2. Apply corresponding geometric transformations
3. Minimal refinement to preserve natural shapes
"""

import numpy as np
import cv2
from typing import Dict, Optional, Any, List
import os
from tqdm import tqdm
import json
import warnings
from skimage.morphology import thin
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')

# Import the pattern refiner to inherit from it
import sys
sys.path.append(os.path.dirname(__file__))
from pattern_refiner import PatternRefiner


class MorphologyCurvatureAnalyzer:
    """Morphology-based analyzer for curvature detection."""
    
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
    
    def analyze_electrode_shape(self, mask: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze electrode shape using morphology analysis."""
        binary = (mask > 127).astype(np.uint8)
        
        # Check if mask is valid
        if np.sum(binary) < 1000:
            return None
        
        # Get centerline
        centerline = thin(binary)
        points = np.argwhere(centerline)
        
        if len(points) < 50:
            return None
        
        # Sort by y (top to bottom in image coordinates)
        points = points[points[:, 0].argsort()]
        y_local = points[:, 0].astype(float)
        x_local = points[:, 1].astype(float)
        
        # Fit polynomial to capture shape
        coeffs = np.polyfit(y_local, x_local, 2)
        
        # Calculate curvature strength
        n_samples = 20
        sample_y = np.linspace(y_local[0], y_local[-1], n_samples)
        fitted_x = np.polyval(coeffs, sample_y)
        
        # Compare to straight line from bottom to top
        straight_x = np.linspace(x_local[-1], x_local[0], n_samples)
        straight_y = np.linspace(y_local[-1], y_local[0], n_samples)
        
        fitted_x_interp = np.interp(straight_y[::-1], sample_y, fitted_x)
        all_deviations = fitted_x_interp - straight_x
        
        # Find the maximum deviation in the top portion (where curvature is most pronounced)
        top_portion = int(len(all_deviations) * 0.3)  # Look at top 30%
        top_deviations = all_deviations[:top_portion]
        
        # Use the maximum deviation in the top portion for direction
        if len(top_deviations) > 0:
            max_idx = np.argmax(np.abs(top_deviations))
            max_deviation = top_deviations[max_idx]
        else:
            max_deviation = all_deviations[np.argmax(np.abs(all_deviations))]
        
        curvature_strength = abs(max_deviation)
        
        # Determine curvature type and parameters
        if abs(coeffs[0]) < 1e-6 or curvature_strength < 3:
            return None  # Too straight
        
        # Direction from bottom-up perspective based on quadratic coefficient
        # The quadratic coefficient tells us the curvature direction more reliably
        # When fitting y vs x (vertical position vs horizontal position):
        # Negative coefficient: ( shape - curves to the LEFT
        # Positive coefficient: ) shape - curves to the RIGHT
        if coeffs[0] < 0:  # Negative quadratic = ( shape = left curve
            direction = 'left'
        else:  # Positive quadratic = ) shape = right curve
            direction = 'right'
        
        # Always treat as head bend (top curvature)
        curve_type = 'head_bend'
        position = 'top_quarter'
        
        # Estimate angle based on curvature strength
        # Most GT electrodes have 10-20px deviation
        # Base angle calculation - will be modulated by transform_intensity
        angle = min(25, int(curvature_strength * 1.0))  # Base angle
        
        result = {
            'has_curvature': True,
            'type': curve_type,
            'direction': direction,
            'angle': angle,
            'position': position,
            'curvature_strength': curvature_strength,
            'quadratic_coeff': coeffs[0],
            'description': f"head bend {angle}° to the {direction}"
        }
        
        if self.debug_mode:
            print(f"  Morphology analysis: {result['description']}")
            print(f"    Quadratic coeff: {coeffs[0]:.2e}")
            print(f"    Curvature strength: {curvature_strength:.1f}px")
        
        return result


class GeometricTransformer:
    """Apply geometric transformations based on morphology analysis."""
    
    def __init__(self, debug_mode=False, transform_intensity=0.6):
        self.debug_mode = debug_mode
        self.transform_intensity = transform_intensity
    
    def apply_morphology_correction(self, mask: np.ndarray, curvature_info: Dict[str, Any], 
                                   electrode_type: str = None) -> np.ndarray:
        """Apply correction to match GT curvature - add curvature to straight electrodes."""
        if not curvature_info or not curvature_info.get('has_curvature', False):
            return mask
        
        mask_binary = (mask > 127).astype(np.uint8)
        
        # We want to add the GT curvature to the refined (straight) electrodes
        # This preserves the natural electrode shape from GT
        
        # Get parameters
        curve_type = curvature_info['type']
        direction = curvature_info['direction']
        angle = curvature_info['angle']
        position = curvature_info['position']
        
        # Adjust angle based on electrode type and transform intensity
        base_factor = self.transform_intensity
        if electrode_type == 'short':
            # Short electrodes get full intensity
            adjusted_angle = int(angle * base_factor * 1.0)
        else:
            # Long electrodes get reduced intensity
            adjusted_angle = int(angle * base_factor * 0.6)
        
        if self.debug_mode:
            print(f"  Applying head bend correction: {adjusted_angle}° {direction} (original: {angle}°, type: {electrode_type})")
        
        # Apply head correction
        result = self._apply_head_correction(mask_binary, direction, adjusted_angle, position)
        
        # Convert back to 0-255 range
        return (result * 255).astype(np.uint8)
    
    def _apply_head_correction(self, mask: np.ndarray, direction: str, angle: float, position: str) -> np.ndarray:
        """Apply smooth natural bend throughout the entire electrode."""
        h, w = mask.shape
        
        # Calculate maximum displacement based on angle
        angle_factor = self.transform_intensity  # Use transform intensity as angle factor
        effective_angle = angle * angle_factor
        
        # Maximum horizontal displacement - adaptive based on electrode width
        if w > 100:  # Wide electrode (likely short type)
            max_displacement = min(int(effective_angle * 3.0), int(w * 0.6))
        else:  # Narrow electrode (likely long type)
            max_displacement = min(int(effective_angle * 1.5), int(w * 0.4))
        
        if self.debug_mode:
            print(f"    Smooth bend: direction={direction}, angle={angle}°, effective={effective_angle:.1f}°")
            print(f"    Mask shape: {h}x{w}, max displacement: {max_displacement}px")
        
        # Create result mask with expanded width to prevent truncation
        pad_width = max_displacement + 5
        result_padded = np.zeros((h, w + 2 * pad_width), dtype=np.uint8)
        
        # Process each row with a single smooth curve
        for y in range(h):
            # Find pixels in this row
            row_pixels = np.where(mask[y] > 0)[0]
            if len(row_pixels) == 0:
                continue
            
            # Smooth continuous curve throughout the electrode
            progress = y / h  # 0 at top, 1 at bottom
            
            # Use exponential decay for a natural, continuous curve
            # This creates strong curvature at the top that gradually diminishes
            # exp(-k*x) where k controls how quickly it straightens
            # Adjust k based on transform intensity - lower intensity means higher k (more gradual)
            k = 3.0 + (1.0 - self.transform_intensity) * 3.0  # k ranges from 3.0 to 6.0
            bend_factor = np.exp(-k * progress)
            
            # Calculate horizontal offset
            offset = int(bend_factor * max_displacement)
            if direction == 'left':
                offset = -offset
            
            # Apply offset to all pixels in the row
            for x in row_pixels:
                new_x = x + offset + pad_width
                if 0 <= new_x < result_padded.shape[1]:
                    result_padded[y, new_x] = 1
        
        # Apply minimal morphological closing to ensure connectivity without removing thin parts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result_padded = cv2.morphologyEx(result_padded, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find the original electrode position
        orig_non_zero = np.where(mask > 0)
        if len(orig_non_zero[0]) > 0:
            orig_min_x = np.min(orig_non_zero[1])
            orig_max_x = np.max(orig_non_zero[1])
            orig_center_x = (orig_min_x + orig_max_x) // 2
        else:
            orig_center_x = w // 2
        
        # Find the actual bounds of the transformed electrode
        non_zero = np.where(result_padded > 0)
        if len(non_zero[0]) > 0:
            min_x = max(0, np.min(non_zero[1]) - 2)
            max_x = min(result_padded.shape[1], np.max(non_zero[1]) + 3)
            
            # Extract the electrode region
            electrode_region = result_padded[:, min_x:max_x]
            electrode_width = max_x - min_x
            
            # Create result maintaining original position
            result = np.zeros_like(mask)
            
            # Calculate offset to maintain original horizontal position
            # We want the center of the transformed electrode to match the original center
            transformed_center_offset = electrode_width // 2
            target_x_start = max(0, orig_center_x - transformed_center_offset)
            target_x_end = min(w, target_x_start + electrode_width)
            
            # Adjust if we're at the edge
            if target_x_end > w:
                target_x_start = w - electrode_width
                target_x_end = w
            if target_x_start < 0:
                target_x_start = 0
                target_x_end = electrode_width
            
            # Place the electrode at the calculated position
            actual_width = target_x_end - target_x_start
            if actual_width > 0:
                result[:, target_x_start:target_x_end] = electrode_region[:, :actual_width]
        else:
            # Fallback to original if transformation failed
            result = mask.copy()
        
        if self.debug_mode:
            diff = np.sum(result > 0) - np.sum(mask > 0)
            print(f"    Pixel count change: {diff} (original: {np.sum(mask > 0)}, new: {np.sum(result > 0)})")
        
        return result
    


class EdgeRefinementOptimizer:
    """Optimize electrode edges using active contour-like methods."""
    
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        
    def refine_edges(self, mask: np.ndarray, reference_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Refine edges using subpixel optimization and smoothing."""
        # Convert to binary
        binary = (mask > 127).astype(np.uint8)
        
        # Step 1: Extract initial contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return mask
            
        # Get the largest contour (main electrode)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Step 2: Smooth contour using active contour energy minimization
        smoothed_contour = self._smooth_contour(main_contour, binary)
        
        # Step 3: Apply subpixel refinement
        refined_contour = self._subpixel_refinement(smoothed_contour, mask)
        
        # Step 4: Create refined mask
        refined_mask = np.zeros_like(mask)
        cv2.drawContours(refined_mask, [refined_contour], -1, 255, -1)
        
        # Step 5: Edge enhancement
        refined_mask = self._enhance_edges(refined_mask, reference_mask)
        
        return refined_mask
    
    def _smooth_contour(self, contour: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """Smooth contour using energy minimization."""
        # Convert contour to point list
        points = contour.reshape(-1, 2).astype(np.float32)
        
        # Apply stronger Gaussian smoothing to reduce noise
        window_size = 7  # Larger window for better smoothing
        smoothed_points = []
        
        for i in range(len(points)):
            # Get neighboring points (circular)
            indices = [(i + j - window_size//2) % len(points) for j in range(window_size)]
            neighbors = points[indices]
            
            # Smoother weighted average
            weights = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
            smoothed_point = np.average(neighbors, axis=0, weights=weights)
            smoothed_points.append(smoothed_point)
        
        smoothed_points = np.array(smoothed_points)
        
        # Energy-based refinement
        # Balance between smoothness and staying close to original
        alpha = 0.5  # Elasticity (increased for smoother curves)
        beta = 0.3   # Stiffness (reduced to allow more flexibility)
        
        for iteration in range(8):  # More iterations for better convergence
            new_points = smoothed_points.copy()
            
            for i in range(len(smoothed_points)):
                # Get neighbors
                prev_idx = (i - 1) % len(smoothed_points)
                next_idx = (i + 1) % len(smoothed_points)
                
                # Elasticity force (minimize distance between consecutive points)
                elastic_force = alpha * (smoothed_points[prev_idx] + smoothed_points[next_idx] - 2 * smoothed_points[i])
                
                # Stiffness force (minimize curvature)
                if i > 0 and i < len(smoothed_points) - 1:
                    curvature_force = beta * (smoothed_points[prev_idx] - 2 * smoothed_points[i] + smoothed_points[next_idx])
                else:
                    curvature_force = 0
                
                # Update point
                new_points[i] += elastic_force + curvature_force
                
                # Ensure point stays within mask bounds
                new_points[i] = np.clip(new_points[i], [0, 0], 
                                       [binary_mask.shape[1]-1, binary_mask.shape[0]-1])
            
            smoothed_points = new_points
        
        return smoothed_points.astype(np.int32).reshape(-1, 1, 2)
    
    def _subpixel_refinement(self, contour: np.ndarray, original_mask: np.ndarray) -> np.ndarray:
        """Refine contour positions to subpixel accuracy."""
        # Convert to grayscale if needed
        if len(original_mask.shape) == 3:
            gray = cv2.cvtColor(original_mask, cv2.COLOR_BGR2GRAY)
        else:
            gray = original_mask.copy()
        
        # Apply Gaussian blur for smoother gradients
        blurred = gaussian_filter(gray.astype(np.float32), sigma=1.0)
        
        # Calculate gradients
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Refine each contour point
        refined_points = []
        points = contour.reshape(-1, 2)
        
        for point in points:
            x, y = point
            
            # Search in a small neighborhood for the strongest edge
            search_radius = 2
            best_pos = point.astype(np.float32)
            best_grad = grad_mag[y, x]
            
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grad_mag.shape[1] and 0 <= ny < grad_mag.shape[0]:
                        if grad_mag[ny, nx] > best_grad:
                            best_grad = grad_mag[ny, nx]
                            # Subpixel refinement using parabolic interpolation
                            if 1 <= nx < grad_mag.shape[1]-1 and 1 <= ny < grad_mag.shape[0]-1:
                                # Fit parabola in x direction
                                gx_m1 = grad_mag[ny, nx-1]
                                gx_0 = grad_mag[ny, nx]
                                gx_p1 = grad_mag[ny, nx+1]
                                dx_sub = 0.5 * (gx_m1 - gx_p1) / (gx_m1 - 2*gx_0 + gx_p1 + 1e-6)
                                
                                # Fit parabola in y direction
                                gy_m1 = grad_mag[ny-1, nx]
                                gy_0 = grad_mag[ny, nx]
                                gy_p1 = grad_mag[ny+1, nx]
                                dy_sub = 0.5 * (gy_m1 - gy_p1) / (gy_m1 - 2*gy_0 + gy_p1 + 1e-6)
                                
                                best_pos = np.array([nx + dx_sub, ny + dy_sub])
                            else:
                                best_pos = np.array([nx, ny], dtype=np.float32)
            
            refined_points.append(best_pos)
        
        return np.array(refined_points, dtype=np.int32).reshape(-1, 1, 2)
    
    def _enhance_edges(self, mask: np.ndarray, reference_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Enhance edges with smoothing while maintaining binary output."""
        # Convert to binary
        binary = (mask > 127).astype(np.uint8)
        
        # Apply morphological closing to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Apply stronger smoothing to reduce artifacts
        # First, dilate slightly to connect nearby edges
        dilated = cv2.dilate(closed, kernel, iterations=1)
        
        # Apply median filter to remove small spikes
        median = cv2.medianBlur(dilated, 5)
        
        # Erode back to original size
        eroded = cv2.erode(median, kernel, iterations=1)
        
        # Final smoothing with Gaussian blur
        blurred = cv2.GaussianBlur(eroded.astype(np.float32), (5, 5), 1.0)
        smooth_binary = (blurred > 0.5).astype(np.uint8)
        
        # Convert back to 255 scale
        return smooth_binary * 255


class GuidedRefiner(PatternRefiner):
    """Guided refiner (morphology-based): direct morphological curvature analysis and geometric transformation."""
    
    def __init__(self, **kwargs):
        """Initialize the morphology-based guided refiner."""
        # Extract the guided refiner specific parameters before passing to parent
        self.save_visualization = kwargs.pop('save_visualization', True)
        self.visualization_dir = kwargs.pop('visualization_dir', 'results/guided_analysis_visualizations')
        self.gt_full_dir = kwargs.pop('gt_full_dir', None)
        self.transform_ratio = kwargs.pop('transform_ratio', 0.2)  # Default 20% of electrodes
        self.transform_intensity = kwargs.pop('transform_intensity', 0.6)  # Default 60% intensity
        
        # Initialize the pattern refiner with remaining parameters
        super().__init__(**kwargs)
        
        # the guided refiner specific initialization
        print("\n[Guided] Morphology + edge refinement enabled:")
        print("  - Based on the pattern refiner's classification")
        print(f"  - Selective curvature: only top {int(self.transform_ratio*100)}% most curved")
        print(f"  - Transform intensity: {int(self.transform_intensity*100)}%")
        print("  - GT-guided curvature detection when available")
        print("  - Edge refinement with active contours")
        print("  - Smooth binary edges")
        
        # Initialize components
        self.morphology_analyzer = MorphologyCurvatureAnalyzer(debug_mode=self.debug_mode)
        self.transformer = GeometricTransformer(debug_mode=self.debug_mode, 
                                                transform_intensity=self.transform_intensity)
        self.edge_optimizer = EdgeRefinementOptimizer(debug_mode=self.debug_mode)
    
    def refine_unified_region(self, region_data: Dict[str, Any]) -> Dict[int, np.ndarray]:
        """Morphology step of the guided refiner: analyze shapes and apply transformations."""
        print("\n[Guided] Processing with morphological curvature analysis...")
        
        # Extract masks
        original_masks = {}
        for inst_id, inst_data in region_data['instance_masks'].items():
            original_masks[inst_id] = inst_data['mask']
        
        # Try to load GT masks
        gt_masks = self._load_gt_masks(region_data)
        self.gt_masks = gt_masks  # Store for edge refinement
        
        # Step 1: Full refinement using parent class method
        print("[Guided] Step 1: Executing full refinement pipeline...")
        # Delegate to the parent refine_unified_region
        # the parent performs pattern-based classification and refinement
        refined_masks = super().refine_unified_region(region_data)
        print(f"[Guided] Full refinement complete, got {len(refined_masks)} masks")
        
        # Step 2: Morphological curvature analysis
        print("[Guided] Step 2: Morphological curvature analysis...")
        if gt_masks:
            print(f"  Using {len(gt_masks)} GT masks for reference")
            curvature_info = self._analyze_curvatures_morphology(refined_masks, gt_masks)
        else:
            print("  Analyzing refined mask morphology")
            curvature_info = self._analyze_curvatures_morphology(refined_masks, refined_masks)
        
        # Step 3: Apply geometric transformations
        transformed_ids = []
        if curvature_info:
            print(f"[Guided] Step 3: Applying transformations to {len(curvature_info)} curved electrodes...")
            final_masks, transformed_ids = self._apply_morphology_transformations(refined_masks, curvature_info)
        else:
            print("[Guided] No significant curvatures detected")
            final_masks = refined_masks
        
        # Step 4: Apply edge refinement to all electrodes
        print(f"[Guided] Step 4: Applying edge refinement to all {len(final_masks)} electrodes...")
        edge_refined_masks = {}
        for inst_id, mask in final_masks.items():
            # Skip if already refined during transformation
            if inst_id in transformed_ids:
                edge_refined_masks[inst_id] = mask
            else:
                # Apply edge refinement to non-transformed electrodes
                reference_gt = gt_masks.get(inst_id) if gt_masks else None
                edge_refined_masks[inst_id] = self.edge_optimizer.refine_edges(mask, reference_gt)
        
        final_masks = edge_refined_masks
        
        # Save visualization if enabled - only show transformed electrodes
        if self.save_visualization and transformed_ids:
            # Show full pipeline only for transformed electrodes
            self._save_full_pipeline_visualization(original_masks, refined_masks, final_masks, 
                                                 gt_masks, curvature_info, region_data, transformed_ids)
        
        return final_masks
    
    def _load_gt_masks(self, region_data: Dict[str, Any]) -> Dict[int, np.ndarray]:
        """Load GT masks using connected components from full GT."""
        if not self.gt_full_dir:   # GT-free inference (default)
            return {}
        gt_masks = {}
        
        # Get image name from metadata
        image_name = None
        for inst_data in region_data['instance_masks'].values():
            if 'metadata' in inst_data and 'image_name' in inst_data['metadata']:
                image_name = inst_data['metadata']['image_name'].replace('.png', '')
                break
        
        if not image_name:
            return {}
        
        # Load full GT mask and use connected components
        full_gt_path = os.path.join(self.gt_full_dir, f"{image_name}.png")
        if os.path.exists(full_gt_path):
            full_gt = cv2.imread(full_gt_path, cv2.IMREAD_GRAYSCALE)
            if full_gt is not None:
                # Get connected components
                from skimage.measure import label, regionprops
                binary_gt = (full_gt > 127).astype(np.uint8)
                labeled_mask = label(binary_gt, connectivity=2)
                
                # For each instance, find the corresponding connected component
                for inst_id, inst_data in region_data['instance_masks'].items():
                    bbox = inst_data.get('bbox')
                    rel_pos = inst_data.get('relative_pos')
                    
                    if bbox and rel_pos:
                        # Get absolute position in original image
                        x1, y1, x2, y2 = region_data['bounds']
                        abs_x = x1 + rel_pos[0]
                        abs_y = y1 + rel_pos[1]
                        
                        # Find label at center of bbox
                        center_x = abs_x + bbox[2] // 2
                        center_y = abs_y + bbox[3] // 2
                        
                        if 0 <= center_y < labeled_mask.shape[0] and 0 <= center_x < labeled_mask.shape[1]:
                            label_at_center = labeled_mask[center_y, center_x]
                            
                            if label_at_center > 0:
                                # Extract this connected component in the bbox region
                                cc_mask = (labeled_mask == label_at_center).astype(np.uint8) * 255
                                gt_region = cc_mask[abs_y:abs_y+bbox[3], abs_x:abs_x+bbox[2]]
                                
                                if gt_region.size > 0 and np.sum(gt_region) > 1000:
                                    gt_masks[inst_id] = gt_region
                                    if self.debug_mode:
                                        print(f"  Loaded CC label {label_at_center} for instance {inst_id}")
        
        return gt_masks
    
    def _analyze_curvatures_morphology(self, masks: Dict[int, np.ndarray], 
                                     reference_masks: Dict[int, np.ndarray]) -> Dict[int, Dict]:
        """Analyze curvatures using morphological analysis."""
        curvature_info = {}
        
        for inst_id in masks:
            if inst_id in reference_masks:
                ref_mask = reference_masks[inst_id]
                
                if self.debug_mode:
                    print(f"\n[Guided] Analyzing instance {inst_id}")
                
                # Analyze the reference mask morphology
                analysis = self.morphology_analyzer.analyze_electrode_shape(ref_mask)
                
                if analysis:
                    curvature_info[inst_id] = analysis
        
        if self.debug_mode:
            print(f"\n[Guided] Found {len(curvature_info)} electrodes with curvature")
        
        return curvature_info
    
    def _classify_electrodes_by_pattern(self, masks: Dict[int, np.ndarray]) -> Dict[int, str]:
        """Classify electrodes as long/short based on alternating pattern (from the pattern refiner)."""
        electrode_info = {}
        
        # Analyze electrode characteristics
        for inst_id, mask in masks.items():
            y_coords, x_coords = np.where(mask > 127)
            if len(x_coords) > 0 and len(y_coords) > 0:
                width = np.max(x_coords) - np.min(x_coords) + 1
                height = np.max(y_coords) - np.min(y_coords) + 1
                aspect_ratio = height / max(width, 1)
                center_x = np.mean(x_coords)
                
                electrode_info[inst_id] = {
                    "aspect_ratio": aspect_ratio,
                    "center_x": center_x
                }
            else:
                electrode_info[inst_id] = {
                    "aspect_ratio": 1.0,
                    "center_x": 0
                }
        
        # Sort by x position
        sorted_ids = sorted(electrode_info.keys(), 
                           key=lambda id: electrode_info[id]["center_x"])
        
        # Classify based on alternating pattern
        types = {}
        for i, inst_id in enumerate(sorted_ids):
            if i % 2 == 0:
                # Even indices - determine type from first pair
                if i == 0 and len(sorted_ids) > 1:
                    ar0 = electrode_info[sorted_ids[0]]["aspect_ratio"]
                    ar1 = electrode_info[sorted_ids[1]]["aspect_ratio"]
                    if ar0 > ar1 * 1.2:  # First is longer
                        types[inst_id] = "long"
                    else:
                        types[inst_id] = "short"
                else:
                    # Follow established pattern
                    types[inst_id] = types[sorted_ids[0]]
            else:
                # Odd indices - opposite of even
                first_type = types[sorted_ids[0]]
                types[inst_id] = "short" if first_type == "long" else "long"
        
        return types
    
    def _apply_morphology_transformations(self, masks: Dict[int, np.ndarray], 
                                        curvature_info: Dict[int, Dict]) -> tuple[Dict[int, np.ndarray], List[int]]:
        """Apply geometric transformations based on morphological analysis."""
        if not curvature_info:
            return masks, []
        
        transformed = masks.copy()
        transformed_ids = []  # Track which electrodes were actually transformed
        
        # First classify electrodes as long/short based on alternating pattern
        electrode_types = self._classify_electrodes_by_pattern(masks)
        
        # Separate into long and short groups
        long_electrodes = [(id, info) for id, info in curvature_info.items() 
                          if electrode_types.get(id) == 'long']
        short_electrodes = [(id, info) for id, info in curvature_info.items() 
                           if electrode_types.get(id) == 'short']
        
        # Sort each group by curvature strength
        long_sorted = sorted(long_electrodes, key=lambda x: x[1]['curvature_strength'], reverse=True)
        short_sorted = sorted(short_electrodes, key=lambda x: x[1]['curvature_strength'], reverse=True)
        
        # Transform specified ratio of each group (minimum 2 per group if available)
        if long_sorted:
            num_long = max(2, int(len(long_sorted) * self.transform_ratio))
            num_long = min(num_long, len(long_sorted))
        else:
            num_long = 0
            
        if short_sorted:
            num_short = max(2, int(len(short_sorted) * self.transform_ratio))
            num_short = min(num_short, len(short_sorted))
        else:
            num_short = 0
        
        if self.debug_mode:
            print(f"\n[Guided] Electrode classification:")
            print(f"  Long electrodes with curvature: {len(long_sorted)}")
            print(f"  Short electrodes with curvature: {len(short_sorted)}")
            print(f"  Will transform: {num_long} long, {num_short} short (ratio: {self.transform_ratio:.1%})")
            if long_sorted:
                print(f"  Long curvature range: {long_sorted[0][1]['curvature_strength']:.1f} - "
                      f"{long_sorted[-1][1]['curvature_strength']:.1f}")
            if short_sorted:
                print(f"  Short curvature range: {short_sorted[0][1]['curvature_strength']:.1f} - "
                      f"{short_sorted[-1][1]['curvature_strength']:.1f}")
        
        # Apply transformation to top long electrodes
        for idx, (inst_id, curve_desc) in enumerate(long_sorted[:num_long]):
            if inst_id not in masks:
                continue
            
            if self.debug_mode:
                print(f"\n[Guided] Transforming LONG electrode {inst_id} (rank {idx+1}/{len(long_sorted)}): "
                      f"curvature strength {curve_desc['curvature_strength']:.1f}, "
                      f"{curve_desc['angle']}° {curve_desc['direction']}")
            
            # Apply transformation with variation
            modified_curve = curve_desc.copy()
            if idx % 3 == 1:
                modified_curve['angle'] = int(curve_desc['angle'] * 0.8)
            elif idx % 3 == 2:
                modified_curve['angle'] = int(curve_desc['angle'] * 0.9)
            
            original_mask = masks[inst_id]
            transformed_mask = self.transformer.apply_morphology_correction(
                original_mask, modified_curve, electrode_type='long'
            )
            
            # Apply edge refinement
            reference_gt = self.gt_masks.get(inst_id) if hasattr(self, 'gt_masks') else None
            refined_mask = self.edge_optimizer.refine_edges(transformed_mask, reference_gt)
            
            transformed[inst_id] = refined_mask
            transformed_ids.append(inst_id)
            
            if self.debug_mode:
                orig_pixels = np.sum(original_mask > 127)
                trans_pixels = np.sum(transformed_mask > 127)
                refined_pixels = np.sum(refined_mask > 127)
                print(f"      Original pixels: {orig_pixels}, Transformed: {trans_pixels}, Edge refined: {refined_pixels}")
        
        # Apply transformation to top short electrodes
        for idx, (inst_id, curve_desc) in enumerate(short_sorted[:num_short]):
            if inst_id not in masks:
                continue
            
            if self.debug_mode:
                print(f"\n[Guided] Transforming SHORT electrode {inst_id} (rank {idx+1}/{len(short_sorted)}): "
                      f"curvature strength {curve_desc['curvature_strength']:.1f}, "
                      f"{curve_desc['angle']}° {curve_desc['direction']}")
            
            # Apply transformation with variation
            modified_curve = curve_desc.copy()
            if idx % 3 == 1:
                modified_curve['angle'] = int(curve_desc['angle'] * 0.8)
            elif idx % 3 == 2:
                modified_curve['angle'] = int(curve_desc['angle'] * 0.9)
            
            original_mask = masks[inst_id]
            transformed_mask = self.transformer.apply_morphology_correction(
                original_mask, modified_curve, electrode_type='short'
            )
            
            # Apply edge refinement
            reference_gt = self.gt_masks.get(inst_id) if hasattr(self, 'gt_masks') else None
            refined_mask = self.edge_optimizer.refine_edges(transformed_mask, reference_gt)
            
            transformed[inst_id] = refined_mask
            transformed_ids.append(inst_id)
            
            if self.debug_mode:
                orig_pixels = np.sum(original_mask > 127)
                trans_pixels = np.sum(transformed_mask > 127)
                refined_pixels = np.sum(refined_mask > 127)
                print(f"      Original pixels: {orig_pixels}, Transformed: {trans_pixels}, Edge refined: {refined_pixels}")
        
        # Ensure no new overlaps after transformation
        transformed = self._fix_overlaps_after_transformation(transformed)
        
        return transformed, transformed_ids
    
    def _fix_overlaps_after_transformation(self, masks: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Fix any overlaps created by transformations."""
        # For now, skip overlap fixing as it's causing issues with different mask sizes
        # and may be erasing entire electrodes
        if self.debug_mode:
            print("[Guided] Skipping overlap fixing to preserve electrode integrity")
        
        # Simply ensure all masks are in proper 0-255 range
        fixed = {}
        for inst_id, mask in masks.items():
            # Ensure mask is in 0-255 range
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8) * 255
            elif np.max(mask) <= 1:
                mask = mask * 255
            fixed[inst_id] = mask
        
        return fixed
    
    def _save_full_pipeline_visualization(self, original_masks: Dict[int, np.ndarray],
                                        lattice_refined: Dict[int, np.ndarray],
                                        final_masks: Dict[int, np.ndarray], 
                                        gt_masks: Dict[int, np.ndarray],
                                        curvature_info: Dict[int, Dict],
                                        region_data: Dict[str, Any],
                                        transformed_ids: List[int]) -> None:
        """Save full-pipeline visualization (only electrodes actually transformed)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Create output directory
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Get image name
        image_name = None
        for inst_data in region_data['instance_masks'].values():
            if 'metadata' in inst_data and 'image_name' in inst_data['metadata']:
                image_name = inst_data['metadata']['image_name']
                break
        
        if not image_name:
            image_name = 'unknown'
        
        # Only visualize transformed electrodes
        transformed_info = {inst_id: curvature_info[inst_id] 
                          for inst_id in transformed_ids 
                          if inst_id in curvature_info}
        
        if not transformed_info:
            print("[Guided] No transformed electrodes to visualize")
            return
        
        # Create figure with 4 columns: Original, Lattice Refined, Curvature Adjusted, GT
        n_show = min(len(transformed_info), 8)
        fig, axes = plt.subplots(n_show, 4, figsize=(16, 4*n_show))
        
        if n_show == 1:
            axes = axes.reshape(1, -1)
        
        # Sort transformed electrodes by curvature strength
        sorted_transformed = sorted(transformed_info.items(), 
                                  key=lambda x: x[1]['curvature_strength'], 
                                  reverse=True)
        
        for idx, (inst_id, curve_desc) in enumerate(sorted_transformed[:n_show]):
            if inst_id not in original_masks:
                continue
            
            # Get masks
            original = original_masks[inst_id]
            lattice_ref = lattice_refined.get(inst_id, original)
            final = final_masks.get(inst_id, lattice_ref)
            gt = gt_masks.get(inst_id, original)
            
            # Original
            axes[idx, 0].imshow(original, cmap='gray')
            axes[idx, 0].set_title('Original', fontsize=10)
            axes[idx, 0].axis('off')
            
            # Lattice refined
            axes[idx, 1].imshow(lattice_ref, cmap='gray')
            axes[idx, 1].set_title('Lattice Refined', fontsize=10)
            axes[idx, 1].axis('off')
            
            # Curvature Adjusted
            axes[idx, 2].imshow(final, cmap='gray')
            axes[idx, 2].set_title('Curvature Adjusted', fontsize=10)
            axes[idx, 2].axis('off')
            
            # GT
            axes[idx, 3].imshow(gt, cmap='gray')
            axes[idx, 3].set_title('Ground Truth', fontsize=10)
            axes[idx, 3].axis('off')
            
            # Add curvature info as row title
            row_title = f"ID {inst_id}: {curve_desc['description']}, Strength: {curve_desc['curvature_strength']:.1f}px"
            axes[idx, 0].text(-0.1, 0.5, row_title, transform=axes[idx, 0].transAxes,
                            rotation=90, va='center', ha='right', fontsize=11)
        
        # Add overall title
        fig.suptitle(f'Guided Refiner Full Pipeline - {image_name}\n'
                    f'Original → Lattice Refinement → Curvature Adjustment\n'
                    f'Showing {n_show} transformed electrodes (out of {len(transformed_ids)} total transformed)',
                    fontsize=14)
        
        plt.tight_layout()
        
        # Generate filename with batch info
        # Get batch info from first instance ID
        first_inst_id = min(transformed_ids)
        batch_info = f"batch_{first_inst_id//50}"  # Assuming 50 instances per batch
        
        # Save figure with batch name
        output_path = os.path.join(self.visualization_dir, 
                                 f'{image_name}_{batch_info}_transformed_only.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.debug_mode:
            print(f"[Guided] Saved full pipeline visualization to: {output_path}")
        
        # Save summary with pipeline info - only for transformed electrodes
        summary = {
            'image_name': image_name,
            'pipeline': 'guided_refiner_full_pipeline',
            'batch_info': batch_info,
            'steps': ['original', 'lattice_refinement', 'curvature_adjustment'],
            'total_electrodes': len(original_masks),
            'curved_electrodes_detected': len(curvature_info),
            'electrodes_transformed': len(transformed_ids),
            'transformed_ids': transformed_ids,
            'transformation_details': {}
        }
        
        # Only include details for transformed electrodes
        for inst_id in transformed_ids:
            if inst_id in curvature_info:
                curve_desc = curvature_info[inst_id]
                summary['transformation_details'][str(inst_id)] = {
                    'type': curve_desc['type'],
                    'direction': curve_desc['direction'],
                    'angle': curve_desc['angle'],
                    'strength': float(curve_desc['curvature_strength']),
                    'quadratic_coeff': float(curve_desc['quadratic_coeff'])
                }
        
        json_path = os.path.join(self.visualization_dir, 
                               f'{image_name}_{batch_info}_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    """Test the morphology-based guided refiner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Guided refiner: morphology-based curvature refinement')
    parser.add_argument('--data-dir', type=str, 
                       default='data/Full_Instances',
                       help='Base directory containing instance data')
    parser.add_argument('--output-dir', type=str,
                       default='results/refined_masks_guided',
                       help='Output directory')
    parser.add_argument('--test-image', type=str, default=None,
                       help='Specific image to test')
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--thickness-factor", type=float, default=0.5,
                       help="Thickness control factor (0.1-2.0, lower=thinner)")
    parser.add_argument("--min-separation", type=int, default=5,
                       help="Minimum enforced separation between electrodes in pixels")
    parser.add_argument('--visualization-dir', type=str,
                       default='results/guided_analysis_visualizations',
                       help='Directory for saving visualization results')
    # Remove gt-full-dir argument as it will be derived from origin-dir
    # Remove mask-dir argument as it will be derived from data-dir
    parser.add_argument('--transform-ratio', type=float, default=0.8,
                       help='Ratio of electrodes to apply curvature transformation (0.0-1.0)')
    parser.add_argument('--transform-intensity', type=float, default=1.0,
                       help='Intensity of curvature transformation (0.0-1.0, higher=more aggressive)')
    parser.add_argument('--origin-dir', type=str,
                       default='data/Origin',
                       help='Directory containing original full images and masks')
    
    parser.add_argument('--gt-full-dir', type=str, default=None,
                        help='Enable the GT-guided diagnostic regime by passing the GT mask dir; '
                             'default None = deployment-faithful GT-free inference')
    args = parser.parse_args()
    
    # Initialize the morphology-based guided refiner
    refiner = GuidedRefiner(
        instances_per_region=50,
        pyramid_levels=1,  # number of pyramid levels (1 = single scale)
        enable_joint_refinement=True,
        enforce_separation=True,
        min_separation_pixels=args.min_separation,
        thickness_factor=args.thickness_factor,
        debug_mode=args.debug,
        visualization_dir=args.visualization_dir,
        gt_full_dir=args.gt_full_dir,
        transform_ratio=args.transform_ratio,
        transform_intensity=args.transform_intensity
    )
    
    # Process instances
    os.makedirs(args.output_dir, exist_ok=True)
    
    instance_info_dir = os.path.join(args.data_dir, 'instance_info')
    enhanced_image_dir = os.path.join(args.origin_dir, 'images_enhanced')
    mask_dir = os.path.join(args.data_dir, 'repaired_masks')
    
    # Get instance info files
    info_files = sorted([f for f in os.listdir(instance_info_dir) if f.endswith('_info.json')])
    
    if args.test_image:
        info_files = [f for f in info_files if args.test_image in f]
    
    print(f"Processing {len(info_files)} images with the morphology-based guided refiner...")
    
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
                
            print(f"\nProcessing group with {len(instance_data)} instances...")
            
            # Create unified region with metadata
            region_data = refiner.create_unified_region(instance_data, full_image)
            
            # Add metadata for GT loading
            for inst_id in region_data['instance_masks']:
                region_data['instance_masks'][inst_id]['metadata'] = {
                    'image_name': base_name
                }
            
            # Refine using morphology analysis
            refined_masks = refiner.refine_unified_region(region_data)
            
            # Save results  
            for inst_id, refined_mask in refined_masks.items():
                if inst_id not in region_data['instance_masks']:
                    continue
                
                rel_pos = region_data['instance_masks'][inst_id]['relative_pos']
                bbox = instance_data[inst_id]['bbox']
                
                # Extract instance portion
                inst_refined = refined_mask[rel_pos[1]:rel_pos[1]+bbox[3], 
                                           rel_pos[0]:rel_pos[0]+bbox[2]]
                
                output_path = os.path.join(args.output_dir, f"{base_name}_instance_{inst_id}.png")
                cv2.imwrite(output_path, inst_refined)
    
    print("\nGuided refiner (morphology-based) run complete!")


if __name__ == '__main__':
    main()
