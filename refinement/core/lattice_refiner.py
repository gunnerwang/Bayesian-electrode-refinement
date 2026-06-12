#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lattice Refiner
Addresses electrode adhesion issues with electrode lattice model

Key improvements:
1. Electrode lattice model with direct boundary enforcement
2. Efficient adhesion detection using overlap checking  
3. Fast separation enforcement without complex optimization
4. Integrated instance-to-strip mapping for better refinement
5. Fallback to the base refiner for small groups to ensure stability
"""

import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from typing import Dict, List, Optional, Tuple
import os
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Import the base refiner to inherit from it
import sys
sys.path.append(os.path.dirname(__file__))
from bayesian_base import BayesianRefinerBase


class ElectrodeLatticeModel:
    """
    Enhanced electrode model with adaptive separation enforcement.
    
    This model works in two steps:
    1. fit_from_instances: Analyzes masks to extract electrode positions and create ordered list
    2. generate_refined_masks: Uses the ordering to enforce separation with adaptive strategies
    
    Key features:
    - Adaptive boundary enforcement based on local context
    - Width consistency constraints
    - Soft boundary cutting with energy minimization
    - Local optimization for boundary placement
    """
    
    def __init__(self, 
                 min_spacing: float = 2.0,       # Minimum spacing between electrodes
                 max_width_ratio: float = 1.5,   # Maximum width relative to median
                 width_consistency_weight: float = 0.3,  # Weight for width consistency
                 boundary_smoothness_weight: float = 0.2,  # Weight for smooth boundaries
                 use_soft_boundaries: bool = True,  # Use soft boundary cutting
                 edge_smoothing_iterations: int = 2,  # Number of edge smoothing iterations
                 vertical_continuity_weight: float = 0.4,  # Weight for vertical continuity
                 use_shape_regularization: bool = True,  # Apply shape regularization
                 use_skeleton_guidance: bool = True):  # Use skeleton for shape preservation
        """Initialize enhanced lattice model."""
        self.min_spacing = min_spacing
        self.max_width_ratio = max_width_ratio
        self.width_consistency_weight = width_consistency_weight
        self.boundary_smoothness_weight = boundary_smoothness_weight
        self.use_soft_boundaries = use_soft_boundaries
        self.edge_smoothing_iterations = edge_smoothing_iterations
        self.vertical_continuity_weight = vertical_continuity_weight
        self.use_shape_regularization = use_shape_regularization
        self.use_skeleton_guidance = use_skeleton_guidance
        
        # Model state
        self.K = None
        self.electrodes = []  # Sorted list of electrode info (by x-coordinate)
        self.target_spacing = None
        self.median_width = None
        self.max_width = None
        self.instance_to_strip = {}  # Maps instance_id -> strip_index
        self.width_profile = None  # Expected width profile for each electrode
        self.electrode_skeletons = {}  # Store skeleton data for each electrode
        
    def fit_from_instances(self, instance_masks: Dict[int, np.ndarray], 
                          image: np.ndarray, edges: np.ndarray) -> Dict:
        """
        Simple fitting from instance masks without complex optimization.
        """
        # Extract electrode information
        self.electrodes = []
        for inst_id, mask in instance_masks.items():
            if np.sum(mask > 127) < 100:
                continue
            
            mask_binary = (mask > 127).astype(np.uint8)
            y_coords, x_coords = np.where(mask_binary)
            
            if len(x_coords) == 0:
                continue
                
            # Calculate electrode properties
            center_x = np.mean(x_coords)
            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            width = max_x - min_x + 1
            
            # Extract skeleton if enabled
            if self.use_skeleton_guidance:
                from skimage.morphology import skeletonize
                skeleton = skeletonize(mask_binary.astype(bool))
                
                # Compute distance transform for width estimation
                dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
                
                self.electrode_skeletons[inst_id] = {
                    'skeleton': skeleton,
                    'dist_transform': dist_transform
                }
            
            self.electrodes.append({
                'id': inst_id,
                'center_x': center_x,
                'min_x': min_x,
                'max_x': max_x,
                'width': width,
                'mask': mask
            })
        
        if not self.electrodes:
            return {'success': False}
        
        # Sort by x-coordinate
        self.electrodes.sort(key=lambda e: e['center_x'])
        self.K = len(self.electrodes)
        
        # Map instances to strips
        for idx, electrode in enumerate(self.electrodes):
            self.instance_to_strip[electrode['id']] = idx
        
        # Calculate spacing
        if self.K > 1:
            spacings = []
            for i in range(self.K - 1):
                spacing = self.electrodes[i+1]['center_x'] - self.electrodes[i]['center_x']
                spacings.append(spacing)
            self.target_spacing = np.median(spacings)
        else:
            self.target_spacing = 30.0
        
        # Calculate width statistics
        widths = [e['width'] for e in self.electrodes]
        self.median_width = np.median(widths)
        self.max_width = self.median_width * self.max_width_ratio
        
        # Calculate width profile for each electrode
        self._compute_width_profile()
        
        print(f"Fitted {self.K} electrodes, target spacing: {self.target_spacing:.1f}, median width: {self.median_width:.1f}")
        
        
        return {
            'success': True,
            'K': self.K,
            'target_spacing': self.target_spacing,
            'instance_to_strip': self.instance_to_strip
        }
    
    def _compute_width_profile(self):
        """Compute expected width profile for each electrode based on neighbors."""
        self.width_profile = np.zeros(self.K)
        
        for i in range(self.K):
            # Use local neighborhood to determine expected width
            neighbors = []
            if i > 0:
                neighbors.append(self.electrodes[i-1]['width'])
            neighbors.append(self.electrodes[i]['width'])
            if i < self.K - 1:
                neighbors.append(self.electrodes[i+1]['width'])
            
            # Weighted average with emphasis on current electrode
            if len(neighbors) > 1:
                weights = [0.2] * len(neighbors)
                weights[len(neighbors)//2] = 0.6  # Higher weight for current
                self.width_profile[i] = np.average(neighbors, weights=weights)
            else:
                self.width_profile[i] = self.electrodes[i]['width']
    
    def _compute_boundary_energy(self, boundary_x: float, left_mask: np.ndarray, 
                                right_mask: np.ndarray, image_gradient: np.ndarray) -> float:
        """Compute energy for placing boundary at given x position."""
        H = left_mask.shape[0]
        energy = 0.0
        
        # Data term: prefer boundaries at low gradient regions
        if image_gradient is not None:
            boundary_col = int(boundary_x)
            if 0 <= boundary_col < image_gradient.shape[1]:
                energy += np.sum(image_gradient[:, boundary_col])
        
        # Smoothness term: prefer straight boundaries
        boundary_col = int(boundary_x)
        
        # Width consistency term
        left_width = np.sum(left_mask[:, :boundary_col] > 0) / H
        right_width = np.sum(right_mask[:, boundary_col:] > 0) / H
        
        if hasattr(self, 'width_profile'):
            left_idx = self._find_electrode_index(left_mask)
            right_idx = self._find_electrode_index(right_mask)
            
            if left_idx >= 0:
                energy += self.width_consistency_weight * abs(left_width - self.width_profile[left_idx])
            if right_idx >= 0:
                energy += self.width_consistency_weight * abs(right_width - self.width_profile[right_idx])
        
        return energy
    
    def _find_electrode_index(self, mask: np.ndarray) -> int:
        """Find electrode index for given mask."""
        mask_center = np.mean(np.where(mask > 127)[1]) if np.any(mask > 127) else -1
        
        for idx, electrode in enumerate(self.electrodes):
            if abs(electrode['center_x'] - mask_center) < 5:  # 5 pixel tolerance
                return idx
        return -1
    
    def _find_optimal_boundary(self, left_electrode: Dict, right_electrode: Dict,
                              left_mask: np.ndarray, right_mask: np.ndarray,
                              image_gradient: Optional[np.ndarray] = None) -> int:
        """Find optimal boundary between two electrodes using skeleton guidance if available."""
        # Default boundary (midpoint)
        default_boundary = int((left_electrode['max_x'] + right_electrode['min_x']) / 2)
        
        # If skeleton guidance is enabled and available, use it to refine boundary
        if self.use_skeleton_guidance and left_electrode['id'] in self.electrode_skeletons:
            left_skeleton_data = self.electrode_skeletons[left_electrode['id']]
            right_skeleton_data = self.electrode_skeletons[right_electrode['id']]
            
            # Get distance transforms and skeletons
            left_dist = left_skeleton_data['dist_transform']
            right_dist = right_skeleton_data['dist_transform']
            left_skeleton = left_skeleton_data['skeleton']
            right_skeleton = right_skeleton_data['skeleton']
            
            # Get skeleton boundaries to ensure we don't cut through them
            left_skel_points = np.argwhere(left_skeleton)
            right_skel_points = np.argwhere(right_skeleton)
            
            if len(left_skel_points) > 0 and len(right_skel_points) > 0:
                left_skel_max_x = np.max(left_skel_points[:, 1])
                right_skel_min_x = np.min(right_skel_points[:, 1])
                
                # Ensure we don't cut through skeletons
                safe_min = left_skel_max_x + 3  # 3 pixel buffer from skeleton
                safe_max = right_skel_min_x - 3
                
                if safe_min < safe_max:
                    # Find optimal position within safe range
                    overlap_region = range(int(safe_min), int(safe_max + 1))
                    
                    min_separation_x = default_boundary
                    max_separation = 0
                    
                    # Find the x position with maximum separation between electrodes
                    for x in overlap_region:
                        if 0 <= x < left_mask.shape[1]:
                            # Count non-electrode pixels at this column
                            left_col = left_mask[:, x] if x < left_mask.shape[1] else np.zeros(left_mask.shape[0])
                            right_col = right_mask[:, x] if x < right_mask.shape[1] else np.zeros(right_mask.shape[0])
                            
                            # Calculate separation (fewer electrode pixels = better separation)
                            separation = np.sum(left_col == 0) + np.sum(right_col == 0)
                            
                            if separation > max_separation:
                                max_separation = separation
                                min_separation_x = x
                    
                    # Use the position with maximum separation
                    optimal_boundary = min_separation_x
                else:
                    # If no safe range, use midpoint but warn
                    optimal_boundary = default_boundary
                    print(f"Warning: No safe boundary between electrodes at x={default_boundary}")
            else:
                # Fallback to default if no skeleton data
                optimal_boundary = default_boundary
            
            # Final safety check
            optimal_boundary = max(int(left_electrode['max_x'] + self.min_spacing),
                                 min(int(right_electrode['min_x'] - self.min_spacing),
                                     optimal_boundary))
            
            return optimal_boundary
        
        if not self.use_soft_boundaries:
            # Hard boundary with minimum spacing
            return int(left_electrode['max_x'] + self.min_spacing)
        
        # Original soft boundary search
        search_start = int(left_electrode['max_x'] + self.min_spacing)
        search_end = int(right_electrode['min_x'] - self.min_spacing)
        
        if search_end <= search_start:
            return default_boundary
        
        # Evaluate energy at different positions
        best_boundary = default_boundary
        best_energy = float('inf')
        
        for boundary_x in range(search_start, search_end + 1, 2):  # Step by 2 for efficiency
            energy = self._compute_boundary_energy(boundary_x, left_mask, right_mask, image_gradient)
            
            if energy < best_energy:
                best_energy = energy
                best_boundary = boundary_x
        
        return best_boundary
    
    def _smooth_electrode_edges(self, mask: np.ndarray) -> np.ndarray:
        """Smooth electrode edges to reduce jaggedness."""
        # Apply morphological operations to smooth edges
        # Adjust kernel size based on thickness_factor in parent class
        thickness_factor = getattr(self, 'thickness_factor', 1.0) if hasattr(self, 'thickness_factor') else 1.0
        
        if thickness_factor < 0.3:
            # Extremely thin - aggressive thinning
            kernel_size = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
            # Multiple opening passes to thin electrodes
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
            mask = cv2.erode(mask, kernel, iterations=1)
        elif thickness_factor < 0.5:
            # Very thin - moderate thinning
            kernel_size = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        elif thickness_factor < 0.7:
            # Thin - light thinning
            kernel_size = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        elif thickness_factor < 0.9:
            # Normal - minimal smoothing
            kernel_size = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            # Slightly thick - light closing
            kernel_size = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Apply median filter for additional smoothing
        if self.edge_smoothing_iterations > 0:
            for _ in range(self.edge_smoothing_iterations):
                mask = cv2.medianBlur(mask, 3)
                
        return mask
    
    def _apply_vertical_continuity(self, mask: np.ndarray, electrode_idx: int) -> np.ndarray:
        """Apply vertical continuity constraints to ensure smooth vertical edges."""
        H, W = mask.shape
        
        # Extract left and right boundaries
        left_boundary = []
        right_boundary = []
        
        for y in range(H):
            row_pixels = np.where(mask[y, :] > 127)[0]
            if len(row_pixels) > 0:
                left_boundary.append(np.min(row_pixels))
                right_boundary.append(np.max(row_pixels))
            else:
                # Handle gaps
                if left_boundary:
                    left_boundary.append(left_boundary[-1])
                    right_boundary.append(right_boundary[-1])
                else:
                    left_boundary.append(W//2)
                    right_boundary.append(W//2)
        
        if not left_boundary:
            return mask
            
        # Smooth boundaries using moving average
        window_size = 5
        left_smooth = np.convolve(left_boundary, np.ones(window_size)/window_size, mode='same')
        right_smooth = np.convolve(right_boundary, np.ones(window_size)/window_size, mode='same')
        
        # Apply additional smoothing for very jagged edges
        if self.vertical_continuity_weight > 0.3:
            # Use Gaussian filter for stronger smoothing
            left_smooth = gaussian_filter1d(left_smooth, sigma=2.0)
            right_smooth = gaussian_filter1d(right_smooth, sigma=2.0)
        
        # Create refined mask with smooth boundaries
        refined_mask = np.zeros_like(mask)
        for y in range(H):
            left = int(left_smooth[y])
            right = int(right_smooth[y]) + 1
            
            # Ensure boundaries are within image
            left = max(0, min(W-1, left))
            right = max(left+1, min(W, right))
            
            # Blend with original mask
            if self.vertical_continuity_weight < 1.0:
                original_row = mask[y, :]
                smooth_row = np.zeros_like(original_row)
                smooth_row[left:right] = 255
                refined_mask[y, :] = (
                    (1 - self.vertical_continuity_weight) * original_row +
                    self.vertical_continuity_weight * smooth_row
                ).astype(np.uint8)
            else:
                refined_mask[y, left:right] = 255
                
        return refined_mask
    
    def _regularize_electrode_shape(self, mask: np.ndarray, expected_width: float) -> np.ndarray:
        """Regularize electrode shape to be more uniform."""
        H, W = mask.shape
        
        # Find electrode center column
        y_coords, x_coords = np.where(mask > 127)
        if len(x_coords) == 0:
            return mask
            
        center_x = np.median(x_coords)
        
        # Create regularized mask with consistent width
        regularized = np.zeros_like(mask)
        half_width = expected_width / 2
        
        for y in range(H):
            # Find current row's pixels
            row_pixels = np.where(mask[y, :] > 127)[0]
            if len(row_pixels) > 0:
                # Use local center for this row
                row_center = np.mean(row_pixels)
                
                # Blend between original center and row center
                blended_center = 0.7 * center_x + 0.3 * row_center
                
                # Set pixels with smooth width
                left = int(blended_center - half_width)
                right = int(blended_center + half_width)
                
                # Ensure within bounds
                left = max(0, left)
                right = min(W, right)
                
                regularized[y, left:right] = 255
                
        return regularized
    
    def generate_refined_masks(self, instance_masks: Dict[int, np.ndarray], 
                              image_gradient: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
        """Generate refined masks with adaptive separation guarantees."""
        refined = {}
        
        # Pre-compute boundaries between adjacent electrodes
        boundaries = {}
        for i in range(self.K - 1):
            left_id = self.electrodes[i]['id']
            right_id = self.electrodes[i + 1]['id']
            
            if left_id in instance_masks and right_id in instance_masks:
                left_mask = instance_masks[left_id]
                right_mask = instance_masks[right_id]
                
                # Find optimal boundary
                optimal_boundary = self._find_optimal_boundary(
                    self.electrodes[i], self.electrodes[i + 1],
                    left_mask, right_mask, image_gradient
                )
                boundaries[(i, i + 1)] = optimal_boundary
        
        # Process each electrode using the strip index mapping
        for inst_id, strip_idx in self.instance_to_strip.items():
            if inst_id not in instance_masks:
                continue
                
            orig_mask = instance_masks[inst_id]
            mask_binary = (orig_mask > 127).astype(np.uint8)
            
            # Get current electrode info
            electrode = self.electrodes[strip_idx]
            
            # Get actual mask bounds
            y_coords, x_coords = np.where(mask_binary > 0)
            if len(x_coords) == 0:
                continue  # Skip empty masks
                
            actual_min_x = np.min(x_coords)
            actual_max_x = np.max(x_coords)
            
            # Calculate boundaries using pre-computed optimal boundaries
            if strip_idx > 0 and (strip_idx - 1, strip_idx) in boundaries:
                min_x_allowed = boundaries[(strip_idx - 1, strip_idx)]
            elif strip_idx > 0:
                # Fallback to hard boundary
                prev_electrode = self.electrodes[strip_idx - 1]
                min_x_allowed = int(prev_electrode['max_x'] + self.min_spacing)
            else:
                min_x_allowed = 0
            
            if strip_idx < self.K - 1 and (strip_idx, strip_idx + 1) in boundaries:
                max_x_allowed = boundaries[(strip_idx, strip_idx + 1)]
            elif strip_idx < self.K - 1:
                # Fallback to hard boundary
                next_electrode = self.electrodes[strip_idx + 1]
                max_x_allowed = int(next_electrode['min_x'] - self.min_spacing)
            else:
                max_x_allowed = mask_binary.shape[1]
            
            # Ensure we don't completely eliminate the mask
            # Allow at least 50% of the original mask width
            mask_center = (actual_min_x + actual_max_x) / 2
            min_required_width = (actual_max_x - actual_min_x) * 0.5
            
            # Adjust boundaries if they're too restrictive
            if max_x_allowed - min_x_allowed < min_required_width:
                # Center the allowed region around the mask center
                half_width = min_required_width / 2
                min_x_allowed = max(0, int(mask_center - half_width))
                max_x_allowed = min(mask_binary.shape[1], int(mask_center + half_width))
            
            # Check if boundaries are reasonable
            if max_x_allowed <= min_x_allowed:
                print(f"WARNING: Invalid boundaries for electrode {strip_idx}: min={min_x_allowed}, max={max_x_allowed}")
                # Use electrode's original boundaries with some margin
                min_x_allowed = max(0, electrode['min_x'] - 5)
                max_x_allowed = min(mask_binary.shape[1], electrode['max_x'] + 5)
            
            # Apply boundaries with optional soft transition
            refined_mask = mask_binary.copy()
            
            # Check if boundaries would cut the electrode too much
            if inst_id in self.electrode_skeletons and self.use_skeleton_guidance:
                skeleton_data = self.electrode_skeletons[inst_id]
                skeleton = skeleton_data['skeleton']
                
                # Ensure skeleton connectivity within boundaries
                skel_points = np.argwhere(skeleton)
                if len(skel_points) > 0:
                    skel_x_coords = skel_points[:, 1]
                    skel_min_x = np.min(skel_x_coords)
                    skel_max_x = np.max(skel_x_coords)
                    
                    # Adjust boundaries to preserve skeleton
                    if min_x_allowed > skel_min_x + 2:  # Allow 2 pixel margin
                        min_x_allowed = max(0, skel_min_x - 2)
                    if max_x_allowed < skel_max_x - 2:
                        max_x_allowed = min(mask_binary.shape[1], skel_max_x + 2)
            
            if self.use_soft_boundaries:
                # Smooth transition at boundaries - ensure no overlap
                transition_width = 2
                
                # Left boundary
                if min_x_allowed > 0:
                    refined_mask[:, :min_x_allowed] = 0
                    # Add smooth transition only within the mask
                    for t in range(transition_width):
                        col = min_x_allowed + t
                        if col < refined_mask.shape[1] and col < max_x_allowed:
                            weight = (t + 1) / (transition_width + 1)
                            refined_mask[:, col] = (refined_mask[:, col] * weight).astype(np.uint8)
                
                # Right boundary  
                if max_x_allowed < mask_binary.shape[1]:
                    refined_mask[:, max_x_allowed:] = 0
                    # Add smooth transition only within the mask
                    for t in range(transition_width):
                        col = max_x_allowed - 1 - t
                        if col >= 0 and col >= min_x_allowed:
                            weight = (t + 1) / (transition_width + 1)
                            refined_mask[:, col] = (refined_mask[:, col] * weight).astype(np.uint8)
            else:
                # Hard boundaries
                if min_x_allowed > 0:
                    refined_mask[:, :min_x_allowed] = 0
                if max_x_allowed < mask_binary.shape[1]:
                    refined_mask[:, max_x_allowed:] = 0
            
            # Apply width constraints conservatively
            if hasattr(self, 'width_profile') and strip_idx < len(self.width_profile) and self.width_consistency_weight > 0:
                expected_width = self.width_profile[strip_idx]
                actual_width = electrode['width']
                
                if actual_width > expected_width * 1.2:  # Too wide
                    # Erode from edges
                    y_coords, x_coords = np.where(refined_mask > 0)
                    if len(x_coords) > 0:
                        center_x = np.mean(x_coords)
                        target_half_width = expected_width / 2
                        
                        for y in range(refined_mask.shape[0]):
                            row_pixels = np.where(refined_mask[y, :] > 0)[0]
                            if len(row_pixels) > 0:
                                row_center = np.mean(row_pixels)
                                refined_mask[y, :int(row_center - target_half_width)] = 0
                                refined_mask[y, int(row_center + target_half_width):] = 0
            
            # Check for connectivity before filling holes
            if inst_id in self.electrode_skeletons and self.use_skeleton_guidance:
                # Ensure skeleton pixels are preserved
                skeleton = self.electrode_skeletons[inst_id]['skeleton']
                skel_points = np.argwhere(skeleton)
                for y, x in skel_points:
                    if 0 <= x < refined_mask.shape[1] and 0 <= y < refined_mask.shape[0]:
                        refined_mask[y, x] = 255
            
            # Fill holes - binary_fill_holes expects boolean input
            refined_mask_bool = refined_mask > 0
            refined_mask = binary_fill_holes(refined_mask_bool).astype(np.uint8) * 255
            
            # Apply shape regularization if enabled (skip if mask would be too small)
            if self.use_shape_regularization and hasattr(self, 'width_profile'):
                if strip_idx < len(self.width_profile):
                    expected_width = self.width_profile[strip_idx]
                    # Only regularize if it won't make the mask too small
                    current_pixels = np.sum(refined_mask > 127)
                    if current_pixels > 100:  # Minimum pixel threshold
                        regularized = self._regularize_electrode_shape(refined_mask, expected_width)
                        # Check if regularization preserved enough pixels
                        if np.sum(regularized > 127) > current_pixels * 0.5:
                            refined_mask = regularized
            
            # Apply vertical continuity constraints
            if self.vertical_continuity_weight > 0:
                refined_mask = self._apply_vertical_continuity(refined_mask, strip_idx)
            
            # Apply edge smoothing
            if self.edge_smoothing_iterations > 0:
                refined_mask = self._smooth_electrode_edges(refined_mask)
            
            # Final morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Ensure mask is binary
            refined_mask = (refined_mask > 127).astype(np.uint8) * 255
            
            refined[inst_id] = refined_mask
        
        return refined
    


class LatticeRefiner(BayesianRefinerBase):
    """
    Lattice refiner with robust anti-adhesion constraints.
    Inherits directly from the base refiner for better stability.
    """
    
    def __init__(self,
                 # the base refiner parameters (passed to parent)
                 instances_per_region: int = 20,
                 pyramid_levels: int = 3,
                 scale_weights: Optional[List[float]] = None,
                 edge_methods: List[str] = ['sobel', 'canny', 'laplacian'],
                 edge_fusion_weights: Optional[Dict[str, float]] = None,
                 adaptive_window_size: int = 51,
                 threshold_sensitivity: float = 0.5,
                 structure_weight: float = 0.3,
                 consistency_weight: float = 0.3,
                 intensity_weight: float = 0.2,
                 smoothness_weight: float = 0.2,
                 min_electrode_width: int = 10,
                 straight_edge_threshold: float = 0.7,
                 boundary_quality_threshold: float = 0.8,
                 enable_straight_edge_enhancement: bool = True,
                 enable_adaptive_boundary_preservation: bool = True,
                 # Lattice-refiner-specific parameters
                 enable_joint_refinement: bool = True,
                 enforce_separation: bool = True,
                 min_separation_pixels: int = 2,
                 min_instances_for_joint: int = 3,
                 # Thickness control parameter
                 thickness_factor: float = 1.0):
        """Initialize the lattice refiner."""
        super().__init__(
            instances_per_region=instances_per_region,
            pyramid_levels=pyramid_levels,
            scale_weights=scale_weights,
            edge_methods=edge_methods,
            edge_fusion_weights=edge_fusion_weights,
            adaptive_window_size=adaptive_window_size,
            threshold_sensitivity=threshold_sensitivity,
            structure_weight=structure_weight,
            consistency_weight=consistency_weight,
            intensity_weight=intensity_weight,
            smoothness_weight=smoothness_weight,
            min_electrode_width=min_electrode_width,
            straight_edge_threshold=straight_edge_threshold,
            boundary_quality_threshold=boundary_quality_threshold,
            enable_straight_edge_enhancement=enable_straight_edge_enhancement,
            enable_adaptive_boundary_preservation=enable_adaptive_boundary_preservation
        )
        
        # Lattice-refiner-specific attributes
        self.enable_joint_refinement = enable_joint_refinement
        self.enforce_separation = enforce_separation
        self.min_separation_pixels = min_separation_pixels
        self.min_instances_for_joint = min_instances_for_joint
        self.thickness_factor = thickness_factor
        
        # Use enhanced lattice model with skeleton guidance
        self.lattice_model = ElectrodeLatticeModel(
            min_spacing=min_separation_pixels,
            max_width_ratio=1.5,
            width_consistency_weight=0.2,  # Reduced
            boundary_smoothness_weight=0.2,
            use_soft_boundaries=True,
            edge_smoothing_iterations=1,  # Reduced
            vertical_continuity_weight=0.2,  # Reduced
            use_shape_regularization=True,  # Enable shape regularization
            use_skeleton_guidance=True  # Enable skeleton-based shape preservation
        )
    
    def _smooth_base_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply smoothing to the base refiner refined masks."""
        # Convert to proper format
        mask_binary = (mask > 127).astype(np.uint8)
        
        # Apply strong smoothing
        # 1. Gaussian blur
        mask_smooth = cv2.GaussianBlur(mask_binary * 255, (7, 7), 2.0)
        
        # 2. Threshold back
        _, mask_smooth = cv2.threshold(mask_smooth, 127, 255, cv2.THRESH_BINARY)
        
        # 3. Morphological smoothing - adjust based on thickness_factor
        if self.thickness_factor < 0.3:
            # Extremely thin - very aggressive erosion
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            mask_smooth = cv2.erode(mask_smooth, kernel, iterations=3)
            mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel, iterations=2)
            # No median filter to preserve thinness
        elif self.thickness_factor < 0.5:
            # Very thin - aggressive erosion
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            mask_smooth = cv2.erode(mask_smooth, kernel, iterations=2)
            mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
            # Smaller median filter
            mask_smooth = cv2.medianBlur(mask_smooth, 3)
        elif self.thickness_factor < 0.7:
            # Thin - moderate erosion
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
            mask_smooth = cv2.erode(mask_smooth, kernel, iterations=1)
            # Small median filter
            mask_smooth = cv2.medianBlur(mask_smooth, 3)
        elif self.thickness_factor < 0.9:
            # Normal - light smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
            # Medium median filter
            mask_smooth = cv2.medianBlur(mask_smooth, 5)
        else:
            # Slightly thick - light dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel)
            # Standard median filter
            mask_smooth = cv2.medianBlur(mask_smooth, 5)
        
        # 5. Final smoothing with bilateral filter for edge preservation
        mask_smooth = cv2.bilateralFilter(mask_smooth, 9, 75, 75)
        
        # Ensure binary
        _, mask_smooth = cv2.threshold(mask_smooth, 127, 255, cv2.THRESH_BINARY)
        
        return mask_smooth
    
    def _apply_final_thickness_control(self, masks_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Apply final thickness control as a post-processing step.
        This ensures the thickness_factor is applied regardless of the refinement path.
        """
        print(f"Applying final thickness control with factor: {self.thickness_factor:.2f}")
        print(f"  Processing {len(masks_dict)} masks")
        
        controlled_masks = {}
        for idx, (inst_id, mask) in enumerate(masks_dict.items()):
            print(f"  Processing mask {idx+1}/{len(masks_dict)}, ID: {inst_id}")
            
            # Ensure binary
            mask_binary = (mask > 127).astype(np.uint8) * 255
            
            # Debug: check initial mask properties
            y_coords, x_coords = np.where(mask_binary > 127)
            if len(x_coords) > 0:
                initial_width = np.max(x_coords) - np.min(x_coords) + 1
                initial_height = np.max(y_coords) - np.min(y_coords) + 1
                initial_area = np.sum(mask_binary > 127)
                print(f"    Initial: width={initial_width}, height={initial_height}, area={initial_area}")
            
            if self.thickness_factor < 0.3:
                # Extremely thin - very aggressive erosion
                print(f"  Applying very aggressive erosion for extremely thin electrodes (mask {inst_id})")
                # Get original area for comparison
                original_area = np.sum(mask_binary > 127)
                
                # First pass with larger kernel
                kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_binary = cv2.erode(mask_binary, kernel_large, iterations=2)
                
                # Second pass with cross kernel for thinning
                kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                mask_binary = cv2.erode(mask_binary, kernel_cross, iterations=3)  # Increased iterations
                
                # Final thinning with vertical bias
                kernel_vert = np.array([[0,1,0],[0,1,0],[0,1,0]], dtype=np.uint8)
                mask_binary = cv2.erode(mask_binary, kernel_vert, iterations=1)
                
                # Clean up
                mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_cross, iterations=2)
                
                final_area = np.sum(mask_binary > 127)
                print(f"    Area reduced from {original_area} to {final_area} ({final_area/original_area*100:.1f}%)")
                
                # Safety check: if electrode was completely eroded, revert
                if final_area == 0:
                    print(f"    ERROR: Electrode completely eroded! Reverting to original")
                    mask_binary = (mask > 127).astype(np.uint8) * 255
                
            elif self.thickness_factor < 0.5:
                # Very thin - aggressive erosion
                print(f"  Applying aggressive erosion for very thin electrodes")
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_binary = cv2.erode(mask_binary, kernel, iterations=3)
                # Additional opening
                mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel, iterations=1)
                
            elif self.thickness_factor < 0.7:
                # Thin - moderate erosion
                print(f"  Applying moderate erosion for thin electrodes")
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_binary = cv2.erode(mask_binary, kernel, iterations=2)
                
            elif self.thickness_factor < 0.9:
                # Normal - light erosion to prevent adhesion
                print(f"  Normal thickness - light erosion")
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_binary = cv2.erode(mask_binary, kernel, iterations=1)
                mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel, iterations=1)
                
            else:
                # Slightly thick - minimal dilation
                print(f"  Slightly thick - minimal dilation")
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_binary = cv2.dilate(mask_binary, kernel, iterations=1)
                mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Always check final width for thin electrodes
            if self.thickness_factor < 0.7:  # For all thin/normal electrodes
                # Calculate final dimensions
                y_coords, x_coords = np.where(mask_binary > 127)
                if len(x_coords) > 0:
                    final_width = np.max(x_coords) - np.min(x_coords) + 1
                    final_height = np.max(y_coords) - np.min(y_coords) + 1
                    final_area = np.sum(mask_binary > 127)
                    
                    # Target width based on thickness factor
                    target_width_ratio = 0.08 + 0.07 * self.thickness_factor  # 0.08-0.15
                    max_width = int(final_height * target_width_ratio)
                    
                    # If still too wide, apply targeted thinning
                    if final_width > max_width:
                        extra_erosion = (final_width - max_width) // 2
                        print(f"    Electrode {inst_id} width {final_width}px > target {max_width}px, extra erosion: {extra_erosion}")
                        
                        # Use horizontal erosion to specifically reduce width
                        for i in range(extra_erosion):
                            kernel_horiz = np.array([[0,0,0],[1,1,1],[0,0,0]], dtype=np.uint8)
                            mask_binary = cv2.erode(mask_binary, kernel_horiz, iterations=1)
                            
                            # Check if we've reduced enough
                            y_coords, x_coords = np.where(mask_binary > 127)
                            if len(x_coords) > 0:
                                current_width = np.max(x_coords) - np.min(x_coords) + 1
                                if current_width <= max_width:
                                    break
                    
                    print(f"    Final: width={final_width}, height={final_height}, area={final_area}")
            
            # Ensure we didn't completely erode the mask
            final_pixels = np.sum(mask_binary > 127)
            if final_pixels == 0:
                print(f"    ERROR: Mask {inst_id} was completely eroded! Using original")
                controlled_masks[inst_id] = mask
            elif final_pixels < 50:  # Too small
                print(f"    Warning: Mask {inst_id} became too small ({final_pixels} pixels), using original")
                controlled_masks[inst_id] = mask
            else:
                controlled_masks[inst_id] = mask_binary
                
        return controlled_masks
    
    def refine_unified_region(self, region_data: Dict) -> Dict[int, np.ndarray]:
        """
        Refine multiple instances with strong anti-adhesion constraints.
        """
        print(f"\n[Lattice] refine_unified_region called with {len(region_data.get('instance_masks', {}))} instances")
        image = region_data['image']
        instance_masks = region_data['instance_masks']
        
        # Extract masks dictionary
        masks_dict = {}
        for inst_id, data in instance_masks.items():
            mask = data['mask']
            if np.sum(mask > 127) >= 100:
                masks_dict[inst_id] = mask
        
        if not self.enable_joint_refinement or len(masks_dict) < self.min_instances_for_joint:
            # Fall back to the base refiner refinement
            print(f"Using the base refiner refinement for {len(masks_dict)} instances (min required: {self.min_instances_for_joint})")
            base_result = self.apply_base_refinement(masks_dict, image)
            # Apply final thickness control
            return self._apply_final_thickness_control(base_result)
        
        print(f"Processing region with {len(masks_dict)} electrodes using the lattice refiner")
        
        # Step 1: the base refiner individual refinement
        base_refined = self.apply_base_refinement(masks_dict, image)
        
        # Step 2: Check for adhesion in the base refiner results
        adhesion_detected = self._detect_adhesion(base_refined)
        
        if not adhesion_detected:
            print("No adhesion detected in the base refiner results - applying smoothing only")
            # Even without adhesion, apply smoothing to reduce jaggedness
            smoothed_masks = {}
            for inst_id, mask in base_refined.items():
                # Apply edge smoothing
                smoothed = self._smooth_base_mask(mask)
                smoothed_masks[inst_id] = smoothed
            # Apply final thickness control
            return self._apply_final_thickness_control(smoothed_masks)
        
        print("Adhesion detected - applying lattice model with skeleton guidance")
        
        # Step 3: Use lattice model for joint refinement
        try:
            # Compute features for optimization
            features = self.compute_unified_features(image)
            edges = features['edges']
            
            # Compute image gradient for soft boundary finding
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
            gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            image_gradient = np.abs(gradient_x)
            
            # Fit lattice model (now with integrated skeleton guidance)
            print("Fitting lattice model with skeleton guidance...")
            fit_result = self.lattice_model.fit_from_instances(base_refined, image, edges)
            
            if fit_result['success']:
                print(f"Lattice model fitted successfully with K={fit_result['K']} electrodes")
                
                # Generate refined masks using lattice model with gradient information
                joint_refined = self.lattice_model.generate_refined_masks(base_refined, image_gradient)
                
                # Apply additional separation enforcement if needed
                if self.enforce_separation:
                    final_refined = self._enforce_minimum_separation(joint_refined)
                    # Apply final thickness control
                    return self._apply_final_thickness_control(final_refined)
                else:
                    # Apply final thickness control
                    return self._apply_final_thickness_control(joint_refined)
            else:
                print("Lattice model fitting failed - using fast separation")
                separated = self._enforce_minimum_separation(base_refined)
                # Apply final thickness control
                return self._apply_final_thickness_control(separated)
                
        except Exception as e:
            print(f"Error in lattice optimization: {e}")
            print("Falling back to fast separation enforcement")
            separated = self._enforce_minimum_separation(base_refined)
            # Apply final thickness control
            return self._apply_final_thickness_control(separated)
    
    def _detect_adhesion(self, masks: Dict[int, np.ndarray]) -> bool:
        """Detect if electrodes are adhering."""
        if len(masks) < 2:
            return False
            
        # Create combined mask to check for overlap
        combined = np.zeros_like(next(iter(masks.values())), dtype=np.int32)
        
        for idx, (inst_id, mask) in enumerate(masks.items()):
            mask_binary = (mask > 127).astype(np.uint8)
            # Add each mask with a unique ID
            combined[mask_binary > 0] += (idx + 1)
        
        # Any pixel with value > max(idx) means overlap
        if np.any(combined > len(masks)):
            return True
        
        # Quick proximity check using dilation
        if self.min_separation_pixels > 0:
            kernel = np.ones((self.min_separation_pixels * 2 + 1, self.min_separation_pixels * 2 + 1), dtype=np.uint8)
            
            for idx, (inst_id, mask) in enumerate(masks.items()):
                mask_binary = (mask > 127).astype(np.uint8)
                mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
                
                # Check if dilated mask overlaps with any other mask
                for idx2, (inst_id2, mask2) in enumerate(masks.items()):
                    if idx >= idx2:  # Skip self and already checked pairs
                        continue
                    mask2_binary = (mask2 > 127).astype(np.uint8)
                    if np.any(mask_dilated & mask2_binary):
                        return True
        
        return False
    
    def _enforce_minimum_separation(self, masks: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Enforce minimum separation between masks."""
        # Sort masks by average x position
        sorted_items = sorted(masks.items(), 
                            key=lambda item: np.mean(np.where(item[1] > 127)[1]))
        
        refined = {}
        accumulated_forbidden = None
        
        for inst_id, mask in sorted_items:
            mask_binary = (mask > 127).astype(np.uint8)
            
            if accumulated_forbidden is not None:
                # Remove overlap with previous masks
                mask_binary = mask_binary & (~accumulated_forbidden)
                
                # Ensure minimum separation
                kernel = np.ones((1, self.min_separation_pixels), dtype=np.uint8)
                forbidden_dilated = cv2.dilate(accumulated_forbidden.astype(np.uint8), kernel)
                mask_binary = mask_binary & (~forbidden_dilated.astype(bool))
            
            # Update accumulated forbidden region
            if accumulated_forbidden is None:
                accumulated_forbidden = mask_binary > 0
            else:
                accumulated_forbidden = accumulated_forbidden | (mask_binary > 0)
            
            # Convert back to uint8
            refined[inst_id] = mask_binary.astype(np.uint8) * 255
        
        return refined
    
    def apply_base_refinement(self, masks_dict: Dict[int, np.ndarray], 
                           image: np.ndarray) -> Dict[int, np.ndarray]:
        """Apply the base refiner refinement to each instance."""
        refined_masks = {}
        features = self.compute_unified_features(image)
        
        for inst_id, mask in masks_dict.items():
            structural_constraint = np.ones_like(mask, dtype=np.float32)
            
            # Use the base refiner's refine_instance_vectorized method
            refined = self.refine_instance_vectorized(
                mask, features, structural_constraint,
                use_multiscale=(mask.shape[0] > 50 and mask.shape[1] > 50)
            )
            
            refined_masks[inst_id] = refined
        
        # Apply thickness control to the base refiner results as well
        print(f"[Lattice] Applying final thickness control to {len(refined_masks)} masks")
        final_masks = self._apply_final_thickness_control(refined_masks)
        
        # Debug: check if any masks were lost or became empty
        for inst_id in refined_masks.keys():
            if inst_id not in final_masks:
                print(f"[Lattice] WARNING: Mask {inst_id} lost during thickness control!")
            elif np.sum(final_masks[inst_id] > 127) == 0:
                print(f"[Lattice] WARNING: Mask {inst_id} became empty after thickness control!")
        
        return final_masks


def main():
    """Test the lattice refiner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Lattice refiner: anti-adhesion refinement')
    parser.add_argument('--data-dir', type=str, default='data/Full_Instances',
                        help='Instance data directory')
    parser.add_argument('--output-dir', type=str, default='results/refined_masks_lattice',
                        help='Output directory')
    parser.add_argument('--test-image', type=str, default=None,
                        help='Specific image to test')
    parser.add_argument("--thickness-factor", type=float, default=1.0,
                        help="Thickness control factor (0.1-2.0, lower=thinner)")
    parser.add_argument('--origin-dir', type=str, default='data/Origin',
                        help='Directory containing original full images and masks')
    
    args = parser.parse_args()
    
    # Initialize the lattice refiner with strong anti-adhesion settings
    refiner = LatticeRefiner(
        instances_per_region=10,  # Moderate group size
        pyramid_levels=1,  # Single scale for speed
        enable_joint_refinement=True,
        enforce_separation=False,
        min_separation_pixels=5,
        thickness_factor=args.thickness_factor  # Pass thickness factor
    )
    
    # Process instances
    os.makedirs(args.output_dir, exist_ok=True)
    
    instance_info_dir = os.path.join(args.data_dir, 'instance_info')
    enhanced_image_dir = os.path.join(args.origin_dir, 'images_enhanced')
    mask_dir = os.path.join(args.data_dir, 'repaired_masks')
    
    # Get instance info files
    info_files = [f for f in os.listdir(instance_info_dir) if f.endswith('_info.json')]
    
    # Filter for specific image if requested
    if args.test_image:
        info_files = [f for f in info_files if args.test_image in f]
    
    print(f"Processing {len(info_files)} images with the lattice refiner anti-adhesion refinement...")
    
    for info_file in tqdm(info_files):
        base_name = info_file.replace('_info.json', '')
        
        # Load instance info
        with open(os.path.join(instance_info_dir, info_file), 'r') as f:
            info_data = json.load(f)
        
        # Load full image
        full_image_path = os.path.join(enhanced_image_dir, f"{base_name}.png")
        if not os.path.exists(full_image_path):
            continue
        
        full_image = cv2.imread(full_image_path)
        instances = info_data['instances']
        
        # Process in groups
        for i in range(0, len(instances), refiner.instances_per_region):
            group = instances[i:i+refiner.instances_per_region]
            
            # Prepare instance data
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
            
            # Create region and refine
            print("Creating unified region...")
            region_data = refiner.create_unified_region(instance_data, full_image)
            print(f"Region size: {region_data['image'].shape}")
            
            print("Starting refinement...")
            refined_masks = refiner.refine_unified_region(region_data)
            print(f"Refinement complete, got {len(refined_masks)} masks")
            
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
    
    print(f"Refinement complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()