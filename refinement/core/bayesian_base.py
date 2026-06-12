#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Base Refiner

Improvements over the early prototypes:
1. Multi-scale processing - Process at multiple scales and combine results
2. Adaptive thresholding - Dynamically determine optimal thresholds
3. Enhanced edge detection - Use Canny and other advanced edge detectors

Key features:
- Pyramid-based multi-scale refinement
- Adaptive threshold learning from local statistics
- Multiple edge detection methods with fusion
- Scale-aware parameter adjustment
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage import morphology, filters, feature
from typing import Tuple, Dict, Optional, List
import os
from tqdm import tqdm
import json


class BayesianRefinerBase:
    """
    Base Bayesian refiner with multi-scale processing, adaptive thresholding, and enhanced edge detection.
    """
    
    def __init__(self, 
                 instances_per_region: int = 4,
                 # Multi-scale parameters
                 pyramid_levels: int = 3,
                 scale_weights: Optional[List[float]] = None,
                 # Edge detection parameters
                 edge_methods: List[str] = ['sobel', 'canny', 'laplacian'],
                 edge_fusion_weights: Optional[Dict[str, float]] = None,
                 # Adaptive threshold parameters
                 adaptive_window_size: int = 51,
                 threshold_sensitivity: float = 0.5,
                 # Parameters inherited from the early prototype
                 structure_weight: float = 0.3,
                 consistency_weight: float = 0.3,
                 intensity_weight: float = 0.2,
                 smoothness_weight: float = 0.2,
                 # Automatic enhancement parameters
                 min_electrode_width: int = 10,
                 straight_edge_threshold: float = 0.7,
                 boundary_quality_threshold: float = 0.8,
                 # Compatibility parameters
                 enable_straight_edge_enhancement: bool = True,
                 enable_adaptive_boundary_preservation: bool = True):
        """Initialize the base Bayesian refinement model."""
        self.instances_per_region = instances_per_region
        
        # Multi-scale parameters
        self.pyramid_levels = pyramid_levels
        self.scale_weights = scale_weights or [0.25, 0.5, 0.25]  # Fine, medium, coarse
        
        # Edge detection parameters
        self.edge_methods = edge_methods
        self.edge_fusion_weights = edge_fusion_weights or {
            'sobel': 0.4,
            'canny': 0.4,
            'laplacian': 0.2
        }
        
        # Adaptive threshold parameters
        self.adaptive_window_size = adaptive_window_size
        self.threshold_sensitivity = threshold_sensitivity
        
        # Weights inherited from the early prototype
        self.structure_weight = structure_weight
        self.consistency_weight = consistency_weight
        self.intensity_weight = intensity_weight
        self.smoothness_weight = smoothness_weight
        
        # Automatic enhancement parameters
        self.min_electrode_width = min_electrode_width
        self.straight_edge_threshold = straight_edge_threshold
        self.boundary_quality_threshold = boundary_quality_threshold
        
        # Compatibility settings
        self.enable_straight_edge_enhancement = enable_straight_edge_enhancement
        self.enable_adaptive_boundary_preservation = enable_adaptive_boundary_preservation
    
    def build_pyramid(self, image: np.ndarray, mask: np.ndarray, 
                     levels: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Build image pyramid for multi-scale processing.
        
        Args:
            image: Input image
            mask: Input mask
            levels: Number of pyramid levels
            
        Returns:
            List of (image, mask) tuples at different scales
        """
        pyramid = [(image, mask)]
        
        for i in range(1, levels):
            # Downsample by factor of 2
            img_down = cv2.pyrDown(pyramid[-1][0])
            mask_down = cv2.pyrDown(pyramid[-1][1])
            pyramid.append((img_down, mask_down))
        
        return pyramid
    
    def compute_multi_method_edges(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute edges using multiple methods and fuse results.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Fused edge probability map
        """
        edge_maps = {}
        
        # Sobel edges
        if 'sobel' in self.edge_methods:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_maps['sobel'] = 1 / (1 + np.exp(-sobel_mag / 10))
        
        # Canny edges
        if 'canny' in self.edge_methods:
            # Adaptive Canny thresholds based on image statistics
            median_val = np.median(gray)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * median_val))
            upper = int(min(255, (1.0 + sigma) * median_val))
            
            canny = cv2.Canny(gray.astype(np.uint8), lower, upper)
            # Distance transform to create soft edges
            dist = distance_transform_edt(~canny.astype(bool))
            edge_maps['canny'] = np.exp(-dist / 3)
        
        # Laplacian edges
        if 'laplacian' in self.edge_methods:
            # Convert to uint8 for Laplacian computation to avoid format issues
            gray_uint8 = (gray * 255).astype(np.uint8) if gray.dtype != np.uint8 else gray
            laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
            edge_maps['laplacian'] = 1 / (1 + np.exp(-np.abs(laplacian) / 10))
        
        # Ridge detection (good for thin structures)
        if 'ridge' in self.edge_methods:
            ridge = feature.hessian_matrix_eigvals(filters.gaussian(gray, 1))[0]
            edge_maps['ridge'] = 1 / (1 + np.exp(-ridge / 10))
        
        # Fuse edge maps
        fused_edges = np.zeros_like(gray, dtype=np.float32)
        total_weight = 0
        
        for method, edge_map in edge_maps.items():
            weight = self.edge_fusion_weights.get(method, 1.0 / len(edge_maps))
            fused_edges += weight * edge_map
            total_weight += weight
        
        if total_weight > 0:
            fused_edges /= total_weight
        
        return fused_edges
    
    def compute_adaptive_threshold(self, posterior: np.ndarray, 
                                 mask: np.ndarray) -> float:
        """
        Compute adaptive threshold based on local statistics.
        
        Args:
            posterior: Posterior probability map
            mask: Current mask
            
        Returns:
            Adaptive threshold value
        """
        # Get foreground and background regions
        fg_values = posterior[mask > 127]
        bg_values = posterior[mask <= 127]
        
        if len(fg_values) > 0 and len(bg_values) > 0:
            # Otsu-like threshold between foreground and background distributions
            all_values = np.concatenate([fg_values, bg_values])
            threshold = filters.threshold_otsu(all_values)
            
            # Adjust based on sensitivity
            # Higher sensitivity = lower threshold (more inclusive)
            threshold *= (1 - self.threshold_sensitivity * 0.3)
            
            # Ensure threshold is in reasonable range
            threshold = np.clip(threshold, 0.3, 0.7)
        else:
            # Fallback to default
            threshold = 0.5
        
        return threshold
    
    def assess_boundary_quality(self, mask: np.ndarray, edges: np.ndarray) -> float:
        """
        Assess the quality of mask boundaries.
        
        Args:
            mask: Input mask
            edges: Edge probability map
            
        Returns:
            Boundary quality score (0-1, higher is better)
        """
        # Extract mask boundary
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel)
        boundary = mask.astype(np.uint8) - eroded
        
        # Calculate how well boundaries align with edges
        boundary_coords = np.where(boundary > 0)
        if len(boundary_coords[0]) == 0:
            return 0.0
        
        # Sample edge values at boundary locations
        edge_values = edges[boundary_coords]
        
        # Calculate metrics with more conservative scoring
        mean_edge_strength = np.mean(edge_values)
        edge_consistency = np.clip(1.0 - 2 * np.std(edge_values), 0, 1)  # Penalize inconsistency more
        
        # Check smoothness of boundary
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Calculate perimeter to area ratio
            perimeter = cv2.arcLength(contours[0], True)
            area = cv2.contourArea(contours[0])
            if area > 0:
                # Normalize compactness score to be less sensitive
                compactness = np.clip(4 * np.pi * area / (perimeter * perimeter), 0, 1)
                # Penalize very non-compact shapes
                if compactness < 0.5:
                    compactness *= 0.5
            else:
                compactness = 0
        else:
            compactness = 0
        
        # Check for holes in the mask (indicates poor quality)
        filled = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        has_holes = np.sum(filled - mask) > 10  # More than 10 pixels difference
        
        # Combine metrics with penalty for holes
        quality_score = 0.3 * mean_edge_strength + 0.3 * edge_consistency + 0.4 * compactness
        if has_holes:
            quality_score *= 0.7  # Reduce quality score if mask has holes
        
        return np.clip(quality_score, 0, 1)
    
    def refine_at_scale(self, mask: np.ndarray,
                       features: Dict, structural_constraint: np.ndarray,
                       scale_factor: float = 1.0) -> np.ndarray:
        """
        Refine mask at a specific scale.
        
        Args:
            mask: Mask at current scale
            features: Pre-computed features
            structural_constraint: Structural constraints
            scale_factor: Current scale relative to original
            
        Returns:
            Refined mask at current scale
        """
        gray = features['gray']
        edges = features['edges']
        
        # Scale-aware parameters
        # Larger scales focus more on structure, smaller scales on details
        intensity_weight = self.intensity_weight * (1 + 0.3 * (1 - scale_factor))
        structure_weight = self.structure_weight * (1 + 0.3 * scale_factor)
        
        # Intensity likelihood
        inside = gray[mask > 127]
        
        # Adaptive window for background sampling
        window_size = max(15, int(30 * scale_factor))
        dilated = cv2.dilate(mask.astype(np.uint8), 
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                   (window_size, window_size)))
        outside_mask = dilated & (~(mask > 127))
        outside = gray[outside_mask > 0]
        
        if len(inside) > 0 and len(outside) > 0:
            # Fit distributions
            mu_in, sigma_in = np.mean(inside), np.std(inside) + 1e-6
            mu_out, sigma_out = np.mean(outside), np.std(outside) + 1e-6
            
            # Compute likelihood
            from scipy.stats import norm
            p_in = norm.pdf(gray, mu_in, sigma_in)
            p_out = norm.pdf(gray, mu_out, sigma_out)
            
            intensity_likelihood = p_in / (p_in + p_out + 1e-10)
        else:
            intensity_likelihood = np.ones_like(gray) * 0.5
        
        # Combine likelihoods with scale-aware weights
        likelihood = intensity_weight * intensity_likelihood + (1 - intensity_weight) * edges
        
        # Distance transforms
        dist_inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform((~(mask > 127)).astype(np.uint8), cv2.DIST_L2, 5)
        
        if self.enable_adaptive_boundary_preservation:
            # Assess boundary quality to determine refinement approach
            boundary_quality = self.assess_boundary_quality(mask, edges)
            use_boundary_preservation = boundary_quality > self.boundary_quality_threshold
        else:
            # Original behavior - no adaptive preservation
            boundary_quality = 0.0
            use_boundary_preservation = False
        
        if use_boundary_preservation:
            # High quality boundaries - preserve them with minimal changes
            # Simply use the original mask as a strong prior
            mask_prior = mask.astype(np.float32) / 255.0
            
            # Small morphological smoothing to clean up boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_prior = cv2.morphologyEx(mask_prior, cv2.MORPH_CLOSE, kernel)
            mask_prior = cv2.GaussianBlur(mask_prior, (5, 5), 1.0)
            
            # Combine with weighted structural constraint
            weighted_structural_constraint = structure_weight * structural_constraint + (1 - structure_weight)
            
            # Strong emphasis on maintaining original shape
            posterior = likelihood * mask_prior * weighted_structural_constraint
        else:
            # Lower quality boundaries - allow more refinement
            # Adaptive refinement based on boundary quality
            preservation_factor = boundary_quality / self.boundary_quality_threshold
            
            # Width prior with adaptive strength - slightly increased to reduce expansion
            width_prior_strength = 0.25 + 0.1 * preservation_factor  # 0.25-0.35 based on quality
            width_prior = np.exp(-width_prior_strength * dist_inside / scale_factor)
            # Prevent width prior from becoming too small to avoid holes
            width_prior = np.clip(width_prior, 0.5, 1.0)
            
            # Boundary influence based on quality
            if boundary_quality > 0.5:  # Medium quality
                boundary_band = 5.0 / scale_factor
                boundary_influence = np.exp(-dist_outside / boundary_band) * preservation_factor
                width_prior = width_prior * (1 - 0.3 * boundary_influence)
            
            # Smoothness prior
            smooth_mask = cv2.GaussianBlur(mask.astype(np.float32), 
                                          (int(5 / scale_factor) * 2 + 1, 
                                           int(5 / scale_factor) * 2 + 1), 
                                          1.0)
            
            # Combine with weighted structural constraint
            weighted_structural_constraint = structure_weight * structural_constraint + (1 - structure_weight)
            posterior = likelihood * width_prior * smooth_mask * weighted_structural_constraint
        
        # Adaptive thresholding
        threshold = self.compute_adaptive_threshold(posterior, mask)
        refined = (posterior > threshold).astype(np.uint8) * 255
        
        # Scale-aware morphological operations with slightly smaller kernel
        kernel_size = max(3, int(4 * scale_factor))  # Reduced from 5 to 4
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        
        return refined
    
    def refine_multiscale(self, image: np.ndarray, mask: np.ndarray,
                         structural_constraint: np.ndarray) -> np.ndarray:
        """
        Perform multi-scale refinement.
        
        Args:
            image: Input image
            mask: Input mask
            structural_constraint: Structural constraints
            
        Returns:
            Refined mask
        """
        # Build pyramid
        pyramid = self.build_pyramid(image, mask, self.pyramid_levels)
        
        # Process each scale
        refined_pyramid = []
        
        for level, (img_scale, mask_scale) in enumerate(pyramid):
            scale_factor = 1.0 / (2 ** level)
            
            # Compute features at this scale
            if len(img_scale.shape) == 3:
                gray_scale = cv2.cvtColor(img_scale, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            else:
                gray_scale = img_scale.astype(np.float32)
                if gray_scale.max() > 1.0:
                    gray_scale = gray_scale / 255.0
            
            # Enhanced edge detection
            edges_scale = self.compute_multi_method_edges(gray_scale)
            
            features = {
                'gray': gray_scale,
                'edges': edges_scale
            }
            
            # Resize structural constraint to current scale
            h, w = mask_scale.shape
            constraint_scale = cv2.resize(structural_constraint, (w, h))
            
            # Refine at current scale
            refined = self.refine_at_scale(mask_scale, features,
                                         constraint_scale, scale_factor)
            
            refined_pyramid.append(refined)
        
        # Combine scales (coarse to fine)
        combined = refined_pyramid[-1]  # Start with coarsest
        
        for level in range(len(refined_pyramid) - 2, -1, -1):
            # Upsample to next level
            h, w = refined_pyramid[level].shape
            upsampled = cv2.resize(combined, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Weighted combination
            weight = self.scale_weights[level]
            combined = (weight * refined_pyramid[level] + 
                       (1 - weight) * upsampled).astype(np.uint8)
        
        # Final threshold
        combined = (combined > 127).astype(np.uint8) * 255
        
        return combined
    
    def compute_unified_features(self, image: np.ndarray) -> Dict:
        """Compute image features with enhanced edge detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = image.astype(np.float32)
            if gray.max() > 1.0:
                gray = gray / 255.0
        
        # Enhanced edge detection
        edges = self.compute_multi_method_edges(gray)
        
        # Gradient information
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        return {
            'gray': gray,
            'edges': edges,
            'grad_x': grad_x,
            'grad_y': grad_y
        }
    
    def detect_electrode_orientation(self, mask: np.ndarray) -> str:
        """Detect if electrode is primarily vertical or horizontal."""
        coords = np.where(mask > 127)
        if len(coords[0]) == 0:
            return 'vertical'
        
        min_y, max_y = coords[0].min(), coords[0].max()
        min_x, max_x = coords[1].min(), coords[1].max()
        
        height = max_y - min_y
        width = max_x - min_x
        
        return 'vertical' if height > width else 'horizontal'
    
    def extract_electrode_core(self, mask: np.ndarray) -> Dict:
        """Extract the core structure of the electrode for straight edge reconstruction."""
        # Consolidate fragmented regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
        consolidated = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Fill holes
        from scipy.ndimage import binary_fill_holes
        filled = binary_fill_holes(consolidated > 127).astype(np.uint8) * 255
        
        # Get main component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        if num_labels <= 1:
            return None
        
        # Find largest component
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        main_mask = (labels == largest_idx).astype(np.uint8) * 255
        
        # Detect orientation
        orientation = self.detect_electrode_orientation(main_mask)
        coords = np.where(main_mask > 127)
        
        if orientation == 'vertical':
            min_y, max_y = coords[0].min(), coords[0].max()
            
            # Calculate median width
            widths = []
            x_centers = []
            for y in range(min_y, max_y, 5):
                row_coords = coords[1][coords[0] == y]
                if len(row_coords) > 0:
                    widths.append(row_coords.max() - row_coords.min() + 1)
                    x_centers.append((row_coords.min() + row_coords.max()) // 2)
            
            if widths:
                return {
                    'orientation': 'vertical',
                    'bounds': (min_y, max_y),
                    'center': int(np.median(x_centers)),
                    'width': max(int(np.median(widths)), self.min_electrode_width),
                    'original_mask': main_mask
                }
        else:
            min_x, max_x = coords[1].min(), coords[1].max()
            
            # Calculate median height
            heights = []
            y_centers = []
            for x in range(min_x, max_x, 5):
                col_coords = coords[0][coords[1] == x]
                if len(col_coords) > 0:
                    heights.append(col_coords.max() - col_coords.min() + 1)
                    y_centers.append((col_coords.min() + col_coords.max()) // 2)
            
            if heights:
                return {
                    'orientation': 'horizontal',
                    'bounds': (min_x, max_x),
                    'center': int(np.median(y_centers)),
                    'width': max(int(np.median(heights)), self.min_electrode_width),
                    'original_mask': main_mask
                }
        
        return None
    
    def create_straight_electrode(self, core_info: Dict, shape: Tuple[int, int]) -> np.ndarray:
        """Create a straight electrode mask based on core information."""
        h, w = shape
        straight_mask = np.zeros((h, w), dtype=np.uint8)
        
        if core_info['orientation'] == 'vertical':
            min_y, max_y = core_info['bounds']
            half_width = core_info['width'] // 2
            
            x_start = max(0, core_info['center'] - half_width)
            x_end = min(w, core_info['center'] + half_width)
            
            straight_mask[min_y:max_y+1, x_start:x_end] = 255
        else:
            min_x, max_x = core_info['bounds']
            half_height = core_info['width'] // 2
            
            y_start = max(0, core_info['center'] - half_height)
            y_end = min(h, core_info['center'] + half_height)
            
            straight_mask[y_start:y_end, min_x:max_x+1] = 255
        
        return straight_mask
    
    def compute_structural_constraint(self, instance_masks: Dict, shape: Tuple) -> np.ndarray:
        """
        Compute structural constraint (same as an early prototype).
        """
        h, w = shape
        constraint = np.ones((h, w), dtype=np.float32)
        
        inst_ids = list(instance_masks.keys())
        
        for i in range(len(inst_ids)):
            for j in range(i + 1, len(inst_ids)):
                mask1 = instance_masks[inst_ids[i]]['mask'] > 0
                mask2 = instance_masks[inst_ids[j]]['mask'] > 0
                
                dilated1 = cv2.dilate(mask1.astype(np.uint8), np.ones((7, 7)))
                dilated2 = cv2.dilate(mask2.astype(np.uint8), np.ones((7, 7)))
                
                overlap = dilated1 & dilated2
                constraint[overlap > 0] *= 0.1
        
        return constraint
    
    def refine_instance_vectorized(self, mask: np.ndarray, features: Dict, 
                                 structural_constraint: np.ndarray,
                                 use_multiscale: bool = True) -> np.ndarray:
        """
        Refine a single instance mask with automatic straight edge enhancement.
        
        Args:
            mask: Instance mask
            features: Pre-computed features
            structural_constraint: Structural constraints
            use_multiscale: Whether to use multi-scale refinement
            
        Returns:
            Refined mask
        """
        # First apply standard refinement
        if use_multiscale and mask.shape[0] > 50 and mask.shape[1] > 50:
            # Use multi-scale for larger masks
            image = cv2.cvtColor(features['gray'].astype(np.uint8), cv2.COLOR_GRAY2BGR)
            refined_mask = self.refine_multiscale(image, mask, structural_constraint)
        else:
            # Use single-scale refinement for small masks
            refined_mask = self.refine_at_scale(mask, features, 
                                               structural_constraint, scale_factor=1.0)
        
        # Apply straight edge enhancement if enabled and the electrode shape warrants it
        if self.enable_straight_edge_enhancement:
            core_info = self.extract_electrode_core(refined_mask)
            
            if core_info is not None:
                # Create ideal straight electrode
                straight_mask = self.create_straight_electrode(core_info, refined_mask.shape)
                
                # Calculate how well the refined mask matches a straight electrode
                original_mask = core_info['original_mask']
                overlap = np.logical_and(original_mask > 127, straight_mask > 127).sum()
                union = np.logical_or(original_mask > 127, straight_mask > 127).sum()
                
                if union > 0:
                    iou = overlap / union
                    
                    # Only apply straight edge enhancement if the electrode is already quite straight
                    # This prevents forcing straight edges on naturally curved or complex electrodes
                    if iou > self.straight_edge_threshold:
                        if iou > 0.9:
                            # Already very straight, just minor smoothing
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
                        else:
                            # Blend refined and straight versions
                            # Use stronger straight edge weight for electrodes that are already fairly straight
                            alpha = 0.6 + 0.3 * (iou - self.straight_edge_threshold) / (1 - self.straight_edge_threshold)
                            blended = alpha * straight_mask + (1 - alpha) * refined_mask
                            refined_mask = (blended > 127).astype(np.uint8) * 255
        
        return refined_mask
    
    def create_unified_region(self, instance_data: Dict, full_image: np.ndarray) -> Dict:
        """Create a unified region containing multiple instances (same as an early prototype)."""
        # Calculate region bounds
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        
        for inst_id, data in instance_data.items():
            bbox = data['bbox']
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[0] + bbox[2])
            max_y = max(max_y, bbox[1] + bbox[3])
        
        # Add margin
        margin = 30
        x1 = max(0, min_x - margin)
        y1 = max(0, min_y - margin)
        x2 = min(full_image.shape[1], max_x + margin)
        y2 = min(full_image.shape[0], max_y + margin)
        
        # Extract region
        region_image = full_image[y1:y2, x1:x2]
        region_h, region_w = y2 - y1, x2 - x1
        
        # Create separate mask for each instance
        instance_masks = {}
        
        for inst_id, data in instance_data.items():
            mask = cv2.imread(data['mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            inst_mask = np.zeros((region_h, region_w), dtype=np.uint8)
            
            bbox = data['bbox']
            rel_x = bbox[0] - x1
            rel_y = bbox[1] - y1
            
            inst_mask[rel_y:rel_y+bbox[3], rel_x:rel_x+bbox[2]] = mask
            
            instance_masks[inst_id] = {
                'mask': inst_mask,
                'bbox': bbox,
                'relative_pos': (rel_x, rel_y)
            }
        
        return {
            'image': region_image,
            'instance_masks': instance_masks,
            'bounds': (x1, y1, x2, y2),
            'shape': (region_h, region_w)
        }
    
    def refine_unified_region(self, region_data: Dict, iterations: int = 3) -> Dict[int, np.ndarray]:
        """
        Refine multiple instances efficiently with the base refiner improvements.
        """
        image = region_data['image']
        instance_masks = region_data['instance_masks']
        shape = region_data['shape']
        
        # Compute shared features once
        features = self.compute_unified_features(image)
        
        # Compute structural constraint once
        structural_constraint = self.compute_structural_constraint(instance_masks, shape)
        
        # Refine each instance
        refined_masks = {}
        
        for inst_id, inst_data in instance_masks.items():
            current_mask = inst_data['mask']
            
            # Determine if we should use multi-scale
            bbox = inst_data['bbox']
            use_multiscale = bbox[2] > 50 and bbox[3] > 50
            
            # Refine using shared features
            for iteration in range(iterations):
                current_mask = self.refine_instance_vectorized(
                    current_mask, features, structural_constraint,
                    use_multiscale=use_multiscale and iteration == 0  # Only first iteration
                )
            
            refined_masks[inst_id] = current_mask
        
        return refined_masks


def process_with_base_refiner(data_dir: str, output_dir: str, use_repaired: bool = True, origin_dir: str = 'data/Origin', test_image: str = None):
    """Process instances using the base refiner with multi-scale and adaptive thresholding."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize refiner with automatic enhancements
    refiner = BayesianRefinerBase(
        instances_per_region=4,
        pyramid_levels=3,
        edge_methods=['sobel', 'canny', 'laplacian'],
        threshold_sensitivity=0.5,
        min_electrode_width=10,
        straight_edge_threshold=0.7,  # Automatically apply straight edges when appropriate
        boundary_quality_threshold=0.8  # Automatically preserve high-quality boundaries
    )
    
    # Paths
    instance_info_dir = os.path.join(data_dir, 'instance_info')
    enhanced_image_dir = os.path.join(origin_dir, 'images_enhanced')
    
    # Choose mask directory based on use_repaired flag
    if use_repaired:
        coarse_mask_dir = os.path.join(data_dir, 'repaired_masks')
        print(f"Using repaired masks from: {coarse_mask_dir}")
    else:
        coarse_mask_dir = os.path.join(data_dir, 'coarse_masks')
        print(f"Using original coarse masks from: {coarse_mask_dir}")
    
    # Get all instance info files
    info_files = [f for f in os.listdir(instance_info_dir) if f.endswith('_info.json')]
    
    # Filter by test_image if provided
    if test_image:
        info_files = [f for f in info_files if test_image in f]
        if not info_files:
            print(f"No instance info files found matching '{test_image}'")
            return
    
    print(f"Processing {len(info_files)} images with the base refiner multi-scale refinement{' (filtered)' if test_image else ''}...")
    
    total_processed = 0
    
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
        
        # Group instances by proximity
        instances = info_data['instances']
        
        # Process in groups
        for i in range(0, len(instances), refiner.instances_per_region):
            group = instances[i:i + refiner.instances_per_region]
            
            if len(group) < 2:
                # For single instances, just copy
                for inst in group:
                    src = os.path.join(coarse_mask_dir, f"{base_name}_instance_{inst['id']}.png")
                    dst = os.path.join(output_dir, f"{base_name}_instance_{inst['id']}.png")
                    mask = cv2.imread(src)
                    if mask is not None:
                        cv2.imwrite(dst, mask)
                        total_processed += 1
                continue
            
            # Prepare instance data
            instance_data = {}
            for inst in group:
                inst_id = inst['id']
                mask_path = os.path.join(coarse_mask_dir, f"{base_name}_instance_{inst_id}.png")
                
                instance_data[inst_id] = {
                    'bbox': inst['bbox'],
                    'mask_path': mask_path
                }
            
            # Create region
            region_data = refiner.create_unified_region(instance_data, full_image)
            
            # Refine with the base refiner improvements
            refined_masks = refiner.refine_unified_region(region_data)
            
            # Save refined masks
            for inst_id, refined_mask in refined_masks.items():
                # Extract instance portion
                rel_pos = region_data['instance_masks'][inst_id]['relative_pos']
                bbox = instance_data[inst_id]['bbox']
                
                inst_refined = refined_mask[rel_pos[1]:rel_pos[1]+bbox[3], 
                                           rel_pos[0]:rel_pos[0]+bbox[2]]
                
                # Save
                output_path = os.path.join(output_dir, f"{base_name}_instance_{inst_id}.png")
                cv2.imwrite(output_path, inst_refined)
                total_processed += 1
    
    print(f"\nProcessed {total_processed} instances with the base refiner refinement")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bayesian instance mask refinement (base refiner)')
    parser.add_argument('--data-dir', type=str, default="data/Full_Instances",
                        help='Base data directory')
    parser.add_argument('--output-dir', type=str, default="results/refined_masks_base",
                        help='Output directory for refined masks')
    parser.add_argument('--use-original', action='store_true', default=False,
                        help='Use original coarse masks instead of repaired masks')
    parser.add_argument('--origin-dir', type=str, default='data/Origin',
                        help='Directory containing original full images and masks')
    parser.add_argument('--test-image', type=str, default=None,
                        help='Specific image to test')
    
    args = parser.parse_args()
    
    use_repaired = not args.use_original
    process_with_base_refiner(args.data_dir, args.output_dir, use_repaired, args.origin_dir, args.test_image)