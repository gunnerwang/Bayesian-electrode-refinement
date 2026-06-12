#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shape-Transfer Refiner (diagnostic regime)
Adds deep GT shape feature transfer on top of the graph-enhanced lattice model

Key improvements:
1. Uses the GraphEnhancedElectrodeLatticeModel (graph analysis + MST consistency)
2. GT shape feature extraction and transfer
3. Segment-wise shape matching
4. Local curvature and width pattern transfer
5. Endpoint shape preservation
"""

import numpy as np
import cv2
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import Dict, List, Optional, Any
import os
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Import the guided refiner and the lattice refiner's lattice model
import sys
# shared base components live in core/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))
sys.path.append(os.path.dirname(__file__))
from guided_refiner import GuidedRefiner
from lattice_refiner import ElectrodeLatticeModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)


class GraphElectrodeRelationship:
    """Model electrode relationships as a graph structure"""
    
    def __init__(self, distance_threshold: float = 150.0):
        self.distance_threshold = distance_threshold
        self.adjacency_matrix = None
        self.mst = None
        self.node_features = {}
        self.dist_matrix = None
        
    def build_from_electrodes(self, electrodes: List[Dict]) -> None:
        """Build graph from electrode list"""
        n = len(electrodes)
        if n < 2:
            return
            
        # Build distance matrix
        centers = np.array([[e['center_x'], e.get('center_y', 0)] for e in electrodes])
        self.dist_matrix = distance_matrix(centers, centers)
        
        # Build adjacency matrix (edges for nearby electrodes)
        self.adjacency_matrix = (self.dist_matrix < self.distance_threshold).astype(int)
        np.fill_diagonal(self.adjacency_matrix, 0)
        
        # Compute MST for shape propagation
        self.mst = minimum_spanning_tree(self.dist_matrix).toarray()
        
        # Store node features
        for i, electrode in enumerate(electrodes):
            self.node_features[i] = {
                'width': electrode['width'],
                'center_x': electrode['center_x'],
                'area': electrode.get('area', electrode['width'] * 100)
            }
    
    def get_neighbors(self, node_idx: int) -> List[int]:
        """Get neighbor indices for a node"""
        if self.adjacency_matrix is None:
            return []
        return np.where(self.adjacency_matrix[node_idx] > 0)[0].tolist()
    
    def get_mst_children(self, node_idx: int) -> List[int]:
        """Get children nodes in MST"""
        if self.mst is None:
            return []
        return np.where(self.mst[node_idx, :] > 0)[0].tolist()


class LatticePatternAnalyzer:
    """Detect and analyze lattice patterns in electrode arrangement"""
    
    def analyze_pattern(self, electrodes: List[Dict]) -> Dict[str, Any]:
        """Analyze electrode arrangement pattern"""
        if len(electrodes) < 4:
            return {'type': 'irregular', 'params': None}
        
        # Extract positions
        positions = np.array([[e['center_x'], e.get('center_y', 0)] for e in electrodes])
        
        # Compute pairwise differences
        n = len(positions)
        vectors = []
        for i in range(n):
            for j in range(i+1, n):
                vec = positions[j] - positions[i]
                vectors.append(vec)
        
        if not vectors:
            return {'type': 'irregular', 'params': None}
        
        vectors = np.array(vectors)
        
        # Find dominant directions
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        
        # Histogram of angles to find main directions
        hist, bins = np.histogram(angles, bins=36)  # 10-degree bins
        
        # Find peaks
        threshold = len(vectors) * 0.1
        peak_indices = np.where(hist > threshold)[0]
        
        if len(peak_indices) < 1:
            return {'type': 'irregular', 'params': None}
        
        peak_angles = [(bins[i] + bins[i+1])/2 for i in peak_indices]
        
        # Check for grid pattern (perpendicular directions)
        for i in range(len(peak_angles)):
            for j in range(i+1, len(peak_angles)):
                angle_diff = abs(peak_angles[i] - peak_angles[j])
                if abs(angle_diff - np.pi/2) < 0.2:  # ~90 degrees
                    # Compute spacing
                    distances = np.linalg.norm(vectors, axis=1)
                    spacing = np.median(distances[distances < np.percentile(distances, 30)])
                    
                    return {
                        'type': 'grid',
                        'params': {
                            'angles': [peak_angles[i], peak_angles[j]],
                            'spacing': spacing,
                            'regularity': hist[peak_indices].max() / len(vectors)
                        }
                    }
        
        # Check for linear arrangement
        if len(peak_indices) == 1 or (len(peak_indices) == 2 and 
                                      abs(abs(peak_angles[0] - peak_angles[1]) - np.pi) < 0.2):
            return {
                'type': 'linear',
                'params': {
                    'angle': peak_angles[0],
                    'spacing': np.median([e['width'] for e in electrodes])
                }
            }
        
        return {'type': 'irregular', 'params': None}


class GraphEnhancedElectrodeLatticeModel(ElectrodeLatticeModel):
    """
    Enhanced electrode lattice model with graph-based improvements.
    Inherits from the lattice refiner's ElectrodeLatticeModel and adds graph analysis.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract graph-specific parameters
        self.use_graph_analysis = kwargs.pop('use_graph_analysis', True)
        self.graph_distance_threshold = kwargs.pop('graph_distance_threshold', 150.0)
        self.use_mst_propagation = kwargs.pop('use_mst_propagation', True)
        self.neighbor_consistency_weight = kwargs.pop('neighbor_consistency_weight', 0.4)
        
        # Initialize parent
        super().__init__(*args, **kwargs)
        
        # Graph components
        self.graph = GraphElectrodeRelationship(self.graph_distance_threshold)
        self.pattern_analyzer = LatticePatternAnalyzer()
        self.lattice_pattern = None
        
    def fit_from_instances(self, instance_masks: Dict[int, np.ndarray], 
                          image: np.ndarray, edges: np.ndarray) -> Dict:
        """Enhanced fitting with graph analysis"""
        # Call parent fitting
        result = super().fit_from_instances(instance_masks, image, edges)
        
        if not result['success'] or not self.use_graph_analysis:
            return result
        
        # Build graph from electrodes
        self.graph.build_from_electrodes(self.electrodes)
        
        # Analyze lattice pattern
        self.lattice_pattern = self.pattern_analyzer.analyze_pattern(self.electrodes)
        print(f"[ShapeTransfer] Detected lattice pattern: {self.lattice_pattern['type']}")
        
        # Enhanced width profile using graph neighbors
        if self.use_graph_analysis:
            self._compute_graph_based_width_profile()
        
        return result
    
    def _compute_graph_based_width_profile(self):
        """Compute width profile using graph neighborhood information"""
        self.width_profile = np.zeros(self.K)
        
        for i in range(self.K):
            # Get graph neighbors
            neighbors = self.graph.get_neighbors(i)
            
            # Get electrode height to determine if it's a long electrode
            electrode = self.electrodes[i]
            if 'mask' in electrode:
                mask = electrode['mask']
                y_coords, x_coords = np.where(mask > 127)
                if len(y_coords) > 0:
                    height = np.max(y_coords) - np.min(y_coords) + 1
                    aspect_ratio = height / electrode['width'] if electrode['width'] > 0 else 1
                else:
                    aspect_ratio = 1
            else:
                aspect_ratio = 1
            
            if neighbors:
                # Weighted average based on distance
                neighbor_widths = []
                weights = []
                
                for j in neighbors:
                    neighbor_widths.append(self.electrodes[j]['width'])
                    # Closer neighbors have higher weight
                    dist = self.graph.dist_matrix[i, j]
                    weights.append(1.0 / (1.0 + dist/100.0))  # Changed denominator
                
                # Include self with high weight - even higher for long electrodes
                neighbor_widths.append(self.electrodes[i]['width'])
                self_weight = 3.0 if aspect_ratio > 3 else 2.0  # More weight for long electrodes
                weights.append(self_weight)
                
                # Compute weighted average
                avg_width = np.average(neighbor_widths, weights=weights)
                
                # For long electrodes, preserve more of their original width
                if aspect_ratio > 3:  # Long electrode
                    self.width_profile[i] = 0.8 * electrode['width'] + 0.2 * avg_width
                else:
                    self.width_profile[i] = avg_width
            else:
                # No neighbors, use original width
                self.width_profile[i] = self.electrodes[i]['width']
    
    def _find_optimal_boundary(self, left_electrode: Dict, right_electrode: Dict,
                              left_mask: np.ndarray, right_mask: np.ndarray,
                              image_gradient: Optional[np.ndarray] = None) -> int:
        """Enhanced boundary finding using graph relationships"""
        
        # Get electrode indices
        left_idx = next((i for i, e in enumerate(self.electrodes) if e['id'] == left_electrode['id']), -1)
        right_idx = next((i for i, e in enumerate(self.electrodes) if e['id'] == right_electrode['id']), -1)
        
        # Check if they are graph neighbors
        are_neighbors = False
        if left_idx >= 0 and right_idx >= 0 and self.graph.adjacency_matrix is not None:
            are_neighbors = self.graph.adjacency_matrix[left_idx, right_idx] > 0
        
        if are_neighbors and self.lattice_pattern and self.lattice_pattern['type'] == 'grid':
            # Use regular spacing for grid pattern
            params = self.lattice_pattern['params']
            
            # Place boundary to maintain regular spacing
            left_center = left_electrode['center_x']
            right_center = right_electrode['center_x']
            expected_boundary = left_center + (right_center - left_center) * \
                               (self.width_profile[left_idx] / 
                                (self.width_profile[left_idx] + self.width_profile[right_idx]))
            
            # Ensure minimum spacing
            boundary = max(left_electrode['max_x'] + self.min_spacing,
                          min(right_electrode['min_x'] - self.min_spacing,
                              int(expected_boundary)))
            
            return boundary
        else:
            # Fall back to parent method
            return super()._find_optimal_boundary(left_electrode, right_electrode,
                                                left_mask, right_mask, image_gradient)
    
    def generate_refined_masks(self, instance_masks: Dict[int, np.ndarray], 
                              image_gradient: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
        """Generate refined masks with graph-based consistency"""
        # First apply parent refinement (the lattice refiner's ElectrodeLatticeModel expects only 2 args)
        refined_masks = super().generate_refined_masks(instance_masks, image_gradient)
        
        if not self.use_graph_analysis or self.graph.mst is None:
            return refined_masks
        
        # Apply MST-based consistency propagation if enabled
        if self.use_mst_propagation:
            # Convert to numpy array format for MST consistency
            H, W = next(iter(instance_masks.values())).shape
            masks_array = np.zeros((H, W), dtype=np.int32)
            
            for inst_id, mask in refined_masks.items():
                strip_idx = self.instance_to_strip.get(inst_id, None)
                if strip_idx is not None:
                    masks_array[mask > 127] = strip_idx + 1
            
            # Apply MST consistency
            masks_array = self._apply_mst_consistency(masks_array)
            
            # Convert back to dict format
            for inst_id, strip_idx in self.instance_to_strip.items():
                if inst_id in refined_masks:
                    mask = (masks_array == strip_idx + 1).astype(np.uint8) * 255
                    refined_masks[inst_id] = mask
        
        return refined_masks
    
    def _apply_mst_consistency(self, masks: np.ndarray) -> np.ndarray:
        """Apply consistency constraints along MST"""
        if self.K < 3:
            return masks
        
        # Find root node (electrode with median area - likely good quality)
        areas = [self.graph.node_features[i]['area'] for i in range(self.K)]
        median_area = np.median(areas)
        root_idx = np.argmin([abs(a - median_area) for a in areas])
        
        # BFS traversal to propagate consistency
        visited = set()
        queue = [(root_idx, None)]
        refined_masks = masks.copy()
        
        while queue:
            node_idx, parent_idx = queue.pop(0)
            if node_idx in visited:
                continue
            visited.add(node_idx)
            
            electrode = self.electrodes[node_idx]
            
            if parent_idx is not None and self.neighbor_consistency_weight > 0:
                # Adjust based on parent
                parent_electrode = self.electrodes[parent_idx]
                parent_width = parent_electrode['width']
                current_width = electrode['width']
                
                # Check if this is a long electrode
                y_coords, x_coords = np.where(masks == electrode['id'])
                if len(y_coords) > 0:
                    height = np.max(y_coords) - np.min(y_coords) + 1
                    aspect_ratio = height / current_width if current_width > 0 else 1
                    is_long = aspect_ratio > 3
                else:
                    is_long = False
                
                # If widths are too different, apply gentle correction
                # Use different threshold for long electrodes
                threshold = 0.4 if is_long else 0.3
                if abs(current_width - parent_width) / parent_width > threshold:
                    # For long electrodes, preserve more of their width
                    consistency_weight = 0.1 if is_long else self.neighbor_consistency_weight
                    target_width = (consistency_weight * parent_width + 
                                  (1 - consistency_weight) * current_width)
                    
                    # Apply morphological operation to adjust width
                    scale = target_width / current_width
                    if scale > 1.05:  # Lower threshold to allow some thickening
                        kernel_size = min(int(3 * (scale - 1) + 3), 5)  # Allow slightly larger kernel
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, 3))
                        strip_mask = (masks == electrode['id']).astype(np.uint8)
                        dilated = cv2.dilate(strip_mask, kernel, iterations=1)
                        # Only apply if it doesn't cause significant overlap
                        overlap = np.sum((refined_masks > 0) & (dilated > 0) & (refined_masks != electrode['id']))
                        if overlap < 10:  # Allow tiny overlap
                            refined_masks[dilated > 0] = electrode['id']
                    elif scale < 0.95 and not is_long:  # Don't erode long electrodes
                        kernel_size = min(int(2 * (1 - scale) + 3), 3)  # Smaller kernel
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, 3))
                        strip_mask = (masks == electrode['id']).astype(np.uint8)
                        eroded = cv2.erode(strip_mask, kernel, iterations=1)
                        # Clear old mask and apply eroded
                        refined_masks[masks == electrode['id']] = 0
                        refined_masks[eroded > 0] = electrode['id']
            
            # Add children to queue
            children = self.graph.get_mst_children(node_idx)
            for child in children:
                if child not in visited:
                    queue.append((child, node_idx))
        
        return refined_masks


class GTShapeFeatureTransfer:
    """Transfer shape features from GT to prediction masks"""
    
    def __init__(self, min_separation: int = 5, 
                 transfer_mode: str = 'conservative'):
        """
        Initialize GT shape feature transfer
        
        Args:
            min_separation: Minimum separation between electrodes
            transfer_mode: Transfer strategy
                - 'conservative': Minimal changes, focus on quality
                - 'moderate': Balanced shape transfer
                - 'boundary_only': Only optimize boundaries
        """
        self.min_separation = min_separation
        self.debug_mode = False
        self.transfer_mode = transfer_mode
        
    def extract_shape_features(self, mask: np.ndarray) -> Dict:
        """Extract comprehensive shape features from mask"""
        features = {}
        
        # Get contour
        contours, _ = cv2.findContours(
            (mask > 127).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        
        if not contours:
            return features
            
        contour = contours[0]
        if len(contour.shape) == 3:
            contour = contour.squeeze(axis=1)  # Remove middle dimension
        
        # 1. Width profile along centerline
        features['width_profile'] = self._compute_width_profile(mask)
        
        # 2. Local curvature distribution
        features['curvature'] = self._compute_local_curvature(contour)
        
        # 3. Endpoint shapes (top and bottom regions)
        features['endpoints'] = self._extract_endpoint_shapes(mask)
        
        # 4. Edge smoothness
        features['smoothness'] = self._compute_edge_smoothness(contour)
        
        # 5. Skeleton for medial axis
        from skimage.morphology import skeletonize
        features['skeleton'] = skeletonize((mask > 127))
        
        # 6. Area and basic geometry
        features['area'] = np.sum(mask > 127)
        features['bbox'] = cv2.boundingRect(contour)
        features['contour'] = contour
        
        return features
    
    def transfer_features(self, pred_mask: np.ndarray, gt_mask: np.ndarray,
                         electrode_id: int = -1) -> np.ndarray:
        """Transfer GT shape features to prediction mask"""
        
        if self.debug_mode:
            print(f"\n[ShapeTransfer] Transferring shape features for electrode {electrode_id}")
        
        # Extract features from both masks
        pred_features = self.extract_shape_features(pred_mask)
        gt_features = self.extract_shape_features(gt_mask)
        
        if not pred_features or not gt_features:
            return pred_mask
        
        # Start with original mask
        refined = pred_mask.copy()
        
        if self.transfer_mode == 'boundary_only':
            # Only optimize boundaries without changing shape
            refined = self._optimize_boundaries(refined, pred_features, gt_features)
        else:
            # Apply shape transfer based on mode
            if self.transfer_mode in ['moderate', 'conservative']:
                # 1. Transfer endpoint shapes (only in moderate mode)
                if self.transfer_mode == 'moderate':
                    refined = self._transfer_endpoint_shape(refined, pred_features, gt_features)
                
                # 2. Transfer width pattern (conservative in both modes)
                refined = self._transfer_width_pattern(refined, pred_features, gt_features)
                
                # 3. Adjust local curvature (only if significant difference)
                refined = self._adjust_local_curvature(refined, pred_features, gt_features)
        
        # 4. Ensure separation from other electrodes (always)
        refined = self._ensure_separation(refined, electrode_id)
        
        # 5. Final pass: ensure minimum width throughout (always)
        refined = self._ensure_minimum_width(refined)
        
        return refined
    
    def _optimize_boundaries(self, mask: np.ndarray, 
                           pred_features: Dict, 
                           gt_features: Dict) -> np.ndarray:
        """Optimize boundaries without changing overall shape"""
        refined = mask.copy()
        
        # Apply gentle smoothing if GT is smoother
        pred_smoothness = pred_features.get('smoothness', 0)
        gt_smoothness = gt_features.get('smoothness', 0)
        
        if gt_smoothness > pred_smoothness * 1.2:
            # Very gentle smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            if self.debug_mode:
                print(f"  Applied gentle boundary smoothing")
        
        return refined
    
    def _ensure_minimum_width(self, mask: np.ndarray) -> np.ndarray:
        """Ensure minimum width is maintained throughout the electrode"""
        # Find electrode regions
        y_coords = np.where(np.any(mask > 127, axis=1))[0]
        if len(y_coords) == 0:
            return mask
        
        # Calculate statistics
        widths = []
        for y in y_coords:
            x_coords = np.where(mask[y] > 127)[0]
            if len(x_coords) > 0:
                width = np.max(x_coords) - np.min(x_coords) + 1
                widths.append(width)
        
        if not widths:
            return mask
            
        avg_width = np.mean(widths)
        min_width = max(3, int(avg_width * 0.2))  # At least 20% of average or 3 pixels
        
        if self.debug_mode:
            print(f"  Width protection: avg={avg_width:.1f}, min={min_width}")
        
        # Fix thin regions
        refined = mask.copy()
        thin_count = 0
        
        for y in y_coords:
            x_coords = np.where(mask[y] > 127)[0]
            if len(x_coords) > 0:
                width = np.max(x_coords) - np.min(x_coords) + 1
                if width < min_width:
                    center = (np.min(x_coords) + np.max(x_coords)) // 2
                    half_width = min_width // 2
                    new_x_min = max(0, center - half_width)
                    new_x_max = min(mask.shape[1]-1, new_x_min + min_width - 1)
                    refined[y, new_x_min:new_x_max+1] = 255
                    thin_count += 1
        
        if self.debug_mode and thin_count > 0:
            print(f"  Fixed {thin_count} thin regions")
        
        return refined
    
    def _compute_width_profile(self, mask: np.ndarray) -> np.ndarray:
        """Compute width at different heights along the electrode"""
        # Find y-coordinates with mask
        y_coords = np.where(mask > 127)[0]
        if len(y_coords) == 0:
            return np.array([])
            
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        height = y_max - y_min + 1
        
        # Sample at regular intervals
        n_samples = min(50, height // 10)
        if n_samples < 5:
            n_samples = min(5, height)
            
        width_profile = []
        sample_positions = np.linspace(y_min, y_max, n_samples, dtype=int)
        
        for y in sample_positions:
            row = mask[y, :]
            x_coords = np.where(row > 127)[0]
            if len(x_coords) > 0:
                width = np.max(x_coords) - np.min(x_coords) + 1
                width_profile.append(width)
            else:
                width_profile.append(0)
                
        return np.array(width_profile)
    
    def _compute_local_curvature(self, contour: np.ndarray) -> np.ndarray:
        """Compute local curvature along contour"""
        if len(contour) < 10:
            return np.array([])
            
        # Smooth contour first
        from scipy.ndimage import gaussian_filter1d
        smooth_x = gaussian_filter1d(contour[:, 0], sigma=3, mode='wrap')
        smooth_y = gaussian_filter1d(contour[:, 1], sigma=3, mode='wrap')
        smooth_contour = np.column_stack([smooth_x, smooth_y])
        
        # Compute curvature using finite differences
        dx = np.gradient(smooth_contour[:, 0])
        dy = np.gradient(smooth_contour[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        # Curvature formula: κ = |dx*d2y - dy*d2x| / (dx² + dy²)^(3/2)
        numerator = np.abs(dx * d2y - dy * d2x)
        denominator = np.power(dx**2 + dy**2, 1.5)
        
        # Avoid division by zero
        curvature = np.zeros_like(numerator)
        valid = denominator > 1e-6
        curvature[valid] = numerator[valid] / denominator[valid]
        
        return curvature
    
    def _extract_endpoint_shapes(self, mask: np.ndarray) -> Dict:
        """Extract shape characteristics of electrode endpoints"""
        y_coords = np.where(mask > 127)[0]
        if len(y_coords) == 0:
            return {}
            
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        height = y_max - y_min + 1
        
        # Define endpoint regions (top/bottom 20%)
        endpoint_size = max(5, int(height * 0.2))
        
        endpoints = {
            'top': mask[y_min:y_min+endpoint_size, :],
            'bottom': mask[y_max-endpoint_size+1:y_max+1, :],
            'top_range': (y_min, y_min+endpoint_size),
            'bottom_range': (y_max-endpoint_size+1, y_max+1)
        }
        
        return endpoints
    
    def _compute_edge_smoothness(self, contour: np.ndarray) -> float:
        """Compute edge smoothness metric"""
        if len(contour) < 3:
            return 0.0
            
        # Compute angle changes along contour
        angles = []
        for i in range(len(contour)):
            p1 = contour[i-1]
            p2 = contour[i]
            p3 = contour[(i+1) % len(contour)]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
                
        if angles:
            # Lower variance means smoother
            return 1.0 / (1.0 + np.var(angles))
        return 0.0
    
    def _transfer_endpoint_shape(self, mask: np.ndarray, 
                                pred_features: Dict, 
                                gt_features: Dict) -> np.ndarray:
        """Transfer endpoint shapes from GT to prediction with smooth transitions"""
        
        if 'endpoints' not in pred_features or 'endpoints' not in gt_features:
            return mask
            
        refined = mask.copy()
        
        # Transfer top endpoint with gradual transition
        if 'top' in gt_features['endpoints'] and 'top_range' in pred_features['endpoints']:
            top_y_start, top_y_end = pred_features['endpoints']['top_range']
            gt_top = gt_features['endpoints']['top']
            
            # Analyze GT top shape pattern
            gt_shape_profile = []
            for y in range(min(gt_top.shape[0], top_y_end - top_y_start)):
                x_coords = np.where(gt_top[y] > 127)[0]
                if len(x_coords) > 0:
                    width = np.max(x_coords) - np.min(x_coords) + 1
                    center = (np.min(x_coords) + np.max(x_coords)) / 2
                    gt_shape_profile.append({'width': width, 'center': center, 'valid': True})
                else:
                    gt_shape_profile.append({'valid': False})
            
            # Apply shape with gradual blending
            transition_zone = min(5, len(gt_shape_profile) // 3)  # Transition over 5 pixels or 1/3 of endpoint
            
            for i, y in enumerate(range(top_y_start, min(top_y_end, refined.shape[0]))):
                if i >= len(gt_shape_profile):
                    break
                    
                x_coords = np.where(refined[y] > 127)[0]
                if len(x_coords) > 0 and gt_shape_profile[i]['valid']:
                    current_width = np.max(x_coords) - np.min(x_coords) + 1
                    current_center = (np.min(x_coords) + np.max(x_coords)) / 2
                    
                    # Calculate blending factor (0 to 1)
                    if i < transition_zone:
                        blend_factor = i / transition_zone
                    else:
                        blend_factor = 1.0
                    
                    # Blend GT shape with current shape (more aggressive)
                    target_width = current_width * (1 - blend_factor * 0.5) + gt_shape_profile[i]['width'] * (blend_factor * 0.5)
                    target_width = int(np.clip(target_width, current_width * 0.7, current_width * 1.3))
                    
                    # Apply refined width
                    new_half_width = target_width // 2
                    new_x_min = max(0, int(current_center - new_half_width))
                    new_x_max = min(refined.shape[1]-1, int(current_center + new_half_width))
                    
                    refined[y, :] = 0
                    refined[y, new_x_min:new_x_max+1] = 255
        
        # Transfer bottom endpoint with similar approach
        if 'bottom' in gt_features['endpoints'] and 'bottom_range' in pred_features['endpoints']:
            bottom_y_start, bottom_y_end = pred_features['endpoints']['bottom_range']
            gt_bottom = gt_features['endpoints']['bottom']
            
            # Analyze GT bottom shape (from bottom up)
            gt_shape_profile = []
            for y in range(min(gt_bottom.shape[0], bottom_y_end - bottom_y_start)):
                # Read from bottom of GT shape
                gt_y = gt_bottom.shape[0] - 1 - y
                if gt_y >= 0:
                    x_coords = np.where(gt_bottom[gt_y] > 127)[0]
                    if len(x_coords) > 0:
                        width = np.max(x_coords) - np.min(x_coords) + 1
                        center = (np.min(x_coords) + np.max(x_coords)) / 2
                        gt_shape_profile.append({'width': width, 'center': center, 'valid': True})
                    else:
                        gt_shape_profile.append({'valid': False})
            
            # Apply from bottom up
            transition_zone = min(5, len(gt_shape_profile) // 3)
            
            for i, y in enumerate(reversed(range(max(bottom_y_start, 0), bottom_y_end))):
                if i >= len(gt_shape_profile):
                    break
                    
                x_coords = np.where(refined[y] > 127)[0]
                if len(x_coords) > 0 and gt_shape_profile[i]['valid']:
                    current_width = np.max(x_coords) - np.min(x_coords) + 1
                    current_center = (np.min(x_coords) + np.max(x_coords)) / 2
                    
                    # Calculate blending factor
                    if i < transition_zone:
                        blend_factor = i / transition_zone
                    else:
                        blend_factor = 1.0
                    
                    # More aggressive blending
                    target_width = current_width * (1 - blend_factor * 0.5) + gt_shape_profile[i]['width'] * (blend_factor * 0.5)
                    target_width = int(np.clip(target_width, current_width * 0.7, current_width * 1.3))
                    
                    # Apply refined width
                    new_half_width = target_width // 2
                    new_x_min = max(0, int(current_center - new_half_width))
                    new_x_max = min(refined.shape[1]-1, int(current_center + new_half_width))
                    
                    refined[y, :] = 0
                    refined[y, new_x_min:new_x_max+1] = 255
        
        return refined
    
    def _transfer_width_pattern(self, mask: np.ndarray,
                               pred_features: Dict,
                               gt_features: Dict) -> np.ndarray:
        """Transfer width variation pattern from GT"""
        
        pred_profile = pred_features.get('width_profile', np.array([]))
        gt_profile = gt_features.get('width_profile', np.array([]))
        
        if len(pred_profile) == 0 or len(gt_profile) == 0:
            return mask
            
        refined = mask.copy()
        
        # Compute average width ratio
        pred_avg_width = np.mean(pred_profile[pred_profile > 0])
        gt_avg_width = np.mean(gt_profile[gt_profile > 0])
        
        if pred_avg_width > 0:
            # Conservative scaling factor
            scale_factor = gt_avg_width / pred_avg_width
            scale_factor = np.clip(scale_factor, 0.9, 1.1)  # Conservative: ±10%
            
            if self.debug_mode:
                print(f"  Width scale factor: {scale_factor:.2f}")
            
            # Apply scaling along the electrode with smooth transitions
            y_coords = np.where(mask > 127)[0]
            if len(y_coords) > 0:
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                
                # First pass: collect target widths
                target_widths = []
                centers = []
                
                for y in range(y_min, y_max+1):
                    x_coords = np.where(refined[y] > 127)[0]
                    if len(x_coords) > 0:
                        center_x = (np.min(x_coords) + np.max(x_coords)) / 2
                        current_width = np.max(x_coords) - np.min(x_coords) + 1
                        
                        # Interpolate scale factor based on position
                        rel_pos = (y - y_min) / (y_max - y_min) if y_max > y_min else 0
                        
                        # Get local scale from GT profile
                        gt_idx = int(rel_pos * (len(gt_profile) - 1))
                        pred_idx = int(rel_pos * (len(pred_profile) - 1))
                        
                        if pred_profile[pred_idx] > 0 and gt_profile[gt_idx] > 0:
                            local_scale = gt_profile[gt_idx] / pred_profile[pred_idx]
                            # Conservative scaling to avoid breaking connectivity
                            local_scale = np.clip(local_scale, 0.9, 1.1)
                        else:
                            local_scale = 1.0
                        
                        target_widths.append(current_width * local_scale)
                        centers.append(center_x)
                    else:
                        target_widths.append(0)
                        centers.append(0)
                
                # Smooth target widths to avoid sudden changes
                if len(target_widths) > 5:
                    from scipy.ndimage import gaussian_filter1d
                    valid_mask = np.array(target_widths) > 0
                    if np.sum(valid_mask) > 3:
                        # Only smooth valid widths
                        smoothed_widths = gaussian_filter1d(target_widths, sigma=2)
                        # Preserve original zeros
                        smoothed_widths[~valid_mask] = 0
                        target_widths = smoothed_widths
                
                # Calculate minimum width based on electrode characteristics
                min_width = 3  # Absolute minimum width in pixels
                if len(target_widths) > 0:
                    avg_width = np.mean([w for w in target_widths if w > 0])
                    min_width = max(3, int(avg_width * 0.25))  # At least 25% of average width
                
                # Second pass: apply smoothed widths with minimum width protection
                for i, y in enumerate(range(y_min, y_max+1)):
                    if target_widths[i] > 0 and centers[i] > 0:
                        # Enforce minimum width
                        protected_width = max(target_widths[i], min_width)
                        new_half_width = int(protected_width / 2)
                        
                        new_x_min = int(centers[i] - new_half_width)
                        new_x_max = int(centers[i] + new_half_width)
                        
                        # Ensure within bounds
                        new_x_min = max(0, new_x_min)
                        new_x_max = min(refined.shape[1]-1, new_x_max)
                        
                        # Ensure minimum width is maintained
                        if new_x_max - new_x_min + 1 < min_width:
                            # Adjust to ensure minimum width
                            center = (new_x_min + new_x_max) // 2
                            half_min = min_width // 2
                            new_x_min = max(0, center - half_min)
                            new_x_max = min(refined.shape[1]-1, new_x_min + min_width - 1)
                        
                        # Update mask
                        refined[y, :] = 0
                        refined[y, new_x_min:new_x_max+1] = 255
        
        return refined
    
    def _adjust_local_curvature(self, mask: np.ndarray,
                               pred_features: Dict,
                               gt_features: Dict) -> np.ndarray:
        """Adjust local curvature to match GT pattern using active contour-like approach"""
        
        pred_curvature = pred_features.get('curvature', np.array([]))
        gt_curvature = gt_features.get('curvature', np.array([]))
        
        if len(pred_curvature) == 0 or len(gt_curvature) == 0:
            return mask
            
        # Get contour for refinement
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return mask
            
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) < 10:
            return mask
            
        refined = mask.copy()
        
        # Analyze curvature patterns
        pred_smoothness = pred_features.get('smoothness', 0)
        gt_smoothness = gt_features.get('smoothness', 0)
        
        # Match curvature distribution
        if len(pred_curvature) == len(gt_curvature):
            # Compute curvature difference
            curvature_diff = gt_curvature - pred_curvature
            
            # Find high curvature regions in GT that need enhancement
            high_curv_threshold = np.percentile(np.abs(gt_curvature), 80)
            high_curv_regions = np.where(np.abs(gt_curvature) > high_curv_threshold)[0]
            
            if len(high_curv_regions) > 0 and self.debug_mode:
                print(f"  Found {len(high_curv_regions)} high curvature points to enhance")
            
            # Apply targeted morphological operations
            if gt_smoothness > pred_smoothness * 1.1:
                # GT is smoother - apply smoothing
                kernel_size = 3 if gt_smoothness > pred_smoothness * 1.3 else 2
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                
                # Progressive smoothing with more iterations
                refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=2)
                refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=2)
                
                if self.debug_mode:
                    print(f"  Applied smoothing with kernel size {kernel_size}")
                    
            elif pred_smoothness > gt_smoothness * 1.05:  # Lower threshold for enhancement
                # Prediction is too smooth - enhance features
                # Use gradient-based enhancement
                gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, 
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
                
                # Enhance regions with high GT curvature
                for idx in high_curv_regions:
                    # Map contour index to image coordinates
                    if idx < len(largest_contour):
                        pt = largest_contour[idx][0]
                        x, y = pt[0], pt[1]
                        
                        # Local enhancement based on curvature sign
                        # Check local width before modification
                        local_y_range = range(max(0, y-3), min(refined.shape[0], y+4))
                        local_widths = []
                        for ly in local_y_range:
                            x_coords = np.where(refined[ly] > 127)[0]
                            if len(x_coords) > 0:
                                local_widths.append(np.max(x_coords) - np.min(x_coords) + 1)
                        
                        if local_widths:
                            avg_local_width = np.mean(local_widths)
                            min_safe_width = max(3, int(avg_local_width * 0.4))
                            
                            if gt_curvature[idx] > 0:  # Convex region
                                # Slight dilation to enhance convexity
                                cv2.circle(refined, (x, y), 1, 255, -1)
                            elif min(local_widths) > min_safe_width:  # Concave region - only if safe
                                # Slight erosion to enhance concavity
                                cv2.circle(refined, (x, y), 1, 0, -1)
                
                if self.debug_mode:
                    print(f"  Enhanced curvature at {len(high_curv_regions)} points")
        
        # Ensure result is still connected and valid
        refined = self._ensure_connectivity(refined, mask)
        
        return refined
    
    def _ensure_connectivity(self, refined: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Ensure refined mask maintains connectivity"""
        # Find connected components
        num_labels, labels = cv2.connectedComponents(refined.astype(np.uint8))
        
        if num_labels <= 2:  # Only background and one component
            return refined
            
        # Find the largest component (excluding background)
        component_sizes = []
        for i in range(1, num_labels):
            size = np.sum(labels == i)
            component_sizes.append((size, i))
        
        if not component_sizes:
            return original
            
        # Keep only the largest component
        largest_size, largest_label = max(component_sizes)
        result = np.zeros_like(refined)
        result[labels == largest_label] = 255
        
        # If we lost too much area, return original
        if np.sum(result > 0) < 0.7 * np.sum(original > 0):
            return original
            
        return result
    
    def _ensure_separation(self, mask: np.ndarray, electrode_id: int) -> np.ndarray:
        """Ensure minimum separation from other electrodes"""
        # This would need access to other electrode masks
        # For now, just apply conservative erosion at boundaries
        
        # Detect thin regions that might cause adhesion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(mask, kernel, iterations=1)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        
        # Find potentially problematic regions
        diff = dilated - mask
        
        # If there are many boundary pixels, apply slight erosion
        if np.sum(diff > 0) > 100:
            return eroded
            
        return mask


class ShapeTransferRefiner(GuidedRefiner):
    """
    Deep GT shape feature transfer
    
    Inherits from the guided refiner and adds GT shape feature transfer on top of the graph-enhanced lattice model.
    Key additions:
    - Shape feature extraction (width profile, curvature, endpoints)
    - Feature-wise transfer from GT to predictions
    - Separation-aware shape adjustment
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize parent (the guided refiner)
        super().__init__(*args, **kwargs)
        
        # Replace the lattice model (inherited from the lattice refiner) with graph-enhanced version
        self.lattice_model = GraphEnhancedElectrodeLatticeModel(
            min_spacing=self.min_separation_pixels,
            max_width_ratio=1.8,  # Increased from 1.5 to allow thicker electrodes
            width_consistency_weight=0.15,  # Slightly increased from 0.1
            boundary_smoothness_weight=0.1,  # Keep low to prevent adhesion
            use_soft_boundaries=True,
            edge_smoothing_iterations=0,  # Keep disabled to prevent adhesion
            vertical_continuity_weight=0.15,  # Slightly increased from 0.1
            use_shape_regularization=True,  # Re-enabled with care
            use_skeleton_guidance=True,
            # Graph-specific parameters
            use_graph_analysis=True,
            graph_distance_threshold=80.0,  # Further reduced to only immediate neighbors
            use_mst_propagation=True,
            neighbor_consistency_weight=0.15  # Reduced from 0.2 for gentler corrections
        )
        
        # GT shape transfer component
        # Default to conservative mode to prevent connectivity issues
        self.shape_transfer = GTShapeFeatureTransfer(
            min_separation=self.min_separation_pixels,
            transfer_mode='conservative'  # Options: 'conservative', 'moderate', 'boundary_only'
        )
        self._gt_masks_cache = None
        
        print("\n[ShapeTransfer] Initialized with GT shape feature transfer")
        print("        - Graph-enhanced electrode lattice model (graph analysis + MST consistency)")
        print("        - Deep shape feature extraction and transfer")
        print("        - Width profile and curvature matching")
        print("        (Inheritance: ShapeTransferRefiner → GuidedRefiner → PatternRefiner → PriorIntegrationRefiner → LatticeRefiner → BayesianRefinerBase)")


    def refine_unified_region(self, region_data: Dict) -> Dict[int, np.ndarray]:
        """Override to add GT shape transfer after base refinement"""
        
        # Load GT masks if available
        gt_masks = self._load_gt_masks(region_data)
        if gt_masks:
            print(f"[ShapeTransfer] Loaded GT masks for {len(gt_masks)} electrodes")
            self._gt_masks_cache = gt_masks
            # Pass to lattice model
            self.lattice_model.gt_masks = gt_masks
            
            # Enable debug mode for shape transfer
            self.shape_transfer.debug_mode = self.debug_mode
        
        # Get initial refinement from parent (graph-enhanced lattice behavior)
        refined_masks = super().refine_unified_region(region_data)
        
        # Apply GT shape feature transfer if GT is available
        if gt_masks:
            print("\n[ShapeTransfer] Applying GT shape feature transfer...")
            
            shape_refined_masks = {}
            for inst_id, mask in refined_masks.items():
                if inst_id in gt_masks:
                    # Transfer shape features from GT
                    shape_refined = self.shape_transfer.transfer_features(
                        mask, gt_masks[inst_id], inst_id
                    )
                    shape_refined_masks[inst_id] = shape_refined
                    
                    # Debug: compare areas
                    if self.debug_mode:
                        orig_area = np.sum(mask > 127)
                        refined_area = np.sum(shape_refined > 127)
                        gt_area = np.sum(gt_masks[inst_id] > 127)
                        print(f"  Electrode {inst_id}: {orig_area} -> {refined_area} (GT: {gt_area})")
                else:
                    shape_refined_masks[inst_id] = mask
            
            return shape_refined_masks
        
        return refined_masks


def main():
    """Test the shape-transfer refiner with GT shape feature transfer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Shape-transfer refinement (GT diagnostic regime)')
    parser.add_argument('--data-dir', type=str, 
                       default='data/Full_Instances',
                       help='Base directory containing instance data')
    parser.add_argument('--output-dir', type=str,
                       default='results/refined_masks_shape_transfer',
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
                       default='results/shape_transfer_visualizations',
                       help='Directory for saving visualization results')
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
    
    # Initialize the shape-transfer refiner: guided refiner + graph-enhanced lattice + GT shape transfer
    refiner = ShapeTransferRefiner(
        instances_per_region=50,
        pyramid_levels=1,
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
    
    # Process instances (same as the guided refiner)
    os.makedirs(args.output_dir, exist_ok=True)
    
    instance_info_dir = os.path.join(args.data_dir, 'instance_info')
    enhanced_image_dir = os.path.join(args.origin_dir, 'images_enhanced')
    mask_dir = os.path.join(args.data_dir, 'repaired_masks')
    
    # Get instance info files
    info_files = sorted([f for f in os.listdir(instance_info_dir) if f.endswith('_info.json')])
    
    if args.test_image:
        info_files = [f for f in info_files if args.test_image in f]
    
    print(f"Processing {len(info_files)} images with the shape-transfer refiner...")
    
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
            
            # Refine using the shape-transfer refiner (graph-enhanced lattice model + GT shape transfer)
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
    
    print("\nShape-transfer refinement complete!")


if __name__ == "__main__":
    main()