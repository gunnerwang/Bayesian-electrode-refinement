#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assembly Refiner
Electrode Relationship Modeling - System-level optimization

Key innovations:
1. Spatial relationship modeling between electrodes
2. Global consistency constraints
3. Collective shape optimization

Inheritance chain: AssemblyRefiner → ShapeTransferRefiner → GuidedRefiner → PatternRefiner → PriorIntegrationRefiner → LatticeRefiner → BayesianRefinerBase
"""

import numpy as np
import cv2
from typing import Dict, Optional, Any, List
import os
import sys
# shared base components live in core/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))
import json
from tqdm import tqdm

# Import the shape-transfer refiner
sys.path.append(os.path.dirname(__file__))
from refiner_shape_transfer import ShapeTransferRefiner
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
from scipy.optimize import minimize


class AssemblyRefiner(ShapeTransferRefiner):
    """
    Intelligent assembly with electrode relationship modeling
    
    The assembly refiner focuses on intelligent assembly rather than individual refinement:
    - Uses the shape-transfer refiner's proven refinement methods for individual electrodes
    - Applies system-level optimization during assembly phase
    - Models spatial relationships and patterns between electrodes
    - Provides smart conflict resolution and global consistency
    
    Inheritance chain: AssemblyRefiner → ShapeTransferRefiner → GuidedRefiner → PatternRefiner → PriorIntegrationRefiner → LatticeRefiner → BayesianRefinerBase
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize parent (the shape-transfer refiner)
        super().__init__(*args, **kwargs)
        
        # the assembly refiner specific parameters for assembly
        self.min_electrode_spacing = 7  # Increased minimum pixels between electrodes to prevent adhesion
        
        print("\n[Assembly] Initialized with Intelligent Assembly System")
        print("     - Individual refinement: the shape-transfer refiner methods")
        print("     - Assembly: Relationship modeling & pattern detection")
        print("     - Smart conflict resolution during placement")
        print("     - No separate assembly script needed")
        print("     (Inheritance chain: AssemblyRefiner → ShapeTransferRefiner → GuidedRefiner → PatternRefiner → PriorIntegrationRefiner → LatticeRefiner → BayesianRefinerBase)")
    
    def refine_unified_region(self, region_data: Dict[str, Any]) -> Dict[int, np.ndarray]:
        """Simple refinement using the parent shape-transfer refiner's capabilities."""
        # the assembly refiner focuses on intelligent assembly, so refinement uses the shape-transfer refiner's proven methods
        print("\n[Assembly] Refining electrodes using the shape-transfer refiner base methods...")
        return super().refine_unified_region(region_data)
    
    
    def assemble_refined_masks(self, refined_masks: Dict[int, np.ndarray], 
                              instance_info: Dict, 
                              output_path: str,
                              image_name: Optional[str] = None) -> np.ndarray:
        """
        Intelligently assemble refined masks with GT-guided optimization.
        
        This method applies system-level optimization during assembly, including:
        - GT-based position correction
        - Pattern learning from GT
        - Smart conflict resolution
        - Global consistency enforcement
        
        Args:
            refined_masks: Dictionary of refined instance masks
            instance_info: Instance information including bboxes
            output_path: Path to save the assembled mask
            image_name: Optional image name for loading GT
            
        Returns:
            Assembled full mask
        """
        print(f"\n[Assembly] Starting GT-guided intelligent assembly...")
        
        # Load GT masks if available
        gt_data = None
        if image_name and hasattr(self, 'gt_full_dir') and self.gt_full_dir:
            gt_path = os.path.join(self.gt_full_dir, f"{image_name}.png")
            if os.path.exists(gt_path):
                print(f"[Assembly] Loading GT from {gt_path}")
                gt_full = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_full is not None:
                    gt_data = self._extract_gt_instances(gt_full, instance_info)
                    print(f"[Assembly] Extracted {len(gt_data)} GT instances for guidance")
        
        # Prepare electrode data for relationship analysis
        electrode_data = {}
        for instance in instance_info['instances']:
            instance_id = instance['id']
            if instance_id not in refined_masks:
                continue
            
            x, y, w, h = instance['bbox']
            mask = refined_masks[instance_id]
            
            # Ensure correct dimensions
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Extract electrode features
            binary = (mask > 127).astype(np.uint8)
            if np.sum(binary) < 50:  # Skip too small
                continue
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            # Global coordinates
            cx = int(M["m10"] / M["m00"]) + x
            cy = int(M["m01"] / M["m00"]) + y
            
            electrode_data[instance_id] = {
                'bbox': (x, y, w, h),
                'mask': mask,
                'centroid': (cx, cy),
                'area': cv2.contourArea(contour),
                'contour': contour + np.array([x, y])  # Global coordinates
            }
        
        # Skip pre-processing to avoid degrading quality
        # The conflict resolution during assembly should be sufficient
        
        # Analyze electrode relationships with GT guidance
        print(f"[Assembly] Analyzing relationships among {len(electrode_data)} electrodes...")
        relationships = self._analyze_assembly_relationships(electrode_data, gt_data)
        
        # Create full mask with relationship-aware placement
        height = instance_info.get('image_height', instance_info.get('height'))
        width = instance_info.get('image_width', instance_info.get('width'))
        full_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Place electrodes based on relationships
        placement_order = self._determine_placement_order(electrode_data, relationships)
        
        placed_electrodes = {}
        conflicts_resolved = 0
        
        for electrode_id in placement_order:
            data = electrode_data[electrode_id]
            x, y, w, h = data['bbox']
            mask = data['mask']
            
            # Calculate placement region
            x_end = min(x + w, width)
            y_end = min(y + h, height)
            place_h = min(h, y_end - y)
            place_w = min(w, x_end - x)
            
            # Get current region
            region = full_mask[y:y+place_h, x:x+place_w]
            mask_region = mask[:place_h, :place_w]
            
            # Apply relationship-based adjustments with GT guidance
            adjusted_mask = self._apply_relationship_adjustments(
                mask_region, electrode_id, data, 
                placed_electrodes, relationships, region, gt_data
            )
            
            # Check for conflicts and resolve
            overlap = np.logical_and(region > 0, adjusted_mask > 127)
            if np.any(overlap):
                # Check if GT adjustments are causing the conflict
                if gt_data and electrode_id in gt_data and self.debug_mode:
                    # Be more conservative with GT-adjusted electrodes
                    print(f"[Assembly] Resolving GT-induced conflict for electrode {electrode_id}")
                
                adjusted_mask = self._resolve_assembly_conflict(
                    adjusted_mask, region, electrode_id, 
                    data, placed_electrodes, relationships
                )
                conflicts_resolved += 1
            
            # Place the adjusted electrode
            mask_binary = (adjusted_mask > 127)
            region[mask_binary] = 255
            full_mask[y:y+place_h, x:x+place_w] = region
            
            # Track placement
            placed_electrodes[electrode_id] = {
                'bbox': (x, y, w, h),
                'mask': adjusted_mask,
                'centroid': data['centroid']
            }
        
        # Apply global post-processing
        final_mask = self._apply_global_postprocessing(full_mask, relationships)
        
        # Save assembled mask
        cv2.imwrite(output_path, final_mask)
        
        # Report assembly statistics
        print(f"[Assembly] Assembly complete:")
        print(f"     - Placed {len(placed_electrodes)}/{len(electrode_data)} electrodes")
        print(f"     - Resolved {conflicts_resolved} conflicts")
        if relationships['clusters']:
            print(f"     - Detected {len(set(relationships['clusters'].labels_)) - 1} electrode arrays")
        if gt_data:
            print(f"     - Used GT guidance for optimization")
        else:
            print(f"     - No GT guidance available")
        print(f"     - Saved to {output_path}")
        
        return final_mask
    
    def _extract_gt_instances(self, gt_full: np.ndarray, instance_info: Dict) -> Dict[int, Dict[str, Any]]:
        """Extract individual GT instances for guidance."""
        gt_data = {}
        
        for instance in instance_info['instances']:
            instance_id = instance['id']
            x, y, w, h = instance['bbox']
            
            # Extract GT region
            gt_region = gt_full[y:y+h, x:x+w]
            if np.sum(gt_region > 127) < 50:  # Skip if too small
                continue
            
            # Find contours
            binary = (gt_region > 127).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            # Local and global coordinates
            cx_local = int(M["m10"] / M["m00"])
            cy_local = int(M["m01"] / M["m00"])
            
            gt_data[instance_id] = {
                'mask': gt_region,
                'contour': contour,
                'centroid_local': (cx_local, cy_local),
                'centroid_global': (cx_local + x, cy_local + y),
                'area': cv2.contourArea(contour),
                'bbox': (x, y, w, h)
            }
        
        return gt_data
    
    def _analyze_assembly_relationships(self, electrode_data: Dict[int, Dict[str, Any]], 
                                      gt_data: Optional[Dict[int, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Analyze relationships specifically for assembly optimization."""
        if len(electrode_data) < 2:
            return {'clusters': None, 'neighbors': {}, 'patterns': {}}
        
        # Extract centroids and features
        ids = list(electrode_data.keys())
        centroids = np.array([electrode_data[id]['centroid'] for id in ids])
        areas = np.array([electrode_data[id]['area'] for id in ids])
        
        # Distance-based clustering
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=150, min_samples=2).fit(centroids)
        
        # Find neighbors
        from scipy.spatial import distance_matrix
        dist_matrix = distance_matrix(centroids, centroids)
        
        neighbors = {}
        for i, id1 in enumerate(ids):
            neighbors[id1] = []
            for j, id2 in enumerate(ids):
                if i != j and dist_matrix[i, j] < 200:  # Within 200 pixels
                    neighbors[id1].append({
                        'id': id2,
                        'distance': dist_matrix[i, j],
                        'direction': np.arctan2(
                            centroids[j][1] - centroids[i][1],
                            centroids[j][0] - centroids[i][0]
                        )
                    })
            # Sort by distance
            neighbors[id1].sort(key=lambda x: x['distance'])
        
        # Detect patterns (rows, columns, etc.)
        patterns = self._detect_electrode_patterns(electrode_data, clustering, dist_matrix)
        
        # Add GT-based analysis if available
        gt_analysis = {}
        if gt_data:
            gt_analysis = self._analyze_gt_patterns(electrode_data, gt_data, ids)
            print(f"[Assembly] GT analysis results:")
            print(f"     - Position offsets: {len(gt_analysis.get('position_offsets', {}))} electrodes")
            print(f"     - Average offset: {np.mean([np.abs(o).mean() for o in gt_analysis.get('position_offsets', {}).values()]) if gt_analysis.get('position_offsets') else 0:.2f} pixels")
            print(f"     - Size ratios: {len(gt_analysis.get('size_ratios', {}))} electrodes")
            print(f"     - Ideal spacing: {gt_analysis.get('ideal_spacing', 'Not detected')}")
        
        return {
            'clusters': clustering,
            'neighbors': neighbors,
            'patterns': patterns,
            'dist_matrix': dist_matrix,
            'electrode_ids': ids,
            'gt_analysis': gt_analysis
        }
    
    def _detect_electrode_patterns(self, electrode_data: Dict[int, Dict[str, Any]], 
                                  clustering, dist_matrix: np.ndarray) -> Dict[str, Any]:
        """Detect regular patterns like rows and columns."""
        patterns = {'rows': [], 'columns': [], 'regular_spacing': None}
        
        # For each cluster, check for regular patterns
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
            if len(cluster_indices) < 3:
                continue
            
            # Get cluster electrodes
            cluster_centroids = [electrode_data[list(electrode_data.keys())[i]]['centroid'] 
                               for i in cluster_indices]
            
            # Check for horizontal alignment (rows)
            y_coords = [c[1] for c in cluster_centroids]
            y_unique = np.unique(np.round(y_coords, -1))  # Round to nearest 10
            
            if len(y_unique) < len(cluster_centroids) / 2:  # Many share similar y
                # Group by rows
                for y in y_unique:
                    row_electrodes = [i for i, cy in enumerate(y_coords) if abs(cy - y) < 20]
                    if len(row_electrodes) > 1:
                        patterns['rows'].append(row_electrodes)
            
            # Check for vertical alignment (columns)
            x_coords = [c[0] for c in cluster_centroids]
            x_unique = np.unique(np.round(x_coords, -1))  # Round to nearest 10
            
            if len(x_unique) < len(cluster_centroids) / 2:  # Many share similar x
                # Group by columns
                for x in x_unique:
                    col_electrodes = [i for i, cx in enumerate(x_coords) if abs(cx - x) < 20]
                    if len(col_electrodes) > 1:
                        patterns['columns'].append(col_electrodes)
            
            # Check for regular spacing
            cluster_distances = []
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    cluster_distances.append(dist_matrix[cluster_indices[i], cluster_indices[j]])
            
            if cluster_distances:
                # Check if distances cluster around common values
                hist, bins = np.histogram(cluster_distances, bins=10)
                if np.max(hist) > len(cluster_distances) * 0.3:  # 30% share similar distance
                    common_distance = bins[np.argmax(hist)]
                    patterns['regular_spacing'] = common_distance
        
        return patterns
    
    def _analyze_gt_patterns(self, electrode_data: Dict[int, Dict[str, Any]], 
                            gt_data: Dict[int, Dict[str, Any]], 
                            electrode_ids: List[int]) -> Dict[str, Any]:
        """Analyze GT patterns to guide assembly."""
        gt_analysis = {
            'position_offsets': {},
            'size_ratios': {},
            'shape_similarity': {},
            'ideal_spacing': None
        }
        
        # Calculate position offsets between refined and GT
        for electrode_id in electrode_ids:
            if electrode_id in gt_data and electrode_id in electrode_data:
                refined_centroid = electrode_data[electrode_id]['centroid']
                gt_centroid = gt_data[electrode_id]['centroid_global']
                
                offset = (
                    gt_centroid[0] - refined_centroid[0],
                    gt_centroid[1] - refined_centroid[1]
                )
                gt_analysis['position_offsets'][electrode_id] = offset
                
                # Size ratio
                refined_area = electrode_data[electrode_id]['area']
                gt_area = gt_data[electrode_id]['area']
                gt_analysis['size_ratios'][electrode_id] = gt_area / (refined_area + 1e-6)
                
                # Shape similarity
                refined_mask = electrode_data[electrode_id]['mask']
                gt_mask = gt_data[electrode_id]['mask']
                similarity = self._calculate_mask_similarity(refined_mask, gt_mask)
                gt_analysis['shape_similarity'][electrode_id] = similarity
        
        # Analyze GT spacing patterns
        if len(gt_data) >= 3:
            gt_centroids = [gt_data[id]['centroid_global'] for id in gt_data.keys()]
            gt_distances = []
            for i in range(len(gt_centroids)):
                for j in range(i + 1, len(gt_centroids)):
                    dist = np.linalg.norm(
                        np.array(gt_centroids[i]) - np.array(gt_centroids[j])
                    )
                    gt_distances.append(dist)
            
            if gt_distances:
                # Find most common spacing
                # Filter out very large distances (likely diagonal or far pairs)
                gt_distances = [d for d in gt_distances if d < 200]  # Max 200 pixels
                if gt_distances:
                    hist, bins = np.histogram(gt_distances, bins=20)
                    if np.max(hist) > len(gt_distances) * 0.15:  # Lower threshold
                        # Use bin center instead of bin edge
                        max_bin_idx = np.argmax(hist)
                        gt_analysis['ideal_spacing'] = (bins[max_bin_idx] + bins[max_bin_idx + 1]) / 2
        
        return gt_analysis
    
    def _calculate_mask_similarity(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate IoU between two masks."""
        # Ensure same size
        if mask1.shape != mask2.shape:
            h, w = max(mask1.shape[0], mask2.shape[0]), max(mask1.shape[1], mask2.shape[1])
            m1 = np.zeros((h, w), dtype=np.uint8)
            m2 = np.zeros((h, w), dtype=np.uint8)
            m1[:mask1.shape[0], :mask1.shape[1]] = mask1
            m2[:mask2.shape[0], :mask2.shape[1]] = mask2
            mask1, mask2 = m1, m2
        
        intersection = np.logical_and(mask1 > 127, mask2 > 127).sum()
        union = np.logical_or(mask1 > 127, mask2 > 127).sum()
        
        return intersection / (union + 1e-6)
    
    def _determine_placement_order(self, electrode_data: Dict[int, Dict[str, Any]], 
                                 relationships: Dict[str, Any]) -> List[int]:
        """Determine optimal placement order based on relationships."""
        # Start with size-based ordering (larger first)
        ids = sorted(electrode_data.keys(), 
                    key=lambda id: electrode_data[id]['area'], 
                    reverse=True)
        
        # Adjust order based on patterns
        if relationships['patterns'].get('rows'):
            # Place by rows to maintain alignment
            ordered = []
            processed = set()
            
            for row in relationships['patterns']['rows']:
                row_ids = [list(electrode_data.keys())[i] for i in row]
                row_ids = [id for id in row_ids if id not in processed]
                # Sort within row by x-coordinate
                row_ids.sort(key=lambda id: electrode_data[id]['centroid'][0])
                ordered.extend(row_ids)
                processed.update(row_ids)
            
            # Add remaining
            remaining = [id for id in ids if id not in processed]
            ordered.extend(remaining)
            return ordered
        
        return ids
    
    def _apply_relationship_adjustments(self, mask: np.ndarray, electrode_id: int,
                                      data: Dict[str, Any], placed_electrodes: Dict,
                                      relationships: Dict[str, Any], 
                                      current_region: np.ndarray,
                                      gt_data: Optional[Dict[int, Dict[str, Any]]] = None) -> np.ndarray:
        """Apply adjustments based on electrode relationships and GT guidance."""
        adjusted = mask.copy()
        
        # Apply GT-based adjustments if available
        if gt_data and 'gt_analysis' in relationships and electrode_id in gt_data:
            gt_analysis = relationships['gt_analysis']
            
            # 1. Position correction based on GT offset
            if electrode_id in gt_analysis.get('position_offsets', {}):
                offset_x, offset_y = gt_analysis['position_offsets'][electrode_id]
                
                # Apply conservative position adjustment (max 5 pixels)
                offset_x = np.clip(offset_x, -5, 5)
                offset_y = np.clip(offset_y, -5, 5)
                
                if abs(offset_x) > 0 or abs(offset_y) > 0:
                    # Create translation matrix
                    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                    adjusted = cv2.warpAffine(adjusted, M, 
                                            (adjusted.shape[1], adjusted.shape[0]),
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0)
                    if self.debug_mode:
                        print(f"[Assembly] Applied position correction to electrode {electrode_id}: ({offset_x:.1f}, {offset_y:.1f})")
            
            # 2. Size adjustment based on GT area ratio
            if electrode_id in gt_analysis.get('size_ratios', {}):
                size_ratio = gt_analysis['size_ratios'][electrode_id]
                
                # Conservative size adjustment (±20% max)
                size_ratio = np.clip(size_ratio, 0.8, 1.2)
                
                if abs(size_ratio - 1.0) > 0.05:  # Only adjust if significant
                    if size_ratio > 1.0:
                        # Dilate to increase size
                        kernel_size = int(3 * (size_ratio - 1.0) / 0.2) + 3
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                         (kernel_size, kernel_size))
                        adjusted = cv2.dilate(adjusted, kernel, iterations=1)
                    else:
                        # Erode to decrease size
                        kernel_size = int(3 * (1.0 - size_ratio) / 0.2) + 3
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (kernel_size, kernel_size))
                        adjusted = cv2.erode(adjusted, kernel, iterations=1)
                    
                    if self.debug_mode:
                        print(f"[Assembly] Applied size adjustment to electrode {electrode_id}: ratio={size_ratio:.2f}")
            
            # 3. Shape refinement based on similarity
            if electrode_id in gt_analysis.get('shape_similarity', {}):
                similarity = gt_analysis['shape_similarity'][electrode_id]
                
                # If shape similarity is low, try to improve it
                if similarity < 0.7 and electrode_id in gt_data:
                    gt_mask = gt_data[electrode_id]['mask']
                    
                    # Ensure same size
                    if gt_mask.shape != adjusted.shape:
                        gt_mask = cv2.resize(gt_mask, 
                                           (adjusted.shape[1], adjusted.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                    
                    # Blend with GT shape (conservative: 20% GT influence)
                    adjusted = cv2.addWeighted(adjusted.astype(np.float32), 0.8, 
                                             gt_mask.astype(np.float32), 0.2, 0)
                    adjusted = (adjusted > 127).astype(np.uint8) * 255
                    
                    if self.debug_mode:
                        print(f"[Assembly] Applied shape refinement to electrode {electrode_id}: similarity={similarity:.2f}")
        
        # Post-process GT adjustments to ensure separation
        if gt_data and electrode_id in gt_data:
            # Check if the GT adjustments made the electrode too large
            original_area = np.sum(mask > 127)
            adjusted_area = np.sum(adjusted > 127)
            if adjusted_area > original_area * 1.3:  # Too much growth
                # Apply slight erosion to prevent adhesion
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                adjusted = cv2.erode(adjusted, kernel, iterations=1)
                if self.debug_mode:
                    print(f"[Assembly] Applied post-GT erosion to prevent adhesion for electrode {electrode_id}")
        
        # Apply spacing constraints based on neighbors
        if electrode_id in relationships['neighbors']:
            for neighbor_info in relationships['neighbors'][electrode_id]:
                neighbor_id = neighbor_info['id']
                if neighbor_id not in placed_electrodes:
                    continue
                
                # Use GT ideal spacing if available
                ideal_spacing = relationships.get('gt_analysis', {}).get('ideal_spacing')
                if ideal_spacing is None:
                    ideal_spacing = self.min_electrode_spacing
                
                # For very close neighbors, ensure minimum spacing
                if neighbor_info['distance'] < ideal_spacing * 1.2:
                    # Apply slight erosion to maintain spacing
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    adjusted = cv2.erode(adjusted, kernel, iterations=1)
                    break
        
        return adjusted
    
    def _resolve_assembly_conflict(self, mask: np.ndarray, region: np.ndarray,
                                  electrode_id: int, data: Dict[str, Any],
                                  placed_electrodes: Dict, 
                                  relationships: Dict[str, Any]) -> np.ndarray:
        """Resolve placement conflicts using relationship information."""
        # Create conflict mask
        conflict_mask = np.logical_and(region > 0, mask > 127)
        
        if not np.any(conflict_mask):
            return mask
        
        # Intelligent conflict resolution
        adjusted = mask.copy()
        
        # Find which electrodes we're conflicting with
        conflicting_values = np.unique(region[conflict_mask])
        conflicting_values = conflicting_values[conflicting_values > 0]
        
        # Calculate conflict severity
        conflict_ratio = np.sum(conflict_mask) / np.sum(mask > 127)
        
        if conflict_ratio > 0.15:  # Significant conflict
            # Apply more aggressive separation
            if self.debug_mode:
                print(f"[Assembly] Significant conflict detected for electrode {electrode_id} ({conflict_ratio:.2%} overlap)")
            
            # Use distance transform for better separation
            dist_transform = cv2.distanceTransform((mask > 127).astype(np.uint8), 
                                                  cv2.DIST_L2, 5)
            
            # Create separation zone
            for conf_val in conflicting_values:
                conflict_region = (region == conf_val).astype(np.uint8)
                
                # Find boundary between electrodes
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                conflict_dilated = cv2.dilate(conflict_region, kernel, iterations=1)
                boundary_zone = conflict_dilated & (mask > 127)
                
                # Apply graduated erosion based on distance from center
                if np.any(boundary_zone):
                    # Use distance transform to erode more at boundaries
                    threshold = np.percentile(dist_transform[boundary_zone], 30)
                    adjusted[boundary_zone & (dist_transform < threshold)] = 0
        else:
            # Minor conflict - use standard approach
            for conf_val in conflicting_values:
                # Apply targeted erosion only at conflict boundaries
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                
                # Create boundary mask
                conflict_region = (region == conf_val)
                dilated = cv2.dilate(conflict_region.astype(np.uint8), kernel, iterations=2)
                boundary = dilated & (adjusted > 127)
                
                # Erode only at boundaries
                if np.any(boundary):
                    adjusted[boundary] = 0
        
        # Ensure minimum separation
        if np.any(adjusted > 127):
            # Apply morphological opening to ensure separation
            kernel_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            adjusted = cv2.morphologyEx(adjusted, cv2.MORPH_OPEN, kernel_sep)
        
        # Ensure we didn't erode too much
        if np.sum(adjusted > 127) < np.sum(mask > 127) * 0.7:  # Lower threshold
            # Too much erosion, use conservative approach
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            adjusted = cv2.erode(mask, kernel_small, iterations=1)
        
        return adjusted
    
    def _apply_global_postprocessing(self, mask: np.ndarray, 
                                   relationships: Dict[str, Any]) -> np.ndarray:
        """Apply global post-processing based on detected patterns."""
        result = mask.copy()
        
        # First ensure proper separation between electrodes
        # Use distance transform to identify potential adhesion points
        dist_transform = cv2.distanceTransform(result, cv2.DIST_L2, 5)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(result)
        
        # Check for suspiciously large components that might be adhered electrodes
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id)
            component_area = np.sum(component_mask)
            
            # If component is too large, it might be adhered electrodes
            if component_area > 50000:  # Threshold for suspiciously large area
                # Use watershed to separate
                dist = cv2.distanceTransform(component_mask.astype(np.uint8), cv2.DIST_L2, 5)
                
                # Find peaks
                _, max_val, _, _ = cv2.minMaxLoc(dist)
                if max_val > 15:  # Wide enough to be multiple electrodes
                    # Apply watershed separation
                    _, markers = cv2.connectedComponents((dist > max_val * 0.7).astype(np.uint8))
                    
                    if np.max(markers) > 1:  # Multiple peaks found
                        # Separate using erosion at boundaries
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        separated = cv2.morphologyEx(component_mask.astype(np.uint8) * 255, 
                                                   cv2.MORPH_OPEN, kernel)
                        result[component_mask] = separated[component_mask]
                        
                        if self.debug_mode:
                            print(f"[Assembly] Separated adhered electrodes in component {label_id}")
        
        # Remove small isolated noise
        num_labels, labels = cv2.connectedComponents(result)
        min_size = 50
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id)
            if np.sum(component_mask) < min_size:
                result[component_mask] = 0
        
        return result
    
    def _ensure_electrode_separation(self, mask: np.ndarray) -> np.ndarray:
        """Final aggressive pass to ensure all electrodes are separated."""
        result = mask.copy()
        
        # Use distance transform to find potential adhesion points
        dist_transform = cv2.distanceTransform(result, cv2.DIST_L2, 5)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(result)
        
        # For each component, check if it might be adhered electrodes
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id)
            component_area = np.sum(component_mask)
            
            # Check aspect ratio and area to identify potential adhesions
            if component_area > 50000:  # Threshold for suspiciously large area
                # Get bounding box
                y_coords, x_coords = np.where(component_mask)
                min_y, max_y = y_coords.min(), y_coords.max()
                min_x, max_x = x_coords.min(), x_coords.max()
                width = max_x - min_x + 1
                height = max_y - min_y + 1
                
                # If too wide relative to expected electrode width
                if width > 60:  # Wider than typical electrode
                    # Find the narrowest points using distance transform
                    component_dist = dist_transform * component_mask
                    
                    # Find local minima in horizontal profiles
                    for y in range(min_y, max_y, 50):  # Check every 50 pixels
                        if y >= component_dist.shape[0]:
                            continue
                        profile = component_dist[y, min_x:max_x+1]
                        if np.max(profile) > 10:  # Has significant width
                            # Find minima
                            minima = []
                            for x in range(1, len(profile)-1):
                                if profile[x] < profile[x-1] and profile[x] < profile[x+1]:
                                    if profile[x] < np.max(profile) * 0.6:  # Significant minimum
                                        minima.append(min_x + x)
                            
                            # Create vertical separation lines at minima
                            for x_min in minima:
                                if x_min > min_x + 20 and x_min < max_x - 20:  # Not too close to edges
                                    # Create separation line
                                    sep_width = 5
                                    x_start = max(x_min - sep_width//2, 0)
                                    x_end = min(x_min + sep_width//2 + 1, result.shape[1])
                                    result[min_y:max_y+1, x_start:x_end] = 0
                                    
                                    if self.debug_mode:
                                        print(f"[Assembly] Applied vertical separation at x={x_min}")
        
        # Apply final morphological opening to ensure clean separation
        final_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, final_kernel)
        
        return result


def main():
    """Test the assembly refiner with electrode relationship modeling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Assembly refinement with electrode relationship modeling')
    parser.add_argument('--data-dir', type=str, 
                       default='data/Full_Instances',
                       help='Base directory containing instance data')
    parser.add_argument('--output-dir', type=str,
                       default='results/refined_masks_assembly',
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
                       default='results/assembly_visualizations',
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
    parser.add_argument('--assembled-dir', type=str, default=None,
                        help='Directory for assembled full-image masks (default: <output-dir>/assembled)')
    args = parser.parse_args()
    
    # Initialize the assembly refiner - Electrode relationship modeling
    refiner = AssemblyRefiner(
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
    
    # Process instances with intelligent assembly
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create directory for assembled masks
    assembled_dir = args.assembled_dir or os.path.join(args.output_dir, 'assembled')
    os.makedirs(assembled_dir, exist_ok=True)
    
    instance_info_dir = os.path.join(args.data_dir, 'instance_info')
    enhanced_image_dir = os.path.join(args.origin_dir, 'images_enhanced')
    mask_dir = os.path.join(args.data_dir, 'repaired_masks')
    
    # Get instance info files
    info_files = sorted([f for f in os.listdir(instance_info_dir) if f.endswith('_info.json')])
    
    if args.test_image:
        info_files = [f for f in info_files if args.test_image in f]
    
    print(f"Processing {len(info_files)} images with the assembly refiner...")
    
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
        
        # Process all instances for this image
        instances = info_data['instances']
        all_refined_masks = {}
        
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
            
            # Refine using the assembly refiner with relationship modeling
            refined_masks = refiner.refine_unified_region(region_data)
            
            # Extract individual instance masks
            for inst_id, refined_mask in refined_masks.items():
                if inst_id not in region_data['instance_masks']:
                    continue
                
                rel_pos = region_data['instance_masks'][inst_id]['relative_pos']
                bbox = instance_data[inst_id]['bbox']
                
                # Extract instance portion
                inst_refined = refined_mask[rel_pos[1]:rel_pos[1]+bbox[3], 
                                           rel_pos[0]:rel_pos[0]+bbox[2]]
                
                # Save individual mask
                output_path = os.path.join(args.output_dir, f"{base_name}_instance_{inst_id}.png")
                cv2.imwrite(output_path, inst_refined)
                
                # Store for assembly
                all_refined_masks[inst_id] = inst_refined
        
        # Intelligent assembly of all refined masks for this image
        if all_refined_masks:
            assembled_path = os.path.join(assembled_dir, f"{base_name}.png")
            refiner.assemble_refined_masks(all_refined_masks, info_data, assembled_path, image_name=base_name)
    
    print("\nAssembly refinement complete!")
    print(f"Individual masks saved to: {args.output_dir}")
    print(f"Assembled masks saved to: {assembled_dir}")


if __name__ == "__main__":
    main()