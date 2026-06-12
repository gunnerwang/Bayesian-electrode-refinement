#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electrode-specific metrics for mask evaluation.

This module provides specialized metrics for evaluating electrode masks,
including electrode counting, classification, and geometric measurements.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class ElectrodeInfo:
    """Information about a single electrode."""
    label: int
    area: int
    length: float
    width: float
    aspect_ratio: float
    center: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    contour: np.ndarray
    type: str  # 'anode' or 'cathode'
    orientation: float  # angle in degrees


@dataclass
class ElectrodeMetrics:
    """Electrode-specific metrics for a mask."""
    num_electrodes: int
    num_anodes: int  # Long electrodes (positive)
    num_cathodes: int  # Short electrodes (negative)
    anode_cathode_ratio: float
    avg_electrode_area: float
    avg_anode_area: float
    avg_cathode_area: float
    avg_aspect_ratio: float
    avg_anode_aspect_ratio: float
    avg_cathode_aspect_ratio: float
    anode_length_mean: float  # Average length of anodes
    anode_width_mean: float   # Average width of anodes
    cathode_length_mean: float  # Average length of cathodes
    cathode_width_mean: float   # Average width of cathodes
    vertex_distance_mean: float  # Mean distance between anode vertices
    vertex_distance_std: float   # Std of distances between anode vertices
    alignment_score: float       # How well aligned the anodes are
    spacing_uniformity: float    # How uniform the electrode spacing is
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'num_electrodes': self.num_electrodes,
            'num_anodes': self.num_anodes,
            'num_cathodes': self.num_cathodes,
            'anode_cathode_ratio': self.anode_cathode_ratio,
            'avg_electrode_area': self.avg_electrode_area,
            'avg_anode_area': self.avg_anode_area,
            'avg_cathode_area': self.avg_cathode_area,
            'avg_aspect_ratio': self.avg_aspect_ratio,
            'avg_anode_aspect_ratio': self.avg_anode_aspect_ratio,
            'avg_cathode_aspect_ratio': self.avg_cathode_aspect_ratio,
            'anode_length_mean': self.anode_length_mean,
            'anode_width_mean': self.anode_width_mean,
            'cathode_length_mean': self.cathode_length_mean,
            'cathode_width_mean': self.cathode_width_mean,
            'vertex_distance_mean': self.vertex_distance_mean,
            'vertex_distance_std': self.vertex_distance_std,
            'alignment_score': self.alignment_score,
            'spacing_uniformity': self.spacing_uniformity
        }


class ElectrodeAnalyzer:
    """Analyze electrode-specific features in masks."""
    
    def __init__(self, min_area_threshold: int = 100, verbose: bool = False):
        """
        Initialize the analyzer.
        
        Args:
            min_area_threshold: Minimum area to consider as valid electrode
            verbose: Whether to print debug information
        """
        self.min_area_threshold = min_area_threshold
        self.verbose = verbose
    
    def extract_electrodes(self, mask: np.ndarray) -> List[ElectrodeInfo]:
        """Extract individual electrodes from a mask."""
        # Binary threshold
        binary = (mask > 127).astype(np.uint8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary)
        
        electrodes = []
        
        for label in range(1, num_labels):  # Skip background (0)
            # Extract component
            component_mask = (labels == label).astype(np.uint8)
            
            # Calculate area
            area = np.sum(component_mask)
            if area < self.min_area_threshold:
                continue
            
            # Find contour
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            contour = contours[0]
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate properties
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            length = max(w, h)
            width = min(w, h)
            
            # Get center
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2
                cy = y + h / 2
            
            # Determine orientation (angle of minimum area rectangle)
            if len(contour) >= 5:
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
            else:
                angle = 0
            
            # Type will be determined later by clustering
            electrode = ElectrodeInfo(
                label=label,
                area=area,
                length=length,
                width=width,
                aspect_ratio=aspect_ratio,
                center=(cx, cy),
                bbox=(x, y, w, h),
                contour=contour,
                type='unknown',  # Will be classified later
                orientation=angle
            )
            
            electrodes.append(electrode)
        
        # Classify electrodes using clustering
        if len(electrodes) >= 2:
            self._classify_electrodes_by_clustering(electrodes)
        elif len(electrodes) == 1:
            # If only one electrode, classify based on aspect ratio
            electrodes[0].type = 'anode' if electrodes[0].aspect_ratio > 2.5 else 'cathode'
        
        return electrodes
    
    def _classify_electrodes_by_clustering(self, electrodes: List[ElectrodeInfo]):
        """Classify electrodes as anodes or cathodes using improved clustering."""
        n_electrodes = len(electrodes)
        
        if n_electrodes < 2:
            # Too few electrodes to cluster
            for electrode in electrodes:
                electrode.type = 'anode' if electrode.aspect_ratio > 40 else 'cathode'
            return
        
        # Extract features for classification
        aspect_ratios = np.array([e.aspect_ratio for e in electrodes])
        lengths = np.array([e.length for e in electrodes])
        
        # Method 1: Use expected ratio (typically ~50% anodes in battery electrodes)
        expected_anode_ratio = 0.515  # Based on typical battery electrode composition
        
        # Check if we have a clear bimodal distribution
        # Calculate the standard deviation to assess distribution spread
        aspect_std = np.std(aspect_ratios)
        aspect_mean = np.mean(aspect_ratios)
        
        # Check if we should use length-based classification
        # This is more robust when aspect ratios are distorted (e.g., after aggressive refinement)
        length_std = np.std(lengths)
        length_mean = np.mean(lengths)
        
        # If aspect ratio distribution is distorted but length distribution is cleaner, use length
        use_length_classification = False
        if aspect_std > 15 or max(aspect_ratios) > 70:
            # Check if length provides better separation
            length_range = np.max(lengths) - np.min(lengths)
            if length_range > 200 and length_std / length_mean < aspect_std / aspect_mean:
                use_length_classification = True
                if self.verbose:
                    print(f"Using length-based classification (length_std/mean={length_std/length_mean:.2f} < aspect_std/mean={aspect_std/aspect_mean:.2f})")
        
        if use_length_classification:
            # Use length as primary feature
            sorted_indices = np.argsort(lengths)
            n_expected_anodes = int(n_electrodes * expected_anode_ratio)
            
            # The electrodes with longest lengths are anodes
            anode_indices = set(sorted_indices[-n_expected_anodes:])
            
            # Validate using gap analysis
            if n_expected_anodes > 0 and n_expected_anodes < n_electrodes:
                threshold_idx = sorted_indices[-n_expected_anodes]
                threshold_length = lengths[threshold_idx]
                
                if n_expected_anodes > 1:
                    prev_idx = sorted_indices[-n_expected_anodes-1]
                    prev_length = lengths[prev_idx]
                    gap = threshold_length - prev_length
                    
                    # If gap is too small, try two-feature clustering
                    if gap < 50:  # Length gap threshold in pixels
                        try:
                            from sklearn.cluster import KMeans
                            from sklearn.preprocessing import StandardScaler
                            
                            # Combine length and aspect ratio features
                            features = np.column_stack((lengths, aspect_ratios))
                            
                            # Normalize features
                            scaler = StandardScaler()
                            features_scaled = scaler.fit_transform(features)
                            
                            # Use k-means with 2 clusters
                            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                            labels = kmeans.fit_predict(features_scaled)
                            
                            # Determine which cluster is anodes (higher length)
                            cluster_0_mean_length = np.mean(lengths[labels == 0])
                            cluster_1_mean_length = np.mean(lengths[labels == 1])
                            
                            anode_label = 0 if cluster_0_mean_length > cluster_1_mean_length else 1
                            anode_indices = set(np.where(labels == anode_label)[0])
                            
                            if self.verbose:
                                print(f"Used two-feature clustering (length + aspect ratio)")
                        except ImportError:
                            if self.verbose:
                                print("Sklearn not available, using simple length threshold")
        
        elif aspect_std > 15 or max(aspect_ratios) > 70:
            # Original high variance handling for aspect ratio
            # High variance suggests a distorted aspect-ratio distribution
            # Use a more conservative approach
            
            # Find natural clusters using percentile-based approach
            p25 = np.percentile(aspect_ratios, 25)
            p75 = np.percentile(aspect_ratios, 75)
            
            # If p75 is much higher than typical anode threshold, adjust expectation
            if p75 > 55:  # Unusually high aspect ratios
                # Look for a gap in the middle range
                mid_range_mask = (aspect_ratios > 35) & (aspect_ratios < 60)
                mid_range_aspects = aspect_ratios[mid_range_mask]
                
                if len(mid_range_aspects) > 10:
                    # Find the median of middle range as potential split point
                    mid_median = np.median(mid_range_aspects)
                    # Count how many are above this threshold
                    n_above_threshold = np.sum(aspect_ratios > mid_median)
                    expected_anode_ratio = n_above_threshold / n_electrodes
                    
                    if self.verbose:
                        print(f"Adjusted expected anode ratio to {expected_anode_ratio:.3f} due to spread distribution")
                else:
                    # If middle range is sparse, likely means clear separation
                    # Count electrodes with very high aspect ratios as anodes
                    n_high_aspect = np.sum(aspect_ratios > 50)
                    if n_high_aspect > 0.2 * n_electrodes:  # At least 20% have high aspect
                        expected_anode_ratio = n_high_aspect / n_electrodes
                        if self.verbose:
                            print(f"Using high aspect ratio count: {n_high_aspect} electrodes")
            else:
                # Even with high variance, if p75 is reasonable, keep original ratio
                if self.verbose:
                    print(f"Keeping original ratio despite high variance (p75={p75:.1f})")
        
        # Default case: use aspect ratio classification
        if not use_length_classification:
            # Sort electrodes by aspect ratio
            sorted_indices = np.argsort(aspect_ratios)
            
            # Determine split point based on expected ratio
            n_expected_anodes = int(n_electrodes * expected_anode_ratio)
            
            # The electrodes with highest aspect ratios are anodes
            anode_indices = set(sorted_indices[-n_expected_anodes:])
        
        # Validate the classification using natural gap in aspect ratios
        if n_expected_anodes > 0 and n_expected_anodes < n_electrodes:
            # Find the gap between presumed anodes and cathodes
            threshold_idx = sorted_indices[-n_expected_anodes]
            threshold_aspect = aspect_ratios[threshold_idx]
            
            # Check if there's a clear gap
            if n_expected_anodes > 1:
                prev_idx = sorted_indices[-n_expected_anodes-1]
                prev_aspect = aspect_ratios[prev_idx]
                gap = threshold_aspect - prev_aspect
                
                # If gap is too small, try GMM for better separation
                if gap < 2.0:  # Aspect ratio gap threshold
                    try:
                        from sklearn.mixture import GaussianMixture
                        
                        # Use Gaussian Mixture Model for overlapping distributions
                        gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
                        aspect_ratios_reshaped = aspect_ratios.reshape(-1, 1)
                        labels = gmm.fit_predict(aspect_ratios_reshaped)
                        
                        # Determine which cluster is anodes
                        cluster_0_mean = np.mean(aspect_ratios[labels == 0])
                        cluster_1_mean = np.mean(aspect_ratios[labels == 1])
                        
                        anode_label = 0 if cluster_0_mean > cluster_1_mean else 1
                        
                        # Update anode indices based on GMM
                        anode_indices = set(np.where(labels == anode_label)[0])
                        
                        # Validate the ratio is reasonable
                        gmm_ratio = len(anode_indices) / n_electrodes
                        if gmm_ratio < 0.3 or gmm_ratio > 0.7:
                            # GMM result is unreasonable, revert to original method
                            anode_indices = set(sorted_indices[-n_expected_anodes:])
                            
                    except Exception as e:
                        # If GMM fails, stick with the original method
                        print(f"GMM classification failed: {e}. Using aspect ratio ranking.")
        
        # Assign types based on classification
        for i, electrode in enumerate(electrodes):
            if i in anode_indices:
                electrode.type = 'anode'
            else:
                electrode.type = 'cathode'
        
        # Post-validation: ensure we have reasonable counts
        n_anodes = sum(1 for e in electrodes if e.type == 'anode')
        n_cathodes = n_electrodes - n_anodes
        
        # Log the classification result for debugging
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Classified {n_anodes} anodes and {n_cathodes} cathodes (ratio: {n_anodes/n_electrodes:.2f})")
    
    def get_anode_vertices(self, electrodes: List[ElectrodeInfo]) -> np.ndarray:
        """Get the top vertices of anodes."""
        vertices = []
        
        for electrode in electrodes:
            if electrode.type == 'anode':
                # Get the topmost point of the electrode
                contour = electrode.contour
                top_point_idx = np.argmin(contour[:, 0, 1])
                top_point = contour[top_point_idx, 0]
                vertices.append(top_point)
        
        return np.array(vertices) if vertices else np.array([]).reshape(0, 2)
    
    def calculate_electrode_metrics(self, mask: np.ndarray) -> ElectrodeMetrics:
        """Calculate all electrode-specific metrics for a mask."""
        electrodes = self.extract_electrodes(mask)
        
        if not electrodes:
            # Return empty metrics
            return ElectrodeMetrics(
                num_electrodes=0, num_anodes=0, num_cathodes=0, anode_cathode_ratio=0,
                avg_electrode_area=0, avg_anode_area=0, avg_cathode_area=0,
                avg_aspect_ratio=0, avg_anode_aspect_ratio=0, avg_cathode_aspect_ratio=0,
                anode_length_mean=0, anode_width_mean=0,
                cathode_length_mean=0, cathode_width_mean=0,
                vertex_distance_mean=0, vertex_distance_std=0,
                alignment_score=0, spacing_uniformity=0
            )
        
        # Separate anodes and cathodes
        anodes = [e for e in electrodes if e.type == 'anode']
        cathodes = [e for e in electrodes if e.type == 'cathode']
        
        # Basic counts
        num_electrodes = len(electrodes)
        num_anodes = len(anodes)
        num_cathodes = len(cathodes)
        anode_cathode_ratio = num_anodes / (num_cathodes + 1e-6)
        
        # Area statistics
        all_areas = [e.area for e in electrodes]
        avg_electrode_area = np.mean(all_areas) if all_areas else 0
        
        anode_areas = [e.area for e in anodes]
        avg_anode_area = np.mean(anode_areas) if anode_areas else 0
        
        cathode_areas = [e.area for e in cathodes]
        avg_cathode_area = np.mean(cathode_areas) if cathode_areas else 0
        
        # Aspect ratio statistics
        all_aspects = [e.aspect_ratio for e in electrodes]
        avg_aspect_ratio = np.mean(all_aspects) if all_aspects else 0
        
        anode_aspects = [e.aspect_ratio for e in anodes]
        avg_anode_aspect_ratio = np.mean(anode_aspects) if anode_aspects else 0
        
        cathode_aspects = [e.aspect_ratio for e in cathodes]
        avg_cathode_aspect_ratio = np.mean(cathode_aspects) if cathode_aspects else 0
        
        # Length and width statistics
        anode_lengths = [e.length for e in anodes]
        anode_length_mean = np.mean(anode_lengths) if anode_lengths else 0
        
        anode_widths = [e.width for e in anodes]
        anode_width_mean = np.mean(anode_widths) if anode_widths else 0
        
        cathode_lengths = [e.length for e in cathodes]
        cathode_length_mean = np.mean(cathode_lengths) if cathode_lengths else 0
        
        cathode_widths = [e.width for e in cathodes]
        cathode_width_mean = np.mean(cathode_widths) if cathode_widths else 0
        
        # Vertex analysis for anodes
        vertices = self.get_anode_vertices(electrodes)
        
        if len(vertices) >= 2:
            # Calculate pairwise distances
            distances = cdist(vertices, vertices)
            # Get upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(len(vertices), k=1)
            pairwise_distances = distances[triu_indices]
            
            vertex_distance_mean = np.mean(pairwise_distances)
            vertex_distance_std = np.std(pairwise_distances)
            
            # Calculate alignment score (how well vertices form a line)
            if len(vertices) >= 3:
                # Fit a line to vertices and calculate R²
                x = vertices[:, 0]
                y = vertices[:, 1]
                coeffs = np.polyfit(x, y, 1)
                poly = np.poly1d(coeffs)
                y_pred = poly(x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                alignment_score = 1 - (ss_res / (ss_tot + 1e-6))
            else:
                alignment_score = 1.0
            
            # Calculate spacing uniformity
            # Sort vertices by x-coordinate
            sorted_vertices = vertices[vertices[:, 0].argsort()]
            spacings = np.diff(sorted_vertices[:, 0])
            if len(spacings) > 0:
                spacing_uniformity = 1 - (np.std(spacings) / (np.mean(spacings) + 1e-6))
            else:
                spacing_uniformity = 1.0
        else:
            vertex_distance_mean = 0
            vertex_distance_std = 0
            alignment_score = 0
            spacing_uniformity = 0
        
        return ElectrodeMetrics(
            num_electrodes=num_electrodes,
            num_anodes=num_anodes,
            num_cathodes=num_cathodes,
            anode_cathode_ratio=anode_cathode_ratio,
            avg_electrode_area=avg_electrode_area,
            avg_anode_area=avg_anode_area,
            avg_cathode_area=avg_cathode_area,
            avg_aspect_ratio=avg_aspect_ratio,
            avg_anode_aspect_ratio=avg_anode_aspect_ratio,
            avg_cathode_aspect_ratio=avg_cathode_aspect_ratio,
            anode_length_mean=anode_length_mean,
            anode_width_mean=anode_width_mean,
            cathode_length_mean=cathode_length_mean,
            cathode_width_mean=cathode_width_mean,
            vertex_distance_mean=vertex_distance_mean,
            vertex_distance_std=vertex_distance_std,
            alignment_score=alignment_score,
            spacing_uniformity=spacing_uniformity
        )
    
    def compare_electrode_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict:
        """Compare electrode metrics between predicted and ground truth masks."""
        pred_metrics = self.calculate_electrode_metrics(pred_mask)
        gt_metrics = self.calculate_electrode_metrics(gt_mask)
        
        # Calculate differences
        comparison = {
            'pred_metrics': pred_metrics.to_dict(),
            'gt_metrics': gt_metrics.to_dict(),
            'differences': {
                'num_electrodes_diff': pred_metrics.num_electrodes - gt_metrics.num_electrodes,
                'num_anodes_diff': pred_metrics.num_anodes - gt_metrics.num_anodes,
                'num_cathodes_diff': pred_metrics.num_cathodes - gt_metrics.num_cathodes,
                'anode_cathode_ratio_diff': pred_metrics.anode_cathode_ratio - gt_metrics.anode_cathode_ratio,
                'avg_area_diff': pred_metrics.avg_electrode_area - gt_metrics.avg_electrode_area,
                'anode_length_diff': pred_metrics.anode_length_mean - gt_metrics.anode_length_mean,
                'cathode_length_diff': pred_metrics.cathode_length_mean - gt_metrics.cathode_length_mean,
                'vertex_distance_diff': pred_metrics.vertex_distance_mean - gt_metrics.vertex_distance_mean,
                'alignment_score_diff': pred_metrics.alignment_score - gt_metrics.alignment_score,
                'spacing_uniformity_diff': pred_metrics.spacing_uniformity - gt_metrics.spacing_uniformity
            },
            'relative_errors': {
                'num_electrodes_error': abs(pred_metrics.num_electrodes - gt_metrics.num_electrodes) / (gt_metrics.num_electrodes + 1e-6),
                'num_anodes_error': abs(pred_metrics.num_anodes - gt_metrics.num_anodes) / (gt_metrics.num_anodes + 1e-6),
                'num_cathodes_error': abs(pred_metrics.num_cathodes - gt_metrics.num_cathodes) / (gt_metrics.num_cathodes + 1e-6),
                'avg_area_error': abs(pred_metrics.avg_electrode_area - gt_metrics.avg_electrode_area) / (gt_metrics.avg_electrode_area + 1e-6),
                'anode_length_error': abs(pred_metrics.anode_length_mean - gt_metrics.anode_length_mean) / (gt_metrics.anode_length_mean + 1e-6),
                'cathode_length_error': abs(pred_metrics.cathode_length_mean - gt_metrics.cathode_length_mean) / (gt_metrics.cathode_length_mean + 1e-6),
                'vertex_distance_error': abs(pred_metrics.vertex_distance_mean - gt_metrics.vertex_distance_mean) / (gt_metrics.vertex_distance_mean + 1e-6)
            }
        }
        
        return comparison
    
    def visualize_electrode_analysis(self, mask: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
        """Visualize electrode analysis results."""
        electrodes = self.extract_electrodes(mask)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original mask
        ax1.imshow(mask, cmap='gray')
        ax1.set_title('Original Mask')
        ax1.axis('off')
        
        # Analysis visualization
        ax2.imshow(mask, cmap='gray', alpha=0.3)
        
        # Different colors for anodes and cathodes
        for electrode in electrodes:
            x, y, w, h = electrode.bbox
            color = 'red' if electrode.type == 'anode' else 'blue'
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            
            # Add center point
            ax2.plot(electrode.center[0], electrode.center[1], 'o', color=color, markersize=4)
            
            # Add label with type
            label_text = f'{electrode.label}\n({electrode.type[0].upper()})'
            ax2.text(electrode.center[0], electrode.center[1], label_text, 
                    fontsize=8, ha='center', va='center', color='white', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        
        # Draw vertex connections for anodes
        vertices = self.get_anode_vertices(electrodes)
        if len(vertices) > 1:
            # Sort by x-coordinate
            sorted_vertices = vertices[vertices[:, 0].argsort()]
            ax2.plot(sorted_vertices[:, 0], sorted_vertices[:, 1], 'g-', linewidth=2, label='Anode Vertices')
            ax2.plot(sorted_vertices[:, 0], sorted_vertices[:, 1], 'go', markersize=6)
        
        ax2.set_title(f'Electrode Analysis (Red: Anode, Blue: Cathode)')
        ax2.legend()
        ax2.axis('off')
        
        # Add metrics text
        metrics = self.calculate_electrode_metrics(mask)
        metrics_text = f"Total: {metrics.num_electrodes}\n"
        metrics_text += f"Anodes: {metrics.num_anodes} (L={metrics.anode_length_mean:.1f}, W={metrics.anode_width_mean:.1f})\n"
        metrics_text += f"Cathodes: {metrics.num_cathodes} (L={metrics.cathode_length_mean:.1f}, W={metrics.cathode_width_mean:.1f})\n"
        metrics_text += f"Alignment: {metrics.alignment_score:.3f}\n"
        metrics_text += f"Spacing Uniformity: {metrics.spacing_uniformity:.3f}"
        
        ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def visualize_electrode_comparison(self, baseline_mask: np.ndarray, refined_mask: np.ndarray, 
                                     gt_mask: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
        """Create a comparison visualization of electrode analysis across masks."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        masks = [baseline_mask, refined_mask, gt_mask]
        titles = ['Baseline', 'Refined', 'Ground Truth']
        
        # First row: Original masks
        for i, (mask, title) in enumerate(zip(masks, titles)):
            ax = axes[0, i]
            ax.imshow(mask, cmap='gray')
            ax.set_title(f'{title} Mask')
            ax.axis('off')
        
        # Second row: Electrode analysis
        for i, (mask, title) in enumerate(zip(masks, titles)):
            ax = axes[1, i]
            ax.imshow(mask, cmap='gray', alpha=0.3)
            
            electrodes = self.extract_electrodes(mask)
            metrics = self.calculate_electrode_metrics(mask)
            
            # Draw electrodes
            for electrode in electrodes:
                x, y, w, h = electrode.bbox
                color = 'red' if electrode.type == 'anode' else 'blue'
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
            
            # Draw anode vertices connections
            vertices = self.get_anode_vertices(electrodes)
            if len(vertices) > 1:
                sorted_vertices = vertices[vertices[:, 0].argsort()]
                ax.plot(sorted_vertices[:, 0], sorted_vertices[:, 1], 'g-', linewidth=2, alpha=0.7)
                ax.plot(sorted_vertices[:, 0], sorted_vertices[:, 1], 'go', markersize=4)
            
            ax.set_title(f'{title} Analysis')
            ax.axis('off')
            
            # Add metrics text
            metrics_text = f"Total: {metrics.num_electrodes}\n"
            metrics_text += f"Anodes: {metrics.num_anodes}\n"
            metrics_text += f"Cathodes: {metrics.num_cathodes}"
            
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add overall comparison metrics
        baseline_comparison = self.compare_electrode_metrics(baseline_mask, gt_mask)
        refined_comparison = self.compare_electrode_metrics(refined_mask, gt_mask)
        
        # Calculate improvements
        electrode_imp = (baseline_comparison['relative_errors']['num_electrodes_error'] - 
                        refined_comparison['relative_errors']['num_electrodes_error'])
        anode_imp = (baseline_comparison['relative_errors']['num_anodes_error'] - 
                    refined_comparison['relative_errors']['num_anodes_error'])
        cathode_imp = (baseline_comparison['relative_errors']['num_cathodes_error'] - 
                      refined_comparison['relative_errors']['num_cathodes_error'])
        
        comparison_text = f"Error Improvements (Baseline → Refined):\n"
        comparison_text += f"Total Electrodes: {electrode_imp:+.3f}\n"
        comparison_text += f"Anodes: {anode_imp:+.3f}\n"
        comparison_text += f"Cathodes: {cathode_imp:+.3f}"
        
        fig.text(0.5, 0.02, comparison_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.suptitle('Electrode Analysis Comparison', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return fig


def create_electrode_metrics_visualization(results: Dict, output_dir: str, max_vis: int = 20,
                                         baseline_dir: str = None, refined_dir: str = None, 
                                         gt_dir: str = None):
    """
    Create comprehensive visualizations for electrode metrics.
    
    Args:
        results: Dictionary with electrode evaluation results
        output_dir: Directory to save visualizations
        max_vis: Maximum number of individual image visualizations
        baseline_dir: Optional directory with baseline masks for individual visualizations
        refined_dir: Optional directory with refined masks for individual visualizations
        gt_dir: Optional directory with ground truth masks for individual visualizations
    """
    if not results or 'detailed_results' not in results:
        print("No results to visualize")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary comparison plot
    create_electrode_summary_plot(results['summary'], output_dir)
    
    # 2. Per-image improvement scatter plot
    create_improvement_scatter_plot(results['detailed_results'], output_dir)
    
    # 3. Electrode count distribution plots
    create_electrode_distribution_plots(results['detailed_results'], output_dir)
    
    # 4. Top improved images visualization
    create_top_improved_visualizations(results['detailed_results'], output_dir, max_vis)
    
    # 5. Error distribution histograms
    create_error_distribution_plots(results['detailed_results'], output_dir)
    
    # 6. Individual image comparisons (if directories provided)
    if all([baseline_dir, refined_dir, gt_dir]):
        create_individual_comparisons(results['detailed_results'], output_dir, 
                                    baseline_dir, refined_dir, gt_dir, max_vis)
    
    print(f"Saved electrode metrics visualizations to: {output_dir}")


def create_electrode_summary_plot(summary: Dict, output_dir: str):
    """Create summary bar plot comparing baseline vs refined electrode metrics."""
    if not summary:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Electrode count errors
    ax = axes[0]
    metrics = ['Baseline', 'Refined']
    errors = [summary['avg_baseline_electrode_error'], summary['avg_refined_electrode_error']]
    bars = ax.bar(metrics, errors, color=['#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Average Relative Error')
    ax.set_title('Total Electrode Count Error')
    ax.set_ylim(0, max(errors) * 1.2 if max(errors) > 0 else 1)
    
    # Add improvement text
    improvement = summary['avg_electrode_error_improvement']
    ax.text(0.5, max(errors) * 1.1, f'Improvement: {improvement:+.3f}', 
            ha='center', fontsize=10, fontweight='bold')
    
    # Anode count errors
    ax = axes[1]
    errors = [summary['avg_baseline_anode_error'], summary['avg_refined_anode_error']]
    bars = ax.bar(metrics, errors, color=['#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Average Relative Error')
    ax.set_title('Anode Count Error')
    ax.set_ylim(0, max(errors) * 1.2 if max(errors) > 0 else 1)
    
    improvement = summary['avg_anode_error_improvement']
    ax.text(0.5, max(errors) * 1.1, f'Improvement: {improvement:+.3f}', 
            ha='center', fontsize=10, fontweight='bold')
    
    # Cathode count errors
    ax = axes[2]
    errors = [summary['avg_baseline_cathode_error'], summary['avg_refined_cathode_error']]
    bars = ax.bar(metrics, errors, color=['#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Average Relative Error')
    ax.set_title('Cathode Count Error')
    ax.set_ylim(0, max(errors) * 1.2 if max(errors) > 0 else 1)
    
    improvement = summary['avg_cathode_error_improvement']
    ax.text(0.5, max(errors) * 1.1, f'Improvement: {improvement:+.3f}', 
            ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'Electrode Metrics Summary ({summary["num_images"]} images)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'electrode_metrics_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_improvement_scatter_plot(results: List[Dict], output_dir: str):
    """Create scatter plot showing per-image improvements."""
    if not results:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    images = [r['image'] for r in results]
    electrode_imp = [r['improvements']['num_electrodes_error_improvement'] for r in results]
    anode_imp = [r['improvements']['num_anodes_error_improvement'] for r in results]
    cathode_imp = [r['improvements']['num_cathodes_error_improvement'] for r in results]
    
    # Create scatter plot
    x = range(len(images))
    ax.scatter(x, electrode_imp, label='Total Electrodes', alpha=0.7, s=50)
    ax.scatter(x, anode_imp, label='Anodes', alpha=0.7, s=50)
    ax.scatter(x, cathode_imp, label='Cathodes', alpha=0.7, s=50)
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Customize plot
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Error Improvement (positive = better)')
    ax.set_title('Per-Image Electrode Metrics Improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics
    stats_text = f'Mean Improvements:\nElectrodes: {np.mean(electrode_imp):+.3f}\n'
    stats_text += f'Anodes: {np.mean(anode_imp):+.3f}\nCathodes: {np.mean(cathode_imp):+.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'electrode_improvement_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_electrode_distribution_plots(results: List[Dict], output_dir: str):
    """Create distribution plots for electrode counts."""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract data
    gt_electrodes = [r['baseline_comparison']['gt_metrics']['num_electrodes'] for r in results]
    baseline_electrodes = [r['baseline_comparison']['pred_metrics']['num_electrodes'] for r in results]
    refined_electrodes = [r['refined_comparison']['pred_metrics']['num_electrodes'] for r in results]
    
    gt_anodes = [r['baseline_comparison']['gt_metrics']['num_anodes'] for r in results]
    baseline_anodes = [r['baseline_comparison']['pred_metrics']['num_anodes'] for r in results]
    refined_anodes = [r['refined_comparison']['pred_metrics']['num_anodes'] for r in results]
    
    gt_cathodes = [r['baseline_comparison']['gt_metrics']['num_cathodes'] for r in results]
    baseline_cathodes = [r['baseline_comparison']['pred_metrics']['num_cathodes'] for r in results]
    refined_cathodes = [r['refined_comparison']['pred_metrics']['num_cathodes'] for r in results]
    
    # Total electrodes distribution
    ax = axes[0, 0]
    ax.hist([gt_electrodes, baseline_electrodes, refined_electrodes], 
            label=['Ground Truth', 'Baseline', 'Refined'], alpha=0.7, bins=15)
    ax.set_xlabel('Number of Electrodes')
    ax.set_ylabel('Frequency')
    ax.set_title('Total Electrode Count Distribution')
    ax.legend()
    
    # Anodes distribution
    ax = axes[0, 1]
    ax.hist([gt_anodes, baseline_anodes, refined_anodes], 
            label=['Ground Truth', 'Baseline', 'Refined'], alpha=0.7, bins=15)
    ax.set_xlabel('Number of Anodes')
    ax.set_ylabel('Frequency')
    ax.set_title('Anode Count Distribution')
    ax.legend()
    
    # Cathodes distribution
    ax = axes[0, 2]
    ax.hist([gt_cathodes, baseline_cathodes, refined_cathodes], 
            label=['Ground Truth', 'Baseline', 'Refined'], alpha=0.7, bins=15)
    ax.set_xlabel('Number of Cathodes')
    ax.set_ylabel('Frequency')
    ax.set_title('Cathode Count Distribution')
    ax.legend()
    
    # Scatter plots: predicted vs ground truth
    # Total electrodes
    ax = axes[1, 0]
    ax.scatter(gt_electrodes, baseline_electrodes, alpha=0.5, label='Baseline', s=30)
    ax.scatter(gt_electrodes, refined_electrodes, alpha=0.5, label='Refined', s=30)
    ax.plot([0, max(gt_electrodes)], [0, max(gt_electrodes)], 'k--', alpha=0.5)
    ax.set_xlabel('Ground Truth Count')
    ax.set_ylabel('Predicted Count')
    ax.set_title('Total Electrode Count: Predicted vs GT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Anodes
    ax = axes[1, 1]
    ax.scatter(gt_anodes, baseline_anodes, alpha=0.5, label='Baseline', s=30)
    ax.scatter(gt_anodes, refined_anodes, alpha=0.5, label='Refined', s=30)
    if max(gt_anodes) > 0:
        ax.plot([0, max(gt_anodes)], [0, max(gt_anodes)], 'k--', alpha=0.5)
    ax.set_xlabel('Ground Truth Count')
    ax.set_ylabel('Predicted Count')
    ax.set_title('Anode Count: Predicted vs GT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cathodes
    ax = axes[1, 2]
    ax.scatter(gt_cathodes, baseline_cathodes, alpha=0.5, label='Baseline', s=30)
    ax.scatter(gt_cathodes, refined_cathodes, alpha=0.5, label='Refined', s=30)
    if max(gt_cathodes) > 0:
        ax.plot([0, max(gt_cathodes)], [0, max(gt_cathodes)], 'k--', alpha=0.5)
    ax.set_xlabel('Ground Truth Count')
    ax.set_ylabel('Predicted Count')
    ax.set_title('Cathode Count: Predicted vs GT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Electrode Count Distributions and Comparisons', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'electrode_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_top_improved_visualizations(results: List[Dict], output_dir: str, max_vis: int = 20):
    """Create visualizations for top improved images."""
    if not results:
        return
    
    # Sort by total electrode improvement
    sorted_results = sorted(results, 
                          key=lambda x: x['improvements']['num_electrodes_error_improvement'], 
                          reverse=True)
    
    # Create comparison table
    top_results = sorted_results[:max_vis]
    
    fig, ax = plt.subplots(figsize=(12, len(top_results) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    headers = ['Image', 'Baseline\nElectrodes', 'Refined\nElectrodes', 'GT\nElectrodes', 
               'Error\nImprovement', 'Baseline\nAnodes', 'Refined\nAnodes', 'GT\nAnodes',
               'Anode\nImprovement']
    
    table_data = []
    for r in top_results:
        row = [
            r['image'][:20] + '...' if len(r['image']) > 20 else r['image'],
            f"{r['baseline_comparison']['pred_metrics']['num_electrodes']}",
            f"{r['refined_comparison']['pred_metrics']['num_electrodes']}",
            f"{r['baseline_comparison']['gt_metrics']['num_electrodes']}",
            f"{r['improvements']['num_electrodes_error_improvement']:+.3f}",
            f"{r['baseline_comparison']['pred_metrics']['num_anodes']}",
            f"{r['refined_comparison']['pred_metrics']['num_anodes']}",
            f"{r['baseline_comparison']['gt_metrics']['num_anodes']}",
            f"{r['improvements']['num_anodes_error_improvement']:+.3f}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color cells based on improvement
    for i in range(len(table_data)):
        # Color improvement cells
        cell = table[(i+1, 4)]
        value = float(table_data[i][4])
        if value > 0:
            cell.set_facecolor('#90EE90')  # Light green
        elif value < 0:
            cell.set_facecolor('#FFB6C1')  # Light red
            
        cell = table[(i+1, 8)]
        value = float(table_data[i][8])
        if value > 0:
            cell.set_facecolor('#90EE90')  # Light green
        elif value < 0:
            cell.set_facecolor('#FFB6C1')  # Light red
    
    plt.title(f'Top {len(top_results)} Images with Best Electrode Count Improvement', 
              fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_improved_table.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_error_distribution_plots(results: List[Dict], output_dir: str):
    """Create error distribution histograms."""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract errors
    baseline_errors = [r['baseline_comparison']['relative_errors']['num_electrodes_error'] for r in results]
    refined_errors = [r['refined_comparison']['relative_errors']['num_electrodes_error'] for r in results]
    improvements = [r['improvements']['num_electrodes_error_improvement'] for r in results]
    
    # Error distribution
    ax = axes[0, 0]
    ax.hist([baseline_errors, refined_errors], label=['Baseline', 'Refined'], 
            alpha=0.7, bins=20, edgecolor='black')
    ax.set_xlabel('Relative Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Electrode Count Error Distribution')
    ax.legend()
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Improvement distribution
    ax = axes[0, 1]
    ax.hist(improvements, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Error Improvement')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Improvement Distribution')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Add statistics
    stats_text = f'Mean: {np.mean(improvements):+.3f}\nStd: {np.std(improvements):.3f}\n'
    stats_text += f'Improved: {sum(1 for x in improvements if x > 0)}\n'
    stats_text += f'Degraded: {sum(1 for x in improvements if x < 0)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Box plot comparison
    ax = axes[1, 0]
    ax.boxplot([baseline_errors, refined_errors], labels=['Baseline', 'Refined'])
    ax.set_ylabel('Relative Error')
    ax.set_title('Error Distribution Comparison')
    ax.grid(True, alpha=0.3)
    
    # Cumulative improvement plot
    ax = axes[1, 1]
    sorted_improvements = sorted(improvements)
    cumulative = np.arange(1, len(sorted_improvements) + 1) / len(sorted_improvements)
    ax.plot(sorted_improvements, cumulative, linewidth=2)
    ax.set_xlabel('Error Improvement')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Error Improvement')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Add percentile markers
    percentiles = [25, 50, 75]
    for p in percentiles:
        value = np.percentile(sorted_improvements, p)
        ax.axhline(y=p/100, color='gray', linestyle=':', alpha=0.5)
        ax.text(value, p/100, f'{p}%: {value:.3f}', fontsize=8, ha='right')
    
    plt.suptitle('Electrode Metrics Error Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_individual_comparisons(results: List[Dict], output_dir: str, 
                                baseline_dir: str, refined_dir: str, gt_dir: str,
                                max_vis: int = 20):
    """Create individual mask comparison visualizations for top improved images."""
    if not results:
        return
    
    # Sort by total electrode improvement
    sorted_results = sorted(results, 
                          key=lambda x: x['improvements']['num_electrodes_error_improvement'], 
                          reverse=True)
    
    # Create subdirectories
    comparison_dir = os.path.join(output_dir, 'individual_comparisons')
    analysis_dir = os.path.join(output_dir, 'individual_analyses')
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    analyzer = ElectrodeAnalyzer()
    
    # Create visualizations for top improved images
    for i, result in enumerate(sorted_results[:max_vis]):
        image_name = result['image']
        
        # Load masks
        baseline_path = os.path.join(baseline_dir, f"{image_name}.png")
        refined_path = os.path.join(refined_dir, f"{image_name}.png")
        gt_path = os.path.join(gt_dir, f"{image_name}.png")
        
        if not all(os.path.exists(p) for p in [baseline_path, refined_path, gt_path]):
            continue
        
        baseline = cv2.imread(baseline_path, cv2.IMREAD_GRAYSCALE)
        refined = cv2.imread(refined_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if any(img is None for img in [baseline, refined, gt]):
            continue
        
        # Ensure same dimensions
        if refined.shape != gt.shape:
            refined = cv2.resize(refined, (gt.shape[1], gt.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        if baseline.shape != gt.shape:
            baseline = cv2.resize(baseline, (gt.shape[1], gt.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Create comparison visualization
        save_path = os.path.join(comparison_dir, f"comparison_{i+1}_{image_name}.png")
        analyzer.visualize_electrode_comparison(baseline, refined, gt, save_path)
        
        # Create individual analysis for each mask type
        for mask_type, mask in [('baseline', baseline), ('refined', refined), ('gt', gt)]:
            analysis_path = os.path.join(analysis_dir, f"analysis_{i+1}_{image_name}_{mask_type}.png")
            analyzer.visualize_electrode_analysis(mask, analysis_path)
    
    # Also create analyses for some random images (not just top improved)
    import random
    random_indices = random.sample(range(len(results)), min(max_vis, len(results)))
    
    for idx, i in enumerate(random_indices):
        result = results[i]
        image_name = result['image']
        
        # Load refined mask
        refined_path = os.path.join(refined_dir, f"{image_name}.png")
        if not os.path.exists(refined_path):
            continue
            
        refined = cv2.imread(refined_path, cv2.IMREAD_GRAYSCALE)
        if refined is None:
            continue
        
        # Create analysis visualization
        analysis_path = os.path.join(analysis_dir, f"random_{idx+1}_{image_name}_refined.png")
        analyzer.visualize_electrode_analysis(refined, analysis_path)
    
    print(f"Created {min(max_vis, len(sorted_results))} comparison visualizations")
    print(f"Created {min(max_vis*3, len(sorted_results)*3)} individual electrode analyses")


def evaluate_electrode_metrics(baseline_dir: str, refined_dir: str, gt_dir: str, 
                              output_path: str, create_visualizations: bool = True) -> Dict:
    """
    Evaluate electrode-specific metrics for a set of masks.
    
    Args:
        baseline_dir: Directory with baseline masks
        refined_dir: Directory with refined masks
        gt_dir: Directory with ground truth masks
        output_path: Path to save results
        create_visualizations: Whether to create visualization plots
        
    Returns:
        Dictionary with evaluation results
    """
    analyzer = ElectrodeAnalyzer()
    
    # Get list of masks
    mask_files = sorted([f for f in os.listdir(refined_dir) if f.endswith('.png')])
    
    if not mask_files:
        print("No masks found for evaluation")
        return {}
    
    results = []
    
    for mask_file in tqdm(mask_files, desc="Evaluating electrode metrics"):
        baseline_path = os.path.join(baseline_dir, mask_file)
        refined_path = os.path.join(refined_dir, mask_file)
        gt_path = os.path.join(gt_dir, mask_file)
        
        # Check if all files exist
        if not all(os.path.exists(p) for p in [baseline_path, refined_path, gt_path]):
            continue
        
        # Load masks
        baseline = cv2.imread(baseline_path, cv2.IMREAD_GRAYSCALE)
        refined = cv2.imread(refined_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if any(img is None for img in [baseline, refined, gt]):
            continue
        
        # Ensure same dimensions
        if refined.shape != gt.shape:
            refined = cv2.resize(refined, (gt.shape[1], gt.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        if baseline.shape != gt.shape:
            baseline = cv2.resize(baseline, (gt.shape[1], gt.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Compare metrics
        baseline_comparison = analyzer.compare_electrode_metrics(baseline, gt)
        refined_comparison = analyzer.compare_electrode_metrics(refined, gt)
        
        # Calculate improvements
        improvements = {}
        for key in refined_comparison['relative_errors']:
            baseline_error = baseline_comparison['relative_errors'][key]
            refined_error = refined_comparison['relative_errors'][key]
            improvements[key + '_improvement'] = baseline_error - refined_error
        
        result = {
            'image': mask_file.replace('.png', ''),
            'baseline_comparison': baseline_comparison,
            'refined_comparison': refined_comparison,
            'improvements': improvements
        }
        
        results.append(result)
    
    # Calculate summary statistics
    if results:
        summary = {
            'num_images': len(results),
            'avg_baseline_electrode_error': np.mean([r['baseline_comparison']['relative_errors']['num_electrodes_error'] for r in results]),
            'avg_refined_electrode_error': np.mean([r['refined_comparison']['relative_errors']['num_electrodes_error'] for r in results]),
            'avg_electrode_error_improvement': np.mean([r['improvements']['num_electrodes_error_improvement'] for r in results]),
            'avg_baseline_anode_error': np.mean([r['baseline_comparison']['relative_errors']['num_anodes_error'] for r in results]),
            'avg_refined_anode_error': np.mean([r['refined_comparison']['relative_errors']['num_anodes_error'] for r in results]),
            'avg_anode_error_improvement': np.mean([r['improvements']['num_anodes_error_improvement'] for r in results]),
            'avg_baseline_cathode_error': np.mean([r['baseline_comparison']['relative_errors']['num_cathodes_error'] for r in results]),
            'avg_refined_cathode_error': np.mean([r['refined_comparison']['relative_errors']['num_cathodes_error'] for r in results]),
            'avg_cathode_error_improvement': np.mean([r['improvements']['num_cathodes_error_improvement'] for r in results]),
        }
    else:
        summary = {}
    
    # Save results
    output = {
        'summary': summary,
        'detailed_results': results
    }
    
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Save CSV
    if results:
        # Flatten results for CSV
        csv_data = []
        for r in results:
            row = {'image': r['image']}
            
            # Add baseline metrics
            for k, v in r['baseline_comparison']['pred_metrics'].items():
                row[f'baseline_{k}'] = v
            
            # Add refined metrics
            for k, v in r['refined_comparison']['pred_metrics'].items():
                row[f'refined_{k}'] = v
            
            # Add improvements
            for k, v in r['improvements'].items():
                row[k] = v
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = output_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved electrode metrics CSV to: {csv_path}")
    
    # Create visualizations
    if create_visualizations and results:
        vis_output_dir = os.path.join(os.path.dirname(output_path), 'electrode_visualizations')
        create_electrode_metrics_visualization(output, vis_output_dir, 
                                             baseline_dir=baseline_dir,
                                             refined_dir=refined_dir,
                                             gt_dir=gt_dir)
    
    # Print summary
    if summary:
        print(f"\n{'='*60}")
        print("Electrode Metrics Summary")
        print(f"{'='*60}")
        print(f"Images evaluated: {summary['num_images']}")
        print(f"Avg electrode count error - Baseline: {summary['avg_baseline_electrode_error']:.3f}")
        print(f"Avg electrode count error - Refined: {summary['avg_refined_electrode_error']:.3f}")
        print(f"Avg electrode count error improvement: {summary['avg_electrode_error_improvement']:.3f}")
        print(f"Avg anode error - Baseline: {summary['avg_baseline_anode_error']:.3f}")
        print(f"Avg anode error - Refined: {summary['avg_refined_anode_error']:.3f}")
        print(f"Avg anode error improvement: {summary['avg_anode_error_improvement']:.3f}")
        print(f"Avg cathode error - Baseline: {summary['avg_baseline_cathode_error']:.3f}")
        print(f"Avg cathode error - Refined: {summary['avg_refined_cathode_error']:.3f}")
        print(f"Avg cathode error improvement: {summary['avg_cathode_error_improvement']:.3f}")
    
    return output


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate electrode-specific metrics')
    parser.add_argument('--baseline-dir', type=str, required=True,
                       help='Directory with baseline masks')
    parser.add_argument('--refined-dir', type=str, required=True,
                       help='Directory with refined masks')
    parser.add_argument('--gt-dir', type=str, required=True,
                       help='Directory with ground truth masks')
    parser.add_argument('--output', type=str, default='electrode_metrics.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    evaluate_electrode_metrics(args.baseline_dir, args.refined_dir, 
                              args.gt_dir, args.output)