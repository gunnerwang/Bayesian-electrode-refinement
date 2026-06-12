#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prior-Integration Refiner (legacy path, disabled by default)
Deep integration of LLM-generated Bayesian priors with the lattice refiner

This version ensures perfect compatibility with the lattice refiner when LLM is disabled:
1. Inherits directly from the lattice refiner
2. Only overrides necessary methods for LLM integration
3. All other behavior remains exactly as the lattice refiner
"""

import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from typing import Dict, List, Optional, Tuple, Any
import os
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# For LLM integration
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.stats import norm
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)
except ImportError:
    pass

# Import the base refiner to inherit from it (the lattice refiner inherits from the base refiner)
sys.path.append(os.path.dirname(__file__))
from bayesian_base import BayesianRefinerBase

# Import the lattice refiner classes directly
from lattice_refiner import ElectrodeLatticeModel, LatticeRefiner


class LLMBayesianPriorGenerator:
    """Generates Bayesian priors using LLM visual understanding."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, debug_mode: bool = False):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4o')  # Default to gpt-4o if not specified
        self.client = None
        self.debug_mode = debug_mode
        
        if self.api_key:
            try:
                from openai import OpenAI
                # Check if custom API base URL is specified
                api_base = os.getenv('OPENAI_API_BASE')
                if api_base:
                    self.client = OpenAI(api_key=self.api_key, base_url=api_base)
                else:
                    self.client = OpenAI(api_key=self.api_key)
                print(f"LLM Bayesian Prior Generator initialized with model: {self.model}")
            except ImportError:
                print("Warning: OpenAI package not installed")
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
    
    def generate_batch_bayesian_priors(self, batch_data: Dict[int, Dict[str, Any]], 
                                     unified_mask: Optional[np.ndarray] = None,
                                     unified_gray: Optional[np.ndarray] = None,
                                     unified_gt: Optional[np.ndarray] = None) -> Dict[int, Dict[str, Any]]:
        """Generate Bayesian priors for a batch of instances."""
        if not self.client:
            return self._get_default_bayesian_priors(batch_data)
        
        try:
            # Cache unified region data if provided
            if unified_mask is not None:
                self._unified_mask = unified_mask
                self._unified_gray = unified_gray
                self._unified_gt = unified_gt
            
            # Create visualization for LLM
            vis_base64 = self._prepare_bayesian_visualization(batch_data)
            
            # Clean up cached data
            if hasattr(self, '_unified_mask'):
                delattr(self, '_unified_mask')
            if hasattr(self, '_unified_gray'):
                delattr(self, '_unified_gray')
            if hasattr(self, '_unified_gt'):
                delattr(self, '_unified_gt')
            
            # Get LLM analysis
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_bayesian_prior_prompt()},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Analyze these {len(batch_data)} electrodes in the unified region and generate Bayesian priors for refinement:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{vis_base64}"}}
                    ]}
                ],
                max_completion_tokens=2000
            )
            
            # Parse response into Bayesian priors
            return self._parse_bayesian_priors(response.choices[0].message.content, batch_data)
            
        except Exception as e:
            print(f"LLM prior generation failed: {e}, using defaults")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return self._get_default_bayesian_priors(batch_data)
    
    def _prepare_bayesian_visualization(self, batch_data: Dict[int, Dict[str, Any]]) -> str:
        """Prepare visualization for Bayesian prior generation - show unified region."""
        # Reconstruct unified region mask and gray image
        if hasattr(self, '_unified_mask') and hasattr(self, '_unified_gray'):
            # Use cached unified region data
            unified_mask = self._unified_mask
            unified_gray = self._unified_gray
            unified_gt = self._unified_gt if hasattr(self, '_unified_gt') else None
        else:
            # Fallback to individual visualization
            return self._prepare_individual_visualization(batch_data)
        
        # Check if GT is available
        has_gt = unified_gt is not None
        
        # Create figure with appropriate layout
        if has_gt:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes = axes.reshape(1, -1)
        
        # Show current mask
        ax = axes[0, 0] if has_gt else axes[0, 0]
        ax.imshow(unified_mask, cmap='gray')
        ax.set_title(f'Current Mask ({len(batch_data)} electrodes)')
        ax.axis('off')
        
        # Show grayscale image
        ax = axes[0, 1] if has_gt else axes[0, 1]
        ax.imshow(unified_gray, cmap='gray')
        ax.set_title('Grayscale Image')
        ax.axis('off')
        
        # Show GT mask if available
        if has_gt:
            ax = axes[1, 0]
            ax.imshow(unified_gt, cmap='gray')
            ax.set_title('Ground Truth Mask')
            ax.axis('off')
            
            # Show overlay
            ax = axes[1, 1]
            overlay = np.zeros((unified_mask.shape[0], unified_mask.shape[1], 3))
            overlay[:,:,0] = unified_mask / 255.0  # Current in red
            overlay[:,:,1] = unified_gt / 255.0    # GT in green
            overlay[:,:,2] = 0
            ax.imshow(overlay)
            ax.set_title('Overlay (Red=Current, Green=GT)')
            ax.axis('off')
        else:
            # Show intensity histogram for all electrodes
            ax = axes[0, 2]
            
            # Compute overall statistics
            inside_pixels = unified_gray[unified_mask > 127]
            outside_pixels = unified_gray[unified_mask <= 127]
            
            if len(inside_pixels) > 0:
                ax.hist(inside_pixels, bins=50, alpha=0.5, label=f'Inside ({len(inside_pixels)} pixels)', density=True)
            if len(outside_pixels) > 0:
                ax.hist(outside_pixels, bins=50, alpha=0.5, label=f'Outside ({len(outside_pixels)} pixels)', density=True)
            
            ax.set_xlabel('Intensity')
            ax.set_ylabel('Density')
            ax.legend()
            ax.set_title('Overall Intensity Distribution')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _prepare_individual_visualization(self, batch_data: Dict[int, Dict[str, Any]]) -> str:
        """Fallback to individual instance visualization."""
        num_instances = len(batch_data)
        cols = min(3, num_instances)
        rows = (num_instances + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if num_instances == 1:
            axes = np.array([axes])
        axes = axes.reshape(-1)
        
        for idx, (inst_id, data) in enumerate(batch_data.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            if 'mask' in data and data['mask'] is not None:
                ax.imshow(data['mask'], cmap='gray')
                ax.set_title(f'Instance {inst_id}')
            else:
                ax.text(0.5, 0.5, f'Instance {inst_id}\nNo data', ha='center', va='center')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_instances, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _get_bayesian_prior_prompt(self) -> str:
        """Prompt for Bayesian prior generation."""
        return """You are an expert in Bayesian image analysis for electrode refinement.
Analyze the unified region containing multiple electrode masks to generate probability priors for Bayesian inference.

The image shows:
1. Current Mask - all electrodes in the unified region
2. Grayscale Image - the intensity data
3. Ground Truth (if available) - the target refinement
4. Intensity Distribution - histogram of pixel intensities

Your task is to generate Bayesian priors that will guide the refinement of each electrode.
Consider:
- Electrodes should be darker than the background
- Boundaries should be smooth but preserve electrode shape
- Adjacent electrodes should remain separated
- Thickness should be consistent along each electrode

For each electrode (numbered 0, 1, 2, etc.), provide:
1. Electrode type (long/short/medium based on aspect ratio)
2. Intensity model parameters (Gaussian distributions for inside/outside)
3. Edge confidence (how reliable are the current boundaries)
4. Shape regularity (how regular/smooth the shape should be)
5. Separation requirements (probability that electrodes need separation)
6. Boundary protection (areas that should be preserved during refinement)

Output JSON format with instance IDs as keys:
{
    "0": {
        "electrode_type": "long",  // "long", "short", or "medium"
        "intensity_model": {
            "inside": {"mean": 50, "std": 15, "confidence": 0.9},
            "outside": {"mean": 150, "std": 25, "confidence": 0.8},
            "separability": 0.85
        },
        "edge_prior": {
            "confidence": 0.8,
            "smoothness": 0.7,
            "sharpness": 0.6
        },
        "shape_prior": {
            "regularity": 0.8,
            "compactness": 0.7,
            "symmetry": 0.6
        },
        "separation_prior": {
            "needs_separation": 0.2,
            "separation_strength": 0.5
        },
        "boundary_protection": {
            "protect_long_edges": true,  // For long electrodes
            "protection_strength": 0.8    // 0-1, higher means stronger protection
        }
    },
    "1": { ... }
}"""
    
    def _parse_bayesian_priors(self, response: str, batch_data: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Parse LLM response into Bayesian priors."""
        try:
            # Try to extract JSON more robustly
            import re
            
            # First try to find JSON block in code fence
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                # Try to find raw JSON
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1) if '```' in json_match.group(0) else json_match.group(0)
                
                # Clean up common JSON errors
                json_str = json_str.strip()
                # Remove trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                # Remove comments
                json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
                
                priors_data = json.loads(json_str)
                
                # Convert to internal format
                bayesian_priors = {}
                for inst_id, data in batch_data.items():
                    inst_id_str = str(inst_id)
                    if inst_id_str in priors_data:
                        bayesian_priors[inst_id] = priors_data[inst_id_str]
                    else:
                        # Use defaults for missing instances
                        bayesian_priors[inst_id] = self._get_default_prior_for_instance(data)
                
                return bayesian_priors
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM priors (JSON error at line {e.lineno}, col {e.colno}): {e.msg}")
            if self.debug_mode:
                print(f"Response excerpt: {response[:500]}...")
        except Exception as e:
            print(f"Failed to parse LLM priors: {e}")
        
        return self._get_default_bayesian_priors(batch_data)
    
    def _get_default_bayesian_priors(self, batch_data: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Generate default Bayesian priors when LLM is unavailable."""
        priors = {}
        for inst_id, data in batch_data.items():
            priors[inst_id] = self._get_default_prior_for_instance(data)
        return priors
    
    def _get_default_prior_for_instance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate default prior for a single instance."""
        # Analyze the mask and image if available
        mask = data['mask']
        gray = data.get('gray')
        
        # Default intensity model
        intensity_model = {
            "inside": {"mean": 50, "std": 20, "confidence": 0.7},
            "outside": {"mean": 150, "std": 30, "confidence": 0.7},
            "separability": 0.7
        }
        
        if gray is not None:
            # Compute actual statistics
            inside = gray[mask > 127]
            # Create a dilated mask to sample nearby outside pixels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated_mask = cv2.dilate((mask > 127).astype(np.uint8), kernel)
            outside_mask = (dilated_mask > 0) & (mask <= 127)
            outside = gray[outside_mask]
            
            if len(inside) > 10 and len(outside) > 10:
                inside_mean = float(np.mean(inside))
                outside_mean = float(np.mean(outside))
                
                # Check if we need to swap (electrodes should be darker than background)
                if inside_mean > outside_mean:
                    # Swap if inside is brighter than outside
                    intensity_model["inside"]["mean"] = outside_mean
                    intensity_model["inside"]["std"] = float(np.std(outside) + 1e-6)
                    intensity_model["outside"]["mean"] = inside_mean
                    intensity_model["outside"]["std"] = float(np.std(inside) + 1e-6)
                else:
                    intensity_model["inside"]["mean"] = inside_mean
                    intensity_model["inside"]["std"] = float(np.std(inside) + 1e-6)
                    intensity_model["outside"]["mean"] = outside_mean
                    intensity_model["outside"]["std"] = float(np.std(outside) + 1e-6)
                
                # Compute separability
                sep = abs(intensity_model["inside"]["mean"] - intensity_model["outside"]["mean"]) / \
                      (intensity_model["inside"]["std"] + intensity_model["outside"]["std"])
                intensity_model["separability"] = float(np.clip(sep / 2, 0, 1))
        
        # Determine electrode type based on aspect ratio
        if mask is not None:
            y_coords, x_coords = np.where(mask > 127)
            if len(x_coords) > 0 and len(y_coords) > 0:
                width = np.max(x_coords) - np.min(x_coords) + 1
                height = np.max(y_coords) - np.min(y_coords) + 1
                aspect_ratio = height / max(width, 1)
                
                if aspect_ratio > 4:  # Consistent with enforce_minimum_separation
                    electrode_type = "long"
                elif aspect_ratio < 2:
                    electrode_type = "short"
                else:
                    electrode_type = "medium"
            else:
                electrode_type = "medium"
        else:
            electrode_type = "medium"
        
        return {
            "electrode_type": electrode_type,
            "intensity_model": intensity_model,
            "edge_prior": {
                "confidence": 0.7,
                "smoothness": 0.6,
                "sharpness": 0.6
            },
            "shape_prior": {
                "regularity": 0.7,
                "compactness": 0.6,
                "symmetry": 0.5
            },
            "separation_prior": {
                "needs_separation": 0.3,
                "separation_strength": 0.5
            },
            "boundary_protection": {
                "protect_long_edges": electrode_type == "long",
                "protection_strength": 0.8 if electrode_type == "long" else 0.5
            }
        }


class PriorIntegrationRefiner(LatticeRefiner):
    """Prior-integration refiner: exact LatticeRefiner behavior with optional LLM prior injection."""
    
    def __init__(self, 
                 enable_llm_priors: bool = False,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 debug_mode: bool = False,
                 **kwargs):
        """Initialize the prior-integration refiner with the lattice refiner parameters plus LLM settings."""
        # Initialize the lattice refiner with all its parameters
        super().__init__(**kwargs)
        
        # Store LLM-specific parameters
        self.enable_llm_priors = enable_llm_priors
        self.api_key = api_key
        self.model = model
        self.debug_mode = debug_mode
        
        # Initialize LLM prior generator if enabled
        if self.enable_llm_priors:
            self.prior_generator = LLMBayesianPriorGenerator(self.api_key, self.model, self.debug_mode)
            self._llm_priors_cache = {}  # Cache for LLM priors
        
        print("Prior-integration refiner: LLM-Bayesian Deep Integration")
        print(f"  - LLM Bayesian priors: {'Enabled' if self.enable_llm_priors else 'Disabled (plain lattice-refiner behavior)'}")
        print(f"  - Base: the lattice refiner with enhanced Bayesian inference")
    
    def refine_instance_vectorized(self, mask: np.ndarray,
                                   features: Dict[str, np.ndarray],
                                   structural_constraint: np.ndarray,
                                   use_multiscale: bool = True) -> np.ndarray:
        """Override the base refiner's method to inject LLM priors when available."""
        # If LLM is disabled or no priors cached, use the lattice refiner's original method
        if not self.enable_llm_priors or not hasattr(self, '_current_llm_prior'):
            return super().refine_instance_vectorized(mask, features, structural_constraint, use_multiscale)
        
        # Use LLM-enhanced refinement
        llm_prior = self._current_llm_prior
        
        # Extract features
        gray = features['gray']
        edges = features['edges']
        
        # Get LLM priors
        intensity_model = llm_prior.get('intensity_model', {})
        edge_prior = llm_prior.get('edge_prior', {})
        shape_prior = llm_prior.get('shape_prior', {})
        
        # Scale-aware parameters modified by LLM confidence
        edge_confidence = edge_prior.get('confidence', 0.7)
        intensity_weight = self.intensity_weight * intensity_model.get('separability', 0.7)
        structure_weight = self.structure_weight * shape_prior.get('regularity', 0.7)
        
        # Intensity likelihood with LLM priors
        if intensity_model and 'inside' in intensity_model and 'outside' in intensity_model:
            # Use LLM-provided distribution parameters
            mu_in = intensity_model['inside']['mean']
            sigma_in = max(intensity_model['inside']['std'], 1.0)  # Ensure minimum std
            mu_out = intensity_model['outside']['mean']
            sigma_out = max(intensity_model['outside']['std'], 1.0)  # Ensure minimum std
            
            # Confidence weights
            conf_in = intensity_model['inside'].get('confidence', 0.8)
            conf_out = intensity_model['outside'].get('confidence', 0.8)
        else:
            # Fall back to data-driven estimation
            inside = gray[mask > 127]
            # Create dilated mask to sample nearby outside pixels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated_mask = cv2.dilate((mask > 127).astype(np.uint8), kernel)
            outside_mask = (dilated_mask > 0) & (mask <= 127)
            outside = gray[outside_mask]
            
            if len(inside) > 0 and len(outside) > 0:
                mu_in, sigma_in = np.mean(inside), np.std(inside) + 1e-6
                mu_out, sigma_out = np.mean(outside), np.std(outside) + 1e-6
                conf_in = conf_out = 1.0
            else:
                # No data, use flat prior
                return mask
        
        # Compute likelihood with LLM confidence weighting
        p_in = norm.pdf(gray, mu_in, sigma_in) * conf_in
        p_out = norm.pdf(gray, mu_out, sigma_out) * conf_out
        
        # Normalize to get probability
        intensity_likelihood = p_in / (p_in + p_out + 1e-10)
        
        # Edge likelihood enhanced by LLM edge prior
        edge_likelihood = edges * edge_confidence + (1 - edge_confidence) * 0.5
        
        # Combine likelihoods
        likelihood = intensity_weight * intensity_likelihood + (1 - intensity_weight) * edge_likelihood
        
        # Shape prior from LLM
        compactness = shape_prior.get('compactness', 0.6)
        smoothness = shape_prior.get('regularity', 0.7)
        
        # Distance transform for shape prior
        dist_transform = distance_transform_edt(mask > 127)
        max_dist = np.max(dist_transform)
        if max_dist > 0:
            # Shape prior favors compact, smooth shapes
            shape_penalty = 1 - np.exp(-dist_transform / (max_dist * compactness))
            shape_prior_map = 1 - shape_penalty * (1 - smoothness)
        else:
            shape_prior_map = np.ones_like(mask, dtype=float)
        
        # Compute posterior
        posterior = likelihood * shape_prior_map * structural_constraint
        
        # Apply smoothness constraint from LLM
        smoothness = edge_prior.get('smoothness', 0.6)
        if smoothness > 0.5:
            sigma = 1 + smoothness
            posterior = gaussian_filter(posterior, sigma)
        
        # Debug output
        if self.debug_mode:
            print(f"  LLM refinement debug:")
            print(f"    - Mask shape: {mask.shape}, non-zero pixels in mask: {np.sum(mask > 127)}")
            print(f"    - Intensity model: inside={mu_in:.1f}±{sigma_in:.1f}, outside={mu_out:.1f}±{sigma_out:.1f}")
            print(f"    - Likelihood range: [{np.min(likelihood):.3f}, {np.max(likelihood):.3f}]")
            print(f"    - Posterior range: [{np.min(posterior):.3f}, {np.max(posterior):.3f}]")
            print(f"    - Non-zero posterior pixels: {np.sum(posterior > 0.1)}")
            
            # Check intensity distribution
            inside_pixels = gray[mask > 127]
            outside_pixels = gray[mask <= 127]
            if len(inside_pixels) > 0:
                print(f"    - Actual inside mean: {np.mean(inside_pixels):.1f}±{np.std(inside_pixels):.1f}")
            if len(outside_pixels) > 0:
                print(f"    - Actual outside mean: {np.mean(outside_pixels):.1f}±{np.std(outside_pixels):.1f}")
            
            # Check what percentage of mask is being classified as electrode
            high_posterior = posterior > 0.5
            print(f"    - Pixels with posterior > 0.5: {np.sum(high_posterior)} ({100*np.sum(high_posterior)/posterior.size:.1f}%)")
        
        # Scale posterior to 0-255 range for compatibility with compute_adaptive_threshold
        posterior_scaled = (posterior * 255).astype(np.uint8)
        
        # Adaptive thresholding
        threshold = self.compute_adaptive_threshold(posterior_scaled, mask)
        
        if self.debug_mode:
            print(f"    - Adaptive threshold: {threshold:.3f}")
            print(f"    - Pixels above threshold: {np.sum(posterior_scaled > threshold)}")
            print(f"    - Original mask pixels: {np.sum(mask > 127)}")
            print(f"    - Ratio refined/original: {np.sum(posterior_scaled > threshold) / max(1, np.sum(mask > 127)):.2f}")
        
        refined = (posterior_scaled > threshold).astype(np.uint8) * 255
        
        # Safety check: if refinement resulted in too few pixels, fall back to original
        if np.sum(refined > 127) < 50:  # Less than 50 pixels
            if self.debug_mode:
                print(f"    WARNING: LLM refinement resulted in only {np.sum(refined > 127)} pixels, falling back to the lattice refiner")
            # Fall back to parent's method
            return super().refine_instance_vectorized(mask, features, structural_constraint, use_multiscale)
        
        # Post-processing based on separation prior
        sep_prior = llm_prior.get('separation_prior', {})
        if sep_prior.get('needs_separation', 0) > 0.5:
            # Apply stronger morphological operations to separate
            kernel_size = int(3 + 2 * sep_prior.get('separation_strength', 0.5))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        
        return refined
    
    def apply_base_refinement(self, masks_dict: Dict[int, np.ndarray], 
                           image: np.ndarray) -> Dict[int, np.ndarray]:
        """Override the lattice refiner's method to generate and inject LLM priors."""
        # If LLM is disabled, use the lattice refiner's original implementation exactly
        if not self.enable_llm_priors:
            # Call the lattice refiner's apply_base_refinement directly, which includes thickness control
            return super().apply_base_refinement(masks_dict, image)
        
        print("[Prior] Applying LLM-enhanced base refinement...")
        
        # Get positions from instance_masks that were passed in create_unified_region
        # masks_dict contains masks in the unified region coordinate system
        # We need the original positions to extract local regions for LLM
        if not hasattr(self, '_current_instance_data'):
            # If position data not available, fall back to the lattice refiner
            print("[Prior] No position data available, falling back to the lattice refiner")
            return super().apply_base_refinement(masks_dict, image)
        
        # Prepare batch data for LLM
        batch_data = {}
        # image here is the unified region image, not the full image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        for inst_id, mask in masks_dict.items():
            # Get instance region from instance_masks which has the original bbox
            if inst_id not in self._current_instance_data:
                continue
                
            orig_bbox = self._current_instance_data[inst_id].get('bbox')
            rel_pos = self._current_instance_data[inst_id].get('relative_pos')
            
            if not orig_bbox or not rel_pos:
                continue
                
            # Extract mask using original bbox dimensions from relative position
            y1, x1 = rel_pos[1], rel_pos[0]
            y2, x2 = y1 + orig_bbox[3], x1 + orig_bbox[2]  # height, width
            
            # Add margin for context
            margin = 30
            y_min_m = max(0, y1 - margin)
            y_max_m = min(gray.shape[0], y2 + margin)
            x_min_m = max(0, x1 - margin)
            x_max_m = min(gray.shape[1], x2 + margin)
            
            # Extract local mask and gray regions with margin
            mask_local = mask[y_min_m:y_max_m, x_min_m:x_max_m]
            gray_local = gray[y_min_m:y_max_m, x_min_m:x_max_m]
            
            batch_data[inst_id] = {
                'mask': mask_local,
                'gray': gray_local,
                'gt_mask': self.gt_masks.get(inst_id)[y_min_m:y_max_m, x_min_m:x_max_m] if hasattr(self, 'gt_masks') and inst_id in self.gt_masks else None
            }
        
        # Prepare unified region mask for visualization
        unified_mask = np.zeros_like(gray)
        for inst_id, m in masks_dict.items():
            unified_mask = np.maximum(unified_mask, m)
        
        # Prepare unified GT mask if available
        unified_gt = None
        if hasattr(self, 'gt_masks') and self.gt_masks:
            unified_gt = np.zeros_like(gray)
            for inst_id, gt in self.gt_masks.items():
                if gt.shape == gray.shape:
                    unified_gt = np.maximum(unified_gt, gt)
        
        # Get LLM priors for batch with unified visualization
        llm_priors_batch = self.prior_generator.generate_batch_bayesian_priors(
            batch_data, unified_mask, gray, unified_gt
        )
        
        # Cache LLM priors for use in separation enforcement
        self._llm_priors_cache = llm_priors_batch
        
        if self.debug_mode:
            print(f"  Generated priors for {len(llm_priors_batch)} instances")
        
        # Process each instance with its LLM prior
        refined_masks = {}
        features = self.compute_unified_features(image)
        
        for inst_id, mask in masks_dict.items():
            # Get position info for this instance
            if inst_id not in self._current_instance_data:
                continue
                
            orig_bbox = self._current_instance_data[inst_id].get('bbox')
            rel_pos = self._current_instance_data[inst_id].get('relative_pos')
            
            if not orig_bbox or not rel_pos:
                # Fall back to processing full mask if no position info
                self._current_llm_prior = llm_priors_batch.get(inst_id, {})
                refined = self.refine_instance_vectorized(
                    mask, features, np.ones_like(mask, dtype=np.float32),
                    use_multiscale=(mask.shape[0] > 50 and mask.shape[1] > 50)
                )
                refined_masks[inst_id] = refined
                continue
            
            # Extract local region for this electrode
            y1, x1 = rel_pos[1], rel_pos[0]
            y2, x2 = y1 + orig_bbox[3], x1 + orig_bbox[2]
            
            # Extract local mask
            local_mask = mask[y1:y2, x1:x2].copy()
            
            # Extract local features
            local_features = {
                'gray': features['gray'][y1:y2, x1:x2],
                'edges': features['edges'][y1:y2, x1:x2]
            }
            
            # Set current LLM prior for this instance
            self._current_llm_prior = llm_priors_batch.get(inst_id, {})
            
            if self.debug_mode:
                print(f"\n  Processing instance {inst_id}:")
                print(f"    - Local mask shape: {local_mask.shape}")
                print(f"    - Non-zero pixels in local mask: {np.sum(local_mask > 127)}")
                print(f"    - Position: ({y1}, {x1})")
            
            # Refine local mask
            local_constraint = np.ones_like(local_mask, dtype=np.float32)
            refined_local = self.refine_instance_vectorized(
                local_mask, local_features, local_constraint,
                use_multiscale=(local_mask.shape[0] > 50 and local_mask.shape[1] > 50)
            )
            
            # Put refined local mask back in full mask
            refined_full = np.zeros_like(mask)
            refined_full[y1:y2, x1:x2] = refined_local
            
            refined_masks[inst_id] = refined_full
        
        # Clear current prior
        if hasattr(self, '_current_llm_prior'):
            delattr(self, '_current_llm_prior')
        
        # Apply thickness control to the base refiner results as the lattice refiner does
        print(f"[Lattice] Applying final thickness control to {len(refined_masks)} masks")
        final_masks = self._apply_final_thickness_control(refined_masks)
        
        return final_masks
    
    def create_unified_region(self, instance_data: Dict, full_image: np.ndarray) -> Dict:
        """Override to store bbox information for LLM processing."""
        # Call parent's create_unified_region
        region_data = super().create_unified_region(instance_data, full_image)
        
        # Add bbox information to instance_masks
        for inst_id in region_data['instance_masks']:
            if inst_id in instance_data and 'bbox' in instance_data[inst_id]:
                region_data['instance_masks'][inst_id]['bbox'] = instance_data[inst_id]['bbox']
        
        return region_data
    
    def refine_unified_region(self, region_data: Dict) -> Dict[int, np.ndarray]:
        """Use the lattice refiner's method with GT mask support and position data for LLM."""
        # Store GT masks if provided
        if 'gt_masks' in region_data:
            self.gt_masks = region_data['gt_masks']
        
        # Store instance position data for LLM processing
        # region_data contains 'instance_masks' which has relative positions
        if 'instance_masks' in region_data:
            self._current_instance_data = region_data['instance_masks']
        
        # Call the lattice refiner's original method
        result = super().refine_unified_region(region_data)
        
        # Clean up
        if hasattr(self, 'gt_masks'):
            delattr(self, 'gt_masks')
        if hasattr(self, '_current_instance_data'):
            delattr(self, '_current_instance_data')
            
        return result
    
    def _enforce_minimum_separation(self, masks: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Override the lattice refiner's method to use LLM priors for smarter separation."""
        # If LLM is disabled or no priors available, use parent's method
        if not self.enable_llm_priors or not hasattr(self, '_llm_priors_cache'):
            return super()._enforce_minimum_separation(masks)
        
        print("[Prior] Using LLM-enhanced separation enforcement")
        
        # Get electrode types from LLM priors
        electrode_types = {}
        for inst_id in masks.keys():
            if inst_id in self._llm_priors_cache:
                electrode_types[inst_id] = self._llm_priors_cache[inst_id].get('electrode_type', 'medium')
            else:
                # Fallback: determine type from aspect ratio
                mask = masks[inst_id]
                y_coords, x_coords = np.where(mask > 127)
                if len(x_coords) > 0 and len(y_coords) > 0:
                    width = np.max(x_coords) - np.min(x_coords) + 1
                    height = np.max(y_coords) - np.min(y_coords) + 1
                    aspect_ratio = height / max(width, 1)
                    
                    if aspect_ratio > 4:  # Lowered threshold to catch more long electrodes
                        electrode_types[inst_id] = "long"
                    elif aspect_ratio < 2:
                        electrode_types[inst_id] = "short"
                    else:
                        electrode_types[inst_id] = "medium"
                else:
                    electrode_types[inst_id] = "medium"
        
        # Sort masks by x position, but prioritize long electrodes
        def sort_key(item):
            inst_id, mask = item
            x_pos = np.mean(np.where(mask > 127)[1]) if np.sum(mask > 127) > 0 else 0
            # Long electrodes get priority (lower sort value)
            priority = 0 if electrode_types[inst_id] == "long" else 1
            return (priority, x_pos)
        
        sorted_items = sorted(masks.items(), key=sort_key)
        
        refined = {}
        accumulated_forbidden = None
        long_electrode_regions = {}  # Track long electrode regions
        
        # First pass: process long electrodes
        for inst_id, mask in sorted_items:
            if electrode_types[inst_id] != "long":
                continue
                
            mask_binary = (mask > 127).astype(np.uint8)
            
            # Long electrodes get minimal processing
            if accumulated_forbidden is not None:
                # Only remove actual overlaps, not dilated regions
                mask_binary = mask_binary & (~accumulated_forbidden)
            
            # Store long electrode region for protection
            long_electrode_regions[inst_id] = mask_binary > 0
            
            # Update accumulated forbidden region
            if accumulated_forbidden is None:
                accumulated_forbidden = mask_binary > 0
            else:
                accumulated_forbidden = accumulated_forbidden | (mask_binary > 0)
            
            refined[inst_id] = mask_binary.astype(np.uint8) * 255
        
        # Second pass: process short/medium electrodes
        for inst_id, mask in sorted_items:
            if electrode_types[inst_id] == "long":
                continue  # Already processed
                
            mask_binary = (mask > 127).astype(np.uint8)
            
            if accumulated_forbidden is not None:
                # Remove overlap with all previous masks
                mask_binary = mask_binary & (~accumulated_forbidden)
                
                # Apply separation with dilation, but protect long electrode regions
                kernel = np.ones((1, self.min_separation_pixels), dtype=np.uint8)
                forbidden_dilated = cv2.dilate(accumulated_forbidden.astype(np.uint8), kernel)
                
                # Don't apply dilated separation near long electrodes
                for long_id, long_region in long_electrode_regions.items():
                    # Create larger protection zone around long electrodes
                    # Use a taller kernel to protect the vertical extent of long electrodes
                    protection_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9))
                    protected_area = cv2.dilate(long_region.astype(np.uint8), protection_kernel)
                    
                    # In protected areas, completely disable dilated separation
                    # This ensures short electrodes cannot "bite into" long electrodes
                    mask_in_protected = mask_binary & (protected_area > 0)
                    if np.any(mask_in_protected):
                        # If this mask overlaps with protected area, use more conservative approach
                        # Only remove direct overlaps with the long electrode itself
                        overlap_with_long = mask_binary & long_region
                        if np.any(overlap_with_long):
                            # Remove only the overlapping pixels
                            mask_binary = mask_binary & (~long_region)
                        # Skip dilated separation in protected areas
                        forbidden_dilated = np.where(protected_area > 0, 
                                                   0,  # No forbidden zone in protected area
                                                   forbidden_dilated)
                
                mask_binary = mask_binary & (~forbidden_dilated.astype(bool))
            
            # Update accumulated forbidden region
            if accumulated_forbidden is None:
                accumulated_forbidden = mask_binary > 0
            else:
                accumulated_forbidden = accumulated_forbidden | (mask_binary > 0)
            
            refined[inst_id] = mask_binary.astype(np.uint8) * 255
        
        if self.debug_mode:
            print(f"[Prior] Separation enforcement summary:")
            for inst_id, electrode_type in electrode_types.items():
                original_pixels = np.sum(masks[inst_id] > 127)
                refined_pixels = np.sum(refined[inst_id] > 127)
                print(f"  Instance {inst_id} ({electrode_type}): {original_pixels} -> {refined_pixels} pixels")
        
        return refined


def main():
    """Test the prior-integration refiner; this main mirrors the lattice refiner's."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prior-integration refiner: LLM-enhanced Bayesian refinement')
    parser.add_argument('--data-dir', type=str, 
                       default='data/Full_Instances',
                       help='Base directory containing instance data')
    parser.add_argument('--output-dir', type=str,
                       default='results/refined_masks_prior',
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
    parser.add_argument("--use-gt", action="store_true",
                       help="Use ground truth masks to guide refinement")
    parser.add_argument("--thickness-factor", type=float, default=1.0,
                       help="Thickness control factor (0.1-2.0, lower=thinner)")
    parser.add_argument('--origin-dir', type=str,
                       default='data/Origin',
                       help='Directory containing original full images and masks')
    
    args = parser.parse_args()
    
    # Initialize the prior-integration refiner (parameters adapted from the lattice refiner's main; deviations noted below)
    refiner = PriorIntegrationRefiner(
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
    
    print(f"Processing {len(info_files)} images with the prior-integration refiner...")
    
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
            gt_masks = {}
            
            for inst in group:
                inst_id = inst['id']
                mask_path = os.path.join(mask_dir, f"{base_name}_instance_{inst_id}.png")
                
                if os.path.exists(mask_path):
                    instance_data[inst_id] = {
                        'bbox': inst['bbox'],
                        'mask_path': mask_path
                    }
                    
                    # Load GT if requested
                    if args.use_gt:
                        gt_path = os.path.join(args.data_dir, 'masks', 
                                             f"{base_name}_instance_{inst_id}.png")
                        if os.path.exists(gt_path):
                            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                            if gt_mask is not None:
                                gt_masks[inst_id] = gt_mask
            
            if not instance_data:
                continue
                
            print(f"Processing group with {len(instance_data)} instances...")
            
            # Create unified region
            region_data = refiner.create_unified_region(instance_data, full_image)
            
            # Add GT masks if available
            if gt_masks:
                region_data['gt_masks'] = gt_masks
            
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
    
    print("Prior-integration refinement complete!")


if __name__ == '__main__':
    main()