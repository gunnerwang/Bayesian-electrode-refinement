#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Guided Refiner
LLM-guided parameter adaptation for electrode refinement

Key improvements:
1. LLM analyzes input and suggests optimal parameters for each processing stage
2. Dynamic adjustment of thickness_factor, transform_ratio, separation parameters
3. Per-electrode parameter customization based on pattern analysis
4. Adaptive refinement strategy based on detected issues
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

# Import OpenAI for LLM integration
try:
    import openai
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if OPENAI_API_KEY:
        if hasattr(openai, 'OpenAI'):
            # New API (v1.0+)
            openai.api_key = OPENAI_API_KEY
        else:
            # Old API (v0.28)
            openai.api_key = OPENAI_API_KEY
    HAS_OPENAI = bool(OPENAI_API_KEY)
except ImportError:
    HAS_OPENAI = False
    print("[LLMGuided] OpenAI not available - LLM parameter adaptation disabled")


class LLMParameterAdvisor:
    """LLM-guided parameter advisor for electrode refinement"""
    
    def __init__(self, calibration=None):
        """calibration: optional dict (or path to a JSON file) holding the
        dataset calibration block computed from a small labeled tuning set
        (schema: docs/calibration.md). When provided, the advisor uses the
        calibrated prompt; without it, the legacy anchored prompt is used.
        """
        # Read model from environment, fallback to gpt-4o-mini
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        if os.getenv('OPENAI_BASE_URL') and not os.getenv('OPENAI_MODEL'):
            print("[LLMGuided] WARNING: OPENAI_BASE_URL is set but OPENAI_MODEL is not. "
                  f"The default model '{self.model}' will be requested from the custom "
                  "endpoint and will likely fail; set OPENAI_MODEL to a model that "
                  "endpoint actually serves.")
        self.enabled = HAS_OPENAI
        if isinstance(calibration, str):
            with open(calibration) as f:
                calibration = json.load(f)
        self.calibration = calibration
        
    def analyze_and_suggest_parameters(self, electrodes: List[Dict], 
                                     graph_analysis: Dict,
                                     region_image: np.ndarray,
                                     gt_masks: Optional[Dict[int, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Analyze electrode patterns and suggest optimal parameters for each processing stage.
        
        Returns:
            Dict with suggested parameters for each stage:
            - thickness_factor: float (0.1-2.0)
            - transform_ratio: float (0.0-1.0) 
            - transform_intensity: float (0.0-1.0)
            - min_separation: int (pixels)
            - lattice_params: dict with model-specific parameters
        """
        if not self.enabled or len(electrodes) < 3:
            return self._get_default_params()
            
        try:
            # Prepare analysis data
            analysis_data = self._prepare_analysis_data(electrodes, graph_analysis, gt_masks)
            
            # Create prompt
            prompt = self._create_parameter_prompt(analysis_data)
            print(f"[LLMGuided] Prompt length: {len(prompt)} chars")
            
            # Get LLM suggestions
            if hasattr(openai, 'OpenAI'):
                # New API (v1.0+)
                client = openai.OpenAI()
                
                # Build base parameters
                base_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert in electrode pattern analysis and image processing parameter optimization."},
                        {"role": "user", "content": prompt}
                    ]
                }
                
                # Try different parameter combinations based on model
                for attempt in range(3):
                    try:
                        params = base_params.copy()
                        
                        if attempt == 0:
                            # First try: newer models with max_completion_tokens and temperature
                            params["max_completion_tokens"] = 1500
                            params["temperature"] = 0.1
                        elif attempt == 1:
                            # Second try: older models with max_tokens
                            params["max_tokens"] = 1500
                            params["temperature"] = 0.1
                        else:
                            # Third try: minimal parameters (for models like gpt-5-nano)
                            # No token limit or temperature parameters
                            pass  # params already has base parameters only
                        
                        response = client.chat.completions.create(**params)
                        content = response.choices[0].message.content
                        if self.enabled:  # Add debug output
                            print(f"[LLMGuided] LLM responded successfully on attempt {attempt + 1}")
                            print(f"[LLMGuided] Response length: {len(content)} chars")
                            if len(content) < 50:
                                print(f"[LLMGuided] Short response: {content}")
                        break
                    except Exception as e:
                        if attempt < 2:
                            if self.enabled:  # Add debug output
                                error_msg = str(e)
                                # Extract just the error type if it's too long
                                if 'Unsupported parameter' in error_msg:
                                    print(f"[LLMGuided] LLM attempt {attempt + 1} failed: Unsupported parameter issue")
                                elif 'Unsupported value' in error_msg:
                                    print(f"[LLMGuided] LLM attempt {attempt + 1} failed: Unsupported value issue")
                                else:
                                    print(f"[LLMGuided] LLM attempt {attempt + 1} failed: {error_msg[:100]}")
                            continue
                        else:
                            raise e
            else:
                # Old API (v0.28)
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in electrode pattern analysis and image processing parameter optimization."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500
                )
                content = response.choices[0].message.content
            
            # Check if response is empty
            if not content or len(content.strip()) == 0:
                print(f"[LLMGuided] WARNING: Model '{self.model}' returned empty response!")
                print("[LLMGuided] This may be a mock/test model. Consider using a real OpenAI model.")
                print("[LLMGuided] Falling back to rule-based adaptation...")
                return self._get_rule_based_params(analysis_data)
            
            # Parse response
            suggestions = self._parse_llm_response(content)
            
            # If LLM returned content but failed to parse as JSON, try rule-based
            if suggestions.get('reasoning', '').startswith('Using default parameters'):
                print("[LLMGuided] LLM response couldn't be parsed as parameters, using rule-based adaptation")
                suggestions = self._get_rule_based_params(analysis_data)
            
            print(f"\n[LLMGuided] LLM Parameter Suggestions:")
            print(f"  - Thickness factor: {suggestions['thickness_factor']:.2f}")
            print(f"  - Transform ratio: {suggestions['transform_ratio']:.2f}")
            print(f"  - Transform intensity: {suggestions['transform_intensity']:.2f}")
            print(f"  - Min separation: {suggestions['min_separation']}px")
            if 'reasoning' in suggestions:
                print(f"  - Reasoning: {suggestions['reasoning'][:100]}...")
            
            return suggestions
            
        except Exception as e:
            print(f"[LLMGuided] LLM analysis error: {str(e)}")
            return self._get_default_params()
    
    def _prepare_analysis_data(self, electrodes: List[Dict], 
                              graph_analysis: Dict,
                              gt_masks: Optional[Dict[int, np.ndarray]]) -> Dict:
        """Prepare electrode data for LLM analysis"""
        # Calculate statistics
        widths = [e['width'] for e in electrodes]
        areas = [e.get('area', e['width'] * 100) for e in electrodes]
        
        # Analyze width distribution
        width_stats = {
            'mean': np.mean(widths),
            'std': np.std(widths),
            'min': np.min(widths),
            'max': np.max(widths),
            'cv': np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 0
        }
        
        # Analyze spacing
        if 'adjacency_matrix' in graph_analysis and graph_analysis['adjacency_matrix'] is not None:
            adj_matrix = graph_analysis['adjacency_matrix']
            distances = []
            for i in range(len(electrodes)):
                for j in range(i+1, len(electrodes)):
                    if adj_matrix[i, j] > 0:
                        dist = abs(electrodes[j]['center_x'] - electrodes[i]['center_x'])
                        distances.append(dist)
            
            spacing_stats = {
                'mean': np.mean(distances) if distances else 0,
                'std': np.std(distances) if distances else 0,
                'min': np.min(distances) if distances else 0
            } if distances else {'mean': 0, 'std': 0, 'min': 0}
        else:
            spacing_stats = {'mean': 0, 'std': 0, 'min': 0}
        
        # GT comparison if available
        gt_comparison = None
        if gt_masks:
            gt_widths = []
            for inst_id, gt_mask in gt_masks.items():
                y_coords, x_coords = np.where(gt_mask > 127)
                if len(x_coords) > 0:
                    width = np.max(x_coords) - np.min(x_coords) + 1
                    gt_widths.append(width)
            
            if gt_widths:
                gt_comparison = {
                    'mean_width': np.mean(gt_widths),
                    'width_ratio': np.mean(widths) / np.mean(gt_widths) if np.mean(gt_widths) > 0 else 1
                }
        
        return {
            'num_electrodes': len(electrodes),
            'pattern_type': graph_analysis.get('pattern_type', 'irregular'),
            'width_stats': width_stats,
            'spacing_stats': spacing_stats,
            'gt_comparison': gt_comparison
        }
    
    def _render_calib_block(self) -> str:
        iv = self.calibration['input_vs_gt']
        lines = [
            "Dataset calibration (computed from ground-truth-labeled tuning images "
            "of this dataset; use it as a prior and adapt to the per-image statistics above):",
            f"- The INPUT masks under-cover the true electrodes: vs GT they have "
            f"precision {iv['precision_mean']:.2f} but recall only {iv['recall_mean']:.2f} "
            f"(instance IoU {iv['iou_mean']:.2f}); true electrodes are on average "
            f"{iv['gt_over_input_width_ratio']:.2f}x wider than the input masks.",
            "- Probe configurations evaluated on the tuning images "
            "(per-instance IoU / precision / recall of the refined output vs GT):",
        ]
        for prb in self.calibration['probes']:
            pp = prb['params']
            lines.append(
                f"    thickness_factor={pp['thickness_factor']}, transform_ratio={pp['transform_ratio']}, "
                f"transform_intensity={pp['transform_intensity']}, min_separation={pp['min_separation']}"
                f"  ->  IoU {prb['refined_iou_mean']:.3f}, precision {prb['refined_precision_mean']:.3f}, "
                f"recall {prb['refined_recall_mean']:.3f}")
        return "\n".join(lines)

    def _create_calibrated_prompt(self, analysis_data: Dict) -> str:
        """Calibrated prompt: per-image statistics + tuning-set calibration
        evidence; all parameter decisions are left to the model (no anchored
        defaults). Same output schema as the legacy prompt."""
        prompt = f"""Analyze the following electrode pattern and suggest optimal refinement parameters:

Electrode Pattern Analysis:
- Number of electrodes: {analysis_data['num_electrodes']}
- Pattern type: {analysis_data['pattern_type']}
- Width statistics:
  - Mean: {analysis_data['width_stats']['mean']:.1f} pixels
  - Std: {analysis_data['width_stats']['std']:.1f} pixels
  - Range: [{analysis_data['width_stats']['min']:.0f}, {analysis_data['width_stats']['max']:.0f}] pixels
  - Coefficient of variation: {analysis_data['width_stats']['cv']:.2f}
"""
        if analysis_data['spacing_stats']['mean'] > 0:
            prompt += f"""
- Spacing statistics:
  - Mean: {analysis_data['spacing_stats']['mean']:.1f} pixels
  - Std: {analysis_data['spacing_stats']['std']:.1f} pixels
  - Min: {analysis_data['spacing_stats']['min']:.0f} pixels
"""
        prompt += f"""
{self._render_calib_block()}

Parameters to choose (decide each value from the evidence above; there is no
preferred default — pick what maximizes refined-vs-GT instance IoU):

1. thickness_factor (0.1-2.0): scales electrode thickness in the refined output.
   Values below ~1.0 shrink electrodes, above ~1.0 expand them.
2. transform_ratio (0.0-1.0): fraction of electrodes receiving curvature transformation.
3. transform_intensity (0.0-1.0): strength of the curvature transformation.
4. min_separation (pixels, 2-20): minimum enforced spacing between electrodes;
   consider the observed minimum spacing.
5. lattice_params:
   - max_width_ratio (1.5-2.5): maximum allowed width variation
   - neighbor_consistency_weight (0.1-0.5)
   - width_consistency_weight (0.1-0.3)

Provide your suggestions in JSON format:
{{
  "thickness_factor": <value>,
  "transform_ratio": <value>,
  "transform_intensity": <value>,
  "min_separation": <value>,
  "lattice_params": {{
    "max_width_ratio": <value>,
    "neighbor_consistency_weight": <value>,
    "width_consistency_weight": <value>
  }},
  "reasoning": "<brief explanation>"
}}
"""
        return prompt

    def _create_parameter_prompt(self, analysis_data: Dict) -> str:
        """Create prompt for LLM parameter suggestion (calibrated prompt when a
        calibration block is provided; legacy anchored prompt otherwise)."""
        if self.calibration:
            return self._create_calibrated_prompt(analysis_data)
        prompt = f"""Analyze the following electrode pattern and suggest optimal refinement parameters:

Electrode Pattern Analysis:
- Number of electrodes: {analysis_data['num_electrodes']}
- Pattern type: {analysis_data['pattern_type']}
- Width statistics:
  - Mean: {analysis_data['width_stats']['mean']:.1f} pixels
  - Std: {analysis_data['width_stats']['std']:.1f} pixels  
  - Range: [{analysis_data['width_stats']['min']:.0f}, {analysis_data['width_stats']['max']:.0f}] pixels
  - Coefficient of variation: {analysis_data['width_stats']['cv']:.2f}
"""

        if analysis_data['spacing_stats']['mean'] > 0:
            prompt += f"""
- Spacing statistics:
  - Mean: {analysis_data['spacing_stats']['mean']:.1f} pixels
  - Std: {analysis_data['spacing_stats']['std']:.1f} pixels
  - Min: {analysis_data['spacing_stats']['min']:.0f} pixels
"""

        if analysis_data['gt_comparison']:
            prompt += f"""
- Ground truth comparison:
  - GT mean width: {analysis_data['gt_comparison']['mean_width']:.1f} pixels
  - Current/GT width ratio: {analysis_data['gt_comparison']['width_ratio']:.2f}
"""

        prompt += """
Based on this analysis, suggest optimal parameters:

1. thickness_factor (0.1-2.0): Controls overall electrode thickness. Lower values make electrodes thinner.
   - If current widths are larger than GT, suggest lower values
   - For high width variation (CV > 0.3), use moderate values
   - Default: 0.5 (IMPORTANT: Only change if there's clear evidence of width issues)

2. transform_ratio (0.0-1.0): Fraction of electrodes to apply curvature transformation.
   - For regular patterns (grid/linear), use moderate values (0.6-0.8)
   - For irregular patterns with high variation, use higher values (0.8-1.0)
   - Default: 0.8 (IMPORTANT: Avoid going below 0.6 unless pattern is extremely regular)

3. transform_intensity (0.0-1.0): Strength of curvature transformation.
   - For regular electrodes, use 0.8-1.0
   - Only reduce for very minor corrections
   - Default: 1.0 (IMPORTANT: Keep high unless electrodes are already well-shaped)

4. min_separation (pixels): Minimum spacing between electrodes.
   - Should be based on observed minimum spacing
   - Typically 3-10 pixels
   - Default: 5

5. Additional lattice model parameters:
   - max_width_ratio: Maximum allowed width variation (1.5-2.5)
   - neighbor_consistency_weight: Weight for neighbor consistency (0.1-0.5)
   - width_consistency_weight: Weight for width consistency (0.1-0.3)

Provide your suggestions in JSON format:
{
  "thickness_factor": <value>,
  "transform_ratio": <value>,
  "transform_intensity": <value>,
  "min_separation": <value>,
  "lattice_params": {
    "max_width_ratio": <value>,
    "neighbor_consistency_weight": <value>,
    "width_consistency_weight": <value>
  },
  "reasoning": "<brief explanation>"
}
"""
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract parameters"""
        try:
            # Debug: print first part of response
            print(f"[LLMGuided] LLM response preview: {response_text[:200]}...")
            
            # Extract JSON from response
            import re
            # More flexible JSON extraction - look for any JSON object
            json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    params = json.loads(json_str)
                    # Check if this is our parameter JSON (has required fields)
                    if 'thickness_factor' in params or 'transform_ratio' in params:
                        print(f"[LLMGuided] Found parameter JSON: {json_str[:100]}...")
                        
                        # Validate and clamp values
                        params['thickness_factor'] = np.clip(params.get('thickness_factor', 0.5), 0.1, 2.0)
                        params['transform_ratio'] = np.clip(params.get('transform_ratio', 0.8), 0.0, 1.0)
                        params['transform_intensity'] = np.clip(params.get('transform_intensity', 1.0), 0.0, 1.0)
                        params['min_separation'] = int(np.clip(params.get('min_separation', 5), 2, 20))
                        
                        # Ensure lattice params exist
                        if 'lattice_params' not in params:
                            params['lattice_params'] = {}
                        
                        lattice = params['lattice_params']
                        lattice['max_width_ratio'] = np.clip(lattice.get('max_width_ratio', 1.8), 1.5, 2.5)
                        lattice['neighbor_consistency_weight'] = np.clip(lattice.get('neighbor_consistency_weight', 0.15), 0.1, 0.5)
                        lattice['width_consistency_weight'] = np.clip(lattice.get('width_consistency_weight', 0.15), 0.1, 0.3)
                        
                        # Add reasoning if not present
                        if 'reasoning' not in params:
                            params['reasoning'] = 'Parameters extracted from LLM response'
                        
                        print(f"[LLMGuided] Successfully parsed LLM parameters")
                        return params
                except json.JSONDecodeError:
                    continue
            
            print(f"[LLMGuided] No valid parameter JSON found in response")
        except Exception as e:
            print(f"[LLMGuided] Parse error: {str(e)[:100]}")
        
        return self._get_default_params()
    
    def _get_rule_based_params(self, analysis_data: Dict) -> Dict[str, Any]:
        """Get parameters based on rule-based analysis when LLM fails"""
        params = self._get_default_params()
        
        width_stats = analysis_data.get('width_stats', {})
        spacing_stats = analysis_data.get('spacing_stats', {})
        gt_comparison = analysis_data.get('gt_comparison')
        pattern_type = analysis_data.get('pattern_type', 'irregular')
        
        reasoning = []
        
        # Adjust thickness factor based on GT comparison
        if gt_comparison and gt_comparison['width_ratio'] < 0.7:
            # Electrodes are too thin compared to GT
            params['thickness_factor'] = 0.8
            reasoning.append("Increased thickness factor (electrodes thinner than GT)")
        elif gt_comparison and gt_comparison['width_ratio'] > 1.3:
            # Electrodes are too thick compared to GT
            params['thickness_factor'] = 0.3
            reasoning.append("Decreased thickness factor (electrodes thicker than GT)")
        
        # Adjust transform ratio based on pattern type
        if pattern_type == 'grid':
            params['transform_ratio'] = 0.5  # Less transformation for regular grids
            params['lattice_params']['max_width_ratio'] = 1.5  # Stricter width control
            reasoning.append("Regular grid pattern - reduced transformation")
        elif pattern_type == 'linear':
            params['transform_ratio'] = 0.6
            reasoning.append("Linear pattern - moderate transformation")
        else:
            params['transform_ratio'] = 0.8  # More transformation for irregular patterns
            reasoning.append("Irregular pattern - higher transformation")
        
        # Adjust based on width variation
        cv = width_stats.get('cv', 0)
        if cv > 0.3:
            # High variation - need more consistency
            params['lattice_params']['width_consistency_weight'] = 0.25
            params['lattice_params']['neighbor_consistency_weight'] = 0.3
            reasoning.append(f"High width variation (CV={cv:.2f}) - increased consistency")
        
        # Adjust min separation based on observed spacing
        if spacing_stats and spacing_stats['min'] > 0:
            params['min_separation'] = max(3, min(10, int(spacing_stats['min'] * 0.8)))
            reasoning.append(f"Adjusted min separation based on observed ({spacing_stats['min']:.0f}px)")
        
        params['reasoning'] = "; ".join(reasoning) if reasoning else "Rule-based adaptation"
        return params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters when LLM is not available"""
        return {
            'thickness_factor': 0.5,
            'transform_ratio': 0.8,
            'transform_intensity': 1.0,
            'min_separation': 5,
            'lattice_params': {
                'max_width_ratio': 1.8,
                'neighbor_consistency_weight': 0.15,
                'width_consistency_weight': 0.15
            },
            'reasoning': 'Using default parameters (LLM not available)'
        }


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
        print(f"[LLMGuided] Detected lattice pattern: {self.lattice_pattern['type']}")
        
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


class LLMGuidedRefiner(GuidedRefiner):
    """
    LLM-guided per-image parameter adaptive refinement
    
    Inherits from the guided refiner and adds:
    1. LLM parameter advisor for dynamic parameter optimization
    2. Graph-enhanced lattice model (graph relationship analysis + MST consistency)
    3. Per-region parameter customization based on electrode patterns
    """
    
    def __init__(self, *args, **kwargs):
        # Extract LLM-specific parameters
        self.use_llm_advisor = kwargs.pop('use_llm_advisor', True)
        advisor_calibration = kwargs.pop('advisor_calibration', None)
        
        # Initialize parent (the guided refiner)
        super().__init__(*args, **kwargs)
        
        # Initialize LLM parameter advisor
        self.param_advisor = (LLMParameterAdvisor(calibration=advisor_calibration)
                              if self.use_llm_advisor else None)
        
        # Start with default lattice model - will be updated per region
        self._init_default_lattice_model()
        
        # Store per-region parameters
        self.region_params = {}
        
        print("\n[LLMGuided] Initialized with LLM-guided parameter adaptation")
        if self.use_llm_advisor and HAS_OPENAI:
            print(f"      LLM advisor: Enabled (model: {self.param_advisor.model})")
            if self.param_advisor.calibration:
                print("      Advisor prompt: CALIBRATED (dataset calibration loaded)")
            else:
                print("      Advisor prompt: LEGACY anchored — no calibration loaded "
                      "(consider --calibration, see docs/calibration.md)")
        else:
            print(f"      LLM advisor: Disabled")
        print("      (Inheritance: LLMGuidedRefiner → GuidedRefiner → PatternRefiner → PriorIntegrationRefiner → LatticeRefiner → BayesianRefinerBase, with adaptive parameters)")
    
    def _init_default_lattice_model(self):
        """Initialize default lattice model"""
        self.lattice_model = GraphEnhancedElectrodeLatticeModel(
            min_spacing=self.min_separation_pixels,
            max_width_ratio=1.8,
            width_consistency_weight=0.15,
            boundary_smoothness_weight=0.1,
            use_soft_boundaries=True,
            edge_smoothing_iterations=0,
            vertical_continuity_weight=0.15,
            use_shape_regularization=True,
            use_skeleton_guidance=True,
            use_graph_analysis=True,
            graph_distance_threshold=80.0,
            use_mst_propagation=True,
            neighbor_consistency_weight=0.15
        )


    def refine_unified_region(self, region_data: Dict) -> Dict[int, np.ndarray]:
        """Override to add LLM parameter adaptation"""
        
        # Extract electrode data for analysis
        electrodes = []
        for inst_id, mask_data in region_data['instance_masks'].items():
            mask = mask_data['mask']
            y_coords, x_coords = np.where(mask > 127)
            if len(x_coords) > 0:
                electrodes.append({
                    'id': inst_id,
                    'width': np.max(x_coords) - np.min(x_coords) + 1,
                    'center_x': np.mean(x_coords),
                    'center_y': np.mean(y_coords) if len(y_coords) > 0 else 0,
                    'area': np.sum(mask > 127)
                })
        
        if self.param_advisor and len(electrodes) >= 3:
            # Build graph for analysis
            graph = GraphElectrodeRelationship()
            graph.build_from_electrodes(electrodes)
            
            # Get pattern analysis
            pattern_analyzer = LatticePatternAnalyzer()
            pattern = pattern_analyzer.analyze_pattern(electrodes)
            
            # Prepare graph analysis data
            graph_data = {
                'adjacency_matrix': graph.adjacency_matrix,
                'pattern_type': pattern['type'],
                'pattern_params': pattern['params']
            }
            
            # Load GT masks if available
            gt_masks = None
            if self.gt_full_dir and 'metadata' in region_data['instance_masks'][electrodes[0]['id']]:
                image_name = region_data['instance_masks'][electrodes[0]['id']]['metadata']['image_name']
                # Create temporary region data for GT loading
                temp_region_data = {
                    'instance_masks': {e['id']: region_data['instance_masks'][e['id']] for e in electrodes},
                    'bounds': region_data.get('bounds', (0, 0, region_data['image'].shape[1], region_data['image'].shape[0]))
                }
                gt_masks = self._load_gt_masks(temp_region_data)
            
            # Get LLM suggestions
            params = self.param_advisor.analyze_and_suggest_parameters(
                electrodes, graph_data, region_data['image'], gt_masks
            )
            
            # Update instance parameters
            print(f"\n[LLMGuided] Updating refinement parameters:")
            print(f"  - Thickness factor: {self.thickness_factor:.2f} -> {params['thickness_factor']:.2f}")
            print(f"  - Transform ratio: {self.transform_ratio:.2f} -> {params['transform_ratio']:.2f}")
            print(f"  - Transform intensity: {self.transform_intensity:.2f} -> {params['transform_intensity']:.2f}")
            print(f"  - Min separation: {self.min_separation_pixels} -> {params['min_separation']}")
            
            self.thickness_factor = params['thickness_factor']
            self.transform_ratio = params['transform_ratio']
            self.transform_intensity = params['transform_intensity']
            self.min_separation_pixels = params['min_separation']
            
            # Re-initialize lattice model with suggested parameters
            lattice_params = params['lattice_params']
            self.lattice_model = GraphEnhancedElectrodeLatticeModel(
                min_spacing=self.min_separation_pixels,
                max_width_ratio=lattice_params['max_width_ratio'],
                width_consistency_weight=lattice_params['width_consistency_weight'],
                boundary_smoothness_weight=0.1,  # Keep fixed
                use_soft_boundaries=True,
                edge_smoothing_iterations=0,
                vertical_continuity_weight=0.15,  # Keep fixed
                use_shape_regularization=True,
                use_skeleton_guidance=True,
                use_graph_analysis=True,
                graph_distance_threshold=80.0,
                use_mst_propagation=True,
                neighbor_consistency_weight=lattice_params['neighbor_consistency_weight']
            )
            
            # Store parameters for this region
            self.region_params[id(region_data)] = params
        
        # Call parent refinement with adapted parameters
        return super().refine_unified_region(region_data)


def main():
    """Run the LLM-guided refiner with parameter adaptation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM-guided adaptive refinement')
    parser.add_argument('--data-dir', type=str, 
                       default='data/Full_Instances',
                       help='Base directory containing instance data')
    parser.add_argument('--output-dir', type=str,
                       default='results/refined_masks_llm_guided',
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
                       default='results/llm_guided_visualizations',
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
    parser.add_argument('--calibration', type=str, default=None,
                        help='JSON file with the dataset calibration block (docs/calibration.md); '
                             'enables the calibrated advisor prompt')
    parser.add_argument('--no-use-llm', action='store_true',
                       help='Disable LLM parameter advisor (enabled by default)')
    
    args = parser.parse_args()
    
    # Initialize the LLM-guided refiner with LLM parameter advisor
    refiner = LLMGuidedRefiner(
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
        transform_intensity=args.transform_intensity,
        use_llm_advisor=not args.no_use_llm,
        advisor_calibration=args.calibration
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
    
    print(f"Processing {len(info_files)} images with the LLM-guided refiner...")
    
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
            
            # Refine using the LLM-guided refiner (the guided refiner with LLM-guided parameters)
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
    
    print("\nLLM-guided refinement complete!")


if __name__ == "__main__":
    main()