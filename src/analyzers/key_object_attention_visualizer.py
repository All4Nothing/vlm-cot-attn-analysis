"""
Key Object Attention Visualizer

Step 4: Load annotated JSON with key_objects and generate real-time attention visualizations
Uses inference.py to generate object-specific queries and visualize attention maps.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union
import logging
from PIL import Image
import cv2
import sys
import torch

# Add models path to sys.path for importing model modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'llava'))
from inference import LLaVAInferenceEngine
from config import Config

logger = logging.getLogger(__name__)

class KeyObjectAttentionVisualizer:
    """
    Visualizer for key object attention patterns
    
    This class handles Step 4 of the workflow:
    1. Load JSON with user-annotated key_objects
    2. Generate object-specific queries in real-time
    3. Analyze attention patterns for each key object
    4. Create attention map visualizations
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the visualizer
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.inference_engine = LLaVAInferenceEngine(self.config)
        self.annotated_data = None
        
    def load_model(self):
        """Load the LLaVA model"""
        logger.info("Loading model for attention visualization...")
        self.inference_engine.load_model()
        
    def unload_model(self):
        """Unload the model to free memory"""
        self.inference_engine.unload_model()
    
    def load_annotated_results(self, json_path: str) -> bool:
        """
        Load JSON file with user-annotated key_objects
        
        Args:
            json_path: Path to the annotated JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.annotated_data = json.load(f)
            
            # Validate data structure
            if "results" not in self.annotated_data:
                logger.error("Invalid JSON structure: 'results' key not found")
                return False
            
            # Check for key_objects annotations
            results_with_objects = []
            for result in self.annotated_data["results"]:
                if 'error' not in result and result.get('key_objects'):
                    results_with_objects.append(result)
            
            logger.info(f"Loaded {len(self.annotated_data['results'])} total results")
            logger.info(f"Found {len(results_with_objects)} images with key_objects annotations")
            
            if not results_with_objects:
                logger.warning("No images have key_objects annotations. Please add key_objects to the JSON file.")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading annotated results: {str(e)}")
            return False
    
    def generate_object_query(self, 
                            object_name: str,
                            query_template: str = "Where is the {object} in this image? Point out its location and describe it.") -> str:
        """
        Generate object-specific query
        
        Args:
            object_name: Name of the object
            query_template: Template for generating queries
            
        Returns:
            Generated query string
        """
        return query_template.format(object=object_name)
    
    def analyze_object_attention_from_saved_data(self, 
                                                saved_attention_data: Dict[str, Any],
                                                object_name: str,
                                                description: str,
                                                generated_ids: List = None) -> Dict[str, Any]:
        """
        Analyze attention for a specific object using saved attention data from Step 1-2
        
        Args:
            saved_attention_data: Attention data saved from description generation
            object_name: Name of the object to analyze
            description: Generated description text
            generated_ids: Generated token IDs
            
        Returns:
            Dictionary containing attention analysis results
        """
        try:
            logger.info(f"Analyzing '{object_name}' from saved attention data")
            
            if not saved_attention_data or not saved_attention_data.get('attentions'):
                return {"error": "No saved attention data available"}
            
            # Find object tokens in the generated description
            object_token_positions = self._find_object_tokens_in_description(
                description, object_name, generated_ids
            )
            
            if not object_token_positions:
                return {"error": f"Object '{object_name}' not found in description"}
            
            # Analyze attention patterns for object tokens
            attention_analysis = self._analyze_object_attention_patterns(
                saved_attention_data['attentions'],
                object_name,
                object_token_positions,
                image_token_start=0,
                image_token_end=576  # LLaVA image token length
            )
            
            return {
                "object_name": object_name,
                "object_tokens_found": object_token_positions,
                "description": description,
                "attention_analysis": attention_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing object attention from saved data: {str(e)}")
            return {"error": str(e)}
    
    def _find_object_tokens_in_description(self, 
                                          description: str, 
                                          object_name: str, 
                                          generated_ids: List = None) -> List[int]:
        """
        Find token positions where the object appears in the generated description
        
        Args:
            description: Generated description text
            object_name: Name of the object to find
            generated_ids: Generated token IDs (if available)
            
        Returns:
            List of token positions where object appears
        """
        try:
            # Simple approach: find object name in description text
            # More sophisticated tokenization matching could be added later
            
            positions = []
            description_lower = description.lower()
            object_lower = object_name.lower()
            
            # Find all occurrences of the object name
            start = 0
            while True:
                pos = description_lower.find(object_lower, start)
                if pos == -1:
                    break
                
                # Estimate token position (rough approximation)
                # In a more sophisticated version, we'd use the actual tokenizer
                words_before = len(description[:pos].split())
                positions.append(words_before)
                start = pos + 1
            
            logger.info(f"Found '{object_name}' at estimated token positions: {positions}")
            return positions
            
        except Exception as e:
            logger.error(f"Error finding object tokens: {str(e)}")
            return []
    
    def _analyze_object_attention_patterns(self, 
                                         attentions: List,
                                         object_name: str,
                                         object_token_positions: List[int],
                                         image_token_start: int = 0,
                                         image_token_end: int = 576) -> Dict[str, Any]:
        """
        Analyze attention patterns for specific object tokens
        
        Args:
            attentions: Attention tensors from model generation
            object_name: Name of the object being analyzed
            object_token_positions: Token positions where object appears
            image_token_start: Start index of image tokens
            image_token_end: End index of image tokens
            
        Returns:
            Dictionary containing attention analysis
        """
        try:
            if not attentions:
                return {"error": "No attention data available"}
            
            analysis = {
                "object_name": object_name,
                "object_token_positions": object_token_positions,
                "generation_steps": len(attentions),
                "image_token_range": (image_token_start, image_token_end),
                "step_analysis": [],
                "summary": {
                    "avg_image_attention": [],
                    "max_image_attention": [],
                    "top_attended_patches": []
                }
            }
            
            all_patch_attentions = {}  # Accumulate attention for each patch
            
            # Analyze each generation step
            for step_idx, step_attentions in enumerate(attentions):
                step_data = {
                    "step": step_idx,
                    "layers": []
                }
                
                step_image_attentions = []
                
                # Analyze each layer
                for layer_idx, layer_attention in enumerate(step_attentions):
                    if layer_attention is not None and hasattr(layer_attention, 'shape'):
                        if len(layer_attention.shape) == 4:  # [batch, heads, seq_len, seq_len]
                            
                            # Get attention from OBJECT TOKENS to image tokens
                            object_attentions = []
                            for token_pos in object_token_positions:
                                if token_pos < layer_attention.shape[2]:  # Check bounds
                                    # Get attention from object token to image tokens
                                    object_token_attention = layer_attention[0, :, token_pos, image_token_start:image_token_end]
                                    object_attentions.append(object_token_attention)
                            
                            if object_attentions:
                                # Average across object token occurrences and heads
                                combined_attention = torch.stack(object_attentions).mean(dim=0)  # Average across tokens
                                avg_attention_per_patch = combined_attention.mean(dim=0)  # Average across heads
                                
                                # Accumulate patch attentions
                                for patch_idx, attn_val in enumerate(avg_attention_per_patch):
                                    if patch_idx not in all_patch_attentions:
                                        all_patch_attentions[patch_idx] = []
                                    all_patch_attentions[patch_idx].append(attn_val.item())
                                
                                # Layer statistics
                                avg_attention = combined_attention.mean().item()
                                max_attention = combined_attention.max().item()
                                
                                layer_data = {
                                    "layer": layer_idx,
                                    "avg_attention_to_image": avg_attention,
                                    "max_attention_to_image": max_attention,
                                    "attention_variance": combined_attention.var().item()
                                }
                                
                                step_data["layers"].append(layer_data)
                                step_image_attentions.append(avg_attention)
                
                # Step summary
                if step_image_attentions:
                    step_summary = {
                        "avg_image_attention": np.mean(step_image_attentions),
                        "max_image_attention": np.max(step_image_attentions),
                        "std_image_attention": np.std(step_image_attentions)
                    }
                    step_data["summary"] = step_summary
                    
                    analysis["summary"]["avg_image_attention"].append(np.mean(step_image_attentions))
                    analysis["summary"]["max_image_attention"].append(np.max(step_image_attentions))
                
                analysis["step_analysis"].append(step_data)
            
            # Find top attended patches across all steps
            if all_patch_attentions:
                patch_scores = {}
                for patch_idx, attn_values in all_patch_attentions.items():
                    patch_scores[patch_idx] = {
                        "avg_attention": np.mean(attn_values),
                        "max_attention": np.max(attn_values),
                        "frequency": len(attn_values),
                        "patch_coordinates": (patch_idx // 24, patch_idx % 24)  # 24x24 grid
                    }
                
                # Sort by average attention
                top_patches = sorted(patch_scores.items(), 
                                   key=lambda x: x[1]["avg_attention"], 
                                   reverse=True)[:10]
                
                analysis["summary"]["top_attended_patches"] = [
                    {
                        "patch_index": patch_idx,
                        "coordinates": data["patch_coordinates"],
                        "avg_attention": data["avg_attention"],
                        "max_attention": data["max_attention"],
                        "frequency": data["frequency"]
                    }
                    for patch_idx, data in top_patches
                ]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing object attention patterns: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_attention_patterns(self, 
                                  attentions: List,
                                  object_name: str,
                                  image_token_start: int = 0,
                                  image_token_end: int = 576) -> Dict[str, Any]:
        """
        Analyze attention patterns from the last token to image patches
        
        Args:
            attentions: Attention tensors from model generation
            object_name: Name of the object being analyzed
            image_token_start: Start index of image tokens
            image_token_end: End index of image tokens
            
        Returns:
            Dictionary containing attention analysis
        """
        try:
            if not attentions:
                return {"error": "No attention data available"}
            
            analysis = {
                "object_name": object_name,
                "generation_steps": len(attentions),
                "image_token_range": (image_token_start, image_token_end),
                "step_analysis": [],
                "summary": {
                    "avg_image_attention": [],
                    "max_image_attention": [],
                    "top_attended_patches": []
                }
            }
            
            all_patch_attentions = {}  # Accumulate attention for each patch
            
            # Analyze each generation step
            for step_idx, step_attentions in enumerate(attentions):
                step_data = {
                    "step": step_idx,
                    "layers": []
                }
                
                step_image_attentions = []
                
                # Analyze each layer
                for layer_idx, layer_attention in enumerate(step_attentions):
                    if layer_attention is not None and hasattr(layer_attention, 'shape'):
                        if len(layer_attention.shape) == 4:  # [batch, heads, seq_len, seq_len]
                            # Get attention from LAST TOKEN to image tokens
                            last_token_attention = layer_attention[0, :, -1, image_token_start:image_token_end]
                            
                            # Average across heads
                            avg_attention_per_patch = last_token_attention.mean(dim=0)  # [576]
                            
                            # Accumulate patch attentions
                            for patch_idx, attn_val in enumerate(avg_attention_per_patch):
                                if patch_idx not in all_patch_attentions:
                                    all_patch_attentions[patch_idx] = []
                                all_patch_attentions[patch_idx].append(attn_val.item())
                            
                            # Layer statistics
                            avg_attention = last_token_attention.mean().item()
                            max_attention = last_token_attention.max().item()
                            
                            layer_data = {
                                "layer": layer_idx,
                                "avg_attention_to_image": avg_attention,
                                "max_attention_to_image": max_attention,
                                "attention_variance": last_token_attention.var().item()
                            }
                            
                            step_data["layers"].append(layer_data)
                            step_image_attentions.append(avg_attention)
                
                # Step summary
                if step_image_attentions:
                    step_summary = {
                        "avg_image_attention": np.mean(step_image_attentions),
                        "max_image_attention": np.max(step_image_attentions),
                        "std_image_attention": np.std(step_image_attentions)
                    }
                    step_data["summary"] = step_summary
                    
                    analysis["summary"]["avg_image_attention"].append(np.mean(step_image_attentions))
                    analysis["summary"]["max_image_attention"].append(np.max(step_image_attentions))
                
                analysis["step_analysis"].append(step_data)
            
            # Find top attended patches across all steps
            if all_patch_attentions:
                patch_scores = {}
                for patch_idx, attn_values in all_patch_attentions.items():
                    patch_scores[patch_idx] = {
                        "avg_attention": np.mean(attn_values),
                        "max_attention": np.max(attn_values),
                        "frequency": len(attn_values),
                        "patch_coordinates": (patch_idx // 24, patch_idx % 24)  # 24x24 grid
                    }
                
                # Sort by average attention
                top_patches = sorted(patch_scores.items(), 
                                   key=lambda x: x[1]["avg_attention"], 
                                   reverse=True)[:10]
                
                analysis["summary"]["top_attended_patches"] = [
                    {
                        "patch_index": patch_idx,
                        "coordinates": data["patch_coordinates"],
                        "avg_attention": data["avg_attention"],
                        "max_attention": data["max_attention"],
                        "frequency": data["frequency"]
                    }
                    for patch_idx, data in top_patches
                ]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing attention patterns: {str(e)}")
            return {"error": str(e)}
    
    def create_attention_visualization(self, 
                                     image_path: str,
                                     object_name: str,
                                     attention_analysis: Dict[str, Any],
                                     save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create attention visualization for an object
        
        Args:
            image_path: Path to the original image
            object_name: Name of the object
            attention_analysis: Results from attention analysis
            save_path: Path to save the visualization
            
        Returns:
            Dictionary containing visualization info
        """
        try:
            if 'error' in attention_analysis:
                return {"error": "Cannot visualize - attention analysis failed"}
            
            # Load original image
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Attention Analysis: {object_name} in {os.path.basename(image_path)}', 
                        fontsize=16, fontweight='bold')
            
            # Plot 1: Original image
            axes[0, 0].imshow(img_array)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Plot 2: Attention heatmap
            if attention_analysis["summary"].get("top_attended_patches"):
                heatmap = self._create_attention_heatmap(attention_analysis["summary"]["top_attended_patches"])
                im1 = axes[0, 1].imshow(heatmap, cmap='jet', alpha=0.8)
                axes[0, 1].set_title('Attention Heatmap (24x24 Grid)')
                axes[0, 1].set_xlabel('Patch Column')
                axes[0, 1].set_ylabel('Patch Row')
                plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            # Plot 3: Attention overlay on image
            if attention_analysis["summary"].get("top_attended_patches"):
                overlay = self._create_attention_overlay(img_array, attention_analysis["summary"]["top_attended_patches"])
                axes[0, 2].imshow(overlay)
                axes[0, 2].set_title('Attention Overlay')
                axes[0, 2].axis('off')
            
            # Plot 4: Attention evolution over generation steps
            avg_attention = attention_analysis["summary"]["avg_image_attention"]
            max_attention = attention_analysis["summary"]["max_image_attention"]
            
            if avg_attention and max_attention:
                steps = range(len(avg_attention))
                axes[1, 0].plot(steps, avg_attention, 'b-o', label='Average Attention', linewidth=2)
                axes[1, 0].plot(steps, max_attention, 'r-s', label='Max Attention', linewidth=2)
                axes[1, 0].set_title('Attention Evolution')
                axes[1, 0].set_xlabel('Generation Step')
                axes[1, 0].set_ylabel('Attention Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Top attended patches bar chart
            if attention_analysis["summary"].get("top_attended_patches"):
                top_patches = attention_analysis["summary"]["top_attended_patches"][:8]
                patch_labels = [f"({p['coordinates'][0]},{p['coordinates'][1]})" for p in top_patches]
                patch_values = [p['avg_attention'] for p in top_patches]
                
                bars = axes[1, 1].bar(range(len(patch_labels)), patch_values, 
                                     color='skyblue', alpha=0.7)
                axes[1, 1].set_title('Top Attended Patches')
                axes[1, 1].set_xlabel('Patch Coordinates (Row, Col)')
                axes[1, 1].set_ylabel('Average Attention')
                axes[1, 1].set_xticks(range(len(patch_labels)))
                axes[1, 1].set_xticklabels(patch_labels, rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, patch_values):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Plot 6: Analysis summary text
            axes[1, 2].axis('off')
            summary_text = f"""
Object: {object_name}
Generation Steps: {attention_analysis["generation_steps"]}

Attention Statistics:
"""
            if avg_attention:
                summary_text += f"""
Initial Attention: {avg_attention[0]:.4f}
Final Attention: {avg_attention[-1]:.4f}
Average Attention: {np.mean(avg_attention):.4f}
Max Attention: {np.max(max_attention):.4f}

Trend: {"Increasing" if avg_attention[-1] > avg_attention[0] else "Decreasing"}
Change: {abs(avg_attention[-1] - avg_attention[0]):.4f}
"""
            
            if attention_analysis["summary"].get("top_attended_patches"):
                top_patch = attention_analysis["summary"]["top_attended_patches"][0]
                summary_text += f"""
Most Attended Patch:
  Position: {top_patch['coordinates']}
  Attention: {top_patch['avg_attention']:.4f}
"""
            
            axes[1, 2].text(0.05, 0.95, summary_text, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                           transform=axes[1, 2].transAxes)
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Visualization saved: {save_path}")
                plt.close()
            else:
                plt.show()
            
            return {
                "visualization_created": True,
                "save_path": save_path,
                "object_name": object_name,
                "summary": attention_analysis["summary"]
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return {"error": str(e)}
    
    def _create_attention_heatmap(self, top_patches: List[Dict[str, Any]]) -> np.ndarray:
        """Create 24x24 attention heatmap from top patches"""
        heatmap = np.zeros((24, 24))
        
        for patch in top_patches:
            row, col = patch['coordinates']
            attention = patch['avg_attention']
            heatmap[row, col] = attention
        
        return heatmap
    
    def _create_attention_overlay(self, img_array: np.ndarray, top_patches: List[Dict[str, Any]]) -> np.ndarray:
        """Create attention overlay on original image"""
        try:
            # Create heatmap
            heatmap = self._create_attention_heatmap(top_patches)
            
            # Resize heatmap to image dimensions
            img_h, img_w = img_array.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
            
            # Normalize heatmap
            if heatmap_resized.max() > 0:
                heatmap_resized = heatmap_resized / heatmap_resized.max()
            
            # Apply colormap
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]  # Remove alpha
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            
            # Blend with original image
            alpha = 0.4
            overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
            
            return overlay
            
        except Exception as e:
            logger.error(f"Error creating overlay: {str(e)}")
            return img_array
    
    def visualize_single_object(self, 
                              image_path: str,
                              object_name: str,
                              output_dir: str,
                              query_template: str = "Where is the {object} in this image? Point out its location and describe it.") -> Dict[str, Any]:
        """
        Visualize attention for a single object
        
        Args:
            image_path: Path to the image
            object_name: Name of the object
            output_dir: Directory to save visualization
            query_template: Template for object queries
            
        Returns:
            Dictionary containing results
        """
        try:
            # Analyze attention
            analysis_result = self.analyze_object_attention(image_path, object_name, query_template)
            
            if 'error' in analysis_result:
                return analysis_result
            
            # Create visualization
            filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{object_name}_attention.png"
            save_path = os.path.join(output_dir, filename)
            
            vis_result = self.create_attention_visualization(
                image_path, object_name, analysis_result["attention_analysis"], save_path
            )
            
            return {
                "analysis": analysis_result,
                "visualization": vis_result
            }
            
        except Exception as e:
            logger.error(f"Error in single object visualization: {str(e)}")
            return {"error": str(e)}
    
    def batch_visualize_all(self, 
                          output_dir: str,
                          query_template: str = "Where is the {object} in this image? Point out its location and describe it.") -> Dict[str, Any]:
        """
        Batch visualize all annotated key objects
        
        Args:
            output_dir: Directory to save all visualizations
            query_template: Template for object queries
            
        Returns:
            Dictionary containing batch results
        """
        try:
            if not self.annotated_data:
                return {"error": "No annotated data loaded. Call load_annotated_results() first."}
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            results = []
            total_objects = 0
            successful_visualizations = 0
            
            print(f"\n🎨 Starting batch visualization...")
            print("="*60)
            
            # Process each image with key objects
            for result in self.annotated_data["results"]:
                if 'error' in result or not result.get('key_objects'):
                    continue
                
                image_path = result["image_path"]
                key_objects = result["key_objects"]
                image_name = os.path.basename(image_path)
                description = result["description"]
                attention_data = result.get("attention_data")
                
                print(f"\n🖼️  Processing: {image_name}")
                print(f"   Key objects: {key_objects}")
                
                if not attention_data:
                    print(f"   ❌ No attention data saved for this image")
                    continue
                
                # Load cached attention tensors if attention_data points to a cache_path
                cached_attentions = None
                cached_generated_ids = None
                if isinstance(attention_data, dict) and attention_data.get("cache_path"):
                    cache_path = attention_data["cache_path"]
                    try:
                        cache_blob = torch.load(cache_path, map_location="cpu")
                        cached_attentions = cache_blob.get("attentions")
                        cached_generated_ids = cache_blob.get("generated_ids")
                    except Exception as e:
                        print(f"   ❌ Failed to load cached attention: {str(e)}")
                        continue
                
                image_results = []
                
                for obj in key_objects:
                    total_objects += 1
                    print(f"     Analyzing '{obj}' from saved attention...", end=" ")
                    
                    try:
                        # Use saved attention data instead of re-running inference
                        obj_analysis = self.analyze_object_attention_from_saved_data(
                            {"attentions": cached_attentions}, obj, description, cached_generated_ids
                        )
                        
                        if 'error' not in obj_analysis:
                            # Create visualization
                            filename = f"{os.path.splitext(image_name)[0]}_{obj}_attention.png"
                            save_path = os.path.join(output_dir, filename)
                            
                            vis_result = self.create_attention_visualization(
                                image_path, obj, obj_analysis["attention_analysis"], save_path
                            )
                            
                            if 'error' not in vis_result:
                                successful_visualizations += 1
                                print("✅")
                                image_results.append({
                                    "analysis": obj_analysis,
                                    "visualization": vis_result
                                })
                            else:
                                print(f"❌ Visualization failed: {vis_result['error']}")
                        else:
                            print(f"❌ {obj_analysis['error']}")
                    
                    except Exception as e:
                        print(f"❌ {str(e)}")
                
                if image_results:
                    results.append({
                        "image_path": image_path,
                        "image_name": image_name,
                        "key_objects": key_objects,
                        "object_results": image_results
                    })
            
            print("="*60)
            print(f"✅ Batch visualization completed!")
            print(f"   Total objects processed: {total_objects}")
            print(f"   Successful visualizations: {successful_visualizations}")
            print(f"   Success rate: {successful_visualizations/total_objects*100:.1f}%")
            print(f"   Output directory: {output_dir}")
            
            return {
                "total_objects": total_objects,
                "successful_visualizations": successful_visualizations,
                "success_rate": successful_visualizations / total_objects * 100 if total_objects > 0 else 0,
                "output_directory": output_dir,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in batch visualization: {str(e)}")
            return {"error": str(e)}
