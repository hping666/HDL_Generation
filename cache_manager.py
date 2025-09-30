#!/usr/bin/env python3
"""
Cache Manager for storing and retrieving HDL code with quality scores during MoA layers
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class HDLCacheManager:
    def __init__(self, cache_dir: Path, design_name: str, trial_num: int):
        """
        Initialize cache manager for a specific design and trial
        
        Args:
            cache_dir: Base cache directory (e.g., ./verilog_temp or ./verilogeval_temp)
            design_name: Name of the design being processed
            trial_num: Trial number (1-based)
        """
        self.cache_dir = cache_dir
        self.design_name = design_name
        self.trial_num = trial_num
        
        # Create cache file path: cache_dir/trial_name/design_name.json
        self.trial_cache_dir = cache_dir / f"t{trial_num}"
        self.trial_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.trial_cache_dir / f"{design_name}_cache.json"
        
        # Initialize fresh cache structure (always start clean)
        self.cache_data = {
            "design_name": design_name,
            "trial_num": trial_num,
            "created_at": datetime.now().isoformat(),
            "total_layers": 0,
            "layer_outputs": {},  # layer_idx -> [hdl_entries]
            "metadata": {
                "last_updated": None,
                "total_hdl_codes": 0
            }
        }
        
        # Always start with a fresh cache - remove old cache file if exists
        if self.cache_file.exists():
            self.cache_file.unlink()
        
        # Save the fresh cache immediately
        self._save_cache()
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            self.cache_data["metadata"]["last_updated"] = datetime.now().isoformat()
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save cache for {self.design_name} t{self.trial_num}: {e}")
    
    def add_layer_outputs(self, layer_idx: int, hdl_outputs: List[Dict]):
        """
        Add HDL outputs from a specific layer
        
        Args:
            layer_idx: Layer index (0-based)
            hdl_outputs: List of dicts containing:
                - code: HDL code string
                - model: Model name that generated this code
                - quality_score: Quality score (0.0-1.0)
                - generation_time: Time when generated
                - additional metadata
        """
        layer_key = str(layer_idx)
        
        if layer_key not in self.cache_data["layer_outputs"]:
            self.cache_data["layer_outputs"][layer_key] = []
        
        # Add timestamp and layer info to each output
        for hdl_output in hdl_outputs:
            hdl_entry = {
                "code": hdl_output["code"],
                "model": hdl_output["model"],
                "quality_score": hdl_output["quality_score"],
                "layer_idx": layer_idx,
                "cached_at": datetime.now().isoformat(),
                "generation_info": hdl_output.get("generation_info", {})
            }
            
            self.cache_data["layer_outputs"][layer_key].append(hdl_entry)
        
        # Update metadata
        self.cache_data["total_layers"] = max(self.cache_data["total_layers"], layer_idx + 1)
        self.cache_data["metadata"]["total_hdl_codes"] = sum(
            len(outputs) for outputs in self.cache_data["layer_outputs"].values()
        )
        
        self._save_cache()
    
    def get_top_quality_codes(self, n: int, up_to_layer: Optional[int] = None) -> List[Dict]:
        """
        Get top n HDL codes by quality score from cache
        
        Args:
            n: Number of codes to retrieve
            up_to_layer: Only consider codes from layers 0 to up_to_layer (inclusive).
                        If None, consider all layers.
        
        Returns:
            List of HDL entries sorted by quality score (highest first)
        """
        all_codes = []
        
        for layer_key, outputs in self.cache_data["layer_outputs"].items():
            layer_idx = int(layer_key)
            
            # Filter by layer if specified
            if up_to_layer is not None and layer_idx > up_to_layer:
                continue
            
            all_codes.extend(outputs)
        
        # Sort by quality score (descending) and take top n
        sorted_codes = sorted(all_codes, key=lambda x: x["quality_score"], reverse=True)
        return sorted_codes[:n]
    
    def get_layer_statistics(self) -> Dict:
        """Get statistics about cached data"""
        stats = {
            "total_layers": self.cache_data["total_layers"],
            "total_codes": self.cache_data["metadata"]["total_hdl_codes"],
            "layers_breakdown": {}
        }
        
        for layer_key, outputs in self.cache_data["layer_outputs"].items():
            layer_idx = int(layer_key)
            layer_stats = {
                "count": len(outputs),
                "avg_quality": sum(o["quality_score"] for o in outputs) / len(outputs) if outputs else 0,
                "max_quality": max(o["quality_score"] for o in outputs) if outputs else 0,
                "min_quality": min(o["quality_score"] for o in outputs) if outputs else 0,
                "models": list(set(o["model"] for o in outputs))
            }
            stats["layers_breakdown"][layer_idx] = layer_stats
        
        return stats
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache_data = {
            "design_name": self.design_name,
            "trial_num": self.trial_num,
            "created_at": datetime.now().isoformat(),
            "total_layers": 0,
            "layer_outputs": {},
            "metadata": {
                "last_updated": None,
                "total_hdl_codes": 0
            }
        }
        self._save_cache()
    
    def has_cached_data(self) -> bool:
        """Check if there's any cached data"""
        return self.cache_data["metadata"]["total_hdl_codes"] > 0
    
    def export_analysis_data(self) -> Dict:
        """Export cache data for analysis purposes"""
        return {
            "design_name": self.design_name,
            "trial_num": self.trial_num,
            "statistics": self.get_layer_statistics(),
            "full_cache": self.cache_data
        }

class GlobalCacheManager:
    """Manager for handling multiple design caches"""
    
    def __init__(self, base_cache_dir: Path):
        self.base_cache_dir = base_cache_dir
        base_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_design_cache(self, design_name: str, trial_num: int) -> HDLCacheManager:
        """Get cache manager for a specific design and trial"""
        return HDLCacheManager(self.base_cache_dir, design_name, trial_num)
    
    def cleanup_old_caches(self, max_age_hours: int = 24):
        """Clean up cache files older than specified hours"""
        import time
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        for cache_file in self.base_cache_dir.rglob("*_cache.json"):
            try:
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    print(f"Cleaned up old cache: {cache_file}")
            except Exception as e:
                print(f"Warning: Failed to clean up {cache_file}: {e}")
    
    def clear_all_caches(self):
        """Clear all cache files in the base directory"""
        try:
            for cache_file in self.base_cache_dir.rglob("*_cache.json"):
                cache_file.unlink()
            print(f"Cleared all caches in {self.base_cache_dir}")
        except Exception as e:
            print(f"Warning: Failed to clear caches: {e}")
    
    def generate_global_analysis(self) -> Dict:
        """Generate analysis across all cached trials"""
        analysis = {
            "total_designs": 0,
            "total_trials": 0,
            "cache_summary": {},
            "quality_distribution": {"high": 0, "medium": 0, "low": 0},
            "model_performance": {}
        }
        
        for cache_file in self.base_cache_dir.rglob("*_cache.json"):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                design_name = cache_data["design_name"]
                trial_num = cache_data["trial_num"]
                
                if design_name not in analysis["cache_summary"]:
                    analysis["cache_summary"][design_name] = []
                    analysis["total_designs"] += 1
                
                analysis["cache_summary"][design_name].append(trial_num)
                analysis["total_trials"] += 1
                
                # Analyze quality distribution and model performance
                for layer_outputs in cache_data["layer_outputs"].values():
                    for output in layer_outputs:
                        quality = output["quality_score"]
                        model = output["model"]
                        
                        # Quality distribution
                        if quality >= 0.8:
                            analysis["quality_distribution"]["high"] += 1
                        elif quality >= 0.5:
                            analysis["quality_distribution"]["medium"] += 1
                        else:
                            analysis["quality_distribution"]["low"] += 1
                        
                        # Model performance
                        if model not in analysis["model_performance"]:
                            analysis["model_performance"][model] = {
                                "count": 0, "total_quality": 0, "avg_quality": 0
                            }
                        
                        analysis["model_performance"][model]["count"] += 1
                        analysis["model_performance"][model]["total_quality"] += quality
                
            except Exception as e:
                print(f"Warning: Failed to analyze {cache_file}: {e}")
        
        # Calculate average quality for each model
        for model_stats in analysis["model_performance"].values():
            if model_stats["count"] > 0:
                model_stats["avg_quality"] = model_stats["total_quality"] / model_stats["count"]
        
        return analysis