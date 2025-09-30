#!/usr/bin/env python3
"""
Main script for Multi-Dataset Benchmark Testing with Qwen2.5
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from config import Config
from llm_interface import OllamaInterface
from rtllm_generator import MultiDatasetGenerator
from hdl_tester_enhanced import MultiDatasetHDLTester
from utils import load_designs

def test_model(model_name: str, designs: List, method: str = "direct", 
               dataset: str = "rtllm", temp_mode: str = "low_T"):
    """Test single Qwen2.5 model with specified method, dataset and temperature"""
    print(f"\nTesting {model_name} on {dataset.upper()} dataset")
    print(f"Method: {method}, Temperature: {temp_mode}")
    
    if Config.ENABLE_ITERATIVE_REFINEMENT:
        print(f"Iterative refinement: Enabled (max {Config.MAX_REFINEMENT_ITERATIONS} iterations)")
    else:
        print("Iterative refinement: Disabled")
    
    # Setup directories based on dataset and temperature
    verilog_dir, result_dir = Config.get_output_dirs(model_name, method, temp_mode, dataset)
    
    verilog_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM with temperature-specific parameters
    llm = OllamaInterface(model_name)
    # Update LLM parameters for temperature mode
    llm.params = Config.get_model_params(model_name, temp_mode)
    
    if not llm.test_connection():
        print(f"Model {model_name} not available")
        return None
    
    # Display model configuration
    params = llm.params
    print(f"Model config: temp={params['temperature']}, top_p={params['top_p']}, "
          f"ctx={params['context_length']}, tokens={params['num_predict']}")
    
    print("-" * 60)
    
    # Generate RTL using specified method and dataset
    print(f"Phase 1: {dataset.upper()} RTL Generation (method: {method}, temp: {temp_mode})")
    generator = MultiDatasetGenerator(llm, designs, verilog_dir, method, dataset, temp_mode)
    generator.generate_all()
    
    # Test RTL with dataset-specific validation
    print(f"\nPhase 2: Testing {dataset.upper()} (method: {method}, temp: {temp_mode})")
    dataset_dir = Config.VERILOGEVAL_DIR if dataset == "verilogeval" else Config.RTLLM_DIR
    tester = MultiDatasetHDLTester(verilog_dir, dataset_dir, result_dir, 
                                   f"{model_name}_{method}_{temp_mode}", dataset, temp_mode)
    tester.run_tests()
    
    return result_dir / "results.json"

def main(method: str = "direct", dataset: str = "rtllm", temp_mode: str = "low_T"):
    """Main function with dataset and temperature selection"""
    print("Multi-Dataset Benchmark for Qwen2.5 Series")
    print(f"Dataset: {dataset.upper()}, Method: {method}, Temperature: {temp_mode}")
    print("=" * 70)
    
    # Validate inputs
    if dataset not in Config.DATASET_TYPES:
        print(f"Error: Unsupported dataset '{dataset}'. Supported: {Config.DATASET_TYPES}")
        return
    
    if temp_mode not in Config.TEMPERATURE_MODES:
        print(f"Error: Unsupported temperature mode '{temp_mode}'. Supported: {Config.TEMPERATURE_MODES}")
        return
    
    # Validate dataset directory exists
    dataset_dir = Config.VERILOGEVAL_DIR if dataset == "verilogeval" else Config.RTLLM_DIR
    if not dataset_dir.exists():
        print(f"Error: {dataset_dir} not found")
        return
    
    # Load designs for specified dataset
    designs = load_designs(dataset)
    if not designs:
        print(f"No {dataset} designs found")
        return
    
    print(f"Qwen2.5 Models: {Config.QWEN_MODELS}")
    print(f"Dataset: {dataset} ({len(designs)} designs)")
    print(f"Method: {method}")
    print(f"Temperature: {temp_mode}")
    print(f"Trials per design: {Config.N_SAMPLES}")
    print(f"Evaluation: pass@{Config.K_VALUES}")
    
    # Create base directories for both temperature modes
    for temp in Config.TEMPERATURE_MODES:
        if dataset == "verilogeval":
            (Config.VERILOG_EVAL_BASE_DIR / temp).mkdir(parents=True, exist_ok=True)
            (Config.RESULT_EVAL_BASE_DIR / temp).mkdir(parents=True, exist_ok=True)
        else:
            (Config.VERILOG_BASE_DIR / temp).mkdir(parents=True, exist_ok=True)
            (Config.RESULT_BASE_DIR / temp).mkdir(parents=True, exist_ok=True)
    
    # Test all Qwen2.5 models with specified configuration
    all_results = {
        "benchmark_info": {
            "series": "qwen2.5",
            "models_tested": Config.QWEN_MODELS,
            "dataset": dataset,
            "method_used": method,
            "temperature_mode": temp_mode,
            "total_designs": len(designs),
            "trials_per_design": Config.N_SAMPLES,
            "timestamp": datetime.now().isoformat()
        },
        "results": {}
    }
    
    successful_tests = 0
    
    for i, model in enumerate(Config.QWEN_MODELS, 1):
        print(f"\n{'='*25} [{i}/{len(Config.QWEN_MODELS)}] {model} - {dataset} - {temp_mode} {'='*25}")
        
        result_file = test_model(model, designs, method, dataset, temp_mode)
        
        if result_file and result_file.exists():
            try:
                with open(result_file) as f:
                    model_results = json.load(f)
                    key = f"{model}_{method}_{temp_mode}"
                    all_results["results"][key] = model_results
                    successful_tests += 1
            except Exception as e:
                print(f"Warning: Could not load results for {model}_{method}_{temp_mode}: {e}")
    
    # Save comprehensive comparison
    output_dir = Config.RESULT_EVAL_BASE_DIR if dataset == "verilogeval" else Config.RESULT_BASE_DIR
    comparison_file = output_dir / f"qwen25_{dataset}_{method}_{temp_mode}_comparison.json"
    comparison_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final comparison
    print(f"\n{'='*80}")
    print(f"Final Qwen2.5 {dataset.upper()} Comparison ({method} + {temp_mode})")
    print(f"{'='*80}")
    
    if successful_tests > 0:
        print(f"{'Model + Config':<30} {'Pass@1':<8} {'Pass@3':<8} {'Syntax%':<8} {'Success Designs'}")
        print("-" * 80)
        
        # Sort by performance
        performance_data = []
        for key, results in all_results["results"].items():
            pass_at_k = results.get("pass_at_k", {})
            agg_stats = results.get("aggregate_stats", {})
            performance_data.append({
                "key": key,
                "pass@1": pass_at_k.get("pass@1", 0),
                "pass@3": pass_at_k.get("pass@3", 0), 
                "syntax_rate": agg_stats.get("syntax_success_rate", 0),
                "success_designs": agg_stats.get("designs_with_success", 0)
            })
        
        performance_data.sort(key=lambda x: x["pass@1"], reverse=True)
        
        for perf in performance_data:
            print(f"{perf['key']:<30} {perf['pass@1']:<7.1f}% {perf['pass@3']:<7.1f}% "
                  f"{perf['syntax_rate']:<7.1f}% {perf['success_designs']}")
        
        # Best model summary
        best = performance_data[0]
        print(f"\nBest performing: {best['key']}")
        print(f"  Pass@1: {best['pass@1']:.1f}%")
        print(f"  Pass@3: {best['pass@3']:.1f}%")
        print(f"  Designs with success: {best['success_designs']}/{len(designs)}")
        
        print(f"\nDetailed results: {comparison_file}")
    else:
        print("No successful tests completed")

def run_all_combinations():
    """Run tests for all combinations of datasets and temperature modes"""
    print("Running comprehensive benchmark across all datasets and temperature modes")
    print("=" * 80)
    
    total_combinations = len(Config.DATASET_TYPES) * len(Config.TEMPERATURE_MODES)
    current = 0
    
    for dataset in Config.DATASET_TYPES:
        for temp_mode in Config.TEMPERATURE_MODES:
            current += 1
            print(f"\n[{current}/{total_combinations}] Starting {dataset.upper()} - {temp_mode}")
            print("=" * 60)
            
            try:
                main(dataset=dataset, temp_mode=temp_mode)
            except Exception as e:
                print(f"Error in {dataset} - {temp_mode}: {e}")
                continue
    
    print(f"\nCompleted all {total_combinations} combinations")
    print("Check individual result directories for detailed analysis")

if __name__ == "__main__":
    main()