#!/usr/bin/env python3
"""
Utility functions for Qwen2.5 RTLLM (Verilog) and VerilogEval (SystemVerilog) benchmark
"""

import subprocess
from pathlib import Path
from typing import List, Dict
from config import Config

def check_dependencies() -> bool:
    """Check if required tools are available"""
    try:
        # Check iverilog
        result = subprocess.run(["iverilog", "-V"], capture_output=True, timeout=5)
        if result.returncode != 0:
            return False
            
        # Check vvp
        result = subprocess.run(["vvp", "-V"], capture_output=True, timeout=5)
        if result.returncode != 0:
            return False
            
        # Check ollama
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=10)
        if result.returncode != 0:
            return False
            
        return True
    except:
        return False

def get_available_models() -> List[str]:
    """Get available Ollama models"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return [line.split()[0] for line in result.stdout.strip().split('\n')[1:] if line.strip()]
    except:
        pass
    return []

def load_rtllm_designs() -> List[Dict]:
    """Load RTLLM designs with recursive search"""
    designs = []
    design_paths = {}  # Map design name to full path
    
    if not Config.RTLLM_DIR.exists():
        print(f"  Error: {Config.RTLLM_DIR} does not exist")
        return []
    
    def scan_directory(directory: Path, depth: int = 0):
        """Recursively scan directory for designs"""
        for item in directory.iterdir():
            if item.is_dir():
                # Check if this directory contains the required files
                desc_file = item / "design_description.txt"
                tb_file = item / "testbench.v"
                
                if desc_file.exists() and tb_file.exists():
                    # Found a valid design
                    design_name = item.name
                    designs.append({
                        "name": design_name,
                        "description": desc_file,
                        "testbench": tb_file,
                        "full_path": item,
                        "dataset": "rtllm"
                    })
                    design_paths[design_name] = item
                else:
                    # Continue scanning subdirectories
                    scan_directory(item, depth + 1)
    
    scan_directory(Config.RTLLM_DIR)
    
    # Store the mapping globally for HDLTester to use
    Config.DESIGN_PATHS = design_paths
    
    return designs

def load_verilogeval_designs() -> List[Dict]:
    """Load VerilogEval designs from flat directory structure"""
    designs = []
    
    if not Config.VERILOGEVAL_DIR.exists():
        print(f"  Error: {Config.VERILOGEVAL_DIR} does not exist")
        return []
    
    # Find all prompt files (*.prompt.txt)
    prompt_files = list(Config.VERILOGEVAL_DIR.glob("*_prompt.txt"))
    
    for prompt_file in prompt_files:
        # Extract design name by removing _prompt.txt suffix
        design_name = prompt_file.stem.replace("_prompt", "")
        
        # Find corresponding test file
        test_file = Config.VERILOGEVAL_DIR / f"{design_name}_test.sv"
        
        if test_file.exists():
            designs.append({
                "name": design_name,
                "description": prompt_file,  # prompt.txt instead of design_description.txt
                "testbench": test_file,      # test.sv instead of testbench.v
                "full_path": Config.VERILOGEVAL_DIR,
                "dataset": "verilogeval"
            })
        else:
            print(f"  Warning: Missing testbench for {design_name}")
    
    return sorted(designs, key=lambda x: x["name"])

def load_designs(dataset: str = "rtllm") -> List[Dict]:
    """Load designs based on dataset type"""
    if dataset == "verilogeval":
        return load_verilogeval_designs()
    elif dataset == "rtllm":
        return load_rtllm_designs()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def load_all_designs() -> Dict[str, List[Dict]]:
    """Load designs from both datasets"""
    return {
        "rtllm": load_designs("rtllm"),
        "verilogeval": load_designs("verilogeval")
    }

def print_setup_check():
    """Print setup status for Qwen2.5 with both datasets"""
    print("Qwen2.5 Multi-Dataset Benchmark Setup Check:")
    
    # Dependencies
    deps_ok = check_dependencies()
    print(f"  Dependencies: {'✓' if deps_ok else '✗'}")
    if not deps_ok:
        print("    Required: iverilog, vvp, ollama")
    
    # Qwen2.5 models specifically
    available = get_available_models()
    qwen25_available = [m for m in available if "qwen2.5" in m]
    
    print(f"  Qwen2.5 models available: {len(qwen25_available)}")
    for model in Config.QWEN_MODELS:
        status = "✓" if model in available else "✗"
        print(f"    {status} {model}")
        if status == "✗":
            print(f"      Download: ollama pull {model}")
    
    # RTLLM designs
    rtllm_designs = load_designs("rtllm")
    print(f"  RTLLM designs: {len(rtllm_designs)} found")
    
    if len(rtllm_designs) > 0:
        print(f"    Sample designs: {[d['name'] for d in rtllm_designs[:3]]}")
        if len(rtllm_designs) > 3:
            print(f"    ... and {len(rtllm_designs) - 3} more")
    
    # VerilogEval designs
    verilogeval_designs = load_designs("verilogeval")
    print(f"  VerilogEval designs: {len(verilogeval_designs)} found")
    
    if len(verilogeval_designs) > 0:
        print(f"    Sample designs: {[d['name'] for d in verilogeval_designs[:3]]}")
        if len(verilogeval_designs) > 3:
            print(f"    ... and {len(verilogeval_designs) - 3} more")
    
    # Temperature settings
    print(f"\nTemperature Settings:")
    print(f"  Low-T: temp={Config.LOW_T_PARAMS['temperature']}, top_p={Config.LOW_T_PARAMS['top_p']}")
    print(f"  High-T: temp={Config.HIGH_T_PARAMS['temperature']}, top_p={Config.HIGH_T_PARAMS['top_p']}")
    
    print(f"\nRefinement Settings:")
    print(f"  Enabled: {Config.ENABLE_ITERATIVE_REFINEMENT}")
    if Config.ENABLE_ITERATIVE_REFINEMENT:
        print(f"  Max iterations: {Config.MAX_REFINEMENT_ITERATIONS}")
    
    # Overall readiness
    datasets_ready = len(rtllm_designs) > 0 or len(verilogeval_designs) > 0
    overall_ready = len(qwen25_available) > 0 and deps_ok and datasets_ready
    print(f"\nMulti-Dataset Benchmark Ready: {'✓' if overall_ready else '✗'}")
    
    return overall_ready