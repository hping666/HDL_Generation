#!/usr/bin/env python3
"""
metric_calculate.py - Standalone metric calculator for HDL designs
Reads Verilog/SystemVerilog files and calculates pass@k metrics
"""

import subprocess
import os
import time
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class MetricCalculator:
    def __init__(self, verilog_path: str, dataset: str = "rtllm", 
                 n_samples: int = 10, k_values: List[int] = [1, 3]):
        """
        Initialize metric calculator
        
        Args:
            verilog_path: Path to verilog subfolder
            dataset: Dataset type ('rtllm' or 'verilogeval')
            n_samples: Number of samples per design
            k_values: List of k values for pass@k calculation
        """
        self.verilog_path = Path(verilog_path)
        self.dataset = dataset
        self.n_samples = n_samples
        self.k_values = k_values
        self.file_extension = ".sv" if dataset == "verilogeval" else ".v"
        
        # Dataset directories
        if dataset == "verilogeval":
            self.dataset_dir = Path("./VerilogEval")
        else:
            self.dataset_dir = Path("./RTLLM")
            
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory {self.dataset_dir} not found")
            
        if not self.verilog_path.exists():
            raise ValueError(f"Verilog path {self.verilog_path} not found")
    
    def find_trials(self) -> Dict[str, List[Path]]:
        """Find all trial files organized by design"""
        design_trials = {}
        
        for i in range(1, self.n_samples + 1):
            trial_dir = self.verilog_path / f"t{i}"
            if trial_dir.exists():
                for file in trial_dir.glob(f"*{self.file_extension}"):
                    design_name = file.stem
                    if design_name not in design_trials:
                        design_trials[design_name] = []
                    design_trials[design_name].append(file)
        
        return design_trials
    
    def find_testbench(self, design_name: str) -> Tuple[Optional[Path], Optional[Path]]:
        """Find testbench and reference file for design"""
        if self.dataset == "rtllm":
            # Search for testbench in nested structure
            for testbench in self.dataset_dir.rglob("*/testbench.v"):
                if testbench.parent.name == design_name:
                    return testbench, None
            
            # Direct path attempt
            testbench = self.dataset_dir / design_name / "testbench.v"
            if testbench.exists():
                return testbench, None
                
        elif self.dataset == "verilogeval":
            testbench = self.dataset_dir / f"{design_name}_test.sv"
            ref_file = self.dataset_dir / f"{design_name}_ref.sv"
            
            if testbench.exists() and ref_file.exists():
                return testbench, ref_file
        
        return None, None
    
    def test_file(self, design_file: Path, tb_file: Path, ref_file: Optional[Path] = None) -> bool:
        """Test single Verilog/SystemVerilog file"""
        try:
            temp_out = f"/tmp/test_{design_file.parent.name}_{design_file.stem}_{int(time.time())}.out"
            
            # Syntax check
            syntax_cmd = ["iverilog", "-g2012", "-o", temp_out, str(design_file)]
            syntax_result = subprocess.run(syntax_cmd, capture_output=True, text=True, timeout=30)
            
            if syntax_result.returncode != 0:
                if os.path.exists(temp_out):
                    os.remove(temp_out)
                return False
            
            # Compilation with testbench
            if self.dataset == "verilogeval" and ref_file:
                compile_cmd = ["iverilog", "-g2012", "-o", temp_out, 
                             str(tb_file), str(design_file), str(ref_file)]
            else:
                compile_cmd = ["iverilog", "-g2012", "-o", temp_out, 
                             str(tb_file), str(design_file)]
            
            compile_result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
            
            if compile_result.returncode != 0:
                if os.path.exists(temp_out):
                    os.remove(temp_out)
                return False
            
            # Simulation
            sim_cmd = ["vvp", temp_out]
            sim_result = subprocess.run(sim_cmd, capture_output=True, text=True, timeout=30)
            
            if os.path.exists(temp_out):
                os.remove(temp_out)
            
            # Parse result
            return self.parse_simulation_result(sim_result.stdout, sim_result.stderr)
            
        except Exception:
            return False
    
    def parse_simulation_result(self, stdout: str, stderr: str) -> bool:
        """Parse simulation output"""
        import re
        
        if self.dataset == "verilogeval":
            # VerilogEval specific check
            mismatch_match = re.search(r'Mismatches: (\d+) in (\d+)', stdout)
            if mismatch_match:
                mismatches = int(mismatch_match.group(1))
                return mismatches == 0
            
            if "mismatches: 0" in stdout.lower():
                return True
            elif "mismatches:" in stdout.lower():
                return False
        
        # General check
        output_lower = stdout.lower()
        stderr_lower = stderr.lower()
        
        fail_indicators = ["fail", "error", "mismatch", "assertion", "timeout"]
        has_fail = any(indicator in output_lower or indicator in stderr_lower 
                      for indicator in fail_indicators)
        
        if has_fail:
            return False
        
        pass_indicators = ["pass", "success", "test completed", "simulation finished"]
        has_pass = any(indicator in output_lower for indicator in pass_indicators)
        
        return has_pass or (not has_fail and len(stderr) == 0)
    
    def calculate_pass_at_k(self, n: int, c: int, k: int) -> float:
        """Calculate pass@k probability"""
        if n <= 0 or c < 0 or k <= 0:
            return 0.0
        
        if c == 0:
            return 0.0
        
        if k > n:
            k = n
        
        try:
            if (n - c) < k:
                return 1.0
            
            prob_all_wrong = math.comb(n - c, k) / math.comb(n, k)
            return 1.0 - prob_all_wrong
            
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def get_total_designs(self) -> int:
        """Get total number of designs in dataset"""
        if self.dataset == "verilogeval":
            # Count prompt files
            prompt_files = list(self.dataset_dir.glob("*_prompt.txt"))
            return len(prompt_files)
        else:  # rtllm
            # Count design directories
            designs = set()
            for desc_file in self.dataset_dir.rglob("*/design_description.txt"):
                designs.add(desc_file.parent.name)
            return len(designs)
    
    def calculate_metrics(self):
        """Calculate and print pass@k metrics"""
        print(f"Metric Calculator")
        print(f"Dataset: {self.dataset}")
        print(f"Verilog path: {self.verilog_path}")
        print(f"N_samples: {self.n_samples}")
        print(f"K_values: {self.k_values}")
        print("-" * 50)
        
        # Find trials
        design_trials = self.find_trials()
        if not design_trials:
            print("No trial files found!")
            return
        
        print(f"Found {len(design_trials)} designs with generated files")
        
        # Get total expected designs
        total_expected_designs = self.get_total_designs()
        print(f"Total designs in dataset: {total_expected_designs}")
        
        # Test each design
        design_results = {}
        for i, (design_name, trial_files) in enumerate(design_trials.items(), 1):
            print(f"[{i}/{len(design_trials)}] Testing {design_name}...", end="")
            
            testbench, ref_file = self.find_testbench(design_name)
            if not testbench:
                print(" [No testbench]")
                design_results[design_name] = {"n": len(trial_files), "c": 0}
                continue
            
            # Test all trials
            passed_count = 0
            for trial_file in trial_files:
                if self.test_file(trial_file, testbench, ref_file):
                    passed_count += 1
            
            design_results[design_name] = {"n": len(trial_files), "c": passed_count}
            print(f" {passed_count}/{len(trial_files)} passed")
        
        # Calculate pass@k metrics
        print("\n" + "=" * 50)
        print("RESULTS:")
        
        for k in self.k_values:
            if k > self.n_samples:
                print(f"pass@{k}: Skipped (k > n_samples)")
                continue
            
            total_pass_prob = 0.0
            
            for design_name, result in design_results.items():
                n = result["n"]
                c = result["c"]
                pass_prob = self.calculate_pass_at_k(n, c, k)
                total_pass_prob += pass_prob
            
            # Use total expected designs as denominator
            avg_pass_prob = total_pass_prob / max(1, total_expected_designs)
            pass_at_k = avg_pass_prob * 100
            
            print(f"pass@{k}: {pass_at_k:.2f}%")
        
        # Additional statistics
        total_trials = sum(r["n"] for r in design_results.values())
        total_passed = sum(r["c"] for r in design_results.values())
        designs_with_success = sum(1 for r in design_results.values() if r["c"] > 0)
        
        print("-" * 50)
        print(f"Designs tested: {len(design_results)}/{total_expected_designs}")
        print(f"Total trials: {total_trials}")
        print(f"Total passed: {total_passed}")
        print(f"Trial success rate: {total_passed/max(1,total_trials)*100:.2f}%")
        print(f"Designs with success: {designs_with_success}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Calculate pass@k metrics for HDL designs")
    
    parser.add_argument("--verilog_path", type=str, 
                        default="./verilog/MoA/MoA_high_T_L1_qwen2_5-7b_qwen2_5-coder-7b_llama3_1-8b_AGG_qwen2_5-7b",
                        help="Path to verilog subfolder (default: ./verilog/low_T/qwen2_5_7b_direct_low_T)")
    
    parser.add_argument("--dataset", type=str, default="rtllm", 
                       choices=["rtllm", "verilogeval"],
                       help="Dataset type (default: rtllm)")
    
    parser.add_argument("--n_samples", type=int, default=10,
                       help="Number of samples per design (default: 10)")
    
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 3],
                       help="K values for pass@k calculation (default: 1 3)")
    
    args = parser.parse_args()
    
    # Check iverilog availability
    try:
        result = subprocess.run(["iverilog", "-V"], capture_output=True, timeout=5)
        if result.returncode != 0:
            print("Error: iverilog not available")
            return
    except:
        print("Error: iverilog not available")
        return
    
    # Run calculation
    calculator = MetricCalculator(
        verilog_path=args.verilog_path,
        dataset=args.dataset,
        n_samples=args.n_samples,
        k_values=args.k_values
    )
    
    calculator.calculate_metrics()

if __name__ == "__main__":
    main()