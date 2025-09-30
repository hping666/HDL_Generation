#!/usr/bin/env python3
"""
HDL Generation Test Script
Tests Verilog files for synthesis and simulation
"""

import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

class HDLTester:
    def __init__(self):
        self.base_dir = Path(".")
        self.verilog_dir = self.base_dir / "verilog"
        self.testbench_dir = self.base_dir / "testbench"
        
        # Results storage
        self.results = {
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": 0,
            "synthesis_passed": [],
            "synthesis_failed": [],
            "simulation_passed": [],
            "simulation_failed": [],
            "logs": []
        }
        
    def find_verilog_files(self):
        """Find all verilog files and their testbenches"""
        verilog_files = []
        
        if not self.verilog_dir.exists():
            print(f"Error: {self.verilog_dir} does not exist")
            return []
            
        for v_file in self.verilog_dir.glob("*.v"):
            module_name = v_file.stem
            tb_file = self.testbench_dir / f"{module_name}_tb.v"
            verilog_files.append({
                "module": module_name,
                "verilog": v_file,
                "testbench": tb_file if tb_file.exists() else None
            })
        
        return verilog_files
    
    def test_synthesis(self, v_file):
        """Test if verilog file can be synthesized"""
        try:
            cmd = ["iverilog", "-o", "/tmp/test.out", str(v_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            log_entry = {
                "file": v_file.name,
                "type": "synthesis",
                "command": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            self.results["logs"].append(log_entry)
            
            return result.returncode == 0
            
        except Exception as e:
            log_entry = {
                "file": v_file.name,
                "type": "synthesis",
                "error": str(e)
            }
            self.results["logs"].append(log_entry)
            return False
    
    def test_simulation(self, v_file, tb_file):
        """Test verilog with testbench"""
        try:
            # Compile
            compile_cmd = ["iverilog", "-o", "/tmp/sim.out", str(tb_file), str(v_file)]
            compile_result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=10)
            
            if compile_result.returncode != 0:
                log_entry = {
                    "file": v_file.name,
                    "type": "simulation_compile",
                    "command": " ".join(compile_cmd),
                    "stdout": compile_result.stdout,
                    "stderr": compile_result.stderr,
                    "returncode": compile_result.returncode
                }
                self.results["logs"].append(log_entry)
                return False
            
            # Run simulation
            sim_cmd = ["vvp", "/tmp/sim.out"]
            sim_result = subprocess.run(sim_cmd, capture_output=True, text=True, timeout=10)
            
            log_entry = {
                "file": v_file.name,
                "type": "simulation",
                "compile_cmd": " ".join(compile_cmd),
                "sim_cmd": " ".join(sim_cmd),
                "stdout": sim_result.stdout,
                "stderr": sim_result.stderr,
                "returncode": sim_result.returncode
            }
            self.results["logs"].append(log_entry)
            
            # Check for pass/fail in output
            output = sim_result.stdout.lower()
            if "fail" in output or "error" in output:
                return False
            if "pass" in output or sim_result.returncode == 0:
                return True
            
            return False
            
        except Exception as e:
            log_entry = {
                "file": v_file.name,
                "type": "simulation",
                "error": str(e)
            }
            self.results["logs"].append(log_entry)
            return False
    
    def run_tests(self):
        """Run all tests"""
        print("\n" + "="*50)
        print("HDL Generation Test")
        print("="*50 + "\n")
        
        # Find files
        files = self.find_verilog_files()
        
        if not files:
            print("No verilog files found!")
            return
        
        self.results["total_files"] = len(files)
        
        print(f"Found {len(files)} verilog files\n")
        
        # Test each file
        for file_info in files:
            module = file_info["module"]
            v_file = file_info["verilog"]
            tb_file = file_info["testbench"]
            
            print(f"Testing {module}.v ... ", end="")
            
            # Synthesis test
            if self.test_synthesis(v_file):
                self.results["synthesis_passed"].append(module)
                print("synthesis OK", end="")
                
                # Simulation test only if synthesis passed
                if tb_file:
                    if self.test_simulation(v_file, tb_file):
                        self.results["simulation_passed"].append(module)
                        print(", simulation OK")
                    else:
                        self.results["simulation_failed"].append(module)
                        print(", simulation FAILED")
                else:
                    print(", no testbench")
            else:
                self.results["synthesis_failed"].append(module)
                print("synthesis FAILED")
        
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save results to files"""
        # Save summary
        summary = {
            "test_time": self.results["test_time"],
            "total_files": self.results["total_files"],
            "synthesis_passed_count": len(self.results["synthesis_passed"]),
            "synthesis_passed_files": self.results["synthesis_passed"],
            "synthesis_failed_count": len(self.results["synthesis_failed"]),
            "synthesis_failed_files": self.results["synthesis_failed"],
            "both_passed_count": len(self.results["simulation_passed"]),
            "both_passed_files": self.results["simulation_passed"],
            "simulation_failed_files": self.results["simulation_failed"]
        }
        
        with open("./result/test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save logs
        with open("./result/test_logs.json", "w") as f:
            json.dump(self.results["logs"], f, indent=2)
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*50)
        print("Test Summary")
        print("="*50)
        
        total = self.results["total_files"]
        syn_pass = len(self.results["synthesis_passed"])
        both_pass = len(self.results["simulation_passed"])
        
        print(f"\nTotal tested files: {total}")
        print(f"Synthesis passed: {syn_pass}")
        print(f"Both synthesis and simulation passed: {both_pass}")
        
        print("\nResults saved to:")
        print("  - ./result/test_summary.json (statistics and file lists)")
        print("  - ./result/test_logs.json (detailed logs)")


def main():
    # Check iverilog installation
    try:
        subprocess.run(["iverilog", "-V"], capture_output=True, check=True)
    except:
        print("Error: iverilog not installed!")
        print("Install with: sudo apt-get install iverilog")
        return
    
    tester = HDLTester()
    tester.run_tests()


if __name__ == "__main__":
    main()