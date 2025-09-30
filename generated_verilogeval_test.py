#!/usr/bin/env python3
"""
Fixed Generated VerilogEval Test - Testing LLM-generated SystemVerilog code
This tests the already generated SystemVerilog files against VerilogEval testbenches
"""

import subprocess
import os
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple

class FixedGeneratedVerilogEvalTest:
    def __init__(self, generated_dir: Path, verilogeval_dir: Path):
        self.generated_dir = Path(generated_dir).resolve()  # Use absolute path immediately
        self.verilogeval_dir = Path(verilogeval_dir).resolve()  # Use absolute path immediately
        self.results = {
            "total_designs": 0,
            "compilation_passed": 0,
            "simulation_passed": 0,
            "perfect_matches": 0,
            "compilation_failed": 0,
            "simulation_failed": 0,
            "missing_testbench": 0,
            "missing_generated": 0,
            "details": []
        }
        
    def find_generated_files(self) -> List[Tuple[str, Path, Path, Path]]:
        """Find all generated .sv files and their corresponding test and ref files"""
        designs = []
        
        print(f"DEBUG: Searching in: {self.generated_dir}")
        print(f"DEBUG: VerilogEval dir: {self.verilogeval_dir}")
        
        if not self.generated_dir.exists():
            print(f"ERROR: Generated directory {self.generated_dir} does not exist")
            return designs
        
        if not self.verilogeval_dir.exists():
            print(f"ERROR: VerilogEval directory {self.verilogeval_dir} does not exist")
            return designs
        
        # Find all generated .sv files
        generated_files = list(self.generated_dir.glob("*.sv"))
        print(f"DEBUG: Found {len(generated_files)} .sv files in generated directory")
        
        if len(generated_files) == 0:
            # Debug: show what files are actually there
            all_files = list(self.generated_dir.glob("*"))
            print(f"DEBUG: Directory contains {len(all_files)} total files")
            if all_files:
                print("DEBUG: Sample files:")
                for f in all_files[:10]:
                    print(f"  {f.name}")
            return designs
        
        # Show first few files found
        print(f"DEBUG: First few .sv files found:")
        for f in generated_files[:5]:
            print(f"  {f.name}")
        
        found_pairs = 0
        missing_test = 0
        missing_ref = 0
        
        for generated_file in generated_files:
            # Extract design name from filename
            design_name = generated_file.stem
            
            # Find corresponding test.sv and ref.sv files
            test_file = self.verilogeval_dir / f"{design_name}_test.sv"
            ref_file = self.verilogeval_dir / f"{design_name}_ref.sv"
            
            if test_file.exists() and ref_file.exists():
                designs.append((design_name, generated_file, test_file, ref_file))
                found_pairs += 1
            else:
                missing = []
                if not test_file.exists():
                    missing.append("test")
                    missing_test += 1
                if not ref_file.exists():
                    missing.append("ref")
                    missing_ref += 1
                if len(designs) < 5:  # Only show first few missing files
                    print(f"DEBUG: Missing {'/'.join(missing)} file(s) for {design_name}")
        
        print(f"DEBUG: Found {found_pairs} complete design sets")
        print(f"DEBUG: Missing test files: {missing_test}")
        print(f"DEBUG: Missing ref files: {missing_ref}")
        
        return sorted(designs)
    
    def test_generated_file(self, generated_file: Path, test_file: Path, ref_file: Path, design_name: str) -> Dict:
        """Test generated SystemVerilog file against testbench with reference module"""
        
        temp_output = f"/tmp/generated_test_sim_{design_name}_{int(time.time())}.out"
        
        try:
            # Compile testbench with both generated file and reference file
            compile_result = subprocess.run(
                ["iverilog", "-g2012", "-o", temp_output, str(test_file), str(generated_file), str(ref_file)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if compile_result.returncode != 0:
                return {
                    "passed": False,
                    "stage": "compilation",
                    "errors": [
                        "Compilation failed",
                        compile_result.stderr[:400] if compile_result.stderr else "No error details"
                    ],
                    "mismatch_count": None,
                    "compilation_stdout": compile_result.stdout[:200] if compile_result.stdout else "",
                    "compilation_stderr": compile_result.stderr[:400] if compile_result.stderr else ""
                }
            
            # Run simulation
            sim_result = subprocess.run(
                ["vvp", temp_output],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Analyze simulation output
            output = sim_result.stdout
            stderr = sim_result.stderr
            
            # Extract mismatch information using the same pattern as perfect test
            mismatch_match = re.search(r'Mismatches: (\d+) in (\d+)', output)
            if mismatch_match:
                mismatches = int(mismatch_match.group(1))
                total_samples = int(mismatch_match.group(2))
                
                return {
                    "passed": mismatches == 0,
                    "stage": "simulation",
                    "errors": [] if mismatches == 0 else [f"Logic mismatches: {mismatches}/{total_samples}"],
                    "mismatch_count": mismatches,
                    "total_samples": total_samples,
                    "simulation_output": output[:300],
                    "simulation_stderr": stderr[:200] if stderr else ""
                }
            
            # Check for timeout or other failures
            if "TIMEOUT" in output:
                return {
                    "passed": False,
                    "stage": "simulation",
                    "errors": ["Simulation timeout"],
                    "mismatch_count": None,
                    "simulation_output": output[:300]
                }
            
            # Check for other failure indicators
            if any(indicator in output.lower() for indicator in ['error', 'assertion']) or any(indicator in stderr.lower() for indicator in ['error', 'assertion']):
                return {
                    "passed": False,
                    "stage": "simulation", 
                    "errors": [f"Simulation error detected"],
                    "mismatch_count": None,
                    "simulation_output": output[:300],
                    "simulation_stderr": stderr[:200] if stderr else ""
                }
            
            # If we can't parse the output, assume failure
            return {
                "passed": False,
                "stage": "simulation",
                "errors": [f"Could not parse simulation output properly"],
                "mismatch_count": None,
                "simulation_output": output[:300],
                "simulation_stderr": stderr[:200] if stderr else ""
            }
                
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "stage": "timeout",
                "errors": ["Process timed out"],
                "mismatch_count": None
            }
        except Exception as e:
            return {
                "passed": False,
                "stage": "exception",
                "errors": [f"Exception: {str(e)}"],
                "mismatch_count": None
            }
        finally:
            # Cleanup
            if os.path.exists(temp_output):
                try:
                    os.remove(temp_output)
                except:
                    pass
    
    def test_single_design(self, design_name: str, generated_file: Path, test_file: Path, ref_file: Path) -> Dict:
        """Test a single generated design"""
        print(f"Testing {design_name}: ", end="", flush=True)
        
        result = {
            "design": design_name,
            "generated_file": str(generated_file),
            "test_file": str(test_file),
            "ref_file": str(ref_file),
            "test_result": None,
            "overall_status": "unknown"
        }
        
        # Run test on generated file with reference
        print("compile", end="", flush=True)
        test_result = self.test_generated_file(generated_file, test_file, ref_file, design_name)
        result["test_result"] = test_result
        
        stage = test_result.get("stage", "unknown")
        
        if stage == "compilation":
            print("✗", end="", flush=True)
            self.results["compilation_failed"] += 1
            result["overall_status"] = "compilation_failed"
        elif test_result["passed"]:
            print(" sim✓", end="", flush=True)
            self.results["compilation_passed"] += 1
            self.results["simulation_passed"] += 1
            self.results["perfect_matches"] += 1
            result["overall_status"] = "perfect_match"
        else:
            print(" sim✗", end="", flush=True)
            self.results["compilation_passed"] += 1
            self.results["simulation_failed"] += 1
            result["overall_status"] = "simulation_failed"
        
        print()  # New line
        return result
    
    def run_generated_tests(self):
        """Run tests on all generated files"""
        print("Testing LLM-Generated SystemVerilog Code")
        print("=" * 55)
        print("Testing generated files against VerilogEval testbenches")
        print(f"Generated files: {self.generated_dir}")
        print(f"Testbenches: {self.verilogeval_dir}")
        
        # Check if iverilog is available
        try:
            version_result = subprocess.run(["iverilog", "-V"], capture_output=True, timeout=5)
            if version_result.returncode != 0:
                print("Error: iverilog not found or not working")
                return
            print("✓ iverilog is available")
        except:
            print("Error: iverilog not found")
            return
        
        # Find all generated files
        designs = self.find_generated_files()
        if not designs:
            print("No complete design sets found (need generated + test + ref files)")
            return
        
        print(f"Found {len(designs)} complete designs to test")
        self.results["total_designs"] = len(designs)
        
        print("\nRunning tests on generated code...")
        print("-" * 55)
        
        # Test each design
        for design_name, generated_file, test_file, ref_file in designs:
            result = self.test_single_design(design_name, generated_file, test_file, ref_file)
            self.results["details"].append(result)
            time.sleep(0.05)  # Small delay
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print detailed summary of generated code test results"""
        print("\n" + "=" * 80)
        print("GENERATED CODE TEST SUMMARY")
        print("=" * 80)
        
        total = self.results["total_designs"]
        comp_passed = self.results["compilation_passed"]
        sim_passed = self.results["simulation_passed"]
        perfect = self.results["perfect_matches"]
        comp_failed = self.results["compilation_failed"]
        sim_failed = self.results["simulation_failed"]
        
        print(f"Total generated designs: {total}")
        print(f"Compilation successful: {comp_passed}/{total} ({comp_passed/total*100:.1f}%)")
        if comp_passed > 0:
            print(f"Simulation successful: {sim_passed}/{comp_passed} ({sim_passed/comp_passed*100:.1f}%)")
            print(f"Perfect matches (0 mismatches): {perfect}/{comp_passed} ({perfect/comp_passed*100:.1f}%)")
        else:
            print(f"Simulation successful: {sim_passed}/0 (N/A)")
            print(f"Perfect matches (0 mismatches): {perfect}/0 (N/A)")
        print(f"Compilation failures: {comp_failed}")
        print(f"Simulation failures: {sim_failed}")
        
        # Show perfect matches
        if perfect > 0:
            print(f"\n✓ PERFECT MATCHES ({perfect}):")
            count = 0
            for detail in self.results["details"]:
                if detail["overall_status"] == "perfect_match" and count < 10:
                    test_result = detail.get("test_result", {})
                    mismatch_count = test_result.get("mismatch_count", "?")
                    total_samples = test_result.get("total_samples", "?")
                    print(f"  {detail['design']}: {mismatch_count}/{total_samples} mismatches")
                    count += 1
            if perfect > 10:
                print(f"  ... and {perfect - 10} more perfect matches")
        
        # Show compilation failures with details
        if comp_failed > 0:
            print(f"\n✗ COMPILATION FAILURES ({min(comp_failed, 5)} shown):")
            count = 0
            for detail in self.results["details"]:
                if detail["overall_status"] == "compilation_failed" and count < 5:
                    test_result = detail.get("test_result", {})
                    errors = test_result.get("errors", ["Unknown error"])
                    # Show first error and truncate long messages
                    error_msg = errors[1] if len(errors) > 1 else errors[0] if errors else "No details"
                    if len(error_msg) > 80:
                        error_msg = error_msg[:77] + "..."
                    print(f"  {detail['design']}: {error_msg}")
                    count += 1
        
        # Show simulation failures
        if sim_failed > 0:
            print(f"\n○ SIMULATION FAILURES ({min(sim_failed, 5)} shown):")
            count = 0
            for detail in self.results["details"]:
                if detail["overall_status"] == "simulation_failed" and count < 5:
                    test_result = detail.get("test_result", {})
                    mismatch_count = test_result.get("mismatch_count", "?")
                    total_samples = test_result.get("total_samples", "?")
                    errors = test_result.get("errors", ["Unknown error"])
                    error_summary = errors[0] if errors else "Unknown error"
                    print(f"  {detail['design']}: {error_summary}")
                    if mismatch_count is not None and mismatch_count != "?":
                        print(f"    Mismatches: {mismatch_count}/{total_samples}")
                    count += 1
        
        # Overall assessment
        comp_rate = comp_passed / total * 100 if total > 0 else 0
        sim_rate = sim_passed / comp_passed * 100 if comp_passed > 0 else 0
        perfect_rate = perfect / comp_passed * 100 if comp_passed > 0 else 0
        
        print(f"\nLLM GENERATION QUALITY ASSESSMENT:")
        print(f"Syntax correctness: {comp_rate:.1f}%")
        print(f"Functional correctness: {sim_rate:.1f}%")
        print(f"Perfect implementation rate: {perfect_rate:.1f}%")

def main():
    # Default directories
    generated_dir = Path("./verilog_eval/low_T/qwen2_5_14b_low_T/t1")
    verilogeval_dir = Path("./VerilogEval")
    
    # Allow custom directories via command line
    import sys
    if len(sys.argv) > 1:
        generated_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        verilogeval_dir = Path(sys.argv[2])
    
    print(f"Generated files directory: {generated_dir}")
    print(f"VerilogEval testbench directory: {verilogeval_dir}")
    
    tester = FixedGeneratedVerilogEvalTest(generated_dir, verilogeval_dir)
    tester.run_generated_tests()

if __name__ == "__main__":
    main()