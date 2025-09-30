#!/usr/bin/env python3
"""
VerilogEval Perfect Test - Using RefModule as TopModule
This creates a perfect test scenario where RefModule is renamed to TopModule
and tested against itself to verify iverilog compatibility
"""

import subprocess
import os
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple

class VerilogEvalPerfectTest:
    def __init__(self, verilogeval_dir: Path):
        self.verilogeval_dir = Path(verilogeval_dir)
        self.results = {
            "total_designs": 0,
            "compilation_passed": 0,
            "simulation_passed": 0,
            "perfect_matches": 0,
            "compilation_failed": 0,
            "simulation_failed": 0,
            "missing_testbench": 0,
            "details": []
        }
        
    def find_design_files(self) -> List[Tuple[str, Path, Path]]:
        """Find all ref.sv files and their corresponding test.sv files"""
        designs = []
        
        if not self.verilogeval_dir.exists():
            print(f"Error: Directory {self.verilogeval_dir} does not exist")
            return designs
        
        # Find all ref.sv files
        ref_files = list(self.verilogeval_dir.glob("*_ref.sv"))
        
        for ref_file in ref_files:
            # Extract design name (remove _ref.sv suffix)
            design_name = ref_file.stem.replace("_ref", "")
            
            # Find corresponding test.sv file
            test_file = self.verilogeval_dir / f"{design_name}_test.sv"
            
            if test_file.exists():
                designs.append((design_name, ref_file, test_file))
            else:
                print(f"Warning: No testbench found for {design_name}")
                designs.append((design_name, ref_file, None))
        
        return sorted(designs)
    
    def create_topmodule_from_ref(self, ref_file: Path, design_name: str) -> Path:
        """Create TopModule by renaming RefModule while keeping original RefModule intact"""
        
        # Read reference module
        with open(ref_file, 'r') as f:
            ref_content = f.read()
        
        # Find the RefModule definition and extract it
        ref_module_match = re.search(r'module\s+(\w+)\s*\((.*?)\);(.*?)endmodule', ref_content, re.DOTALL)
        if not ref_module_match:
            return None
        
        original_module_name = ref_module_match.group(1)
        port_declaration = ref_module_match.group(2)
        module_body = ref_module_match.group(3)
        
        # Create a new file with both RefModule and TopModule
        combined_content = f"""// Perfect test file for {design_name}
// Contains both original RefModule and TopModule (renamed copy) for testing

{ref_content}

// TopModule - exact copy of {original_module_name} for testing
module TopModule (
{port_declaration}
);
{module_body}
endmodule
"""
        
        # Write combined file to temporary location
        combined_file = Path(f"/tmp/perfect_test_{design_name}.sv")
        with open(combined_file, 'w') as f:
            f.write(combined_content)
        
        return combined_file
    
    def test_perfect_match(self, ref_file: Path, test_file: Path, design_name: str) -> Dict:
        """Test perfect match scenario: RefModule vs TopModule (same implementation)"""
        
        # Create combined file with both modules
        combined_file = self.create_topmodule_from_ref(ref_file, design_name)
        if not combined_file:
            return {
                "passed": False,
                "stage": "file_creation",
                "errors": ["Failed to create TopModule from RefModule"],
                "mismatch_count": None
            }
        
        temp_output = f"/tmp/perfect_test_sim_{design_name}_{int(time.time())}.out"
        
        try:
            # Compile testbench with combined file
            compile_result = subprocess.run(
                ["iverilog", "-g2012", "-o", temp_output, str(test_file), str(combined_file)],
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
                    "mismatch_count": None
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
            
            # Extract mismatch information
            mismatch_match = re.search(r'Mismatches: (\d+) in (\d+)', output)
            if mismatch_match:
                mismatches = int(mismatch_match.group(1))
                total_samples = int(mismatch_match.group(2))
                
                return {
                    "passed": mismatches == 0,
                    "stage": "simulation",
                    "errors": [] if mismatches == 0 else [f"Unexpected mismatches: {mismatches}/{total_samples}"],
                    "mismatch_count": mismatches,
                    "total_samples": total_samples,
                    "simulation_output": output[:300]
                }
            
            # Check for timeout or other failures
            if "TIMEOUT" in output:
                return {
                    "passed": False,
                    "stage": "simulation",
                    "errors": ["Simulation timeout"],
                    "mismatch_count": None
                }
            
            # Check for other failure indicators
            if any(indicator in output.lower() for indicator in ['error', 'fail']) and 'fail' not in output.lower():
                return {
                    "passed": False,
                    "stage": "simulation", 
                    "errors": [f"Simulation error: {output[:200]}"],
                    "mismatch_count": None
                }
            
            # If we can't parse the output, assume failure
            return {
                "passed": False,
                "stage": "simulation",
                "errors": [f"Could not parse simulation output: {output[:200]}"],
                "mismatch_count": None
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
            for cleanup_file in [temp_output, combined_file]:
                if cleanup_file and os.path.exists(cleanup_file):
                    try:
                        os.remove(cleanup_file)
                    except:
                        pass
    
    def test_single_design(self, design_name: str, ref_file: Path, test_file: Path = None) -> Dict:
        """Test a single design using perfect match approach"""
        print(f"Testing {design_name}: ", end="", flush=True)
        
        result = {
            "design": design_name,
            "ref_file": str(ref_file),
            "test_file": str(test_file) if test_file else None,
            "test_result": None,
            "overall_status": "unknown"
        }
        
        if not test_file:
            print("(no testbench)", end="", flush=True)
            self.results["missing_testbench"] += 1
            result["overall_status"] = "no_testbench"
            print()
            return result
        
        # Run perfect match test
        print("compile", end="", flush=True)
        test_result = self.test_perfect_match(ref_file, test_file, design_name)
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
    
    def run_perfect_tests(self):
        """Run perfect match tests on all designs"""
        print("VerilogEval Perfect Match Test - RefModule as TopModule")
        print("=" * 65)
        print("This test renames RefModule to TopModule and tests it against itself.")
        print("Perfect matches (0 mismatches) indicate iverilog compatibility.")
        
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
        
        # Find all design files
        designs = self.find_design_files()
        if not designs:
            print("No SystemVerilog design files found in VerilogEval directory")
            return
        
        print(f"Found {len(designs)} designs in VerilogEval")
        self.results["total_designs"] = len(designs)
        
        print("\nRunning perfect match tests...")
        print("-" * 65)
        
        # Test each design
        for design_name, ref_file, test_file in designs:
            result = self.test_single_design(design_name, ref_file, test_file)
            self.results["details"].append(result)
            time.sleep(0.05)  # Small delay
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print detailed summary of perfect match test results"""
        print("\n" + "=" * 80)
        print("PERFECT MATCH TEST SUMMARY")
        print("=" * 80)
        
        total = self.results["total_designs"]
        comp_passed = self.results["compilation_passed"]
        sim_passed = self.results["simulation_passed"]
        perfect = self.results["perfect_matches"]
        comp_failed = self.results["compilation_failed"]
        sim_failed = self.results["simulation_failed"]
        missing_tb = self.results["missing_testbench"]
        
        print(f"Total designs: {total}")
        print(f"Compilation successful: {comp_passed}/{total} ({comp_passed/total*100:.1f}%)")
        print(f"Simulation successful: {sim_passed}/{comp_passed} ({sim_passed/max(1,comp_passed)*100:.1f}%)")
        print(f"Perfect matches (0 mismatches): {perfect}/{sim_passed} ({perfect/max(1,sim_passed)*100:.1f}%)")
        print(f"Compilation failures: {comp_failed}")
        print(f"Simulation failures: {sim_failed}")
        print(f"Missing testbenches: {missing_tb}")
        
        # Show perfect matches
        if perfect > 0:
            print(f"\n✓ PERFECT MATCHES ({perfect}):")
            count = 0
            for detail in self.results["details"]:
                if detail["overall_status"] == "perfect_match" and count < 10:
                    mismatch_count = detail["test_result"].get("mismatch_count", "?")
                    total_samples = detail["test_result"].get("total_samples", "?")
                    print(f"  {detail['design']}: {mismatch_count}/{total_samples} mismatches")
                    count += 1
            if perfect > 10:
                print(f"  ... and {perfect - 10} more perfect matches")
        
        # Show compilation failures
        if comp_failed > 0:
            print(f"\n✗ COMPILATION FAILURES ({min(comp_failed, 5)} shown):")
            count = 0
            for detail in self.results["details"]:
                if detail["overall_status"] == "compilation_failed" and count < 5:
                    errors = detail["test_result"].get("errors", ["Unknown error"])
                    print(f"  {detail['design']}: {errors[0]}")
                    count += 1
        
        # Show simulation failures
        if sim_failed > 0:
            print(f"\n○ SIMULATION FAILURES ({min(sim_failed, 5)} shown):")
            count = 0
            for detail in self.results["details"]:
                if detail["overall_status"] == "simulation_failed" and count < 5:
                    mismatch_count = detail["test_result"].get("mismatch_count", "?")
                    total_samples = detail["test_result"].get("total_samples", "?")
                    print(f"  {detail['design']}: {mismatch_count}/{total_samples} mismatches")
                    count += 1
        
        # Overall assessment
        comp_rate = comp_passed / total * 100 if total > 0 else 0
        sim_rate = sim_passed / comp_passed * 100 if comp_passed > 0 else 0
        perfect_rate = perfect / sim_passed * 100 if sim_passed > 0 else 0
        
        print(f"\nOVERALL IVERILOG COMPATIBILITY ASSESSMENT:")
        print(f"Compilation success rate: {comp_rate:.1f}%")
        print(f"Simulation success rate: {sim_rate:.1f}%")
        print(f"Perfect match rate: {perfect_rate:.1f}%")
        
        if comp_rate >= 95 and perfect_rate >= 95:
            print("\n✓ EXCELLENT: iverilog is fully compatible with VerilogEval")
            recommendation = "iverilog is recommended for VerilogEval processing"
        elif comp_rate >= 85 and perfect_rate >= 85:
            print("\n○ GOOD: iverilog has good compatibility with VerilogEval")
            recommendation = "iverilog is suitable but may need minor workarounds"
        elif comp_rate >= 70:
            print("\n⚠ MODERATE: iverilog has moderate compatibility issues")
            recommendation = "iverilog may work but expect significant compatibility issues"
        else:
            print("\n✗ POOR: iverilog has serious compatibility issues")
            recommendation = "Consider alternative simulators for VerilogEval"
        
        print(f"\nRecommendation: {recommendation}")
        
        # Note about the test methodology
        print(f"\nNOTE: This test uses identical implementations (RefModule renamed to TopModule)")
        print(f"Any mismatches indicate simulator issues, not logic errors.")
        print(f"Real LLM-generated code will likely have additional functional errors.")

def main():
    # Default VerilogEval directory
    verilogeval_dir = Path("./VerilogEval")
    
    # Allow custom directory via command line
    import sys
    if len(sys.argv) > 1:
        verilogeval_dir = Path(sys.argv[1])
    
    tester = VerilogEvalPerfectTest(verilogeval_dir)
    tester.run_perfect_tests()

if __name__ == "__main__":
    main()