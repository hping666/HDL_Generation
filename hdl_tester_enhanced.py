#!/usr/bin/env python3
"""
Multi-Dataset HDL Tester optimized for Qwen2.5 generated code with refinement and prescreening analysis
Supports both RTLLM (Verilog .v files) and VerilogEval (SystemVerilog .sv files)
"""

import subprocess
import json
import time
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from config import Config

class MultiDatasetHDLTester:
    def __init__(self, verilog_dir: Path, dataset_dir: Path, result_dir: Path, 
                 model_name: str = "unknown", dataset: str = "rtllm", temp_mode: str = "low_T"):
        self.verilog_dir = verilog_dir
        self.dataset_dir = dataset_dir  # RTLLM_DIR or VERILOGEVAL_DIR
        self.result_dir = result_dir
        self.model_name = model_name
        self.dataset = dataset
        self.temp_mode = temp_mode
        self.file_extension = Config.get_file_extension(dataset)
        self.results = {"design_results": {}, "logs": []}
        self.generation_info = None
        result_dir.mkdir(parents=True, exist_ok=True)
    
    def load_generation_info(self) -> Optional[Dict]:
        """Load generation summary to correlate with test results"""
        gen_summary_path = self.verilog_dir / "generation_summary.json"
        if gen_summary_path.exists():
            try:
                with open(gen_summary_path, 'r', encoding='utf-8') as f:
                    self.generation_info = json.load(f)
                    return self.generation_info
            except Exception as e:
                print(f"Warning: Could not load generation summary: {e}")
        return None
    
    def get_total_designs_from_dataset(self) -> int:
        """Get total number of designs directly from dataset directory"""
        try:
            if self.dataset == "verilogeval":
                # Count prompt files
                prompt_files = list(self.dataset_dir.glob("*_prompt.txt"))
                return len(prompt_files)
            else:  # rtllm
                # Count design directories with design_description.txt
                designs = set()
                for desc_file in self.dataset_dir.rglob("*/design_description.txt"):
                    designs.add(desc_file.parent.name)
                return len(designs)
        except Exception as e:
            print(f"Warning: Could not count designs from dataset: {e}")
            return 0
    
    def analyze_prescreening_effectiveness(self):
        """Analyze how effective prescreening was"""
        if not self.generation_info or not self.generation_info.get('prescreening_enabled'):
            return None
        
        analysis = {
            "prescreening_enabled": True,
            "prescreened_trials": {
                "total": 0,
                "passed_syntax": 0,
                "passed_simulation": 0,
                "final_test_passed": 0
            },
            "fallback_trials": {
                "total": 0,
                "passed_syntax": 0,
                "passed_simulation": 0
            },
            "by_design": {},
            "method_comparison": {
                "direct_prescreened": {"total": 0, "test_passed": 0},
                "fallback_method": {"total": 0, "test_passed": 0}
            },
            "efficiency_metrics": {
                "prescreening_success_rate": 0.0,
                "prescreened_final_success_rate": 0.0,
                "fallback_final_success_rate": 0.0
            }
        }
        
        # Analyze each design
        for design_gen in self.generation_info.get("details", []):
            design_name = design_gen["design"]
            
            if design_name not in self.results["design_results"]:
                continue
            
            test_result = self.results["design_results"][design_name]
            prescreening_stats = design_gen.get("prescreening_stats")
            
            if not prescreening_stats:
                continue
            
            design_analysis = {
                "prescreened": {"total": 0, "passed": 0},
                "fallback": {"total": 0, "passed": 0}
            }
            
            # Analyze trials
            for trial_info in design_gen.get("trials", []):
                if not trial_info.get("success"):
                    continue
                    
                trial_num = trial_info["trial"]
                trial_id = f"t{trial_num}"
                
                # Get generation info
                gen_info = trial_info.get("generation_info", {})
                
                if not gen_info.get("prescreening_attempted"):
                    continue
                
                # Get test result
                if trial_id not in test_result.get("trial_details", {}):
                    continue
                
                test_detail = test_result["trial_details"][trial_id]
                test_passed = test_detail["simulation"]
                
                if gen_info.get("prescreening_passed"):
                    # This trial passed prescreening
                    analysis["prescreened_trials"]["total"] += 1
                    design_analysis["prescreened"]["total"] += 1
                    
                    if test_detail["syntax"]:
                        analysis["prescreened_trials"]["passed_syntax"] += 1
                    
                    if test_passed:
                        analysis["prescreened_trials"]["final_test_passed"] += 1
                        design_analysis["prescreened"]["passed"] += 1
                        analysis["method_comparison"]["direct_prescreened"]["test_passed"] += 1
                    
                    analysis["method_comparison"]["direct_prescreened"]["total"] += 1
                    
                else:
                    # This trial used fallback method
                    analysis["fallback_trials"]["total"] += 1
                    design_analysis["fallback"]["total"] += 1
                    
                    if test_detail["syntax"]:
                        analysis["fallback_trials"]["passed_syntax"] += 1
                    
                    if test_passed:
                        analysis["fallback_trials"]["passed_simulation"] += 1
                        design_analysis["fallback"]["passed"] += 1
                        analysis["method_comparison"]["fallback_method"]["test_passed"] += 1
                    
                    analysis["method_comparison"]["fallback_method"]["total"] += 1
            
            if design_analysis["prescreened"]["total"] > 0 or design_analysis["fallback"]["total"] > 0:
                analysis["by_design"][design_name] = design_analysis
        
        # Calculate efficiency metrics
        if analysis["prescreened_trials"]["total"] > 0:
            # Success rate of prescreened trials in final testing
            analysis["efficiency_metrics"]["prescreened_final_success_rate"] = (
                analysis["prescreened_trials"]["final_test_passed"] / 
                analysis["prescreened_trials"]["total"] * 100
            )
        
        if analysis["fallback_trials"]["total"] > 0:
            # Success rate of fallback trials
            analysis["efficiency_metrics"]["fallback_final_success_rate"] = (
                analysis["fallback_trials"]["passed_simulation"] / 
                analysis["fallback_trials"]["total"] * 100
            )
        
        # Overall prescreening success rate from generation summary
        if self.generation_info.get("prescreening_summary"):
            ps_summary = self.generation_info["prescreening_summary"]
            if ps_summary.get("total_attempts", 0) > 0:
                analysis["efficiency_metrics"]["prescreening_success_rate"] = float(
                    ps_summary.get("success_rate", "0").replace("%", "")
                )
        
        return analysis
    
    def analyze_refinement_effectiveness(self):
        """Analyze how effective refinement was by comparing refined vs non-refined results"""
        if not self.generation_info:
            return None
        
        analysis = {
            "refined_trials": {
                "total": 0,
                "passed_syntax": 0,
                "passed_simulation": 0,
                "needed_refinement": 0,
                "refinement_fixed": 0
            },
            "non_refined_trials": {
                "total": 0,
                "passed_syntax": 0,
                "passed_simulation": 0
            },
            "by_iteration": {},
            "by_design": {}
        }
        
        for design_gen in self.generation_info.get("details", []):
            design_name = design_gen["design"]
            
            if design_name not in self.results["design_results"]:
                continue
            
            test_result = self.results["design_results"][design_name]
            refinement_stats = design_gen.get("refinement_stats")
            
            design_analysis = {
                "refined": {"total": 0, "passed": 0},
                "non_refined": {"total": 0, "passed": 0}
            }
            
            for trial_info in design_gen.get("trials", []):
                trial_id = f"t{trial_info['trial']}"
                
                if trial_id not in test_result.get("trial_details", {}):
                    continue
                
                test_passed = test_result["trial_details"][trial_id]["simulation"]
                
                if refinement_stats and trial_id in refinement_stats.get("trial_details", {}):
                    refine_info = refinement_stats["trial_details"][trial_id]
                    analysis["refined_trials"]["total"] += 1
                    design_analysis["refined"]["total"] += 1
                    
                    if test_result["trial_details"][trial_id]["syntax"]:
                        analysis["refined_trials"]["passed_syntax"] += 1
                    
                    if test_passed:
                        analysis["refined_trials"]["passed_simulation"] += 1
                        design_analysis["refined"]["passed"] += 1
                        
                        if refine_info.get("history"):
                            first_test = refine_info["history"][0]
                            if not first_test.get("passed", False):
                                analysis["refined_trials"]["refinement_fixed"] += 1
                    
                    iterations = refine_info.get("iterations", 1)
                    iter_key = str(iterations)
                    
                    if iter_key not in analysis["by_iteration"]:
                        analysis["by_iteration"][iter_key] = {"total": 0, "passed": 0}
                    
                    analysis["by_iteration"][iter_key]["total"] += 1
                    if test_passed:
                        analysis["by_iteration"][iter_key]["passed"] += 1
                    
                    if refine_info.get("history") and not refine_info["history"][0].get("passed", False):
                        analysis["refined_trials"]["needed_refinement"] += 1
                else:
                    analysis["non_refined_trials"]["total"] += 1
                    design_analysis["non_refined"]["total"] += 1
                    
                    if test_result["trial_details"][trial_id]["syntax"]:
                        analysis["non_refined_trials"]["passed_syntax"] += 1
                    
                    if test_passed:
                        analysis["non_refined_trials"]["passed_simulation"] += 1
                        design_analysis["non_refined"]["passed"] += 1
            
            if design_analysis["refined"]["total"] > 0 or design_analysis["non_refined"]["total"] > 0:
                analysis["by_design"][design_name] = design_analysis
        
        if analysis["refined_trials"]["total"] > 0:
            analysis["refined_trials"]["success_rate"] = (
                analysis["refined_trials"]["passed_simulation"] / 
                analysis["refined_trials"]["total"] * 100
            )
            analysis["refined_trials"]["fix_rate"] = (
                analysis["refined_trials"]["refinement_fixed"] / 
                max(1, analysis["refined_trials"]["needed_refinement"]) * 100
            )
        
        if analysis["non_refined_trials"]["total"] > 0:
            analysis["non_refined_trials"]["success_rate"] = (
                analysis["non_refined_trials"]["passed_simulation"] / 
                analysis["non_refined_trials"]["total"] * 100
            )
        
        for iter_key in analysis["by_iteration"]:
            if analysis["by_iteration"][iter_key]["total"] > 0:
                analysis["by_iteration"][iter_key]["success_rate"] = (
                    analysis["by_iteration"][iter_key]["passed"] / 
                    analysis["by_iteration"][iter_key]["total"] * 100
                )
        
        if analysis["by_iteration"]:
            sorted_iterations = dict(sorted(analysis["by_iteration"].items(), 
                                          key=lambda x: int(x[0])))
            analysis["by_iteration"] = sorted_iterations
        
        return analysis
    
    def analyze_cpp_validation_effectiveness(self):
        """Analyze effectiveness of C++ validation when used"""
        if not self.generation_info:
            return None
        
        # Check if C++ validation was enabled
        cpp_val_enabled = self.generation_info.get('cpp_validation_enabled', False)
        
        if not cpp_val_enabled:
            return None
        
        analysis = {
            "cpp_validation_enabled": True,
            "mode": self.generation_info.get('cpp_validation_mode', 'unknown'),
            "trials_with_cpp_validation": 0,
            "cpp_fix_success": 0,
            "cpp_fixes_applied": 0,
            "by_design": {}
        }
        
        # Analyze each design
        for design_gen in self.generation_info.get("details", []):
            design_name = design_gen["design"]
            
            if design_name not in self.results["design_results"]:
                continue
            
            test_result = self.results["design_results"][design_name]
            cpp_val_stats = design_gen.get("cpp_validation_stats")
            
            if not cpp_val_stats or cpp_val_stats.get('total', 0) == 0:
                continue
            
            design_cpp_analysis = {
                "total": cpp_val_stats.get('total', 0),
                "successful": cpp_val_stats.get('successful', 0),
                "fixes_applied": cpp_val_stats.get('fixes_applied', 0),
                "test_passed": 0
            }
            
            # Check test results for trials with C++ validation
            for trial_id, trial_details in cpp_val_stats.get('trials', {}).items():
                analysis["trials_with_cpp_validation"] += 1
                
                if trial_details.get('iterations', 0) > 1:
                    analysis["cpp_fixes_applied"] += 1
                
                # Check if this trial passed final test
                if trial_id in test_result.get("trial_details", {}):
                    if test_result["trial_details"][trial_id]["simulation"]:
                        if trial_details.get('success'):
                            analysis["cpp_fix_success"] += 1
                            design_cpp_analysis["test_passed"] += 1
            
            analysis["by_design"][design_name] = design_cpp_analysis
        
        # Calculate success rates
        if analysis["trials_with_cpp_validation"] > 0:
            analysis["validation_success_rate"] = (
                analysis["cpp_fix_success"] / analysis["trials_with_cpp_validation"] * 100
            )
        
        if analysis["cpp_fixes_applied"] > 0:
            analysis["fix_effectiveness"] = (
                analysis["cpp_fix_success"] / analysis["cpp_fixes_applied"] * 100
            )
        
        return analysis
    
    def find_trials(self) -> Dict[str, List[Path]]:
        """Find all trial files organized by design"""
        design_trials = {}
        
        for i in range(1, Config.N_SAMPLES + 1):
            trial_dir = self.verilog_dir / f"t{i}"
            if trial_dir.exists():
                for file in trial_dir.glob(f"*{self.file_extension}"):
                    design_name = file.stem
                    if design_name not in design_trials:
                        design_trials[design_name] = []
                    design_trials[design_name].append(file)
        
        return design_trials
    
    def find_testbench(self, design_name: str) -> tuple:
        """Find testbench and reference file for design based on dataset type"""
        if self.dataset == "rtllm":
            # RTLLM: nested directory structure
            if hasattr(Config, 'DESIGN_PATHS') and design_name in Config.DESIGN_PATHS:
                design_dir = Config.DESIGN_PATHS[design_name]
                testbench = design_dir / "testbench.v"
                if testbench.exists():
                    return testbench, None  # No ref file for RTLLM
            
            # Fallback: search in RTLLM directory
            for testbench in self.dataset_dir.rglob("*/testbench.v"):
                parent_dir = testbench.parent
                if parent_dir.name == design_name:
                    return testbench, None
            
            direct_path = self.dataset_dir / design_name / "testbench.v"
            if direct_path.exists():
                return direct_path, None
        
        elif self.dataset == "verilogeval":
            # VerilogEval: flat structure with _test.sv suffix AND _ref.sv
            testbench = self.dataset_dir / f"{design_name}_test.sv"
            ref_file = self.dataset_dir / f"{design_name}_ref.sv"
            
            if testbench.exists() and ref_file.exists():
                return testbench, ref_file
            else:
                return None, None
        
        return None, None
    
    def test_file(self, design_file: Path, tb_file: Path, ref_file: Path = None) -> Dict[str, bool]:
        """Test single Verilog/SystemVerilog file"""
        try:
            temp_out = f"/tmp/test_{design_file.parent.name}_{design_file.stem}_{int(time.time())}.out"
            
            # Syntax check (only for the generated file)
            syntax_cmd = ["iverilog", "-g2012", "-o", temp_out, str(design_file)]
            syntax_result = subprocess.run(syntax_cmd, capture_output=True, text=True, 
                                        timeout=Config.COMPILATION_TIMEOUT)
            
            if syntax_result.returncode != 0:
                self.results["logs"].append({
                    "file": str(design_file),
                    "type": "syntax_error",
                    "stderr": syntax_result.stderr,
                    "timestamp": datetime.now().isoformat()
                })
            
            try:
                if os.path.exists(temp_out):
                    os.remove(temp_out)
            except:
                pass
            
            syntax_ok = syntax_result.returncode == 0
            if not syntax_ok:
                return {"syntax": False, "simulation": False}
            
            # Compilation with testbench (and ref file for VerilogEval)
            if self.dataset == "verilogeval" and ref_file:
                # VerilogEval: compile test + generated + ref
                compile_cmd = ["iverilog", "-g2012", "-o", temp_out, str(tb_file), str(design_file), str(ref_file)]
            else:
                # RTLLM: compile test + generated
                compile_cmd = ["iverilog", "-g2012", "-o", temp_out, str(tb_file), str(design_file)]
                
            compile_result = subprocess.run(compile_cmd, capture_output=True, text=True, 
                                        timeout=Config.COMPILATION_TIMEOUT)
            
            if compile_result.returncode != 0:
                self.results["logs"].append({
                    "file": str(design_file),
                    "type": "compilation_error",
                    "stderr": compile_result.stderr,
                    "timestamp": datetime.now().isoformat()
                })
                return {"syntax": True, "simulation": False}
            
            # Simulation
            sim_cmd = ["vvp", temp_out]
            sim_result = subprocess.run(sim_cmd, capture_output=True, text=True, 
                                    timeout=Config.SIMULATION_TIMEOUT)
            
            try:
                if os.path.exists(temp_out):
                    os.remove(temp_out)
            except:
                pass
            
            # Parse simulation results based on dataset
            sim_ok = self.parse_simulation_result(sim_result.stdout, sim_result.stderr)
            
            if not sim_ok:
                self.results["logs"].append({
                    "file": str(design_file),
                    "type": "simulation_fail",
                    "stdout": sim_result.stdout,
                    "stderr": sim_result.stderr,
                    "timestamp": datetime.now().isoformat()
                })
            
            return {"syntax": True, "simulation": sim_ok}
            
        except Exception as e:
            self.results["logs"].append({
                "file": str(design_file),
                "type": "test_exception",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return {"syntax": False, "simulation": False}
    
    def parse_simulation_result(self, stdout: str, stderr: str) -> bool:
        """Parse simulation result with improved VerilogEval detection"""
        import re
        
        if self.dataset == "verilogeval":
            # VerilogEval: Look for exact "Mismatches: X in Y" pattern
            mismatch_match = re.search(r'Mismatches: (\d+) in (\d+)', stdout)
            if mismatch_match:
                mismatches = int(mismatch_match.group(1))
                return mismatches == 0
            
            # Fallback: simple "mismatches: 0" check
            if "mismatches: 0" in stdout.lower():
                return True
            elif "mismatches:" in stdout.lower():
                return False
        
        # RTLLM or general case: Check for failure indicators
        output_lower = stdout.lower()
        stderr_lower = stderr.lower()
        
        fail_indicators = ["fail", "error", "mismatch", "assertion", "timeout"]
        has_fail = any(indicator in output_lower or indicator in stderr_lower 
                    for indicator in fail_indicators)
        
        if has_fail:
            return False
        
        # Check for success indicators
        pass_indicators = ["pass", "success", "test completed", "simulation finished"]
        has_pass = any(indicator in output_lower for indicator in pass_indicators)
        
        return has_pass or (not has_fail and len(stderr) == 0)
    
    def test_design(self, design_name: str, trial_files: List[Path]) -> Dict:
        """Test all trials for one design"""
        testbench_result = self.find_testbench(design_name)
        
        if isinstance(testbench_result, tuple):
            testbench, ref_file = testbench_result
        else:
            testbench, ref_file = testbench_result, None
        
        if not testbench:
            return {
                "error": "No testbench",
                "n_samples": len(trial_files),
                "syntax_passed": 0,
                "simulation_passed": 0
            }
        
        # For VerilogEval, we need both testbench and ref file
        if self.dataset == "verilogeval" and not ref_file:
            return {
                "error": "No reference file",
                "n_samples": len(trial_files),
                "syntax_passed": 0,
                "simulation_passed": 0
            }
        
        syntax_count = 0
        sim_count = 0
        trial_details = {}
        
        print(f"    Testing {len(trial_files)} trials: ", end="")
        
        for trial_file in trial_files:
            trial_name = trial_file.parent.name
            result = self.test_file(trial_file, testbench, ref_file)
            
            trial_details[trial_name] = result
            
            if result["syntax"]:
                syntax_count += 1
                if result["simulation"]:
                    sim_count += 1
                    print("✓", end="")
                else:
                    print("○", end="")
            else:
                print("✗", end="")
        
        print(f" -> {sim_count}/{len(trial_files)} passed")
        
        return {
            "design": design_name,
            "dataset": self.dataset,
            "n_samples": len(trial_files),
            "syntax_passed": syntax_count,
            "simulation_passed": sim_count,
            "trial_details": trial_details
        }
    
    def run_tests(self):
        """Run all tests with refinement, prescreening and C++ validation analysis"""
        print(f"Testing Qwen2.5 generated {self.dataset} samples (n={Config.N_SAMPLES} per design)")
        print(f"Temperature mode: {self.temp_mode}")
        print(f"File extension: {self.file_extension}")
        
        self.load_generation_info()
        if self.generation_info:
            print("✓ Loaded generation info for analysis")
            if self.generation_info.get('prescreening_enabled'):
                print("✓ Prescreening was enabled during generation")
            if self.generation_info.get('cpp_validation_enabled'):
                print(f"✓ C++ validation was enabled (mode: {self.generation_info.get('cpp_validation_mode', 'unknown')})")
            if self.generation_info.get('refinement_enabled'):
                print("✓ Refinement was enabled during generation")
        
        try:
            result = subprocess.run(["iverilog", "-V"], capture_output=True, timeout=5)
            if result.returncode != 0:
                print("Error: iverilog not available")
                return
            print("✓ Using iverilog with SystemVerilog 2012 support")
        except:
            print("Error: iverilog not available")
            return
        
        design_trials = self.find_trials()
        if not design_trials:
            print("No trial files found!")
            return
        
        print(f"Found {len(design_trials)} designs to test")
        
        for i, (design_name, trial_files) in enumerate(design_trials.items(), 1):
            print(f"[{i}/{len(design_trials)}] {design_name}")
            result = self.test_design(design_name, trial_files)
            self.results["design_results"][design_name] = result
        
        # ===== FIXED: Corrected pass@k calculation =====
        # Get total expected designs from multiple sources
        total_expected_designs = None
        
        # Source 1: generation_info (most reliable if available)
        if self.generation_info:
            # Use designs_attempted (excludes skipped designs)
            total_expected_designs = self.generation_info.get("designs_attempted")
            
            # Fallback to total_designs if designs_attempted not available
            if total_expected_designs is None:
                total_expected_designs = self.generation_info.get("total_designs")
        
        # Source 2: Directly count from dataset (fallback)
        if total_expected_designs is None:
            total_expected_designs = self.get_total_designs_from_dataset()
            if total_expected_designs > 0:
                print(f"Warning: Using dataset directory count for total designs ({total_expected_designs})")
        
        # Source 3: Last resort - use tested designs count (least accurate)
        if total_expected_designs is None or total_expected_designs == 0:
            total_expected_designs = len(design_trials)
            print(f"Warning: No reliable design count available, using tested count ({total_expected_designs})")
        
        # Calculate pass@k metrics with correct denominator
        pass_at_k = {}
        valid_results = [r for r in self.results["design_results"].values() if "error" not in r]
        
        print(f"\nPass@k calculation:")
        print(f"  Total expected designs: {total_expected_designs}")
        print(f"  Designs with test results: {len(valid_results)}")
        print(f"  Designs with no generated files: {total_expected_designs - len(valid_results)}")
        
        for k in Config.K_VALUES:
            if k > Config.N_SAMPLES:
                # Skip if k is larger than the number of samples we generated
                print(f"  Skipping pass@{k} (k={k} > N_SAMPLES={Config.N_SAMPLES})")
                continue
                
            total_pass_prob = 0.0
            
            # Calculate pass probability for designs with test results
            for result in valid_results:
                n = result["n_samples"]
                c = result["simulation_passed"]
                
                # Calculate pass@k for this design
                pass_prob = Config.calculate_pass_at_k(n, c, k)
                total_pass_prob += pass_prob
            
            # Note: Designs with no generated files have pass_prob = 0
            # They don't contribute to total_pass_prob, but they ARE counted in the denominator
            
            # Use correct denominator (total expected designs)
            avg_pass_prob = total_pass_prob / max(1, total_expected_designs)
            pass_at_k[f"pass@{k}"] = avg_pass_prob * 100
            
            print(f"  pass@{k}: {avg_pass_prob * 100:.2f}% (sum={total_pass_prob:.2f}, denom={total_expected_designs})")
        
        self.results["pass_at_k"] = pass_at_k
        self.results["total_expected_designs"] = total_expected_designs
        self.results["total_tested_designs"] = len(design_trials)
        self.results["total_valid_results"] = len(valid_results)
        
        # Aggregate statistics
        total_samples = sum(r["n_samples"] for r in valid_results)
        total_syntax = sum(r["syntax_passed"] for r in valid_results)
        total_sim = sum(r["simulation_passed"] for r in valid_results)
        
        aggregate_stats = {
            "total_designs": len(valid_results),
            "total_expected_designs": total_expected_designs,
            "total_tested_designs": len(design_trials),
            "designs_with_no_files": total_expected_designs - len(valid_results),
            "total_samples": total_samples,
            "syntax_success_rate": total_syntax / max(1, total_samples) * 100,
            "simulation_success_rate": total_sim / max(1, total_syntax) * 100 if total_syntax > 0 else 0,
            "designs_with_success": sum(1 for r in valid_results if r["simulation_passed"] > 0)
        }
        
        self.results["aggregate_stats"] = aggregate_stats
        
        # Prescreening analysis
        prescreening_analysis = None
        if self.generation_info and self.generation_info.get('prescreening_enabled'):
            prescreening_analysis = self.analyze_prescreening_effectiveness()
            if prescreening_analysis:
                self.results["prescreening_analysis"] = prescreening_analysis
        
        # Refinement analysis
        refinement_analysis = None
        if self.generation_info and self.generation_info.get('refinement_enabled'):
            refinement_analysis = self.analyze_refinement_effectiveness()
            if refinement_analysis:
                self.results["refinement_analysis"] = refinement_analysis
        
        # C++ validation analysis
        cpp_validation_analysis = None
        if self.generation_info:
            cpp_validation_analysis = self.analyze_cpp_validation_effectiveness()
            if cpp_validation_analysis:
                self.results["cpp_validation_analysis"] = cpp_validation_analysis
        
        # Save detailed results with error handling
        detailed_results_file = self.result_dir / "detailed_results.json"
        try:
            with open(detailed_results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2)
        except PermissionError as e:
            print(f"\n✗ Permission denied saving detailed results: {e}")
        except OSError as e:
            print(f"\n✗ OS error saving detailed results: {e}")
        except Exception as e:
            print(f"\n✗ Error saving detailed results: {e}")
            # Try to save simplified version
            try:
                simplified_results = {
                    "pass_at_k": self.results.get("pass_at_k", {}),
                    "aggregate_stats": self.results.get("aggregate_stats", {}),
                    "error": f"Full results failed to save: {e}"
                }
                with open(detailed_results_file, 'w', encoding='utf-8') as f:
                    json.dump(simplified_results, f, indent=2)
                print(f"  Saved simplified results instead")
            except:
                print(f"  Failed to save any results")
        
        summary = {
            "model": self.model_name,
            "dataset": self.dataset,
            "temp_mode": self.temp_mode,
            "pass_at_k": pass_at_k,
            "aggregate_stats": aggregate_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        if prescreening_analysis:
            summary["prescreening_summary"] = {
                "prescreened_success_rate": prescreening_analysis["efficiency_metrics"]["prescreened_final_success_rate"],
                "fallback_success_rate": prescreening_analysis["efficiency_metrics"]["fallback_final_success_rate"],
                "prescreening_hit_rate": prescreening_analysis["efficiency_metrics"]["prescreening_success_rate"]
            }
        
        if refinement_analysis:
            summary["refinement_summary"] = {
                "refined_success_rate": refinement_analysis["refined_trials"].get("success_rate", 0),
                "non_refined_success_rate": refinement_analysis["non_refined_trials"].get("success_rate", 0),
                "fix_rate": refinement_analysis["refined_trials"].get("fix_rate", 0)
            }
        
        if cpp_validation_analysis:
            summary["cpp_validation_summary"] = {
                "mode": cpp_validation_analysis.get("mode", "unknown"),
                "trials_with_validation": cpp_validation_analysis.get("trials_with_cpp_validation", 0),
                "fixes_applied": cpp_validation_analysis.get("cpp_fixes_applied", 0),
                "success_after_fix": cpp_validation_analysis.get("cpp_fix_success", 0),
                "validation_success_rate": cpp_validation_analysis.get("validation_success_rate", 0),
                "fix_effectiveness": cpp_validation_analysis.get("fix_effectiveness", 0)
            }
        
        # Save summary with error handling
        summary_file = self.result_dir / "results.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"\n✗ Error saving results summary: {e}")
        
        # Print results
        print(f"\nQwen2.5 {self.dataset.upper()} Benchmark Results ({self.temp_mode}):")
        print(f"  Model: {self.model_name}")
        for k in Config.K_VALUES:
            if f"pass@{k}" in pass_at_k:
                print(f"  pass@{k}: {pass_at_k[f'pass@{k}']:.1f}%")
        if total_expected_designs:
            print(f"  Total expected designs: {total_expected_designs}")
        print(f"  Designs tested: {len(design_trials)}")
        print(f"  Designs with no files: {aggregate_stats['designs_with_no_files']}")
        print(f"  Syntax rate: {aggregate_stats['syntax_success_rate']:.1f}%")
        print(f"  Simulation rate: {aggregate_stats['simulation_success_rate']:.1f}%")
        print(f"  Designs with any success: {aggregate_stats['designs_with_success']}")
        
        if prescreening_analysis:
            print(f"\nPrescreening Analysis:")
            print(f"  Prescreened trials success: {prescreening_analysis['efficiency_metrics']['prescreened_final_success_rate']:.1f}%")
            print(f"  Fallback trials success: {prescreening_analysis['efficiency_metrics']['fallback_final_success_rate']:.1f}%")
        
        if cpp_validation_analysis:
            print(f"\nC++ Validation Analysis:")
            print(f"  Mode: {cpp_validation_analysis.get('mode', 'unknown')}")
            print(f"  Trials with C++ validation: {cpp_validation_analysis['trials_with_cpp_validation']}")
            print(f"  C++ fixes applied: {cpp_validation_analysis['cpp_fixes_applied']}")
            if cpp_validation_analysis.get('validation_success_rate'):
                print(f"  Validation success rate: {cpp_validation_analysis['validation_success_rate']:.1f}%")
            if cpp_validation_analysis.get('fix_effectiveness'):
                print(f"  Fix effectiveness: {cpp_validation_analysis['fix_effectiveness']:.1f}%")
        
        if refinement_analysis:
            print(f"\nRefinement Analysis:")
            print(f"  Refined trials success: {refinement_analysis['refined_trials'].get('success_rate', 0):.1f}%")
            print(f"  Fix rate: {refinement_analysis['refined_trials'].get('fix_rate', 0):.1f}%")
        
        print(f"\nResults saved to {self.result_dir}")