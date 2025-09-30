#!/usr/bin/env python3
"""
Prescreener Module - Fast trial prescreening using direct method
Tests each trial with syntax and simulation before deciding on generation method
"""

import time
import tempfile
import os
import subprocess
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from config import Config

class TrialPrescreener:
    """Prescreener for individual trials using direct method"""
    
    def __init__(self, llm_interface, dataset_dir: Path, dataset: str, temp_mode: str):
        """
        Initialize prescreener
        
        Args:
            llm_interface: LLM interface for code generation
            dataset_dir: Directory containing dataset files
            dataset: Dataset type ('rtllm' or 'verilogeval')
            temp_mode: Temperature mode ('low_T' or 'high_T')
        """
        self.llm = llm_interface
        self.dataset_dir = dataset_dir
        self.dataset = dataset
        self.temp_mode = temp_mode
        self.file_extension = Config.get_file_extension(dataset)
        self.language_name = "SystemVerilog" if dataset == "verilogeval" else "Verilog"
        
    def prescreen_trial(self, design: Dict, trial_num: int, description: str) -> Dict:
        """
        Perform prescreening for a single trial
        Must pass both syntax and simulation to be considered successful
        
        Args:
            design: Design information dictionary
            trial_num: Trial number
            description: Design description text
            
        Returns:
            Dictionary with prescreening results
        """
        start_time = time.time()
        
        # Generate code using direct method
        generated_code = self.generate_direct_code(description, design['name'])
        generation_time = time.time() - start_time
        
        if not generated_code:
            return {
                'trial_num': trial_num,
                'prescreening_passed': False,
                'generated_code': None,
                'syntax_passed': False,
                'simulation_passed': False,
                'generation_time': generation_time,
                'test_time': 0,
                'error_info': 'Failed to generate code'
            }
        
        # Quick test with shorter timeout
        test_start = time.time()
        syntax_passed, simulation_passed, error_msg = self.quick_test_code(
            generated_code, design
        )
        test_time = time.time() - test_start
        
        # Both must pass for prescreening success
        prescreening_passed = syntax_passed and simulation_passed
        
        return {
            'trial_num': trial_num,
            'prescreening_passed': prescreening_passed,
            'generated_code': generated_code if prescreening_passed else None,
            'syntax_passed': syntax_passed,
            'simulation_passed': simulation_passed,
            'generation_time': generation_time,
            'test_time': test_time,
            'error_info': error_msg if not prescreening_passed else None
        }
    
    def generate_direct_code(self, description: str, design_name: str) -> Optional[str]:
        """
        Generate code using direct method
        
        Args:
            description: Design description
            design_name: Name of the design
            
        Returns:
            Generated Verilog/SystemVerilog code or None
        """
        # Build prompt based on dataset type
        if self.dataset == "verilogeval":
            prompt = f"""Generate ONLY the complete SystemVerilog module code.

CRITICAL REQUIREMENTS:
- Module MUST be named "TopModule" exactly
- Write syntactically correct and synthesizable SystemVerilog code
- Include all necessary logic for the specified functionality  
- Use proper signal declarations and assignments
- End with 'endmodule'
- Do not include explanations, comments, or additional text

Task Specification:
{description}

Provide the complete TopModule SystemVerilog implementation:"""
            
            system_role = "You are a professional SystemVerilog RTL designer. Generate syntactically correct, synthesizable SystemVerilog code following best practices for digital design."
        else:
            prompt = f"""Generate ONLY the complete Verilog module code.

Requirements:
- Write syntactically correct and synthesizable Verilog code
- Include all necessary logic for the specified functionality  
- Use proper signal declarations and assignments
- End with 'endmodule'
- Do not include explanations, comments, or additional text

Design Specification:
{description}

Provide the complete Verilog module:"""
            
            system_role = "You are a professional Verilog RTL designer. Generate syntactically correct, synthesizable Verilog code following best practices for digital design."
        
        # Generate response
        response = self.llm.generate_response(prompt, system_role)
        
        if response:
            verilog_code = self.llm.extract_verilog(response)
            if verilog_code:
                # Clean and ensure module naming for VerilogEval
                return self.clean_verilog_for_dataset(verilog_code, design_name)
        
        return None
    
    def clean_verilog_for_dataset(self, code: str, design_name: str) -> str:
        """
        Clean Verilog code based on dataset requirements
        
        Args:
            code: Raw Verilog/SystemVerilog code
            design_name: Design name
            
        Returns:
            Cleaned code
        """
        if not code:
            return ""
        
        lines = code.split('\n')
        cleaned_lines = []
        module_found = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines and comments before module
            if not module_found and not stripped:
                continue
            if not module_found and stripped.startswith('//'):
                continue
                
            # Handle module declaration
            if not module_found and stripped.startswith('module '):
                module_found = True
                # For VerilogEval, enforce TopModule naming
                if self.dataset == "verilogeval":
                    line = re.sub(r'module\s+\w+', 'module TopModule', line)
            
            if module_found:
                cleaned_lines.append(line)
                if stripped in ['endmodule', 'endmodule;']:
                    break
        
        return '\n'.join(cleaned_lines) if cleaned_lines else code
    
    def quick_test_code(self, code: str, design: Dict) -> Tuple[bool, bool, Optional[str]]:
        """
        Quick test of generated code with shorter timeout
        
        Args:
            code: Verilog/SystemVerilog code to test
            design: Design information
            
        Returns:
            Tuple of (syntax_passed, simulation_passed, error_message)
        """
        design_name = design['name']
        
        # Find testbench and reference files
        testbench_file, ref_file = self.find_testbench(design_name)
        
        if not testbench_file or not testbench_file.exists():
            return False, False, "No testbench found"
        
        # For VerilogEval, need both testbench and reference
        if self.dataset == "verilogeval" and (not ref_file or not ref_file.exists()):
            return False, False, "No reference file found"
        
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(suffix=self.file_extension, delete=False, mode='w') as f:
            f.write(code)
            verilog_file = f.name
        
        try:
            # Syntax check
            syntax_passed, syntax_error = self.check_syntax(verilog_file)
            if not syntax_passed:
                return False, False, f"Syntax error: {syntax_error}"
            
            # Simulation test
            sim_passed, sim_error = self.check_simulation(
                verilog_file, testbench_file, ref_file
            )
            
            return syntax_passed, sim_passed, sim_error if not sim_passed else None
            
        finally:
            # Cleanup
            if os.path.exists(verilog_file):
                os.unlink(verilog_file)
    
    def find_testbench(self, design_name: str) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find testbench and reference file for design
        
        Args:
            design_name: Name of the design
            
        Returns:
            Tuple of (testbench_path, reference_path)
        """
        if self.dataset == "rtllm":
            # RTLLM: nested directory structure
            if hasattr(Config, 'DESIGN_PATHS') and design_name in Config.DESIGN_PATHS:
                design_dir = Config.DESIGN_PATHS[design_name]
                testbench = design_dir / "testbench.v"
                if testbench.exists():
                    return testbench, None
            
            # Fallback: search in RTLLM directory
            for testbench in self.dataset_dir.rglob("*/testbench.v"):
                parent_dir = testbench.parent
                if parent_dir.name == design_name:
                    return testbench, None
            
            # Direct path attempt
            direct_path = self.dataset_dir / design_name / "testbench.v"
            if direct_path.exists():
                return direct_path, None
        
        elif self.dataset == "verilogeval":
            # VerilogEval: flat structure with _test.sv and _ref.sv
            testbench = self.dataset_dir / f"{design_name}_test.sv"
            ref_file = self.dataset_dir / f"{design_name}_ref.sv"
            
            if testbench.exists() and ref_file.exists():
                return testbench, ref_file
        
        return None, None
    
    def check_syntax(self, verilog_file: str) -> Tuple[bool, Optional[str]]:
        """
        Check syntax using iverilog
        
        Args:
            verilog_file: Path to Verilog file
            
        Returns:
            Tuple of (passed, error_message)
        """
        temp_out = verilog_file.replace(self.file_extension, '.out')
        
        try:
            result = subprocess.run(
                ['iverilog', '-g2012', '-o', temp_out, verilog_file],
                capture_output=True,
                text=True,
                timeout=Config.PRESCREENING_TIMEOUT
            )
            
            if result.returncode == 0:
                return True, None
            else:
                # Extract first error
                for line in result.stderr.split('\n'):
                    if 'error' in line.lower():
                        return False, line.strip()
                return False, "Compilation failed"
                
        except subprocess.TimeoutExpired:
            return False, "Syntax check timeout"
        except Exception as e:
            return False, str(e)
        finally:
            if os.path.exists(temp_out):
                os.unlink(temp_out)
    
    def check_simulation(self, verilog_file: str, testbench_path: Path, 
                        ref_file: Optional[Path]) -> Tuple[bool, Optional[str]]:
        """
        Run simulation test
        
        Args:
            verilog_file: Path to Verilog file
            testbench_path: Path to testbench
            ref_file: Optional reference file for VerilogEval
            
        Returns:
            Tuple of (passed, error_message)
        """
        temp_out = verilog_file.replace(self.file_extension, '.out')
        
        try:
            # Compile with testbench
            if self.dataset == "verilogeval" and ref_file:
                compile_cmd = ['iverilog', '-g2012', '-o', temp_out, 
                             str(testbench_path), verilog_file, str(ref_file)]
            else:
                compile_cmd = ['iverilog', '-g2012', '-o', temp_out, 
                             str(testbench_path), verilog_file]
            
            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=Config.PRESCREENING_TIMEOUT
            )
            
            if compile_result.returncode != 0:
                return False, "Compilation with testbench failed"
            
            # Run simulation
            sim_result = subprocess.run(
                ['vvp', temp_out],
                capture_output=True,
                text=True,
                timeout=Config.PRESCREENING_TIMEOUT
            )
            
            # Parse result
            sim_passed = self.parse_simulation_result(
                sim_result.stdout, sim_result.stderr
            )
            
            if not sim_passed:
                return False, "Simulation test failed"
            
            return True, None
            
        except subprocess.TimeoutExpired:
            return False, "Simulation timeout"
        except Exception as e:
            return False, str(e)
        finally:
            if os.path.exists(temp_out):
                os.unlink(temp_out)
    
    def parse_simulation_result(self, stdout: str, stderr: str) -> bool:
        """
        Parse simulation output to determine pass/fail
        
        Args:
            stdout: Simulation stdout
            stderr: Simulation stderr
            
        Returns:
            True if simulation passed
        """
        if self.dataset == "verilogeval":
            # VerilogEval: Look for mismatches pattern
            mismatch_match = re.search(r'Mismatches: (\d+) in (\d+)', stdout)
            if mismatch_match:
                mismatches = int(mismatch_match.group(1))
                return mismatches == 0
            
            # Fallback check
            if "mismatches: 0" in stdout.lower():
                return True
            elif "mismatches:" in stdout.lower():
                return False
        
        # General case: Check for failure indicators
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