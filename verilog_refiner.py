#!/usr/bin/env python3
"""
Multi-Dataset Verilog Refiner - Iterative refinement using iverilog feedback with adaptive strategies
Supports both RTLLM (Verilog .v files) and VerilogEval (SystemVerilog .sv files)
"""

import subprocess
import re
import tempfile
import os
import difflib
from pathlib import Path
from typing import Dict, Optional, Tuple
from config import Config

class MultiDatasetVerilogRefiner:
    def __init__(self, llm_interface, max_iterations: int = 3, dataset: str = "rtllm", dataset_dir: Path = None):
        self.llm = llm_interface
        self.max_iterations = max_iterations
        self.dataset = dataset
        self.dataset_dir = dataset_dir or (Config.VERILOGEVAL_DIR if dataset == "verilogeval" else Config.RTLLM_DIR)
        self.file_extension = Config.get_file_extension(dataset)
        self.language_name = "SystemVerilog" if dataset == "verilogeval" else "Verilog"
        
    def find_testbench(self, design_name: str) -> tuple:
        """Find testbench and reference file for design based on dataset type"""
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
            
            direct_path = self.dataset_dir / design_name / "testbench.v"
            if direct_path.exists():
                return direct_path, None
        
        elif self.dataset == "verilogeval":
            # VerilogEval: flat structure with _test.sv suffix AND _ref.sv
            testbench = self.dataset_dir / f"{design_name}_test.sv"
            ref_file = self.dataset_dir / f"{design_name}_ref.sv"
            
            if testbench.exists() and ref_file.exists():
                return testbench, ref_file
        
        return None, None
        
    def test_verilog(self, verilog_code: str, testbench_path: Optional[Path]) -> Dict:
        """Test Verilog/SystemVerilog code with iverilog and extract detailed errors"""
        
        # Handle testbench_path as tuple for VerilogEval
        if isinstance(testbench_path, tuple):
            testbench_file, ref_file = testbench_path
        else:
            testbench_file, ref_file = testbench_path, None
        
        with tempfile.NamedTemporaryFile(suffix=self.file_extension, delete=False, mode='w') as f:
            f.write(verilog_code)
            verilog_file = f.name
        
        try:
            # First test: syntax check only
            syntax_result = self.check_syntax(verilog_file)
            
            if not syntax_result['passed']:
                return {
                    'passed': False,
                    'stage': 'syntax',
                    'errors': syntax_result['errors']
                }
            
            # Second test: functional simulation with testbench
            if testbench_file and testbench_file.exists():
                sim_result = self.check_simulation(verilog_file, testbench_file, ref_file)
                return sim_result
            else:
                return {
                    'passed': True,
                    'stage': 'syntax',
                    'errors': []
                }
                
        finally:
            if os.path.exists(verilog_file):
                os.unlink(verilog_file)
    
    def check_syntax(self, verilog_file: str) -> Dict:
        """Check Verilog/SystemVerilog syntax using iverilog"""
        
        temp_out = verilog_file.replace(self.file_extension, '.out')
        
        try:
            result = subprocess.run(
                ['iverilog', '-g2012', '-o', temp_out, verilog_file],
                capture_output=True,
                text=True,
                timeout=Config.COMPILATION_TIMEOUT
            )
            
            if result.returncode == 0:
                return {'passed': True, 'errors': []}
            else:
                errors = self.parse_iverilog_errors(result.stderr)
                return {'passed': False, 'errors': errors}
                
        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'errors': [{
                    'type': 'timeout',
                    'message': f'Compilation timed out after {Config.COMPILATION_TIMEOUT} seconds'
                }]
            }
        except Exception as e:
            return {
                'passed': False,
                'errors': [{
                    'type': 'exception',
                    'message': f'Compilation error: {str(e)}'
                }]
            }
        finally:
            if os.path.exists(temp_out):
                os.unlink(temp_out)
    
    def check_simulation(self, verilog_file: str, testbench_path: Path, ref_file: Path = None) -> Dict:
        """Run simulation with testbench and optional reference file"""
        
        temp_out = verilog_file.replace(self.file_extension, '.out')
        
        try:
            # Compile with testbench (and ref file for VerilogEval)
            if self.dataset == "verilogeval" and ref_file and ref_file.exists():
                # VerilogEval: compile test + generated + ref
                compile_cmd = ['iverilog', '-g2012', '-o', temp_out, str(testbench_path), verilog_file, str(ref_file)]
            else:
                # RTLLM: compile test + generated
                compile_cmd = ['iverilog', '-g2012', '-o', temp_out, str(testbench_path), verilog_file]
            
            try:
                compile_result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=Config.COMPILATION_TIMEOUT
                )
            except subprocess.TimeoutExpired:
                return {
                    'passed': False,
                    'stage': 'syntax',  # Compilation error is still syntax-level
                    'errors': [{
                        'type': 'timeout',
                        'message': f'Compilation with testbench timed out after {Config.COMPILATION_TIMEOUT} seconds'
                    }]
                }
            
            if compile_result.returncode != 0:
                errors = self.parse_iverilog_errors(compile_result.stderr)
                return {
                    'passed': False,
                    'stage': 'syntax',  # Compilation error with testbench is still syntax
                    'errors': errors
                }
            
            # Run simulation with timeout handling
            try:
                sim_result = subprocess.run(
                    ['vvp', temp_out],
                    capture_output=True,
                    text=True,
                    timeout=Config.SIMULATION_TIMEOUT
                )
            except subprocess.TimeoutExpired:
                # Simulation timeout - likely infinite loop
                return {
                    'passed': False,
                    'stage': 'simulation',
                    'errors': [{
                        'type': 'timeout',
                        'message': f'Simulation timed out after {Config.SIMULATION_TIMEOUT} seconds - possible infinite loop or missing signal changes'
                    }]
                }
            
            # Parse simulation output based on dataset
            sim_passed = self.parse_simulation_result(sim_result.stdout, sim_result.stderr)
            
            if not sim_passed:
                errors = self.parse_simulation_errors(sim_result.stdout, sim_result.stderr)
                return {
                    'passed': False,
                    'stage': 'simulation',
                    'errors': errors
                }
            
            return {'passed': True, 'stage': 'simulation', 'errors': []}
            
        except Exception as e:
            return {
                'passed': False,
                'stage': 'simulation',
                'errors': [{
                    'type': 'exception',
                    'message': f'Simulation error: {str(e)}'
                }]
            }
        finally:
            if os.path.exists(temp_out):
                os.unlink(temp_out)
    
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
    
    def parse_iverilog_errors(self, stderr: str) -> list:
        """Extract meaningful error information from iverilog output"""
        
        errors = []
        lines = stderr.split('\n')
        
        for line in lines:
            if 'error:' in line.lower() or 'syntax error' in line.lower():
                # Extract line number if present
                line_match = re.search(r':(\d+):', line)
                line_num = line_match.group(1) if line_match else None
                
                # Clean up error message
                error_msg = re.sub(r'^.*?error:\s*', '', line, flags=re.IGNORECASE)
                
                errors.append({
                    'line': line_num,
                    'message': error_msg.strip(),
                    'raw': line,
                    'type': 'syntax'
                })
        
        return errors if errors else [{'message': stderr.strip(), 'type': 'unknown'}]
    
    def parse_simulation_errors(self, stdout: str, stderr: str) -> list:
        """Extract simulation failure details"""
        
        errors = []
        combined = stdout + '\n' + stderr
        lines = combined.split('\n')
        
        for line in lines:
            if any(word in line.lower() for word in ['fail', 'error', 'mismatch', 'assert']):
                errors.append({
                    'type': 'simulation',
                    'message': line.strip()
                })
        
        return errors if errors else [{'message': 'Simulation failed without specific error', 'type': 'simulation'}]
    
    def generate_fix_prompt(self, verilog_code: str, errors: list, iteration: int, 
                           stage: str, original_description: str = None) -> str:
        """Generate adaptive prompt based on error type and dataset"""
        
        # Format error summary
        error_summary = '\n'.join([
            f"- Line {e.get('line', '?')}: {e['message']}" if e.get('line') else f"- {e['message']}"
            for e in errors[:10]  # Limit to first 10 errors
        ])
        
        # Check for timeout
        has_timeout = any(e.get('type') == 'timeout' for e in errors)
        
        # Dataset-specific language and module naming requirements
        if self.dataset == "verilogeval":
            language_instruction = f"Fix the {self.language_name} code that is"
            module_requirement = "CRITICAL: Ensure the module is named 'TopModule' exactly.\n"
        else:
            language_instruction = f"Fix the {self.language_name} code that is"
            module_requirement = ""
        
        if has_timeout and stage == 'simulation':
            # Simulation timeout-specific prompt
            prompt = f"""{language_instruction} causing simulation timeout.

{module_requirement}Timeout issues typically indicate:
- Infinite loops in always blocks
- Missing sensitivity list items (especially clock edges)
- Combinational loops
- State machines stuck in one state
- Missing signal updates in always blocks

Focus on these specific issues:
1. Check all always blocks have proper sensitivity lists
2. Ensure state machines can transition between states
3. Verify counters increment/decrement properly
4. Check for missing default cases in case statements

Current {self.language_name} Code:
{verilog_code}

Error Details:
{error_summary}

Provide the complete corrected {self.language_name} module:"""
        
        elif stage == 'syntax':
            # Syntax/compilation error - focused fix
            error_lines = [e.get('line') for e in errors if e.get('line')]
            error_lines_str = ', '.join(filter(None, error_lines)) if error_lines else 'unknown'
            
            prompt = f"""{language_instruction} causing syntax/compilation errors.

{module_requirement}Errors detected at lines: {error_lines_str}

IMPORTANT RULES:
- Focus ONLY on fixing the reported syntax/compilation errors
- Make minimal changes to correct these issues
- Do NOT change logic or functionality
- Preserve all working parts of the code
- Common issues: missing semicolons, mismatched begin/end, incorrect port declarations, undefined signals

Specific Errors:
{error_summary}

Current {self.language_name} Code:
{verilog_code}

Provide the corrected {self.language_name} module with errors fixed:"""
        
        elif stage == 'simulation':
            # Simulation error - broader fix allowed
            prompt = f"""{language_instruction} failing simulation tests.

{module_requirement}The code compiles successfully but fails functional tests. Common causes:
- Incorrect logic implementation
- Wrong arithmetic operations (especially for multipliers/adders)
- Incorrect state machine transitions
- Missing or wrong initial values
- Timing issues with sequential logic
- Off-by-one errors in counters or indices
- Incorrect bit width assignments
- Wrong operators or conditions

GUIDELINES FOR FIXING:
- Analyze the error patterns to identify the root cause
- You may modify multiple related sections if needed
- Ensure the module correctly implements the required functionality
- Keep the module interface (input/output ports) unchanged
- Pay special attention to edge cases and boundary conditions

Current Simulation Errors:
{error_summary}

Current {self.language_name} Code:
{verilog_code}
"""
            # Add original spec on first iteration for simulation errors
            if original_description and iteration == 1:
                prompt += f"\nOriginal Design Specification (for reference):\n{original_description}\n"
            
            prompt += f"\nProvide the complete corrected {self.language_name} module:"
        
        else:
            # Fallback generic prompt
            prompt = f"""{language_instruction} based on the errors reported.

{module_requirement}Current Errors:
{error_summary}

Current {self.language_name} Code:
{verilog_code}

Requirements:
- Fix all reported errors
- Maintain module interface compatibility
- Ensure functional correctness

Provide the complete corrected {self.language_name} module:"""
        
        return prompt
    
    def check_excessive_changes(self, original: str, modified: str) -> bool:
        """Check if modifications are excessive (only for syntax fixes)"""
        # Calculate similarity ratio
        similarity = difflib.SequenceMatcher(None, original, modified).ratio()
        
        # For syntax fixes, expect high similarity (>70%)
        # This is only a warning, not a hard block
        return similarity < 0.7
    
    def refine_verilog(self, initial_code: str, testbench_path: Optional[Path] = None, 
                    original_description: str = None) -> Tuple[str, Dict]:
        """Iteratively refine Verilog/SystemVerilog code with adaptive strategies"""
        
        current_code = initial_code
        history = []
        
        for iteration in range(1, self.max_iterations + 1):
            # Test current code - pass testbench_path directly (might be tuple for VerilogEval)
            test_result = self.test_verilog(current_code, testbench_path)
            
            history.append({
                'iteration': iteration,
                'passed': test_result['passed'],
                'stage': test_result.get('stage'),
                'errors': test_result.get('errors', [])
            })
            
            if test_result['passed']:
                return current_code, {
                    'success': True,
                    'iterations': iteration,
                    'history': history
                }
            
            # If failed and not last iteration, try to fix
            if iteration < self.max_iterations:
                # Select appropriate system role based on error type and dataset
                stage = test_result.get('stage', 'unknown')
                if stage == 'syntax':
                    if self.dataset == "verilogeval":
                        system_role = f"You are a {self.language_name} syntax expert. Fix syntax and compilation errors with minimal, targeted changes. Ensure module is named 'TopModule'. Do not rewrite working code."
                    else:
                        system_role = f"You are a {self.language_name} syntax expert. Fix syntax and compilation errors with minimal, targeted changes. Do not rewrite working code."
                elif stage == 'simulation':
                    if self.dataset == "verilogeval":
                        system_role = f"You are an expert {self.language_name} RTL designer. Analyze and fix functional errors to pass simulation tests. Focus on logic correctness. Ensure module is named 'TopModule'."
                    else:
                        system_role = f"You are an expert {self.language_name} RTL designer. Analyze and fix functional errors to pass simulation tests. Focus on logic correctness."
                else:
                    system_role = f"You are a professional {self.language_name} designer. Fix all errors while maintaining code structure."
                
                # Generate adaptive fix prompt
                fix_prompt = self.generate_fix_prompt(
                    current_code, 
                    test_result['errors'],
                    iteration,
                    stage,
                    original_description if iteration == 1 else None
                )
                
                # Get fixed code from LLM
                fixed_response = self.llm.generate_response(fix_prompt, system_role)
                
                if fixed_response:
                    extracted = self.llm.extract_verilog(fixed_response)
                    if extracted:
                        # Check for excessive changes only for syntax fixes
                        if stage == 'syntax' and self.check_excessive_changes(current_code, extracted):
                            print(f"  [REFINE] Warning: Large changes for syntax fix (iteration {iteration})")
                            # Still accept the changes as they might be necessary
                        
                        current_code = extracted
                    else:
                        print(f"  [REFINE] Failed to extract valid {self.language_name} (iteration {iteration})")
                        break
                else:
                    print(f"  [REFINE] No response from LLM (iteration {iteration})")
                    break
        
        # Max iterations reached without success
        return current_code, {
            'success': False,
            'iterations': self.max_iterations,
            'history': history
        }