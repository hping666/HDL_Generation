#!/usr/bin/env python3
"""
Quality Evaluator for HDL code using iverilog syntax and function testing with severity-weighted scoring
"""

import subprocess
import tempfile
import os
import time
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from config import Config

class HDLQualityEvaluator:
    def __init__(self, dataset_dir: Path, dataset: str = "rtllm"):
        self.dataset_dir = dataset_dir
        self.dataset = dataset
        self.file_extension = Config.get_file_extension(dataset)
        
    def evaluate_quality(self, code: str, design_name: str) -> float:
        """
        Evaluate HDL code quality using iverilog testing with severity-weighted scoring
        Returns: 
        - 1.0: Both syntax and function tests pass (highest quality)
        - 0.45-0.85: Only syntax test passes (weighted by error severity)
        - 0.0-0.6: Neither pass, evaluated by fallback rules
        """
        if not code or not code.strip():
            return 0.0
        
        # Step 1: Syntax test
        syntax_score = self._test_syntax(code)
        if syntax_score == 0.0:
            return self._fallback_evaluation(code)
        
        # Step 2: Function test (only if syntax passes)
        function_score = self._test_function(code, design_name)
        
        if function_score > 0.0:
            return 1.0  # Both syntax and function pass - highest quality
        else:
            # Syntax passes but function fails - severity-weighted evaluation
            return self._severity_weighted_evaluation(code)
    
    def _test_syntax(self, code: str) -> float:
        """Test syntax using iverilog compilation"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=self.file_extension, delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            temp_out = f"/tmp/syntax_test_{int(time.time())}.out"
            
            # Syntax check with iverilog
            syntax_cmd = ["iverilog", "-g2012", "-o", temp_out, temp_file]
            result = subprocess.run(syntax_cmd, capture_output=True, text=True, 
                                  timeout=Config.COMPILATION_TIMEOUT)
            
            # Cleanup
            try:
                os.unlink(temp_file)
                if os.path.exists(temp_out):
                    os.unlink(temp_out)
            except:
                pass
            
            return 1.0 if result.returncode == 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _test_function(self, code: str, design_name: str) -> float:
        """Test function using iverilog + testbench simulation"""
        try:
            testbench_result = self._find_testbench(design_name)
            if isinstance(testbench_result, tuple):
                testbench, ref_file = testbench_result
            else:
                testbench, ref_file = testbench_result, None
                
            if not testbench:
                return 0.0
            
            # For VerilogEval, we need both testbench and ref file
            if self.dataset == "verilogeval" and not ref_file:
                return 0.0
            
            with tempfile.NamedTemporaryFile(mode='w', suffix=self.file_extension, delete=False) as f:
                f.write(code)
                code_file = f.name
            
            temp_out = f"/tmp/func_test_{int(time.time())}.out"
            
            # Compilation with testbench
            if self.dataset == "verilogeval" and ref_file:
                compile_cmd = ["iverilog", "-g2012", "-o", temp_out, str(testbench), code_file, str(ref_file)]
            else:
                compile_cmd = ["iverilog", "-g2012", "-o", temp_out, str(testbench), code_file]
            
            compile_result = subprocess.run(compile_cmd, capture_output=True, text=True, 
                                          timeout=Config.COMPILATION_TIMEOUT)
            
            if compile_result.returncode != 0:
                return 0.0
            
            # Simulation
            sim_cmd = ["vvp", temp_out]
            sim_result = subprocess.run(sim_cmd, capture_output=True, text=True, 
                                      timeout=Config.SIMULATION_TIMEOUT)
            
            # Cleanup
            try:
                os.unlink(code_file)
                if os.path.exists(temp_out):
                    os.unlink(temp_out)
            except:
                pass
            
            # Parse simulation results
            success = self._parse_simulation_result(sim_result.stdout, sim_result.stderr)
            return 1.0 if success else 0.0
            
        except Exception:
            return 0.0
    
    def _severity_weighted_evaluation(self, code: str) -> float:
        """
        Severity-weighted evaluation for code that passes syntax but fails function tests
        Uses different penalty weights based on error severity:
        - Logic errors: -0.1 to -0.2 (severe)
        - Synthesis issues: -0.05 to -0.1 (moderate) 
        - Style issues: -0.01 to -0.03 (minor)
        """
        base_score = 0.85  # Base score for syntax-passing code
        total_penalty = 0.0
        
        # SEVERE: Logic errors (heavy penalties -0.1 to -0.2)
        logic_penalty = self._evaluate_logic_errors(code)
        total_penalty += logic_penalty
        
        # MODERATE: Synthesis/implementation issues (medium penalties -0.05 to -0.1)
        synthesis_penalty = self._evaluate_synthesis_issues(code)
        total_penalty += synthesis_penalty
        
        # MINOR: Style and structure issues (light penalties -0.01 to -0.03)
        style_penalty = self._evaluate_style_issues(code)
        total_penalty += style_penalty
        
        final_score = base_score - total_penalty
        return max(final_score, 0.45)  # Floor at 0.45, ceiling remains at 0.85
    
    def _evaluate_logic_errors(self, code: str) -> float:
        """Evaluate logic errors with severe penalties (0.0-0.4)"""
        penalty = 0.0
        
        # Severe: Mixed signal assignments in single always block
        always_blocks = re.findall(r'always\s*@[^}]*?end', code, re.DOTALL | re.IGNORECASE)
        
        for block in always_blocks:
            # Check for multiple independent signals in same block (major logic error)
            signal_assignments = re.findall(r'(\w+)\s*<=', block)
            unique_signals = set(signal_assignments)
            
            if len(unique_signals) > 3:  # More than 3 different signals
                penalty += 0.2  # Severe penalty
            elif len(unique_signals) > 2:  # 2-3 signals might be problematic
                penalty += 0.1  # Moderate penalty
        
        # Severe: Signal driving conflicts (multiple always blocks driving same signal)
        signal_counts = {}
        for signal in re.findall(r'(\w+)\s*<=', code):
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        for signal in signal_counts:
            block_count = 0
            for block in always_blocks:
                if re.search(f'{signal}\\s*<=', block):
                    block_count += 1
            
            if block_count > 1:  # Signal driven from multiple blocks
                penalty += 0.15  # Severe logic error
        
        # Severe: Incomplete sequential logic
        for block in always_blocks:
            if '@' in block and 'posedge' in block.lower():
                # Sequential block should have proper reset handling
                if not re.search(r'if\s*\(\s*\w*[Rr][Ss][Tt]\w*\s*\)', block):
                    penalty += 0.1  # Missing reset is serious
                    
                # Check for incomplete state transitions
                if_count = len(re.findall(r'\bif\s*\(', block))
                lines = block.split('\n')
                incomplete_logic = 0
                
                for line in lines:
                    # Signal <= Signal pattern often indicates missing logic
                    if re.search(r'(\w+)\s*<=\s*\1\s*;', line):
                        incomplete_logic += 1
                
                if incomplete_logic > 2:  # Multiple "keep current value" assignments
                    penalty += 0.1  # Suggests incomplete logic
        
        return min(penalty, 0.4)  # Cap at 0.4
    
    def _evaluate_synthesis_issues(self, code: str) -> float:
        """Evaluate synthesis/implementation issues with moderate penalties (0.0-0.2)"""
        penalty = 0.0
        
        # Moderate: Mixed edge sensitivity in always blocks
        always_blocks = re.findall(r'always\s*@\s*\([^)]+\)', code, re.IGNORECASE)
        for block in always_blocks:
            # Check for mixed posedge/negedge with other signals
            if ('posedge' in block.lower() or 'negedge' in block.lower()) and 'or' in block.lower():
                # This is actually okay for reset, but check for problematic patterns
                edge_signals = re.findall(r'(?:posedge|negedge)\s+(\w+)', block.lower())
                other_signals = re.findall(r'or\s+(?!posedge|negedge)(\w+)', block.lower())
                
                if len(other_signals) > 1:  # Multiple non-edge signals
                    penalty += 0.05  # Synthesis concern
        
        # Moderate: Combinational loops potential
        assign_statements = re.findall(r'assign\s+(\w+)\s*=', code)
        always_assignments = re.findall(r'(\w+)\s*<=', code)
        
        for signal in assign_statements:
            if signal in always_assignments:
                penalty += 0.1  # Signal driven by both assign and always
        
        # Moderate: Width mismatch potential
        width_declarations = re.findall(r'\[(\d+):(\d+)\]\s*(\w+)', code)
        for width_decl in width_declarations:
            high, low, signal = width_decl
            expected_width = int(high) - int(low) + 1
            
            # Look for assignments that might have width issues
            assignments = re.findall(f'{signal}\\s*<=\\s*([^;]+)', code)
            for assignment in assignments:
                # Simple check for obvious width mismatches
                if "'b" in assignment or "'d" in assignment:
                    # Extract bit width from literals
                    width_match = re.search(r"(\d+)'[bd]", assignment)
                    if width_match:
                        literal_width = int(width_match.group(1))
                        if literal_width != expected_width:
                            penalty += 0.05  # Width mismatch concern
        
        return min(penalty, 0.2)  # Cap at 0.2
    
    def _evaluate_style_issues(self, code: str) -> float:
        """Evaluate style/formatting issues with minor penalties (0.0-0.06)"""
        penalty = 0.0
        
        # Minor: Inconsistent indentation
        lines = [line for line in code.split('\n') if line.strip()]
        if lines:
            indented_lines = [line for line in lines if line.startswith(' ') or line.startswith('\t')]
            if len(indented_lines) < len(lines) * 0.3:  # Less than 30% indented
                penalty += 0.02
        
        # Minor: Inconsistent signal width specifications
        reg_declarations = re.findall(r'reg\s+(?:\[[\d:]+\]\s+)?(\w+)', code)
        width_specs = re.findall(r'reg\s+\[[\d:]+\]', code)
        
        if len(reg_declarations) > len(width_specs) and len(reg_declarations) > 2:
            penalty += 0.01  # Some signals without explicit width
        
        # Minor: Inconsistent reset signal naming
        reset_patterns = re.findall(r'if\s*\(\s*(!?\w*[Rr][Ss][Tt]\w*)\s*\)', code)
        if len(set(reset_patterns)) > 1:  # Multiple different reset patterns
            penalty += 0.01
        
        # Minor: Missing wire/reg declarations consistency
        if 'input wire' in code and 'input' in code and code.count('input wire') < code.count('input'):
            penalty += 0.01  # Mixed input declaration styles
        
        # Minor: Spacing and formatting
        if '(' in code and code.count(' (') < code.count('(') * 0.5:
            penalty += 0.01  # Poor spacing around parentheses
            
        return min(penalty, 0.06)  # Cap at 0.06
    
    def _find_testbench(self, design_name: str) -> tuple:
        """Find testbench and reference file for design based on dataset type"""
        if self.dataset == "rtllm":
            if hasattr(Config, 'DESIGN_PATHS') and design_name in Config.DESIGN_PATHS:
                design_dir = Config.DESIGN_PATHS[design_name]
                testbench = design_dir / "testbench.v"
                if testbench.exists():
                    return testbench, None
            
            for testbench in self.dataset_dir.rglob("*/testbench.v"):
                parent_dir = testbench.parent
                if parent_dir.name == design_name:
                    return testbench, None
            
            direct_path = self.dataset_dir / design_name / "testbench.v"
            if direct_path.exists():
                return direct_path, None
        
        elif self.dataset == "verilogeval":
            testbench = self.dataset_dir / f"{design_name}_test.sv"
            ref_file = self.dataset_dir / f"{design_name}_ref.sv"
            
            if testbench.exists() and ref_file.exists():
                return testbench, ref_file
        
        return None, None
    
    def _parse_simulation_result(self, stdout: str, stderr: str) -> bool:
        """Parse simulation result based on dataset type"""
        if self.dataset == "verilogeval":
            mismatch_match = re.search(r'Mismatches: (\d+) in (\d+)', stdout)
            if mismatch_match:
                mismatches = int(mismatch_match.group(1))
                return mismatches == 0
            
            if "mismatches: 0" in stdout.lower():
                return True
            elif "mismatches:" in stdout.lower():
                return False
        
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
    
    def _fallback_evaluation(self, code: str) -> float:
        """Fallback evaluation using rule-based scoring when iverilog tests fail"""
        score = 0.0
        
        if 'module' in code and 'endmodule' in code:
            score += 0.2
        
        if re.search(r'input|output', code):
            score += 0.15
        
        logic_keywords = ['always', 'assign', 'if', 'case', 'for', 'while']
        logic_count = sum(1 for kw in logic_keywords if kw in code)
        score += min(logic_count * 0.05, 0.15)
        
        if self.dataset == "verilogeval":
            if re.search(r'module\s+TopModule', code):
                score += 0.1
        else:
            if re.search(r'module\s+[a-zA-Z_][a-zA-Z0-9_]*', code):
                score += 0.1
        
        lines = len([l for l in code.split('\n') if l.strip()])
        if 5 <= lines <= 200:
            score += 0.05
        
        if code.count('(') == code.count(')') and code.count('[') == code.count(']'):
            score += 0.05
        
        return min(score, 0.6)  # Cap at 0.6 for fallback evaluation