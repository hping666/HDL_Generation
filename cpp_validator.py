#!/usr/bin/env python3
"""
C++ Validator Module - Validates and refines intermediate C++ code in cpp_chain pipeline
Performs functional validation and HLS compatibility checks for generated C++ code
"""

import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from config import Config

class CppValidator:
    def __init__(self, llm_interface, max_iterations: int = 2):
        """
        Initialize C++ validator
        
        Args:
            llm_interface: LLM interface for validation and refinement
            max_iterations: Maximum refinement iterations for C++ code
        """
        self.llm = llm_interface
        self.max_iterations = max_iterations
        self.validation_cache = {}
        
    def validate_cpp_structure(self, cpp_code: str) -> Dict:
        """
        Check C++ code for HLS compatibility and structural correctness
        
        Args:
            cpp_code: C++ code to validate
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Check for dynamic memory allocation
        if any(keyword in cpp_code for keyword in ['new ', 'delete ', 'malloc', 'free', 'vector<', 'list<', 'map<']):
            issues.append({
                'type': 'dynamic_memory',
                'severity': 'error',
                'message': 'Dynamic memory allocation detected - not HLS compatible'
            })
        
        # Check for recursive functions
        function_pattern = r'(\w+)\s*\([^)]*\)\s*{([^}]*)}'
        functions = re.findall(function_pattern, cpp_code, re.DOTALL)
        for func_name, func_body in functions:
            if func_name in func_body:
                issues.append({
                    'type': 'recursion',
                    'severity': 'error',
                    'message': f'Recursive function {func_name} detected - not HLS compatible'
                })
        
        # Check for proper bit-width types
        if not any(btype in cpp_code for btype in ['uint8_t', 'uint16_t', 'uint32_t', 'int8_t', 'int16_t', 'int32_t', 'bool']):
            issues.append({
                'type': 'bit_width',
                'severity': 'warning',
                'message': 'No explicit bit-width types found - may cause synthesis issues'
            })
        
        # Check for unbounded loops
        while_pattern = r'while\s*\([^)]*\)'
        while_loops = re.findall(while_pattern, cpp_code)
        for loop in while_loops:
            if 'true' in loop.lower() or '1' == loop.strip()[-2]:
                issues.append({
                    'type': 'unbounded_loop',
                    'severity': 'error',
                    'message': 'Potentially unbounded while loop detected'
                })
        
        return {
            'valid': len([i for i in issues if i['severity'] == 'error']) == 0,
            'issues': issues,
            'warnings': len([i for i in issues if i['severity'] == 'warning']),
            'errors': len([i for i in issues if i['severity'] == 'error'])
        }
    
    def validate_cpp_functionality(self, cpp_code: str, design_spec: str) -> Dict:
        """
        Validate C++ functional correctness against design specification
        
        Args:
            cpp_code: C++ code to validate
            design_spec: Original design specification
            
        Returns:
            Dictionary with functional validation results
        """
        prompt = f"""Analyze if this C++ code correctly implements the specified functionality.

Design Specification:
{design_spec}

C++ Code:
{cpp_code}

Check for:
1. Does the C++ code implement all required operations?
2. Are the input/output interfaces correct?
3. Is the algorithmic logic correct?
4. Are there any obvious functional errors?

Provide a structured analysis:
- Correctness: [CORRECT/INCORRECT/PARTIAL]
- Missing features: [list any]
- Logic errors: [list any]
- Interface issues: [list any]

Be concise and specific."""

        system_role = "You are an expert in hardware design and HLS C++ programming. Analyze code functionality against specifications precisely."
        
        response = self.llm.generate_response(prompt, system_role)
        
        if response:
            # Parse LLM response
            correctness = "CORRECT" if "CORRECT" in response and "INCORRECT" not in response else "INCORRECT"
            has_errors = "logic error" in response.lower() or "missing" in response.lower()
            
            return {
                'functionally_correct': correctness == "CORRECT",
                'analysis': response,
                'has_errors': has_errors
            }
        
        return {
            'functionally_correct': False,
            'analysis': 'Failed to analyze',
            'has_errors': True
        }
    
    def should_fix_cpp(self, verilog_errors: List[Dict], cpp_code: str, design_spec: str) -> Dict:
        """
        Determine if Verilog errors stem from C++ issues
        
        Args:
            verilog_errors: List of Verilog simulation errors
            cpp_code: Generated C++ code
            design_spec: Original design specification
            
        Returns:
            Dictionary with decision and reasoning
        """
        # Quick heuristics
        error_messages = ' '.join([e.get('message', '') for e in verilog_errors])
        
        # Indicators that C++ might be wrong
        cpp_indicators = [
            'wrong output',
            'incorrect result',
            'mismatch in expected',
            'logic error',
            'wrong calculation',
            'incorrect algorithm'
        ]
        
        # Indicators that it's a Verilog translation issue
        verilog_indicators = [
            'synthesis',
            'timing',
            'clock',
            'reset',
            'sensitivity list',
            'always block'
        ]
        
        cpp_score = sum(1 for ind in cpp_indicators if ind in error_messages.lower())
        verilog_score = sum(1 for ind in verilog_indicators if ind in error_messages.lower())
        
        # If strong Verilog indicators, don't check C++
        if verilog_score > cpp_score:
            return {'fix_cpp': False, 'reason': 'Likely Verilog translation issue'}
        
        # Validate C++ if indicators suggest functional problems
        if cpp_score > 0 or verilog_score == 0:
            structure_result = self.validate_cpp_structure(cpp_code)
            if not structure_result['valid']:
                return {'fix_cpp': True, 'reason': 'C++ structural issues detected'}
            
            func_result = self.validate_cpp_functionality(cpp_code, design_spec)
            if not func_result['functionally_correct']:
                return {'fix_cpp': True, 'reason': 'C++ functional issues detected'}
        
        return {'fix_cpp': False, 'reason': 'C++ appears correct, likely translation issue'}
    
    def refine_cpp_code(self, cpp_code: str, issues: List[Dict], design_spec: str, iteration: int = 1) -> Optional[str]:
        """
        Refine C++ code based on identified issues
        
        Args:
            cpp_code: Current C++ code
            issues: List of issues to fix
            design_spec: Original design specification
            iteration: Current refinement iteration
            
        Returns:
            Refined C++ code or None if refinement fails
        """
        # Format issues for prompt
        issue_summary = '\n'.join([
            f"- {issue['type']}: {issue['message']}"
            for issue in issues[:5]  # Limit to first 5 issues
        ])
        
        prompt = f"""Fix the HLS C++ code based on these issues:

Issues to fix:
{issue_summary}

Current C++ Code:
{cpp_code}

Original Design Requirements:
{design_spec}

Requirements for fixed code:
- Must be HLS-compatible (no dynamic memory, no recursion)
- Use fixed-size arrays only
- Use explicit bit-width types (uint8_t, uint16_t, etc.)
- Ensure all loops are bounded
- Maintain functional correctness

Provide the complete corrected C++ code:"""

        system_role = "You are an HLS C++ expert. Fix code to be synthesis-compatible while maintaining functionality."
        
        response = self.llm.generate_response(prompt, system_role)
        
        if response:
            # Extract C++ code from response
            return self.extract_cpp_code(response)
        
        return None
    
    def extract_cpp_code(self, response: str) -> Optional[str]:
        """
        Extract C++ code from LLM response
        
        Args:
            response: LLM response containing C++ code
            
        Returns:
            Extracted C++ code or None
        """
        if not response:
            return None
        
        # Look for code blocks
        lines = response.split('\n')
        code_lines = []
        in_block = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('```'):
                if in_block:
                    break
                else:
                    in_block = True
                    continue
            if in_block:
                code_lines.append(line)
        
        # If no code blocks, try to extract function definitions
        if not code_lines:
            for i, line in enumerate(lines):
                stripped = line.strip()
                if any(stripped.startswith(kw) for kw in ['#include', 'void ', 'int ', 'uint', 'bool ', 'class ', 'struct ']):
                    code_lines = lines[i:]
                    break
        
        return '\n'.join(code_lines) if code_lines else response
    
    def validate_and_refine_cpp(self, cpp_code: str, design_spec: str, verilog_errors: List[Dict] = None) -> Tuple[str, Dict]:
        """
        Complete validation and refinement pipeline for C++ code
        
        Args:
            cpp_code: C++ code to validate and refine
            design_spec: Original design specification
            verilog_errors: Optional Verilog errors to consider
            
        Returns:
            Tuple of (refined_cpp_code, validation_info)
        """
        history = []
        current_code = cpp_code
        
        for iteration in range(1, self.max_iterations + 1):
            # Structural validation
            structure_result = self.validate_cpp_structure(current_code)
            
            # Functional validation
            func_result = self.validate_cpp_functionality(current_code, design_spec)
            
            history.append({
                'iteration': iteration,
                'structure_valid': structure_result['valid'],
                'functionally_correct': func_result['functionally_correct'],
                'issues': structure_result['issues']
            })
            
            # If both validations pass, return
            if structure_result['valid'] and func_result['functionally_correct']:
                return current_code, {
                    'success': True,
                    'iterations': iteration,
                    'history': history
                }
            
            # If not last iteration, try to fix
            if iteration < self.max_iterations:
                all_issues = structure_result['issues']
                if not func_result['functionally_correct']:
                    all_issues.append({
                        'type': 'functional',
                        'severity': 'error',
                        'message': 'Functional correctness issues detected'
                    })
                
                refined_code = self.refine_cpp_code(current_code, all_issues, design_spec, iteration)
                
                if refined_code:
                    current_code = refined_code
                else:
                    break
        
        return current_code, {
            'success': False,
            'iterations': self.max_iterations,
            'history': history
        }