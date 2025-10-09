#!/usr/bin/env python3
"""
LLM Interface optimized for Qwen2.5 series with enhanced code extraction
"""

import subprocess
import json
import time
import re
from typing import Optional, List
from config import Config

class OllamaInterface:
    def __init__(self, model_name: str, temp_mode: str = "low_T"):
        self.model_name = model_name
        self.temp_mode = temp_mode
        self.params = Config.get_model_params(model_name, temp_mode)
        
    def update_temperature_mode(self, temp_mode: str):
        """Update temperature mode and refresh parameters"""
        self.temp_mode = temp_mode
        self.params = Config.get_model_params(self.model_name, temp_mode)
        
    def test_connection(self) -> bool:
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            return result.returncode == 0 and self.model_name in result.stdout
        except:
            return False
    
    def generate_response(self, prompt: str, system_role: str = None) -> Optional[str]:
        """Generate response with dynamic temperature settings and dataset-aware defaults"""
        try:
            import requests
            
            # Use provided system role or dataset-specific default
            if system_role is None:
                if "systemverilog" in prompt.lower():
                    system_role = "You are a professional SystemVerilog designer. Provide clean, functional SystemVerilog code without explanations."
                else:
                    system_role = "You are a professional Verilog designer. Provide clean, functional Verilog code without explanations."
            
            api_request = {
                "model": self.model_name,
                "prompt": f"System: {system_role}\n\nUser: {prompt}",
                "stream": False,
                "options": {
                    "temperature": self.params["temperature"],
                    "top_p": self.params["top_p"],
                    "num_predict": self.params["num_predict"],
                    "num_ctx": self.params["context_length"],
                    "stop": ["<|im_end|>", "User:", "System:"]
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=api_request,
                timeout=self.params["timeout"]
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
                
        except ImportError:
            # Fallback to ollama run
            try:
                full_prompt = f"{system_role}\n\n{prompt}" if system_role else prompt
                result = subprocess.run(
                    ["ollama", "run", self.model_name],
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.params["timeout"]
                )
                
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
                
        except Exception:
            pass
        return None
    
    def extract_verilog(self, response: str, dataset: str = "rtllm") -> Optional[str]:
        """
        Enhanced Verilog code extraction with multiple strategies
        Adapted from MoA_verify.py for improved robustness
        """
        if not response:
            return None
        
        response = response.strip()
        
        # Step 1: Remove markdown code blocks using regex
        response = re.sub(r'```(?:systemverilog|verilog|sv|v)?\s*\n?', '', response, flags=re.IGNORECASE)
        response = re.sub(r'```\s*$', '', response, flags=re.MULTILINE)
        
        # Step 2: Remove common prefixes using regex patterns
        prefixes_to_remove = [
            r"Here's the (?:System)?Verilog (?:code|module|implementation):?\s*",
            r"Here is the (?:System)?Verilog (?:code|module|implementation):?\s*",
            r"The (?:System)?Verilog (?:code|module) is:?\s*",
            r"Output:?\s*", 
            r"Solution:?\s*", 
            r"Implementation:?\s*", 
            r"Code:?\s*",
            r"(?:System)?Verilog:?\s*", 
            r"Generated (?:System)?Verilog module:?\s*", 
            r"Module code:?\s*"
        ]
        
        for prefix in prefixes_to_remove:
            response = re.sub(f'^{prefix}', '', response, flags=re.IGNORECASE | re.MULTILINE)
        
        # Step 3: Try multiple regex patterns to find module boundaries
        module_patterns = [
            r'\b(module\s+[a-zA-Z_][a-zA-Z0-9_]*.*?endmodule\s*;?)\b',
            r'(module\s+\w+[^;]*?[\s\S]*?endmodule\s*;?)',
            r'(module[\s\S]+?endmodule)'
        ]
        
        for pattern in module_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                code = matches[0]
                code = self._clean_extracted_code(code, dataset)
                
                if self._validate_extracted_code(code, dataset):
                    return code.strip()
        
        # Step 4: Fallback - extract by lines
        code = self._extract_code_by_lines(response, dataset)
        if code and self._validate_extracted_code(code, dataset):
            return code
        
        # Step 5: Last resort - salvage attempt
        if 'module' in response.lower():
            code = self._salvage_module_code(response, dataset)
            if code and self._validate_extracted_code(code, dataset):
                return code
        
        return None
    
    def _clean_extracted_code(self, code: str, dataset: str) -> str:
        """Clean up extracted code"""
        # Remove comments for SystemVerilog, keep for regular Verilog
        if dataset == "verilogeval":
            code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Remove markdown remnants
        code = re.sub(r'```.*?$', '', code, flags=re.MULTILINE)
        
        # Clean excessive empty lines (keep max 1)
        lines = code.split('\n')
        cleaned_lines = []
        empty_line_count = 0
        
        for line in lines:
            if line.strip():
                empty_line_count = 0
                cleaned_lines.append(line)
            elif empty_line_count < 1:
                empty_line_count += 1
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_code_by_lines(self, response: str, dataset: str) -> Optional[str]:
        """Extract code by processing line by line"""
        lines = response.split('\n')
        module_started = False
        code_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip markdown
            if stripped.startswith('```'):
                continue
            
            # Detect module start
            if not module_started and re.match(r'module\s+\w+', stripped, re.IGNORECASE):
                module_started = True
                code_lines.append(line)
                continue
            
            if module_started:
                code_lines.append(line)
                
                # Detect end
                if re.match(r'endmodule\s*;?\s*$', stripped, re.IGNORECASE):
                    break
                
                # Prevent infinite loops
                if len(code_lines) > 1000:
                    break
        
        if code_lines:
            code = '\n'.join(code_lines)
            
            # Ensure endmodule exists
            if not re.search(r'endmodule\s*;?\s*$', code, re.MULTILINE | re.IGNORECASE):
                code += '\nendmodule'
            
            return code.strip()
        
        return None
    
    def _salvage_module_code(self, response: str, dataset: str) -> Optional[str]:
        """Last resort attempt to salvage module code"""
        module_start = re.search(r'module\s+\w+', response, re.IGNORECASE)
        if not module_start:
            return None
        
        # Extract from module start
        code = response[module_start.start():]
        
        # Find endmodule
        endmodule_match = re.search(r'endmodule\s*;?\s*', code, re.IGNORECASE)
        if endmodule_match:
            code = code[:endmodule_match.end()]
        else:
            code = code + '\nendmodule'
        
        code = self._clean_extracted_code(code, dataset)
        
        return code
    
    def _validate_extracted_code(self, code: str, dataset: str) -> bool:
        """Validate extracted code meets basic requirements"""
        if not code:
            return False
        
        # Must have module declaration
        if not re.search(r'module\s+\w+', code, re.IGNORECASE):
            return False
        
        # Must have endmodule
        if not re.search(r'endmodule', code, re.IGNORECASE):
            return False
        
        # Check module name validity
        if not re.search(r'module\s+[a-zA-Z_][a-zA-Z0-9_]*', code):
            return False
        
        # No markdown remnants
        if '```' in code:
            return False
        
        # Module and endmodule must be paired and unique
        module_count = len(re.findall(r'module\s+\w+', code, re.IGNORECASE))
        endmodule_count = len(re.findall(r'endmodule', code, re.IGNORECASE))
        
        if module_count != 1 or endmodule_count != 1:
            return False
        
        # Minimum length check
        min_length = 25
        if len(code) < min_length:
            return False
        
        return True
    
    def extract_cpp_code(self, response: str) -> Optional[str]:
        """
        Enhanced C++ code extraction with multiple strategies
        """
        if not response:
            return None
        
        response = response.strip()
        
        # Remove markdown code blocks
        response = re.sub(r'```(?:cpp|c\+\+|c)?\s*\n?', '', response, flags=re.IGNORECASE)
        response = re.sub(r'```\s*$', '', response, flags=re.MULTILINE)
        
        # Look for code blocks first
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
                if any(stripped.startswith(kw) for kw in ['#include', 'void ', 'int ', 'class ', 'struct ', 'bool ', 'uint8_t', 'uint16_t', 'uint32_t']):
                    code_lines = lines[i:]
                    break
        
        if code_lines:
            code = '\n'.join(code_lines)
            # Basic validation
            if any(kw in code for kw in ['#include', 'void', 'int', 'class', 'struct']):
                return code
        
        return response if response else None