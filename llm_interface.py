#!/usr/bin/env python3
"""
LLM Interface optimized for Qwen2.5 series with dynamic temperature support
"""

import subprocess
import json
import time
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
                # Determine if this looks like a SystemVerilog request
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
    
    def extract_verilog(self, response: str) -> Optional[str]:
        """Extract Verilog code from response"""
        if not response:
            return None
        
        original_response = response
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Here's the Verilog code:",
            "Here is the Verilog code:",
            "The Verilog code is:",
            "```verilog",
            "```systemverilog",
            "```",
            "Here's the implementation:"
        ]
        
        for prefix in prefixes_to_remove:
            if response.strip().startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Look for code blocks first
        lines = response.split('\n')
        code_lines = []
        in_block = False
        
        # Check for markdown code blocks
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
        
        # If no code blocks, extract module directly
        if not code_lines:
            module_found = False
            for line in lines:
                stripped = line.strip()
                
                # Start from module declaration
                if not module_found and stripped.startswith('module '):
                    module_found = True
                
                if module_found:
                    code_lines.append(line)
                    
                    # Stop at endmodule
                    if stripped in ['endmodule', 'endmodule;'] or stripped.startswith('endmodule'):
                        break
        
        # Validate and return
        if code_lines:
            code = '\n'.join(code_lines)
            
            # Ensure we have both module and endmodule
            if 'module ' in code:
                if 'endmodule' not in code:
                    code += '\nendmodule'
                return code
        
        # Final fallback
        if 'module ' in original_response:
            return original_response
        
        return None
    
    def extract_cpp_code(self, response: str) -> Optional[str]:
        """Extract C++ code from response"""
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
                if any(stripped.startswith(kw) for kw in ['#include', 'void ', 'int ', 'class ', 'struct ', 'bool ', 'uint8_t', 'uint16_t', 'uint32_t']):
                    code_lines = lines[i:]
                    break
        
        return '\n'.join(code_lines) if code_lines else response