#!/usr/bin/env python3
"""
MoA_verify.py - Enhanced Mixture-of-Agents HDL generation with quality-based caching
Implements quality evaluation and caching mechanism to improve MoA performance with increasing layers
Enhanced with robust code extraction and quality-based layer input selection
"""

import json
import time
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from config import Config
from llm_interface import OllamaInterface
from hdl_tester_enhanced import MultiDatasetHDLTester
from utils import load_designs
from quality_evaluator import HDLQualityEvaluator
from cache_manager import HDLCacheManager, GlobalCacheManager

class EnhancedMoAHDLGenerator:
    def __init__(self, layer_models: List[str], aggregator_model: str, 
                 num_layers: int, dataset: str = "rtllm", temp_mode: str = "low_T", 
                 enable_quality_caching: bool = False):
        """
        Initialize Enhanced MoA HDL Generator with optional quality-based caching
        
        Args:
            layer_models: List of models for each layer
            aggregator_model: Final aggregator model
            num_layers: Number of MoA layers (0 for direct generation)
            dataset: 'rtllm' or 'verilogeval'
            temp_mode: 'low_T' or 'high_T'
            enable_quality_caching: True for ResNet-style quality caching, False for standard MoA
        """
        self.layer_models = layer_models
        self.aggregator_model = aggregator_model
        self.num_layers = num_layers
        self.dataset = dataset
        self.temp_mode = temp_mode
        self.enable_quality_caching = enable_quality_caching
        
        # Quality-based caching parameters
        self.n_select = 3  # Number of top-quality codes to select for next layer input
        
        # Dataset-specific paths
        dataset_dir = Config.VERILOGEVAL_DIR if dataset == "verilogeval" else Config.RTLLM_DIR
        
        # Initialize quality evaluator only if caching is enabled
        if self.enable_quality_caching:
            self.quality_evaluator = HDLQualityEvaluator(dataset_dir, dataset)
        else:
            self.quality_evaluator = None
        
        # MoA-specific LLM parameters
        self.moa_llm_params = {
            "qwen2.5:7b": {"context_length": 65536, "num_predict": 4096, "timeout": 120},
            "qwen2.5-coder:7b": {"context_length": 65536, "num_predict": 4096, "timeout": 120},
            "llama3.1:8b": {"context_length": 32768, "num_predict": 4096, "timeout": 120}
        }
        
        # Initialize LLM interfaces
        self.llm_interfaces = {}
        all_models = [aggregator_model] if num_layers == 0 else list(set(layer_models + [aggregator_model]))
        
        for model in all_models:
            llm = OllamaInterface(model, temp_mode)
            if model in self.moa_llm_params:
                base_params = Config.get_model_params(model, temp_mode)
                base_params.update(self.moa_llm_params[model])
                llm.params = base_params
            self.llm_interfaces[model] = llm
        
        # Set file extension and language
        self.file_extension = Config.get_file_extension(dataset)
        self.language = "SystemVerilog" if dataset == "verilogeval" else "Verilog"
        
        # Create output directories
        self.setup_directories()
        
        # Setup cache directory only if quality caching is enabled
        if self.enable_quality_caching:
            self.setup_cache_directory()
        else:
            self.cache_dir = None
            self.global_cache_manager = None
        
        # Setup system prompts
        self.setup_system_prompts()
    
    def setup_directories(self):
        """Setup output directories with descriptive names"""
        if self.num_layers == 0:
            aggregator_str = self.aggregator_model.replace(":", "-").replace(".", "_")
            folder_name = f"Direct_{self.temp_mode}_{aggregator_str}"
        else:
            models_str = "_".join([m.replace(":", "-").replace(".", "_") for m in self.layer_models])
            aggregator_str = self.aggregator_model.replace(":", "-").replace(".", "_")
            folder_name = f"MoA_{self.temp_mode}_L{self.num_layers}_{models_str}_AGG_{aggregator_str}"
            if self.enable_quality_caching:
                folder_name += "_QualityCache"
        
        if self.dataset == "verilogeval":
            self.verilog_dir = Path("./verilog_eval/MoA") / folder_name
            self.result_dir = Path("./result_eval/MoA") / folder_name
        else:
            self.verilog_dir = Path("./verilog/MoA") / folder_name
            self.result_dir = Path("./result/MoA") / folder_name
        
        self.verilog_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(1, Config.N_SAMPLES + 1):
            (self.verilog_dir / f"t{i}").mkdir(exist_ok=True)
    
    def setup_cache_directory(self):
        """Setup cache directory for intermediate HDL storage"""
        if self.dataset == "verilogeval":
            base_cache_dir = Path("./verilogeval_temp")
        else:
            base_cache_dir = Path("./verilog_temp")
        
        if self.num_layers == 0:
            aggregator_str = self.aggregator_model.replace(":", "-").replace(".", "_")
            folder_name = f"Direct_{self.temp_mode}_{aggregator_str}_QualityCache"
        else:
            models_str = "_".join([m.replace(":", "-").replace(".", "_") for m in self.layer_models])
            aggregator_str = self.aggregator_model.replace(":", "-").replace(".", "_")
            folder_name = f"MoA_{self.temp_mode}_L{self.num_layers}_{models_str}_AGG_{aggregator_str}_QualityCache"
        
        self.cache_dir = base_cache_dir / "MoA" / folder_name
        self.global_cache_manager = GlobalCacheManager(self.cache_dir)
        
        # Create trial directories in cache
        for i in range(1, Config.N_SAMPLES + 1):
            (self.cache_dir / f"t{i}").mkdir(parents=True, exist_ok=True)
    
    def setup_system_prompts(self):
        """Setup unified system prompts based on dataset and task type"""
        if self.dataset == "verilogeval":
            self.system_prompt_generate = (
                "You are a professional SystemVerilog RTL designer. "
                "Generate syntactically correct, synthesizable SystemVerilog code following best practices. "
                "The module MUST be named 'TopModule' exactly. "
                "Output ONLY the SystemVerilog code starting with 'module TopModule' and ending with 'endmodule'. "
                "Do NOT include any markdown formatting, explanations, or comments outside the module."
            )
            self.system_prompt_aggregate = (
                "You are an expert SystemVerilog RTL designer specializing in code synthesis. "
                "Analyze multiple implementations and create a superior solution. "
                "The module MUST be named 'TopModule' exactly. "
                "Output ONLY the SystemVerilog code starting with 'module TopModule' and ending with 'endmodule'. "
                "Do NOT include any markdown formatting or explanations."
            )
        else:  # rtllm
            self.system_prompt_generate = (
                "You are a professional Verilog RTL designer. "
                "Generate syntactically correct, synthesizable Verilog code following best practices. "
                "Output ONLY the Verilog code starting with 'module' and ending with 'endmodule'. "
                "Do NOT include any markdown formatting or explanations."
            )
            self.system_prompt_aggregate = (
                "You are an expert Verilog RTL designer specializing in code synthesis. "
                "Analyze multiple implementations and create a superior solution. "
                "Output ONLY the Verilog code starting with 'module' and ending with 'endmodule'. "
                "Do NOT include any markdown formatting or explanations."
            )
    
    def generate_initial_prompt(self, description: str) -> str:
        """Generate initial prompt with stronger format requirements"""
        if self.dataset == "verilogeval":
            return f"""Generate SystemVerilog code for this specification.

CRITICAL REQUIREMENTS:
1. Output ONLY the module code
2. Module name MUST be exactly 'TopModule'
3. Use modern SystemVerilog port declaration style
4. Start directly with: module TopModule
5. End with: endmodule
6. NO markdown formatting (no ```)
7. NO explanations or comments outside the module

Specification:
{description}

Output the SystemVerilog module now:"""
        else:  # rtllm
            module_name_match = re.search(r'Module name:\s*(\w+)', description)
            module_name = module_name_match.group(1) if module_name_match else "module_name"
            
            return f"""Generate Verilog code for this specification.

CRITICAL REQUIREMENTS:
1. Output ONLY the module code
2. Module name should be: {module_name}
3. Can use traditional or ANSI-style port declarations
4. Start directly with: module {module_name}
5. End with: endmodule
6. NO markdown formatting (no ```)
7. NO explanations outside the module

Specification:
{description}

Output the Verilog module now:"""
    
    def generate_aggregation_prompt(self, hdl_data: Union[List[str], List[Dict]], description: str) -> str:
        """
        Generate unified aggregation prompt for both modes
        
        Args:
            hdl_data: List of HDL strings (standard mode) or List of HDL dicts (quality mode)
            description: Original specification
        """
        responses_text = ""
        
        # Handle different input types
        if hdl_data and isinstance(hdl_data[0], dict):
            # Quality caching mode: hdl_data contains dicts with quality info
            for i, code_info in enumerate(hdl_data, 1):
                code = code_info["code"]
                if len(code) > 5000:
                    code = code[:5000] + "\n... [truncated]"
                responses_text += f"\n[Response {i}]\n{code}\n"
        else:
            # Standard mode: hdl_data contains plain strings
            for i, code in enumerate(hdl_data, 1):
                if len(code) > 5000:
                    code = code[:5000] + "\n... [truncated]"
                responses_text += f"\n[Response {i}]\n{code}\n"
        
        if self.dataset == "verilogeval":
            format_requirements = """
CRITICAL OUTPUT FORMAT:
1. Output ONLY the module code
2. Module name MUST be exactly 'TopModule'
3. Start directly with: module TopModule
4. End with: endmodule
5. NO markdown formatting (no ```)
6. NO explanations or text outside the module
7. Synthesize the best solution from the given responses"""
            language_spec = "SystemVerilog"
        else:
            format_requirements = """
CRITICAL OUTPUT FORMAT:
1. Output ONLY the module code
2. Start directly with: module <name>
3. End with: endmodule
4. NO markdown formatting (no ```)
5. NO explanations or text outside the module
6. Synthesize the best solution from the given responses"""
            language_spec = "Verilog"
        
        return f"""Synthesize multiple {language_spec} implementations into one superior solution.

Original specification:
{description}

Model responses to synthesize:
{responses_text}

Requirements:
- Combine best practices from all responses
- Fix any errors found
- Ensure syntactically correct {language_spec}
- Implement all required functionality

{format_requirements}

Output the {language_spec} module now:"""
    
    def extract_code(self, response: str) -> Optional[str]:
        """Enhanced code extraction with multiple strategies"""
        if not response:
            return None
        
        response = response.strip()
        
        # Remove markdown code blocks
        response = re.sub(r'```(?:systemverilog|verilog|sv|v)?\s*\n?', '', response, flags=re.IGNORECASE)
        response = re.sub(r'```\s*$', '', response, flags=re.MULTILINE)
        
        # Remove common prefixes
        prefixes_to_remove = [
            r"Here's the (?:System)?Verilog (?:code|module|implementation):?\s*",
            r"Here is the (?:System)?Verilog (?:code|module|implementation):?\s*",
            r"The (?:System)?Verilog (?:code|module) is:?\s*",
            r"Output:?\s*", r"Solution:?\s*", r"Implementation:?\s*", r"Code:?\s*",
            r"(?:System)?Verilog:?\s*", r"Generated (?:System)?Verilog module:?\s*", r"Module code:?\s*"
        ]
        
        for prefix in prefixes_to_remove:
            response = re.sub(f'^{prefix}', '', response, flags=re.IGNORECASE | re.MULTILINE)
        
        # Find module boundaries
        module_patterns = [
            r'\b(module\s+[a-zA-Z_][a-zA-Z0-9_]*.*?endmodule\s*;?)\b',
            r'(module\s+\w+[^;]*?[\s\S]*?endmodule\s*;?)',
            r'(module[\s\S]+?endmodule)'
        ]
        
        for pattern in module_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                code = matches[0]
                code = self.clean_extracted_code(code)
                
                if self.dataset == "verilogeval":
                    code = self.fix_module_name(code)
                
                if self.validate_extracted_code(code):
                    return code.strip()
        
        # Fallback strategies
        code = self.extract_code_by_lines(response)
        if code and self.validate_extracted_code(code):
            return code
        
        if 'module' in response.lower():
            code = self.salvage_module_code(response)
            if code and self.validate_extracted_code(code):
                return code
        
        return None
    
    def clean_extracted_code(self, code: str) -> str:
        """Clean up extracted code"""
        if self.dataset == "verilogeval":
            code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        else:
            code = re.sub(r'//\s*(TODO|FIXME|NOTE|WARNING):.*?$', '', code, flags=re.MULTILINE | re.IGNORECASE)
        
        code = re.sub(r'```.*?$', '', code, flags=re.MULTILINE)
        
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
    
    def fix_module_name(self, code: str) -> str:
        """Fix module name for VerilogEval (must be TopModule)"""
        if self.dataset != "verilogeval":
            return code
        
        replacements = [
            (r'module\s+\w+(\s*#\s*\([^)]*\)\s*)?(\s*\([^)]*\))', r'module TopModule\1\2'),
            (r'module\s+\w+(\s*\([^)]*\))', r'module TopModule\1'),
            (r'module\s+\w+\s*;', r'module TopModule;'),
            (r'module\s+\w+\s*$', r'module TopModule')
        ]
        
        for pattern, replacement in replacements:
            code = re.sub(pattern, replacement, code, count=1, flags=re.IGNORECASE | re.MULTILINE)
        
        return code
    
    def extract_code_by_lines(self, response: str) -> Optional[str]:
        """Extract code by processing line by line"""
        lines = response.split('\n')
        module_started = False
        code_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('```'):
                continue
            
            if not module_started and re.match(r'module\s+\w+', stripped, re.IGNORECASE):
                module_started = True
                code_lines.append(line)
                continue
            
            if module_started:
                code_lines.append(line)
                
                if re.match(r'endmodule\s*;?\s*$', stripped, re.IGNORECASE):
                    break
                
                if len(code_lines) > 1000:
                    break
        
        if code_lines:
            code = '\n'.join(code_lines)
            
            if not re.search(r'endmodule\s*;?\s*$', code, re.MULTILINE | re.IGNORECASE):
                code += '\nendmodule'
            
            if self.dataset == "verilogeval":
                code = self.fix_module_name(code)
            
            return code.strip()
        
        return None
    
    def salvage_module_code(self, response: str) -> Optional[str]:
        """Last resort attempt to salvage module code"""
        module_start = re.search(r'module\s+\w+', response, re.IGNORECASE)
        if not module_start:
            return None
        
        code = response[module_start.start():]
        
        endmodule_match = re.search(r'endmodule\s*;?\s*', code, re.IGNORECASE)
        if endmodule_match:
            code = code[:endmodule_match.end()]
        else:
            code = code + '\nendmodule'
        
        code = self.clean_extracted_code(code)
        
        if self.dataset == "verilogeval":
            code = self.fix_module_name(code)
        
        return code
    
    def validate_extracted_code(self, code: str) -> bool:
        """Validate extracted code meets basic requirements"""
        if not code:
            return False
        
        if not re.search(r'module\s+\w+', code, re.IGNORECASE):
            return False
        
        if not re.search(r'endmodule', code, re.IGNORECASE):
            return False
        
        if self.dataset == "verilogeval":
            if not re.search(r'module\s+TopModule', code):
                return False
        else:
            if not re.search(r'module\s+[a-zA-Z_][a-zA-Z0-9_]*', code):
                return False
        
        if '```' in code:
            return False
        
        module_count = len(re.findall(r'module\s+\w+', code, re.IGNORECASE))
        endmodule_count = len(re.findall(r'endmodule', code, re.IGNORECASE))
        
        if module_count != 1 or endmodule_count != 1:
            return False
        
        min_length = 25 if self.dataset == "rtllm" else 30
        if len(code) < min_length:
            return False
        
        return True
    
    def generate_direct_trial(self, description: str, trial_num: int, design_name: str) -> Optional[str]:
        """Generate single trial using direct LLM (for num_layers=0)"""
        print(f"  Trial {trial_num}: Direct", end="", flush=True)
        
        llm = self.llm_interfaces[self.aggregator_model]
        initial_prompt = self.generate_initial_prompt(description)
        
        for attempt in range(2):
            if attempt > 0:
                retry_prompt = initial_prompt + "\n\nREMINDER: Output ONLY the code, no explanations!"
                if self.dataset == "verilogeval":
                    retry_prompt += " Module name MUST be 'TopModule'!"
                response = llm.generate_response(retry_prompt, self.system_prompt_generate)
            else:
                response = llm.generate_response(initial_prompt, self.system_prompt_generate)
            
            if response:
                code = self.extract_code(response)
                if code and self.validate_extracted_code(code):
                    print(" OK")
                    return code
        
        print(" FAIL")
        return None
    
    def generate_moa_trial(self, description: str, trial_num: int, design_name: str) -> Optional[str]:
        """Generate single trial using MoA methodology (unified for both modes)"""
        if self.num_layers == 0:
            return self.generate_direct_trial(description, trial_num, design_name)
        
        print(f"  Trial {trial_num}: ", end="", flush=True)
        
        # Initialize cache manager if quality caching is enabled
        if self.enable_quality_caching:
            cache_manager = self.global_cache_manager.get_design_cache(design_name, trial_num)
        else:
            cache_manager = None
        
        initial_prompt = self.generate_initial_prompt(description)
        layer_outputs = []  # For standard mode
        
        # Process each layer
        for layer_idx in range(self.num_layers):
            print(f"L{layer_idx+1}", end="", flush=True)
            
            if layer_idx == 0:
                # First layer: use initial prompt
                current_layer_outputs = []
                
                for model in self.layer_models:
                    llm = self.llm_interfaces[model]
                    response = llm.generate_response(initial_prompt, self.system_prompt_generate)
                    if response:
                        code = self.extract_code(response)
                        if code and self.validate_extracted_code(code):
                            if self.enable_quality_caching:
                                # Quality caching mode: evaluate and store
                                quality_score = self.quality_evaluator.evaluate_quality(code, design_name)
                                hdl_output = {
                                    "code": code,
                                    "model": model,
                                    "quality_score": quality_score,
                                    "generation_info": {
                                        "layer_idx": layer_idx,
                                        "prompt_type": "initial",
                                        "generation_time": datetime.now().isoformat()
                                    }
                                }
                                current_layer_outputs.append(hdl_output)
                            else:
                                # Standard mode: just collect code
                                current_layer_outputs.append(code)
                        else:
                            # Retry with stronger prompt
                            retry_prompt = initial_prompt + "\n\nREMINDER: Output ONLY the code, no explanations!"
                            response = llm.generate_response(retry_prompt, self.system_prompt_generate)
                            if response:
                                code = self.extract_code(response)
                                if code and self.validate_extracted_code(code):
                                    if self.enable_quality_caching:
                                        quality_score = self.quality_evaluator.evaluate_quality(code, design_name)
                                        hdl_output = {
                                            "code": code,
                                            "model": model,
                                            "quality_score": quality_score,
                                            "generation_info": {
                                                "layer_idx": layer_idx,
                                                "prompt_type": "initial_retry",
                                                "generation_time": datetime.now().isoformat()
                                            }
                                        }
                                        current_layer_outputs.append(hdl_output)
                                    else:
                                        current_layer_outputs.append(code)
                
                if self.enable_quality_caching:
                    # Cache layer outputs and continue with quality mode logic
                    if current_layer_outputs:
                        cache_manager.add_layer_outputs(layer_idx, current_layer_outputs)
                else:
                    # Standard mode: store outputs for next layer
                    layer_outputs = current_layer_outputs
            
            else:
                # Subsequent layers
                if self.enable_quality_caching:
                    # Quality mode: get top quality codes from cache
                    top_quality_codes = cache_manager.get_top_quality_codes(self.n_select, up_to_layer=layer_idx-1)
                    
                    if not top_quality_codes:
                        print(" [no cached input] ", end="")
                        return None
                    
                    input_data = top_quality_codes
                else:
                    # Standard mode: use all outputs from previous layer
                    if not layer_outputs:
                        print(" [no input] ", end="")
                        return None
                    
                    input_data = layer_outputs
                
                current_layer_outputs = []
                agg_prompt = self.generate_aggregation_prompt(input_data, description)
                
                for model in self.layer_models:
                    llm = self.llm_interfaces[model]
                    response = llm.generate_response(agg_prompt, self.system_prompt_aggregate)
                    if response:
                        code = self.extract_code(response)
                        if code and self.validate_extracted_code(code):
                            if self.enable_quality_caching:
                                quality_score = self.quality_evaluator.evaluate_quality(code, design_name)
                                hdl_output = {
                                    "code": code,
                                    "model": model,
                                    "quality_score": quality_score,
                                    "generation_info": {
                                        "layer_idx": layer_idx,
                                        "prompt_type": "aggregation",
                                        "input_codes_quality": [c["quality_score"] for c in input_data] if isinstance(input_data[0], dict) else None,
                                        "generation_time": datetime.now().isoformat()
                                    }
                                }
                                current_layer_outputs.append(hdl_output)
                            else:
                                current_layer_outputs.append(code)
                
                if self.enable_quality_caching:
                    if current_layer_outputs:
                        cache_manager.add_layer_outputs(layer_idx, current_layer_outputs)
                else:
                    layer_outputs = current_layer_outputs
            
            output_count = len(current_layer_outputs)
            print(f"({output_count})", end="", flush=True)
        
        # Final aggregation
        print(" AGG", end="", flush=True)
        
        if self.enable_quality_caching:
            final_input = cache_manager.get_top_quality_codes(self.n_select)
        else:
            final_input = layer_outputs
        
        if final_input:
            llm = self.llm_interfaces[self.aggregator_model]
            
            for attempt in range(2):
                final_prompt = self.generate_aggregation_prompt(final_input, description)
                
                if attempt > 0:
                    final_prompt += "\n\nCRITICAL: Output ONLY the module code! No markdown, no explanations!"
                    if self.dataset == "verilogeval":
                        final_prompt += " Module name MUST be 'TopModule'!"
                
                response = llm.generate_response(final_prompt, self.system_prompt_aggregate)
                if response:
                    final_code = self.extract_code(response)
                    if final_code and self.validate_extracted_code(final_code):
                        print(" OK")
                        return final_code
            
            # Last resort for quality caching mode
            if self.enable_quality_caching and final_input:
                print(" OK(best_cached)")
                return final_input[0]["code"]
        
        print(" FAIL")
        return None
    
    def generate_design_trials(self, design: Dict) -> Dict:
        """Generate all trials for a design"""
        design_name = design["name"]
        desc_file = design["description"]
        
        try:
            with open(desc_file, 'r', encoding='utf-8') as f:
                description = f.read().strip()
        except:
            return {"design": design_name, "error": "Cannot read description", "trials": []}
        
        print(f"Generating {design_name}:")
        
        trials = []
        successful_count = 0
        extraction_failures = 0
        
        for i in range(Config.N_SAMPLES):
            trial_num = i + 1
            code = self.generate_moa_trial(description, trial_num, design_name)
            
            if code:
                trial_file = self.verilog_dir / f"t{trial_num}" / f"{design_name}{self.file_extension}"
                try:
                    with open(trial_file, 'w') as f:
                        f.write(code)
                    trials.append({"trial": trial_num, "file": str(trial_file), "success": True})
                    successful_count += 1
                except Exception as e:
                    trials.append({"trial": trial_num, "error": str(e), "success": False})
            else:
                extraction_failures += 1
                trials.append({"trial": trial_num, "error": "Code extraction failed", "success": False})
            
            time.sleep(0.5)
        
        return {
            "design": design_name,
            "trials": trials,
            "successful_count": successful_count,
            "extraction_failures": extraction_failures,
            "total_trials": Config.N_SAMPLES
        }
    
    def run_generation(self, designs: List[Dict]):
        """Run MoA generation for all designs"""
        if self.num_layers == 0:
            print("Direct LLM HDL Generation (num_layers=0)")
            method = "Direct"
        elif self.enable_quality_caching:
            print("Enhanced MoA HDL Generation with Quality-based Caching")
            method = "Enhanced_MoA"
        else:
            print("Standard MoA HDL Generation")
            method = "Standard_MoA"
        
        print(f"Dataset: {self.dataset} ({self.language})")
        print(f"Temperature: {self.temp_mode}")
        
        if self.num_layers == 0:
            print(f"Model: {self.aggregator_model} (Direct generation)")
        elif self.enable_quality_caching:
            print(f"Layers: {self.num_layers}, Models per layer: {self.layer_models}")
            print(f"Final Aggregator: {self.aggregator_model}")
            print(f"Quality-based selection: Top {self.n_select} codes per layer")
            print(f"Cache directory: {self.cache_dir}")
        else:
            print(f"Layers: {self.num_layers}, Models per layer: {self.layer_models}")
            print(f"Final Aggregator: {self.aggregator_model}")
            print("Mode: Standard MoA (no quality caching)")
        print("=" * 60)
        
        results = []
        total_successful = 0
        total_extraction_failures = 0
        
        for i, design in enumerate(designs, 1):
            print(f"[{i}/{len(designs)}] ", end="")
            result = self.generate_design_trials(design)
            results.append(result)
            total_successful += result.get("successful_count", 0)
            total_extraction_failures += result.get("extraction_failures", 0)
            time.sleep(1)
        
        # Save generation summary
        summary = {
            "method": method,
            "dataset": self.dataset,
            "language": self.language,
            "temp_mode": self.temp_mode,
            "num_layers": self.num_layers,
            "layer_models": self.layer_models if self.num_layers > 0 else [],
            "aggregator_model": self.aggregator_model,
            "quality_based_caching": self.enable_quality_caching,
            "n_select_per_layer": self.n_select if self.enable_quality_caching else None,
            "cache_directory": str(self.cache_dir) if self.enable_quality_caching else None,
            "timestamp": datetime.now().isoformat(),
            "total_designs": len(designs),
            "total_trials": len(designs) * Config.N_SAMPLES,
            "successful_trials": total_successful,
            "extraction_failures": total_extraction_failures,
            "success_rate": f"{total_successful/(len(designs)*Config.N_SAMPLES)*100:.1f}%",
            "extraction_failure_rate": f"{total_extraction_failures/(len(designs)*Config.N_SAMPLES)*100:.1f}%",
            "details": results
        }
        
        with open(self.verilog_dir / "generation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nGeneration complete: {total_successful}/{len(designs)*Config.N_SAMPLES} successful")
        print(f"Extraction failures: {total_extraction_failures}")
        print(f"Output: {self.verilog_dir}")
        
        if self.enable_quality_caching and self.num_layers > 0:
            cache_analysis = self.global_cache_manager.generate_global_analysis()
            with open(self.cache_dir / "cache_analysis.json", 'w') as f:
                json.dump(cache_analysis, f, indent=2)
            print(f"Cache analysis: {self.cache_dir / 'cache_analysis.json'}")
        
        return self.verilog_dir, self.result_dir
    
    def run_testing(self):
        """Run testing on generated files"""
        dataset_dir = Config.VERILOGEVAL_DIR if self.dataset == "verilogeval" else Config.RTLLM_DIR
        
        if self.num_layers == 0:
            model_name = f"Direct_{self.aggregator_model.replace(':', '-')}"
        else:
            model_name = f"{'Enhanced' if self.enable_quality_caching else 'Standard'}_MoA_{self.num_layers}L"
        
        tester = MultiDatasetHDLTester(
            self.verilog_dir, dataset_dir, self.result_dir,
            model_name, self.dataset, self.temp_mode
        )
        tester.run_tests()
        
        return self.result_dir / "results.json"

def main():
    """Main function with command line interface"""
    import sys
    
    # Default configuration
    layer_models = ['qwen2.5:7b', 'qwen2.5-coder:7b', 'qwen2.5-coder:7b']
    # layer_models = ['qwen2.5-coder:7b', 'qwen2.5-coder:7b', 'qwen2.5-coder:7b']
    aggregator_model = 'qwen2.5-coder:7b'
    num_layers =  4
    dataset = 'rtllm'
    temp_mode = 'high_T'
    enable_quality_caching = True
    
    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg.startswith('--layers='):
            num_layers = int(arg.split('=')[1])
        elif arg.startswith('--models='):
            models_str = arg.split('=')[1]
            layer_models = models_str.split(',')
        elif arg.startswith('--aggregator='):
            aggregator_model = arg.split('=')[1]
        elif arg.startswith('--dataset='):
            dataset = arg.split('=')[1]
        elif arg.startswith('--temp='):
            temp_mode = arg.split('=')[1]
        elif arg.startswith('--quality_cache='):
            enable_quality_caching = arg.split('=')[1].lower() in ['true', '1', 'yes', 'on']
        elif arg == '--quality_cache':
            enable_quality_caching = True
    
    # Load designs
    designs = load_designs(dataset)
    if not designs:
        print(f"No {dataset} designs found")
        return
    
    print(f"Loaded {len(designs)} designs")
    
    # Initialize generator
    generator = EnhancedMoAHDLGenerator(
        layer_models=layer_models,
        aggregator_model=aggregator_model,
        num_layers=num_layers,
        dataset=dataset,
        temp_mode=temp_mode,
        enable_quality_caching=enable_quality_caching
    )
    
    # Run generation
    verilog_dir, result_dir = generator.run_generation(designs)
    
    # Run testing
    print("\nStarting testing phase...")
    test_results = generator.run_testing()
    
    print(f"\nResults saved to: {test_results}")

if __name__ == "__main__":
    main()