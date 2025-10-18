#!/usr/bin/env python3
"""
MoA_HLS.py - Multi-path MoA with Configurable Intermediate Agents
Implements a hybrid approach with configurable paths (direct, C++, Python) per layer.
Uses dual-layer caching: HDL quality evaluation + intermediate code preservation.
NEW: Added early stopping feature to terminate generation when perfect HDL is found
NEW: Added self-refinement feature to iteratively fix HDL errors using iverilog feedback
"""

import json
import time
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from config import Config
from llm_interface import create_llm_interface
from hdl_tester_enhanced import MultiDatasetHDLTester
from utils import load_designs
from quality_evaluator import HDLQualityEvaluator
from cache_manager import HDLCacheManager, GlobalCacheManager


class DualLayerCacheManager(HDLCacheManager):
    """
    Extended cache manager supporting dual-layer storage:
    - HDL layer: for quality evaluation and selection
    - Intermediate code layer: for path-specific references
    """
    
    def __init__(self, cache_dir: Path, design_name: str, trial_num: int):
        super().__init__(cache_dir, design_name, trial_num)
        
        # Add intermediate code storage
        self.cache_data["intermediate_codes"] = {
            "cpp": [],
            "python": []
        }
    
    def add_layer_outputs_with_intermediate(self, layer_idx: int, hdl_outputs: List[Dict]):
        """
        Add HDL outputs with optional intermediate code
        
        Each hdl_output should contain:
        - code: HDL code string
        - model: Model name
        - quality_score: Quality score
        - path: 'direct', 'cpp_chain', or 'python_chain'
        - intermediate: Dict with 'language' and 'code' (optional)
        - original_quality: Original quality before refinement (for intermediate code eval)
        """
        layer_key = str(layer_idx)
        
        if layer_key not in self.cache_data["layer_outputs"]:
            self.cache_data["layer_outputs"][layer_key] = []
        
        for hdl_output in hdl_outputs:
            hdl_entry = {
                "code": hdl_output["code"],
                "model": hdl_output["model"],
                "quality_score": hdl_output["quality_score"],
                "path": hdl_output.get("path", "unknown"),
                "layer_idx": layer_idx,
                "cached_at": datetime.now().isoformat(),
                "generation_info": hdl_output.get("generation_info", {})
            }
            
            # Store intermediate code if present
            if "intermediate" in hdl_output and hdl_output["intermediate"]:
                intermediate = hdl_output["intermediate"]
                hdl_entry["has_intermediate"] = True
                hdl_entry["intermediate_language"] = intermediate["language"]
                
                # Use original_quality for intermediate code evaluation (not refined quality)
                intermediate_quality = hdl_output.get("original_quality", hdl_output["quality_score"])
                
                # Store in intermediate codes section with reference to HDL
                intermediate_entry = {
                    "code": intermediate["code"],
                    "language": intermediate["language"],
                    "hdl_quality": intermediate_quality,  # Use original quality
                    "layer_idx": layer_idx,
                    "hdl_reference_idx": len(self.cache_data["layer_outputs"][layer_key])
                }
                
                self.cache_data["intermediate_codes"][intermediate["language"]].append(
                    intermediate_entry
                )
            else:
                hdl_entry["has_intermediate"] = False
            
            self.cache_data["layer_outputs"][layer_key].append(hdl_entry)
        
        # Update metadata
        self.cache_data["total_layers"] = max(self.cache_data["total_layers"], layer_idx + 1)
        self.cache_data["metadata"]["total_hdl_codes"] = sum(
            len(outputs) for outputs in self.cache_data["layer_outputs"].values()
        )
        
        self._save_cache()
    
    def get_best_intermediate_code(self, language: str, up_to_layer: Optional[int] = None) -> Optional[Dict]:
        """
        Get the highest quality intermediate code for a specific language
        
        Returns: Dict with 'code', 'language', 'hdl_quality' or None
        """
        candidates = []
        
        for entry in self.cache_data["intermediate_codes"][language]:
            if up_to_layer is not None and entry["layer_idx"] > up_to_layer:
                continue
            candidates.append(entry)
        
        if not candidates:
            return None
        
        # Return the one with highest HDL quality
        best = max(candidates, key=lambda x: x["hdl_quality"])
        return {
            "code": best["code"],
            "language": best["language"],
            "hdl_quality": best["hdl_quality"]
        }


class MoAHLSGenerator:
    """
    Multi-path MoA Generator with Configurable Intermediate Agents
    All paths use the same LLM model
    """
    
    def __init__(self, model: str, num_layers: int, dataset: str = "rtllm",
                 temp_mode: str = "low_T", enable_quality_caching: bool = True,
                 n_select: int = 3, path_config: List[str] = None,
                 enable_early_stopping: bool = False, enable_self_refinement: bool = False,
                 max_self_refinement_iterations: int = 3):
        """
        Initialize MoA-HLS Generator
        
        Args:
            model: LLM model name (used for all paths and aggregation)
            num_layers: Number of MoA layers
            dataset: 'rtllm' or 'verilogeval'
            temp_mode: 'low_T' or 'high_T'
            enable_quality_caching: Whether to use quality-based caching
            n_select: Number of top codes to select per layer
            path_config: List of path types for each MoA layer
                        e.g., ['direct', 'cpp', 'python'] or ['cpp', 'cpp', 'cpp']
                        Valid values: 'direct', 'cpp', 'python'
                        Default: ['direct', 'cpp', 'python']
            enable_early_stopping: True to stop generation when perfect HDL (score=1.0) is found
            enable_self_refinement: True to iteratively fix HDL errors using iverilog feedback
            max_self_refinement_iterations: Maximum refinement iterations (default: 3)
        """
        self.model = model
        self.num_layers = num_layers
        self.dataset = dataset
        self.temp_mode = temp_mode
        self.enable_quality_caching = enable_quality_caching
        self.n_select = n_select
        self.enable_early_stopping = enable_early_stopping
        self.enable_self_refinement = enable_self_refinement
        
        # Self-refinement parameters (only effective when quality caching is enabled)
        self.max_self_refinement_iterations = max_self_refinement_iterations
        
        # Disable self-refinement if quality caching is disabled
        if self.enable_self_refinement and not self.enable_quality_caching:
            print("Warning: Self-refinement requires quality caching. Disabling self-refinement.")
            self.enable_self_refinement = False
        
        # Configure paths for MoA layers
        if path_config is None:
            self.path_config = ['direct', 'cpp', 'python']  # Default configuration
        else:
            # Validate path configuration
            valid_paths = {'direct', 'cpp', 'python'}
            for path in path_config:
                if path not in valid_paths:
                    raise ValueError(f"Invalid path type: {path}. Must be one of {valid_paths}")
            self.path_config = path_config
        
        # Dataset-specific paths
        self.dataset_dir = Config.VERILOGEVAL_DIR if dataset == "verilogeval" else Config.RTLLM_DIR
        
        # Initialize quality evaluator if caching enabled
        if self.enable_quality_caching:
            self.quality_evaluator = HDLQualityEvaluator(self.dataset_dir, dataset)
        else:
            self.quality_evaluator = None
        
        # Initialize LLM interface - Modified: Support both Ollama and OpenAI models
        self.llm = create_llm_interface(model, temp_mode)
        
        # Apply extended context for MoA
        moa_params = {
            "context_length": 65536,
            "num_predict": 4096,
            "timeout": 180
        }
        base_params = Config.get_model_params(model, temp_mode)
        base_params.update(moa_params)
        self.llm.params = base_params
        
        # Set file extension and language
        self.file_extension = Config.get_file_extension(dataset)
        self.language = "SystemVerilog" if dataset == "verilogeval" else "Verilog"
        
        # Setup directories
        self.setup_directories()
        
        # Setup cache directory if quality caching enabled
        if self.enable_quality_caching:
            self.setup_cache_directory()
        else:
            self.cache_dir = None
            self.global_cache_manager = None
        
        # Setup system prompts
        self.setup_system_prompts()
    
    def setup_directories(self):
        """Setup output directories with descriptive folder names"""
        model_str = self.model.replace(":", "-").replace(".", "_")
        
        # Generate path configuration string for folder name
        path_str = "_".join(self.path_config)
        
        folder_parts = [
            "MoA_HLS",
            self.temp_mode,
            f"L{self.num_layers}",
            model_str,
            f"paths_{path_str}"
        ]
        
        if self.enable_quality_caching:
            folder_parts.append(f"QCache_N{self.n_select}")
        
        if self.enable_early_stopping:
            folder_parts.append("EarlyStop")
        
        if self.enable_self_refinement:
            folder_parts.append(f"SelfRef{self.max_self_refinement_iterations}")
        
        folder_name = "_".join(folder_parts)
        
        if self.dataset == "verilogeval":
            self.verilog_dir = Path("./verilog_eval/MoA_HLS") / folder_name
            self.result_dir = Path("./result_eval/MoA_HLS") / folder_name
        else:
            self.verilog_dir = Path("./verilog/MoA_HLS") / folder_name
            self.result_dir = Path("./result/MoA_HLS") / folder_name
        
        self.verilog_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # Create trial directories
        for i in range(1, Config.N_SAMPLES + 1):
            (self.verilog_dir / f"t{i}").mkdir(exist_ok=True)
    
    def setup_cache_directory(self):
        """Setup cache directory for intermediate storage"""
        if self.dataset == "verilogeval":
            base_cache_dir = Path("./verilogeval_temp")
        else:
            base_cache_dir = Path("./verilog_temp")
        
        model_str = self.model.replace(":", "-").replace(".", "_")
        path_str = "_".join(self.path_config)
        folder_name = f"MoA_HLS_{self.temp_mode}_L{self.num_layers}_{model_str}_paths_{path_str}_QCache_N{self.n_select}"
        
        if self.enable_early_stopping:
            folder_name += "_EarlyStop"
        
        if self.enable_self_refinement:
            folder_name += f"_SelfRef{self.max_self_refinement_iterations}"
        
        self.cache_dir = base_cache_dir / "MoA_HLS" / folder_name
        self.global_cache_manager = GlobalCacheManager(self.cache_dir)
        
        # Create trial directories in cache
        for i in range(1, Config.N_SAMPLES + 1):
            (self.cache_dir / f"t{i}").mkdir(parents=True, exist_ok=True)
    
    def setup_system_prompts(self):
        """Setup system prompts for different generation tasks"""
        if self.dataset == "verilogeval":
            self.system_prompt_direct = (
                "You are a professional SystemVerilog RTL designer. "
                "Generate syntactically correct, synthesizable SystemVerilog code. "
                "The module MUST be named 'TopModule' exactly. "
                "Output ONLY the SystemVerilog code starting with 'module TopModule' and ending with 'endmodule'. "
                "Do NOT include markdown formatting or explanations."
            )
            
            self.system_prompt_intermediate = (
                "You are an expert programmer. "
                "Write clear, concise code demonstrating the algorithm. "
                "Focus on showing the logical flow and operations."
            )
            
            self.system_prompt_translate = (
                "You are an expert SystemVerilog RTL designer. "
                "Translate the reference implementation to synthesizable SystemVerilog. "
                "The module MUST be named 'TopModule' exactly. "
                "Output ONLY the SystemVerilog code. No markdown or explanations."
            )
            
            self.system_prompt_refinement = (
                "You are an expert SystemVerilog debugger. "
                "Your task is to analyze compilation/simulation errors and fix them precisely. "
                "Focus on the specific error messages and fix ONLY what's broken. "
                "Output clean, corrected code without explanations."
            )
        else:  # rtllm
            self.system_prompt_direct = (
                "You are a professional Verilog RTL designer. "
                "Generate syntactically correct, synthesizable Verilog code. "
                "Output ONLY the Verilog code starting with 'module' and ending with 'endmodule'. "
                "Do NOT include markdown formatting or explanations."
            )
            
            self.system_prompt_intermediate = (
                "You are an expert programmer. "
                "Write clear, concise code demonstrating the algorithm. "
                "Focus on showing the logical flow and operations."
            )
            
            self.system_prompt_translate = (
                "You are an expert Verilog RTL designer. "
                "Translate the reference implementation to synthesizable Verilog. "
                "Output ONLY the Verilog code. No markdown or explanations."
            )
            
            self.system_prompt_refinement = (
                "You are an expert Verilog debugger. "
                "Your task is to analyze compilation/simulation errors and fix them precisely. "
                "Focus on the specific error messages and fix ONLY what's broken. "
                "Output clean, corrected code without explanations."
            )
    
    def generate_refinement_prompt(self, original_code: str, error_type: str,
                                   error_message: str, description: str, 
                                   iteration: int, intermediate_code: Optional[str] = None,
                                   intermediate_language: Optional[str] = None) -> str:
        """
        Generate refinement prompt for HDL code
        
        Args:
            original_code: The HDL code that failed
            error_type: 'syntax_error', 'compilation_error', or 'simulation_fail'
            error_message: Error message from iverilog/vvp
            description: Original design specification
            iteration: Current refinement iteration (1-based)
            intermediate_code: Optional intermediate code (for cpp/python paths)
            intermediate_language: Language of intermediate code ('cpp' or 'python')
        """
        # Base context
        if intermediate_code and intermediate_language:
            # For HLS paths: include intermediate code reference
            base_context = f"""You are fixing {self.language} code that was translated from {intermediate_language.upper()}.

Original specification:
{description}

{intermediate_language.upper()} reference implementation:
{intermediate_code}

Current {self.language} code (Refinement attempt {iteration}/{self.max_self_refinement_iterations}):
{original_code}

Error encountered:
{error_message}

The {intermediate_language.upper()} reference is correct - focus on fixing the {self.language} translation.
"""
        else:
            # For direct path: standard refinement
            base_context = f"""You are debugging {self.language} code that failed testing.

Original specification:
{description}

Current code (Refinement attempt {iteration}/{self.max_self_refinement_iterations}):
{original_code}

Error encountered:
{error_message}
"""
        
        # Error-specific guidance
        if error_type == "syntax_error":
            specific_guidance = """
SYNTAX ERROR DETECTED. Common issues:
1. Variable/genvar redeclaration - check all loop variables and ensure unique naming
2. Part select with non-constant expressions - use parameters or constants
3. Missing/mismatched module declarations
4. Incorrect port declarations or signal types

Fix the syntax errors while preserving the original logic.
"""
        
        elif error_type == "compilation_error":
            if "Unknown module type" in error_message:
                specific_guidance = """
MISSING/UNKNOWN MODULE ERROR. Possible causes:
1. Module name mismatch with testbench expectations
2. Missing submodule definitions - you must implement ALL modules in a single file
3. Hierarchical design split incorrectly

Solution:
- Implement all required submodules in the SAME file
- Ensure the top-level module name matches the testbench requirement
- Use inline logic instead of module instantiation if appropriate
"""
            else:
                specific_guidance = """
COMPILATION ERROR. Check:
1. Module name matches testbench expectations
2. Port declarations are correct
3. All referenced signals are declared
4. No circular dependencies
"""
        
        elif error_type == "simulation_fail":
            specific_guidance = """
SIMULATION FAILURE. The code compiles but produces incorrect results.
Possible issues:
1. Logic errors in state machines or combinational logic
2. Incorrect edge sensitivity (posedge/negedge)
3. Race conditions or initialization issues
4. Incorrect bit widths or signal ranges

Review and fix the functional logic while maintaining correct syntax.
"""
        
        else:
            specific_guidance = """
TESTING FAILED. Review the error message carefully and fix the issue.
"""
        
        # Add HLS-specific guidance if intermediate code is present
        if intermediate_code and intermediate_language:
            specific_guidance += f"""
Common issues when translating from {intermediate_language.upper()}:
1. Loop constructs (for/while) → always blocks with proper sensitivity
2. Arrays/pointers → wire/reg arrays with correct indexing
3. Functions → modules or combinational logic
4. Sequential operations → state machines or pipelined logic
"""
        
        # Output requirements
        if self.dataset == "verilogeval":
            output_requirements = """
CRITICAL OUTPUT REQUIREMENTS:
1. Module name MUST be exactly 'TopModule'
2. Output ONLY the complete, corrected SystemVerilog code
3. Start with: module TopModule
4. End with: endmodule
5. NO markdown formatting (no ```)
6. NO explanations - only code
7. Include ALL necessary submodules in the SAME file if needed
"""
        else:
            module_name_match = re.search(r'Module name:\s*(\w+)', description)
            module_name = module_name_match.group(1) if module_name_match else "module_name"
            
            output_requirements = f"""
CRITICAL OUTPUT REQUIREMENTS:
1. Module name should be: {module_name}
2. Output ONLY the complete, corrected Verilog code
3. Start with: module {module_name}
4. End with: endmodule
5. NO markdown formatting (no ```)
6. NO explanations - only code
7. Include ALL necessary submodules in the SAME file if needed
"""
        
        # Iteration-specific encouragement
        if iteration == 1:
            iteration_note = "This is your first attempt to fix the error. Focus on the specific error message."
        elif iteration == 2:
            iteration_note = "Previous fix attempt failed. Try a different approach - the issue might be more fundamental."
        else:
            iteration_note = "Multiple fix attempts have failed. Consider simplifying the design or using a completely different implementation approach."
        
        return f"""{base_context}

{specific_guidance}

{iteration_note}

{output_requirements}

Output the corrected {self.language} code now:"""
    
    def refine_hdl_code(self, original_code: str, design_name: str, description: str,
                       intermediate_code: Optional[str] = None,
                       intermediate_language: Optional[str] = None) -> Tuple[str, float, int, float]:
        """
        Iteratively refine HDL code using iverilog feedback
        
        Args:
            original_code: Initial HDL code to refine
            design_name: Name of the design
            description: Original design specification
            intermediate_code: Optional intermediate code (for cpp/python paths)
            intermediate_language: Language of intermediate code ('cpp' or 'python')
        
        Returns:
            Tuple of (best_code, final_quality, iterations_performed, original_quality)
        """
        if not self.enable_self_refinement or not self.enable_quality_caching:
            # Self-refinement disabled
            quality = self.quality_evaluator.evaluate_quality(original_code, design_name)
            return original_code, quality, 0, quality
        
        # Evaluate original code with details
        quality, error_details = self.quality_evaluator.evaluate_quality_with_details(
            original_code, design_name
        )
        original_quality = quality  # Store original quality for intermediate code eval
        
        # If already perfect, return immediately
        if error_details["passed"]:
            return original_code, quality, 0, original_quality
        
        # Start refinement process
        best_code = original_code
        best_quality = quality
        current_code = original_code
        
        for iteration in range(1, self.max_self_refinement_iterations + 1):
            # Generate refinement prompt
            refinement_prompt = self.generate_refinement_prompt(
                current_code,
                error_details["error_type"],
                error_details["error_message"],
                description,
                iteration,
                intermediate_code,
                intermediate_language
            )
            
            # Get refined code from LLM
            response = self.llm.generate_response(refinement_prompt, self.system_prompt_refinement)
            
            if not response:
                break  # LLM failed to respond
            
            # Extract code
            refined_code = self.extract_code(response)
            
            if not refined_code or not self.validate_hdl_code(refined_code):
                break  # Failed to extract valid code
            
            # Evaluate refined code
            refined_quality, refined_error_details = self.quality_evaluator.evaluate_quality_with_details(
                refined_code, design_name
            )
            
            # Update best code if improved
            if refined_quality > best_quality:
                best_code = refined_code
                best_quality = refined_quality
            
            # Check if perfect
            if refined_error_details["passed"]:
                return refined_code, refined_quality, iteration, original_quality
            
            # Prepare for next iteration
            current_code = refined_code
            error_details = refined_error_details
        
        return best_code, best_quality, self.max_self_refinement_iterations if iteration == self.max_self_refinement_iterations else iteration, original_quality
    
    def extract_code(self, response: str) -> Optional[str]:
        """Extract HDL code from LLM response - reuse from OllamaInterface"""
        return self.llm.extract_verilog(response, self.dataset)
    
    def validate_hdl_code(self, code: str) -> bool:
        """Validate extracted HDL code"""
        if not code:
            return False
        
        if not re.search(r'module\s+\w+', code, re.IGNORECASE):
            return False
        
        if not re.search(r'endmodule', code, re.IGNORECASE):
            return False
        
        if self.dataset == "verilogeval":
            if not re.search(r'module\s+TopModule', code):
                return False
        
        if '```' in code:
            return False
        
        module_count = len(re.findall(r'module\s+\w+', code, re.IGNORECASE))
        endmodule_count = len(re.findall(r'endmodule', code, re.IGNORECASE))
        
        if module_count != 1 or endmodule_count != 1:
            return False
        
        return True
    
    def extract_cpp_code(self, response: str) -> Optional[str]:
        """Extract C++ code from LLM response"""
        if not response:
            return None
        
        # Remove markdown
        response = re.sub(r'```(?:cpp|c\+\+|c)?\s*\n?', '', response, flags=re.IGNORECASE)
        response = re.sub(r'```\s*$', '', response, flags=re.MULTILINE)
        
        # Look for includes, functions, or main
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            if re.match(r'^(#include|void |int |class |struct |using |namespace )', stripped):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else response
    
    def extract_python_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response"""
        if not response:
            return None
        
        # Remove markdown
        response = re.sub(r'```python\s*\n?', '', response, flags=re.IGNORECASE)
        response = re.sub(r'```\s*$', '', response, flags=re.MULTILINE)
        
        # Look for function or class definitions
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            if re.match(r'^(def |class |import |from )', stripped):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else response
    
    def generate_initial_prompt(self, description: str) -> str:
        """Generate initial direct generation prompt"""
        if self.dataset == "verilogeval":
            return f"""Generate SystemVerilog code for this specification.

CRITICAL REQUIREMENTS:
1. Module name MUST be exactly 'TopModule'
2. Output ONLY the module code
3. No markdown formatting
4. No explanations

Specification:
{description}

Output the SystemVerilog module:"""
        else:
            module_name_match = re.search(r'Module name:\s*(\w+)', description)
            module_name = module_name_match.group(1) if module_name_match else "module_name"
            
            return f"""Generate Verilog code for this specification.

CRITICAL REQUIREMENTS:
1. Module name should be: {module_name}
2. Output ONLY the module code
3. No markdown formatting
4. No explanations

Specification:
{description}

Output the Verilog module:"""
    
    def generate_aggregation_prompt(self, previous_hdl: List[Dict], description: str,
                                    intermediate_code: Optional[Dict] = None) -> str:
        """
        Generate aggregation prompt with optional intermediate code reference
        
        Args:
            previous_hdl: List of HDL code entries with quality scores
            description: Original specification
            intermediate_code: Optional dict with best intermediate code
                            {'code': str, 'language': str, 'hdl_quality': float}
        
        Returns:
            Formatted prompt for aggregation
        """
        
        # Format previous HDL implementations
        hdl_text = ""
        for i, entry in enumerate(previous_hdl[:3], 1):
            code = entry["code"]
            quality = entry.get("quality_score", 0)
            path = entry.get("path", "unknown")
            
            # Truncate long code
            if len(code) > 1500:
                code = code[:1500] + "\n... [truncated for length]"
            
            hdl_text += f"\n[Implementation {i}] (quality: {quality:.2f}, path: {path})\n{code}\n"
        
        # Add intermediate code reference if available
        intermediate_text = ""
        if intermediate_code:
            lang = intermediate_code["language"].upper()
            code = intermediate_code["code"]
            quality = intermediate_code["hdl_quality"]
            
            # Truncate long code
            if len(code) > 1000:
                code = code[:1000] + "\n... [truncated for length]"
            
            intermediate_text = f"""
Additional reference - {lang} implementation (HDL quality: {quality:.2f}):
{code}
"""
    
        # Generate dataset-specific prompt
        if self.dataset == "verilogeval":
            return f"""Synthesize multiple SystemVerilog implementations into one superior solution.

Original specification:
{description}

Previous implementations to synthesize:
{hdl_text}
{intermediate_text}

Requirements:
- Combine the best practices from all implementations
- Fix any errors or suboptimal designs found
- Ensure syntactically correct and synthesizable SystemVerilog
- Implement complete functionality as specified

CRITICAL OUTPUT FORMAT:
1. Module name MUST be exactly 'TopModule'
2. Output ONLY the module code
3. Start with: module TopModule
4. End with: endmodule
5. NO markdown formatting (no ```)
6. NO explanations or text outside the module

Output the synthesized SystemVerilog module:"""
    
        else:  # rtllm
            module_name_match = re.search(r'Module name:\s*(\w+)', description)
            module_name = module_name_match.group(1) if module_name_match else "module_name"
            
            return f"""Synthesize multiple Verilog implementations into one superior solution.

Original specification:
{description}

Previous implementations to synthesize:
{hdl_text}
{intermediate_text}

Requirements:
- Combine the best practices from all implementations
- Fix any errors or suboptimal designs found
- Ensure syntactically correct and synthesizable Verilog
- Implement complete functionality as specified

CRITICAL OUTPUT FORMAT:
1. Module name should be: {module_name}
2. Output ONLY the module code
3. Start with: module {module_name}
4. End with: endmodule
5. NO markdown formatting (no ```)
6. NO explanations or text outside the module

Output the synthesized Verilog module:"""
    
    def generate_path_direct(self, description: str, previous_hdl: List[Dict] = None) -> Optional[str]:
        """Direct path: generate HDL directly"""
        
        if previous_hdl:
            # Aggregation mode
            prompt = self.generate_aggregation_prompt(previous_hdl, description)
        else:
            # Initial generation
            prompt = self.generate_initial_prompt(description)
        
        response = self.llm.generate_response(prompt, self.system_prompt_direct)
        
        if response:
            return self.extract_code(response)
        
        return None
    
    def generate_path_cpp_chain(self, description: str, 
                                previous_hdl: List[Dict] = None,
                                reference_cpp: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        C++ chain path: Spec → C++ → HDL
        Returns: (hdl_code, cpp_code)
        """
        
        # Step 1: Generate C++ reference
        if reference_cpp:
            # Use provided reference C++ code
            cpp_code = reference_cpp
        else:
            # Generate new C++ code
            if previous_hdl:
                hdl_examples = "\n\n".join([
                    f"Previous implementation {i+1}:\n{entry['code'][:800]}"
                    for i, entry in enumerate(previous_hdl[:2])
                ])
                cpp_prompt = f"""Write C++ code demonstrating the functional logic.

Specification:
{description}

Previous HDL implementations:
{hdl_examples}

Write simple C++ code showing the algorithm:"""
            else:
                cpp_prompt = f"""Write C++ code demonstrating the functional logic.

Specification:
{description}

Write simple C++ code showing the algorithm:"""
            
            cpp_response = self.llm.generate_response(cpp_prompt, self.system_prompt_intermediate)
            
            if not cpp_response:
                return None, None
            
            cpp_code = self.extract_cpp_code(cpp_response)
            if not cpp_code:
                return None, None
        
        # Step 2: Translate C++ to HDL
        hdl_prompt = f"""Translate this C++ reference to {self.language}.

Original specification:
{description}

C++ reference code:
{cpp_code}

Generate the {self.language} module implementing this logic:"""
        
        hdl_response = self.llm.generate_response(hdl_prompt, self.system_prompt_translate)
        
        if hdl_response:
            hdl_code = self.extract_code(hdl_response)
            return hdl_code, cpp_code
        
        return None, cpp_code
    
    def generate_path_python_chain(self, description: str,
                                   previous_hdl: List[Dict] = None,
                                   reference_python: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Python chain path: Spec → Python → HDL
        Returns: (hdl_code, python_code)
        """
        
        # Step 1: Generate Python reference
        if reference_python:
            python_code = reference_python
        else:
            if previous_hdl:
                hdl_examples = "\n\n".join([
                    f"Previous implementation {i+1}:\n{entry['code'][:800]}"
                    for i, entry in enumerate(previous_hdl[:2])
                ])
                py_prompt = f"""Write Python code demonstrating the functional logic.

Specification:
{description}

Previous HDL implementations:
{hdl_examples}

Write simple Python code showing the algorithm:"""
            else:
                py_prompt = f"""Write Python code demonstrating the functional logic.

Specification:
{description}

Write simple Python code showing the algorithm:"""
            
            py_response = self.llm.generate_response(py_prompt, self.system_prompt_intermediate)
            
            if not py_response:
                return None, None
            
            python_code = self.extract_python_code(py_response)
            if not python_code:
                return None, None
        
        # Step 2: Translate Python to HDL
        hdl_prompt = f"""Translate this Python reference to {self.language}.

Original specification:
{description}

Python reference code:
{python_code}

Generate the {self.language} module implementing this logic:"""
        
        hdl_response = self.llm.generate_response(hdl_prompt, self.system_prompt_translate)
        
        if hdl_response:
            hdl_code = self.extract_code(hdl_response)
            return hdl_code, python_code
        
        return None, python_code
    
    def generate_single_path(self, path_type: str, description: str, design_name: str,
                            previous_hdl: List[Dict] = None,
                            reference_cpp: Optional[str] = None,
                            reference_python: Optional[str] = None) -> Optional[Dict]:
        """
        Generate HDL via a single path with optional self-refinement
        
        Args:
            path_type: 'direct', 'cpp', or 'python'
            description: Design specification
            design_name: Name of the design
            previous_hdl: Previous layer HDL codes
            reference_cpp: Reference C++ code (if available)
            reference_python: Reference Python code (if available)
        
        Returns:
            Dict with HDL code, quality score, path type, optional intermediate code, and original_quality
        """
        hdl_code = None
        intermediate_code = None
        intermediate_language = None
        
        if path_type == 'direct':
            print("D", end="", flush=True)
            hdl_code = self.generate_path_direct(description, previous_hdl)
            
        elif path_type == 'cpp':
            print("C", end="", flush=True)
            hdl_code, intermediate_code = self.generate_path_cpp_chain(
                description, previous_hdl, reference_cpp
            )
            intermediate_language = "cpp"
            
        elif path_type == 'python':
            print("P", end="", flush=True)
            hdl_code, intermediate_code = self.generate_path_python_chain(
                description, previous_hdl, reference_python
            )
            intermediate_language = "python"
        
        else:
            raise ValueError(f"Unknown path type: {path_type}")
        
        # Validate and evaluate HDL code
        if hdl_code and self.validate_hdl_code(hdl_code):
            # Apply self-refinement if enabled
            if self.enable_self_refinement and self.enable_quality_caching:
                hdl_code, quality, refine_iters, original_quality = self.refine_hdl_code(
                    hdl_code, design_name, description, intermediate_code, intermediate_language
                )
            else:
                quality = 0.0
                original_quality = 0.0
                if self.enable_quality_caching:
                    quality = self.quality_evaluator.evaluate_quality(hdl_code, design_name)
                    original_quality = quality
            
            result = {
                "code": hdl_code,
                "quality_score": quality,
                "original_quality": original_quality,  # For intermediate code evaluation
                "path": f"{path_type}_chain" if path_type != 'direct' else 'direct',
                "model": self.model,
                "intermediate": None
            }
            
            # Add intermediate code if available
            if intermediate_code and intermediate_language:
                result["intermediate"] = {
                    "language": intermediate_language,
                    "code": intermediate_code
                }
            
            return result
        
        return None
    
    def generate_multipath_layer(self, description: str, layer_idx: int,
                                 design_name: str, cache_manager: Optional[DualLayerCacheManager] = None,
                                 previous_codes: List[Dict] = None) -> List[Dict]:
        """
        Generate HDL via configured paths using same LLM
        
        Returns: List of dicts with HDL code, quality score, path, and optional intermediate code
        """
        layer_outputs = []
        
        # Get optional intermediate code references from cache
        reference_cpp = None
        reference_python = None
        
        if cache_manager and previous_codes and layer_idx > 0:
            # Get best intermediate codes from previous layers
            best_cpp = cache_manager.get_best_intermediate_code("cpp", up_to_layer=layer_idx-1)
            best_python = cache_manager.get_best_intermediate_code("python", up_to_layer=layer_idx-1)
            
            if best_cpp:
                reference_cpp = best_cpp["code"]
            if best_python:
                reference_python = best_python["code"]
        
        # Generate HDL for each configured path
        for path_type in self.path_config:
            result = self.generate_single_path(
                path_type=path_type,
                description=description,
                design_name=design_name,
                previous_hdl=previous_codes,
                reference_cpp=reference_cpp,
                reference_python=reference_python
            )
            
            if result:
                layer_outputs.append(result)
        
        return layer_outputs
    
    def generate_trial(self, description: str, trial_num: int, design_name: str) -> Optional[str]:
        """
        Generate single trial with multi-path MoA and final aggregation
        Follows original MoA architecture: layers → top-n selection → final aggregation
        NEW: Supports early stopping when perfect HDL is found
        NEW: Supports self-refinement for intermediate and final HDL
        """
        print(f"  Trial {trial_num}: ", end="", flush=True)
        
        cache_manager = None
        if self.enable_quality_caching:
            cache_manager = DualLayerCacheManager(self.cache_dir, design_name, trial_num)
        
        current_layer_outputs = []
        
        # Early stopping tracking
        perfect_code_found = None
        early_stop_layer = None
        
        # Process each layer
        for layer_idx in range(self.num_layers):
            print(f"L{layer_idx+1}[", end="", flush=True)
            
            if layer_idx == 0:
                previous_codes = None
            else:
                if self.enable_quality_caching:
                    previous_codes = cache_manager.get_top_quality_codes(
                        self.n_select, up_to_layer=layer_idx-1
                    )
                else:
                    if current_layer_outputs:
                        sorted_codes = sorted(
                            current_layer_outputs,
                            key=lambda x: x.get("quality_score", 0),
                            reverse=True
                        )
                        previous_codes = sorted_codes[:self.n_select]
                    else:
                        previous_codes = None
            
            current_layer_outputs = self.generate_multipath_layer(
                description, layer_idx, design_name, cache_manager, previous_codes
            )
            
            print(f"]({len(current_layer_outputs)})", end="", flush=True)
            
            if self.enable_quality_caching and current_layer_outputs:
                cache_manager.add_layer_outputs_with_intermediate(layer_idx, current_layer_outputs)
                
                # Check for perfect code (early stopping)
                if self.enable_early_stopping and perfect_code_found is None:
                    for output in current_layer_outputs:
                        if output["quality_score"] == 1.0:
                            perfect_code_found = output["code"]
                            early_stop_layer = layer_idx
                            print(f"[PERFECT@L{layer_idx+1}]", end="", flush=True)
                            break
                    
                    # Early stopping - terminate if perfect code found
                    if perfect_code_found is not None:
                        print(" EARLY_STOP")
                        return perfect_code_found
        
        # Final aggregation phase
        print(" AGG", end="", flush=True)
        
        if self.enable_quality_caching and cache_manager:
            # Get top-n codes for final aggregation
            final_input = cache_manager.get_top_quality_codes(self.n_select)
            
            if final_input:
                # Generate final aggregation prompt (no intermediate code)
                final_prompt = self.generate_aggregation_prompt(
                    final_input, description, intermediate_code=None
                )
                
                # LLM performs final aggregation
                for attempt in range(2):
                    if attempt > 0:
                        final_prompt += "\n\nCRITICAL: Output ONLY the module code! No markdown, no explanations!"
                        if self.dataset == "verilogeval":
                            final_prompt += " Module name MUST be 'TopModule'!"
                    
                    response = self.llm.generate_response(final_prompt, self.system_prompt_direct)
                    
                    if response:
                        final_code = self.extract_code(response)
                        if final_code and self.validate_hdl_code(final_code):
                            # Apply self-refinement to final code
                            if self.enable_self_refinement:
                                final_code, final_quality, refine_iters, _ = self.refine_hdl_code(
                                    final_code, design_name, description
                                )
                                if refine_iters > 0:
                                    print(f"[R{refine_iters}]", end="", flush=True)
                            print(" OK")
                            return final_code
                
                # Fallback
                print(" OK(best)")
                return final_input[0]["code"]
        
        elif current_layer_outputs:
            sorted_outputs = sorted(
                current_layer_outputs,
                key=lambda x: x.get("quality_score", 0),
                reverse=True
            )
            top_outputs = sorted_outputs[:min(self.n_select, len(sorted_outputs))]
            
            if len(top_outputs) > 1:
                # No intermediate code in final aggregation
                final_prompt = self.generate_aggregation_prompt(
                    top_outputs, description, intermediate_code=None
                )
                
                for attempt in range(2):
                    if attempt > 0:
                        final_prompt += "\n\nCRITICAL: Output ONLY the module code!"
                    
                    response = self.llm.generate_response(final_prompt, self.system_prompt_direct)
                    
                    if response:
                        final_code = self.extract_code(response)
                        if final_code and self.validate_hdl_code(final_code):
                            print(" OK")
                            return final_code
            
            if top_outputs:
                print(" OK(best)")
                return top_outputs[0]["code"]
        
        print(" FAIL")
        return None
    
    def generate_design_trials(self, design: Dict) -> Dict:
        """Generate all trials for a design"""
        design_name = design["name"]
        desc_file = design["description"]
        
        try:
            with open(desc_file, 'r', encoding='utf-8') as f:
                description = f.read().strip()
        except Exception as e:
            return {
                "design": design_name,
                "error": f"Cannot read description: {e}",
                "trials": [],
                "successful_count": 0
            }
        
        print(f"Generating {design_name}:")
        
        trials = []
        successful_count = 0
        
        for i in range(Config.N_SAMPLES):
            trial_num = i + 1
            code = self.generate_trial(description, trial_num, design_name)
            
            if code:
                trial_file = self.verilog_dir / f"t{trial_num}" / f"{design_name}{self.file_extension}"
                try:
                    with open(trial_file, 'w', encoding='utf-8') as f:
                        f.write(code)
                    trials.append({
                        "trial": trial_num,
                        "file": str(trial_file),
                        "success": True
                    })
                    successful_count += 1
                except Exception as e:
                    trials.append({
                        "trial": trial_num,
                        "error": f"Save failed: {e}",
                        "success": False
                    })
            else:
                trials.append({
                    "trial": trial_num,
                    "error": "Generation failed",
                    "success": False
                })
            
            time.sleep(0.5)
        
        return {
            "design": design_name,
            "trials": trials,
            "successful_count": successful_count,
            "total_trials": Config.N_SAMPLES
        }
    
    def run_generation(self, designs: List[Dict]):
        """Run MoA-HLS generation for all designs"""
        print("MoA-HLS: Multi-path HDL Generation with Configurable Agents")
        print(f"Dataset: {self.dataset} ({self.language})")
        print(f"Model: {self.model}")
        print(f"Temperature: {self.temp_mode}")
        print(f"Layers: {self.num_layers}")
        print(f"Path Configuration: {self.path_config} ({len(self.path_config)} paths per layer)")
        
        # Print early stopping status
        if self.enable_early_stopping:
            print("✓ Early Stopping: ENABLED (will stop when perfect HDL found)")
        else:
            print("✗ Early Stopping: DISABLED")
        
        # Print self-refinement status
        if self.enable_self_refinement:
            print(f"✓ Self-Refinement: ENABLED (max {self.max_self_refinement_iterations} iterations)")
        else:
            print("✗ Self-Refinement: DISABLED")
        
        if self.enable_quality_caching:
            print(f"Quality caching: Enabled (select top-{self.n_select} per layer)")
            print(f"Cache directory: {self.cache_dir}")
        else:
            print("Quality caching: Disabled")
        
        print("=" * 70)
        
        results = []
        total_successful = 0
        
        for i, design in enumerate(designs, 1):
            print(f"[{i}/{len(designs)}] ", end="")
            result = self.generate_design_trials(design)
            results.append(result)
            total_successful += result.get("successful_count", 0)
            time.sleep(1)
        
        # Save generation summary
        summary = {
            "method": "MoA_HLS",
            "model": self.model,
            "dataset": self.dataset,
            "language": self.language,
            "temp_mode": self.temp_mode,
            "num_layers": self.num_layers,
            "path_config": self.path_config,
            "num_paths_per_layer": len(self.path_config),
            "quality_caching": self.enable_quality_caching,
            "early_stopping_enabled": self.enable_early_stopping,
            "self_refinement_enabled": self.enable_self_refinement,
            "max_self_refinement_iterations": self.max_self_refinement_iterations if self.enable_self_refinement else None,
            "n_select": self.n_select if self.enable_quality_caching else None,
            "cache_directory": str(self.cache_dir) if self.enable_quality_caching else None,
            "timestamp": datetime.now().isoformat(),
            "total_designs": len(designs),
            "total_trials": len(designs) * Config.N_SAMPLES,
            "successful_trials": total_successful,
            "success_rate": f"{total_successful/(len(designs)*Config.N_SAMPLES)*100:.1f}%",
            "details": results
        }
        
        with open(self.verilog_dir / "generation_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nGeneration complete: {total_successful}/{len(designs)*Config.N_SAMPLES} successful")
        print(f"Output: {self.verilog_dir}")
        
        if self.enable_quality_caching:
            # Generate cache analysis
            cache_analysis = self.global_cache_manager.generate_global_analysis()
            with open(self.cache_dir / "cache_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(cache_analysis, f, indent=2)
            print(f"Cache analysis: {self.cache_dir / 'cache_analysis.json'}")
        
        return self.verilog_dir, self.result_dir
    
    def run_testing(self):
        """Run testing on generated files"""
        model_name = f"MoA_HLS_{self.model.replace(':', '-')}_L{self.num_layers}_paths_{'_'.join(self.path_config)}"
        
        if self.enable_early_stopping:
            model_name += "_EarlyStop"
        
        if self.enable_self_refinement:
            model_name += f"_SelfRef{self.max_self_refinement_iterations}"
        
        tester = MultiDatasetHDLTester(
            self.verilog_dir,
            self.dataset_dir,
            self.result_dir,
            model_name,
            self.dataset,
            self.temp_mode
        )
        tester.run_tests()
        
        return self.result_dir / "results.json"


def main():
    """Main function with CLI"""
    import sys
    
    # Default configuration
    model = 'gpt-4o-mini'
    num_layers = 4
    dataset = 'rtllm'
    temp_mode = 'high_T'
    enable_quality_caching = True
    n_select = 3
    path_config = None  # Will use default ['direct', 'cpp', 'python']
    enable_early_stopping = True
    enable_self_refinement = True  
    max_self_refinement_iterations = 3 
    
    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg.startswith('--model='):
            model = arg.split('=')[1]
        elif arg.startswith('--layers='):
            num_layers = int(arg.split('=')[1])
        elif arg.startswith('--dataset='):
            dataset = arg.split('=')[1]
        elif arg.startswith('--temp='):
            temp_mode = arg.split('=')[1]
        elif arg.startswith('--n_select='):
            n_select = int(arg.split('=')[1])
        elif arg.startswith('--paths='):
            # Parse path configuration, e.g., --paths=cpp,cpp,cpp or --paths=direct,python
            path_config = arg.split('=')[1].split(',')
        elif arg in ['--no_cache', '--no-cache']:
            enable_quality_caching = False
        elif arg.startswith('--early_stop='):
            enable_early_stopping = arg.split('=')[1].lower() in ['true', '1', 'yes', 'on']
        elif arg == '--early_stop':
            enable_early_stopping = True
        elif arg.startswith('--self_refine='):
            enable_self_refinement = arg.split('=')[1].lower() in ['true', '1', 'yes', 'on']
        elif arg == '--self_refine':
            enable_self_refinement = True
        elif arg == '--no_self_refine':
            enable_self_refinement = False
        elif arg.startswith('--max_refine_iters='):
            max_self_refinement_iterations = int(arg.split('=')[1])
        elif arg == '--help':
            print("MoA-HLS: Multi-path HDL Generation with Configurable Agents")
            print("\nUsage: python MoA_HLS.py [options]")
            print("\nOptions:")
            print("  --model=<name>           LLM model (default: gpt-4o-mini)")
            print("  --layers=<n>             Number of layers (default: 4)")
            print("  --dataset=<name>         Dataset: rtllm|verilogeval (default: rtllm)")
            print("  --temp=<mode>            Temperature: low_T|high_T (default: high_T)")
            print("  --n_select=<n>           Top-n selection per layer (default: 3)")
            print("  --paths=<config>         Path configuration (default: direct,cpp,python)")
            print("                           Examples:")
            print("                             --paths=cpp,cpp,cpp       (3 C++ paths)")
            print("                             --paths=python,python     (2 Python paths)")
            print("                             --paths=direct,cpp        (direct + C++)")
            print("  --no_cache               Disable quality caching")
            print("  --early_stop             Enable early stopping when perfect HDL found")
            print("  --self_refine            Enable self-refinement (default: enabled)")
            print("  --no_self_refine         Disable self-refinement")
            print("  --max_refine_iters=<n>   Max refinement iterations (default: 3)")
            print("\nExample:")
            print("  python MoA_HLS.py --model=gpt-4o-mini --layers=3 --paths=cpp,cpp,cpp --self_refine --max_refine_iters=3")
            return
    
    # Load designs
    designs = load_designs(dataset)
    if not designs:
        print(f"Error: No {dataset} designs found")
        return
    
    print(f"Loaded {len(designs)} designs from {dataset}")
    
    # Initialize generator
    generator = MoAHLSGenerator(
        model=model,
        num_layers=num_layers,
        dataset=dataset,
        temp_mode=temp_mode,
        enable_quality_caching=enable_quality_caching,
        n_select=n_select,
        path_config=path_config,
        enable_early_stopping=enable_early_stopping,
        enable_self_refinement=enable_self_refinement,
        max_self_refinement_iterations=max_self_refinement_iterations
    )
    
    # Run generation
    verilog_dir, result_dir = generator.run_generation(designs)
    
    # Run testing
    print("\nStarting testing phase...")
    test_results = generator.run_testing()
    
    print(f"\nAll complete!")
    print(f"Results: {test_results}")


if __name__ == "__main__":
    main()