#!/usr/bin/env python3
"""
Multi-Dataset Generator with enhanced prompt strategies and code extraction
Supports iterative refinement, prescreening, and C++ validation
"""

import os
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from config import Config
from verilog_refiner import MultiDatasetVerilogRefiner

class MultiDatasetGenerator:
    def __init__(self, llm_interface, designs: List[Dict], output_dir: Path, 
                 method: str = "direct", dataset: str = "rtllm", temp_mode: str = "low_T"):
        self.llm = llm_interface
        self.designs = designs
        self.output_dir = output_dir
        self.method = method
        self.dataset = dataset
        self.temp_mode = temp_mode
        
        # Initialize refiner if enabled
        self.refiner = None
        if Config.ENABLE_ITERATIVE_REFINEMENT:
            dataset_dir = Config.VERILOGEVAL_DIR if dataset == "verilogeval" else Config.RTLLM_DIR
            self.refiner = MultiDatasetVerilogRefiner(llm_interface, Config.MAX_REFINEMENT_ITERATIONS, dataset, dataset_dir)
        
        # Initialize prescreener if enabled
        self.prescreener = None
        self.prescreening_summary = {
            'enabled': Config.ENABLE_PRESCREENING,
            'total_attempts': 0,
            'direct_passed': 0,
            'fallback_used': 0,
            'time_saved': 0.0,
            'by_design': {}
        }
        
        if Config.ENABLE_PRESCREENING:
            from prescreener import TrialPrescreener
            dataset_dir = Config.VERILOGEVAL_DIR if dataset == "verilogeval" else Config.RTLLM_DIR
            self.prescreener = TrialPrescreener(
                llm_interface,
                dataset_dir,
                dataset,
                temp_mode
            )
        
        # Initialize C++ validator if enabled for cpp_chain method
        self.cpp_validator = None
        self.cpp_validation_summary = {
            'enabled': Config.ENABLE_CPP_VALIDATION and method == "cpp_chain",
            'mode': Config.CPP_VALIDATION_MODE if method == "cpp_chain" else "disabled",
            'total_validations': 0,
            'successful_validations': 0,
            'cpp_fixes_applied': 0,
            'by_design': {}
        }
        
        if Config.ENABLE_CPP_VALIDATION and method == "cpp_chain":
            from cpp_validator import CppValidator
            self.cpp_validator = CppValidator(llm_interface, Config.MAX_CPP_REFINEMENT_ITERATIONS)
        
        # Cache for C++ code when using cpp_chain
        self.last_cpp_code = None
        self.last_analysis = None
        self.last_description = None
        
        # Dataset-specific configurations
        self.file_extension = Config.get_file_extension(dataset)
        self.module_name_required = "TopModule" if dataset == "verilogeval" else None
        
        # Enhanced stage-specific system prompts (MoA-inspired)
        if self.dataset == "verilogeval":
            rtl_prompt = (
                "You are a professional SystemVerilog RTL designer. "
                "Generate syntactically correct, synthesizable SystemVerilog code following best practices. "
                "Output ONLY the SystemVerilog code starting with 'module' and ending with 'endmodule'. "
                "Do NOT include any markdown formatting, explanations, or comments outside the module."
            )
        else:
            rtl_prompt = (
                "You are a professional Verilog RTL designer. "
                "Generate syntactically correct, synthesizable Verilog code following best practices. "
                "Output ONLY the Verilog code starting with 'module' and ending with 'endmodule'. "
                "Do NOT include any markdown formatting or explanations."
            )
        
        self.system_prompts = {
            "analyzer": "You are a hardware architecture analyst specializing in digital design. Extract and structure technical requirements from RTL specifications with precision.",
            "cpp_developer": "You are an HLS (High-Level Synthesis) C++ expert. Generate synthesizable C++ code suitable for hardware implementation using fixed-size arrays, explicit loops, and no dynamic memory allocation.",
            "rtl_designer": rtl_prompt
        }
        
        # Create trial folders
        for i in range(1, Config.N_SAMPLES + 1):
            (output_dir / f"t{i}").mkdir(exist_ok=True)
    
    def read_description(self, desc_file: Path) -> str:
        """Read description file (design_description.txt for RTLLM, prompt.txt for VerilogEval)"""
        try:
            with open(desc_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except:
            return ""
    
    def find_testbench(self, design: Dict) -> Optional[Path]:
        """Find testbench for the design based on dataset type"""
        if 'testbench' in design and design['testbench'].exists():
            return design['testbench']
        
        if self.dataset == "rtllm":
            # Try to find in RTLLM directory structure
            if hasattr(Config, 'DESIGN_PATHS') and design['name'] in Config.DESIGN_PATHS:
                design_dir = Config.DESIGN_PATHS[design['name']]
                testbench = design_dir / "testbench.v"
                if testbench.exists():
                    return testbench
        elif self.dataset == "verilogeval":
            # VerilogEval testbenches are in the same directory with _test.sv suffix
            testbench = Config.VERILOGEVAL_DIR / f"{design['name']}_test.sv"
            if testbench.exists():
                return testbench
        
        return None
    
    def clean_verilog_for_dataset(self, code: str, design_name: str) -> str:
        """Clean Verilog code based on dataset requirements"""
        if not code:
            return ""
        
        lines = code.split('\n')
        cleaned_lines = []
        module_found = False
        
        for line in lines:
            stripped = line.strip()
            
            if not module_found and not stripped:
                continue
            if not module_found and stripped.startswith('//'):
                continue
                
            if not module_found and stripped.startswith('module '):
                module_found = True
                # For VerilogEval, enforce TopModule naming
                if self.dataset == "verilogeval" and self.module_name_required:
                    line = re.sub(r'module\s+\w+', f'module {self.module_name_required}', line)
            
            if module_found:
                cleaned_lines.append(line)
                if stripped in ['endmodule', 'endmodule;']:
                    break
        
        return '\n'.join(cleaned_lines) if cleaned_lines else code
    
    def generate_verilogeval_prompt(self, prompt_text: str) -> str:
        """Generate enhanced prompt for VerilogEval with strong format requirements"""
        enhanced = f"""Generate SystemVerilog code for this specification.

CRITICAL REQUIREMENTS:
1. Output ONLY the module code
2. Module name MUST be exactly 'TopModule'
3. Use modern SystemVerilog port declaration style
4. Start directly with: module TopModule
5. End with: endmodule
6. NO markdown formatting (no ```)
7. NO explanations or comments outside the module

Task Specification:
{prompt_text}

OUTPUT THE SYSTEMVERILOG MODULE CODE NOW (nothing else):"""
        
        return enhanced
    
    def enhance_prompt_for_rtllm(self, description: str) -> str:
        """Enhanced prompt for RTLLM with strong format requirements"""
        module_name_match = re.search(r'Module name:\s*(\w+)', description)
        module_name = module_name_match.group(1) if module_name_match else "module_name"
        
        enhanced = f"""Generate Verilog code for this specification.

CRITICAL REQUIREMENTS:
1. Output ONLY the complete module code
2. Module name should be: {module_name}
3. Start directly with: module {module_name}
4. End with: endmodule
5. NO markdown formatting (no ```)
6. NO explanations before or after the code
7. Do NOT add any text outside the module

Design Specification:
{description}

OUTPUT THE VERILOG MODULE CODE NOW (nothing else):"""
        
        return enhanced
    
    def generate_structured_analysis(self, description: str) -> str:
        """Generate structured technical analysis from description"""
        prompt = f"""Analyze this RTL specification and extract key technical requirements.

Provide a structured analysis including:
1. Input signals (name, bit width, purpose)
2. Output signals (name, bit width, purpose)
3. Internal state requirements (registers, counters, FSM states)
4. Core operations and algorithms
5. Timing requirements (combinational vs sequential logic)
6. Critical design constraints

Specification:
{description}

Provide concise technical analysis in structured format:"""
        
        return prompt
    
    def generate_cpp_code_prompt(self, analysis: str) -> str:
        """Generate HLS-compatible C++ from structured analysis"""
        prompt = f"""Generate HLS-compatible C++ code based on this technical analysis.

Requirements:
- Use fixed-size arrays only (no vectors or dynamic allocation)
- Use explicit loops (suitable for unrolling)
- Use appropriate bit-width types (uint8_t, uint16_t, uint32_t)
- Implement the exact functionality described
- Include clear function interfaces

Technical Analysis:
{analysis}

Provide complete HLS-compatible C++ implementation:"""
        
        return prompt
    
    def generate_verilog_from_cpp(self, cpp_code: str, analysis: str) -> str:
        """Generate Verilog from C++ code and analysis with strong format requirements"""
        if self.dataset == "verilogeval":
            module_instruction = "CRITICAL: The module MUST be named 'TopModule' exactly.\n"
            language = "SystemVerilog"
        else:
            module_instruction = ""
            language = "Verilog"
        
        prompt = f"""Convert this HLS C++ implementation to synthesizable {language} RTL.

{module_instruction}CRITICAL OUTPUT REQUIREMENTS:
1. Output ONLY the module code
2. NO markdown formatting (no ```)
3. NO explanations before or after the code
4. Start directly with: module <name>
5. End with: endmodule

Mapping rules:
- C++ loops map to sequential logic with proper clock/reset
- C++ arrays map to register arrays or memory blocks
- C++ if-else maps to Verilog conditional assignments or always blocks
- Ensure all sequential logic uses proper clock edge sensitivity
- Include proper reset logic (synchronous or asynchronous as appropriate)

Technical Requirements:
{analysis}

C++ Implementation:
{cpp_code}

OUTPUT THE {language.upper()} MODULE CODE NOW (nothing else):"""
        
        return prompt
    
    def generate_single_trial_with_prescreening(self, design: Dict, description: str, 
                                               trial_num: int) -> Tuple[Optional[str], Dict]:
        """
        Generate single trial with optional prescreening
        
        Returns: (verilog_code, generation_info)
        """
        generation_info = {
            'trial_num': trial_num,
            'method_attempted': self.method,
            'prescreening_attempted': False,
            'prescreening_passed': False,
            'actual_method_used': None,
            'refinement_info': None,
            'cpp_validation_info': None
        }
        
        # Prescreening logic
        if self.prescreener:
            generation_info['prescreening_attempted'] = True
            
            # Execute prescreening
            print("P", end="", flush=True)
            prescreen_result = self.prescreener.prescreen_trial(
                design, 
                trial_num, 
                description
            )
            
            # Update summary statistics
            self.prescreening_summary['total_attempts'] += 1
            
            if prescreen_result['prescreening_passed']:
                # Prescreening successful
                self.prescreening_summary['direct_passed'] += 1
                generation_info['prescreening_passed'] = True
                generation_info['actual_method_used'] = 'direct_prescreened'
                generation_info['prescreen_details'] = prescreen_result
                print("✓", end="", flush=True)
                
                return prescreen_result['generated_code'], generation_info
            else:
                # Prescreening failed, use fallback
                self.prescreening_summary['fallback_used'] += 1
                generation_info['prescreen_failed_reason'] = prescreen_result.get('error_info')
                print("F", end="", flush=True)
        
        # Use original method (prescreening failed or not enabled)
        generation_info['actual_method_used'] = self.method
        
        if self.method == "cpp_chain":
            code, info_dict = self.generate_single_trial_cpp_chain_with_info(description, design)
            if info_dict:
                generation_info['refinement_info'] = info_dict.get('refinement_info')
                generation_info['cpp_validation_info'] = info_dict.get('cpp_validation_info')
        else:
            code, refine_info = self.generate_single_trial_direct_with_info(description, design)
            generation_info['refinement_info'] = refine_info
        
        return code, generation_info
    
    def generate_single_trial_direct_with_info(self, description: str, design: Dict = None) -> Tuple[Optional[str], Optional[Dict]]:
        """Generate single trial using direct method with enhanced prompt and retry"""
        # Enhance prompt based on dataset
        if self.dataset == "verilogeval":
            enhanced_prompt = self.generate_verilogeval_prompt(description)
        else:
            enhanced_prompt = self.enhance_prompt_for_rtllm(description)
        
        # Try twice with increasingly forceful prompts
        for attempt in range(2):
            if attempt > 0:
                # Second attempt - add even stronger reminder
                retry_prompt = enhanced_prompt + "\n\nREMINDER: Output ONLY the code, NO explanations, NO markdown!"
                response = self.llm.generate_response(retry_prompt, self.system_prompts["rtl_designer"])
            else:
                response = self.llm.generate_response(enhanced_prompt, self.system_prompts["rtl_designer"])
            
            if response:
                # Use enhanced extraction with dataset parameter
                initial_verilog = self.llm.extract_verilog(response, self.dataset)
                
                if initial_verilog:
                    # Apply refinement if enabled and initial code exists
                    if self.refiner and design:
                        # Get testbench path based on dataset
                        if self.dataset == "verilogeval":
                            testbench_result = self.refiner.find_testbench(design['name'])
                            if isinstance(testbench_result, tuple) and testbench_result[0]:
                                testbench_path = testbench_result  # Pass tuple for VerilogEval
                            else:
                                testbench_path = None
                        else:
                            testbench_path = self.refiner.find_testbench(design['name'])
                            if isinstance(testbench_path, tuple):
                                testbench_path = testbench_path[0]  # Extract single path for RTLLM
                        
                        if testbench_path:
                            # First test the initial code
                            test_result = self.refiner.test_verilog(initial_verilog, testbench_path)
                            
                            # Only refine if it failed
                            if not test_result['passed']:
                                refined_verilog, refinement_info = self.refiner.refine_verilog(
                                    initial_verilog, 
                                    testbench_path,
                                    description
                                )
                                return refined_verilog, refinement_info
                            else:
                                # Already passes, no need to refine
                                return initial_verilog, None
                    
                    return initial_verilog, None
        
        return None, None
    
    def generate_single_trial_cpp_chain_with_info(self, description: str, design: Dict = None) -> Tuple[Optional[str], Dict]:
        """Generate single trial using C++ chain method with enhanced prompts and refinement"""
        info_dict = {
            'refinement_info': None,
            'cpp_validation_info': None
        }
        
        # Stage 1: Structured analysis
        analysis_prompt = self.generate_structured_analysis(description)
        analysis_response = self.llm.generate_response(analysis_prompt, self.system_prompts["analyzer"])
        
        if not analysis_response:
            return None, info_dict
        
        # Stage 2: Generate HLS C++ code
        cpp_code_prompt = self.generate_cpp_code_prompt(analysis_response)
        cpp_code_response = self.llm.generate_response(cpp_code_prompt, self.system_prompts["cpp_developer"])
        
        if not cpp_code_response:
            return None, info_dict
        
        cpp_code = self.llm.extract_cpp_code(cpp_code_response)
        
        # Optional C++ validation in "always" mode
        cpp_validation_info = None
        if self.cpp_validator and Config.CPP_VALIDATION_MODE == "always":
            cpp_code, cpp_validation_info = self.cpp_validator.validate_and_refine_cpp(
                cpp_code, description
            )
            self.cpp_validation_summary['total_validations'] += 1
            if cpp_validation_info.get('success'):
                self.cpp_validation_summary['successful_validations'] += 1
            info_dict['cpp_validation_info'] = cpp_validation_info
        
        # Store C++ code for potential later validation
        self.last_cpp_code = cpp_code
        self.last_analysis = analysis_response
        self.last_description = description
        
        # Stage 3: Generate Verilog from C++ with enhanced prompt and retry
        for attempt in range(2):
            verilog_prompt = self.generate_verilog_from_cpp(cpp_code, analysis_response)
            
            if attempt > 0:
                verilog_prompt += "\n\nCRITICAL: Output ONLY the module code! NO markdown, NO explanations!"
            
            verilog_response = self.llm.generate_response(verilog_prompt, self.system_prompts["rtl_designer"])
            
            if verilog_response:
                # Use enhanced extraction with dataset parameter
                initial_verilog = self.llm.extract_verilog(verilog_response, self.dataset)
                
                if initial_verilog:
                    # Apply refinement if enabled and initial code exists
                    if self.refiner and design:
                        # Get testbench path based on dataset
                        if self.dataset == "verilogeval":
                            testbench_result = self.refiner.find_testbench(design['name'])
                            if isinstance(testbench_result, tuple) and testbench_result[0]:
                                testbench_path = testbench_result  # Pass tuple for VerilogEval
                            else:
                                testbench_path = None
                        else:
                            testbench_path = self.refiner.find_testbench(design['name'])
                            if isinstance(testbench_path, tuple):
                                testbench_path = testbench_path[0]  # Extract single path for RTLLM
                        
                        if testbench_path:
                            # First test the initial code
                            test_result = self.refiner.test_verilog(initial_verilog, testbench_path)
                            
                            # If failed, consider C++ validation in "on_failure" mode
                            if not test_result['passed'] and test_result.get('stage') == 'simulation':
                                if self.cpp_validator and Config.CPP_VALIDATION_MODE == "on_failure":
                                    # Check if we should fix C++ instead
                                    cpp_check = self.cpp_validator.should_fix_cpp(
                                        test_result.get('errors', []),
                                        self.last_cpp_code,
                                        self.last_description
                                    )
                                    
                                    if cpp_check['fix_cpp']:
                                        # Fix C++ and regenerate
                                        print(" [CPP-FIX]", end="", flush=True)
                                        refined_cpp, cpp_val_info = self.cpp_validator.validate_and_refine_cpp(
                                            self.last_cpp_code,
                                            self.last_description,
                                            test_result.get('errors', [])
                                        )
                                        
                                        self.cpp_validation_summary['total_validations'] += 1
                                        if cpp_val_info.get('success'):
                                            self.cpp_validation_summary['successful_validations'] += 1
                                            self.cpp_validation_summary['cpp_fixes_applied'] += 1
                                            
                                            # Regenerate Verilog from fixed C++
                                            new_verilog_prompt = self.generate_verilog_from_cpp(refined_cpp, self.last_analysis)
                                            new_verilog_response = self.llm.generate_response(
                                                new_verilog_prompt,
                                                self.system_prompts["rtl_designer"]
                                            )
                                            
                                            if new_verilog_response:
                                                new_verilog = self.llm.extract_verilog(new_verilog_response, self.dataset)
                                                if new_verilog:
                                                    info_dict['cpp_validation_info'] = cpp_val_info
                                                    return new_verilog, info_dict
                            
                            # Standard Verilog refinement if C++ is OK or not checked
                            if not test_result['passed']:
                                refined_verilog, refinement_info = self.refiner.refine_verilog(
                                    initial_verilog,
                                    testbench_path,
                                    description
                                )
                                info_dict['refinement_info'] = refinement_info
                                return refined_verilog, info_dict
                            else:
                                # Already passes, no need to refine
                                return initial_verilog, info_dict
                    
                    return initial_verilog, info_dict
        
        return None, info_dict
    
    def generate_design_trials(self, design: Dict) -> Dict:
        """Generate trials for one design using specified method"""
        design_name = design["name"]
        description = self.read_description(design["description"])
        
        if not description:
            return {"design": design_name, "error": "No description", "trials": [], "successful_count": 0}
        
        # Check existing files
        if not Config.OVERWRITE_EXISTING:
            existing_files = []
            for i in range(1, Config.N_SAMPLES + 1):
                trial_file = self.output_dir / f"t{i}" / f"{design_name}{self.file_extension}"
                if trial_file.exists():
                    existing_files.append(trial_file)
            
            if existing_files:
                print(f"Skipping {design_name} (found {len(existing_files)} existing files)")
                return {
                    "design": design_name, 
                    "skipped": True,
                    "existing_files": len(existing_files),
                    "trials": [],
                    "successful_count": 0
                }
        
        # Build configuration string
        config_parts = [f"{self.method}_{self.temp_mode}"]
        if self.method == "cpp_chain" and self.cpp_validator:
            config_parts.append(f"cppval_{Config.CPP_VALIDATION_MODE}")
        if self.prescreener:
            config_parts.append("prescreen")
        if self.refiner:
            config_parts.append(f"refine{Config.MAX_REFINEMENT_ITERATIONS}")
        
        print(f"Generating {design_name} ({'+'.join(config_parts)}): ", end="")
        
        # Initialize design-level prescreening stats
        design_prescreening = {
            'total': 0,
            'passed': 0,
            'fallback': 0,
            'trials': {}
        }
        
        # Initialize design-level C++ validation stats
        design_cpp_validation = {
            'total': 0,
            'successful': 0,
            'fixes_applied': 0,
            'trials': {}
        }
        
        # Generate trials with refinement tracking
        trials = []
        successful_count = 0
        refinement_stats = {
            'total_refined': 0,
            'refinement_successful': 0,
            'average_iterations': 0,
            'trial_details': {}
        }
        
        for i in range(Config.N_SAMPLES):
            trial_num = i + 1
            print(f"t{trial_num}", end="", flush=True)
            
            # Generate with prescreening if enabled
            verilog_code, generation_info = self.generate_single_trial_with_prescreening(
                design, description, trial_num
            )
            
            # Update design prescreening stats
            if generation_info['prescreening_attempted']:
                design_prescreening['total'] += 1
                design_prescreening['trials'][f't{trial_num}'] = {
                    'passed': generation_info['prescreening_passed'],
                    'method_used': generation_info['actual_method_used']
                }
                
                if generation_info['prescreening_passed']:
                    design_prescreening['passed'] += 1
                else:
                    design_prescreening['fallback'] += 1
            
            # Update C++ validation stats
            if generation_info.get('cpp_validation_info'):
                design_cpp_validation['total'] += 1
                cpp_val_info = generation_info['cpp_validation_info']
                if cpp_val_info.get('success'):
                    design_cpp_validation['successful'] += 1
                if cpp_val_info.get('iterations', 0) > 1:
                    design_cpp_validation['fixes_applied'] += 1
                design_cpp_validation['trials'][f't{trial_num}'] = {
                    'success': cpp_val_info.get('success', False),
                    'iterations': cpp_val_info.get('iterations', 0)
                }
            
            # Process refinement info if present
            trial_refinement = generation_info.get('refinement_info')
            
            if verilog_code:
                print("G", end="")  # Generated
                cleaned = self.clean_verilog_for_dataset(verilog_code, design_name)
                if cleaned:
                    print("C", end="")  # Cleaned
                    # Save to trial folder with correct extension
                    trial_file = self.output_dir / f"t{trial_num}" / f"{design_name}{self.file_extension}"
                    try:
                        with open(trial_file, 'w') as f:
                            f.write(cleaned)
                        
                        # Build trial record with all info
                        trial_record = {
                            "trial": trial_num, 
                            "file": str(trial_file), 
                            "success": True,
                            "generation_info": generation_info
                        }
                        
                        # Add refinement information if available
                        if trial_refinement:
                            trial_record["refinement"] = {
                                "success": trial_refinement.get('success', False),
                                "iterations": trial_refinement.get('iterations', 0)
                            }
                            refinement_stats['total_refined'] += 1
                            if trial_refinement.get('success'):
                                refinement_stats['refinement_successful'] += 1
                            refinement_stats['trial_details'][f't{trial_num}'] = trial_refinement
                        
                        trials.append(trial_record)
                        successful_count += 1
                        print("✓", end="")
                    except Exception as e:
                        trials.append({"trial": trial_num, "error": f"Save: {e}", "success": False})
                        print("x", end="")
                else:
                    trials.append({"trial": trial_num, "error": "Clean failed", "success": False})
                    print("x", end="")
            else:
                trials.append({"trial": trial_num, "error": "No response", "success": False})
                print("x", end="")
            
            if i < Config.N_SAMPLES - 1:
                print(" ", end="")
                time.sleep(0.3)
        
        print(f" -> {successful_count}/{Config.N_SAMPLES}")
        
        # Add prescreening summary if applicable
        if self.prescreener and design_prescreening['total'] > 0:
            print(f"  Prescreening: {design_prescreening['passed']}/{design_prescreening['total']} passed, "
                  f"{design_prescreening['fallback']}/{design_prescreening['total']} used {self.method} fallback")
        
        # Add C++ validation summary if applicable
        if self.cpp_validator and design_cpp_validation['total'] > 0:
            print(f"  C++ Validation: {design_cpp_validation['successful']}/{design_cpp_validation['total']} successful, "
                  f"{design_cpp_validation['fixes_applied']} fixes applied")
        
        # Calculate average iterations for refined trials
        if refinement_stats['total_refined'] > 0:
            total_iters = sum(
                details.get('iterations', 0) 
                for details in refinement_stats['trial_details'].values()
            )
            refinement_stats['average_iterations'] = total_iters / refinement_stats['total_refined']
        
        # Update summaries
        if self.prescreener:
            self.prescreening_summary['by_design'][design_name] = design_prescreening
        
        if self.cpp_validator:
            self.cpp_validation_summary['by_design'][design_name] = design_cpp_validation
        
        return {
            "design": design_name,
            "trials": trials,
            "successful_count": successful_count,
            "method": self.method,
            "dataset": self.dataset,
            "temp_mode": self.temp_mode,
            "prescreening_enabled": self.prescreener is not None,
            "prescreening_stats": design_prescreening if self.prescreener else None,
            "cpp_validation_enabled": self.cpp_validator is not None,
            "cpp_validation_stats": design_cpp_validation if self.cpp_validator else None,
            "refinement_enabled": self.refiner is not None,
            "refinement_stats": refinement_stats if self.refiner else None
        }
    
    def generate_all(self):
        """Generate all designs"""
        dataset_display = "SystemVerilog" if self.dataset == "verilogeval" else "Verilog"
        print(f"Generating {len(self.designs)} {self.dataset} {dataset_display} designs with {Config.N_SAMPLES} trials each")
        print(f"Method: {self.method}, Temperature: {self.temp_mode}")
        print(f"File extension: {self.file_extension}")
        if self.dataset == "verilogeval":
            print("Module naming: TopModule (enforced)")
        
        if self.prescreener:
            print(f"Prescreening: ENABLED (syntax + simulation required)")
        else:
            print("Prescreening: DISABLED")
        
        if self.cpp_validator:
            print(f"C++ Validation: ENABLED (mode: {Config.CPP_VALIDATION_MODE})")
        else:
            print("C++ Validation: DISABLED")
            
        if self.refiner:
            print(f"Iterative refinement: ENABLED (max {Config.MAX_REFINEMENT_ITERATIONS} iterations)")
        else:
            print("Iterative refinement: DISABLED")
        
        results = []
        total_successful = 0
        skipped_count = 0
        generated_count = 0
        total_refinement_successful = 0
        total_refined = 0
        
        for i, design in enumerate(self.designs, 1):
            print(f"[{i}/{len(self.designs)}] ", end="")
            result = self.generate_design_trials(design)
            results.append(result)
            
            if result.get("skipped", False):
                skipped_count += 1
            else:
                generated_count += 1
                total_successful += result.get("successful_count", 0)
                
                # Aggregate refinement stats
                if result.get("refinement_stats"):
                    stats = result["refinement_stats"]
                    total_refined += stats.get("total_refined", 0)
                    total_refinement_successful += stats.get("refinement_successful", 0)
            
            time.sleep(0.5)
        
        # Build comprehensive summary
        summary = {
            "model": self.llm.model_name,
            "method": self.method,
            "dataset": self.dataset,
            "temp_mode": self.temp_mode,
            "prescreening_enabled": self.prescreener is not None,
            "cpp_validation_enabled": self.cpp_validator is not None,
            "cpp_validation_mode": Config.CPP_VALIDATION_MODE if self.cpp_validator else "disabled",
            "refinement_enabled": self.refiner is not None,
            "max_refinement_iterations": Config.MAX_REFINEMENT_ITERATIONS if self.refiner else 0,
            "max_cpp_refinement_iterations": Config.MAX_CPP_REFINEMENT_ITERATIONS if self.cpp_validator else 0,
            "model_series": "qwen2.5",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_designs": len(self.designs),
            "skipped_designs": skipped_count,
            "generated_designs": generated_count,
            "total_trials_requested": generated_count * Config.N_SAMPLES,
            "total_successful": total_successful,
            "success_rate": f"{total_successful/max(1, generated_count * Config.N_SAMPLES)*100:.1f}%" if generated_count > 0 else "0.0%",
        }
        
        # Add prescreening summary if enabled
        if self.prescreener and self.prescreening_summary['total_attempts'] > 0:
            summary["prescreening_summary"] = {
                "total_attempts": self.prescreening_summary['total_attempts'],
                "direct_passed": self.prescreening_summary['direct_passed'],
                "fallback_used": self.prescreening_summary['fallback_used'],
                "success_rate": f"{self.prescreening_summary['direct_passed']/self.prescreening_summary['total_attempts']*100:.1f}%"
            }
        
        # Add C++ validation summary if enabled
        if self.cpp_validator and self.cpp_validation_summary['total_validations'] > 0:
            summary["cpp_validation_summary"] = {
                "mode": self.cpp_validation_summary['mode'],
                "total_validations": self.cpp_validation_summary['total_validations'],
                "successful_validations": self.cpp_validation_summary['successful_validations'],
                "cpp_fixes_applied": self.cpp_validation_summary['cpp_fixes_applied'],
                "success_rate": f"{self.cpp_validation_summary['successful_validations']/self.cpp_validation_summary['total_validations']*100:.1f}%"
            }
        
        # Add refinement summary if enabled
        if self.refiner and total_refined > 0:
            summary["refinement_summary"] = {
                "total_trials_refined": total_refined,
                "refinement_successful": total_refinement_successful,
                "refinement_success_rate": f"{total_refinement_successful/total_refined*100:.1f}%"
            }
        
        summary["details"] = results
        
        # Save detailed summary
        with open(self.output_dir / "generation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nGeneration completed: {total_successful}/{generated_count * Config.N_SAMPLES} successful")
        
        if self.prescreener and self.prescreening_summary['total_attempts'] > 0:
            print(f"Prescreening: {self.prescreening_summary['direct_passed']}/{self.prescreening_summary['total_attempts']} "
                  f"passed direct generation")
        
        if self.cpp_validator and self.cpp_validation_summary['total_validations'] > 0:
            print(f"C++ Validation: {self.cpp_validation_summary['successful_validations']}/{self.cpp_validation_summary['total_validations']} "
                  f"successful, {self.cpp_validation_summary['cpp_fixes_applied']} fixes applied")
            
        if self.refiner and total_refined > 0:
            print(f"Refinement: {total_refinement_successful}/{total_refined} trials passed after refinement")
            
        if skipped_count > 0:
            print(f"Skipped {skipped_count} designs (use --overwrite to regenerate)")
            
        if generated_count > 0:
            print(f"Success rate: {total_successful/(generated_count * Config.N_SAMPLES)*100:.1f}%")
        else:
            print("No new files generated")