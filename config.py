#!/usr/bin/env python3
"""
Configuration for RTLLM (Verilog) and VerilogEval (SystemVerilog) Benchmark Testing with Qwen2.5
"""

from pathlib import Path
import math
import os

class Config:
    # Qwen2.5 models - optimized for coding tasks
    QWEN_MODELS = [
        "qwen2.5:14b",
        "qwen2.5:7b",
        # "qwen2.5:32b",
        # "qwen2.5:72b"
    ]
    
    # OpenAI models - NEW: Added GPT-4o support
    OPENAI_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
    ]
    
    # OpenAI API Configuration - NEW
    OPENAI_API_KEY = ""  # Set via environment or modify here
    OPENAI_BASE_URL = "https://api.openai.com/v1"  # Can be changed for proxy/alternative endpoints
    
    # Supported datasets
    DATASET_TYPES = ["rtllm", "verilogeval"]
    
    # Generation methods
    GENERATION_METHODS = [
        "direct",      # Direct RTL generation (existing)
        "cpp_chain"    # C++ intermediate + RTL generation (new)
    ]
    
    # Temperature settings
    TEMPERATURE_MODES = ["low_T", "high_T"]
    
    # Directories for RTLLM dataset
    RTLLM_DIR = Path("./RTLLM")
    VERILOG_BASE_DIR = Path("./verilog")
    RESULT_BASE_DIR = Path("./result")
    
    # Directories for VerilogEval dataset
    VERILOGEVAL_DIR = Path("./VerilogEval")
    VERILOG_EVAL_BASE_DIR = Path("./verilog_eval")
    RESULT_EVAL_BASE_DIR = Path("./result_eval")
    
    # Generation control
    OVERWRITE_EXISTING = False  # Set to True to overwrite existing files
    
    # Iterative refinement settings
    ENABLE_ITERATIVE_REFINEMENT = False  # Toggle iterative refinement
    MAX_REFINEMENT_ITERATIONS = 3  # Maximum attempts to fix Verilog errors
    
    # Pre-screening settings (independent auxiliary feature)
    ENABLE_PRESCREENING = False  # Enable prescreening
    PRESCREENING_TIMEOUT = 30 # Fast timeout for prescreening (seconds)
    
    # C++ Validation settings
    ENABLE_CPP_VALIDATION = False  # Toggle C++ validation
    CPP_VALIDATION_MODE = "on_failure"  # "always" | "on_failure" | "never"
    MAX_CPP_REFINEMENT_ITERATIONS = 3  # Maximum attempts to fix C++ code
    
    # Simulation settings
    SIMULATION_TIMEOUT = 30  # seconds for vvp simulation
    COMPILATION_TIMEOUT = 30  # seconds for iverilog compilation
    
    # Design paths mapping (populated by utils.load_designs)
    DESIGN_PATHS = {}
    
    # Temperature-specific parameters
    LOW_T_PARAMS = {
        "temperature": 0.0,
        "top_p": 0.01,
    }
    
    HIGH_T_PARAMS = {
        "temperature": 0.8,
        "top_p": 0.95,
    }
   
    # Qwen2.5-optimized base parameters - leveraging its coding strengths
    LLM_PARAMS = {
        "qwen2.5:7b": {
            "context_length": 32768,  # Qwen2.5 supports long context
            "num_predict": 2048,     # Sufficient for code generation
            "timeout": 90
        },
        "qwen2.5:14b": {
            "context_length": 32768,
            "num_predict": 2048,
            "timeout": 120
        },
        "qwen2.5:32b": {
            "context_length": 128000, # Max context for large model
            "num_predict": 3072,
            "timeout": 180
        },
        "qwen2.5:72b": {
            "context_length": 128000,
            "num_predict": 4096,     # Most tokens for best model
            "timeout": 240
        },
        # NEW: OpenAI GPT-4o parameters
        "gpt-4o": {
            "context_length": 128000,
            "num_predict": 4096,
            "timeout": 120
        },
        "gpt-4o-mini": {
            "context_length": 128000,
            "num_predict": 4096,
            "timeout": 90
        }
    }
    
    DEFAULT_PARAMS = {
        "context_length": 32768,
        "num_predict": 2048,
        "timeout": 120
    }
    
    # Evaluation settings
    N_SAMPLES = 10
    K_VALUES = [1, 3] 
    
    @classmethod
    def get_model_params(cls, model_name: str, temp_mode: str = "low_T") -> dict:
        """Get model parameters with temperature mode"""
        base_params = cls.LLM_PARAMS.get(model_name, cls.DEFAULT_PARAMS).copy()
        
        if temp_mode == "low_T":
            base_params.update(cls.LOW_T_PARAMS)
        elif temp_mode == "high_T":
            base_params.update(cls.HIGH_T_PARAMS)
        else:
            raise ValueError(f"Unknown temperature mode: {temp_mode}")
        
        return base_params
    
    @classmethod
    def is_openai_model(cls, model_name: str) -> bool:
        """NEW: Check if model is an OpenAI model"""
        return model_name in cls.OPENAI_MODELS
    
    @classmethod
    def get_folder_name(cls, model_name: str, method: str = "direct", 
                       temp_mode: str = "low_T", dataset: str = "rtllm") -> str:
        """Get folder name for model, method, temperature and dataset combination"""
        base_name = model_name.replace(":", "_").replace(".", "_").replace("-", "_")  # Modified: handle GPT model names
        suffix_parts = []
        
        if method != "direct":
            suffix_parts.append(method)
        
        # Add temperature mode
        suffix_parts.append(temp_mode)
        
        # Add C++ validation indicator with iteration count if applicable
        if method == "cpp_chain" and cls.ENABLE_CPP_VALIDATION:
            # Include the number of C++ refinement iterations in the folder name
            suffix_parts.append(f"cppval{cls.MAX_CPP_REFINEMENT_ITERATIONS}")
        
        # Add prescreening indicator
        if cls.ENABLE_PRESCREENING:
            suffix_parts.append("prescreen")
        
        # Add Verilog refinement indicator with iteration count
        if cls.ENABLE_ITERATIVE_REFINEMENT:
            suffix_parts.append(f"refine{cls.MAX_REFINEMENT_ITERATIONS}")
        
        if suffix_parts:
            return f"{base_name}_{'_'.join(suffix_parts)}"
        return f"{base_name}_{temp_mode}"
    
    @classmethod
    def get_output_dirs(cls, model_name: str, method: str = "direct", 
                       temp_mode: str = "low_T", dataset: str = "rtllm") -> tuple:
        """Get verilog and result output directories based on dataset"""
        folder_name = cls.get_folder_name(model_name, method, temp_mode, dataset)
        temp_folder = Path(temp_mode)
        
        if dataset == "verilogeval":
            verilog_dir = cls.VERILOG_EVAL_BASE_DIR / temp_folder / folder_name
            result_dir = cls.RESULT_EVAL_BASE_DIR / temp_folder / folder_name
        else:  # rtllm
            verilog_dir = cls.VERILOG_BASE_DIR / temp_folder / folder_name
            result_dir = cls.RESULT_BASE_DIR / temp_folder / folder_name
            
        return verilog_dir, result_dir
    
    @classmethod
    def get_design_path(cls, design_name: str) -> Path:
        """Get full path to design directory"""
        return cls.DESIGN_PATHS.get(design_name, cls.RTLLM_DIR / design_name)
    
    @classmethod
    def get_file_extension(cls, dataset: str) -> str:
        """Get file extension based on dataset"""
        return ".sv" if dataset == "verilogeval" else ".v"
    
    @staticmethod
    def calculate_pass_at_k(n: int, c: int, k: int) -> float:
        """
        Calculate pass@k using VerilogEval formula
        
        pass@k = probability that at least 1 of k randomly drawn samples is correct
               = 1 - P(all k samples are incorrect)
               = 1 - C(n-c, k) / C(n, k)
        
        Where:
        - n = total samples per problem  
        - c = number of correct samples
        - k = number of attempts allowed
        """
        if n <= 0 or c < 0 or k <= 0:
            return 0.0
        
        if c == 0:
            return 0.0  # No correct solutions available
            
        if k > n:
            k = n  # Can't draw more samples than we have
        
        try:
            # If there aren't enough incorrect samples to fill k draws,
            # we're guaranteed to get at least one correct sample
            if (n - c) < k:
                return 1.0
            
            # Standard case: P(all k samples incorrect) = C(n-c, k) / C(n, k)
            prob_all_wrong = math.comb(n - c, k) / math.comb(n, k)
            return 1.0 - prob_all_wrong
            
        except (ValueError, ZeroDivisionError):
            return 0.0