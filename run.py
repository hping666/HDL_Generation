#!/usr/bin/env python3
"""
Simple runner script for Qwen2.5 multi-dataset benchmark with temperature, prescreening and C++ validation support
"""

import sys
from utils import print_setup_check, get_available_models, load_all_designs
from config import Config
from main import main, test_model, run_all_combinations
from utils import load_designs

def run_single_model(model_name: str, method: str = "direct", 
                     dataset: str = "rtllm", temp_mode: str = "low_T"):
    """Run benchmark for single Qwen2.5 model with specified configuration"""
    designs = load_designs(dataset)
    if not designs:
        print(f"No {dataset} designs found")
        return
    
    print(f"Testing {model_name} on {dataset.upper()} dataset")
    print(f"Method: {method}, Temperature: {temp_mode}")
    print(f"Designs: {len(designs)}")
    if Config.ENABLE_PRESCREENING:
        print(f"Prescreening: Enabled (syntax + simulation required)")
    if Config.ENABLE_CPP_VALIDATION and method == "cpp_chain":
        print(f"C++ Validation: Enabled (mode: {Config.CPP_VALIDATION_MODE})")
    if Config.ENABLE_ITERATIVE_REFINEMENT:
        print(f"Refinement: Enabled (max {Config.MAX_REFINEMENT_ITERATIONS} iterations)")
    
    test_model(model_name, designs, method, dataset, temp_mode)

def list_models():
    """List available models with Qwen2.5 focus"""
    models = get_available_models()
    print("Available Ollama models:")
    
    qwen25_models = [m for m in models if "qwen2.5" in m]
    other_models = [m for m in models if "qwen2.5" not in m]
    
    print("\nQwen2.5 Models (optimized for this benchmark):")
    for model in qwen25_models:
        status = " (configured)" if model in Config.QWEN_MODELS else ""
        print(f"  {model}{status}")
    
    if other_models:
        print(f"\nOther Models: {len(other_models)} available")

def show_info():
    """Show configuration and dataset information"""
    print("Multi-Dataset Qwen2.5 Benchmark Configuration")
    print("=" * 50)
    
    print(f"\nSupported Datasets: {Config.DATASET_TYPES}")
    print(f"Temperature Modes: {Config.TEMPERATURE_MODES}")
    print(f"Generation Methods: {Config.GENERATION_METHODS}")
    
    print(f"\nTemperature Settings:")
    print(f"  Low-T:  temp={Config.LOW_T_PARAMS['temperature']}, top_p={Config.LOW_T_PARAMS['top_p']}")
    print(f"  High-T: temp={Config.HIGH_T_PARAMS['temperature']}, top_p={Config.HIGH_T_PARAMS['top_p']}")
    
    print(f"\nModel Parameters:")
    for model in Config.QWEN_MODELS:
        params = Config.get_model_params(model, "low_T")
        print(f"\n{model}:")
        print(f"  Context: {params['context_length']:,} tokens")
        print(f"  Max output: {params['num_predict']:,} tokens")
        print(f"  Timeout: {params['timeout']}s")
    
    print(f"\nEvaluation Settings:")
    print(f"  Samples per design: {Config.N_SAMPLES}")
    print(f"  Pass@k metrics: {Config.K_VALUES}")
    print(f"  Overwrite existing: {Config.OVERWRITE_EXISTING}")
    
    print(f"\nPrescreening Settings:")
    print(f"  Enabled: {Config.ENABLE_PRESCREENING}")
    if Config.ENABLE_PRESCREENING:
        print(f"  Timeout: {Config.PRESCREENING_TIMEOUT}s")
        print(f"  Requirement: Must pass both syntax AND simulation")
    
    print(f"\nC++ Validation Settings:")
    print(f"  Enabled: {Config.ENABLE_CPP_VALIDATION}")
    if Config.ENABLE_CPP_VALIDATION:
        print(f"  Mode: {Config.CPP_VALIDATION_MODE}")
        print(f"  Max iterations: {Config.MAX_CPP_REFINEMENT_ITERATIONS}")
    
    print(f"\nRefinement Settings:")
    print(f"  Enabled: {Config.ENABLE_ITERATIVE_REFINEMENT}")
    if Config.ENABLE_ITERATIVE_REFINEMENT:
        print(f"  Max iterations: {Config.MAX_REFINEMENT_ITERATIONS}")
    
    # Show dataset statistics
    all_designs = load_all_designs()
    print(f"\nDataset Statistics:")
    for dataset_name, designs in all_designs.items():
        print(f"  {dataset_name.upper()}: {len(designs)} designs")
        if designs:
            print(f"    Sample: {[d['name'] for d in designs[:3]]}")

def main_cli():
    """Main CLI interface"""
    # Parse prescreening flags first
    if "--prescreening" in sys.argv or "--prescreen" in sys.argv:
        Config.ENABLE_PRESCREENING = True
        print("✓ Prescreening enabled (syntax + simulation required)")
        sys.argv = [arg for arg in sys.argv if arg not in ["--prescreening", "--prescreen"]]
    
    if "--no-prescreening" in sys.argv or "--no-prescreen" in sys.argv:
        Config.ENABLE_PRESCREENING = False
        sys.argv = [arg for arg in sys.argv if arg not in ["--no-prescreening", "--no-prescreen"]]
    
    # Parse C++ validation flags
    if "--cpp-validation" in sys.argv or "--cpp-val" in sys.argv:
        Config.ENABLE_CPP_VALIDATION = True
        print("✓ C++ validation enabled")
        sys.argv = [arg for arg in sys.argv if arg not in ["--cpp-validation", "--cpp-val"]]
    
    if "--no-cpp-validation" in sys.argv:
        Config.ENABLE_CPP_VALIDATION = False
        sys.argv.remove("--no-cpp-validation")
    
    # Parse C++ validation mode
    for arg in sys.argv[:]:
        if arg.startswith("--cpp-val-mode="):
            mode = arg.split("=")[1]
            if mode in ["always", "on_failure", "never"]:
                Config.CPP_VALIDATION_MODE = mode
                print(f"✓ C++ validation mode: {mode}")
            else:
                print(f"Invalid C++ validation mode: {mode}")
            sys.argv.remove(arg)
            break
    
    # Parse C++ refinement iterations
    for arg in sys.argv[:]:
        if arg.startswith("--cpp-refine-iter="):
            try:
                iterations = int(arg.split("=")[1])
                Config.MAX_CPP_REFINEMENT_ITERATIONS = iterations
                print(f"✓ C++ refinement iterations: {iterations}")
            except:
                print("Invalid C++ iteration count")
            sys.argv.remove(arg)
            break
    
    # Parse refinement flags
    if "--no-refine" in sys.argv:
        Config.ENABLE_ITERATIVE_REFINEMENT = False
        print("✓ Refinement disabled")
        sys.argv.remove("--no-refine")
    else:
        # Check for custom iteration count
        for i, arg in enumerate(sys.argv):
            if arg.startswith("--refine-iter="):
                try:
                    iterations = int(arg.split("=")[1])
                    Config.MAX_REFINEMENT_ITERATIONS = iterations
                    Config.ENABLE_ITERATIVE_REFINEMENT = True
                    print(f"✓ Refinement enabled with {iterations} iteration(s)")
                    sys.argv.pop(i)
                    break
                except:
                    print("  Invalid iteration count, using default")
                    sys.argv.pop(i)
                    break
        
        # Simple --refine flag
        if "--refine" in sys.argv:
            Config.ENABLE_ITERATIVE_REFINEMENT = True
            print(f"✓ Refinement enabled (default {Config.MAX_REFINEMENT_ITERATIONS} iterations)")
            sys.argv.remove("--refine")
    
    # Parse overwrite flag
    if "--overwrite" in sys.argv:
        Config.OVERWRITE_EXISTING = True
        print("✓ Overwrite mode enabled")
        sys.argv.remove("--overwrite")
    
    # Parse method flag
    method = "direct"  # default
    if "--cpp-chain" in sys.argv:
        method = "cpp_chain"
        print("✓ C++ chain method enabled")
        sys.argv.remove("--cpp-chain")
    
    # Parse dataset flag
    dataset = "rtllm"  # default
    for arg in sys.argv[:]:
        if arg.startswith("--dataset="):
            dataset = arg.split("=")[1]
            if dataset not in Config.DATASET_TYPES:
                print(f"Error: Invalid dataset '{dataset}'. Supported: {Config.DATASET_TYPES}")
                return
            print(f"✓ Dataset: {dataset}")
            sys.argv.remove(arg)
            break
    
    # Parse temperature flag
    temp_mode = "low_T"  # default
    for arg in sys.argv[:]:
        if arg.startswith("--temp="):
            temp_mode = arg.split("=")[1]
            if temp_mode not in Config.TEMPERATURE_MODES:
                print(f"Error: Invalid temperature mode '{temp_mode}'. Supported: {Config.TEMPERATURE_MODES}")
                return
            print(f"✓ Temperature: {temp_mode}")
            sys.argv.remove(arg)
            break
    
    # Alternative temperature flags
    if "--low-T" in sys.argv:
        temp_mode = "low_T"
        print("✓ Low temperature mode")
        sys.argv.remove("--low-T")
    elif "--high-T" in sys.argv:
        temp_mode = "high_T"
        print("✓ High temperature mode")
        sys.argv.remove("--high-T")
    
    if len(sys.argv) < 2:
        print("Multi-Dataset Qwen2.5 Benchmark")
        print("Usage:")
        print("  python run.py all                           # Test all models on RTLLM with low-T")
        print("  python run.py all --dataset=verilogeval     # Test all models on VerilogEval")
        print("  python run.py all --temp=high_T             # Test all models with high temperature")
        print("  python run.py all --low-T --cpp-chain       # Test all with low-T and C++ chain")
        print("  python run.py all --high-T --refine         # Test all with high-T and refinement")
        print("  python run.py all --cpp-chain --prescreening # Test with C++ chain and prescreening")
        print("  python run.py all --cpp-chain --cpp-validation # Test with C++ chain and validation")
        print("  python run.py comprehensive                 # Test all datasets and temperatures")
        print("  python run.py check                         # Check setup")
        print("  python run.py list                          # List available models") 
        print("  python run.py info                          # Show configuration")
        print("  python run.py <model>                       # Test specific model")
        print("\nDataset Options:")
        print(f"  --dataset=rtllm         # RTLLM dataset (Verilog .v files)")
        print(f"  --dataset=verilogeval   # VerilogEval dataset (SystemVerilog .sv files)")
        print("\nTemperature Options:")
        print(f"  --low-T or --temp=low_T   # Low temperature (T=0.0, top_p=0.01)")
        print(f"  --high-T or --temp=high_T # High temperature (T=0.8, top_p=0.95)")
        print("\nGeneration Options:")
        print("  --cpp-chain             # Use C++ chain method")
        print("  --prescreening          # Enable prescreening (direct method pre-check)")
        print("  --no-prescreening       # Disable prescreening")
        print("\nC++ Validation Options (for cpp-chain method):")
        print("  --cpp-validation        # Enable C++ validation")
        print("  --cpp-val-mode=MODE     # Validation mode: always|on_failure|never")
        print("  --cpp-refine-iter=N     # C++ refinement iterations (default: 2)")
        print("  --no-cpp-validation     # Disable C++ validation")
        print("\nRefinement Options:")
        print("  --refine                # Enable refinement")
        print("  --refine-iter=N         # Custom refinement iterations")
        print("  --no-refine             # Disable refinement")
        print("\nOther Options:")
        print("  --overwrite             # Overwrite existing files")
        print("\nExamples:")
        print("  python run.py qwen2.5:14b --dataset=verilogeval --high-T")
        print("  python run.py qwen2.5:7b --cpp-chain --refine --overwrite")
        print("  python run.py all --dataset=rtllm --low-T --refine-iter=5")
        print("  python run.py all --cpp-chain --prescreening --refine")
        print("  python run.py all --cpp-chain --cpp-validation --cpp-val-mode=always")
        print("  python run.py all --dataset=verilogeval --cpp-chain --cpp-validation")
        print("  python run.py all --dataset=verilogeval --high-T --cpp-chain --prescreening --cpp-validation --cpp-val-mode=on_failure --cpp-refine-iter=3 --refine-iter=3")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "all":
        main(method, dataset, temp_mode)
    elif cmd == "comprehensive":
        run_all_combinations()
    elif cmd == "check":
        if print_setup_check():
            print("✓ Ready to test Qwen2.5 models on multiple datasets!")
        else:
            print("Setup needs attention")
    elif cmd == "list":
        list_models()
    elif cmd == "info":
        show_info()
    else:
        # Single model test
        run_single_model(cmd, method, dataset, temp_mode)

if __name__ == "__main__":
    main_cli()