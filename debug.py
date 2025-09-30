#!/usr/bin/env python3
"""
Debug script to test LLM interface
"""

import sys
from pathlib import Path
from llm_interface import OllamaInterface
from config import Config
from utils import load_designs

def test_simple_prompt():
    """Test with a very simple prompt"""
    model = "qwen3:8b"
    print(f"Testing {model} with simple prompt")
    
    llm = OllamaInterface(model)
    
    simple_prompt = "Write a simple Verilog module that adds two 4-bit numbers:\n\nmodule adder_4bit(\ninput [3:0] a,\ninput [3:0] b,\noutput [3:0] sum\n);\n\n// Your code here\n\nendmodule"
    
    print("Sending prompt...")
    response = llm.generate_response(simple_prompt)
    
    if response:
        print("✓ Got response!")
        print(f"Response length: {len(response)}")
        print("FULL RESPONSE:")
        print("=" * 60)
        print(response)
        print("=" * 60)
        
        # Try extracting Verilog
        verilog = llm.extract_verilog(response)
        if verilog:
            print("\n✓ Extracted Verilog!")
            print("Verilog code:")
            print("-" * 40)
            print(verilog)
            print("-" * 40)
        else:
            print("\n✗ Failed to extract Verilog")
            print("DEBUG: Looking for module/endmodule...")
            print(f"Contains 'module ': {'module ' in response}")
            print(f"Contains 'endmodule': {'endmodule' in response}")
            print(f"Contains '<think>': {'<think>' in response}")
            print(f"Contains '</think>': {'</think>' in response}")
    else:
        print("✗ No response from model")

def test_actual_design():
    """Test with actual RTLLM design"""
    designs = load_designs()
    if not designs:
        print("No designs available for testing")
        return
    
    # Test with calendar design (the one that's failing)
    calendar_design = None
    for design in designs:
        if design["name"] == "calendar":
            calendar_design = design
            break
    
    if not calendar_design:
        calendar_design = designs[0]  # Use first design if calendar not found
    
    design = calendar_design
    print(f"Testing with actual design: {design['name']}")
    
    # Read the actual description
    try:
        with open(design["description"], 'r') as f:
            description = f.read().strip()
        
        print(f"Description length: {len(description)}")
        print("First 500 chars of description:")
        print("-" * 40)
        print(description[:500])
        print("-" * 40)
        
        # Test generation
        model = "qwen3:8b"
        llm = OllamaInterface(model)
        
        print(f"\nTesting generation with {model}...")
        response = llm.generate_response(description)
        
        if response:
            print("✓ Got response!")
            print(f"Response length: {len(response)}")
            
            print("\nFULL RESPONSE:")
            print("=" * 60)
            print(response)
            print("=" * 60)
            
            verilog = llm.extract_verilog(response)
            if verilog:
                print("✓ Successfully extracted Verilog")
                print("EXTRACTED VERILOG:")
                print("-" * 40)
                print(verilog)
                print("-" * 40)
            else:
                print("✗ Failed to extract Verilog from response")
                print("DEBUG INFO:")
                print(f"Contains 'module ': {'module ' in response}")
                print(f"Contains 'endmodule': {'endmodule' in response}")
                print(f"Contains '<think>': {'<think>' in response}")
                print(f"Contains '</think>': {'</think>' in response}")
        else:
            print("✗ No response from model")
            
    except Exception as e:
        print(f"Error reading design description: {e}")

def test_ollama_direct():
    """Test Ollama directly with correct API"""
    import subprocess
    import json
    
    print("Testing Ollama directly...")
    
    try:
        # Test basic connection
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        print(f"Ollama list result: {result.returncode}")
        
        # Test with correct ollama run command
        print("Testing with ollama run...")
        result = subprocess.run(
            ["ollama", "run", "qwen3:8b", "Write a simple hello world in Verilog"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"Run result code: {result.returncode}")
        if result.returncode == 0:
            print(f"Response length: {len(result.stdout)}")
            if result.stdout:
                print("First 200 chars:")
                print(result.stdout[:200])
        else:
            print(f"Error: {result.stderr}")
            
        # Also test API endpoint if available
        print("\nTesting API endpoint...")
        api_request = {
            "model": "qwen3:8b", 
            "prompt": "Write hello world in Verilog",
            "stream": False
        }
        
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=api_request,
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                print(f"API Response length: {len(data.get('response', ''))}")
                print("API works!")
            else:
                print(f"API Error: {response.status_code}")
        except ImportError:
            print("requests not available, skipping API test")
        except Exception as e:
            print(f"API test failed: {e}")
            
    except Exception as e:
        print(f"Direct test error: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python debug.py simple     # Test simple prompt")
        print("  python debug.py design     # Test actual design")
        print("  python debug.py direct     # Test Ollama directly")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "simple":
        test_simple_prompt()
    elif cmd == "design":
        test_actual_design()
    elif cmd == "direct":
        test_ollama_direct()
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()