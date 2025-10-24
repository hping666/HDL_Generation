# MoA-Enhanced HDL Code Generation

A Mixture-of-Agents (MoA) framework for high-quality Hardware Description Language (HDL) code generation with quality-based caching, early stopping, and self-refinement capabilities.

## Features

- **Quality-Based Caching**: Evaluates and caches intermediate HDL codes, selecting top-quality candidates for subsequent layers
- **Early Stopping**: Terminates generation when perfect HDL (score=1.0) is found
- **Self-Refinement**: Iteratively fixes HDL errors using iverilog feedback
- **Multi-Path Generation** (MoA_HLS): Generates HDL through configurable paths (direct, C++→HDL, Python→HDL)
- **Dual Dataset Support**: RTLLM (Verilog) and VerilogEval (SystemVerilog)

## Project Structure

```
├── MoA_verify.py           # Standard MoA with quality caching
├── MoA_HLS.py              # Multi-path MoA with HLS intermediate generation
├── quality_evaluator.py    # HDL quality evaluation with error details
├── cache_manager.py        # Cache management for intermediate HDL
├── llm_interface.py        # LLM interface (Ollama & OpenAI)
├── hdl_tester_enhanced.py  # HDL testing with iverilog/vvp
├── config.py               # Configuration settings
└── utils.py                # Utility functions
```

## Quick Start

### Prerequisites

```bash
# Install iverilog (for HDL testing)
pip install iverilog 

# Setup LLM backend (choose one)
# Option 1: Ollama (local)
ollama pull qwen2.5-coder:7b

# Option 2: OpenAI API
export OPENAI_API_KEY="your-api-key"
```

### Basic Usage

#### MoA_verify.py (Residual MoA)

```bash
# Basic run with quality caching and self-refinement
python MoA_verify.py --models=qwen2.5-coder:7b,qwen2.5-coder:7b,qwen2.5-coder:7b \
                     --aggregator=qwen2.5-coder:7b \
                     --layers=3 \
                     --dataset=rtllm \
                     --temp=high_T \
                     --quality_cache \
                     --self_refine \
                     --max_refine_iters=3
```

#### MoA_HLS.py (Multi-path Residual MoA)

```bash
# Multi-path generation with C++, Python, and direct paths
python MoA_HLS.py --model=gpt-4o-mini \
                  --layers=4 \
                  --dataset=rtllm \
                  --temp=high_T \
                  --paths=direct,cpp,python \
                  --n_select=3 \
                  --self_refine \
                  --max_refine_iters=3
```

## Parameter Guide

### Common Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--dataset` | `rtllm`, `verilogeval` | `rtllm` | Target dataset |
| `--temp` | `low_T`, `high_T` | `high_T` | Temperature mode (low=0.0, high=0.8) |
| `--quality_cache` | flag | enabled | Enable quality-based caching |
| `--early_stop` | flag | disabled | Stop when perfect HDL found |
| `--self_refine` | flag | **enabled** | Enable self-refinement |
| `--no_self_refine` | flag | - | Disable self-refinement |
| `--max_refine_iters` | 1-10 | `3` | Maximum refinement iterations |

### MoA_verify.py Specific Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--layers` | 0-10 | `2` | Number of MoA layers (0=direct) |
| `--models` | comma-separated | `qwen2.5-coder:7b` (×3) | Models for each layer |
| `--aggregator` | model name | `qwen2.5-coder:7b` | Final aggregator model |

**Example Configurations:**

```bash
# Homogeneous 3-layer MoA
python MoA_verify.py --layers=3 \
                     --models=qwen2.5-coder:7b,qwen2.5-coder:7b,qwen2.5-coder:7b

# Heterogeneous MoA with different models
python MoA_verify.py --layers=2 \
                     --models=gpt-4o-mini,qwen2.5-coder:14b \
                     --aggregator=gpt-4o

# Direct generation (no MoA)
python MoA_verify.py --layers=0 --aggregator=gpt-4o-mini
```

### MoA_HLS.py Specific Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--model` | model name | `gpt-4o-mini` | Single model for all paths |
| `--layers` | 1-10 | `4` | Number of MoA layers |
| `--paths` | comma-separated | `direct,cpp,python` | Generation paths per layer |
| `--n_select` | 1-10 | `3` | Top-n selection per layer |

**Path Configuration:**

| Configuration | Description | Use Case |
|---------------|-------------|----------|
| `direct,cpp,python` | One of each path | Balanced approach |
| `cpp,cpp,cpp` | All C++ paths | C++ expertise emphasis |
| `python,python` | All Python paths | Python-centric generation |
| `direct,cpp` | Mix direct + C++ | Faster generation |

**Example Configurations:**

```bash
# Balanced multi-path (default)
python MoA_HLS.py --paths=direct,cpp,python

# C++-focused generation
python MoA_HLS.py --paths=cpp,cpp,cpp,cpp --layers=4

# Python-only paths
python MoA_HLS.py --paths=python,python --layers=2

# Direct + C++ hybrid
python MoA_HLS.py --paths=direct,cpp,direct,cpp --layers=4
```

## Feature Configuration

### Quality-Based Caching

**Enable** (recommended):
```bash
--quality_cache --n_select=3
```

**Effect:**
- Evaluates HDL quality using iverilog
- Selects top-n codes for next layer
- Improves output quality progressively
- **Required for self-refinement**

**Disable:**
```bash
--no_cache
```

### Early Stopping

**Enable:**
```bash
--early_stop
```

**Effect:**
- Stops generation when HDL achieves perfect score (1.0)
- Reduces computation time
- Prevents unnecessary iterations
- Works with both quality caching and self-refinement

**When to use:**
- Time-sensitive scenarios
- When perfect HDL is achievable
- With high-quality base models

### Self-Refinement

**Enable** (default):
```bash
--self_refine --max_refine_iters=3
```

**Effect:**
- Iteratively fixes syntax errors, compilation errors, and simulation failures
- Uses iverilog feedback for targeted fixes
- Applies to both intermediate and final HDL
- Refined HDL updates cache and influences subsequent layers

**Disable:**
```bash
--no_self_refine
```

**When to use:**
- When generated HDL has errors
- To maximize pass rate
- With quality caching enabled

**Iteration Settings:**

| Iterations | Use Case | Trade-off |
|------------|----------|-----------|
| 1-2 | Fast refinement | May miss complex errors |
| 3 (default) | Balanced | Good quality/speed ratio |
| 4-5 | Thorough fixing | Slower, diminishing returns |

### Configuration Combinations

#### Maximum Quality (Slowest)
```bash
python MoA_HLS.py --layers=5 \
                  --paths=cpp,cpp,python,python,direct \
                  --quality_cache --n_select=3 \
                  --self_refine --max_refine_iters=3
```

#### Balanced (Recommended)
```bash
python MoA_HLS.py --layers=4 \
                  --paths=direct,cpp,python \
                  --quality_cache --n_select=3 \
                  --early_stop \
                  --self_refine --max_refine_iters=3
```

#### Fast Generation
```bash
python MoA_verify.py --layers=2 \
                     --models=gpt-4o-mini,gpt-4o-mini \
                     --early_stop \
                     --self_refine --max_refine_iters=2
```

#### No Enhancement (Baseline)
```bash
python MoA_verify.py --layers=3 \
                     --no_cache \
                     --no_self_refine
```

## Output Structure

### Folder Naming Convention

**MoA_verify.py:**
```
MoA_{temp}_{L#}_{models}_AGG_{aggregator}[_QualityCache][_EarlyStop][_SelfRef#]
```

**MoA_HLS.py:**
```
MoA_HLS_{temp}_L{#}_{model}_paths_{config}_QCache_N{#}[_EarlyStop][_SelfRef#]
```

**Examples:**
```
MoA_high_T_L3_qwen2_5-coder-7b_qwen2_5-coder-7b_qwen2_5-coder-7b_AGG_qwen2_5-coder-7b_QualityCache_EarlyStop_SelfRef3

MoA_HLS_high_T_L4_gpt-4o-mini_paths_direct_cpp_python_QCache_N3_EarlyStop_SelfRef3
```

### Directory Structure

```
./verilog/MoA[_HLS]/<folder_name>/
├── t1/
│   ├── design1.v
│   ├── design2.v
│   └── ...
├── t2/
├── ...
├── t10/
└── generation_summary.json

./result/MoA[_HLS]/<folder_name>/
├── results.json
└── detailed_results.json

./verilog_temp/MoA[_HLS]/<folder_name>/  (if quality caching enabled)
├── t1/
│   ├── design1_cache.json
│   └── ...
└── cache_analysis.json
```

## Self-Refinement Details

### How It Works

1. **Initial Generation**: LLM generates HDL code
2. **Quality Evaluation**: Runs iverilog syntax and functional tests
3. **Error Detection**: Identifies error type (syntax/compilation/simulation)
4. **Refinement Prompt**: Generates targeted fix prompt with error details
5. **Iterative Fix**: LLM attempts to fix errors (max 3 iterations)
6. **Cache Update**: Stores best refined HDL with updated quality score

### Error Type Handling

| Error Type | Detection | Fix Strategy |
|------------|-----------|--------------|
| **Syntax Error** | `iverilog` compilation failure | Fix variable redeclaration, part select issues |
| **Compilation Error** | Missing modules, name mismatch | Implement inline logic, fix module names |
| **Simulation Fail** | `vvp` test failure | Fix logic errors, timing issues |

### MoA_HLS Refinement Specifics

- **Direct path**: Standard refinement with error feedback
- **C++/Python paths**: Refinement includes intermediate code reference
- **Intermediate code quality**: Uses **original** HDL quality (before refinement)
- **Cache behavior**: Refined HDL quality used for top-n selection

### Refinement Process Example

```
Layer 1:
├─ Model A: HDL (quality=0.6) 
│   └─ Refinement Iter 1: Fix syntax → quality=0.7
│   └─ Refinement Iter 2: Fix logic → quality=0.85
│   └─ Store in cache: quality=0.85
│
├─ Model B: HDL (quality=1.0) → Perfect, no refinement needed
│   └─ Store in cache: quality=1.0
│
└─ Model C: HDL (quality=0.5)
    └─ Refinement Iter 1: Still fails → quality=0.6
    └─ Refinement Iter 2: Improved → quality=0.75
    └─ Store in cache: quality=0.75

Layer 2 input: Top-3 from cache [1.0, 0.85, 0.75]
```

## Performance Tips

1. **Use quality caching**: Essential for self-refinement and better results
2. **Enable early stopping**: Saves time when perfect HDL is found
3. **Tune refinement iterations**: 3 is optimal for most cases
4. **Choose appropriate paths** (MoA_HLS):
   - C++ paths: Better for algorithmic designs
   - Python paths: Good for data processing logic
   - Direct paths: Faster, suitable for simple designs
5. **Model selection**:
   - GPT-4o-mini: Fast, cost-effective
   - Qwen2.5-coder: Strong coding capability, local deployment
   - Mix models: Leverage diverse strengths

## Troubleshooting

**Issue**: Self-refinement not working
- **Solution**: Ensure `--quality_cache` is enabled

**Issue**: Slow generation
- **Solution**: Reduce `--layers`, use `--early_stop`, or decrease `--max_refine_iters`

**Issue**: Low pass rate
- **Solution**: Enable `--self_refine`, increase `--max_refine_iters`, or use better base models

**Issue**: Out of memory
- **Solution**: Reduce `--n_select` or `--layers`

