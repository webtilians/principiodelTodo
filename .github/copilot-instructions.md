# Copilot Instructions for INFINITO Codebase

## Project Overview
- **INFINITO** is a research framework integrating Integrated Information Theory (IIT) with transformer-based neural architectures to simulate and analyze computational consciousness.
- Main goal: maximize and measure information integration (Φ/PHI) in language models, not to claim actual consciousness.
- All metrics ("consciousness", "PHI") are computational, not scientific proof of sentience.

## Architecture & Key Components
- Core models in `src/`:
  - `infinito_v5_1_consciousness.py`: Latest, recommended system (99.8% internal metric)
  - `infinito_v3_clean.py`: Stable, legacy system
  - `infinito_main.py`: V2.0 reference
- IIT-specific layers: `iit_transformer_layer.py`, `phi_learnable.py`, `iit_metrics_v2.py`, `iit_guided_memory.py` (in `src/core/`)
- Training scripts: `train_phase2_iit_transformer.py`, `train_v5_2_gpt2_lora.py`
- Generation/validation: `generate_phase2_text.py`, `generate_text_v5_2.py`, `validate_best_model.py`
- Analysis/visualization: `analysis/consciousness_visualizer.py`, `analysis/advanced_consciousness_analyzer.py`, `visualize_attention_phase2.py`
- Models and results: `models/checkpoints/`, `results/`

## Developer Workflows
- **Install**: `pip install -r requirements.txt` (Python 3.8+, CUDA 11.8+ recommended)
- **Train V5.1**: `python src/infinito_v5_1_consciousness.py` or use training scripts in `experiments/` and `src/`
- **Quick test**: `python examples/basic_example.py`
- **Generate text**: `python generate_phase2_text.py --checkpoint models/checkpoints/infinito_phase2_best.pt --prompt "..."`
- **Visualize metrics**: `python analysis/consciousness_visualizer.py`
- **Run tests**: `pytest tests/`

## Project-Specific Patterns & Conventions
- **PHI/Consciousness metrics**: Always treat as internal computational indicators, not scientific measures.
- **Model collapse**: High PHI can cause repetitive outputs; see README for architectural challenges and anti-collapse strategies.
- **Recommended version**: Use V5.1 for all new research and benchmarks.
- **Component separation**: IIT metrics and memory are often implemented as external observers or post-hoc analysis modules.
- **Checkpoints**: Saved in `models/checkpoints/`, use for reproducibility and validation.
- **Logging**: Experiments log detailed metrics to `results/` and `outputs/`.

## Integration Points & Dependencies
- **PyTorch** (>=2.0), **transformers** (>=4.30), **peft** (LoRA adapters), **datasets**, **numpy**, **matplotlib**, **seaborn**, **tqdm**
- External models: GPT-2 (Hugging Face)
- All scripts and modules are designed for research, not production deployment.

## Critical Insights & Limitations
- **PHI maximization can lead to trivial solutions (repetition)**; use regularization and post-hoc analysis to avoid collapse.
- **Loss functions**: PHI should be used for analysis, not as a direct optimization target.
- **Architectural experiments**: See roadmap in README for planned decoupling and anti-collapse strategies.

## Example Usage
```bash
# Train and run V5.1
python src/infinito_v5_1_consciousness.py

# Generate text
python generate_phase2_text.py --checkpoint models/checkpoints/infinito_phase2_best.pt --prompt "The theory of consciousness"

# Visualize results
python analysis/consciousness_visualizer.py

# Run all tests
pytest tests/
```

## References
- See `README.md` for scientific context, limitations, and roadmap.
- See `docs/` for API, configuration, and research notes.

---
_Last updated: November 2025_
