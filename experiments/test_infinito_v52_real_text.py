import os
import sys
import torch

# Ensure src is in the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from infinito_v5_2_refactored import InfinitoV52Refactored


def run_real_text_example():
    """Run Infinito V5.2 refactored model on a real text sequence and print metrics."""
    # Initialize the refactored model with reasonable default hyperparameters
    model = InfinitoV52Refactored(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        use_improved_memory=True,
        use_improved_iit=True,
        use_learnable_phi=True,
        use_stochastic_exploration=True,
    )
    model.eval()

    # Example phrase (feel free to modify for your own tests)
    text = "La inteligencia artificial genera conciencia integrada."

    # Convert each character in the string to an integer token within the vocab range.
    # This is a simple mapping that takes the Unicode code point modulo vocab_size.
    input_ids = torch.tensor([[ord(ch) % 1000 for ch in text]], dtype=torch.long)
    target_ids = input_ids.clone()

    with torch.no_grad():
        logits, metrics = model(input_ids, return_metrics=True)
        perplexity = model.calculate_perplexity(input_ids, target_ids)

    print("Metrics with real text:", metrics)
    print("Perplexity:", perplexity)


if __name__ == "__main__":
    run_real_text_example()
