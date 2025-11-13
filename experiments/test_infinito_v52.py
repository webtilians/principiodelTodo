"""
Test script for Infinito V5.2 refactored model.

This script initializes the Infinito V5.2 refactored model with default parameters, generates random input and target tensors, performs a forward pass with return_metrics=True, prints metrics and calculates perplexity.
"""

import torch
from src.infinito_v5_2_refactored import InfinitoV52Refactored


def main():
    # Create model instance
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
    
    # Generate sample inputs
    batch_size, seq_len = 4, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    target_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    model.eval()
    
    # Forward pass and metrics
    with torch.no_grad():
        logits, metrics = model(input_ids, return_metrics=True)
        print("Metrics:", metrics)
        perplexity = model.calculate_perplexity(input_ids, target_ids)
        print("Perplexity:", perplexity)


if __name__ == "__main__":
    main()
