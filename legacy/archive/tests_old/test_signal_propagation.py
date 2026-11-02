"""
🔬 ANÁLISIS PROFUNDO: Propagación de la señal semántica en la arquitectura
==========================================================================

Objetivo: Rastrear cómo el embedding semántico se propaga (o se pierde) a través de:
1. Input inicial
2. Forward pass (capas de la red)
3. Loss computation
4. Backward pass (gradientes)
5. Weight updates

Método: Usar hooks de PyTorch para capturar activaciones en cada capa
"""

import torch
import torch.nn as nn
import sys
sys.path.append('src')

from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough, SemanticTextEmbedder
from argparse import Namespace
import numpy as np

class ActivationCapture:
    """Captura activaciones de capas específicas"""
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def register_hook(self, module, name):
        """Registra un hook para capturar la salida de un módulo"""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().clone()
            elif isinstance(output, tuple):
                self.activations[name] = output[0].detach().clone() if len(output) > 0 else None
        
        hook = module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def clear(self):
        """Limpia activaciones capturadas"""
        self.activations = {}
    
    def remove_hooks(self):
        """Remueve todos los hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def analyze_signal_propagation():
    """
    Analiza cómo se propaga la señal semántica a través de la arquitectura
    """
    
    print("="*80)
    print("🔬 ANÁLISIS DE PROPAGACIÓN DE SEÑAL SEMÁNTICA")
    print("="*80)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    # Textos con máxima diferencia semántica
    textos = [
        "mi perro es rojo",      # Animal + color cálido
        "yo pienso, luego existo"  # Filosofía pura
    ]
    
    # Primero verificar que los embeddings son diferentes
    print("🔍 PASO 1: Verificar embeddings semánticos iniciales")
    print("-"*80)
    
    embedder = SemanticTextEmbedder(embed_dim=256, use_glove=False)
    
    emb1 = embedder.text_to_tensor(textos[0], device)
    emb2 = embedder.text_to_tensor(textos[1], device)
    
    diff_embedding = (emb1 - emb2).norm().item()
    
    print(f"\n📝 '{textos[0]}'")
    print(f"   Embedding norm: {emb1.norm().item():.6f}")
    print(f"   Embedding mean: {emb1.mean().item():.6f}")
    
    print(f"\n📝 '{textos[1]}'")
    print(f"   Embedding norm: {emb2.norm().item():.6f}")
    print(f"   Embedding mean: {emb2.mean().item():.6f}")
    
    print(f"\n📊 Diferencia entre embeddings: {diff_embedding:.6f}")
    
    if diff_embedding < 0.1:
        print("   ❌ Embeddings muy similares, análisis puede no ser concluyente")
        return
    else:
        print("   ✅ Embeddings suficientemente diferentes")
    
    # Crear modelo y registrar hooks
    print("\n" + "="*80)
    print("🔍 PASO 2: Rastrear propagación a través de la red")
    print("="*80)
    
    args = Namespace(
        batch_size=4,
        input_dim=256,
        hidden_dim=512,
        attention_heads=8,
        memory_slots=256,
        lr=1e-3,
        text_mode=True,
        input_text=None,
        quantum_active=False,
        target_consciousness=0.6,
        max_consciousness=0.9
    )
    
    results = {}
    
    for texto in textos:
        print(f"\n\n{'='*80}")
        print(f"📝 Procesando: '{texto}'")
        print('='*80)
        
        # Crear modelo fresh para cada texto
        model = InfinitoV51ConsciousnessBreakthrough(args)
        
        # Acceder al modelo interno (puede estar wrapped en DataParallel)
        net = model.model.module if hasattr(model.model, 'module') else model.model
        
        # Registrar hooks en capas clave
        capture = ActivationCapture()
        
        print("\n🎣 Registrando hooks en capas clave...")
        
        # Hook en las primeras capas de procesamiento
        if hasattr(net, 'visual_module') and hasattr(net.visual_module, 'dense_layers'):
            capture.register_hook(net.visual_module.dense_layers[0], 'visual_layer0')
            if len(net.visual_module.dense_layers) > 1:
                capture.register_hook(net.visual_module.dense_layers[-1], 'visual_layer_final')
        
        if hasattr(net, 'executive_module') and hasattr(net.executive_module, 'dense_layers'):
            capture.register_hook(net.executive_module.dense_layers[0], 'executive_layer0')
            if len(net.executive_module.dense_layers) > 1:
                capture.register_hook(net.executive_module.dense_layers[-1], 'executive_layer_final')
        
        # Hook en integración consciente
        if hasattr(net, 'conscious_integration'):
            capture.register_hook(net.conscious_integration, 'conscious_integration')
        
        # Hook en memoria
        if hasattr(net, 'memory'):
            capture.register_hook(net.memory, 'memory_output')
        
        print(f"   ✅ {len(capture.hooks)} hooks registrados")
        
        # Capturar input inicial
        with torch.no_grad():
            input_tensor = model.generate_text_based_input(texto)
            
            print(f"\n📥 Input inicial:")
            print(f"   Shape: {input_tensor.shape}")
            print(f"   Norm: {input_tensor.norm().item():.6f}")
            print(f"   Mean: {input_tensor.mean().item():.6f}")
            print(f"   Std: {input_tensor.std().item():.6f}")
        
        # Ejecutar UNA iteración con captura de activaciones
        print(f"\n🔄 Ejecutando forward pass...")
        
        model.optimizer.zero_grad()
        
        # Forward pass (sin train_step para control fino)
        with torch.set_grad_enabled(True):
            try:
                # Forward a través del modelo
                output = net(input_tensor)
                
                print(f"\n📤 Output de la red:")
                print(f"   Shape: {output.shape}")
                print(f"   Norm: {output.norm().item():.6f}")
                print(f"   Mean: {output.mean().item():.6f}")
                print(f"   Std: {output.std().item():.6f}")
                
            except Exception as e:
                print(f"   ⚠️ Error en forward pass: {e}")
                capture.remove_hooks()
                continue
        
        # Analizar activaciones capturadas
        print(f"\n📊 Activaciones capturadas en capas intermedias:")
        print("-"*80)
        
        layer_stats = {}
        
        for layer_name, activation in capture.activations.items():
            if activation is not None:
                norm = activation.norm().item()
                mean = activation.mean().item()
                std = activation.std().item()
                
                layer_stats[layer_name] = {
                    'norm': norm,
                    'mean': mean,
                    'std': std,
                    'shape': activation.shape
                }
                
                print(f"\n   {layer_name}:")
                print(f"      Shape: {activation.shape}")
                print(f"      Norm:  {norm:.6f}")
                print(f"      Mean:  {mean:.6f}")
                print(f"      Std:   {std:.6f}")
        
        # Guardar resultados
        results[texto] = {
            'input_norm': input_tensor.norm().item(),
            'input_mean': input_tensor.mean().item(),
            'output_norm': output.norm().item() if output is not None else 0,
            'output_mean': output.mean().item() if output is not None else 0,
            'layer_stats': layer_stats,
            'embedding_raw': embedder.text_to_tensor(texto, device)
        }
        
        capture.remove_hooks()
    
    # Análisis comparativo
    print("\n\n" + "="*80)
    print("📊 ANÁLISIS COMPARATIVO ENTRE TEXTOS")
    print("="*80)
    
    texto1, texto2 = textos[0], textos[1]
    
    print(f"\n🔴 INPUT (embeddings crudos):")
    emb_diff = (results[texto1]['embedding_raw'] - results[texto2]['embedding_raw']).norm().item()
    print(f"   L2 distance: {emb_diff:.6f}")
    
    print(f"\n🟡 INPUT (post-transform en generate_text_based_input):")
    input_diff = abs(results[texto1]['input_norm'] - results[texto2]['input_norm'])
    print(f"   Norm difference: {input_diff:.6f}")
    print(f"   Texto 1 norm: {results[texto1]['input_norm']:.6f}")
    print(f"   Texto 2 norm: {results[texto2]['input_norm']:.6f}")
    
    print(f"\n🟢 OUTPUT (salida final de la red):")
    output_diff = abs(results[texto1]['output_norm'] - results[texto2]['output_norm'])
    print(f"   Norm difference: {output_diff:.6f}")
    print(f"   Texto 1 norm: {results[texto1]['output_norm']:.6f}")
    print(f"   Texto 2 norm: {results[texto2]['output_norm']:.6f}")
    
    # Analizar pérdida de señal en cada capa
    print(f"\n🔵 CAPAS INTERMEDIAS:")
    
    common_layers = set(results[texto1]['layer_stats'].keys()) & set(results[texto2]['layer_stats'].keys())
    
    for layer_name in sorted(common_layers):
        stats1 = results[texto1]['layer_stats'][layer_name]
        stats2 = results[texto2]['layer_stats'][layer_name]
        
        diff_norm = abs(stats1['norm'] - stats2['norm'])
        diff_mean = abs(stats1['mean'] - stats2['mean'])
        
        print(f"\n   {layer_name}:")
        print(f"      Norm diff:  {diff_norm:.6f}")
        print(f"      Mean diff:  {diff_mean:.6f}")
        print(f"      Texto 1: norm={stats1['norm']:.4f}, mean={stats1['mean']:.4f}")
        print(f"      Texto 2: norm={stats2['norm']:.4f}, mean={stats2['mean']:.4f}")
    
    # Diagnóstico final
    print("\n\n" + "="*80)
    print("🎯 DIAGNÓSTICO DE PÉRDIDA DE SEÑAL")
    print("="*80)
    
    signal_loss_ratio = output_diff / emb_diff if emb_diff > 0 else 0
    
    print(f"\n1️⃣ Embedding crudo → Input transformado:")
    if input_diff / emb_diff < 0.1:
        print(f"   ❌ PÉRDIDA CRÍTICA: {(1 - input_diff/emb_diff)*100:.1f}% de la señal")
        print(f"   → El problema está en generate_text_based_input()")
    else:
        print(f"   ✅ Señal preservada: {(input_diff/emb_diff)*100:.1f}% retenido")
    
    print(f"\n2️⃣ Input → Output de la red:")
    if output_diff / input_diff < 0.1 if input_diff > 0 else False:
        print(f"   ❌ PÉRDIDA CRÍTICA en la red neuronal")
        print(f"   → Las capas están homogeneizando la señal")
    else:
        print(f"   ⚠️ Señal se mantiene parcialmente")
    
    print(f"\n3️⃣ Ratio de preservación total (Embedding → Output):")
    print(f"   Signal preservation: {signal_loss_ratio*100:.2f}%")
    
    if signal_loss_ratio < 0.01:
        print(f"\n❌ CONCLUSIÓN: PÉRDIDA CATASTRÓFICA DE SEÑAL")
        print(f"   La diferencia semántica de {emb_diff:.4f} se reduce a {output_diff:.4f}")
        print(f"   Reducción de {(1-signal_loss_ratio)*100:.1f}%")
    elif signal_loss_ratio < 0.1:
        print(f"\n⚠️ CONCLUSIÓN: PÉRDIDA SEVERA DE SEÑAL")
        print(f"   La red homogeneiza las diferencias semánticas")
    else:
        print(f"\n✅ CONCLUSIÓN: SEÑAL SE PRESERVA PARCIALMENTE")
        print(f"   La red mantiene {signal_loss_ratio*100:.1f}% de la diferenciación")
    
    # Identificar la capa más problemática
    print(f"\n4️⃣ Capa con mayor pérdida de señal:")
    
    if common_layers:
        max_loss_layer = None
        max_loss = 0
        
        for layer_name in common_layers:
            stats1 = results[texto1]['layer_stats'][layer_name]
            stats2 = results[texto2]['layer_stats'][layer_name]
            diff = abs(stats1['norm'] - stats2['norm'])
            
            if max_loss_layer is None or diff < max_loss:
                max_loss_layer = layer_name
                max_loss = diff
        
        print(f"   🔴 {max_loss_layer}")
        print(f"      Diferencia mínima: {max_loss:.6f}")
        print(f"      → Esta capa homogeneiza más que otras")

if __name__ == "__main__":
    analyze_signal_propagation()
