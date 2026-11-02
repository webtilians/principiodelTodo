"""
Análisis de Generación y Calidad de Texto - INFINITO
====================================================
Análisis enfocados en la calidad de generación:
1. Análisis de diversidad léxica
2. Análisis de coherencia semántica
3. Comparación con/sin memoria IIT
4. Análisis de repetición
5. Mapa de calor de atención
6. Benchmark de velocidad con PHI
"""

import torch
import numpy as np
import json
import os
import time
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/infinito_gpt2_spanish_phi.pt"
RESULTS_DIR = "results"

def load_model():
    """Carga el modelo entrenado"""
    from train_gpt2_with_phi_observer import InfinitoGPT2WithObserver
    
    print("Cargando modelo...")
    model = InfinitoGPT2WithObserver()
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✓ Modelo cargado desde {MODEL_PATH}")
    
    model = model.to(DEVICE)
    model.eval()
    return model

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8, top_k=50, use_memory=False):
    """Genera texto con parámetros configurables"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    
    generated = input_ids.clone()
    phi_values = []
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs, metrics = model(generated, return_phi=True, use_memory=use_memory)
            phi_values.append(metrics['phi'].mean().item())
            
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Stop en EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text, phi_values

def analyze_lexical_diversity(model, tokenizer):
    """Análisis de diversidad léxica del texto generado"""
    print("\n" + "="*60)
    print("ANÁLISIS 1: Diversidad Léxica")
    print("="*60)
    
    prompts = [
        "El futuro de la humanidad",
        "La inteligencia artificial puede",
        "En un mundo donde",
        "Los científicos descubrieron",
        "La tecnología del mañana"
    ]
    
    results = []
    
    for prompt in prompts:
        text, phi = generate_text(model, tokenizer, prompt, max_length=100)
        
        # Tokenizar para análisis
        words = text.split()
        unique_words = set(words)
        
        # Métricas de diversidad
        ttr = len(unique_words) / len(words) if words else 0  # Type-Token Ratio
        
        # Hapax legomena (palabras que aparecen una sola vez)
        word_counts = Counter(words)
        hapax = sum(1 for count in word_counts.values() if count == 1)
        hapax_ratio = hapax / len(words) if words else 0
        
        result = {
            'prompt': prompt,
            'text': text[:150] + "...",
            'total_words': len(words),
            'unique_words': len(unique_words),
            'ttr': ttr,
            'hapax_ratio': hapax_ratio,
            'mean_phi': np.mean(phi)
        }
        results.append(result)
        
        print(f"\n{prompt}...")
        print(f"  Palabras: {len(words)} | Únicas: {len(unique_words)} | TTR: {ttr:.3f}")
        print(f"  Hapax ratio: {hapax_ratio:.3f} | PHI medio: {np.mean(phi):.3f}")
    
    avg_ttr = np.mean([r['ttr'] for r in results])
    print(f"\n→ TTR promedio: {avg_ttr:.3f}")
    
    return results

def analyze_repetition(model, tokenizer):
    """Análisis de patrones de repetición"""
    print("\n" + "="*60)
    print("ANÁLISIS 2: Patrones de Repetición")
    print("="*60)
    
    prompts = [
        "La vida es",
        "El universo contiene",
        "Los seres humanos"
    ]
    
    results = []
    
    for prompt in prompts:
        text, phi = generate_text(model, tokenizer, prompt, max_length=150)
        
        # Analizar n-gramas repetidos
        words = text.split()
        
        # Bigramas
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
        
        # Trigramas
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)
        
        # Frases repetidas (4+ palabras)
        phrases = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        phrase_counts = Counter(phrases)
        repeated_phrases = sum(1 for count in phrase_counts.values() if count > 1)
        
        repetition_score = (repeated_bigrams * 0.3 + repeated_trigrams * 0.5 + repeated_phrases * 1.0) / len(words) * 100
        
        result = {
            'prompt': prompt,
            'repeated_bigrams': repeated_bigrams,
            'repeated_trigrams': repeated_trigrams,
            'repeated_phrases': repeated_phrases,
            'repetition_score': repetition_score,
            'text_preview': text[:100]
        }
        results.append(result)
        
        print(f"\n{prompt}...")
        print(f"  Bigramas repetidos: {repeated_bigrams}")
        print(f"  Trigramas repetidos: {repeated_trigrams}")
        print(f"  Frases repetidas (4+): {repeated_phrases}")
        print(f"  Score de repetición: {repetition_score:.2f}%")
    
    return results

def compare_memory_modes(model, tokenizer):
    """Comparar generación con y sin memoria IIT"""
    print("\n" + "="*60)
    print("ANÁLISIS 3: Comparación Con/Sin Memoria IIT")
    print("="*60)
    
    prompts = [
        "La consciencia humana es",
        "El aprendizaje profundo permite",
        "En el futuro, los robots"
    ]
    
    results = []
    
    for prompt in prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        
        # Sin memoria
        text_no_mem, phi_no_mem = generate_text(model, tokenizer, prompt, max_length=50, use_memory=False)
        
        # Resetear memoria antes de usar
        model.iit_memory.reset()
        
        # Con memoria
        text_with_mem, phi_with_mem = generate_text(model, tokenizer, prompt, max_length=50, use_memory=True)
        
        result = {
            'prompt': prompt,
            'without_memory': {
                'text': text_no_mem,
                'phi_mean': np.mean(phi_no_mem),
                'phi_std': np.std(phi_no_mem)
            },
            'with_memory': {
                'text': text_with_mem,
                'phi_mean': np.mean(phi_with_mem),
                'phi_std': np.std(phi_with_mem)
            }
        }
        results.append(result)
        
        print(f"\n  SIN MEMORIA:")
        print(f"    PHI: {np.mean(phi_no_mem):.3f} ± {np.std(phi_no_mem):.3f}")
        print(f"    Texto: {text_no_mem[:80]}...")
        
        print(f"\n  CON MEMORIA:")
        print(f"    PHI: {np.mean(phi_with_mem):.3f} ± {np.std(phi_with_mem):.3f}")
        print(f"    Texto: {text_with_mem[:80]}...")
    
    return results

def analyze_attention_patterns(model, tokenizer):
    """Análisis de patrones de atención"""
    print("\n" + "="*60)
    print("ANÁLISIS 4: Patrones de Atención")
    print("="*60)
    
    prompt = "La inteligencia artificial revoluciona el mundo moderno"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Tokens: {tokens}")
    
    with torch.no_grad():
        gpt2_outputs = model.gpt2(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )
    
    # Analizar atención de la última capa
    last_attention = gpt2_outputs.attentions[-1]  # [1, heads, seq, seq]
    avg_attention = last_attention.mean(dim=1).squeeze().cpu().numpy()  # [seq, seq]
    
    # Métricas de atención
    attention_entropy = []
    for i in range(avg_attention.shape[0]):
        row = avg_attention[i]
        row = row + 1e-10  # Evitar log(0)
        entropy = -np.sum(row * np.log(row))
        attention_entropy.append(entropy)
    
    results = {
        'prompt': prompt,
        'tokens': tokens,
        'attention_matrix': avg_attention.tolist(),
        'entropy_per_token': attention_entropy,
        'mean_entropy': np.mean(attention_entropy)
    }
    
    print(f"\nEntropía de atención por token:")
    for i, (token, entropy) in enumerate(zip(tokens, attention_entropy)):
        bar = "█" * int(entropy * 5)
        print(f"  {token:15s}: {entropy:.3f} {bar}")
    
    print(f"\nEntropía promedio: {np.mean(attention_entropy):.3f}")
    
    return results

def benchmark_speed(model, tokenizer):
    """Benchmark de velocidad con PHI"""
    print("\n" + "="*60)
    print("ANÁLISIS 5: Benchmark de Velocidad")
    print("="*60)
    
    prompt = "La inteligencia artificial"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model(input_ids, return_phi=True, use_memory=False)
    
    # Benchmark sin PHI
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    iterations = 50
    
    for _ in range(iterations):
        with torch.no_grad():
            model.gpt2(input_ids)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_without_phi = (time.time() - start) / iterations * 1000
    
    # Benchmark con PHI
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    for _ in range(iterations):
        with torch.no_grad():
            model(input_ids, return_phi=True, use_memory=False)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_with_phi = (time.time() - start) / iterations * 1000
    
    # Benchmark con memoria
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    for _ in range(iterations):
        with torch.no_grad():
            model(input_ids, return_phi=True, use_memory=True)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_with_memory = (time.time() - start) / iterations * 1000
    
    overhead_phi = ((time_with_phi - time_without_phi) / time_without_phi) * 100
    overhead_memory = ((time_with_memory - time_with_phi) / time_with_phi) * 100
    
    results = {
        'time_without_phi_ms': time_without_phi,
        'time_with_phi_ms': time_with_phi,
        'time_with_memory_ms': time_with_memory,
        'overhead_phi_percent': overhead_phi,
        'overhead_memory_percent': overhead_memory,
        'device': str(DEVICE)
    }
    
    print(f"\n📊 Resultados (promedio de {iterations} iteraciones):")
    print(f"  Sin PHI:      {time_without_phi:.2f} ms")
    print(f"  Con PHI:      {time_with_phi:.2f} ms (+{overhead_phi:.1f}%)")
    print(f"  Con Memoria:  {time_with_memory:.2f} ms (+{overhead_memory:.1f}% adicional)")
    print(f"\n  Dispositivo: {DEVICE}")
    
    return results

def analyze_temperature_effects(model, tokenizer):
    """Análisis del efecto de temperatura en PHI"""
    print("\n" + "="*60)
    print("ANÁLISIS 6: Efecto de Temperatura en PHI")
    print("="*60)
    
    prompt = "El futuro de la inteligencia artificial"
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5]
    
    results = {}
    
    for temp in temperatures:
        text, phi_values = generate_text(model, tokenizer, prompt, max_length=50, temperature=temp)
        
        # Diversidad léxica
        words = text.split()
        ttr = len(set(words)) / len(words) if words else 0
        
        results[temp] = {
            'text': text,
            'phi_mean': np.mean(phi_values),
            'phi_std': np.std(phi_values),
            'ttr': ttr,
            'word_count': len(words)
        }
        
        print(f"\n🌡️ Temperatura: {temp}")
        print(f"  PHI: {np.mean(phi_values):.3f} ± {np.std(phi_values):.3f}")
        print(f"  Diversidad (TTR): {ttr:.3f}")
        print(f"  Texto: {text[:70]}...")
    
    return results

def create_visualizations(all_results):
    """Crear visualizaciones de los análisis"""
    print("\n" + "="*60)
    print("Generando Visualizaciones...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análisis de Generación - INFINITO', fontsize=16, fontweight='bold')
    
    # 1. Diversidad léxica
    if 'lexical' in all_results and all_results['lexical']:
        ax = axes[0, 0]
        data = all_results['lexical']
        prompts = [d['prompt'][:15] + '...' for d in data]
        ttr = [d['ttr'] for d in data]
        phi = [d['mean_phi'] for d in data]
        
        x = np.arange(len(prompts))
        width = 0.35
        ax.bar(x - width/2, ttr, width, label='TTR', color='steelblue', alpha=0.7)
        ax2 = ax.twinx()
        ax2.bar(x + width/2, phi, width, label='PHI', color='coral', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(prompts, rotation=45, ha='right')
        ax.set_ylabel('TTR', color='steelblue')
        ax2.set_ylabel('PHI', color='coral')
        ax.set_title('Diversidad Léxica vs PHI')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    # 2. Patrones de repetición
    if 'repetition' in all_results:
        ax = axes[0, 1]
        data = all_results['repetition']
        prompts = [d['prompt'][:15] + '...' for d in data]
        scores = [d['repetition_score'] for d in data]
        
        colors = ['green' if s < 10 else 'yellow' if s < 20 else 'red' for s in scores]
        bars = ax.bar(prompts, scores, color=colors, alpha=0.7)
        ax.set_ylabel('Score de Repetición (%)')
        ax.set_title('Patrones de Repetición')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Bueno')
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Alto')
        ax.legend()
    
    # 3. Comparación memoria
    if 'memory_compare' in all_results:
        ax = axes[0, 2]
        data = all_results['memory_compare']
        prompts = [d['prompt'][:15] + '...' for d in data]
        phi_no = [d['without_memory']['phi_mean'] for d in data]
        phi_mem = [d['with_memory']['phi_mean'] for d in data]
        
        x = np.arange(len(prompts))
        width = 0.35
        ax.bar(x - width/2, phi_no, width, label='Sin Memoria', color='gray', alpha=0.7)
        ax.bar(x + width/2, phi_mem, width, label='Con Memoria', color='purple', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(prompts, rotation=45, ha='right')
        ax.set_ylabel('PHI')
        ax.set_title('PHI: Sin vs Con Memoria IIT')
        ax.legend()
    
    # 4. Mapa de calor de atención
    if 'attention' in all_results:
        ax = axes[1, 0]
        data = all_results['attention']
        matrix = np.array(data['attention_matrix'])
        tokens = [t[:8] for t in data['tokens']]
        
        im = ax.imshow(matrix, cmap='viridis')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title('Mapa de Atención (Última Capa)')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 5. Benchmark de velocidad
    if 'benchmark' in all_results:
        ax = axes[1, 1]
        data = all_results['benchmark']
        modes = ['Sin PHI', 'Con PHI', 'Con Memoria']
        times = [data['time_without_phi_ms'], data['time_with_phi_ms'], data['time_with_memory_ms']]
        colors = ['green', 'blue', 'purple']
        
        bars = ax.bar(modes, times, color=colors, alpha=0.7)
        ax.set_ylabel('Tiempo (ms)')
        ax.set_title(f'Benchmark de Velocidad ({data["device"]})')
        
        for bar, time in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{time:.1f}ms', ha='center', va='bottom', fontsize=10)
    
    # 6. Efecto de temperatura
    if 'temperature' in all_results:
        ax = axes[1, 2]
        data = all_results['temperature']
        temps = sorted(data.keys())
        phi_means = [data[t]['phi_mean'] for t in temps]
        ttrs = [data[t]['ttr'] for t in temps]
        
        ax.plot(temps, phi_means, 'o-', color='coral', label='PHI', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(temps, ttrs, 's-', color='steelblue', label='TTR', linewidth=2)
        ax.set_xlabel('Temperatura')
        ax.set_ylabel('PHI', color='coral')
        ax2.set_ylabel('TTR (Diversidad)', color='steelblue')
        ax.set_title('Efecto de Temperatura')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(RESULTS_DIR, 'phi_generation_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualización guardada: {output_path}")
    
    plt.close()

def main():
    print("="*60)
    print("  ANÁLISIS DE GENERACIÓN - INFINITO")
    print("="*60)
    
    # Cargar modelo
    model = load_model()
    tokenizer = model.tokenizer
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = {}
    
    # Ejecutar análisis
    try:
        all_results['lexical'] = analyze_lexical_diversity(model, tokenizer)
    except Exception as e:
        print(f"Error en análisis léxico: {e}")
    
    try:
        all_results['repetition'] = analyze_repetition(model, tokenizer)
    except Exception as e:
        print(f"Error en análisis de repetición: {e}")
    
    try:
        all_results['memory_compare'] = compare_memory_modes(model, tokenizer)
    except Exception as e:
        print(f"Error en comparación de memoria: {e}")
    
    try:
        all_results['attention'] = analyze_attention_patterns(model, tokenizer)
    except Exception as e:
        print(f"Error en análisis de atención: {e}")
    
    try:
        all_results['benchmark'] = benchmark_speed(model, tokenizer)
    except Exception as e:
        print(f"Error en benchmark: {e}")
    
    try:
        all_results['temperature'] = analyze_temperature_effects(model, tokenizer)
    except Exception as e:
        print(f"Error en análisis de temperatura: {e}")
    
    # Crear visualizaciones
    try:
        create_visualizations(all_results)
    except Exception as e:
        print(f"Error en visualizaciones: {e}")
    
    # Guardar resultados
    output_file = os.path.join(RESULTS_DIR, 'phi_generation_analysis.json')
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Resultados guardados: {output_file}")
    
    # Resumen final
    print("\n" + "="*60)
    print("  RESUMEN DE ANÁLISIS DE GENERACIÓN")
    print("="*60)
    
    if 'lexical' in all_results:
        avg_ttr = np.mean([r['ttr'] for r in all_results['lexical']])
        print(f"\n• Diversidad léxica promedio (TTR): {avg_ttr:.3f}")
    
    if 'repetition' in all_results:
        avg_rep = np.mean([r['repetition_score'] for r in all_results['repetition']])
        print(f"• Score de repetición promedio: {avg_rep:.2f}%")
    
    if 'benchmark' in all_results:
        data = all_results['benchmark']
        print(f"• Overhead del PHI Observer: {data['overhead_phi_percent']:.1f}%")
        print(f"• Overhead de Memoria IIT: {data['overhead_memory_percent']:.1f}%")
    
    if 'temperature' in all_results:
        temps = all_results['temperature']
        best_temp = max(temps.items(), key=lambda x: x[1]['phi_mean'])
        print(f"• Mejor temperatura para PHI: {best_temp[0]} (PHI={best_temp[1]['phi_mean']:.3f})")
    
    print("\n" + "="*60)
    print("  ANÁLISIS COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    main()
