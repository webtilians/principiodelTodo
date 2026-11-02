"""
Análisis Avanzado de PHI - INFINITO
===================================
Análisis adicionales para comprender mejor el comportamiento del PHI:
1. Análisis por longitud de prompt
2. Análisis de estabilidad temporal (ventanas deslizantes)
3. Heatmap de interacciones entre componentes
4. Análisis de tokens críticos (cuáles tokens activan más PHI)
5. Comparación de patrones entre categorías
6. Distribución de PHI por capa
"""

import torch
import numpy as np
import json
import os
from collections import defaultdict
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

def analyze_by_length(model, tokenizer):
    """Análisis de PHI según longitud del prompt"""
    print("\n" + "="*60)
    print("ANÁLISIS 1: PHI por Longitud de Prompt")
    print("="*60)
    
    prompts_by_length = {
        "muy_corto": [  # 1-5 tokens
            "Hola",
            "Sí",
            "No",
            "Bien",
            "Gracias"
        ],
        "corto": [  # 5-15 tokens
            "¿Cómo estás hoy?",
            "El cielo es azul",
            "Me gusta la música",
            "Hace buen tiempo",
            "Tengo hambre"
        ],
        "medio": [  # 15-30 tokens
            "La inteligencia artificial está transformando nuestra forma de vivir",
            "El cambio climático es un problema que debemos resolver juntos",
            "La educación es la base del desarrollo de cualquier sociedad",
            "Los avances en medicina han salvado millones de vidas",
            "La tecnología nos conecta pero también puede aislarnos"
        ],
        "largo": [  # 30-50 tokens
            "En un mundo cada vez más interconectado, la colaboración entre naciones se vuelve esencial para abordar desafíos globales como el cambio climático, las pandemias y la desigualdad económica",
            "La revolución digital ha transformado profundamente la manera en que trabajamos, nos comunicamos y accedemos a la información, creando nuevas oportunidades pero también nuevos desafíos sociales",
            "El estudio de la consciencia humana representa uno de los mayores misterios de la ciencia moderna, combinando perspectivas de la neurociencia, la filosofía y la inteligencia artificial",
            "Las ciudades inteligentes utilizan tecnología de sensores y análisis de datos para optimizar el transporte, el consumo de energía y la calidad de vida de sus habitantes",
            "La exploración espacial continúa fascinando a la humanidad, con misiones a Marte y el desarrollo de turismo espacial marcando el inicio de una nueva era"
        ]
    }
    
    results = {}
    
    for length_cat, prompts in prompts_by_length.items():
        phi_values = []
        components_values = defaultdict(list)
        token_counts = []
        
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            token_counts.append(input_ids.shape[1])
            
            with torch.no_grad():
                outputs, metrics = model(input_ids, return_phi=True, use_memory=False)
            
            phi_values.append(metrics['phi'].mean().item())
            for comp, val in metrics['raw_components'].items():
                components_values[comp].append(val)
        
        results[length_cat] = {
            'phi_mean': np.mean(phi_values),
            'phi_std': np.std(phi_values),
            'avg_tokens': np.mean(token_counts),
            'components': {k: np.mean(v) for k, v in components_values.items()}
        }
        
        print(f"\n{length_cat.upper()} (~{np.mean(token_counts):.0f} tokens):")
        print(f"  PHI: {np.mean(phi_values):.4f} ± {np.std(phi_values):.4f}")
        for comp, val in results[length_cat]['components'].items():
            print(f"  {comp}: {val:.4f}")
    
    return results

def analyze_temporal_stability(model, tokenizer):
    """Análisis de estabilidad del PHI con generación token a token"""
    print("\n" + "="*60)
    print("ANÁLISIS 2: Estabilidad Temporal del PHI")
    print("="*60)
    
    prompts = [
        "La consciencia es un fenómeno",
        "El universo se expande constantemente",
        "La inteligencia artificial puede"
    ]
    
    results = {}
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        phi_evolution = []
        generated_tokens = []
        
        # Generar token a token y medir PHI
        current_ids = input_ids.clone()
        
        for step in range(20):  # Generar 20 tokens
            with torch.no_grad():
                outputs, metrics = model(current_ids, return_phi=True, use_memory=False)
                
                phi_evolution.append(metrics['phi'].mean().item())
                
                # Obtener siguiente token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                generated_tokens.append(tokenizer.decode(next_token[0]))
        
        # Calcular métricas de estabilidad
        phi_diff = np.diff(phi_evolution)
        
        results[prompt] = {
            'phi_evolution': phi_evolution,
            'phi_mean': np.mean(phi_evolution),
            'phi_std': np.std(phi_evolution),
            'phi_range': max(phi_evolution) - min(phi_evolution),
            'volatility': np.std(phi_diff),
            'generated_text': ''.join(generated_tokens[:10])
        }
        
        print(f"  PHI promedio: {np.mean(phi_evolution):.4f}")
        print(f"  Volatilidad: {np.std(phi_diff):.4f}")
        print(f"  Rango: {results[prompt]['phi_range']:.4f}")
        print(f"  Generado: {results[prompt]['generated_text']}...")
    
    return results

def analyze_component_interactions(model, tokenizer):
    """Análisis de correlaciones entre componentes IIT"""
    print("\n" + "="*60)
    print("ANÁLISIS 3: Matriz de Interacciones entre Componentes")
    print("="*60)
    
    # Usar prompts variados para capturar diferentes patrones
    prompts = [
        "Hola, ¿cómo estás?",
        "La física cuántica estudia las partículas subatómicas",
        "Me siento muy feliz hoy",
        "Escribe un poema sobre el mar",
        "¿Cuál es la capital de Francia?",
        "El amor es un sentimiento complejo",
        "Explica el teorema de Pitágoras",
        "Era una noche oscura y tormentosa",
        "¿Qué opinas sobre la inteligencia artificial?",
        "Los robots del futuro serán conscientes",
        "La música es el lenguaje universal",
        "El sol brilla con fuerza en verano",
        "Necesito ayuda con mi tarea",
        "La democracia es el gobierno del pueblo",
        "Los sueños revelan nuestro subconsciente"
    ]
    
    # Recolectar datos de componentes
    component_data = defaultdict(list)
    
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        
        with torch.no_grad():
            outputs, metrics = model(input_ids, return_phi=True, use_memory=False)
        
        for comp, val in metrics['raw_components'].items():
            component_data[comp].append(val)
    
    # Calcular matriz de correlación
    components = list(component_data.keys())
    n_components = len(components)
    correlation_matrix = np.zeros((n_components, n_components))
    
    for i, comp1 in enumerate(components):
        for j, comp2 in enumerate(components):
            correlation_matrix[i, j] = np.corrcoef(
                component_data[comp1], 
                component_data[comp2]
            )[0, 1]
    
    print("\nMatriz de Correlación:")
    print("="*50)
    header = "          " + "  ".join([c[:8].center(8) for c in components])
    print(header)
    for i, comp in enumerate(components):
        row = f"{comp[:8]:8s}  "
        for j in range(n_components):
            val = correlation_matrix[i, j]
            row += f"{val:+.3f}   "
        print(row)
    
    return {
        'components': components,
        'correlation_matrix': correlation_matrix.tolist(),
        'component_means': {k: np.mean(v) for k, v in component_data.items()},
        'component_stds': {k: np.std(v) for k, v in component_data.items()}
    }

def analyze_layer_distribution(model, tokenizer):
    """Análisis detallado de PHI por capa del transformer"""
    print("\n" + "="*60)
    print("ANÁLISIS 4: Distribución de PHI por Capa")
    print("="*60)
    
    prompts = [
        "La consciencia emerge de la complejidad neuronal",
        "El universo está compuesto de materia y energía",
        "Los sentimientos humanos son difíciles de explicar"
    ]
    
    layer_phi_all = defaultdict(list)
    
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        
        with torch.no_grad():
            # Acceso directo al GPT-2 para obtener hidden states y attentions
            gpt2_outputs = model.gpt2(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
            
            hidden_states = gpt2_outputs.hidden_states[1:]  # Skip embedding layer
            attentions = gpt2_outputs.attentions
            
            for layer_idx, (hidden, attn) in enumerate(zip(hidden_states, attentions)):
                # Calcular PHI por capa individual usando métricas simplificadas
                # Temporal: autocorrelación entre tokens
                hidden_flat = hidden.view(hidden.size(0), -1)
                temporal = torch.sigmoid(hidden_flat.std()).item()
                
                # Integration usando solo esta capa
                if layer_idx > 0:
                    prev_hidden = hidden_states[layer_idx - 1]
                    prev_flat = prev_hidden.view(prev_hidden.size(0), -1)
                    # Correlación entre capas
                    integration = torch.cosine_similarity(hidden_flat, prev_flat, dim=-1).mean().item()
                    integration = (integration + 1) / 2  # Normalizar a [0, 1]
                else:
                    integration = 0.5
                
                # Complexity: entropía aproximada
                complexity = torch.sigmoid(hidden.std(dim=-1).mean()).item()
                
                # Attention diversity
                if attn is not None:
                    # Calcular entropía de atención
                    attn_float = attn.float()
                    attn_flat = attn_float.mean(dim=1)  # Promediar sobre cabezas
                    attn_clamped = torch.clamp(attn_flat, min=1e-8)
                    entropy = -torch.sum(attn_clamped * torch.log(attn_clamped), dim=-1)
                    max_entropy = torch.log(torch.tensor(attn_flat.shape[-1], dtype=torch.float32))
                    attention_div = (entropy / max_entropy).mean().item()
                else:
                    attention_div = 0.5
                
                layer_phi = (0.3 * temporal + 0.3 * integration + 
                           0.2 * complexity + 0.2 * attention_div) * 10
                
                layer_phi_all[layer_idx].append(layer_phi)
    
    # Calcular estadísticas por capa
    results = {}
    print("\nPHI por Capa:")
    print("-" * 40)
    
    for layer_idx in sorted(layer_phi_all.keys()):
        values = layer_phi_all[layer_idx]
        mean_phi = np.mean(values)
        std_phi = np.std(values)
        
        results[f"layer_{layer_idx}"] = {
            'mean': mean_phi,
            'std': std_phi,
            'min': min(values),
            'max': max(values)
        }
        
        bar = "█" * int(mean_phi * 3)
        print(f"  Capa {layer_idx:2d}: {mean_phi:.3f} ± {std_phi:.3f} {bar}")
    
    # Identificar patrones
    means = [results[f"layer_{i}"]['mean'] for i in range(len(results))]
    peak_layer = np.argmax(means)
    
    print(f"\n  → Capa pico: {peak_layer} (PHI = {means[peak_layer]:.3f})")
    print(f"  → Tendencia: {'Ascendente' if means[-1] > means[0] else 'Descendente'}")
    
    return results

def analyze_token_importance(model, tokenizer):
    """Análisis de qué tokens contribuyen más al PHI"""
    print("\n" + "="*60)
    print("ANÁLISIS 5: Importancia de Tokens para PHI")
    print("="*60)
    
    prompt = "La consciencia artificial es un tema fascinante de estudio"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Tokens: {tokens}")
    
    # PHI base
    with torch.no_grad():
        outputs, base_metrics = model(input_ids, return_phi=True, use_memory=False)
        base_phi = base_metrics['phi'].mean().item()
    
    print(f"\nPHI base: {base_phi:.4f}")
    
    # Calcular importancia eliminando cada token
    token_importance = []
    
    for i in range(len(tokens)):
        # Crear input sin el token i
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        mask[0, i] = False
        masked_ids = input_ids[mask].unsqueeze(0)
        
        if masked_ids.shape[1] == 0:
            continue
            
        with torch.no_grad():
            outputs, metrics = model(masked_ids, return_phi=True, use_memory=False)
            phi_without = metrics['phi'].mean().item()
        
        importance = base_phi - phi_without  # Positivo = token importante
        token_importance.append({
            'token': tokens[i],
            'position': i,
            'importance': importance,
            'phi_without': phi_without
        })
    
    # Ordenar por importancia
    token_importance.sort(key=lambda x: abs(x['importance']), reverse=True)
    
    print("\nTokens por Importancia para PHI:")
    print("-" * 50)
    for item in token_importance[:10]:
        sign = "+" if item['importance'] > 0 else ""
        bar = "▓" * int(abs(item['importance']) * 50)
        print(f"  {item['token']:15s} : {sign}{item['importance']:.4f} {bar}")
    
    return token_importance

def analyze_category_patterns(model, tokenizer):
    """Análisis de patrones distintivos entre categorías"""
    print("\n" + "="*60)
    print("ANÁLISIS 6: Patrones por Categoría")
    print("="*60)
    
    categories = {
        "tecnico": [
            "El algoritmo tiene complejidad O(n log n)",
            "La función recursiva calcula el factorial",
            "El protocolo TCP garantiza la entrega de paquetes"
        ],
        "filosofico": [
            "¿Qué significa existir realmente?",
            "La consciencia es el mayor misterio",
            "El libre albedrío es una ilusión"
        ],
        "emocional": [
            "Me siento muy triste hoy",
            "El amor llena mi corazón de alegría",
            "La nostalgia me invade al recordar"
        ],
        "narrativo": [
            "Había una vez en un reino lejano",
            "El detective examinó la escena del crimen",
            "La nave espacial atravesó la galaxia"
        ]
    }
    
    results = {}
    
    for cat, prompts in categories.items():
        component_profiles = defaultdict(list)
        phi_values = []
        
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            
            with torch.no_grad():
                outputs, metrics = model(input_ids, return_phi=True, use_memory=False)
            
            phi_values.append(metrics['phi'].mean().item())
            for comp, val in metrics['raw_components'].items():
                component_profiles[comp].append(val)
        
        # Calcular perfil de la categoría
        profile = {k: np.mean(v) for k, v in component_profiles.items()}
        
        results[cat] = {
            'phi_mean': np.mean(phi_values),
            'phi_std': np.std(phi_values),
            'profile': profile,
            'dominant': max(profile, key=profile.get)
        }
        
        print(f"\n{cat.upper()}:")
        print(f"  PHI: {np.mean(phi_values):.4f}")
        print(f"  Componente dominante: {results[cat]['dominant']}")
        print(f"  Perfil: ", end="")
        for comp, val in profile.items():
            print(f"{comp[:4]}={val:.2f} ", end="")
        print()
    
    return results

def create_visualizations(all_results):
    """Crear visualizaciones de todos los análisis"""
    print("\n" + "="*60)
    print("Generando Visualizaciones...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análisis Avanzado de PHI - INFINITO', fontsize=16, fontweight='bold')
    
    # 1. PHI por longitud
    if 'length' in all_results:
        ax = axes[0, 0]
        data = all_results['length']
        lengths = list(data.keys())
        phi_means = [data[l]['phi_mean'] for l in lengths]
        phi_stds = [data[l]['phi_std'] for l in lengths]
        
        bars = ax.bar(lengths, phi_means, yerr=phi_stds, capsize=5, color='steelblue', alpha=0.7)
        ax.set_ylabel('PHI')
        ax.set_title('PHI por Longitud de Prompt')
        ax.set_ylim([6, 8])
        
        for bar, mean in zip(bars, phi_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Estabilidad temporal
    if 'temporal' in all_results:
        ax = axes[0, 1]
        data = all_results['temporal']
        for prompt, vals in data.items():
            short_prompt = prompt[:25] + "..."
            ax.plot(vals['phi_evolution'], label=short_prompt, marker='o', markersize=3)
        ax.set_xlabel('Paso de Generación')
        ax.set_ylabel('PHI')
        ax.set_title('Evolución Temporal del PHI')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 3. Matriz de correlación
    if 'interactions' in all_results:
        ax = axes[0, 2]
        data = all_results['interactions']
        matrix = np.array(data['correlation_matrix'])
        components = data['components']
        
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(components)))
        ax.set_yticks(range(len(components)))
        ax.set_xticklabels([c[:6] for c in components], rotation=45, ha='right')
        ax.set_yticklabels([c[:6] for c in components])
        ax.set_title('Correlación entre Componentes')
        
        # Añadir valores
        for i in range(len(components)):
            for j in range(len(components)):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 4. PHI por capa
    if 'layers' in all_results:
        ax = axes[1, 0]
        data = all_results['layers']
        layers = sorted([int(k.split('_')[1]) for k in data.keys()])
        means = [data[f'layer_{l}']['mean'] for l in layers]
        stds = [data[f'layer_{l}']['std'] for l in layers]
        
        ax.fill_between(layers, 
                       [m-s for m, s in zip(means, stds)],
                       [m+s for m, s in zip(means, stds)],
                       alpha=0.3, color='green')
        ax.plot(layers, means, 'o-', color='green', linewidth=2)
        ax.set_xlabel('Capa del Transformer')
        ax.set_ylabel('PHI')
        ax.set_title('PHI por Capa')
        ax.grid(True, alpha=0.3)
        
        peak = np.argmax(means)
        ax.axvline(x=peak, color='red', linestyle='--', alpha=0.5, label=f'Pico: capa {peak}')
        ax.legend()
    
    # 5. Importancia de tokens
    if 'tokens' in all_results:
        ax = axes[1, 1]
        data = all_results['tokens'][:8]  # Top 8
        tokens = [d['token'] for d in data]
        importance = [d['importance'] for d in data]
        
        colors = ['green' if i > 0 else 'red' for i in importance]
        bars = ax.barh(tokens, importance, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Importancia (ΔPHI)')
        ax.set_title('Importancia de Tokens')
    
    # 6. Patrones por categoría (radar chart simplificado)
    if 'categories' in all_results:
        ax = axes[1, 2]
        data = all_results['categories']
        
        categories = list(data.keys())
        phi_values = [data[c]['phi_mean'] for c in categories]
        dominants = [data[c]['dominant'] for c in categories]
        
        x = range(len(categories))
        bars = ax.bar(x, phi_values, color='purple', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylabel('PHI')
        ax.set_title('PHI por Categoría')
        ax.set_ylim([6, 8])
        
        for bar, dom in zip(bars, dominants):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   dom[:4], ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(RESULTS_DIR, 'phi_advanced_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualización guardada: {output_path}")
    
    plt.close()

def main():
    print("="*60)
    print("  ANÁLISIS AVANZADO DE PHI - INFINITO")
    print("="*60)
    
    # Cargar modelo
    model = load_model()
    tokenizer = model.tokenizer
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = {}
    
    # Ejecutar todos los análisis
    try:
        all_results['length'] = analyze_by_length(model, tokenizer)
    except Exception as e:
        print(f"Error en análisis de longitud: {e}")
    
    try:
        all_results['temporal'] = analyze_temporal_stability(model, tokenizer)
    except Exception as e:
        print(f"Error en análisis temporal: {e}")
    
    try:
        all_results['interactions'] = analyze_component_interactions(model, tokenizer)
    except Exception as e:
        print(f"Error en análisis de interacciones: {e}")
    
    try:
        all_results['layers'] = analyze_layer_distribution(model, tokenizer)
    except Exception as e:
        print(f"Error en análisis de capas: {e}")
    
    try:
        all_results['tokens'] = analyze_token_importance(model, tokenizer)
    except Exception as e:
        print(f"Error en análisis de tokens: {e}")
    
    try:
        all_results['categories'] = analyze_category_patterns(model, tokenizer)
    except Exception as e:
        print(f"Error en análisis de categorías: {e}")
    
    # Crear visualizaciones
    try:
        create_visualizations(all_results)
    except Exception as e:
        print(f"Error en visualizaciones: {e}")
    
    # Guardar resultados
    output_file = os.path.join(RESULTS_DIR, 'phi_advanced_analysis.json')
    
    # Convertir numpy arrays a listas para JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Resultados guardados: {output_file}")
    
    # Resumen final
    print("\n" + "="*60)
    print("  RESUMEN DE ANÁLISIS AVANZADO")
    print("="*60)
    
    if 'length' in all_results:
        best_length = max(all_results['length'].items(), key=lambda x: x[1]['phi_mean'])
        print(f"\n• Longitud óptima: {best_length[0]} (PHI={best_length[1]['phi_mean']:.3f})")
    
    if 'temporal' in all_results:
        avg_volatility = np.mean([v['volatility'] for v in all_results['temporal'].values()])
        print(f"• Volatilidad temporal promedio: {avg_volatility:.4f}")
    
    if 'interactions' in all_results:
        print(f"• Componente más correlacionado: {all_results['interactions']['components'][0]}")
    
    if 'layers' in all_results:
        means = [all_results['layers'][f'layer_{i}']['mean'] for i in range(len(all_results['layers']))]
        print(f"• Capa pico de PHI: {np.argmax(means)} (PHI={max(means):.3f})")
    
    if 'tokens' in all_results and all_results['tokens']:
        top_token = all_results['tokens'][0]
        print(f"• Token más importante: '{top_token['token']}' (Δ={top_token['importance']:.4f})")
    
    if 'categories' in all_results:
        best_cat = max(all_results['categories'].items(), key=lambda x: x[1]['phi_mean'])
        print(f"• Mejor categoría: {best_cat[0]} (PHI={best_cat[1]['phi_mean']:.3f})")
    
    print("\n" + "="*60)
    print("  ANÁLISIS COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    main()
