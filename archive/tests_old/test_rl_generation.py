#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba de generación de texto usando el agente RL entrenado.
"""

import torch
import numpy as np
from stable_baselines3 import PPO
from src.rl.infinito_rl_env import InfinitoRLEnv
import json
from datetime import datetime

def load_model_and_env(checkpoint_path):
    """Cargar modelo y entorno."""
    print(f"📦 Cargando modelo desde: {checkpoint_path}")
    
    # Cargar configuración del entorno
    config_path = "outputs/rl_phi_text_scheduler/env_config.json"
    with open(config_path, 'r') as f:
        env_config = json.load(f)
    
    print("✅ Configuración cargada")
    
    # Crear entorno
    env = InfinitoRLEnv(config=env_config)
    print("✅ Entorno creado")
    
    # Cargar modelo PPO
    model = PPO.load(checkpoint_path, env=env)
    print("✅ Modelo PPO cargado")
    
    return model, env

def generate_with_rl_agent(
    model,
    env,
    prompt="The nature of consciousness",
    max_steps=100,
    temperature=0.8,
    top_k=40,
    verbose=True
):
    """
    Generar texto usando el agente RL entrenado.
    
    Args:
        model: Modelo PPO entrenado
        env: Entorno INFINITO RL
        prompt: Texto inicial
        max_steps: Pasos máximos de generación
        temperature: Temperatura de muestreo
        top_k: Top-k para muestreo
        verbose: Mostrar progreso
    """
    
    print("\n" + "="*70)
    print("🚀 GENERACIÓN CON AGENTE RL")
    print("="*70)
    print(f"Prompt: '{prompt}'")
    print(f"Max steps: {max_steps}")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    print("="*70)
    
    # Resetear entorno con prompt
    obs, info = env.reset()
    
    # Inicializar texto con el prompt
    tokenizer = env.model.tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(env.device)
    generated_text = prompt
    
    # Métricas tracking
    actions_taken = []
    phi_history = []
    consciousness_history = []
    ppl_history = []
    rewards_history = []
    
    print("\n📝 Generando texto...\n")
    print(f"Inicio: {prompt}")
    print("-" * 70)
    
    step = 0
    done = False
    
    while step < max_steps and not done:
        # Agente decide la acción
        action, _states = model.predict(obs, deterministic=False)
        action_name = ['TEXT', 'PHI', 'MIXED'][action]
        
        # Ejecutar acción en el entorno
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Guardar métricas
        actions_taken.append(action)
        phi_history.append(info.get('phi', 0))
        consciousness_history.append(info.get('consciousness', 0))
        ppl_history.append(info.get('perplexity', 0))
        rewards_history.append(reward)
        
        # Mostrar progreso cada 10 steps
        if verbose and (step + 1) % 10 == 0:
            phi = info.get('phi', 0)
            c = info.get('consciousness', 0)
            ppl = info.get('perplexity', 0)
            
            print(f"Step {step+1:3d}: {action_name:5s} | "
                  f"Φ={phi:5.2f} | C={c:4.2f} | PPL={ppl:6.1f} | "
                  f"Reward={reward:+6.3f}")
        
        step += 1
    
    # Obtener texto generado final
    # El texto está en el modelo del entorno
    try:
        # Intentar obtener el texto del último forward
        with torch.no_grad():
            generated_ids = env.model.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + step * 5,  # Aproximado
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"⚠️ No se pudo generar texto completo: {e}")
        generated_text = prompt + " [generación en progreso durante RL]"
    
    print("\n" + "="*70)
    print("📊 ESTADÍSTICAS DE GENERACIÓN")
    print("="*70)
    
    # Analizar acciones
    actions_count = {
        'TEXT': actions_taken.count(0),
        'PHI': actions_taken.count(1),
        'MIXED': actions_taken.count(2)
    }
    
    total = len(actions_taken)
    print(f"\n🎮 Distribución de acciones ({total} pasos):")
    for name, count in actions_count.items():
        pct = (count / total * 100) if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {name:5s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Métricas INFINITO
    print(f"\n🧠 Métricas INFINITO:")
    print(f"  PHI (Φ):")
    print(f"    Promedio: {np.mean(phi_history):6.3f}")
    print(f"    Rango:    [{np.min(phi_history):6.3f}, {np.max(phi_history):6.3f}]")
    print(f"    Std:      {np.std(phi_history):6.3f}")
    
    print(f"\n  Consciousness (C):")
    print(f"    Promedio: {np.mean(consciousness_history):6.3f}")
    print(f"    Rango:    [{np.min(consciousness_history):6.3f}, {np.max(consciousness_history):6.3f}]")
    
    print(f"\n  Perplexity (PPL):")
    print(f"    Promedio: {np.mean(ppl_history):7.2f}")
    print(f"    Rango:    [{np.min(ppl_history):7.2f}, {np.max(ppl_history):7.2f}]")
    
    # Rewards
    print(f"\n💰 Rewards:")
    print(f"    Total:    {np.sum(rewards_history):+8.3f}")
    print(f"    Promedio: {np.mean(rewards_history):+8.3f}")
    print(f"    Rango:    [{np.min(rewards_history):+8.3f}, {np.max(rewards_history):+8.3f}]")
    
    # Verificar objetivos
    print(f"\n✅ Verificación de objetivos:")
    phi_in_range = np.mean([(3 <= p <= 6) for p in phi_history]) * 100
    ppl_safe = np.mean([p >= 10 for p in ppl_history]) * 100
    mixed_used = (actions_count['MIXED'] / total * 100) if total > 0 else 0
    
    print(f"  Φ en [3-6]:  {phi_in_range:5.1f}% {'✅' if phi_in_range > 70 else '⚠️'}")
    print(f"  PPL >= 10:   {ppl_safe:5.1f}% {'✅' if ppl_safe > 90 else '⚠️'}")
    print(f"  Uso MIXED:   {mixed_used:5.1f}% {'✅' if mixed_used > 15 else '⚠️'}")
    
    print("\n" + "="*70)
    print("📄 TEXTO GENERADO")
    print("="*70)
    print(generated_text)
    print("="*70)
    
    return {
        'text': generated_text,
        'actions': actions_taken,
        'phi_history': phi_history,
        'consciousness_history': consciousness_history,
        'ppl_history': ppl_history,
        'rewards_history': rewards_history,
        'metrics': {
            'phi_mean': np.mean(phi_history),
            'phi_std': np.std(phi_history),
            'phi_in_range': phi_in_range,
            'ppl_mean': np.mean(ppl_history),
            'ppl_safe': ppl_safe,
            'mixed_pct': mixed_used,
            'total_reward': np.sum(rewards_history),
        }
    }

def main():
    """Probar diferentes checkpoints."""
    
    checkpoints = [
        ("Mejor Modelo (30K)", "outputs/rl_phi_text_scheduler/checkpoints/ppo_infinito_scheduler_30000_steps.zip"),
        ("Modelo Final (50K)", "outputs/rl_phi_text_scheduler/ppo_infinito_scheduler_final.zip"),
    ]
    
    prompts = [
        "The nature of consciousness",
        "Artificial intelligence can",
        "In the beginning there was",
    ]
    
    print("="*70)
    print("🧪 TEST DE GENERACIÓN CON AGENTES RL ENTRENADOS")
    print("="*70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoints a probar: {len(checkpoints)}")
    print(f"Prompts: {len(prompts)}")
    print("="*70)
    
    results = {}
    
    for ckpt_name, ckpt_path in checkpoints:
        print(f"\n\n{'='*70}")
        print(f"🎯 Probando: {ckpt_name}")
        print(f"{'='*70}")
        
        try:
            # Cargar modelo
            model, env = load_model_and_env(ckpt_path)
            
            results[ckpt_name] = {}
            
            # Probar cada prompt
            for prompt in prompts:
                print(f"\n\n{'─'*70}")
                print(f"Prompt: '{prompt}'")
                print(f"{'─'*70}")
                
                result = generate_with_rl_agent(
                    model=model,
                    env=env,
                    prompt=prompt,
                    max_steps=50,
                    temperature=0.8,
                    top_k=40,
                    verbose=True
                )
                
                results[ckpt_name][prompt] = result
            
            # Cerrar entorno
            env.close()
            
        except Exception as e:
            print(f"\n❌ Error probando {ckpt_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Resumen comparativo
    print("\n\n" + "="*70)
    print("📊 RESUMEN COMPARATIVO")
    print("="*70)
    
    for ckpt_name, ckpt_results in results.items():
        if not ckpt_results:
            continue
        
        print(f"\n{ckpt_name}:")
        
        # Promediar métricas
        all_metrics = [r['metrics'] for r in ckpt_results.values()]
        
        avg_phi = np.mean([m['phi_mean'] for m in all_metrics])
        avg_mixed = np.mean([m['mixed_pct'] for m in all_metrics])
        avg_phi_in_range = np.mean([m['phi_in_range'] for m in all_metrics])
        avg_reward = np.mean([m['total_reward'] for m in all_metrics])
        
        print(f"  Φ promedio:      {avg_phi:6.3f}")
        print(f"  Φ en [3-6]:      {avg_phi_in_range:5.1f}%")
        print(f"  Uso MIXED:       {avg_mixed:5.1f}%")
        print(f"  Reward total:    {avg_reward:+8.3f}")
    
    print("\n" + "="*70)
    print("✅ PRUEBA COMPLETADA")
    print("="*70)

if __name__ == "__main__":
    main()
