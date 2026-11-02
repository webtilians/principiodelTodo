"""
Monitor de Entrenamiento INFINITO V5.2
Revisa el progreso del entrenamiento cada 5 minutos
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

def check_log_file():
    """Lee las últimas líneas del log de entrenamiento."""
    log_file = Path('train_log.txt')
    
    if not log_file.exists():
        return None
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # Últimas 30 líneas
            return lines[-30:] if len(lines) >= 30 else lines
    except Exception as e:
        return None

def check_checkpoints():
    """Verifica checkpoints guardados."""
    checkpoint_dir = Path('models/checkpoints')
    
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = list(checkpoint_dir.glob('infinito_v5.2_real_*.pt'))
    checkpoints_info = []
    
    for ckpt in checkpoints:
        stat = ckpt.stat()
        checkpoints_info.append({
            'name': ckpt.name,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%H:%M:%S')
        })
    
    return sorted(checkpoints_info, key=lambda x: x['modified'], reverse=True)

def check_training_history():
    """Lee el archivo de historial de entrenamiento."""
    history_dir = Path('results/training')
    
    if not history_dir.exists():
        return None
    
    history_files = list(history_dir.glob('training_history_real_*.json'))
    
    if not history_files:
        return None
    
    # Tomar el más reciente
    latest = max(history_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest, 'r') as f:
            history = json.load(f)
            return history
    except:
        return None

def extract_epoch_info(lines):
    """Extrae información de época de las líneas del log."""
    if not lines:
        return None
    
    info = {
        'current_epoch': None,
        'progress': None,
        'train_ppl': None,
        'val_ppl': None,
        'last_update': datetime.now().strftime('%H:%M:%S')
    }
    
    # Buscar información en las últimas líneas
    for line in reversed(lines):
        # Época actual
        if 'ÉPOCA' in line and '/' in line:
            try:
                parts = line.split('/')
                epoch = parts[0].split('ÉPOCA')[-1].strip()
                info['current_epoch'] = epoch
            except:
                pass
        
        # Progress bar
        if 'Época' in line and '%' in line:
            try:
                if '|' in line:
                    progress_part = line.split('|')[0]
                    if '%' in progress_part:
                        percent = progress_part.split('%')[0].split()[-1]
                        info['progress'] = f"{percent}%"
            except:
                pass
        
        # Resultados
        if 'Train PPL:' in line:
            try:
                ppl = line.split('Train PPL:')[1].split()[0].replace(',', '')
                info['train_ppl'] = float(ppl)
            except:
                pass
        
        if 'Val PPL:' in line:
            try:
                ppl = line.split('Val PPL:')[1].split()[0].replace(',', '')
                info['val_ppl'] = float(ppl)
            except:
                pass
    
    return info

def print_status():
    """Imprime el estado actual del entrenamiento."""
    print("\n" + "="*70)
    print(f"   MONITOR DE ENTRENAMIENTO - {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)
    
    # 1. Revisar log
    log_lines = check_log_file()
    if log_lines:
        info = extract_epoch_info(log_lines)
        
        print(f"\n📊 PROGRESO ACTUAL:")
        if info['current_epoch']:
            print(f"  Época: {info['current_epoch']}/20")
        if info['progress']:
            print(f"  Avance: {info['progress']}")
        if info['train_ppl']:
            print(f"  Train PPL: {info['train_ppl']:,.2f}")
        if info['val_ppl']:
            print(f"  Val PPL: {info['val_ppl']:,.2f}")
            
            # Proyección
            if info['val_ppl'] < 100:
                print(f"  🎯 EXCELENTE - Ya debajo de 100 PPL!")
            elif info['val_ppl'] < 200:
                print(f"  ✅ MUY BIEN - Camino al objetivo 50-80")
            elif info['val_ppl'] < 500:
                print(f"  📈 PROGRESANDO - Mejora constante")
        
        print(f"  Última actualización: {info['last_update']}")
    else:
        print("\n⚠️  No se encontró train_log.txt")
    
    # 2. Revisar checkpoints
    checkpoints = check_checkpoints()
    if checkpoints:
        print(f"\n💾 CHECKPOINTS GUARDADOS ({len(checkpoints)}):")
        for ckpt in checkpoints[:3]:  # Mostrar solo los 3 más recientes
            print(f"  • {ckpt['name']}")
            print(f"    {ckpt['size_mb']:.1f} MB - {ckpt['modified']}")
    else:
        print("\n💾 CHECKPOINTS: Ninguno guardado aún")
    
    # 3. Revisar historial
    history = check_training_history()
    if history:
        epochs_completed = len(history)
        print(f"\n📈 HISTORIAL: {epochs_completed} época(s) completada(s)")
        
        if epochs_completed > 0:
            last_epoch = history[-1]
            print(f"  Última época:")
            print(f"    Train PPL: {last_epoch.get('train_ppl', 'N/A')}")
            print(f"    Val PPL: {last_epoch.get('val_ppl', 'N/A')}")
            
            # Tendencia
            if epochs_completed >= 2:
                prev_ppl = history[-2].get('val_ppl', 0)
                curr_ppl = last_epoch.get('val_ppl', 0)
                if prev_ppl > 0 and curr_ppl > 0:
                    improvement = ((prev_ppl - curr_ppl) / prev_ppl) * 100
                    if improvement > 0:
                        print(f"    📉 Mejora: -{improvement:.1f}% vs época anterior")
                    else:
                        print(f"    📊 Cambio: +{abs(improvement):.1f}% vs época anterior")
    
    # 4. Tiempo estimado
    if log_lines:
        print(f"\n⏱️  ESTIMACIÓN:")
        if info['current_epoch']:
            try:
                current = int(info['current_epoch'])
                remaining = 20 - current
                mins_per_epoch = 28  # Estimado
                total_mins = remaining * mins_per_epoch
                hours = total_mins // 60
                mins = total_mins % 60
                print(f"  Épocas restantes: {remaining}")
                print(f"  Tiempo estimado: ~{hours}h {mins}min")
            except:
                pass
    
    print("\n" + "="*70)
    print("Próxima actualización en 5 minutos...")
    print("="*70 + "\n")

def main():
    """Loop principal de monitoreo."""
    print("\n🔍 INICIANDO MONITOR DE ENTRENAMIENTO")
    print("Actualizaciones cada 5 minutos")
    print("Presiona Ctrl+C para detener\n")
    
    try:
        while True:
            print_status()
            time.sleep(300)  # 5 minutos = 300 segundos
    except KeyboardInterrupt:
        print("\n\n✋ Monitor detenido por el usuario")
        print("El entrenamiento continúa en background\n")

if __name__ == '__main__':
    main()
