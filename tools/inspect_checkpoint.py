import sys
import torch
from pathlib import Path

def inspect(path):
    p = Path(path)
    if not p.exists():
        print(f"ERROR: {p} no encontrado")
        return 1
    try:
        ck = torch.load(str(p), map_location='cpu')
    except Exception as e:
        print(f"ERROR al cargar checkpoint: {e}")
        return 2

    print(f"Checkpoint: {p.name}\n")
    # print available top-level keys
    print("Top-level keys:")
    for k in sorted(ck.keys()):
        print(f"  - {k}")
    print("")

    def get(k):
        return ck.get(k, None)

    print("Epoch:", get('epoch'))
    print("Train loss:", get('train_loss'))
    print("Val loss:", get('val_loss'))
    print("Train PPL:", get('train_ppl'))
    print("Val PPL:", get('val_ppl'))
    print("Config keys present:", list(get('config', {}).keys()) if isinstance(get('config', {}), dict) else 'N/A')

    # If model_state_dict exists, show any keys related to phi or learnable
    msd = get('model_state_dict')
    if msd is None and 'state_dict' in ck:
        msd = ck.get('state_dict')
    if msd is None:
        print("No se encontró model_state_dict en checkpoint.")
    else:
        # count params
        keys = list(msd.keys())
        print(f"model_state_dict keys: {len(keys)} entries (showing phi-related keys):")
        phi_keys = [k for k in keys if 'phi' in k.lower() or 'learnable' in k.lower()]
        for k in phi_keys[:50]:
            print(f"  - {k}")
        if not phi_keys:
            print("  (no se encontraron keys relacionadas con PHI/learnable en model_state_dict)")

    # Quick heuristic: was this the buggy quick_experiment? Check for presence of 'collate_fn' or a flag
    if 'notes' in ck:
        print('\nNotes in checkpoint:')
        print(ck['notes'])

    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Uso: python inspect_checkpoint.py <path_to_checkpoint>')
        sys.exit(3)
    sys.exit(inspect(sys.argv[1]))
