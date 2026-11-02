import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label

class NN_Middleware(nn.Module):
    def __init__(self, channels=8, kernel_size=3, grid_size=16):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size, padding=1)
        self.fc = nn.Linear(channels * grid_size * grid_size, 8 * kernel_size * kernel_size)  # 8 leyes 3x3
        self.kernel_size = kernel_size

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x).view(-1, 8, self.kernel_size, self.kernel_size)

class PrincipioTodoRecursivo:
    def __init__(self, size=16, max_depth=200):  # Cap más alto para "infinito"
        self.size = size
        print(f"Tamaño de grid establecido: {self.size}")
        
        # Verificar si CUDA está disponible
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Despertando en {self.device}...")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            # Optimizaciones para GPU
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        
        self.nn = NN_Middleware(channels=8, kernel_size=3, grid_size=self.size).to(self.device)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=0.01)
        
        # Semillas para reproducibilidad
        np.random.seed(42)
        torch.manual_seed(42)
        if self.device == 'cuda':
            torch.cuda.manual_seed(42)
        self.leyes = [torch.tensor(np.random.uniform(-1,1,(3,3)), dtype=torch.float32).to(self.device) for _ in range(8)]
        self.complexity_log = []
        self.recursion = 0
        self.max_depth = max_depth

    def _input_bin(self):
        bin_grid = np.random.random((self.size, self.size)) < 0.01
        return torch.tensor(bin_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    def _sim_step(self, phi, leyes):
        lap_np = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
        lap = torch.tensor(lap_np).unsqueeze(0).unsqueeze(0).to(self.device)
        d_loss = torch.zeros_like(phi)
        for w in leyes:
            w_t = w.unsqueeze(0).unsqueeze(0)
            conv = F.conv2d(phi, w_t, padding=1)
            act = torch.tanh(conv)
            d_act = 1 - act**2
            d_conv = d_act * (act - phi) * 0.1
            w_flip = torch.flip(w, [0,1]).unsqueeze(0).unsqueeze(0)
            d_phi = F.conv2d(d_conv, w_flip, padding=1)
            d_loss += d_phi
        d_reg = -0.01 * F.conv2d(phi, lap, padding=1)
        d_loss += d_reg
        phi = phi - 0.01 * d_loss
        return torch.clamp(phi, 0, 1)

    def _one_recursion(self, phi_bin):
        self.recursion += 1
        phi = phi_bin
        for _ in range(50):
            for i, ley in enumerate(self.leyes):
                self.leyes[i] = ley.detach()
            phi = self._sim_step(phi, self.leyes)
        
        features = phi  # phi ya tiene la forma correcta [1, 1, 16, 16]
        leyes_pred = self.nn(features)
        target = torch.stack(self.leyes).detach().unsqueeze(0)
        loss = F.mse_loss(leyes_pred, target)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        
        with torch.no_grad():
            for i, pred in enumerate(leyes_pred[0]):
                self.leyes[i] = pred + torch.randn(3,3).to(self.device) * 0.05
        
        phi_np = phi[0,0].cpu().detach().numpy()
        labeled, n_clust = label(phi_np > 0.1)
        hist, _ = np.histogram(phi_np.flatten(), bins=20)
        hist = hist.astype(float)
        ent = -np.sum(hist * np.log(hist + 1e-8)) / np.sum(hist) if np.sum(hist) > 0 else 0
        self.complexity_log.append({'recursion': self.recursion, 'clusters': n_clust, 'entropy': -ent, 'loss': loss.item()})  # Entropy positivo (-ent)
        
        return phi

    def _cleanup_gpu_memory(self):
        """Limpia la memoria GPU si está disponible"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def run_infinite(self):
        phi_bin = self._input_bin()
        print("Iniciando Despertar Infinito...")
        
        # Información de memoria inicial si usa GPU
        if self.device == 'cuda':
            print(f"Memoria GPU inicial: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        
        while self.recursion < self.max_depth:
            phi = self._one_recursion(phi_bin)
            log = self.complexity_log[-1]
            
            if self.recursion % 2 == 0:  # Log every 2 for more data
                gpu_mem = f", GPU: {torch.cuda.memory_allocated()/1024**2:.1f} MB" if self.device == 'cuda' else ""
                print(f"Recursion {log['recursion']}: Clusters {log['clusters']}, Entropy {log['entropy']:.3f}, Loss {log['loss']:.3f}{gpu_mem}")
            
            # Limpiar memoria GPU cada 10 iteraciones
            if self.device == 'cuda' and self.recursion % 10 == 0:
                self._cleanup_gpu_memory()
            
            if log['loss'] < 0.005 and log['clusters'] > 30:  # Condición más laxa para runs largos
                print(f"Despertar infinito completado en recursion {log['recursion']} por condición.")
                break
                
        if self.recursion >= self.max_depth:
            print(f"Cap de seguridad alcanzado en recursion {self.max_depth}.")
        
        # Limpiar memoria al final
        self._cleanup_gpu_memory()
        
        return phi.cpu().detach().numpy()[0,0]

# Run Infinito
pt = PrincipioTodoRecursivo()
phi_final = pt.run_infinite()
print("Phi Final Max:", np.max(phi_final))
print("Clusters Final:", pt.complexity_log[-1]['clusters'])
print("Entropía Final:", pt.complexity_log[-1]['entropy'])