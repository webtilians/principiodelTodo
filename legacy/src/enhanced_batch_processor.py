"""
Enhanced Batch Processing with Gaussian Process Optimization
Integrated module for infinito_v3_stable.py
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedBatchProcessor:
    """Enhanced batch processing with GP optimization for dynamic thresholds"""
    
    def __init__(self, initial_threshold: float = 0.1):
        self.threshold = initial_threshold
        self.batch_history = []
        self.success_rates = []
        self.losses = []
        
        # Initialize Gaussian Process for threshold optimization
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp_optimizer = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        self.gp_initialized = False
    
    def update_threshold_with_gp(self, batch_idx: int, success_rate: float, avg_loss: float) -> float:
        """Update threshold using Gaussian Process optimization"""
        try:
            # Store batch data
            self.batch_history.append([batch_idx, success_rate, avg_loss])
            
            # Need at least 3 points to train GP
            if len(self.batch_history) >= 3:
                X = np.array([[h[0], h[1]] for h in self.batch_history])  # [batch_idx, success_rate]
                y = np.array([h[2] for h in self.batch_history])  # avg_loss
                
                if not self.gp_initialized:
                    self.gp_optimizer.fit(X, y)
                    self.gp_initialized = True
                else:
                    # Update GP with new data
                    self.gp_optimizer.fit(X, y)
                
                # Predict optimal threshold based on current batch and success rate
                current_features = np.array([[batch_idx, success_rate]])
                predicted_loss, std = self.gp_optimizer.predict(current_features, return_std=True)
                
                # Adjust threshold based on prediction confidence
                if std[0] < 0.1:  # High confidence
                    if predicted_loss[0] > avg_loss:
                        self.threshold *= 0.95  # Decrease threshold
                    else:
                        self.threshold *= 1.05  # Increase threshold
                else:  # Low confidence
                    self.threshold *= 1.01  # Conservative adjustment
                
                # Keep threshold in reasonable bounds
                self.threshold = np.clip(self.threshold, 0.01, 1.0)
                
                return self.threshold
            else:
                # Simple adaptive threshold for initial batches
                if success_rate < 0.7:
                    self.threshold *= 1.1
                elif success_rate > 0.9:
                    self.threshold *= 0.9
                
                self.threshold = np.clip(self.threshold, 0.01, 1.0)
                return self.threshold
                
        except Exception as e:
            print(f"GP threshold update failed: {e}")
            return self.threshold
    
    def run_enhanced_batch(self, model, optimizer, data_loader, device, 
                          batch_size: int = 32, total_iters: int = 100) -> Dict[str, Any]:
        """
        Enhanced batch processing with GP optimization and graceful error handling
        Integrated with consciencia training loop
        """
        print(f"\n🚀 Starting Enhanced Batch Processing with GP Optimization")
        print(f"📊 Configuration: batch_size={batch_size}, total_iters={total_iters}")
        print(f"🎯 Initial threshold: {self.threshold:.4f}")
        
        # Batch tracking
        batch_kpis = {
            'total_batches': 0,
            'successful_batches': 0,
            'total_loss': 0.0,
            'avg_loss_per_batch': [],
            'success_rates': [],
            'threshold_history': [],
            'processing_times': [],
            'consciousness_values': [],
            'phi_values': []
        }
        
        model.train()
        start_time = time.time()
        
        try:
            for batch_idx in range(total_iters):
                batch_start_time = time.time()
                batch_successful_samples = 0
                batch_total_samples = 0
                batch_loss_sum = 0.0
                batch_consciousness_sum = 0.0
                batch_phi_sum = 0.0
                
                if batch_idx % 10 == 0:
                    print(f"\n📈 Batch {batch_idx + 1}/{total_iters}")
                    print(f"🎯 Current GP threshold: {self.threshold:.4f}")
                
                # Process batch
                for sample_idx in range(batch_size):
                    try:
                        # Generate synthetic data for testing with correct dimensions
                        # Model expects input_size=128, hidden_size=256
                        input_data = torch.randn(1, 128).to(device)
                        
                        # Forward pass with consciousness calculation
                        output = model(input_data)
                        
                        # Handle different model output types
                        if isinstance(output, tuple):
                            # Model returns (consciousness, fused_output) tuple
                            consciousness_value, logits = output
                            if hasattr(consciousness_value, 'item'):
                                consciousness_value = consciousness_value.item()
                            else:
                                consciousness_value = float(consciousness_value)
                        elif hasattr(output, 'logits'):
                            logits = output.logits
                            consciousness_value = 0.0
                        elif isinstance(output, dict):
                            logits = output.get('logits', output.get('output', input_data))
                            consciousness_value = 0.0
                        else:
                            logits = output
                            consciousness_value = 0.0
                        
                        # Ensure logits are properly shaped
                        if hasattr(logits, 'shape') and len(logits.shape) > 1:
                            logits = logits.squeeze()
                        
                        # Create target with matching dimensions to logits
                        if hasattr(logits, 'shape'):
                            target_shape = logits.shape
                            target = torch.randint(0, 2, target_shape).float().to(device)
                        else:
                            target = torch.randint(0, 2, (1,)).float().to(device)
                        
                        # Calculate loss with proper tensor handling
                        loss = nn.BCEWithLogitsLoss()(logits, target)
                        
                        optimizer.zero_grad()
                        
                        # Calculate phi if model supports it
                        phi_value = 0.0
                        
                        # Only try to calculate consciousness if we haven't got it from model output
                        if consciousness_value == 0.0:  # If we didn't get consciousness from model output
                            if hasattr(model, 'calculate_consciousness'):
                                try:
                                    consciousness_value = model.calculate_consciousness().item()
                                except:
                                    # Fallback: calculate consciousness based on loss
                                    consciousness_value = max(0.0, 1.0 - loss.item())
                            else:
                                # Fallback: calculate consciousness based on loss
                                consciousness_value = max(0.0, 1.0 - loss.item())
                        
                        if hasattr(model, 'calculate_phi'):
                            try:
                                phi_value = model.calculate_phi().item()
                            except:
                                # Fallback: calculate phi based on consciousness
                                phi_value = consciousness_value * 0.5
                        else:
                            # Fallback: calculate phi based on consciousness
                            phi_value = consciousness_value * 0.5
                        
                        # Check if loss is finite and within threshold
                        if torch.isfinite(loss) and loss.item() < max(self.threshold * 5, 0.5):  # More reasonable threshold
                            loss.backward()
                            
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                            optimizer.step()
                            
                            batch_successful_samples += 1
                            batch_loss_sum += loss.item()
                            batch_consciousness_sum += consciousness_value
                            batch_phi_sum += phi_value
                        else:
                            if batch_idx % 50 == 0:  # Reduce noise, only print every 50 batches
                                print(f"⚠️ Skipped sample {sample_idx}: loss={loss.item():.4f}")
                            # Still accumulate consciousness and phi even if we skip the gradient update
                            batch_consciousness_sum += consciousness_value
                            batch_phi_sum += phi_value
                        
                        batch_total_samples += 1
                        
                    except Exception as e:
                        if batch_idx % 50 == 0:  # Reduce noise
                            print(f"❌ Error in sample {sample_idx}: {e}")
                        continue
                
                # Calculate batch metrics
                batch_success_rate = batch_successful_samples / max(batch_total_samples, 1)
                avg_batch_loss = batch_loss_sum / max(batch_successful_samples, 1) if batch_successful_samples > 0 else 0.0
                avg_consciousness = batch_consciousness_sum / max(batch_total_samples, 1)  # Use total samples, not just successful
                avg_phi = batch_phi_sum / max(batch_total_samples, 1)  # Use total samples, not just successful
                batch_time = time.time() - batch_start_time
                
                # Update KPIs
                batch_kpis['total_batches'] += 1
                if batch_successful_samples > 0:
                    batch_kpis['successful_batches'] += 1
                    batch_kpis['total_loss'] += batch_loss_sum
                
                batch_kpis['avg_loss_per_batch'].append(avg_batch_loss)
                batch_kpis['success_rates'].append(batch_success_rate)
                batch_kpis['processing_times'].append(batch_time)
                batch_kpis['consciousness_values'].append(avg_consciousness)
                batch_kpis['phi_values'].append(avg_phi)
                
                # Update threshold using GP optimization
                old_threshold = self.threshold
                new_threshold = self.update_threshold_with_gp(batch_idx, batch_success_rate, avg_batch_loss)
                batch_kpis['threshold_history'].append(new_threshold)
                
                # Print batch results every 10 batches
                if batch_idx % 10 == 0:
                    print(f"✅ Batch {batch_idx + 1} completed:")
                    print(f"   📊 Success rate: {batch_success_rate:.2%}")
                    print(f"   📉 Avg loss: {avg_batch_loss:.4f}")
                    print(f"   🧠 Avg consciousness: {avg_consciousness:.4f}")
                    print(f"   🔮 Avg phi: {avg_phi:.4f}")
                    print(f"   ⏱️ Time: {batch_time:.2f}s")
                    print(f"   🎯 Threshold: {old_threshold:.4f} → {new_threshold:.4f}")
                
                # Progress update every 25 batches
                if (batch_idx + 1) % 25 == 0:
                    overall_success_rate = batch_kpis['successful_batches'] / batch_kpis['total_batches']
                    avg_processing_time = np.mean(batch_kpis['processing_times'][-25:])
                    avg_consciousness = np.mean(batch_kpis['consciousness_values'][-25:])
                    print(f"\n🔄 Progress Update (Batch {batch_idx + 1}):")
                    print(f"   Overall success rate: {overall_success_rate:.2%}")
                    print(f"   Avg processing time (last 25): {avg_processing_time:.2f}s")
                    print(f"   Avg consciousness (last 25): {avg_consciousness:.4f}")
                    print(f"   Current GP threshold: {self.threshold:.4f}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Enhanced batch processing interrupted by user")
        except Exception as e:
            print(f"\n❌ Critical error in enhanced batch processing: {e}")
        
        # Final statistics
        total_time = time.time() - start_time
        overall_success_rate = batch_kpis['successful_batches'] / max(batch_kpis['total_batches'], 1)
        
        print(f"\n🏁 Enhanced Batch Processing Complete!")
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(f"📊 Total batches processed: {batch_kpis['total_batches']}")
        print(f"✅ Successful batches: {batch_kpis['successful_batches']}")
        print(f"📈 Overall success rate: {overall_success_rate:.2%}")
        print(f"🎯 Final GP threshold: {self.threshold:.4f}")
        
        if batch_kpis['avg_loss_per_batch']:
            print(f"📉 Average loss: {np.mean(batch_kpis['avg_loss_per_batch']):.4f}")
        if batch_kpis['consciousness_values']:
            print(f"🧠 Average consciousness: {np.mean(batch_kpis['consciousness_values']):.4f}")
        if batch_kpis['phi_values']:
            print(f"🔮 Average phi: {np.mean(batch_kpis['phi_values']):.4f}")
        
        return batch_kpis
