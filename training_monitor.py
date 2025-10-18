"""Training monitoring utilities for stability detection and analysis."""


class TrainingMonitor:
    """Monitor training stability and detect issues"""
    def __init__(self, window_size=20, spike_threshold=0.5, oscillation_threshold=0.1):
        self.window_size = window_size
        self.spike_threshold = spike_threshold  # Relative increase that counts as a spike
        self.oscillation_threshold = oscillation_threshold  # StdDev threshold for oscillation detection
        
        self.loss_history = []
        self.grad_norm_history = []
        self.lr_history = []
        
        self.spike_count = 0
        self.oscillation_count = 0
        self.explosion_count = 0
        
    def update(self, loss, grad_norm, lr, step):
        """Update monitoring with new values"""
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.lr_history.append(lr)
        
        # Keep only recent history
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size:]
            self.grad_norm_history = self.grad_norm_history[-self.window_size:]
            self.lr_history = self.lr_history[-self.window_size:]
    
    def check_loss_spike(self):
        """Detect sudden loss increases"""
        if len(self.loss_history) < 3:
            return False, ""
            
        recent_loss = self.loss_history[-1]
        prev_loss = self.loss_history[-2]
        
        if prev_loss > 0 and (recent_loss - prev_loss) / prev_loss > self.spike_threshold:
            self.spike_count += 1
            return True, f"Loss spike detected: {prev_loss:.4f} → {recent_loss:.4f} (+{((recent_loss-prev_loss)/prev_loss)*100:.1f}%)"
        return False, ""
    
    def check_oscillation(self):
        """Detect loss oscillation in recent window"""
        if len(self.loss_history) < self.window_size:
            return False, ""
            
        recent_losses = self.loss_history[-self.window_size:]
        mean_loss = sum(recent_losses) / len(recent_losses)
        variance = sum((l - mean_loss) ** 2 for l in recent_losses) / len(recent_losses)
        std_dev = variance ** 0.5
        
        if mean_loss > 0 and (std_dev / mean_loss) > self.oscillation_threshold:
            self.oscillation_count += 1
            return True, f"Loss oscillation detected: std/mean = {(std_dev/mean_loss)*100:.1f}% (threshold: {self.oscillation_threshold*100:.1f}%)"
        return False, ""
    
    def check_gradient_explosion(self, max_grad_norm=1.0):
        """Detect gradient explosion"""
        if len(self.grad_norm_history) < 2:
            return False, ""
            
        recent_grad = self.grad_norm_history[-1]
        if recent_grad >= max_grad_norm * 0.98:  # Only trigger very close to actual clipping
            self.explosion_count += 1
            return True, f"Gradient norm near explosion: {recent_grad:.4f} (max: {max_grad_norm})"
        return False, ""
    
    def get_stats(self):
        """Get monitoring statistics"""
        if not self.loss_history:
            return "No data yet"
            
        recent_loss = self.loss_history[-1]
        recent_grad = self.grad_norm_history[-1] if self.grad_norm_history else 0
        recent_lr = self.lr_history[-1] if self.lr_history else 0
        
        return (f"Recent: loss={recent_loss:.4f}, grad_norm={recent_grad:.4f}, lr={recent_lr:.2e} | "
                f"Issues: spikes={self.spike_count}, oscillations={self.oscillation_count}, explosions={self.explosion_count}")