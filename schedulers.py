from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math

class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    """
    Cosine annealing with warm restarts, decaying peak learning rates, and linear warmup.
    
    At each restart, the peak learning rate is multiplied by decay_factor,
    so the restarts gradually decay over time. Includes a linear warmup period.
    """
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, decay_factor=0.9, warmup_epochs=0, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_0 (int): Number of iterations for the first restart.
            T_mult (int, optional): A factor increases T_i after a restart. Default: 1.
            eta_min (float, optional): Minimum learning rate. Default: 0.
            decay_factor (float, optional): Factor to decay peak LR at each restart. Default: 0.9.
            warmup_epochs (int, optional): Number of epochs for linear warmup. Default: 0.
            last_epoch (int, optional): The index of last epoch. Default: -1.
        """
        self.decay_factor = decay_factor
        self.warmup_epochs = warmup_epochs
        self.restart_count = 0
        super(CosineAnnealingWarmRestartsDecay, self).__init__(optimizer, T_0, T_mult, eta_min, last_epoch)

    def get_lr(self):
        # Handle warmup phase (only during the first cycle, before any restarts)
        if self.restart_count == 0 and self.T_cur < self.warmup_epochs:
            # Linear warmup phase
            warmup_factor = (self.T_cur + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # After warmup or after restarts, calculate the current peak LR based on restart count
        current_peak_lr = [base_lr * (self.decay_factor ** self.restart_count) for base_lr in self.base_lrs]
        
        # Use cosine annealing with the current peak LR
        if self.restart_count == 0:
            # First cycle: account for warmup
            cycle_step = self.T_cur - self.warmup_epochs
        else:
            # After restarts: no warmup, use T_cur directly
            cycle_step = self.T_cur
            
        return [
            self.eta_min + (peak_lr - self.eta_min) * 
            (1 + math.cos(math.pi * cycle_step / self.T_i)) / 2
            for peak_lr in current_peak_lr
        ]

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            
            # Check if we need to restart
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur % self.T_i
                self.T_i = self.T_i * self.T_mult
                self.restart_count += 1
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        # Update learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]