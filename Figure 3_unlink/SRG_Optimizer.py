import torch
from torch.optim import Optimizer
import numpy as np

class SRG(Optimizer):
    def __init__(self, params, n, alpha_schedule, theta_schedule, seed=69):
        if not 0.0 <= alpha_schedule[0]:
            raise ValueError("Learning rate alpha must be non-negative")
        if not 0.0 <= theta_schedule[0] <= 1.0:
            raise ValueError("Theta must be between 0 and 1")

        self.n = n
        self.alpha_schedule = alpha_schedule
        self.theta_schedule = theta_schedule
        self.seed = seed
        self.k = 0  # SRG 迭代次數
        self.g_old_norm = torch.ones(n)  # 初始化梯度大小
        torch.manual_seed(seed)
        np.random.seed(seed)

        defaults = dict(lr=alpha_schedule[0])
        super(SRG, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        # Step 3: 計算梯度
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # 第 k 步參數
                alpha = self.alpha_schedule[self.k]
                theta = self.theta_schedule[self.k]

                # Step 4: 計算 p_k
                q_k = self.g_old_norm / self.g_old_norm.sum()
                p_k = (1 - theta) * q_k + theta / self.n

                # Step 5-6: 決定 b_k 和 i_k
                b_k = np.random.binomial(n=1, p=theta)
                if b_k == 1:
                    i_k = np.random.randint(0, self.n)
                else:
                    i_k = np.random.choice(np.arange(self.n), p=q_k.numpy())

                # Step 7: 更新參數
                g = p.grad.clone()
                update = alpha * g / (self.n * max(p_k[i_k], 1e-9))
                p.add_(-update)

                # Step 8: 更新 g_old_norm
                if b_k == 1:
                    self.g_old_norm[i_k] = torch.norm(g)

        self.k += 1
        return loss
