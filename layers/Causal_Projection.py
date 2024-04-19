import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalProjectionLayer(nn.Module):
    def __init__(self, d_model, p_steps, sequence_length):
        super(CausalProjectionLayer, self).__init__()
        self.A = nn.Parameter(torch.randn(p_steps, d_model))
        self.A_prime = nn.Parameter(torch.randn(p_steps, d_model))
        self.delta_t = nn.Parameter(torch.ones(sequence_length, 1))

    def forward(self, x, update_delta=False):
        B, L, D = x.shape
        p_steps = self.A.shape[0]

        delta_t_expanded = self.delta_t.expand(L, B, D).permute(1, 0, 2)

        delta_x = torch.zeros_like(x)
        for j in range(1, p_steps + 1):
            delta_x[:, j:, :] = x[:, j:, :] - x[:, :-j, :]
            delta_x[:, :j, :] = 0

            x = x + self.A[j-1].unsqueeze(0).unsqueeze(2) * delta_x

            derivative = delta_x / delta_t_expanded
            x = x + self.A_prime[j-1].unsqueeze(0).unsqueeze(2) * derivative

        if update_delta:
            delta_t_loss = self.calculate_delta_t_loss(x, target)
            self.delta_t.grad = torch.autograd.grad(delta_t_loss, self.delta_t, create_graph=True)[0]
            self.delta_t.data = self.delta_t.data - learning_rate * self.delta_t.grad

        return x

    def calculate_delta_t_loss(self, predictions, target):
        loss = F.mse_loss(predictions, target)
        return loss