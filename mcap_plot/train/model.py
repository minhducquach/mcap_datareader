import torch
from torch import nn
import math

torch.manual_seed(16)

class MLP(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.nl1 = nn.Softplus()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print(x)
        x = self.fc1(x)
        x = self.nl1(x)
        x = self.fc2(x)
        # print(x)
        return x

# import torch
# from torch import nn

# class MLP(nn.Module):
#     def __init__(self, input_size=2, hidden_size=64, output_size=1):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.nl1 = nn.Softplus()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#         # Learnable parameters A and B
#         self.A = nn.Parameter(torch.randn(1))  # Initialize A as a learnable parameter
#         self.B = nn.Parameter(torch.randn(1))  # Initialize B as a learnable parameter
#         # self.C = nn.Parameter(torch.randn(1))  # Initialize B as a learnable parameter

#     def forward(self, x):
#         # Separate features
#         distance, cosine_a, cosine_b = x[:, 0], x[:, 1], x[:, 2]

#         # Apply MLP to cosine_a and cosine_b
#         # x_cosine = torch.stack((cosine_a, cosine_b), dim=1)
#         # print(x_cosine)
#         # print(distance)
#         # print(x_cosine.shape)
#         mlp_output = self.fc1(torch.abs(cosine_b.unsqueeze(1)))
#         # print(cosine_b.unsqueeze(1).shape)
#         # mlp_output = self.fc1(cosine_b.unsqueeze(1))
#         # print(mlp_output.shape)
#         mlp_output = self.nl1(mlp_output)
#         # print(mlp_output.shape)
#         mlp_output = self.fc2(mlp_output)
#         # print(mlp_output.shape)

#         # Calculate output based on the given formula
#         output = (mlp_output.squeeze() / (math.exp(-self.A) + distance**2) + self.B) * 0.6/3.14 * cosine_a
#         # print(output)
#         # print("FINAL", output.shape)
#         return output