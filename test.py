# import torch

# # Initialize a 3x3 tensor
# tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# reshaped_original = tensor.reshape(-1)

# # Rotate it 90 degrees clockwise
# rotated_tensor = torch.rot90(tensor, 1)

# # Reshape the rotated tensor into a 1D tensor
# reshaped_tensor = rotated_tensor.reshape(-1)

# print(f"Original tensor:\n{tensor}")
# print(f"Reshaped original tensor:\n{reshaped_original}")
# print(f"Rotated tensor:\n{rotated_tensor}")
# print(f"Reshaped tensor:\n{reshaped_tensor}")



# import torch
import pdb
# # Initialize your tensor
# m  = torch.tensor([[0, 1, 0],[1, 0, 1],[0, 1, 0]], dtype=torch.int)

# # Get the indices of the ones
# indices = torch.where(m)

# # Initialize a tensor filled with zeros
# splits = torch.zeros(indices[0].size(0), *m.size(), dtype=torch.int)

# # Use advanced indexing to place the ones at the correct places
# splits[torch.arange(indices[0].size(0)), indices[0], indices[1]] = 1

# print(splits)
# pdb.set_trace()
# ...


import torch

# Initialize your tensor
m  = torch.tensor([[[0, 1, 0],[1, 0, 1],[0, 1, 0]],
                   [[0, 1, 0],[1, 0, 1],[0, 1, 0]]], dtype=torch.int)

# Get the indices of the ones
indices = torch.where(m)

# Initialize a tensor filled with zeros
splits = torch.zeros((indices[0].size(0),) + m.shape, dtype=torch.int)

# Use advanced indexing to place the ones at the correct places
splits[(torch.arange(indices[0].size(0)),) + indices] = 1

print(splits)
pdb.set_trace()