from constant import *
import torch

class WLS:
    def __init__(self, init_w=None):
        if init_w is None:
            rand_tensor = torch.rand(g_landmarks_num) * 0.01
        else:
            rand_tensor = init_w
        self.w = rand_tensor.to(default_device).requires_grad_(True)  # define weight
        self.w_diag = torch.diag(self.w).to(default_device)

    def compute_loss(self, L_d, L_g_list):
        distance_list = []
        for L_k in L_g_list:
            # cal affine transform
            h_L_k = torch.cat([L_k, torch.ones(1, g_landmarks_num).to(default_device)],
                              dim=0)  # to homogeneous coordinate
            inner_matrix = h_L_k @ self.w_diag @ h_L_k.transpose(0, 1)
            inverse_matrix = torch.inverse(inner_matrix)
            A = L_d @ self.w_diag @ h_L_k.transpose(0, 1) @ inverse_matrix
            error = (A @ h_L_k - L_d) * self.w
            affine_distance = torch.linalg.norm(error, ord='fro') ** 2
            distance_list.append(affine_distance)
        distance_tensor = torch.stack(distance_list).to(default_device)
        shortest_distance, min_index = torch.min(distance_tensor, dim=0)
        remaining_distances = torch.cat([distance_tensor[:min_index], distance_tensor[min_index + 1:]])
        loss_tensor = torch.max(torch.zeros_like(remaining_distances), 1.0 - (remaining_distances - shortest_distance))
        loss = torch.sum(loss_tensor)
        return loss, min_index
