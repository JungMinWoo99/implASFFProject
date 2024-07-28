import constant
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_img_deformation_from_mls(L_d, L_g, p): # [2,68], [2,68], [2,1]
    W_p = torch.diag(1.0/(torch.norm(L_d-p[:, None], dim=0)**2 + constant.eps)) #[68, 68]
    h_L_d = torch.cat([L_d, torch.ones(1, constant.g_landmarks_num)], dim=0)  # to homogeneous coordinate
    M_p_f = L_g @ W_p @ torch.transpose(h_L_d, 0, 1)
    M_p_b = torch.inverse(h_L_d @ W_p @ torch.transpose(h_L_d, 0, 1))
    M_p = M_p_f @ M_p_b
    return M_p


class MLS(nn.Module):
    def __init__(self):
        super(MLS, self).__init__()

    def forward(self, fea_g, L_d, L_g):
        grid_size = list(fea_g.shape)
        batch_size = grid_size[0]
        height = grid_size[2]
        width = grid_size[3]
        base_grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0), size=grid_size, align_corners=False)
        dynamic_grid = base_grid.clone()
        for i in range(height):
            for j in range(width):
                p = base_grid[0, i, j]
                M_p = get_img_deformation_from_mls(L_d, L_g, p)
                h_coord = torch.cat([p, torch.ones(1)], dim=0)
                p_transed = (M_p @ h_coord.unsqueeze(1)).squeeze(1)
                for h in range(batch_size):
                    dynamic_grid[h, i, j] = p_transed
        output_tensor = F.grid_sample(fea_g, dynamic_grid, mode='bilinear', padding_mode='zeros',
                                      align_corners=False)
        return output_tensor


if __name__ == '__main__':
    print("test")