'''
출처: https://github.com/Jarvis73/Moving-Least-Squares/tree/master
'''
import numpy as np
import torch
from constant import *

def mls_affine_deformation_inv_final(height, width, channel, p, q, alpha=1.0, density=1.0):
    ''' Affine inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width * density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height * density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2 + 0.00000001, axis=1) ** alpha  # [ctrls, grow, gcol]
    w[w == np.inf] = 2 ** 31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    qhat = reshaped_q - qstar  # [ctrls, 2, grow, gcol]

    reshaped_phat = phat.reshape(ctrls, 2, 1, grow, gcol)  # [ctrls, 2, 1, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]
    pTwq = np.sum(reshaped_phat * reshaped_w * reshaped_qhat, axis=0)  # [2, 2, grow, gcol]
    try:
        inv_pTwq = np.linalg.inv(pTwq.transpose(2, 3, 0, 1))  # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(pTwq.transpose(2, 3, 0, 1))  # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(1, 1, grow, gcol)  # [1, 1, grow, gcol]
        adjoint = pTwq[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]  # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]  # [2, 2, grow, gcol]
        inv_pTwq = (adjoint / reshaped_det).transpose(2, 3, 0, 1)  # [grow, gcol, 2, 2]
    mul_left = reshaped_v - qstar  # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)  # [grow, gcol, 1, 2]
    mul_right = np.sum(reshaped_phat * reshaped_w * reshaped_phat2, axis=0)  # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.transpose(2, 3, 0, 1)  # [grow, gcol, 2, 2]
    temp = np.matmul(np.matmul(reshaped_mul_left, inv_pTwq), reshaped_mul_right)  # [grow, gcol, 1, 2]
    reshaped_temp = temp.reshape(grow, gcol, 2).transpose(2, 0, 1)  # [2, grow, gcol]

    # Get final image transfomer -- 3-D array
    transformers = reshaped_temp + pstar  # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf  # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    return transformers

def roi_mls_whole_final(feature, d_point, g_point, step=1):
    '''
    :param feature: itorchut guidance feature [C, H, W]
    :param d_point: landmark for degraded feature [N, 2]
    :param g_point: landmark for guidance feature [N, 2]
    :param step: step of landmark choose, number of control points: landmark_number/step
    :return: transformed feature [C, H, W]
    '''
    # feature 3 * 256 * 256

    channel = feature.size(0)
    height = feature.size(1)  # 256 * 256
    width = feature.size(2)

    # ignore the boarder point of face
    g_land = g_point[0::step, :]
    d_land = d_point[0::step, :]

    # mls

    # featureTmp = feature.permute(1,2,0)
    # grid, timg = mls_rigid_deformation_inv_wy(featureTmp.cpu(), height, width, channel, g_land.cpu(), d_land.cpu(), density=1.) # 2 * 256 * 256 # wenyu
    grid = mls_affine_deformation_inv_final(height, width, channel, g_land.cpu().numpy(), d_land.cpu().numpy(),
                                            density=1.)  #affine prefered
    # grid, timg_sim = mls_similarity_deformation_inv(featureTmp.cpu(), height, width, channel, g_land.cpu().numpy(), d_land.cpu().numpy(), density=1.) # similarity
    # grid, timg_rigid = mls_rigid_deformation_inv(featureTmp.cpu(), height, width, channel, g_land.cpu().numpy(), d_land.cpu().numpy(), density=1.) #rigid

    grid = (grid - height / 2) / (height / 2)

    gridNew = torch.from_numpy(grid[[1, 0], :, :]).float().permute(1, 2, 0).unsqueeze(0)

    if torch.cuda.is_available():
        gridNew = gridNew.cuda()
    # HWC -> CHW
    return gridNew

class MLS(torch.nn.Module):
    def __init__(self):
        super(MLS, self).__init__()

    def forward(self, fea_g, L_d, L_g):

        Grids = torch.zeros(fea_g.size(0), fea_g.size(2), fea_g.size(3), 2).type_as(fea_g)
        for i in range(fea_g.size(0)):
            Grids[i, :, :, :] = roi_mls_whole_final(
                fea_g[i, :, :, :], L_d[i, 17:, :], L_g[i, 17:, :]
            )

        MLS_Ref_Feature = torch.nn.functional.grid_sample(fea_g, Grids, mode='bilinear')  # v

        return MLS_Ref_Feature