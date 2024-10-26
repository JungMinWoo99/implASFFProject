'''
출처: https://github.com/Jarvis73/Moving-Least-Squares/tree/master
'''
import torch
from constant import *

"""
Image deformation using moving least squares.

    * Affine deformation
    * Similarity deformation
    * Rigid deformation

For more details please refer to the Chinese documentation: 

    ./doc/Image Deformation.pdf

or the original paper: 

    Image deformation using moving least squares
    Schaefer, Mcphail, Warren. 

Note:
    In the original paper, the author missed the weight w_j in formular (5).
    In addition, all the formulars in section 2.1 miss the w_j. 
    And I have corrected this point in my documentation.

@author: Jian-Wei ZHANG
@email: zjw.cs@zju.edu.cn
@date: 2022/01/12: PyTorch implementation
"""


def interp(xnew, x, y):
    """
    Linear 1D interpolation on the GPU for Pytorch.
    This function returns interpolated values of a set of 1-D functions at
    the desired query points `xnew`.
    This function is working similarly to Matlab™ or scipy functions with
    the `linear` interpolation mode on, except that it parallelises over
    any number of desired interpolation problems.
    The code will run on GPU if all the tensors provided are on a cuda
    device.
    Parameters
    ----------
    x : (N, ) or (D, N) Pytorch Tensor
        A 1-D or 2-D tensor of real values.
    y : (N,) or (D, N) Pytorch Tensor
        A 1-D or 2-D tensor of real values. The length of `y` along its
        last dimension must be the same as that of `x`
    xnew : (P,) or (D, P) Pytorch Tensor
        A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
        _both_ `x` and `y` are 1-D. Otherwise, its length along the first
        dimension must be the same as that of whichever `x` and `y` is 2-D.
    """
    # making the vectors at least 2D
    is_flat = {}
    require_grad = {}
    v = {}
    device = []
    eps = torch.finfo(y.dtype).eps
    for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
        assert len(vec.shape) <= 2, 'interp1d: all inputs must be ' \
                                    'at most 2-D.'
        if len(vec.shape) == 1:
            v[name] = vec[None, :]
        else:
            v[name] = vec
        is_flat[name] = v[name].shape[0] == 1
        require_grad[name] = vec.requires_grad
        device = list(set(device + [str(vec.device)]))
    assert len(device) == 1, 'All parameters must be on the same device.'
    device = device[0]

    # Checking for the dimensions
    assert (v['x'].shape[1] == v['y'].shape[1]
            and (
                    v['x'].shape[0] == v['y'].shape[0]
                    or v['x'].shape[0] == 1
                    or v['y'].shape[0] == 1
            )
            ), ("x and y must have the same number of columns, and either "
                "the same number of row or one of them having only one "
                "row.")

    reshaped_xnew = False
    if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
            and (v['xnew'].shape[0] > 1)):
        # if there is only one row for both x and y, there is no need to
        # loop over the rows of xnew because they will all have to face the
        # same interpolation problem. We should just stack them together to
        # call interp1d and put them back in place afterwards.
        original_xnew_shape = v['xnew'].shape
        v['xnew'] = v['xnew'].contiguous().view(1, -1)
        reshaped_xnew = True

    # identify the dimensions of output and check if the one provided is ok
    D = max(v['x'].shape[0], v['xnew'].shape[0])
    shape_ynew = (D, v['xnew'].shape[-1])

    ynew = torch.zeros(*shape_ynew, device=device)

    # moving everything to the desired device in case it was not there
    # already (not handling the case things do not fit entirely, user will
    # do it if required.)
    for name in v:
        v[name] = v[name].to(device)

    # calling searchsorted on the x values.
    ind = ynew.long()

    # expanding xnew to match the number of rows of x in case only one xnew is
    # provided
    if v['xnew'].shape[0] == 1:
        v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

    torch.searchsorted(v['x'].contiguous(),
                       v['xnew'].contiguous(), out=ind)

    # the `-1` is because searchsorted looks for the index where the values
    # must be inserted to preserve order. And we want the index of the
    # preceeding value.
    ind -= 1
    # we clamp the index, because the number of intervals is x.shape-1,
    # and the left neighbour should hence be at most number of intervals
    # -1, i.e. number of columns in x -2
    ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

    # helper function to select stuff according to the found indices.
    def sel(name):
        if is_flat[name]:
            return v[name].contiguous().view(-1)[ind]
        return torch.gather(v[name], 1, ind)

    # activating gradient storing for everything now
    enable_grad = False
    saved_inputs = []
    for name in ['x', 'y', 'xnew']:
        if require_grad[name]:
            enable_grad = True
            saved_inputs += [v[name]]
        else:
            saved_inputs += [None, ]
    # assuming x are sorted in the dimension 1, computing the slopes for
    # the segments
    is_flat['slopes'] = is_flat['x']
    # now we have found the indices of the neighbors, we start building the
    # output. Hence, we start also activating gradient tracking
    v['slopes'] = (
            (v['y'][:, 1:] - v['y'][:, :-1])
            /
            (eps + (v['x'][:, 1:] - v['x'][:, :-1]))
    )

    # now build the linear interpolation
    ynew = sel('y') + sel('slopes') * (v['xnew'] - sel('x'))

    if reshaped_xnew:
        ynew = ynew.view(original_xnew_shape)

    return ynew


def mls_affine_deformation(vy, vx, p, q, alpha=1.0, eps=1e-8):
    """
    Affine deformation

    Parameters
    ----------
    vy, vx: torch.Tensor
        coordinate grid, generated by torch.meshgrid(gridX, gridY)
    p: torch.Tensor
        an array with size [n, 2], original control points, in (y, x) formats
    q: torch.Tensor
        an array with size [n, 2], final control points, in (y, x) formats
    alpha: float
        parameter used by weights
    eps: float
        epsilon

    Return
    ------
        A deformed image.
    """

    device = q.device
    # Change (x, y) to (row, col)
    q = q.short()
    p = p.short()

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Precompute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_v = torch.cat((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)), dim=0)  # [2, grow, gcol]

    w = 1.0 / (torch.sum((reshaped_p - reshaped_v).float() ** 2, dim=1) + eps) ** alpha  # [ctrls, grow, gcol]
    w /= torch.sum(w, dim=0, keepdim=True)  # [ctrls, grow, gcol]

    pstar = torch.zeros((2, grow, gcol), dtype=torch.float32).to(device)
    for i in range(ctrls):
        pstar += w[i] * reshaped_p[i]  # [2, grow, gcol]

    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    phat = phat.reshape(ctrls, 2, 1, grow, gcol)  # [ctrls, 2, 1, grow, gcol]
    phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]
    pTwp = torch.zeros((2, 2, grow, gcol), dtype=torch.float32).to(device)
    for i in range(ctrls):
        pTwp += phat[i] * reshaped_w[i] * phat1[i]
    del phat1

    try:
        inv_pTwp = torch.inverse(pTwp.permute(2, 3, 0, 1))  # [grow, gcol, 2, 2]
        flag = False
    except RuntimeError as error:
        # print(error)
        flag = True
        det = torch.det(pTwp.permute(2, 3, 0, 1))  # [grow, gcol]
        det[det < 1e-8] = float("Inf")
        reshaped_det = det.reshape(1, 1, grow, gcol)  # [1, 1, grow, gcol]
        adjoint = pTwp[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]  # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]  # [2, 2, grow, gcol]
        inv_pTwp = (adjoint / reshaped_det).permute(2, 3, 0, 1)  # [grow, gcol, 2, 2]

    mul_left = reshaped_v - pstar  # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).permute(2, 3, 0, 1)  # [grow, gcol, 1, 2]
    mul_right = torch.mul(reshaped_w, phat, out=phat)  # [ctrls, 2, 1, grow, gcol]
    reshaped_mul_right = mul_right.permute(0, 3, 4, 1, 2)  # [ctrls, grow, gcol, 2, 1]
    out_A = mul_right.reshape(2, ctrls, grow, gcol, 1, 1)[0]  # [ctrls, grow, gcol, 1, 1]
    A = torch.matmul(torch.matmul(reshaped_mul_left, inv_pTwp), reshaped_mul_right,
                     out=out_A)  # [ctrls, grow, gcol, 1, 1]
    A = A.reshape(ctrls, 1, grow, gcol)  # [ctrls, 1, grow, gcol]
    del mul_right, reshaped_mul_right, phat

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    qstar = torch.zeros((2, grow, gcol), dtype=torch.float32).to(device)
    for i in range(ctrls):
        qstar += w[i] * reshaped_q[i]  # [2, grow, gcol]
    del w, reshaped_w

    # Get final image transfomer -- 3-D array
    transformers = torch.zeros((2, grow, gcol), dtype=torch.float32).to(device)
    for i in range(ctrls):
        transformers += A[i] * (reshaped_q[i] - qstar)
    transformers += qstar
    del A

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == float('Inf')  # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > grow - 1] = 0
    transformers[1][transformers[1] > gcol - 1] = 0

    return transformers.long()

class MLS(torch.nn.Module):
    def __init__(self):
        super(MLS, self).__init__()

    def forward(self, fea_g, L_d, L_g):

        batch_size,channel,height, width = fea_g.shape
        gridX = torch.arange(width, dtype=torch.int16).to(default_device)
        gridY = torch.arange(height, dtype=torch.int16).to(default_device)
        vy, vx = torch.meshgrid(gridX, gridY)
        # !!! Pay attention !!!: the shape of returned tensors are different between numpy.meshgrid and torch.meshgrid
        vy, vx = vy.transpose(0, 1), vx.transpose(0, 1)

        output_tensor = torch.ones_like(fea_g).to(default_device)
        for b in range(batch_size):
            affine = mls_affine_deformation(vy, vx, L_d[b].t(), L_g[b].t(), alpha=1)
            for c in range(channel):
                output_tensor[b][c][vx.long(), vy.long()] = fea_g[b][c][tuple(affine)]

        return output_tensor