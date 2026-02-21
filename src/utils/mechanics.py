import numpy as np
import torch

def stress_to_traction(sxx, syy, sxy, n):
    """
    Convert stress components to traction vector on a plane with normal vector n.
    Inputs:
        sxx: Normal stress component in x direction: (N, 1)
        syy: Normal stress component in y direction: (N, 1)
        sxy: Shear stress component: (N, 1)
        n: Normal vector of the plane: (N, 2)
    Outputs:
        t = Traction vector tau n: (N,2) = tau * n
    """

    n = n / torch.norm(n, dim=1, keepdim=True)

    tx = sxx.flatten() * n[:, 0] + sxy.flatten() * n[:, 1]  
    ty = sxy.flatten() * n[:, 0] + syy.flatten() * n[:, 1]

    return torch.cat((tx[:, None], ty[:, None]), axis=1)

def von_mises_stress(sxx, syy, sxy):
    return torch.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)

def dot(v1, v2):
    return torch.sum(v1 * v2, axis=1, keepdim=True)

def MSE(res, weight=1.0):
    return torch.mean(torch.sum((res**2) * weight, dim=1))

class PDELoss:
    def __init__(self, material, scales):
        self.mat = material
        self.scales = scales

    def residual(self, model, coords, b=None):
        x = coords[:, 0:1].requires_grad_(True)
        y = coords[:, 1:2].requires_grad_(True)
        
        ux, uy, sxx, syy, sxy = model(x, y)

        dux = torch.autograd.grad(ux, [x, y], grad_outputs=torch.ones_like(ux), create_graph=True)
        duy = torch.autograd.grad(uy, [x, y], grad_outputs=torch.ones_like(uy), create_graph=True)
        
        exx = dux[0]
        eyy = duy[1]
        exy = 0.5 * (dux[1] + duy[0])

        ### Constitutive Law
        res_consti_xx = (sxx - (self.mat.lam * (exx + eyy) + 2 * self.mat.mu * exx)) / self.scales.s0
        res_consti_yy = (syy - (self.mat.lam * (exx + eyy) + 2 * self.mat.mu * eyy)) / self.scales.s0
        res_consti_xy = (sxy - (2 * self.mat.mu * exy)) / self.scales.s0

        ### Equilibrium
        dsxx_dx = torch.autograd.grad(sxx, x, grad_outputs=torch.ones_like(sxx), create_graph=True)[0]
        grads   = torch.autograd.grad(sxy, [x, y], grad_outputs=torch.ones_like(sxy), create_graph=True)
        dsxy_dx = grads[0]
        dsxy_dy = grads[1]
        dsyy_dy = torch.autograd.grad(syy, y, grad_outputs=torch.ones_like(syy), create_graph=True)[0]

        res_eq_x = (dsxx_dx + dsxy_dy) * (self.scales.L / self.scales.s0)
        res_eq_y = (dsxy_dx + dsyy_dy) * (self.scales.L / self.scales.s0)

        if b is not None:
            res_eq_x += b[0]* (self.scales.L / self.scales.s0)
            res_eq_y += b[1]* (self.scales.L / self.scales.s0)

        return torch.cat([res_consti_xx, res_consti_yy, res_consti_xy, res_eq_x, res_eq_y], dim=1)
    