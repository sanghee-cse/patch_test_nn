import torch

class ElasticMaterial:
    def __init__(self, E, nu, mode='plane_strain', dtype=torch.float32):
        self.E = torch.as_tensor(E, dtype=dtype)
        self.nu = torch.as_tensor(nu, dtype=dtype)
        self.mode = mode

    @property
    def mu(self):
        return self.E / (2 * (1 + self.nu))

    @property
    def lam(self):
        if self.mode == 'plane_strain':
            return (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        else: # plane_stress
            return (self.E * self.nu) / (1 - self.nu**2)

    def to(self, device, dtype=None):
        target_dtype = dtype if dtype is not None else self.L.dtype
        self.E = self.E.to(device=device, dtype=target_dtype)
        self.nu = self.nu.to(device=device, dtype=target_dtype)
        return self