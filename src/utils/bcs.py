import torch
from utils.mechanics import stress_to_traction, dot

def MSE(res, weight=1.0):
    return torch.mean(torch.sum((res**2) * weight, dim=1))

class BCManager:
    def __init__(self, bc_configs):
        """
        bc_configs: [(BC_instance, num_sample, weight, sample_kwargs), ...]
        """
        self.configs = bc_configs
        self.data_map = []

    def resample(self, device, dtype):
        self.data_map = []
        for bc, n, weight, s_kwargs in self.configs:
            coords = bc.boundary.sample(n, device=device, dtype=dtype, **s_kwargs)
            self.data_map.append({
                'bc': bc, 'coords': coords, 'weight': weight })

    def get_loss(self, model, scales):
        loss = 0.0
        for entry in self.data_map:
            res = entry['bc'].residual(model, entry['coords'], scales)
            loss += MSE(res, weight = entry['weight'])
        return loss


class TractionBC:
    """
    Enforce tau*n = traction(coords)
     or 0 if traction is None
    """
    def __init__(self, boundary, traction_=None, label = None):
        self.boundary = boundary
        self.traction_ = traction_  # Function that takes coords and returns target traction vector
        if label is not None:
            self.label = label
        else:
            if self._is_free():
                self.label = "Free Traction"
            else:
                self.label = "Traction BC"

    def _is_free(self):
        if self.traction_ is None:
            return True
        
        try:
            t_tensor = torch.as_tensor(self.traction_)
            if torch.all(t_tensor == 0):
                return True
        except:
            pass
            
        return False

    def _eval_target(self, coords):
        if self.traction_ is None:
            return None
        if callable(self.traction_):
            t = self.traction_(coords)
        else:
            t = torch.as_tensor(self.traction_, device=coords.device, dtype=coords.dtype).view(1,2)
            t = t.repeat(coords.shape[0], 1)
        return t
    
    def residual(self, model, coords, scales):
        x, y = coords[:, 0:1], coords[:, 1:2]
        _, _, sxx, syy, sxy = model(x, y)
        n = self.boundary.normal(coords)
        t_model = stress_to_traction(sxx, syy, sxy, n)
        
        if self.traction_ is None:
            res = t_model/scales.s0
        else:
            t_target = self._eval_target(coords)
            res = (t_model - t_target)/scales.s0   

        return res

class RollerBC:
    """
    Frictionless roller:
        u*n = 0
        (tau*n)*tangent = 0
    """
    def __init__(self, boundary, label="Roller BC"):
        self.boundary = boundary
        self.label = label
    
    def residual(self, model, coords, scales):
        x, y = coords[:, 0:1], coords[:, 1:2]
        ux, uy, sxx, syy, sxy = model(x, y)

        n = self.boundary.normal(coords)
        t = self.boundary.tangent(coords)
        
        u = torch.cat((ux, uy), axis=1)
        traction = stress_to_traction(sxx, syy, sxy, n)

        res_u = dot(u, n)/scales.u0
        res_t = dot(traction, t)/scales.s0

        return torch.cat((res_u, res_t), axis=1)
    
class PointRollerBC:
    """
    Roller support at a point:
        u*n = 0
    """
    def __init__(self, boundary, normal_vec, label="Roller BC"):
        self.boundary = boundary
        self.n = torch.tensor(normal_vec)
        self.label = label

    def residual(self, model, coords, scales):
        x, y = coords[:, 0:1], coords[:, 1:2]
        ux, uy, _, _, _ = model(x, y)
        
        u = torch.cat((ux, uy), axis=1)
        n = self.n.to(device=coords.device, dtype=coords.dtype)
        
        res_u = dot(u, n) / scales.u0
        
        return res_u

class PinBC:
    """
    Pin support:
        u = 0
    """
    def __init__(self, boundary, label="Pin BC"):
        self.boundary = boundary
        self.label = label
    
    def residual(self, model, coords, scales):
        x, y = coords[:, 0:1], coords[:, 1:2]
        ux, uy, _, _, _ = model(x, y)

        u = torch.cat((ux, uy), axis=1)
        res = u / scales.u0

        return res

class SpringBC:
    """
    Spring support:
        (tau*n)·dir  - k*(u·dir) + q_val = 0
        q_val: scalar constant
    
    """
    def __init__(self, boundary, k, spring_vec, q_val=0.0, label="Spring BC"):
        self.boundary = boundary
        self.k = k
        self.spring_vec = torch.tensor(spring_vec, dtype=torch.float32).view(1,2)
        self.q = q_val
        self.label = label

    def residual(self, model, coords, scales):
        x, y = coords[:, 0:1], coords[:, 1:2]
        ux, uy, sxx, syy, sxy = model(x, y)

        u = torch.cat([ux, uy], dim=1)
        n = self.boundary.normal(coords)
        traction = stress_to_traction(sxx, syy, sxy, n)

        d = self.spring_vec.to(device=coords.device, dtype=coords.dtype)
        
        t_d = dot(traction, d)
        u_d = dot(u, d)

        # t_d - k * u_d + q_val = 0
        res = (t_d - self.k * u_d + self.q) / scales.s0
        return res
    
class DisplacementBC:
    def __init__(self, boundary, u_val, label="Displacement BC"):
        """
        u_val: [target_ux, target_uy] (or none if unconstrained)
        """
        self.boundary = boundary
        self.u_val = u_val
        self.label = label

    def residual(self, model, coords, scales):
        x, y = coords[:, 0:1], coords[:, 1:2]
        ux, uy, sxx, syy, sxy = model(x, y)
        
        n = self.boundary.normal(coords)
        traction = stress_to_traction(sxx, syy, sxy, n)
        
        res = []
        u_preds = [ux, uy]
        
        for i, val in enumerate(self.u_val):
            if val is not None:
                res.append((u_preds[i] - val) / scales.u0)
            else:
                res.append(traction[:, i:i+1] / scales.s0)        
        
        return torch.cat(res, axis=1)    