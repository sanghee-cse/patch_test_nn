import torch
from abc import ABC, abstractmethod

class InteriorGeometry2D(ABC):
    def __init__(self, eps=1e-6):
        self.eps = eps

    @abstractmethod
    def sample(self, num_points, device='cpu', dtype=torch.float32):
        pass

    @abstractmethod
    def contains(self, coords):
        pass

    @abstractmethod
    def on_boundary(self, coords, tol=None):
        pass

class Boundary2D(ABC):
    @abstractmethod
    def sample(self, num_points, sampling='uniform', device='cpu', dtype=torch.float32, focus='p1'):
        """
        Sample points on the boundary.
        Inputs:
            num_points: Number of points to sample
            sampling: Sampling strategy ('uniform', 'random', 'power_law')
            device: Torch device
            dtype: Torch data type
            focus: For 'power_law' sampling, focus point ('p1', 'p2', 'both')
        Outputs:
            samples: Sampled points on the boundary (num_points, 2)
        """
        pass
    @abstractmethod
    def normal(self, coords):
        """
        Compute normal vectors at given coordinates on the boundary.
        Inputs:
            coords: Coordinates on the boundary (N, 2)
        Outputs:
            normals: Normal vectors at the coordinates (N, 2)
        """
        pass
    
    def tangent(self, coords):
        """
        Compute tangent vectors at given coordinates on the boundary.
        The tangent is derived by rotating the normal vector (nx, ny) 90 degrees 
        counterclockwise to yield (-ny, nx),
        """
        n = self.normal(coords)
        tx = -n[:, 1:2]
        ty =  n[:, 0:1]
        return torch.cat([tx, ty], dim=1)

class Point2D:
    def __init__(self, coord):
        """coord: [x, y]"""
        self.coord = torch.tensor(coord, dtype=torch.float32)

    def sample(self, num_points, device='cpu', dtype=torch.float32):
        p = self.coord.to(device=device, dtype=dtype).view(1, 2)
        return p.expand(num_points, 2).contiguous()

#### Implementations ####
class Rectangle2D(InteriorGeometry2D):
    def __init__(self, x_min, x_max, y_min, y_max, eps=1e-3):
        super().__init__(eps)
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

    def sample(self, num_points, sampling='grid', device='cpu', dtype=torch.float32):
        x_start, x_end = self.x_min + self.eps, self.x_max - self.eps
        y_start, y_end = self.y_min + self.eps, self.y_max - self.eps
        if sampling == 'random':
            r = torch.rand((num_points, 2), device=device, dtype=dtype)
            x = x_start + (x_end - x_start) * r[:, 0:1]
            y = y_start + (y_end - y_start) * r[:, 1:2]
            return torch.cat((x, y), axis=1)
        elif sampling == 'grid':
            n_side = int(torch.tensor(num_points, dtype=dtype)**0.5)
            x = torch.linspace(x_start, x_end, n_side, device=device, dtype=dtype)
            y = torch.linspace(y_start, y_end, n_side, device=device, dtype=dtype)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            return torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)

        elif sampling == 'rotated_grid':
            n_side = int(torch.tensor(num_points, dtype=dtype)**0.5)
            cx, cy = (x_start + x_end) / 2, (y_start + y_end) / 2
            u = torch.linspace(-1, 1, 2*n_side, device=device, dtype=dtype)
            v = torch.linspace(-1, 1, 2*n_side, device=device, dtype=dtype)
            U, V = torch.meshgrid(u, v, indexing='ij')
            theta = torch.tensor(torch.pi / 4) 
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            X_rot = cx + (x_end - x_start) * (U * cos_t - V * sin_t)
            Y_rot = cy + (y_end - y_start) * (U * sin_t + V * cos_t)
            
            X_rot = torch.clamp(X_rot, x_start, x_end)
            Y_rot = torch.clamp(Y_rot, y_start, y_end)
            
            return torch.cat((X_rot.reshape(-1, 1), Y_rot.reshape(-1, 1)), axis=1)
        
    def contains(self, coords):
        x, y = coords[:, 0], coords[:, 1]
        return (x > self.x_min) & (x < self.x_max) & (y > self.y_min) & (y < self.y_max)
    
    def on_boundary(self, coords, tol=1.0e-6):
        x, y = coords[:, 0], coords[:, 1]
        on_left = torch.abs(x - self.x_min) < tol
        on_right = torch.abs(x - self.x_max) < tol
        on_bottom = torch.abs(y - self.y_min) < tol
        on_top = torch.abs(y - self.y_max) < tol
        return on_left | on_right | on_bottom | on_top
    
    def get_boundaries(self):
        return {
        "bottom": LineEdge2D([self.x_min, self.y_min], [self.x_max, self.y_min]),
        "right":  LineEdge2D([self.x_max, self.y_min], [self.x_max, self.y_max]),
        "top":    LineEdge2D([self.x_max, self.y_max], [self.x_min, self.y_max]),
        "left":   LineEdge2D([self.x_min, self.y_max], [self.x_min, self.y_min])
    }
    
class Circle2D(InteriorGeometry2D):
    def __init__(self, center, radius, eps=1e-3):
        super().__init__(eps)
        self.center = torch.tensor(center, dtype=torch.float32)
        self.radius = radius

    def sample(self, num_points, sampling='random', device='cpu', dtype=torch.float32):
        r = torch.rand((num_points, 1), device=device, dtype=dtype)
        c = self.center.to(device=device, dtype=dtype)
        theta = 2 * torch.pi * torch.rand((num_points, 1), device=device, dtype=dtype)
        x = c[0] + (self.radius - self.eps) * torch.sqrt(r) * torch.cos(theta)
        y = c[1] + (self.radius - self.eps) * torch.sqrt(r) * torch.sin(theta)
        return torch.cat((x, y), axis=1)

    def contains(self, coords):
        c = self.center.to(device=coords.device, dtype=coords.dtype)
        x, y = coords[:, 0], coords[:, 1]
        dist_sq = (x - c[0])**2 + (y - c[1])**2
        return dist_sq < self.radius**2
    
    def on_boundary(self, coords, tol=1.0e-6):
        x, y = coords[:, 0], coords[:, 1]
        c = self.center.to(device=coords.device, dtype=coords.dtype)
        dist = torch.sqrt((x - c[0])**2 + (y - c[1])**2)
        return torch.abs(dist - self.radius) < tol
    
    def get_boundaries(self, theta_min=0.0, theta_max=2*torch.pi):
        return {"arc": ArcEdge2D(self.center.tolist(), self.radius, theta_min, theta_max)}

class SubtractedGeometry(InteriorGeometry2D):
    def __init__(self, main_geom, sub_geom, eps=1e-6):
        self.main = main_geom
        self.sub = sub_geom
        self.eps = eps

    def sample(self, num_points, sampling='random', device='cpu', dtype=torch.float32):
        if sampling == 'random':
            points = []
            count = 0
            n_to_sample = int(num_points * 1.2)
            
            while count < num_points:
                candidate = self.main.sample(n_to_sample, sampling=sampling, device=device, dtype=dtype)
                is_inside = ~self.sub.contains(candidate)
                filtered = candidate[is_inside]
                
                points.append(filtered)
                count += len(filtered)
                
                n_to_sample = max(int((num_points - count) * 1.5), 10)

            return torch.cat(points, dim=0)[:num_points]        

        else:
            candidate = self.main.sample(num_points, sampling=sampling, device=device, dtype=dtype)
            is_inside = ~self.sub.contains(candidate)
            return candidate[is_inside]



    def contains(self, coords):
        in_main = self.main.contains(coords)
        in_sub = self.sub.contains(coords) | self.sub.on_boundary(coords)
        return in_main & (~in_sub)
    
    def on_boundary(self, coords, tol=1.0e-6):
        on_main = self.main.on_boundary(coords, tol)
        in_main = self.main.contains(coords)
        on_sub = self.sub.on_boundary(coords, tol)
        in_sub = self.sub.contains(coords)
        return (on_main & ~in_sub) | (on_sub & in_main)
    

class LineEdge2D(Boundary2D):
    def __init__(self, p1, p2, eps=1e-12):
            # The order of p1, p2 matters
            self.p1 = torch.tensor(p1)
            self.p2 = torch.tensor(p2)
            self.eps = eps
            
            v = self.p2 - self.p1
            self._t_const = v / (torch.norm(v) + self.eps)
            self._n_const = torch.tensor([self._t_const[1], -self._t_const[0]])
    
    def sample(self, num_points, sampling='uniform', device='cpu', dtype=torch.float32, focus='p1', power=2.0):
        if sampling == 'uniform':
            t = torch.linspace(0, 1, num_points, device=device, dtype=dtype).view(-1, 1)
        
        elif sampling == 'random':
            t = torch.rand((num_points, 1), device=device, dtype=dtype)
            
        elif sampling == 'power_law':
            r = torch.rand((num_points, 1), device=device, dtype=dtype)
            if focus == 'p1':
                t = torch.pow(r, power)
            elif focus == 'p2':
                t = 1 - torch.pow(r, power)
            elif focus == 'both':
                t = torch.where(r < 0.5, 
                                0.5 * torch.pow(2*r, power), 
                                1 - 0.5 * torch.pow(2*(1-r), power))
                
        P1, P2 = self.p1.to(device, dtype), self.p2.to(device, dtype)
        samples = (1 - t) * P1 + t * P2
        return samples
    
    def normal(self, coords):
        return self._n_const.to(device=coords.device, dtype=coords.dtype).expand(coords.shape[0], 2)

    def tangent(self, coords):
        return self._t_const.to(coords.device).expand(coords.shape[0], 2)
    
    def plot(self, ax, **kwargs):
        style = {'color':'gray', 'lw':2, 'zorder':10, 'label': '_nolegend_'}
        style.update(kwargs)
        return ax.plot([self.p1[0], self.p2[0]], 
                       [self.p1[1], self.p2[1]], **style)



class ArcEdge2D(Boundary2D):
    def __init__(self, center, radius, theta_start=0.0, theta_end=2*torch.pi, eps=1e-12):
        self.center = torch.tensor(center, dtype=torch.float32)
        self.radius = radius
        self.theta_start = theta_start
        self.theta_end = theta_end
        self.eps = eps

    def sample(self, num_points, sampling='uniform', device='cpu', dtype=torch.float32, focus=None):
        if sampling == 'uniform':
            theta = torch.linspace(self.theta_start, self.theta_end, num_points, device=device, dtype=dtype).view(-1, 1)
        else: # random
            r = torch.rand((num_points, 1), device=device, dtype=dtype)
            theta = self.theta_start + (self.theta_end - self.theta_start) * r

        c = self.center.to(device, dtype)
        x = c[0] + self.radius * torch.cos(theta)
        y = c[1] + self.radius * torch.sin(theta)
        return torch.cat((x, y), dim=1)

    def normal(self, coords):
        c = self.center.to(device=coords.device, dtype=coords.dtype)
        n = (coords - c)
        if self.theta_end <= self.theta_start:
            n = -n
        return n / (torch.norm(n, dim=1, keepdim=True) + self.eps)

    def plot(self, ax, **kwargs):
            style = {'color': 'gray', 'lw': 2, 'zorder': 10, 'label': '_nolegend_'}
            style.update(kwargs)
            theta = torch.linspace(self.theta_start, self.theta_end, 100)
            cx, cy = self.center
            r = self.radius
            return ax.plot(cx + r*torch.cos(theta), cy + r*torch.sin(theta), **style)
