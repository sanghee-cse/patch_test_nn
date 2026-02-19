import torch
import copy
from dataclasses     import dataclass
from utils.materials import ElasticMaterial
from utils.bcs       import BCManager, TractionBC, RollerBC, PointRollerBC, PinBC, SpringBC, DisplacementBC
from utils.geometry  import Point2D, Rectangle2D, Circle2D, LineEdge2D, ArcEdge2D, SubtractedGeometry


@dataclass
class Scales:
    L: torch.Tensor
    s0: torch.Tensor
    u0: torch.Tensor

    def to(self, device, dtype=None):
        target_dtype = dtype if dtype is not None else self.L.dtype
        self.L = self.L.to(device=device, dtype=target_dtype)
        self.s0 = self.s0.to(device=device, dtype=target_dtype)
        self.u0 = self.u0.to(device=device, dtype=target_dtype)
        return self

@dataclass
class PhysicsConfig:
    E:      float = 55.0
    nu:     float = 0.3
    mode:   str = 'plane_strain'
    L:      float = 10.0
    l_bc:   float = 0.1
    q:      float = None
    ks_val: float = None
    delta:  float = None

@dataclass
class TrainConfig:
    n_pde:  int = 1000
    n_bc:   int = 100
    lr:     float = 1e-4
    epochs: int = 10000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype:  torch.dtype = torch.float32
    activation: str = 'tanh'
    sampling_method: str = 'random'

STRESS_NOISE = {
    'vmin': -0.01, 'vmax': 0.01, 
    'tick_step': 0.01, 
    'cmap': 'coolwarm', 
}

DISP_NOISE = {
    'vmin': -0.001, 'vmax': 0.001, 
    'tick_step': 0.001, 
    'cmap': 'coolwarm', 
}

DEFAULT_PLOT_STYLE = {
    'u_mag': {'cmap': 'magma', 'num_ticks': 5},
    'vm':    {'cmap': 'magma', 'num_ticks': 5},
    'ux':    {'cmap': 'coolwarm', 'num_ticks': 5},
    'uy':    {'cmap': 'coolwarm', 'num_ticks': 5},
    'sxx':   {'cmap': 'coolwarm', 'num_ticks': 5},
    'syy':   {'cmap': 'coolwarm', 'num_ticks': 5},
    'sxy':   {'cmap': 'coolwarm', 'num_ticks': 5},
}

class BaseConfig:
    def __init__(self, train_cfg):
        self.train   = train_cfg
        self.device = self.train.device
        self.dtype = self.train.dtype
        
        self.physics = None
        self.material = None
        self.scales = None

        self.geom = None
        self.bc = None
        self.edges = None

        self.plot_style = copy.deepcopy(DEFAULT_PLOT_STYLE)

    def _bake_physics(self):
        self.material = ElasticMaterial(
            E    = self.physics.E, 
            nu   = self.physics.nu, 
            mode = self.physics.mode,
            dtype = self.dtype
        ).to(self.device, dtype=self.dtype)
        
        if self.physics.q is not None:
            s0_val = self.physics.q
            u0_val = (s0_val * self.physics.L) / self.physics.E
        else: 
            u0_val = self.physics.delta
            s0_val = (u0_val*self.physics.E)/self.physics.L

        self.scales = Scales(
            L=torch.tensor(self.physics.L),
            s0=torch.tensor(s0_val),
            u0=torch.tensor(u0_val)
        ).to(self.device, dtype=self.dtype)



    def print_summary(self):
        print(f"--- PINN Configuration Summary ---")
        print(f"Mode: {self.physics.mode} on {self.train.device}")
        print(f"Scales -> s0: {self.scales.s0.item():.2e}, u0: {self.scales.u0.item():.2e}")
        print(f"Lame Params -> lambda: {self.material.lam.item():.2f}, mu: {self.material.mu.item():.2f}")

    def setup(self):
        raise NotImplementedError
    
    def resample(self):
        if self.bc_manager is not None:
            self.bc_manager.resample(self.device, self.dtype)
        if self.geom is not None:
            sampling_method = getattr(self.train, 'sampling', 'rotated_grid')
            self.coords_pde = self.geom.sample(
                self.train.n_pde, 
                sampling=sampling_method,
                device=self.device, 
                dtype=self.dtype
            )
        return self
    

class PatchXTractionProblem(BaseConfig):
    def setup(self):
        self.physics = PhysicsConfig(
            E = 55.0,
            nu = 0.3,
            L = 10.0,    
            q = 0.55,   
            mode = "plane_strain"
        )
        
        self._bake_physics()

        L = self.physics.L
        n_bc = self.train.n_bc
        
        self.geom = Rectangle2D(0, L, 0, L)
        self.edges = self.geom.get_boundaries()

        self.pin_point = Point2D([0.0, 0.0])
        self.roller_point = Point2D([0.0, L])

        bc_configs = [
            (PinBC(self.pin_point), 10, 1.0, {}),
            (PointRollerBC(self.roller_point, normal_vec=[1.0, 0.0]), 10, 1.0, {}),
            (TractionBC(self.edges["left"],   traction_=[-self.physics.q, 0.0]), n_bc, 1.0, {}),
            (TractionBC(self.edges["right"],  traction_=[self.physics.q, 0.0]),  n_bc, 1.0, {}),
            (TractionBC(self.edges["top"],    traction_=None),                   n_bc, 1.0, {}),
            (TractionBC(self.edges["bottom"], traction_=None),                   n_bc, 1.0, {})
        ]

        self.bc_manager = BCManager(bc_configs)
        self.resample()

        self.plot_style.update({
            'sxx': {'vmin': 0.54, 'vmax': 0.56, 'tick_step': 0.01, 'cmap': 'magma'},
            'vm': {'vmin': 0.54, 'vmax': 0.56, 'tick_step': 0.01, 'cmap': 'magma'},
            'syy': STRESS_NOISE,
            'sxy': STRESS_NOISE,
        })

        return self
        

class PatchShearProblem(BaseConfig):
    def setup(self):
        self.physics = PhysicsConfig(
            E=55.0, nu=0.3, L=10.0, q=0.55, mode="plane_strain"
        )
        self._bake_physics()

        L = self.physics.L
        n_bc = self.train.n_bc
        self.geom = Rectangle2D(0, L, 0, L)
        self.edges = self.geom.get_boundaries()

        self.pin_point = Point2D([0.0, 0.0]) 
        self.roller_point = Point2D([L, 0.0]) 

        bc_configs = [
            (PinBC(self.pin_point), 10, 10.0, {}),
            (PointRollerBC(self.roller_point, normal_vec=[0.0, 1.0]), 10, 1.0, {}),
            
            (TractionBC(self.edges["right"],  traction_=[0.0,  self.physics.q]), n_bc, 1.0, {}),
            (TractionBC(self.edges["left"],   traction_=[0.0, -self.physics.q]), n_bc, 1.0, {}),
            (TractionBC(self.edges["top"],    traction_=[self.physics.q,  0.0]), n_bc, 1.0, {}),
            (TractionBC(self.edges["bottom"], traction_=[-self.physics.q, 0.0]), n_bc, 1.0, {})
        ]

        self.bc_manager = BCManager(bc_configs)
        self.resample()

        self.plot_style.update({
            'sxx': STRESS_NOISE,
            'syy': STRESS_NOISE,
            'sxy': {'vmin': 0.54, 'vmax': 0.56, 'tick_step': 0.01, 'cmap': 'magma'},
            'vm': {'vmin': 0.9426, 'vmax': 0.9626, 'tick_step': 0.01, 'cmap': 'magma'},
            'ux': {'cmap': 'coolwarm'},
            'uy': DISP_NOISE,
        })
        return self

class PatchXTranslationProblem(BaseConfig):
    def setup(self):
        self.physics = PhysicsConfig(
            E = 55.0,
            nu = 0.3,
            L = 10.0,    
            q = 0.55,
            ks_val = 5.5,
            l_bc = 0.1,
            mode = "plane_strain",
        )

        self._bake_physics()

        L = self.physics.L
        self.geom = Rectangle2D(0, L, 0, L)
        self.edges = self.geom.get_boundaries()
        
        l_bc = self.physics.l_bc
        n_bc = self.train.n_bc
        self.left_spring_top    = LineEdge2D([0.0, L], [0.0, L - l_bc])
        self.left_free_mid      = LineEdge2D([0.0, L - l_bc], [0.0, l_bc])
        self.left_spring_bottom = LineEdge2D([0.0, l_bc], [0.0, 0.0])

        bc_configs = [
            (SpringBC(self.left_spring_bottom, k=self.physics.ks_val, 
                      spring_vec=[1.0, 0.0], q_val=self.physics.q), 
             n_bc, 1.0, {'sampling': 'power_law', 'focus': 'p2'}),
            
            (SpringBC(self.left_spring_top, k=self.physics.ks_val, 
                      spring_vec=[1.0, 0.0], q_val=self.physics.q), 
             n_bc, 1.0, {'sampling': 'power_law', 'focus': 'p1'}),
            
            (RollerBC(self.edges["bottom"]), n_bc, 1.0, {}),
            
            (TractionBC(self.edges["top"],    traction_=None), n_bc, 1.0, {}),
            (TractionBC(self.edges["right"],  traction_=None), n_bc, 1.0, {}),
            (TractionBC(self.left_free_mid,  traction_=None), n_bc, 1.0, {})
        ]

        self.bc_manager = BCManager(bc_configs)
        self.resample()            
        
        
        self.plot_style.update({
            'sxx': STRESS_NOISE,
            'syy': STRESS_NOISE,
            'sxy': STRESS_NOISE,
            'vm':  STRESS_NOISE,
            'ux': {'cmap': 'magma','vmin': 0.099, 'vmax': 0.101, 'tick_step': 0.001},
            'u_mag': {'cmap': 'magma', 'vmin': 0.099, 'vmax': 0.101, 'tick_step': 0.001},
            'uy': DISP_NOISE,
        })
            
        return self
    
    def resample(self):
        self.bc_manager.resample(self.device, self.dtype)
        self.coords_pde = self.geom.sample(self.train.n_pde,
                                           device=self.device, dtype=self.dtype)
        
        return self
    
class PatchRotationProblem(BaseConfig):
    def setup(self):
        self.physics = PhysicsConfig(
            E = 55.0,
            nu = 0.3,
            L = 10.0,    
            q = 0.55,
            ks_val = 5.5,
            l_bc = 0.1,
            mode = "plane_strain",
        )

        self._bake_physics()        
        
        L = self.physics.L
        l_bc = self.physics.l_bc
        n_bc = self.train.n_bc
        self.geom = Rectangle2D(0, L, 0, L)
        self.edges = self.geom.get_boundaries()
        
        self.bottom_free = LineEdge2D([0.0, 0.0], [L - l_bc, 0.0])
        self.bottom_spring = LineEdge2D([L - l_bc, 0.0], [L, 0.0])
        self.pin_point = Point2D([0.0, 0.0])

        self.bc_configs = [
            (PinBC(self.pin_point), 10, 10.0, {}),
            
            (SpringBC(self.bottom_spring, k=self.physics.ks_val, 
                      spring_vec=[0.0, 1.0], q_val=self.physics.q), 
             n_bc, 1.0, {'sampling': 'power_law', 'focus': 'p2'}),
            
            (TractionBC(self.edges["top"],    traction_=None), n_bc, 1.0, {}),
            (TractionBC(self.edges["left"],   traction_=None), n_bc, 1.0, {}),
            (TractionBC(self.edges["right"],  traction_=None), n_bc, 1.0, {}),
            (TractionBC(self.bottom_free, traction_=None), n_bc, 1.0, {})
        ]

        self.bc_manager = BCManager(self.bc_configs)
        self.resample()

        self.plot_style.update({
            'sxx': STRESS_NOISE,
            'syy': STRESS_NOISE,
            'sxy': STRESS_NOISE,
            'vm':  STRESS_NOISE, 
        })

        return self
    
class PlateWithAHoleDispProblem(BaseConfig):
    def setup(self):
        L_x, L_y = 10.0, 28.0
        r_hole = 5.0

        self.physics = PhysicsConfig(
            E = 55.0,
            nu = 0.3,
            L = max(L_x, L_y),    
            delta = 0.5,
            mode = "plane_stress",
        )

        self._bake_physics()

        rect = Rectangle2D(0, L_x, 0, L_y)
        hole = Circle2D([0.0, 0.0], r_hole)
        self.geom = SubtractedGeometry(rect, hole)
        self.edges = {
            "top":        LineEdge2D([L_x, L_y], [0.0, L_y]),          
            "right":      LineEdge2D([L_x, 0.0], [L_x, L_y]),        
            "left_sym":   LineEdge2D([0.0, L_y], [0.0, r_hole]),  
            "bottom_sym": LineEdge2D([r_hole, 0.0], [L_x, 0.0]),
            "hole":       ArcEdge2D([0, 0], r_hole, torch.pi/2, 0), 
        }

        n_bc = self.train.n_bc
        bc_configs = [
            (RollerBC(self.edges["left_sym"]),   n_bc, torch.tensor([10.0, 1.0]), {}),
            (RollerBC(self.edges["bottom_sym"]), n_bc, torch.tensor([10.0, 1.0]), {}),
            
            (DisplacementBC(self.edges["top"], u_val=[None, self.physics.delta]), 
             n_bc, torch.tensor([1.0, 10.0]), {}),
            
            (TractionBC(self.edges["right"], traction_=None), n_bc, 1.0, {}),
            (TractionBC(self.edges["hole"],  traction_=None), n_bc, 1.0, {})
        ]

        self.bc_manager = BCManager(bc_configs)
        self.resample()

        return self



class FullPlateWithAHoleTractionProblem(BaseConfig):
    def setup(self):
        L_x, L_y = 10.0, 28.0
        self.physics = PhysicsConfig(
            E=55.0, 
            nu=0.3, 
            L = 2*max(L_x, L_y),
            q=1.0, 
            mode="plane_stress"
        )
        self._bake_physics()

        L = self.physics.L
        R = 5.0  
        q = self.physics.q
        n_bc = self.train.n_bc

        rect = Rectangle2D(-L_x, L_x, -L_y, L_y)
        hole = Circle2D([0.0, 0.0], R)
        self.geom = SubtractedGeometry(rect, hole)

        rect_bounds = rect.get_boundaries()
        hole_bounds = hole.get_boundaries(theta_min=2*torch.pi, theta_max=0.0)
        self.edges = {**rect_bounds, **hole_bounds}

        self.pin_point = Point2D([0.0, -L_y]) 

        bc_configs = [
            (RollerBC(self.edges["bottom"]), n_bc, 1.0, {}),
            
            (PinBC(self.pin_point), 10, 10.0, {}),

            (TractionBC(self.edges["top"], traction_=[0.0, q]), n_bc, 1.0, {}),

            (TractionBC(self.edges["left"],   traction_=None), n_bc, 1.0, {}),
            (TractionBC(self.edges["right"],  traction_=None), n_bc, 1.0, {}),
            (TractionBC(self.edges["arc"],    traction_=None), n_bc, 1.0, {})
        ]

        self.bc_manager = BCManager(bc_configs)
        self.resample()

        return self

EXPERIMENTAL_CASES = {
    "Xtraction": {
        "class": PatchXTractionProblem,
        "save_dir": "/Patch_X_traction",
        "description": "Uniaxial traction test for verification"
    },

    "shear": {
        "class": PatchShearProblem,
        "save_dir": "/Patch_shear",
        "description": "Pure shear patch test"
    },
    "rotation": {
        "class": PatchRotationProblem,
        "save_dir": "/Patch_rotation",
        "description": "Rotation test for rigid body mode test"
    },
    "Xtranslation": {
        "class": PatchXTranslationProblem,
        "save_dir": "/Patch_X_translation",
        "description": "Translation test for rigid body mode test"
    },
    "quaterhole": {
        "class": PlateWithAHoleDispProblem, 
        "save_dir": "/plate_hole_qtr_disp",
        "description": "Quarter model with prescribed displacement (Symmetry check)"
    },

    "fullhole": {
        "class": FullPlateWithAHoleTractionProblem, 
        "save_dir": "/plate_hole_full_traction",
        "description": "Full model with remote traction (Benchmark solution)"
    }
}

""" "Ytraction": {
        "class": PatchYTractionProblem,
        "save_dir": "/Patch_Y_traction",
        "description": "Uniaxial traction test for verification"
    },"""
""" "Ytranslation": {
    "class": PatchYTranslationProblem,
    "save_dir": "/Patch_Y_translation",
    "description": "Translation test for rigid body mode test"
},"""
"""  """