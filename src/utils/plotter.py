import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib import ticker
from utils.mechanics import von_mises_stress

class Plotter:
    def __init__(self, config, model, save_dir="results"):
        self.config = config
        self.model = model
        self.device = config.device
        self.dtype  = config.dtype
        self.scales = config.scales
        self.save_dir = save_dir
        
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _draw_boundaries(self, ax, **kwargs):
        if hasattr(self.config, 'edges') and self.config.edges is not None:
            edges = self.config.edges
            edge_list = edges.values() if isinstance(edges, dict) else edges
            
            for edge in edge_list:
                if hasattr(edge, 'plot'):
                    edge.plot(ax, **kwargs)
                    
        elif hasattr(self.config.geom, 'get_boundaries'):
            boundaries = self.config.geom.get_boundaries()
            edges = boundaries.values() if isinstance(boundaries, dict) else boundaries
            for edge in edges:
                if hasattr(edge, 'plot'):
                    edge.plot(ax, **kwargs)

    def plot_sampling_points(self, save=False, filename="sampling_points.png"):
        fig, ax = plt.subplots(figsize=(5,5))
        
        self._draw_boundaries(ax, zorder=1, marker=None)
        pde = self.config.coords_pde.detach().cpu().numpy()
        ax.scatter(pde[:, 0], pde[:, 1], s=5, label='PDE Interior', alpha=0.7, color='gray')
        
        colors_iter = plt.get_cmap('tab10')
        plotted_labels = {}
        color_idx = 0
        
        for i, entry in enumerate(self.config.bc_manager.data_map):
            bc_obj = entry['bc']
            coords = entry['coords']
            label = getattr(bc_obj, 'label', bc_obj.__class__.__name__)
            pts = coords.detach().cpu().numpy().reshape(-1, 2)

            if label not in plotted_labels:
                plotted_labels[label] = colors_iter(color_idx % 10)
                color_idx += 1
                legend_label = label
            else:
                legend_label = "_nolegend_" 
                
            color = plotted_labels[label]
            
            if bc_obj.__class__.__name__ in ["PinBC", "PointRollerBC", "SpringBC"]:
                s = 40  
            else:
                s = 10 
            
            ax.scatter(pts[:, 0], pts[:, 1], s=s, label=legend_label, 
                       color=color, zorder=10 if s > 10 else 5)
            
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
        ax.set_title("Sampling Points", fontsize=12)
        ax.axis('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        if save: self._save_fig(fig, filename)
        plt.show()

    def _pcolor_ax(self, ax, x, y, z, title, cbar_kwargs=None):
            cbar_kwargs = cbar_kwargs or {}
            
            cmap = cbar_kwargs.pop('cmap', 'jet')
            
            vmin = cbar_kwargs.pop('vmin', z.min())
            vmax = cbar_kwargs.pop('vmax', z.max())
            
            im = ax.scatter(x, y, c=z, cmap=cmap, s=12, vmin=vmin, vmax=vmax, edgecolors='none', clip_on=True,)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis("equal")
            ax.axis("off")

            cbar_defaults = {'fraction': 0.046, 'pad': 0.04}
            label     = cbar_kwargs.pop('label', None)
            num_ticks = cbar_kwargs.pop('num_ticks', 5)
            sci_mode  = cbar_kwargs.pop('sci_mode', False)
            tick_step = cbar_kwargs.pop('tick_step', None)
            
            from matplotlib import ticker
            cbar = plt.colorbar(im, ax=ax, **{**cbar_defaults, **cbar_kwargs})
            if tick_step is not None:    
                cbar.locator = ticker.MultipleLocator(tick_step)
            else:
                cbar.locator = ticker.MaxNLocator(nbins=num_ticks)
            
            
            if label: cbar.set_label(label)
            if sci_mode:
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

            return im

    def plot_results(self, n_points=10000, save=False, prefix="result", custom_configs=None):
        """
        custom_configs example:
        {
            'u_mag': {'label': 'mm', 'sci_mode': True, 'num_ticks': 3},
            'vm':    {'label': 'MPa', 'vmax': 100, 'num_ticks': 5}
        }
        """

        d = self._get_predictions(n_points)
        
        plot_groups = [
            ([('u_mag', "Displacement Magnitude"), ('vm', "Von Mises Stress")], "mag_vm"),
            ([('ux', r"Displacement $u_{x}$"), ('uy', r"Displacement $u_{y}$")], "ux_uy"),
            ([('sxx', r"Stress $\tau_{xx}$"), ('syy', r"Stress $\tau_{yy}$"), ('sxy', r"Stress $\tau_{xy}$")], "stresses")
        ]

        if custom_configs is None: custom_configs = {}
        
        for plots, suffix in plot_groups:
            n_cols = len(plots)
            fig, axs = plt.subplots(1, n_cols, figsize=(6*n_cols, 5))
            if n_cols == 1: axs = [axs]
            
            for i, (key, title) in enumerate(plots):
                self._draw_boundaries(axs[i], zorder=10, lw=2)
                config = custom_configs.get(key, {}).copy() 
                self._pcolor_ax(axs[i], d['x'], d['y'], d[key], title, cbar_kwargs=config)
                axs[i].set_title(title)
                axs[i].axis("equal")
                axs[i].axis("off")
                axs[i].set_xlabel('x')
                axs[i].set_xlabel('y')
            
            if save: self._save_fig(fig, f"{prefix}_{suffix}.png")
            plt.show()

    def plot_deformed(self, scale_factor=20.0, n_points=5000, save=False, filename="deformed.png"):
        d = self._get_predictions(n_points)
        fig, ax = plt.subplots(figsize=(7, 6))

        x_def = d['x'] + scale_factor * d['ux']
        y_def = d['y'] + scale_factor * d['uy']
        
        sc = ax.scatter(x_def, y_def, c=d['u_mag'], cmap='jet', s=10, edgecolors='none')
        self._draw_boundaries(ax, color='k', linestyle='--', alpha=0.7, lw=2, zorder=10)
        

        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Displacement Magnitude', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        
        ax.set_title(f"Deformed Shape (Scale: {scale_factor}x)", fontsize=14, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        if save: self._save_fig(fig, filename)
        plt.show()
        
    def _get_predictions(self, n_points):
        coords = self.config.geom.sample(n_points, sampling='grid', device=self.device, dtype=self.dtype)
        x, y = coords[:, 0:1], coords[:, 1:2]
        self.model.eval()
        with torch.no_grad():
            ux, uy, sxx, syy, sxy = self.model(x, y)
        
        data = {
            'x': x.cpu().numpy().flatten(), 'y': y.cpu().numpy().flatten(),
            'ux': (ux).cpu().numpy().flatten(), 'uy': (uy).cpu().numpy().flatten(),
            'sxx': (sxx).cpu().numpy().flatten(), 'syy': (syy).cpu().numpy().flatten(), 'sxy': (sxy).cpu().numpy().flatten()
        }
        data['u_mag'] = np.sqrt(data['ux']**2 + data['uy']**2)
        data['vm'] = von_mises_stress(torch.tensor(data['sxx']), torch.tensor(data['syy']), torch.tensor(data['sxy'])).numpy()
        return data

    def _save_fig(self, fig, filename):
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f"Saved: {path}")

    def plot_history(self, history, save=False, filename="loss_history.png"):
        """
        history: {'train_loss': [], 'pde_loss': [], 'bc_loss': []} 
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        epochs = range(len(history['train_loss']))
        
        ax.plot(epochs, history['train_loss'], label='Total Loss', color='black')
        ax.plot(epochs, history['pde_loss'], label='PDE Loss', linestyle='--', alpha=0.7)
        ax.plot(epochs, history['bc_loss'], label='BC Loss', linestyle='--', alpha=0.7)
        
        ax.set_yscale('log')
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        if save: self._save_fig(fig, filename)
        plt.show()