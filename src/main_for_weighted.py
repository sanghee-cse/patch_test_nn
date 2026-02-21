import matplotlib
matplotlib.use('Agg') 

import copy
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import time
from datetime import timedelta

from utils.configs import TrainConfig, EXPERIMENTAL_CASES
from utils.models import MLP, SkewMLP, SxxMLP, SyyMLP
from utils.plotter import Plotter
from utils.mechanics import PDELoss

torch.manual_seed(42)
np.random.seed(42)
dtype = torch.float64
torch.set_default_dtype(dtype)

def run_experiment_logic(case_name, case_data, ModelClass, train_cfg, loss_weights, save_dir):
    start_time = time.time()
    model_name = ModelClass.__name__
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"\n{'='*60}")
    print(f"Running Case: [{case_name}] | Model: [{model_name}]")
    print(f"Save Directory: {save_dir}")
    print(f"Loss Weights: {loss_weights}")
    print(f"{'='*60}")

    ProblemClass = case_data["class"]
    config = ProblemClass(train_cfg).setup()
    config.resample() 

    model = ModelClass(
        hidden_dim=64, num_hidden_layers=3, 
        L=config.physics.L, activation=train_cfg.activation
    ).to(train_cfg.device)
    
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    pde_loss_fn = PDELoss(config.material, config.scales)
    
    history = {'train_loss': [], 'pde_loss': [], 'bc_loss': []}

    # --- Adam Training ---
    pbar = tqdm(range(train_cfg.epochs), desc="Adam", leave=False)
    for epoch in pbar:
        optimizer.zero_grad()
        loss_pde = pde_loss_fn.get_loss(model, config.coords_pde, weight=loss_weights)
        loss_bc = config.bc_manager.get_loss(model, config.scales)
        total_loss = loss_pde + loss_bc
        total_loss.backward()
        optimizer.step()
        
        history['train_loss'].append(total_loss.item())
        history['pde_loss'].append(loss_pde.item())
        history['bc_loss'].append(loss_bc.item())

    # --- L-BFGS Fine-tuning ---
    lbfgs = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1000,
        history_size=50,
        tolerance_grad=1e-8,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )
    
    if case_name == 'fullhole': n_lbfgs_steps = 150
    elif case_name == 'quaterhole': n_lbfgs_steps = 50
    else: n_lbfgs_steps = 20

    pbar_lbfgs = tqdm(range(n_lbfgs_steps), desc="L-BFGS", leave=False)

    for k in pbar_lbfgs:

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            lp = pde_loss_fn.get_loss(model, config.coords_pde, weight=loss_weights)
            lb = config.bc_manager.get_loss(model, config.scales)
            total = lp + lb
            total.backward()
            return total

        try:
            lbfgs.step(closure)
        except Exception as e:
            print(f"[Warning] L-BFGS skipped due to error: {e}")

        lbfgs.zero_grad(set_to_none=True)
        lp_now = pde_loss_fn.get_loss(model, config.coords_pde, weight=loss_weights)
        lb_now = config.bc_manager.get_loss(model, config.scales)
        total_now = (lp_now + lb_now).item()

        history['train_loss'].append(total_now)
        history['pde_loss'].append(lp_now.item())
        history['bc_loss'].append(lb_now.item())

        if total_now < 1e-7:
            print(f"Target loss was reached. Training completed after {k} lbfgs_steps")

            break

    plotter = Plotter(config, model, save_dir=save_dir)
    plotter.plot_sampling_points(save=True)
    plotter.plot_results(custom_configs=config.plot_style, save=True)
    plotter.plot_deformed(scale_factor=20.0, save=True)
    plotter.plot_history(history, save=True)

    config.save(save_dir, case_name=case_name, model_name=model_name)
    torch.save(model.state_dict(), os.path.join(save_dir, f"weights.pth"))
    
    plt.close('all')
    elapsed_time = time.time() - start_time

    print(f"Done: {save_dir}")
    print(f"Time: {str(timedelta(seconds=int(elapsed_time)))}")

if __name__ == "__main__":
    train_cfg_base = TrainConfig(
        epochs=2000, 
        lr=1e-4, 
        n_pde=1000, 
        n_bc=100, 
        dtype=dtype, 
        activation='tanh'
    )
    
    base_result_dir = "../results_weighted"
    
    # Weight order: [res_consti_xx, res_consti_yy, res_consti_xy, res_eq_x, res_eq_y]
    experiments = [
        {
            "name": "const_weighted",
            "weights": torch.tensor([0.001, 0.001, 0.001, 1.0, 1.0], dtype=dtype)
        },
        {
            "name": "eq_x_weighted",
            "weights": torch.tensor([1.0, 1.0, 1.0, 0.001, 1.0], dtype=dtype)
        }
    ]
    
    for exp in experiments:
        print(f"\n>>> RUNNING EXPERIMENT: {exp['name']}")
        current_weights = exp['weights'].to(train_cfg_base.device)
        
        for case_name, case_data in EXPERIMENTAL_CASES.items():            
            train_cfg = copy.deepcopy(train_cfg_base)
            
            if case_name not in ["quaterhole", "fullhole"]:
                train_cfg.n_pde = 250
                print(f"[{case_name}] Setting n_pde to 250")
            else:
                print(f"[{case_name}] Keeping n_pde at {train_cfg.n_pde}")

            for ModelClass in [MLP, SkewMLP, SxxMLP, SyyMLP]:
                save_path = os.path.join(base_result_dir, exp['name'], case_name, ModelClass.__name__)
                try:
                    run_experiment_logic(case_name, case_data, ModelClass, train_cfg, current_weights, save_path)
                except Exception as e:
                    print(f"!!! CRITICAL ERROR in {exp['name']} - {case_name} !!!")
                    print(e)
                    continue 
    
    print("\nAll Experiments Completed.")