import matplotlib
matplotlib.use('Agg') 

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import json
import numpy as np

from utils.configs import TrainConfig, EXPERIMENTAL_CASES
from utils.models import MLP, SkewMLP, SxxMLP, SyyMLP
from utils.plotter import Plotter
from utils.mechanics import PDELoss
from dataclasses import asdict

import time
from datetime import timedelta

torch.manual_seed(42)
np.random.seed(42)
dtype = torch.float64
torch.set_default_dtype(dtype)


def run_experiment(case_name, case_data, ModelClass, train_cfg, base_result_dir="../results"):
    
    start_time = time.time()
    model_name = ModelClass.__name__
    save_dir = os.path.join(base_result_dir, case_name, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"\n{'='*60}")
    print(f"Running Case: [{case_name}] | Model: [{model_name}]")
    print(f"Save Directory: {save_dir}")
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

    pbar = tqdm(range(train_cfg.epochs), desc="Adam", leave=False)
    for epoch in pbar:
        optimizer.zero_grad()
        loss_pde = pde_loss_fn.get_loss(model, config.coords_pde)
        loss_bc = config.bc_manager.get_loss(model, config.scales)
        total_loss = loss_pde + loss_bc
        total_loss.backward()
        optimizer.step()
        
        history['train_loss'].append(total_loss.item())
        history['pde_loss'].append(loss_pde.item())
        history['bc_loss'].append(loss_bc.item())


    print("Fine-tuning with L-BFGS...")
    lbfgs = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1000,
        history_size=50,
        tolerance_grad=1e-8,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )
    if case_name == 'fullhole': n_lbfgs_steps = 70
    elif case_name == 'quaterhole': n_lbfgs_steps = 50
    else: n_lbfgs_steps = 10

    pbar_lbfgs = tqdm(range(n_lbfgs_steps), desc="L-BFGS", leave=False)

    for k in pbar_lbfgs:

        def closure():
            lbfgs.zero_grad(set_to_none=True)

            loss_pde = pde_loss_fn.get_loss(model, config.coords_pde)
            loss_bc  = config.bc_manager.get_loss(model, config.scales)
            total_loss = loss_pde + loss_bc

            total_loss.backward()
            return total_loss

        try:
           lbfgs.step(closure)
        except Exception as e:
            print(f"[Warning] L-BFGS skipped due to error: {e}")

        lbfgs.zero_grad(set_to_none=True)
        loss_pde = pde_loss_fn.get_loss(model, config.coords_pde)
        loss_bc  = config.bc_manager.get_loss(model, config.scales)
        total = (loss_pde + loss_bc).item()

        history['train_loss'].append(total)
        history['pde_loss'].append(loss_pde.item())
        history['bc_loss'].append(loss_bc.item())

        if total < 1e-7:
            print(f"Target loss was reached. Training completed after {k} lbfgs_steps")
            break
            

    print("Saving results...")
    
    plotter = Plotter(config, model, save_dir=save_dir)
    plotter.plot_sampling_points(save=True)
    plotter.plot_results(custom_configs=config.plot_style, save=True)
    plotter.plot_deformed(scale_factor=20.0, save=True)
    plotter.plot_history(history, save=True)
    
    config_to_save = {
        "train_info": asdict(train_cfg),
        "case_name": case_name,
        "model_name": model_name,
        "dtype": str(train_cfg.dtype) 
    }

    config_path = os.path.join(save_dir, "exp_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=4)


    model_path = os.path.join(save_dir, f"{model_name}_weights.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to: {model_path}")
    
    plt.close('all')

    elapsed_time = time.time() - start_time

    print(f"Done: {save_dir}")
    print(f"Total Time: {str(timedelta(seconds=int(elapsed_time)))}")


if __name__ == "__main__":
    start_time = time.time()
    train_cfg = TrainConfig(
        epochs=2000, 
        lr=1e-4, 
        n_pde=1000, 
        n_bc=100, 
        dtype=dtype, 
        activation='tanh'
    )
    
    target_models = [MLP, SkewMLP, SxxMLP, SyyMLP]
    
    print(f"Total Cases Found: {len(EXPERIMENTAL_CASES)}")
    
    for case_name, case_data in EXPERIMENTAL_CASES.items():            
            for ModelClass in target_models:
                try:
                    run_experiment(case_name, case_data, ModelClass, train_cfg)
                except Exception as e:
                    print(f"!!! CRITICAL ERROR in {case_name} - {ModelClass.__name__} !!!")
                    print(e)
                    continue 
    
    print("\nAll Experiments Completed.\n")
    elapsed_time = time.time() - start_time
    print(f"Total Time: {str(timedelta(seconds=int(elapsed_time)))}")