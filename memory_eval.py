#!/usr/bin/env python3
"""
Memory Evaluation Script

This script evaluates trained world models on memory-specific tasks by comparing
world model rollouts to environment rollouts on memory tasks like object recall,
spatial memory, and navigation.

Usage:
    python memory_eval.py --config-name memory_eval --ckpt_base_path /path/to/checkpoints --model_name model_name
"""

import os
import hydra
import torch
import wandb
import logging
import warnings
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, open_dict

from memory_plan import memory_planning_main, build_memory_plan_cfg_dicts
from utils import cfg_to_dict, seed

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def run_memory_evaluation_suite(
    ckpt_base_path: str,
    model_name: str,
    model_epoch: str = "final",
    memory_test_modes: list = None,
    n_memory_objects_list: list = None,
    goal_H_list: list = None,
    n_evals: int = 10,
    seed: int = 42,
    wandb_logging: bool = True
):
    """
    Run a comprehensive memory evaluation suite across different configurations.
    
    Args:
        ckpt_base_path: Base path to model checkpoints
        model_name: Name of the trained model
        model_epoch: Epoch to load (or "final" for latest)
        memory_test_modes: List of memory test modes to evaluate
        n_memory_objects_list: List of numbers of memory objects to test
        goal_H_list: List of planning horizons to test
        n_evals: Number of evaluation episodes per configuration
        seed: Random seed
        wandb_logging: Whether to log to wandb
    """
    if memory_test_modes is None:
        memory_test_modes = ["object_recall", "color_memory", "sequential_memory", "navigation"]
    if n_memory_objects_list is None:
        n_memory_objects_list = [3, 4, 5]
    if goal_H_list is None:
        goal_H_list = [5, 10, 15]
    
    # Build configuration dictionaries for all combinations
    plan_cfg_path = "conf/memory_plan.yaml"
    cfg_dicts = build_memory_plan_cfg_dicts(
        plan_cfg_path=plan_cfg_path,
        ckpt_base_path=ckpt_base_path,
        model_name=model_name,
        model_epoch=model_epoch,
        memory_test_mode=memory_test_modes,
        n_memory_objects=n_memory_objects_list,
        memory_object_types=[["ball", "box", "key"]],
        goal_H=goal_H_list,
        alpha=[0.1, 1.0]
    )
    
    print(f"Running memory evaluation suite with {len(cfg_dicts)} configurations")
    
    results = {}
    
    for i, cfg_dict in enumerate(cfg_dicts):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(cfg_dicts)}")
        print(f"Memory Test Mode: {cfg_dict['memory_test_mode']}")
        print(f"Memory Objects: {cfg_dict['n_memory_objects']}")
        print(f"Goal Horizon: {cfg_dict['goal_H']}")
        print(f"Alpha: {cfg_dict['objective']['alpha']}")
        print(f"{'='*60}")
        
        # Set additional configuration
        cfg_dict.update({
            "n_evals": n_evals,
            "seed": seed,
            "wandb_logging": wandb_logging,
            "saved_folder": f"memory_outputs/{model_name}_{cfg_dict['memory_test_mode']}_{cfg_dict['n_memory_objects']}obj_H{cfg_dict['goal_H']}_alpha{cfg_dict['objective']['alpha']}"
        })
        
        try:
            # Run memory evaluation
            logs = memory_planning_main(cfg_dict)
            
            # Store results
            config_key = f"{cfg_dict['memory_test_mode']}_{cfg_dict['n_memory_objects']}obj_H{cfg_dict['goal_H']}_alpha{cfg_dict['objective']['alpha']}"
            results[config_key] = logs
            
            print(f"Completed configuration {i+1}/{len(cfg_dicts)}")
            print(f"Success rate: {logs.get('memory_eval/success_rate', 'N/A'):.3f}")
            
        except Exception as e:
            print(f"Error in configuration {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary results
    print_memory_evaluation_summary(results)
    
    return results


def print_memory_evaluation_summary(results):
    """Print a summary of memory evaluation results."""
    print(f"\n{'='*80}")
    print("MEMORY EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    if not results:
        print("No results to display.")
        return
    
    # Group results by memory test mode
    by_mode = {}
    for config_key, logs in results.items():
        mode = config_key.split('_')[0] + '_' + config_key.split('_')[1]
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append((config_key, logs))
    
    for mode, mode_results in by_mode.items():
        print(f"\n{mode.upper()} RESULTS:")
        print("-" * 40)
        
        for config_key, logs in mode_results:
            success_rate = logs.get('memory_eval/success_rate', 0.0)
            visual_div = logs.get('memory_eval/mean_div_visual_emb', 0.0)
            proprio_div = logs.get('memory_eval/mean_div_proprio_emb', 0.0)
            
            print(f"  {config_key}:")
            print(f"    Success Rate: {success_rate:.3f}")
            print(f"    Visual Divergence: {visual_div:.3f}")
            print(f"    Proprio Divergence: {proprio_div:.3f}")
            
            # Print memory-specific metrics
            for key, value in logs.items():
                if key.startswith('memory_eval/') and 'memory' in key.lower():
                    print(f"    {key.split('/')[-1]}: {value:.3f}")
    
    print(f"\n{'='*80}")


@hydra.main(config_path="conf", config_name="memory_eval")
def main(cfg: OmegaConf):
    """Main function for memory evaluation."""
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Memory evaluation results saved dir: {cfg['saved_folder']}")
    
    # Set random seed
    seed(cfg.seed)
    
    # Run memory evaluation suite
    results = run_memory_evaluation_suite(
        ckpt_base_path=cfg.ckpt_base_path,
        model_name=cfg.model_name,
        model_epoch=cfg.model_epoch,
        memory_test_modes=cfg.memory_test_modes,
        n_memory_objects_list=cfg.n_memory_objects_list,
        goal_H_list=cfg.goal_H_list,
        n_evals=cfg.n_evals,
        seed=cfg.seed,
        wandb_logging=cfg.wandb_logging
    )
    
    # Save results
    import json
    with open("memory_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nMemory evaluation completed. Results saved to memory_evaluation_results.json")


if __name__ == "__main__":
    main()
