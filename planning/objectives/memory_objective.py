import torch
import numpy as np
from typing import Dict, Any


class MemoryObjective:
    """
    Memory-specific objective function for evaluating world model memory capabilities.
    This objective focuses on memory-related metrics rather than just goal reaching.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Weight for memory-specific terms in the objective
        """
        self.alpha = alpha
    
    def __call__(
        self,
        wm,
        obs_0: Dict[str, torch.Tensor],
        obs_g: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        memory_objects: list = None,
        memory_questions: list = None,
        memory_test_mode: str = "object_recall",
        **kwargs
    ) -> torch.Tensor:
        """
        Compute memory-specific objective.
        
        Args:
            wm: World model
            obs_0: Initial observations
            obs_g: Goal observations
            actions: Action sequence
            memory_objects: List of memory objects in the environment
            memory_questions: List of memory questions
            memory_test_mode: Type of memory test being performed
            
        Returns:
            Objective value (lower is better)
        """
        with torch.no_grad():
            # Rollout world model
            z_obses, _ = wm.rollout(obs_0, actions)
            
            # Basic goal reaching objective
            goal_objective = self._compute_goal_objective(z_obses, obs_g)
            
            # Memory-specific objectives
            memory_objective = self._compute_memory_objective(
                z_obses, memory_objects, memory_questions, memory_test_mode
            )
            
            # Combine objectives
            total_objective = goal_objective + self.alpha * memory_objective
            
            return total_objective
    
    def _compute_goal_objective(
        self, 
        z_obses: Dict[str, torch.Tensor], 
        obs_g: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute basic goal reaching objective."""
        # Get final predicted observations
        final_z_obs = {}
        for key in z_obses.keys():
            final_z_obs[key] = z_obses[key][:, -1, ...]
        
        # Compute distance to goal in latent space
        goal_distances = []
        for key in final_z_obs.keys():
            if key in obs_g:
                # Encode goal observations
                goal_z = obs_g[key]  # Assuming obs_g is already encoded
                pred_z = final_z_obs[key]
                
                # Compute L2 distance
                distance = torch.norm(pred_z - goal_z, dim=-1).mean()
                goal_distances.append(distance)
        
        return torch.stack(goal_distances).mean() if goal_distances else torch.tensor(0.0)
    
    def _compute_memory_objective(
        self,
        z_obses: Dict[str, torch.Tensor],
        memory_objects: list,
        memory_questions: list,
        memory_test_mode: str
    ) -> torch.Tensor:
        """Compute memory-specific objective based on test mode."""
        if memory_test_mode == "object_recall":
            return self._compute_object_recall_objective(z_obses, memory_objects)
        elif memory_test_mode == "color_memory":
            return self._compute_color_memory_objective(z_obses, memory_objects)
        elif memory_test_mode == "sequential_memory":
            return self._compute_sequential_memory_objective(z_obses, memory_objects)
        elif memory_test_mode == "navigation":
            return self._compute_navigation_objective(z_obses, memory_objects)
        else:
            return torch.tensor(0.0)
    
    def _compute_object_recall_objective(
        self, 
        z_obses: Dict[str, torch.Tensor], 
        memory_objects: list
    ) -> torch.Tensor:
        """Compute object recall objective."""
        # This would measure how well the world model can predict
        # the presence and location of memory objects
        # For now, return a placeholder
        return torch.tensor(0.0)
    
    def _compute_color_memory_objective(
        self, 
        z_obses: Dict[str, torch.Tensor], 
        memory_objects: list
    ) -> torch.Tensor:
        """Compute color memory objective."""
        # This would measure how well the world model can predict
        # color-object associations
        # For now, return a placeholder
        return torch.tensor(0.0)
    
    def _compute_sequential_memory_objective(
        self, 
        z_obses: Dict[str, torch.Tensor], 
        memory_objects: list
    ) -> torch.Tensor:
        """Compute sequential memory objective."""
        # This would measure how well the world model can predict
        # temporal sequences of object placements
        # For now, return a placeholder
        return torch.tensor(0.0)
    
    def _compute_navigation_objective(
        self, 
        z_obses: Dict[str, torch.Tensor], 
        memory_objects: list
    ) -> torch.Tensor:
        """Compute navigation objective."""
        # This would measure how well the world model can predict
        # navigation paths to remembered objects
        # For now, return a placeholder
        return torch.tensor(0.0)
