#!/usr/bin/env python3
"""
Scaling law validation for pre-intelligent initialization
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from ..config.schemas import BangkongConfig
from ..models.config_loader import get_models_config

class ScalingLawValidator:
    """Validate scaling laws for pre-intelligent initialization."""
    
    def __init__(self, config: BangkongConfig, output_dir: str = None):
        """
        Initialize scaling law validator.
        
        Args:
            config: Bangkong configuration
            output_dir: Directory to save results
        """
        self.config = config
        self.models_config = get_models_config()
        
        # Get output directory from config if not provided
        if output_dir is None:
            output_dir = self.models_config.get("models.scaling_law_validator.output_dir", "./scaling_law_results")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Scaling law parameters (can be calibrated)
        scaling_config = self.models_config.get("models.scaling_law_validator", {})
        self.alpha = getattr(config.training, 'scaling_law_alpha', scaling_config.get('alpha', 0.8))  # Exponent in T = a * N^alpha
        self.target_N_ref = getattr(config.training, 'scaling_law_N_ref', scaling_config.get('target_N_ref', 0.35e9))  # 350M params reference
        self.target_T_ref = getattr(config.training, 'scaling_law_T_ref', scaling_config.get('target_T_ref', 1.0e9))   # 1 billion tokens reference
        
        # Compute coefficient a so that baseline matches reference point
        self.a = self.target_T_ref / (self.target_N_ref ** self.alpha)
        
    def compute_baseline_tokens(self, N: float) -> float:
        """
        Compute baseline token requirement.
        
        Args:
            N: Number of parameters
            
        Returns:
            Required tokens for baseline model
        """
        return self.a * (N ** self.alpha)
        
    def compute_preint_tokens(self, N: float, reduction_factor: float = None) -> float:
        """
        Compute pre-intelligent token requirement.
        
        Args:
            N: Number of parameters
            reduction_factor: Reduction factor (default from config or 0.4)
            
        Returns:
            Required tokens for pre-intelligent model
        """
        if reduction_factor is None:
            reduction_factor = getattr(self.config.training, 'preint_reduction_factor', 0.4)
            
        baseline = self.compute_baseline_tokens(N)
        return (1.0 - reduction_factor) * baseline
        
    def estimate_gpu_hours(self, N: float, T: float, gpu_hours_ref: float = None) -> float:
        """
        Estimate GPU hours required.
        
        Args:
            N: Number of parameters
            T: Number of tokens
            gpu_hours_ref: Reference GPU hours
            
        Returns:
            Estimated GPU hours
        """
        # Get GPU hours reference from config if not provided
        if gpu_hours_ref is None:
            scaling_config = self.models_config.get("models.scaling_law_validator", {})
            gpu_hours_ref = getattr(self.config.training, 'scaling_gpu_hours_ref', scaling_config.get('gpu_hours_ref', 1000.0))
            
        # Compute conversion factor
        c = gpu_hours_ref / (self.target_T_ref * self.target_N_ref)
        return T * N * c
        
    def run_validation_experiment(self, model_sizes: List[float], 
                                reduction_factor: float = None) -> pd.DataFrame:
        """
        Run validation experiment across model sizes.
        
        Args:
            model_sizes: List of model sizes (parameters)
            reduction_factor: Reduction factor for pre-intelligent initialization
            
        Returns:
            DataFrame with results
        """
        if reduction_factor is None:
            reduction_factor = getattr(self.config.training, 'preint_reduction_factor', 0.4)
            
        results = []
        
        for N in model_sizes:
            T_baseline = self.compute_baseline_tokens(N)
            T_preint = self.compute_preint_tokens(N, reduction_factor)
            savings_tokens = T_baseline - T_preint
            savings_pct = 100.0 * savings_tokens / T_baseline
            
            # GPU hour estimates
            gpu_hours_ref = getattr(self.config.training, 'scaling_gpu_hours_ref', 1000.0)
            gpu_hours_base = self.estimate_gpu_hours(N, T_baseline, gpu_hours_ref)
            gpu_hours_pi = self.estimate_gpu_hours(N, T_preint, gpu_hours_ref)
            gpu_hours_saved = gpu_hours_base - gpu_hours_pi
            
            results.append({
                "N_params": int(N),
                "T_baseline_tokens": int(T_baseline),
                "T_preint_tokens": int(T_preint),
                "tokens_saved": int(savings_tokens),
                "pct_saved": round(savings_pct, 1),
                "GPUh_baseline": round(gpu_hours_base, 1),
                "GPUh_preint": round(gpu_hours_pi, 1),
                "GPUh_saved": round(gpu_hours_saved, 1)
            })
            
        return pd.DataFrame(results)
        
    def plot_scaling_laws(self, df: pd.DataFrame, save_path: str = None):
        """
        Plot scaling laws.
        
        Args:
            df: DataFrame with results
            save_path: Path to save plot
        """
        # Get plotting parameters from config
        scaling_config = self.models_config.get("models.scaling_law_validator", {})
        plot_range = scaling_config.get("plot_range", {"min": 50e6, "max": 10e9})
        plot_points = scaling_config.get("plot_points", 200)
        
        # Model sizes for plotting curve
        N_plot = np.logspace(np.log10(plot_range["min"]), np.log10(plot_range["max"]), plot_points)
        T_base_plot = [self.compute_baseline_tokens(N) for N in N_plot]
        T_pi_plot = [self.compute_preint_tokens(N) for N in N_plot]
        
        plt.figure(figsize=(10, 6))
        plt.loglog(N_plot, T_base_plot, label="Baseline (random init)", linewidth=2)
        plt.loglog(N_plot, T_pi_plot, label=f"Pre-Intelligent (-{int(getattr(self.config.training, 'preint_reduction_factor', 0.4)*100)}% tokens)", linewidth=2)
        plt.scatter(df["N_params"], df["T_baseline_tokens"], c='C0', s=50, zorder=5)
        plt.scatter(df["N_params"], df["T_preint_tokens"], c='C1', s=50, zorder=5)
        plt.xlabel("Model size N (parameters)")
        plt.ylabel("Required tokens to reach target perf (T)")
        plt.title("Scaling: Baseline vs Pre-Intelligent Init")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.4)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_report(self, df: pd.DataFrame, save_path: str = None):
        """
        Generate detailed report.
        
        Args:
            df: DataFrame with results
            save_path: Path to save report
        """
        # Make readable format
        df_display = df.copy()
        df_display["N_readable"] = df_display["N_params"].apply(
            lambda x: f"{x/1e6:.0f}M" if x < 1e9 else f"{x/1e9:.1f}B"
        )
        df_display = df_display[[
            "N_readable", "T_baseline_tokens", "T_preint_tokens", 
            "tokens_saved", "pct_saved", "GPUh_baseline", 
            "GPUh_preint", "GPUh_saved"
        ]].rename(columns={
            "N_readable": "Model size",
            "T_baseline_tokens": "Baseline tokens",
            "T_preint_tokens": "PreInt tokens",
            "tokens_saved": "Tokens saved",
            "pct_saved": "% saved",
            "GPUh_baseline": "GPUh baseline",
            "GPUh_preint": "GPUh PreInt",
            "GPUh_saved": "GPUh saved"
        })
        
        print("Scaling Law Validation Results:")
        print("=" * 80)
        print(df_display.to_string(index=False))
        
        if save_path:
            df_display.to_csv(save_path, index=False)
            
        return df_display