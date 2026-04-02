#!/usr/bin/env python3
"""
Scaling law validation script for Bangkong LLM Training System
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.config.loader import ConfigLoader
from bangkong.validation.scaling_law_validator import ScalingLawValidator

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate scaling laws for pre-intelligent initialization")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="./scaling_law_results", 
                       help="Directory to save results")
    parser.add_argument("--reduction-factor", type=float, default=None,
                       help="Reduction factor for pre-intelligent initialization")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config_loader = ConfigLoader(args.config)
        config = config_loader.config
    else:
        # Create a default config for demonstration
        from bangkong.config.schemas import BangkongConfig
        config = BangkongConfig()
    
    # Initialize validator
    validator = ScalingLawValidator(config, args.output_dir)
    
    # Model sizes to evaluate
    model_sizes = [50e6, 150e6, 350e6, 1e9, 3e9, 10e9]  # 50M to 10B params
    
    # Run validation
    print("Running scaling law validation...")
    reduction_factor = args.reduction_factor or getattr(config.training, 'preint_reduction_factor', 0.4)
    df = validator.run_validation_experiment(model_sizes, reduction_factor)
    
    # Generate plots and reports
    plot_path = Path(args.output_dir) / "scaling_law_plot.png"
    report_path = Path(args.output_dir) / "scaling_law_report.csv"
    
    try:
        validator.plot_scaling_laws(df, str(plot_path))
    except Exception as e:
        print(f"Warning: Could not generate plot: {e}")
        
    validator.generate_report(df, str(report_path))
    
    print(f"\nResults saved to:")
    print(f"  - Report: {report_path}")

if __name__ == "__main__":
    main()