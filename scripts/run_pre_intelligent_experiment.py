#!/usr/bin/env python3
"""
Experiment runner for pre-intelligent initialization
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.experiments.pre_intelligent_experiment import PreIntelligentExperiment

def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run pre-intelligent initialization experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--experiment-name", type=str, help="Name of experiment")
    parser.add_argument("--ablation-only", action="store_true", help="Run only ablation study")
    parser.add_argument("--scaling-only", action="store_true", help="Run only scaling validation")
    parser.add_argument("--strategies", type=str, nargs="+", 
                       help="Initialization strategies to test (default: random xavier kaiming pre_intelligent)")
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = PreIntelligentExperiment(args.config, args.experiment_name)
    
    ablation_df = None
    scaling_df = None
    
    # Run ablation study
    if not args.scaling_only:
        print("Running ablation study...")
        ablation_df = experiment.run_ablation_study(args.strategies)
        print("Ablation study results:")
        print(ablation_df.to_string())
        
    # Run scaling validation
    if not args.ablation_only:
        print("Running scaling validation...")
        scaling_df = experiment.run_scaling_validation()
        print("Scaling validation results:")
        print(scaling_df.to_string())
        
    # Generate comprehensive report
    print("Generating comprehensive report...")
    report = experiment.generate_comprehensive_report(ablation_df, scaling_df)
    
    print(f"Experiment completed. Results saved in {experiment.output_dir}")

if __name__ == "__main__":
    main()