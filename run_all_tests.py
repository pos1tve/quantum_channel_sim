"""
Main runner script for Holevo capacity additivity testing.
This script runs all tests and generates comprehensive reports.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')  # Suppress matplotlib warnings

# Import all modules
from test_additivity import AdditivityTester
from statistical_models import perform_advanced_statistical_analysis
from counterexamples import (
    test_known_counterexamples, 
    analyze_violation_landscape,
    test_entanglement_breaking_threshold,
    test_higher_order_additivity
)


def main():
    """Main function to run all additivity tests."""
    print("=" * 80)
    print("COMPREHENSIVE HOLEVO CAPACITY ADDITIVITY TESTING FRAMEWORK")
    print("=" * 80)
    print("\nThis framework tests the additivity conjecture for quantum channel capacities.")
    print("The conjecture asks whether χ(N₁⊗N₂) = χ(N₁) + χ(N₂) for all channels.")
    print("\n" + "=" * 80)
    
    # Part 1: Systematic testing
    print("\nPART 1: SYSTEMATIC TESTING OF RANDOM CHANNELS")
    print("-" * 60)
    
    tester = AdditivityTester(results_dir="results")
    
    # Run systematic tests with fewer samples for demonstration
    print("\nRunning systematic tests on various channel types...")
    tester.run_systematic_tests(num_samples_per_type=20)
    
    # Test specific extreme cases
    print("\nTesting extreme parameter cases...")
    tester.test_specific_counterexamples()
    
    # Generate basic analysis
    print("\nGenerating basic analysis...")
    analysis = tester.analyze_results()
    
    print(f"\nBasic Results Summary:")
    print(f"  Total tests: {analysis['total_tests']}")
    print(f"  Additivity rate: {analysis['additivity_rate']*100:.1f}%")
    print(f"  Mean violation: {analysis['mean_violation']:.6f}")
    print(f"  Max violation: {analysis['max_violation']:.6f}")
    print(f"  Std deviation: {analysis['std_violation']:.6f}")
    
    # Part 2: Statistical analysis
    print("\n" + "=" * 80)
    print("PART 2: ADVANCED STATISTICAL ANALYSIS")
    print("-" * 60)
    
    if tester.results:
        print("\nPerforming advanced statistical analysis...")
        stat_results = perform_advanced_statistical_analysis(tester.results)
        
        # Print key findings
        bayes = stat_results['bayesian']
        print(f"\nBayesian Analysis:")
        print(f"  Posterior mean violation: {bayes['posterior_mean']:.6f}")
        print(f"  95% Credible interval: [{bayes['credible_interval'][0]:.6f}, {bayes['credible_interval'][1]:.6f}]")
        print(f"  Evidence: {bayes['evidence_interpretation']}")
        
        mixture = stat_results['mixture_model']
        print(f"\nMixture Model Analysis:")
        print(f"  Optimal components: {mixture['optimal_components']}")
        print(f"  Component means: {[f'{m:.6f}' for m in mixture['means']]}")
        
        regression = stat_results['regression']
        print(f"\nPredictive Modeling:")
        print(f"  Random Forest R² (test): {regression['test_r2']:.3f}")
        print(f"  Top predictive features:")
        for _, row in regression['feature_importance'].head(3).iterrows():
            print(f"    - {row['feature']}: {row['importance']:.3f}")
    
    # Part 3: Known counterexamples
    print("\n" + "=" * 80)
    print("PART 3: TESTING KNOWN COUNTEREXAMPLES")
    print("-" * 60)
    
    print("\nTesting theoretical counterexamples to additivity...")
    counterexample_results = test_known_counterexamples()
    
    # Part 4: Violation landscape
    print("\n" + "=" * 80)
    print("PART 4: ANALYZING VIOLATION LANDSCAPE")
    print("-" * 60)
    
    violation_map = analyze_violation_landscape()
    
    # Part 5: Special cases
    print("\n" + "=" * 80)
    print("PART 5: SPECIAL CASES AND BOUNDARIES")
    print("-" * 60)
    
    # Test entanglement-breaking threshold
    eb_results = test_entanglement_breaking_threshold()
    
    # Test higher-order additivity
    print("\nTesting higher-order tensor products...")
    higher_order = test_higher_order_additivity()
    
    # Generate plots and save results
    print("\n" + "=" * 80)
    print("GENERATING FINAL REPORTS AND VISUALIZATIONS")
    print("-" * 60)
    
    # Generate plots
    print("\nGenerating visualization plots...")
    tester.plot_results()
    
    # Save all results
    print("\nSaving results to JSON...")
    tester.save_results()
    
    # Generate comprehensive report
    print("\nGenerating final report...")
    report = tester.generate_report()
    
    # Additional summary
    print("\n" + "=" * 80)
    print("TESTING COMPLETE - SUMMARY OF FINDINGS")
    print("=" * 80)
    
    print(f"""
Key Findings:
1. Tested {analysis['total_tests']} channel pairs across multiple types
2. Found additivity violations in {(1-analysis['additivity_rate'])*100:.1f}% of cases
3. Maximum observed violation: {analysis['max_violation']:.6f}
4. Statistical evidence: {bayes['evidence_interpretation']}

The results {'support' if analysis['additivity_rate'] > 0.95 else 'challenge'} the general additivity conjecture.
While many channels appear to satisfy additivity within numerical precision,
{'no significant' if analysis['additivity_rate'] > 0.95 else 'some'} violations were observed.

All results have been saved to the 'results' directory for further analysis.
""")
    
    print("\nThank you for using the Holevo Capacity Additivity Testing Framework!")
    print("=" * 80)


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc() 