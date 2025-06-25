import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from typing import List, Dict, Tuple
import json
from datetime import datetime
import os

from holevo_capacity import (
    QuantumChannel, DepolarizingChannel, AmplitudeDampingChannel, 
    ErasureChannel, test_additivity, generate_random_channel,
    holevo_capacity, tensor_product_channel
)


class AdditivityTester:
    """Comprehensive testing framework for Holevo capacity additivity."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = []
        
    def test_channel_pair(self, channel1: QuantumChannel, channel2: QuantumChannel,
                         num_trials: int = 100, name: str = None) -> dict:
        """Test additivity for a specific pair of channels."""
        result = test_additivity(channel1, channel2, num_trials)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['channel1_type'] = channel1.__class__.__name__
        result['channel2_type'] = channel2.__class__.__name__
        result['num_trials'] = num_trials
        
        if name:
            result['test_name'] = name
            
        # Add channel parameters if available
        if hasattr(channel1, 'p'):
            result['channel1_p'] = channel1.p
        if hasattr(channel1, 'gamma'):
            result['channel1_gamma'] = channel1.gamma
        if hasattr(channel2, 'p'):
            result['channel2_p'] = channel2.p
        if hasattr(channel2, 'gamma'):
            result['channel2_gamma'] = channel2.gamma
            
        self.results.append(result)
        return result
    
    def run_systematic_tests(self, num_samples_per_type: int = 50):
        """Run systematic tests on different channel combinations."""
        print("Running systematic additivity tests...")
        
        # Test 1: Depolarizing channels with varying parameters
        print("\n1. Testing depolarizing channels...")
        for _ in range(num_samples_per_type):
            p1 = np.random.uniform(0, 1)
            p2 = np.random.uniform(0, 1)
            ch1 = DepolarizingChannel(p1, dim=2)
            ch2 = DepolarizingChannel(p2, dim=2)
            self.test_channel_pair(ch1, ch2, num_trials=50, 
                                 name=f"Depol({p1:.3f})⊗Depol({p2:.3f})")
        
        # Test 2: Amplitude damping channels
        print("\n2. Testing amplitude damping channels...")
        for _ in range(num_samples_per_type):
            gamma1 = np.random.uniform(0, 1)
            gamma2 = np.random.uniform(0, 1)
            ch1 = AmplitudeDampingChannel(gamma1)
            ch2 = AmplitudeDampingChannel(gamma2)
            self.test_channel_pair(ch1, ch2, num_trials=50,
                                 name=f"AD({gamma1:.3f})⊗AD({gamma2:.3f})")
        
        # Test 3: Mixed channel types
        print("\n3. Testing mixed channel types...")
        for _ in range(num_samples_per_type // 2):
            p = np.random.uniform(0, 1)
            gamma = np.random.uniform(0, 1)
            ch1 = DepolarizingChannel(p, dim=2)
            ch2 = AmplitudeDampingChannel(gamma)
            self.test_channel_pair(ch1, ch2, num_trials=50,
                                 name=f"Depol({p:.3f})⊗AD({gamma:.3f})")
        
        # Test 4: Erasure channels
        print("\n4. Testing erasure channels...")
        for _ in range(num_samples_per_type // 2):
            p1 = np.random.uniform(0, 1)
            p2 = np.random.uniform(0, 1)
            ch1 = ErasureChannel(p1, dim=2)
            ch2 = ErasureChannel(p2, dim=2)
            self.test_channel_pair(ch1, ch2, num_trials=50,
                                 name=f"Erasure({p1:.3f})⊗Erasure({p2:.3f})")
        
        # Test 5: Random channels
        print("\n5. Testing random quantum channels...")
        for _ in range(num_samples_per_type // 2):
            ch1 = generate_random_channel(2, 'random')
            ch2 = generate_random_channel(2, 'random')
            self.test_channel_pair(ch1, ch2, num_trials=50,
                                 name="Random⊗Random")
    
    def test_specific_counterexamples(self):
        """Test known cases where additivity might fail."""
        print("\nTesting potential counterexamples...")
        
        # Test highly noisy channels
        for p in [0.99, 0.999]:
            ch1 = DepolarizingChannel(p, dim=2)
            ch2 = DepolarizingChannel(p, dim=2)
            result = self.test_channel_pair(ch1, ch2, num_trials=200,
                                          name=f"HighNoise_Depol({p})")
            print(f"High noise depolarizing p={p}: violation={result['violation']:.6f}")
        
        # Test extreme amplitude damping
        for gamma in [0.99, 0.999]:
            ch1 = AmplitudeDampingChannel(gamma)
            ch2 = AmplitudeDampingChannel(gamma)
            result = self.test_channel_pair(ch1, ch2, num_trials=200,
                                          name=f"Extreme_AD({gamma})")
            print(f"Extreme amplitude damping γ={gamma}: violation={result['violation']:.6f}")
    
    def analyze_results(self) -> Dict:
        """Analyze all test results and generate statistics."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        analysis = {
            'total_tests': len(df),
            'additive_cases': len(df[df['is_additive']]),
            'non_additive_cases': len(df[~df['is_additive']]),
            'additivity_rate': len(df[df['is_additive']]) / len(df),
            'mean_violation': df['violation'].mean(),
            'std_violation': df['violation'].std(),
            'max_violation': df['violation'].max(),
            'min_violation': df['violation'].min(),
            'mean_relative_violation': df['relative_violation'].mean(),
            'max_relative_violation': df['relative_violation'].max()
        }
        
        # Group by channel types
        type_analysis = {}
        for group_name, group_df in df.groupby(['channel1_type', 'channel2_type']):
            type_analysis[f"{group_name[0]}⊗{group_name[1]}"] = {
                'count': len(group_df),
                'additivity_rate': len(group_df[group_df['is_additive']]) / len(group_df),
                'mean_violation': group_df['violation'].mean(),
                'max_violation': group_df['violation'].max()
            }
        
        analysis['by_channel_type'] = type_analysis
        
        return analysis
    
    def plot_results(self):
        """Generate comprehensive plots of the results."""
        if not self.results:
            print("No results to plot!")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Holevo Capacity Additivity Analysis', fontsize=16)
        
        # 1. Violation distribution
        ax = axes[0, 0]
        ax.hist(df['violation'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', label='Perfect additivity')
        ax.set_xlabel('Violation (χ(N₁⊗N₂) - χ(N₁) - χ(N₂))')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Additivity Violations')
        ax.legend()
        
        # 2. Relative violation distribution
        ax = axes[0, 1]
        relative_violations = df['relative_violation'] * 100  # Convert to percentage
        ax.hist(relative_violations, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Relative Violation (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Relative Violations')
        
        # 3. Scatter plot: Individual capacities vs tensor product capacity
        ax = axes[0, 2]
        ax.scatter(df['additive_capacity'], df['chi_tensor'], alpha=0.5)
        min_val = min(df['additive_capacity'].min(), df['chi_tensor'].min())
        max_val = max(df['additive_capacity'].max(), df['chi_tensor'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect additivity')
        ax.set_xlabel('χ(N₁) + χ(N₂)')
        ax.set_ylabel('χ(N₁⊗N₂)')
        ax.set_title('Tensor Product vs Sum of Individual Capacities')
        ax.legend()
        
        # 4. Violation by channel type
        ax = axes[1, 0]
        channel_types = df.groupby(['channel1_type', 'channel2_type'])['violation'].mean()
        channel_types.plot(kind='bar', ax=ax)
        ax.set_xlabel('Channel Types')
        ax.set_ylabel('Mean Violation')
        ax.set_title('Mean Violation by Channel Type Combination')
        ax.tick_params(axis='x', rotation=45)
        
        # 5. Heatmap for depolarizing channels (if available)
        ax = axes[1, 1]
        depol_data = df[(df['channel1_type'] == 'DepolarizingChannel') & 
                       (df['channel2_type'] == 'DepolarizingChannel')]
        if len(depol_data) > 0 and 'channel1_p' in depol_data.columns:
            # Create bins for parameters
            p1_bins = np.linspace(0, 1, 11)
            p2_bins = np.linspace(0, 1, 11)
            
            # Create 2D histogram
            violation_map = np.zeros((10, 10))
            counts = np.zeros((10, 10))
            
            for _, row in depol_data.iterrows():
                i = min(int(row['channel1_p'] * 10), 9)
                j = min(int(row['channel2_p'] * 10), 9)
                violation_map[i, j] += row['violation']
                counts[i, j] += 1
            
            # Average violations
            with np.errstate(divide='ignore', invalid='ignore'):
                violation_map = np.where(counts > 0, violation_map / counts, np.nan)
            
            im = ax.imshow(violation_map, cmap='RdBu_r', origin='lower', 
                          extent=[0, 1, 0, 1], aspect='auto')
            ax.set_xlabel('p₁ (Channel 1 parameter)')
            ax.set_ylabel('p₂ (Channel 2 parameter)')
            ax.set_title('Mean Violation for Depolarizing Channels')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No depolarizing channel data', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # 6. Time series of violations (if multiple runs)
        ax = axes[1, 2]
        ax.plot(df['violation'].values, alpha=0.7)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Test Number')
        ax.set_ylabel('Violation')
        ax.set_title('Violations Over Test Sequence')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.results_dir, f'additivity_analysis_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def statistical_hypothesis_test(self) -> Dict:
        """Perform statistical hypothesis testing on additivity."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        violations = df['violation'].values
        
        # Test 1: One-sample t-test (H0: mean violation = 0)
        t_stat, p_value = stats.ttest_1samp(violations, 0)
        
        # Test 2: Normality test
        shapiro_stat, shapiro_p = stats.shapiro(violations)
        
        # Test 3: Sign test (non-parametric)
        positive_violations = np.sum(violations > 0)
        negative_violations = np.sum(violations < 0)
        sign_test_p = stats.binom_test(positive_violations, 
                                      positive_violations + negative_violations, 
                                      p=0.5)
        
        # Bootstrap confidence interval for mean violation
        bootstrap_means = []
        for _ in range(10000):
            sample = np.random.choice(violations, size=len(violations), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        return {
            'one_sample_t_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'reject_null': p_value < 0.05,
                'interpretation': 'Evidence against perfect additivity' if p_value < 0.05 
                                else 'No evidence against perfect additivity'
            },
            'normality_test': {
                'shapiro_statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            },
            'sign_test': {
                'positive_violations': positive_violations,
                'negative_violations': negative_violations,
                'p_value': sign_test_p,
                'interpretation': 'Systematic bias in violations' if sign_test_p < 0.05
                                else 'No systematic bias in violations'
            },
            'bootstrap_ci': {
                'mean_violation': np.mean(violations),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'contains_zero': ci_lower <= 0 <= ci_upper
            }
        }
    
    def save_results(self, filename: str = None):
        """Save all results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"additivity_results_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert numpy values to Python types for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    serializable_result[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    serializable_result[key] = int(value)
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(filepath, 'w') as f:
            json.dump({
                'results': serializable_results,
                'analysis': self.analyze_results(),
                'hypothesis_tests': self.statistical_hypothesis_test()
            }, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def generate_report(self):
        """Generate a comprehensive report of all findings."""
        analysis = self.analyze_results()
        hypothesis_tests = self.statistical_hypothesis_test()
        
        report = f"""
HOLEVO CAPACITY ADDITIVITY TEST REPORT
=====================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SUMMARY STATISTICS
------------------
Total tests performed: {analysis.get('total_tests', 0)}
Cases satisfying additivity: {analysis.get('additive_cases', 0)} ({analysis.get('additivity_rate', 0)*100:.1f}%)
Cases violating additivity: {analysis.get('non_additive_cases', 0)} ({(1-analysis.get('additivity_rate', 0))*100:.1f}%)

VIOLATION STATISTICS
-------------------
Mean violation: {analysis.get('mean_violation', 0):.6f}
Standard deviation: {analysis.get('std_violation', 0):.6f}
Maximum violation: {analysis.get('max_violation', 0):.6f}
Minimum violation: {analysis.get('min_violation', 0):.6f}
Mean relative violation: {analysis.get('mean_relative_violation', 0)*100:.2f}%
Maximum relative violation: {analysis.get('max_relative_violation', 0)*100:.2f}%

STATISTICAL HYPOTHESIS TESTING
------------------------------
"""
        
        if hypothesis_tests:
            t_test = hypothesis_tests.get('one_sample_t_test', {})
            report += f"""
One-sample t-test (H₀: mean violation = 0):
  t-statistic: {t_test.get('t_statistic', 0):.4f}
  p-value: {t_test.get('p_value', 1):.6f}
  Conclusion: {t_test.get('interpretation', 'N/A')}

Bootstrap 95% CI for mean violation:
  [{hypothesis_tests.get('bootstrap_ci', {}).get('ci_lower', 0):.6f}, 
   {hypothesis_tests.get('bootstrap_ci', {}).get('ci_upper', 0):.6f}]
  Contains zero: {hypothesis_tests.get('bootstrap_ci', {}).get('contains_zero', True)}
"""
        
        # Add channel-specific analysis
        if 'by_channel_type' in analysis:
            report += "\nANALYSIS BY CHANNEL TYPE\n"
            report += "-" * 50 + "\n"
            for channel_pair, stats in analysis['by_channel_type'].items():
                report += f"\n{channel_pair}:\n"
                report += f"  Tests: {stats['count']}\n"
                report += f"  Additivity rate: {stats['additivity_rate']*100:.1f}%\n"
                report += f"  Mean violation: {stats['mean_violation']:.6f}\n"
                report += f"  Max violation: {stats['max_violation']:.6f}\n"
        
        report += """
INTERPRETATION
--------------
The additivity conjecture states that χ(N₁⊗N₂) = χ(N₁) + χ(N₂) for all quantum channels.
Positive violations indicate super-additivity (χ(N₁⊗N₂) > χ(N₁) + χ(N₂)).
Negative violations indicate sub-additivity (χ(N₁⊗N₂) < χ(N₁) + χ(N₂)).

Based on the statistical analysis:
"""
        
        if hypothesis_tests and hypothesis_tests.get('one_sample_t_test', {}).get('p_value', 1) < 0.05:
            report += """
- There is statistically significant evidence against perfect additivity.
- The mean violation is significantly different from zero.
"""
        else:
            report += """
- No statistically significant evidence against perfect additivity was found.
- Observed violations may be due to numerical errors or sampling noise.
"""
        
       
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f'additivity_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\nReport saved to: {report_path}")
        
        return report


def main():
    """Main function to run comprehensive additivity tests."""
    print("=" * 60)
    print("HOLEVO CAPACITY ADDITIVITY TESTING FRAMEWORK")
    print("=" * 60)
    
    tester = AdditivityTester()
    
    tester.run_systematic_tests(num_samples_per_type=30)
    
    tester.test_specific_counterexamples()
    
    print("\nAnalyzing results...")
    analysis = tester.analyze_results()
    
    print(f"\nTotal tests: {analysis['total_tests']}")
    print(f"Additivity rate: {analysis['additivity_rate']*100:.1f}%")
    print(f"Mean violation: {analysis['mean_violation']:.6f}")
    print(f"Max violation: {analysis['max_violation']:.6f}")
    
    tester.plot_results()
    
    tester.save_results()
    
    tester.generate_report()
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main() 