"""
Simple demo of the Holevo capacity additivity testing framework.
This script shows basic usage without running the full test suite.
"""

import numpy as np
import matplotlib.pyplot as plt
from holevo_capacity import (
    DepolarizingChannel, AmplitudeDampingChannel, 
    holevo_capacity, test_additivity
)
from counterexamples import SpecificCounterexamples


def demo_basic_test():
    """Demonstrate basic additivity testing."""
    print("=" * 60)
    print("DEMO: Basic Additivity Test")
    print("=" * 60)
    
    # Create two quantum channels
    print("\nCreating quantum channels:")
    ch1 = DepolarizingChannel(p=0.7, dim=2)
    ch2 = AmplitudeDampingChannel(gamma=0.3)
    
    print(f"  Channel 1: Depolarizing channel (p=0.7)")
    print(f"  Channel 2: Amplitude damping channel (γ=0.3)")
    
    # Test additivity
    print("\nTesting additivity...")
    result = test_additivity(ch1, ch2, num_trials=50)
    
    print(f"\nResults:")
    print(f"  χ(N₁) = {result['chi1']:.6f}")
    print(f"  χ(N₂) = {result['chi2']:.6f}")
    print(f"  χ(N₁) + χ(N₂) = {result['additive_capacity']:.6f}")
    print(f"  χ(N₁⊗N₂) = {result['chi_tensor']:.6f}")
    print(f"  Violation = {result['violation']:.6f}")
    print(f"  Relative violation = {result['relative_violation']*100:.2f}%")
    
    if result['is_additive']:
        print("\n✓ Additivity satisfied (within tolerance)")
    else:
        print("\n✗ Additivity violated!")
    
    return result


def demo_parameter_sweep():
    """Demonstrate how violations depend on channel parameters."""
    print("\n" + "=" * 60)
    print("DEMO: Parameter Sweep Analysis")
    print("=" * 60)
    
    print("\nAnalyzing how depolarizing channel violations depend on p...")
    
    # Parameter values to test
    p_values = np.linspace(0.1, 0.9, 9)
    violations = []
    
    # Test each parameter value
    for p in p_values:
        ch = DepolarizingChannel(p, dim=2)
        result = test_additivity(ch, ch, num_trials=30)
        violations.append(result['violation'])
        print(f"  p = {p:.1f}: violation = {result['violation']:.6f}")
    
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(p_values, violations, 'bo-', markersize=8, linewidth=2)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Perfect additivity')
    plt.xlabel('Depolarizing parameter p', fontsize=12)
    plt.ylabel('Additivity violation', fontsize=12)
    plt.title('Additivity Violations vs Channel Parameter', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return p_values, violations


def demo_known_counterexample():
    """Demonstrate a known counterexample to additivity."""
    print("\n" + "=" * 60)
    print("DEMO: Known Counterexample (Werner-Holevo Channel)")
    print("=" * 60)
    
    print("\nTesting Werner-Holevo channel (known to violate additivity)...")
    
    # Create Werner-Holevo channel
    ch = SpecificCounterexamples.werner_holevo_channel(p=0.5, d=3)
    
    print(f"  Channel: Werner-Holevo (p=0.5, d=3)")
    
    # Test additivity
    result = test_additivity(ch, ch, num_trials=100)
    
    print(f"\nResults:")
    print(f"  χ(N) = {result['chi1']:.6f}")
    print(f"  χ(N⊗N) = {result['chi_tensor']:.6f}")
    print(f"  Expected (if additive) = {result['additive_capacity']:.6f}")
    print(f"  Violation = {result['violation']:.6f}")
    print(f"  Relative violation = {result['relative_violation']*100:.2f}%")
    
    if result['violation'] > 0:
        print("\n→ Super-additivity detected! χ(N⊗N) > 2χ(N)")
    
    return result


def demo_channel_comparison():
    """Compare different channel types."""
    print("\n" + "=" * 60)
    print("DEMO: Channel Type Comparison")
    print("=" * 60)
    
    print("\nComparing additivity properties of different channel types...")
    
    # Define channels to test
    channels = [
        ("Depolarizing (p=0.5)", DepolarizingChannel(0.5, dim=2)),
        ("Depolarizing (p=0.9)", DepolarizingChannel(0.9, dim=2)),
        ("Amplitude Damping (γ=0.5)", AmplitudeDampingChannel(0.5)),
        ("Amplitude Damping (γ=0.9)", AmplitudeDampingChannel(0.9)),
    ]
    
    results = []
    
    # Test each channel with itself
    for name, channel in channels:
        print(f"\n  Testing {name}...")
        result = test_additivity(channel, channel, num_trials=50)
        result['name'] = name
        results.append(result)
        print(f"    Violation: {result['violation']:.6f}")
        print(f"    Capacity: {result['chi1']:.6f}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    names = [r['name'] for r in results]
    violations = [r['violation'] for r in results]
    capacities = [r['chi1'] for r in results]
    
    # Violations plot
    ax1.bar(range(len(names)), violations, color=['blue', 'lightblue', 'green', 'lightgreen'])
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Additivity Violation')
    ax1.set_title('Violations by Channel Type')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Capacities plot
    ax2.bar(range(len(names)), capacities, color=['blue', 'lightblue', 'green', 'lightgreen'])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Holevo Capacity')
    ax2.set_title('Channel Capacities')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return results


def main():
    """Run all demos."""
    print("HOLEVO CAPACITY ADDITIVITY - DEMONSTRATION")
    print("==========================================\n")
    
    # Run demos
    demo_basic_test()
    demo_parameter_sweep()
    demo_known_counterexample()
    demo_channel_comparison()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nFor comprehensive testing, run: python run_all_tests.py")
    print("For more examples, see the README.md file")


if __name__ == "__main__":
    main() 