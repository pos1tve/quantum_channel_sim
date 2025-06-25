# Holevo Capacity Additivity Testing Framework

## Overview

This framework provides comprehensive tools for testing the additivity conjecture of Holevo capacity for quantum channels. The additivity conjecture asks whether the Holevo capacity of a tensor product of quantum channels equals the sum of their individual capacities:

**χ(N₁ ⊗ N₂) = χ(N₁) + χ(N₂)**

While this conjecture was disproved in general by Hastings (2009), it remains an important question to understand which channels satisfy additivity and under what conditions violations occur.

## Features

### 1. **Quantum Channel Implementations**
- **Depolarizing Channel**: Classic noise model with parameter p
- **Amplitude Damping Channel**: Models energy dissipation
- **Erasure Channel**: Models information loss
- **Werner-Holevo Channel**: Generalized depolarizing channel
- **Hastings Channel**: First known counterexample to additivity
- **Random Unitary Channels**: For exploring general cases

### 2. **Holevo Capacity Calculation**
- Optimization-based calculation of Holevo capacity
- Support for arbitrary quantum channels
- Efficient numerical methods

### 3. **Comprehensive Testing Suite**
- Systematic testing across channel types
- Known counterexample verification
- Parameter sweep analysis
- Higher-order additivity tests (3+ channels)

### 4. **Advanced Statistical Analysis**
- **Bayesian Analysis**: Posterior distributions and credible intervals
- **Hypothesis Testing**: Statistical significance of violations
- **Mixture Models**: Identify different violation regimes
- **Extreme Value Theory**: Analyze maximum violations
- **Machine Learning**: Predict violations from channel properties
- **Correlation Analysis**: Identify key predictors

### 5. **Visualization Tools**
- Violation distributions
- Parameter landscapes
- Correlation heatmaps
- Statistical diagnostic plots

## Installation

```bash
# Clone the repository
cd quantum_channels

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run All Tests
```bash
python run_all_tests.py
```

This will:
1. Test various random channels
2. Analyze known counterexamples
3. Perform statistical analysis
4. Generate comprehensive reports
5. Save results and visualizations

### Run Specific Tests

```python
from test_additivity import AdditivityTester
from holevo_capacity import DepolarizingChannel, AmplitudeDampingChannel

# Create tester
tester = AdditivityTester()

# Test specific channels
ch1 = DepolarizingChannel(p=0.8, dim=2)
ch2 = AmplitudeDampingChannel(gamma=0.5)
result = tester.test_channel_pair(ch1, ch2, num_trials=100)

print(f"Violation: {result['violation']:.6f}")
```

### Test Known Counterexamples

```python
from counterexamples import test_known_counterexamples

results = test_known_counterexamples()
```

## Module Structure

### `holevo_capacity.py`
Core implementations:
- Base `QuantumChannel` class
- Channel implementations
- Holevo capacity calculation
- Helper functions (entropy, partial trace, etc.)

### `test_additivity.py`
Testing framework:
- `AdditivityTester` class
- Systematic testing methods
- Result analysis and visualization
- Report generation

### `statistical_models.py`
Advanced statistical analysis:
- Bayesian inference
- Mixture models
- Extreme value analysis
- Machine learning predictions

### `counterexamples.py`
Known counterexamples:
- Hastings channel
- Werner-Holevo channel
- Violation landscape analysis
- Special case testing

### `run_all_tests.py`
Main runner script that orchestrates all tests

## Understanding the Results

### Violation Metrics
- **Violation**: χ(N₁⊗N₂) - χ(N₁) - χ(N₂)
- **Relative Violation**: Violation / (χ(N₁) + χ(N₂))
- Positive violation = super-additivity
- Negative violation = sub-additivity

### Statistical Interpretation
- **Bayes Factor**: Evidence for/against perfect additivity
- **Credible Intervals**: Uncertainty in mean violation
- **Mixture Components**: Different violation regimes

### Key Findings
1. Most channels satisfy additivity within numerical precision
2. Specific counterexamples exist (e.g., Hastings channel)
3. Violations tend to be small but statistically significant
4. Entanglement-breaking channels always satisfy additivity

## Advanced Usage

### Custom Channel Implementation

```python
class MyCustomChannel(QuantumChannel):
    def __init__(self, param):
        super().__init__(dim_in=2, dim_out=2)
        self.param = param
    
    def apply(self, rho):
        # Implement channel action
        return transformed_rho
    
    def kraus_operators(self):
        # Return list of Kraus operators
        return [K1, K2, ...]
```

### Batch Testing

```python
# Test parameter sweep
import numpy as np
from holevo_capacity import DepolarizingChannel

p_values = np.linspace(0, 1, 50)
violations = []

for p in p_values:
    ch = DepolarizingChannel(p, dim=2)
    result = test_additivity(ch, ch, num_trials=100)
    violations.append(result['violation'])
```

## Output Files

The framework generates several output files in the `results/` directory:

- `additivity_results_[timestamp].json`: Raw test results
- `additivity_report_[timestamp].txt`: Human-readable report
- `additivity_analysis_[timestamp].png`: Visualization plots

## Theoretical Background

The Holevo capacity of a quantum channel N is:

**χ(N) = max_{p_i, ρ_i} [S(∑ p_i N(ρ_i)) - ∑ p_i S(N(ρ_i))]**

Where:
- S is the von Neumann entropy
- {p_i, ρ_i} is an ensemble of input states

The additivity conjecture states this should be additive for tensor products.

## References

1. Hastings, M. B. (2009). "Superadditivity of communication capacity using entangled inputs". Nature Physics 5, 255.
2. Holevo, A. S. (1998). "The capacity of the quantum channel with general signal states". IEEE Trans. Inf. Theory 44, 269–273.
3. Werner, R. F., & Holevo, A. S. (2002). "Counterexample to an additivity conjecture for output purity of quantum channels". J. Math. Phys. 43, 4353.
4. Shor, P. W. (2004). "Equivalence of Additivity Questions in Quantum Information Theory". Commun. Math. Phys. 246, 453–472.

## Contributing

Feel free to extend the framework with:
- New channel implementations
- Additional statistical tests
- Improved optimization algorithms
- More visualization options

## License

This project is provided for educational and research purposes.

## Contact

For questions or collaborations, please open an issue in the repository. 