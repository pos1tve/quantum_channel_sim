import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from holevo_capacity import (
    QuantumChannel, holevo_capacity, test_additivity, 
    von_neumann_entropy, tensor_product_channel
)


class SpecificCounterexamples:
    """Implementation of specific known counterexamples to additivity."""
    
    @staticmethod
    def hastings_channel(d: int = 4) -> 'HastingsChannel':
        """
        Hastings' counterexample channel (2009).
        This was the first explicit counterexample to additivity.
        """
        return HastingsChannel(d)
    
    @staticmethod
    def werner_holevo_channel(p: float) -> 'WernerHolevoChannel':
        """Werner-Holevo channel - known to violate additivity for certain p."""
        return WernerHolevoChannel(p)
    
    @staticmethod
    def antisymmetric_channel(d: int = 3) -> 'AntisymmetricChannel':
        """Antisymmetric subspace channel."""
        return AntisymmetricChannel(d)


class HastingsChannel(QuantumChannel):
    """
    Hastings' counterexample to additivity (2009).
    Based on random unitary channels with specific structure.
    """
    
    def __init__(self, d: int = 4):
        super().__init__(d, d)
        self.d = d
        self._generate_unitaries()
        
    def _generate_unitaries(self):
        """Generate the specific unitaries for Hastings' construction."""
        # This is a simplified version of Hastings' construction
        # The actual construction involves specific random unitaries
        np.random.seed(42)  # For reproducibility
        
        self.unitaries = []
        self.probabilities = []
        
        # Generate d² random unitaries
        num_unitaries = self.d ** 2
        
        for _ in range(num_unitaries):
            # Generate random unitary via QR decomposition
            A = np.random.randn(self.d, self.d) + 1j * np.random.randn(self.d, self.d)
            Q, R = np.linalg.qr(A)
            # Ensure proper phase
            D = np.diag(np.diag(R) / np.abs(np.diag(R)))
            U = Q @ D
            self.unitaries.append(U)
            
        # Equal probabilities
        self.probabilities = np.ones(num_unitaries) / num_unitaries
        
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply the channel to a density matrix."""
        result = np.zeros_like(rho, dtype=complex)
        
        for p, U in zip(self.probabilities, self.unitaries):
            result += p * (U @ rho @ U.conj().T)
            
        return result
    
    def kraus_operators(self) -> List[np.ndarray]:
        """Return Kraus operators."""
        return [np.sqrt(p) * U for p, U in zip(self.probabilities, self.unitaries)]


class WernerHolevoChannel(QuantumChannel):
    """
    Werner-Holevo channel, known to exhibit non-additivity for certain parameters.
    This is a d-dimensional generalization of the depolarizing channel.
    """
    
    def __init__(self, p: float, d: int = 3):
        super().__init__(d, d)
        self.p = p
        self.d = d
        self._generate_weyl_operators()
        
    def _generate_weyl_operators(self):
        """Generate generalized Pauli (Weyl) operators."""
        self.weyl_ops = []
        
        # Clock and shift matrices
        omega = np.exp(2j * np.pi / self.d)
        
        # Clock matrix
        Z = np.diag([omega**k for k in range(self.d)])
        
        # Shift matrix
        X = np.zeros((self.d, self.d), dtype=complex)
        for i in range(self.d):
            X[i, (i+1) % self.d] = 1
            
        # Generate all Weyl operators
        for a in range(self.d):
            for b in range(self.d):
                W = np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b)
                self.weyl_ops.append(W)
                
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply the Werner-Holevo channel."""
        result = self.p * rho
        
        # Add noise term
        noise_weight = (1 - self.p) / (self.d ** 2)
        for W in self.weyl_ops:
            result += noise_weight * (W @ rho @ W.conj().T)
            
        return result
    
    def kraus_operators(self) -> List[np.ndarray]:
        """Return Kraus operators."""
        kraus_ops = [np.sqrt(self.p) * np.eye(self.d)]
        
        noise_weight = np.sqrt((1 - self.p) / (self.d ** 2))
        for W in self.weyl_ops:
            kraus_ops.append(noise_weight * W)
            
        return kraus_ops


class AntisymmetricChannel(QuantumChannel):
    """
    Channel projecting onto antisymmetric subspace.
    Known to exhibit interesting additivity properties.
    """
    
    def __init__(self, d: int = 3):
        # For antisymmetric subspace of d ⊗ d
        self.d_single = d
        self.d_in = d * d
        self.d_out = d * (d - 1) // 2  # Dimension of antisymmetric subspace
        super().__init__(self.d_in, self.d_out)
        self._construct_projection()
        
    def _construct_projection(self):
        """Construct projection onto antisymmetric subspace."""
        # Create antisymmetric projector
        self.basis_vectors = []
        
        for i in range(self.d_single):
            for j in range(i + 1, self.d_single):
                # |ij⟩ - |ji⟩ (unnormalized)
                vec = np.zeros(self.d_in, dtype=complex)
                vec[i * self.d_single + j] = 1 / np.sqrt(2)
                vec[j * self.d_single + i] = -1 / np.sqrt(2)
                self.basis_vectors.append(vec)
        
        # Projection matrix from full space to antisymmetric subspace
        self.P = np.array(self.basis_vectors).T
        
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply the antisymmetric projection channel."""
        # Project down to antisymmetric subspace
        rho_proj = self.P.conj().T @ rho @ self.P
        return rho_proj
    
    def kraus_operators(self) -> List[np.ndarray]:
        """Return Kraus operators (just the projection)."""
        return [self.P.conj().T]


def test_known_counterexamples():
    """Test various known counterexamples to additivity."""
    print("=" * 60)
    print("TESTING KNOWN COUNTEREXAMPLES TO ADDITIVITY")
    print("=" * 60)
    
    results = []
    
    # Test 1: Hastings channel
    print("\n1. Hastings Channel (d=4):")
    ch_hastings = SpecificCounterexamples.hastings_channel(d=4)
    result_hastings = test_additivity(ch_hastings, ch_hastings, num_trials=500)
    results.append(result_hastings)
    print(f"   Violation: {result_hastings['violation']:.6f}")
    print(f"   Relative violation: {result_hastings['relative_violation']*100:.2f}%")
    
    # Test 2: Werner-Holevo channel with various parameters
    print("\n2. Werner-Holevo Channel:")
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        ch_wh = SpecificCounterexamples.werner_holevo_channel(p, d=3)
        result_wh = test_additivity(ch_wh, ch_wh, num_trials=200)
        results.append(result_wh)
        print(f"   p={p}: violation={result_wh['violation']:.6f}")
    
    # Test 3: Mixed dimensions
    print("\n3. Mixed Dimension Test:")
    ch1 = SpecificCounterexamples.hastings_channel(d=3)
    ch2 = SpecificCounterexamples.werner_holevo_channel(0.5, d=3)
    result_mixed = test_additivity(ch1, ch2, num_trials=300)
    results.append(result_mixed)
    print(f"   Hastings(d=3) ⊗ Werner-Holevo(p=0.5,d=3):")
    print(f"   Violation: {result_mixed['violation']:.6f}")
    
    # Test 4: Antisymmetric channel
    print("\n4. Antisymmetric Channel Test:")
    ch_anti = SpecificCounterexamples.antisymmetric_channel(d=3)
    # Note: This channel has different input/output dimensions
    # So we test it differently
    chi_anti = holevo_capacity(ch_anti, num_samples=500)
    print(f"   Holevo capacity: {chi_anti:.6f}")
    
    return results


def analyze_violation_landscape():
    """Analyze how violations depend on channel parameters."""
    print("\nAnalyzing violation landscape...")
    
    # Parameter ranges
    p_values = np.linspace(0.1, 0.9, 20)
    d_values = [2, 3, 4, 5]
    
    # Store results
    violation_map = {}
    
    # Test Werner-Holevo channels
    for d in d_values:
        violations = []
        
        for p in p_values:
            ch = SpecificCounterexamples.werner_holevo_channel(p, d)
            result = test_additivity(ch, ch, num_trials=100)
            violations.append(result['violation'])
            
        violation_map[d] = violations
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    for d, violations in violation_map.items():
        plt.plot(p_values, violations, 'o-', label=f'd={d}', markersize=6)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Werner-Holevo parameter p')
    plt.ylabel('Additivity violation')
    plt.title('Additivity Violations for Werner-Holevo Channels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return violation_map


def test_entanglement_breaking_threshold():
    """
    Test channels near the entanglement-breaking threshold.
    Entanglement-breaking channels are known to be additive.
    """
    print("\nTesting near entanglement-breaking threshold...")
    
    # For depolarizing channel, EB threshold is p = 1/(d+1)
    d = 2
    eb_threshold = 1 / (d + 1)  # = 1/3 for qubits
    
    # Test around the threshold
    p_values = np.linspace(eb_threshold - 0.1, eb_threshold + 0.1, 20)
    violations = []
    is_eb = []
    
    for p in p_values:
        from holevo_capacity import DepolarizingChannel
        ch = DepolarizingChannel(p, d)
        
        # Test additivity
        result = test_additivity(ch, ch, num_trials=100)
        violations.append(result['violation'])
        
        # Check if entanglement-breaking
        is_eb.append(p <= eb_threshold)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Color points based on EB property
    colors = ['red' if eb else 'blue' for eb in is_eb]
    plt.scatter(p_values, violations, c=colors, s=50, alpha=0.7)
    
    plt.axvline(x=eb_threshold, color='green', linestyle='--', 
                label=f'EB threshold (p={eb_threshold:.3f})')
    plt.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    
    plt.xlabel('Depolarizing parameter p')
    plt.ylabel('Additivity violation')
    plt.title('Additivity Near Entanglement-Breaking Threshold')
    plt.legend(['EB threshold', 'Perfect additivity', 'EB channels', 'Non-EB channels'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nEntanglement-breaking threshold: p = {eb_threshold:.3f}")
    print(f"Maximum violation for EB channels: {max([v for v, eb in zip(violations, is_eb) if eb]):.6f}")
    print(f"Maximum violation for non-EB channels: {max([v for v, eb in zip(violations, is_eb) if not eb]):.6f}")
    
    return p_values, violations, is_eb


def test_higher_order_additivity():
    """Test additivity for tensor products of more than two channels."""
    print("\nTesting higher-order additivity...")
    
    from holevo_capacity import DepolarizingChannel
    
    # Test with three channels
    p_values = [0.5, 0.7, 0.9]
    channels = [DepolarizingChannel(p, dim=2) for p in p_values]
    
    # Calculate individual capacities
    individual_capacities = [holevo_capacity(ch, num_samples=200) for ch in channels]
    sum_capacities = sum(individual_capacities)
    
    # Calculate pairwise tensor products
    ch_12 = tensor_product_channel(channels[0], channels[1])
    ch_23 = tensor_product_channel(channels[1], channels[2])
    ch_13 = tensor_product_channel(channels[0], channels[2])
    
    chi_12 = holevo_capacity(ch_12, num_samples=200)
    chi_23 = holevo_capacity(ch_23, num_samples=200)
    chi_13 = holevo_capacity(ch_13, num_samples=200)
    
    # Calculate triple tensor product
    ch_123_temp = tensor_product_channel(channels[0], channels[1])
    ch_123 = tensor_product_channel(ch_123_temp, channels[2])
    chi_123 = holevo_capacity(ch_123, num_samples=200)
    
    print(f"\nIndividual capacities: {individual_capacities}")
    print(f"Sum of individual capacities: {sum_capacities:.6f}")
    print(f"\nPairwise capacities:")
    print(f"  χ(N₁⊗N₂) = {chi_12:.6f}, violation = {chi_12 - individual_capacities[0] - individual_capacities[1]:.6f}")
    print(f"  χ(N₂⊗N₃) = {chi_23:.6f}, violation = {chi_23 - individual_capacities[1] - individual_capacities[2]:.6f}")
    print(f"  χ(N₁⊗N₃) = {chi_13:.6f}, violation = {chi_13 - individual_capacities[0] - individual_capacities[2]:.6f}")
    print(f"\nTriple tensor product:")
    print(f"  χ(N₁⊗N₂⊗N₃) = {chi_123:.6f}")
    print(f"  Violation from sum: {chi_123 - sum_capacities:.6f}")
    print(f"  Violation from pairwise: {chi_123 - chi_12 - individual_capacities[2]:.6f}")
    
    return {
        'individual': individual_capacities,
        'pairwise': {'12': chi_12, '23': chi_23, '13': chi_13},
        'triple': chi_123,
        'violations': {
            'from_sum': chi_123 - sum_capacities,
            'from_pairwise': chi_123 - chi_12 - individual_capacities[2]
        }
    }


if __name__ == "__main__":
    # Run all counterexample tests
    results = test_known_counterexamples()
    
    # Analyze violation landscape
    violation_landscape = analyze_violation_landscape()
    
    # Test EB threshold
    eb_results = test_entanglement_breaking_threshold()
    
    # Test higher-order additivity
    higher_order_results = test_higher_order_additivity()
    
    print("\n" + "=" * 60)
    print("COUNTEREXAMPLE TESTING COMPLETE")
    print("=" * 60) 