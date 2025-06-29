import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
from itertools import product


class QuantumChannel:
    """Base class for quantum channels."""
    
    def __init__(self, dim_in: int, dim_out: int):
        self.dim_in = dim_in
        self.dim_out = dim_out
    
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply the channel to a density matrix."""
        raise NotImplementedError
    
    def kraus_operators(self) -> List[np.ndarray]:
        """Return the Kraus operators for the channel."""
        raise NotImplementedError


class DepolarizingChannel(QuantumChannel):
    """Depolarizing channel: ρ → p*ρ + (1-p)*I/d"""
    
    def __init__(self, p: float, dim: int = 2):
        super().__init__(dim, dim)
        self.p = p
        self.dim = dim
        
    def apply(self, rho: np.ndarray) -> np.ndarray:
        identity = np.eye(self.dim) / self.dim
        return self.p * rho + (1 - self.p) * identity
    
    def kraus_operators(self) -> List[np.ndarray]:
        # Kraus operators for depolarizing channel
        kraus_ops = []
        
        # Identity component
        kraus_ops.append(np.sqrt(self.p) * np.eye(self.dim))
        
        # Pauli components (for qubit case)
        if self.dim == 2:
            pauli_weight = np.sqrt((1 - self.p) / 3)
            # Pauli X
            kraus_ops.append(pauli_weight * np.array([[0, 1], [1, 0]]))
            # Pauli Y
            kraus_ops.append(pauli_weight * np.array([[0, -1j], [1j, 0]]))
            # Pauli Z
            kraus_ops.append(pauli_weight * np.array([[1, 0], [0, -1]]))
        
        return kraus_ops


class AmplitudeDampingChannel(QuantumChannel):
    """Amplitude damping channel modeling energy dissipation."""
    
    def __init__(self, gamma: float):
        super().__init__(2, 2)
        self.gamma = gamma
        
    def apply(self, rho: np.ndarray) -> np.ndarray:
        K0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]])
        K1 = np.array([[0, np.sqrt(self.gamma)], [0, 0]])
        
        return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
    
    def kraus_operators(self) -> List[np.ndarray]:
        K0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]])
        K1 = np.array([[0, np.sqrt(self.gamma)], [0, 0]])
        return [K0, K1]


class ErasureChannel(QuantumChannel):
    """Erasure channel that replaces state with erasure flag with probability p."""
    
    def __init__(self, p: float, dim: int = 2):
        # Output dimension is dim + 1 (original space + erasure flag)
        super().__init__(dim, dim + 1)
        self.p = p
        self.dim = dim
        
    def apply(self, rho: np.ndarray) -> np.ndarray:
        # Extend rho to larger space
        rho_extended = np.zeros((self.dim_out, self.dim_out), dtype=complex)
        rho_extended[:self.dim, :self.dim] = (1 - self.p) * rho
        # Add erasure component
        rho_extended[self.dim, self.dim] = self.p
        return rho_extended
    
    def kraus_operators(self) -> List[np.ndarray]:
        kraus_ops = []
        
        # Non-erasure operator
        K0 = np.zeros((self.dim_out, self.dim_in), dtype=complex)
        K0[:self.dim_in, :self.dim_in] = np.sqrt(1 - self.p) * np.eye(self.dim_in)
        kraus_ops.append(K0)
        
        # Erasure operator
        K1 = np.zeros((self.dim_out, self.dim_in), dtype=complex)
        K1[self.dim_in, :] = np.sqrt(self.p) * np.ones(self.dim_in)
        kraus_ops.append(K1)
        
        return kraus_ops


def tensor_product_channel(channel1: QuantumChannel, channel2: QuantumChannel) -> QuantumChannel:
    """Create tensor product of two quantum channels."""
    
    class TensorProductChannel(QuantumChannel):
        def __init__(self, ch1: QuantumChannel, ch2: QuantumChannel):
            self.ch1 = ch1
            self.ch2 = ch2
            super().__init__(ch1.dim_in * ch2.dim_in, ch1.dim_out * ch2.dim_out)
            
        def apply(self, rho: np.ndarray) -> np.ndarray:
            dim1_in, dim2_in = self.ch1.dim_in, self.ch2.dim_in
            dim1_out, dim2_out = self.ch1.dim_out, self.ch2.dim_out
            
            # Reshape and apply channels
            result = np.zeros((dim1_out * dim2_out, dim1_out * dim2_out), dtype=complex)
            
            # Apply using Kraus operators
            kraus1 = self.ch1.kraus_operators()
            kraus2 = self.ch2.kraus_operators()
            
            for K1 in kraus1:
                for K2 in kraus2:
                    K_tensor = np.kron(K1, K2)
                    result += K_tensor @ rho @ K_tensor.conj().T
                    
            return result
        
        def kraus_operators(self) -> List[np.ndarray]:
            kraus1 = self.ch1.kraus_operators()
            kraus2 = self.ch2.kraus_operators()
            return [np.kron(K1, K2) for K1 in kraus1 for K2 in kraus2]
    
    return TensorProductChannel(channel1, channel2)


def von_neumann_entropy(rho: np.ndarray, epsilon: float = 1e-12) -> float:
    """Calculate von Neumann entropy of a density matrix."""
    eigenvalues = la.eigvalsh(rho)
    # Filter out numerical zeros
    eigenvalues = eigenvalues[eigenvalues > epsilon]
    return -np.sum(eigenvalues * np.log2(eigenvalues))


def mutual_information(rho_AB: np.ndarray, dim_A: int, dim_B: int) -> float:
    """Calculate mutual information I(A:B) = S(A) + S(B) - S(AB)."""
    # Calculate reduced density matrices
    rho_A = partial_trace(rho_AB, dim_A, dim_B, 'B')
    rho_B = partial_trace(rho_AB, dim_A, dim_B, 'A')
    
    # Calculate entropies
    S_A = von_neumann_entropy(rho_A)
    S_B = von_neumann_entropy(rho_B)
    S_AB = von_neumann_entropy(rho_AB)
    
    return S_A + S_B - S_AB


def partial_trace(rho: np.ndarray, dim_A: int, dim_B: int, system: str) -> np.ndarray:
    """Compute partial trace over system A or B."""
    if system == 'A':
        # Trace out system A
        result = np.zeros((dim_B, dim_B), dtype=complex)
        for i in range(dim_A):
            for j in range(dim_B):
                for k in range(dim_B):
                    result[j, k] += rho[i*dim_B + j, i*dim_B + k]
    else:  # system == 'B'
        # Trace out system B
        result = np.zeros((dim_A, dim_A), dtype=complex)
        for i in range(dim_A):
            for j in range(dim_A):
                for k in range(dim_B):
                    result[i, j] += rho[i*dim_B + k, j*dim_B + k]
    
    return result


def holevo_capacity(channel: QuantumChannel, num_samples: int = 1000, 
                   num_states: int = None) -> float:
    """
    Calculate Holevo capacity of a quantum channel using optimization.
    
    χ(N) = max_{p_i, ρ_i} [S(∑ p_i N(ρ_i)) - ∑ p_i S(N(ρ_i))]
    """
    dim = channel.dim_in
    
    if num_states is None:
        num_states = min(dim**2, 10)  # Limit for computational efficiency
    
    best_capacity = 0
    
    for _ in range(num_samples):
        probabilities = np.random.dirichlet(np.ones(num_states))
        states = []
        
        for _ in range(num_states):
            psi = np.random.randn(dim) + 1j * np.random.randn(dim)
            psi = psi / np.linalg.norm(psi)
            rho = np.outer(psi, psi.conj())
            states.append(rho)
        
        output_states = [channel.apply(rho) for rho in states]
        
        #
        avg_output = sum(p * rho for p, rho in zip(probabilities, output_states))
        
        chi = von_neumann_entropy(avg_output)
        chi -= sum(p * von_neumann_entropy(rho) for p, rho in zip(probabilities, output_states))
        
        best_capacity = max(best_capacity, chi)
    
    return best_capacity


def test_additivity(channel1: QuantumChannel, channel2: QuantumChannel, 
                   num_trials: int = 100) -> dict:
    """
    Test additivity of Holevo capacity for two channels.
    
    Additivity holds if: χ(N1 ⊗ N2) = χ(N1) + χ(N2)
    """
    print(f"Testing additivity for {channel1.__class__.__name__} ⊗ {channel2.__class__.__name__}")
    
    # 
    chi1 = holevo_capacity(channel1, num_samples=num_trials)
    chi2 = holevo_capacity(channel2, num_samples=num_trials)
    

    tensor_channel = tensor_product_channel(channel1, channel2)
    chi_tensor = holevo_capacity(tensor_channel, num_samples=num_trials)
    
    additive_capacity = chi1 + chi2
    violation = chi_tensor - additive_capacity
    
    results = {
        'chi1': chi1,
        'chi2': chi2,
        'chi_tensor': chi_tensor,
        'additive_capacity': additive_capacity,
        'violation': violation,
        'relative_violation': violation / additive_capacity if additive_capacity > 0 else 0,
        'is_additive': abs(violation) < 0.01  # Tolerance for numerical errors
    }
    
    return results


def generate_random_channel(dim: int, channel_type: str = 'random') -> QuantumChannel:
    """Generate a random quantum channel of specified type."""
    if channel_type == 'depolarizing':
        p = np.random.uniform(0, 1)
        return DepolarizingChannel(p, dim)
    elif channel_type == 'amplitude_damping' and dim == 2:
        gamma = np.random.uniform(0, 1)
        return AmplitudeDampingChannel(gamma)
    elif channel_type == 'erasure':
        p = np.random.uniform(0, 1)
        return ErasureChannel(p, dim)
    else:
        class RandomChannel(QuantumChannel):
            def __init__(self, kraus_ops):
                self._kraus_ops = kraus_ops
                super().__init__(kraus_ops[0].shape[1], kraus_ops[0].shape[0])
                
            def apply(self, rho: np.ndarray) -> np.ndarray:
                result = np.zeros_like(rho, dtype=complex)
                for K in self._kraus_ops:
                    result += K @ rho @ K.conj().T
                return result
            
            def kraus_operators(self) -> List[np.ndarray]:
                return self._kraus_ops
        
       
        num_kraus = np.random.randint(1, dim**2 + 1)
        kraus_ops = []
        
        for _ in range(num_kraus):
            K = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
            kraus_ops.append(K)
        
        completeness = sum(K.conj().T @ K for K in kraus_ops)
        factor = la.sqrtm(la.inv(completeness))
        kraus_ops = [K @ factor for K in kraus_ops]
        
        return RandomChannel(kraus_ops) 
