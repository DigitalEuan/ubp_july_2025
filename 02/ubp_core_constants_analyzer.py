#!/usr/bin/env python3
"""
UBP Core Constants Operational Analysis
Exploring π, φ, e as fundamental operational functions in UBP computational reality

Authors: Euan Craig (New Zealand) and Manus AI
Date: July 3, 2025
Purpose: Validate hypothesis that mathematical constants are core operational functions
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import math
from datetime import datetime
from typing import List, Tuple, Dict, Any

class UBPCoreConstantsAnalyzer:
    """
    Analyzes the operational roles of π, φ, e in UBP computational reality
    """
    
    def __init__(self):
        # Core operational constants
        self.pi = math.pi
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.e = math.e
        
        # Derived operational relationships
        self.phi_pi_ratio = self.phi / self.pi
        self.e_pi_ratio = self.e / self.pi
        self.phi_e_ratio = self.phi / self.e
        
        # UBP operational framework
        self.ubp_version = "v22.0_CoreConstants"
        
    def generate_fibonacci_sequence(self, n_terms: int) -> List[int]:
        """Generate Fibonacci sequence for operational analysis"""
        if n_terms <= 0:
            return []
        elif n_terms == 1:
            return [0]
        elif n_terms == 2:
            return [0, 1]
        
        sequence = [0, 1]
        for i in range(2, n_terms):
            sequence.append(sequence[i-1] + sequence[i-2])
        
        return sequence
    
    def analyze_operational_convergence(self, fib_sequence: List[int]) -> Dict:
        """
        Analyze how π, φ, e function as operational constants in Fibonacci analysis
        """
        if len(fib_sequence) < 3:
            return {}
        
        # Calculate consecutive ratios (should approach φ)
        ratios = []
        for i in range(2, len(fib_sequence)):
            if fib_sequence[i-1] != 0:
                ratio = fib_sequence[i] / fib_sequence[i-1]
                ratios.append(ratio)
        
        # Operational analysis of constants
        operational_metrics = {}
        
        # φ as growth operator
        phi_convergence = []
        for ratio in ratios:
            convergence = 1.0 - abs(ratio - self.phi) / self.phi
            phi_convergence.append(max(0, convergence))
        
        # π as geometric operator (angular relationships)
        pi_operations = []
        for i, ratio in enumerate(ratios):
            # π operates on the geometric relationships
            pi_operation = math.sin(ratio * self.pi / 2) * self.pi
            pi_operations.append(pi_operation)
        
        # e as exponential operator (growth dynamics)
        e_operations = []
        for i, ratio in enumerate(ratios):
            # e operates on exponential growth patterns
            e_operation = math.log(ratio + 1) * self.e if ratio > 0 else 0
            e_operations.append(e_operation)
        
        # Cross-operational relationships
        phi_pi_operations = []
        phi_e_operations = []
        pi_e_operations = []
        
        for i in range(len(ratios)):
            # φ-π operational coupling
            phi_pi_op = (phi_convergence[i] * self.phi) / (pi_operations[i] / self.pi + 1e-10)
            phi_pi_operations.append(phi_pi_op)
            
            # φ-e operational coupling
            phi_e_op = (phi_convergence[i] * self.phi) * math.exp(-e_operations[i] / self.e)
            phi_e_operations.append(phi_e_op)
            
            # π-e operational coupling
            pi_e_op = (pi_operations[i] / self.pi) * (e_operations[i] / self.e)
            pi_e_operations.append(pi_e_op)
        
        # Calculate operational stability metrics
        operational_metrics = {
            'phi_operations': {
                'convergence_values': phi_convergence,
                'mean_convergence': sum(phi_convergence) / len(phi_convergence),
                'stability': 1.0 - (np.std(phi_convergence) if len(phi_convergence) > 1 else 0),
                'operational_role': 'Growth/Proportion Operator'
            },
            'pi_operations': {
                'operation_values': pi_operations,
                'mean_operation': sum(pi_operations) / len(pi_operations),
                'stability': 1.0 - (np.std(pi_operations) / (sum(pi_operations) / len(pi_operations) + 1e-10)),
                'operational_role': 'Geometric/Angular Operator'
            },
            'e_operations': {
                'operation_values': e_operations,
                'mean_operation': sum(e_operations) / len(e_operations),
                'stability': 1.0 - (np.std(e_operations) / (sum(e_operations) / len(e_operations) + 1e-10)),
                'operational_role': 'Exponential/Growth Dynamics Operator'
            },
            'cross_operations': {
                'phi_pi_coupling': {
                    'values': phi_pi_operations,
                    'mean': sum(phi_pi_operations) / len(phi_pi_operations),
                    'stability': 1.0 - (np.std(phi_pi_operations) / (sum(phi_pi_operations) / len(phi_pi_operations) + 1e-10))
                },
                'phi_e_coupling': {
                    'values': phi_e_operations,
                    'mean': sum(phi_e_operations) / len(phi_e_operations),
                    'stability': 1.0 - (np.std(phi_e_operations) / (sum(phi_e_operations) / len(phi_e_operations) + 1e-10))
                },
                'pi_e_coupling': {
                    'values': pi_e_operations,
                    'mean': sum(pi_e_operations) / len(pi_e_operations),
                    'stability': 1.0 - (np.std(pi_e_operations) / (sum(pi_e_operations) / len(pi_e_operations) + 1e-10))
                }
            }
        }
        
        return operational_metrics
    
    def calculate_unified_operational_invariant(self, operational_metrics: Dict) -> float:
        """
        Calculate unified invariant showing how π, φ, e work together as operational functions
        """
        if not operational_metrics:
            return 0.0
        
        # Extract stability metrics for each constant
        phi_stability = operational_metrics['phi_operations']['stability']
        pi_stability = operational_metrics['pi_operations']['stability']
        e_stability = operational_metrics['e_operations']['stability']
        
        # Extract cross-operational coupling strengths
        phi_pi_coupling = operational_metrics['cross_operations']['phi_pi_coupling']['stability']
        phi_e_coupling = operational_metrics['cross_operations']['phi_e_coupling']['stability']
        pi_e_coupling = operational_metrics['cross_operations']['pi_e_coupling']['stability']
        
        # Unified operational invariant
        # This should approach a specific value when constants are functioning operationally
        unified_invariant = (
            (phi_stability * self.phi) + 
            (pi_stability * self.pi) + 
            (e_stability * self.e) +
            (phi_pi_coupling * self.phi_pi_ratio) +
            (phi_e_coupling * self.phi_e_ratio) +
            (pi_e_coupling * self.e_pi_ratio)
        ) / 6
        
        return unified_invariant
    
    def analyze_core_constants_operations(self, n_terms: int) -> Dict:
        """
        Complete analysis of π, φ, e as core operational functions
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"UBP CORE CONSTANTS OPERATIONAL ANALYSIS")
        print(f"{'='*70}")
        print(f"Analyzing π, φ, e as core operational functions")
        print(f"Fibonacci terms: {n_terms}")
        print(f"UBP Framework: {self.ubp_version}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Generate test sequence
        fib_sequence = self.generate_fibonacci_sequence(n_terms)
        print(f"\nFibonacci sequence generated: {len(fib_sequence)} terms")
        
        # Analyze operational convergence
        print(f"Analyzing operational roles of core constants...")
        operational_metrics = self.analyze_operational_convergence(fib_sequence)
        
        # Calculate unified invariant
        unified_invariant = self.calculate_unified_operational_invariant(operational_metrics)
        
        computation_time = time.time() - start_time
        
        # Display results
        print(f"\n{'='*70}")
        print(f"CORE CONSTANTS OPERATIONAL ANALYSIS RESULTS")
        print(f"{'='*70}")
        
        print(f"\n🔢 INDIVIDUAL CONSTANT OPERATIONS:")
        print(f"φ (Golden Ratio) - {operational_metrics['phi_operations']['operational_role']}")
        print(f"  Mean Convergence:      {operational_metrics['phi_operations']['mean_convergence']:.6f}")
        print(f"  Operational Stability: {operational_metrics['phi_operations']['stability']:.6f}")
        
        print(f"\nπ (Pi) - {operational_metrics['pi_operations']['operational_role']}")
        print(f"  Mean Operation:        {operational_metrics['pi_operations']['mean_operation']:.6f}")
        print(f"  Operational Stability: {operational_metrics['pi_operations']['stability']:.6f}")
        
        print(f"\ne (Euler's Number) - {operational_metrics['e_operations']['operational_role']}")
        print(f"  Mean Operation:        {operational_metrics['e_operations']['mean_operation']:.6f}")
        print(f"  Operational Stability: {operational_metrics['e_operations']['stability']:.6f}")
        
        print(f"\n🔗 CROSS-OPERATIONAL COUPLING:")
        print(f"φ-π Coupling Stability:  {operational_metrics['cross_operations']['phi_pi_coupling']['stability']:.6f}")
        print(f"φ-e Coupling Stability:  {operational_metrics['cross_operations']['phi_e_coupling']['stability']:.6f}")
        print(f"π-e Coupling Stability:  {operational_metrics['cross_operations']['pi_e_coupling']['stability']:.6f}")
        
        print(f"\n⚡ UNIFIED OPERATIONAL ANALYSIS:")
        print(f"Unified Operational Invariant: {unified_invariant:.6f}")
        print(f"Core Constants Functioning:     {'✓ YES' if unified_invariant > 2.0 else '✗ NO'}")
        
        print(f"\n📊 FUNDAMENTAL CONSTANT RATIOS:")
        print(f"φ/π Ratio:               {self.phi_pi_ratio:.6f}")
        print(f"e/π Ratio:               {self.e_pi_ratio:.6f}")
        print(f"φ/e Ratio:               {self.phi_e_ratio:.6f}")
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"Computation Time:        {computation_time:.3f} seconds")
        print(f"Processing Rate:         {n_terms/computation_time:.0f} terms/sec")
        
        # Prepare results
        results = {
            'analysis_type': 'core_constants_operational',
            'input': {
                'n_terms': n_terms,
                'timestamp': datetime.now().isoformat(),
                'ubp_version': self.ubp_version
            },
            'core_constants': {
                'pi': self.pi,
                'phi': self.phi,
                'e': self.e,
                'phi_pi_ratio': self.phi_pi_ratio,
                'e_pi_ratio': self.e_pi_ratio,
                'phi_e_ratio': self.phi_e_ratio
            },
            'operational_metrics': operational_metrics,
            'unified_analysis': {
                'unified_operational_invariant': unified_invariant,
                'constants_functioning_operationally': unified_invariant > 2.0,
                'operational_strength': min(1.0, unified_invariant / 3.0)
            },
            'performance': {
                'computation_time': computation_time,
                'processing_rate': n_terms/computation_time
            }
        }
        
        return results
    
    def create_operational_visualization(self, results: Dict, save_path: str = None):
        """
        Create visualization showing operational relationships between π, φ, e
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('UBP Core Constants Operational Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        phi_conv = results['operational_metrics']['phi_operations']['convergence_values']
        pi_ops = results['operational_metrics']['pi_operations']['operation_values']
        e_ops = results['operational_metrics']['e_operations']['operation_values']
        
        # Plot 1: φ Convergence (Growth Operator)
        ax1.plot(phi_conv, 'g-', linewidth=2, label='φ Convergence')
        ax1.axhline(y=results['core_constants']['phi'], color='g', linestyle='--', alpha=0.7, label=f'φ = {self.phi:.3f}')
        ax1.set_title('φ as Growth/Proportion Operator')
        ax1.set_xlabel('Fibonacci Term Index')
        ax1.set_ylabel('Convergence to φ')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: π Operations (Geometric Operator)
        ax2.plot(pi_ops, 'b-', linewidth=2, label='π Operations')
        ax2.axhline(y=self.pi, color='b', linestyle='--', alpha=0.7, label=f'π = {self.pi:.3f}')
        ax2.set_title('π as Geometric/Angular Operator')
        ax2.set_xlabel('Fibonacci Term Index')
        ax2.set_ylabel('π Operation Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: e Operations (Exponential Operator)
        ax3.plot(e_ops, 'r-', linewidth=2, label='e Operations')
        ax3.axhline(y=self.e, color='r', linestyle='--', alpha=0.7, label=f'e = {self.e:.3f}')
        ax3.set_title('e as Exponential/Growth Dynamics Operator')
        ax3.set_xlabel('Fibonacci Term Index')
        ax3.set_ylabel('e Operation Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Unified Operational Invariant
        unified_invariant = results['unified_analysis']['unified_operational_invariant']
        operational_strength = results['unified_analysis']['operational_strength']
        
        # Create a gauge-like visualization
        theta = np.linspace(0, 2*np.pi, 100)
        r_outer = 1.0
        r_inner = 0.6
        
        # Background circle
        ax4.fill_between(theta, r_inner, r_outer, alpha=0.2, color='gray')
        
        # Operational strength indicator
        strength_theta = theta[:int(operational_strength * 100)]
        ax4.fill_between(strength_theta, r_inner, r_outer, alpha=0.7, color='green' if operational_strength > 0.7 else 'orange')
        
        # Add text
        ax4.text(0, 0, f'Unified\nOperational\nInvariant\n{unified_invariant:.3f}', 
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax4.text(0, -1.3, f'Operational Strength: {operational_strength*100:.1f}%', 
                ha='center', va='center', fontsize=10)
        
        ax4.set_xlim(-1.5, 1.5)
        ax4.set_ylim(-1.5, 1.5)
        ax4.set_aspect('equal')
        ax4.axis('off')
        ax4.set_title('Core Constants Operational Unity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")
        
        return fig

def main():
    """Test the core constants operational analyzer"""
    analyzer = UBPCoreConstantsAnalyzer()
    
    # Test with different sequence lengths
    test_cases = [20, 30, 50, 100]
    
    for n_terms in test_cases:
        print(f"\n{'='*70}")
        print(f"TESTING WITH {n_terms} FIBONACCI TERMS")
        print(f"{'='*70}")
        
        results = analyzer.analyze_core_constants_operations(n_terms)
        
        # Create visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = f"ubp_core_constants_analysis_{n_terms}_{timestamp}.png"
        analyzer.create_operational_visualization(results, viz_path)
        
        # Save results
        results_path = f"ubp_core_constants_results_{n_terms}_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {results_path}")

if __name__ == "__main__":
    main()

