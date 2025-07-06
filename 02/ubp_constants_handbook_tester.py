#!/usr/bin/env python3
"""
UBP Constants Handbook Tester
Systematic testing of mathematical constants as operational functions in computational reality

Authors: Euan Craig (New Zealand) and Manus AI
Date: July 3, 2025
Purpose: Create comprehensive "Handbook of Computational Reality"
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import math
from datetime import datetime
from typing import List, Tuple, Dict, Any

class UBPConstantsHandbookTester:
    """
    Systematic tester for mathematical constants as operational functions
    """
    
    def __init__(self):
        # Core validated constants
        self.pi = math.pi
        self.phi = (1 + math.sqrt(5)) / 2
        self.e = math.e
        
        # Mathematical constants to test
        self.constants_catalog = {
            # Category A: Fundamental Operators (High Priority)
            'sqrt_2': {
                'value': math.sqrt(2),
                'name': 'Square Root of 2',
                'symbol': '√2',
                'category': 'A',
                'expected_role': 'Dimensional Scaling Operator',
                'description': 'Pythagoras constant, diagonal ratio'
            },
            'sqrt_3': {
                'value': math.sqrt(3),
                'name': 'Square Root of 3', 
                'symbol': '√3',
                'category': 'A',
                'expected_role': 'Triangular/Hexagonal Geometry Operator',
                'description': 'Theodorus constant'
            },
            'sqrt_5': {
                'value': math.sqrt(5),
                'name': 'Square Root of 5',
                'symbol': '√5', 
                'category': 'A',
                'expected_role': 'Golden Ratio Foundation Operator',
                'description': 'Related to golden ratio (φ = (1+√5)/2)'
            },
            'cbrt_2': {
                'value': 2**(1/3),
                'name': 'Cube Root of 2',
                'symbol': '∛2',
                'category': 'A', 
                'expected_role': 'Three-Dimensional Scaling Operator',
                'description': 'Cubic scaling constant'
            },
            'tau': {
                'value': 2 * math.pi,
                'name': 'Tau',
                'symbol': 'τ',
                'category': 'A',
                'expected_role': 'Full-Circle Geometric Operator', 
                'description': 'Circle constant (2π)'
            },
            'euler_gamma': {
                'value': 0.5772156649015329,  # Euler-Mascheroni constant
                'name': 'Euler-Mascheroni Constant',
                'symbol': 'γ',
                'category': 'A',
                'expected_role': 'Harmonic/Series Convergence Operator',
                'description': 'Limiting difference between harmonic series and natural log'
            },
            'ln_2': {
                'value': math.log(2),
                'name': 'Natural Logarithm of 2',
                'symbol': 'ln(2)',
                'category': 'A',
                'expected_role': 'Binary/Logarithmic Scaling Operator',
                'description': 'Logarithmic scaling constant'
            },
            
            # Category B: Special Functions (Medium Priority)
            'apery': {
                'value': 1.2020569031595942,  # ζ(3)
                'name': "Apéry's Constant",
                'symbol': 'ζ(3)',
                'category': 'B',
                'expected_role': 'Higher-Order Series Operator',
                'description': 'Riemann zeta function at 3'
            },
            'catalan': {
                'value': 0.9159655941772190,  # Catalan's constant
                'name': "Catalan's Constant", 
                'symbol': 'G',
                'category': 'B',
                'expected_role': 'Dirichlet Series Operator',
                'description': 'Dirichlet beta function'
            },
            'omega': {
                'value': 0.5671432904097838,  # Omega constant
                'name': 'Omega Constant',
                'symbol': 'Ω',
                'category': 'B',
                'expected_role': 'Self-Referential/Recursive Operator',
                'description': 'Lambert W function constant'
            },
            
            # Category C: Geometric Ratios (Medium Priority)
            'silver_ratio': {
                'value': 1 + math.sqrt(2),
                'name': 'Silver Ratio',
                'symbol': 'δ_S',
                'category': 'C',
                'expected_role': 'Secondary Proportion Operator',
                'description': '1 + √2, octagonal geometry'
            },
            'plastic_number': {
                'value': 1.3247179572447460,  # Real root of x³ = x + 1
                'name': 'Plastic Number',
                'symbol': 'ρ',
                'category': 'C', 
                'expected_role': 'Cubic Proportion Operator',
                'description': 'Real root of x³ = x + 1'
            },
            'supergolden': {
                'value': 1.4655712318767680,  # Tribonacci constant
                'name': 'Supergolden Ratio',
                'symbol': 'ψ',
                'category': 'C',
                'expected_role': 'Tribonacci Sequence Operator',
                'description': 'Tribonacci constant'
            }
        }
        
        # Leech Lattice parameters
        self.leech_dimension = 24
        self.kissing_number = 196560
        self.error_correction_levels = [3, 6, 9]
        
        # UBP framework
        self.ubp_version = "v22.0_ConstantsHandbook"
        
    def generate_test_sequence(self, n_terms: int, sequence_type: str = 'fibonacci') -> List[int]:
        """Generate test sequence for constant analysis"""
        if sequence_type == 'fibonacci':
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
        
        elif sequence_type == 'natural':
            return list(range(1, n_terms + 1))
        
        elif sequence_type == 'squares':
            return [i*i for i in range(1, n_terms + 1)]
        
        else:
            return list(range(1, n_terms + 1))
    
    def test_constant_as_operator(self, constant_key: str, test_sequence: List[int]) -> Dict:
        """
        Test a specific mathematical constant as an operational function
        """
        constant_info = self.constants_catalog[constant_key]
        constant_value = constant_info['value']
        
        print(f"\n{'='*60}")
        print(f"TESTING: {constant_info['name']} ({constant_info['symbol']})")
        print(f"Value: {constant_value:.10f}")
        print(f"Expected Role: {constant_info['expected_role']}")
        print(f"{'='*60}")
        
        # Encode sequence using the constant as operator
        offbits = self.encode_constant_offbits(test_sequence, constant_value, constant_key)
        
        # Calculate 24D Leech Lattice positions
        positions = self.calculate_constant_leech_positions(offbits, constant_value)
        
        # Analyze error correction capabilities
        error_correction = self.analyze_constant_error_correction(offbits, positions, constant_value)
        
        # Calculate operational metrics
        operational_metrics = self.calculate_constant_operational_metrics(
            offbits, positions, constant_value, constant_key
        )
        
        # Calculate unified invariant contribution
        unified_contribution = self.calculate_constant_unified_contribution(
            operational_metrics, constant_value
        )
        
        # Determine operational classification
        operational_class = self.classify_constant_operation(operational_metrics, unified_contribution)
        
        results = {
            'constant_info': constant_info,
            'test_parameters': {
                'sequence_length': len(test_sequence),
                'sequence_type': 'fibonacci',
                'leech_dimension': self.leech_dimension
            },
            'offbits_created': len(offbits),
            'error_correction': error_correction,
            'operational_metrics': operational_metrics,
            'unified_contribution': unified_contribution,
            'operational_classification': operational_class,
            'comparison_to_core': {
                'vs_pi': unified_contribution / (self.pi / 3),
                'vs_phi': unified_contribution / (self.phi / 3), 
                'vs_e': unified_contribution / (self.e / 3)
            }
        }
        
        # Display results
        self.display_constant_results(results)
        
        return results
    
    def encode_constant_offbits(self, sequence: List[int], constant: float, constant_key: str) -> List[Dict]:
        """
        Encode sequence as OffBits using the constant as operational function
        """
        offbits = []
        
        for i, num in enumerate(sequence):
            # Convert to 24-bit representation
            binary_rep = format(num % (2**24), '024b')
            
            # Split into 4 layers (6 bits each)
            layers = [
                binary_rep[0:6],    # Reality layer
                binary_rep[6:12],   # Information layer
                binary_rep[12:18],  # Activation layer
                binary_rep[18:24]   # Unactivated layer
            ]
            
            # Apply constant as operational function to each layer
            layer_operations = []
            for j, layer in enumerate(layers):
                layer_val = int(layer, 2)
                
                # Different operational modes based on constant type
                if constant_key in ['sqrt_2', 'sqrt_3', 'sqrt_5']:
                    # Square root constants: scaling operations
                    operation = (layer_val * constant) / 64
                elif constant_key in ['cbrt_2']:
                    # Cube root: 3D scaling
                    operation = (layer_val * (constant ** 3)) / 64
                elif constant_key in ['tau']:
                    # Tau: full circle operations
                    operation = (layer_val * math.sin(constant * j / 4)) / 64
                elif constant_key in ['euler_gamma', 'ln_2']:
                    # Logarithmic constants: convergence operations
                    operation = (layer_val * math.log(constant + 1)) / 64
                elif constant_key in ['apery', 'catalan']:
                    # Special function constants: series operations
                    operation = (layer_val * constant * (j + 1)) / 64
                else:
                    # Default: direct scaling
                    operation = (layer_val * constant) / 64
                
                layer_operations.append(operation)
            
            offbit = {
                'index': i,
                'sequence_value': num,
                'binary_representation': binary_rep,
                'layers': layers,
                'layer_operations': layer_operations,
                'total_operation': sum(layer_operations),
                'constant_applied': constant
            }
            
            offbits.append(offbit)
        
        return offbits
    
    def calculate_constant_leech_positions(self, offbits: List[Dict], constant: float) -> List[Tuple]:
        """
        Calculate 24D Leech Lattice positions using constant as geometric operator
        """
        positions = []
        
        for i, offbit in enumerate(offbits):
            coordinates = []
            
            # Generate 24 coordinates using constant as operator
            for dim in range(self.leech_dimension):
                layer_idx = dim % 4
                operation_val = offbit['layer_operations'][layer_idx]
                
                # Apply constant-specific geometric transformations
                if dim < 6:  # First 6 dimensions: direct constant operation
                    coord = operation_val * math.cos(dim * constant)
                elif dim < 12:  # Next 6: constant with π interaction
                    coord = operation_val * math.sin(dim * constant / self.pi)
                elif dim < 18:  # Next 6: constant with φ interaction  
                    coord = operation_val * math.cos(dim * constant / self.phi)
                else:  # Last 6: constant with e interaction
                    coord = operation_val * math.sin(dim * constant / self.e)
                
                # Apply Fibonacci index modulation
                coord *= (1 + i * constant / 1000)
                
                coordinates.append(coord)
            
            positions.append(tuple(coordinates))
        
        return positions
    
    def analyze_constant_error_correction(self, offbits: List[Dict], positions: List[Tuple], constant: float) -> Dict:
        """
        Analyze error correction capabilities using the constant
        """
        error_correction = {
            'level_3': {'corrections': 0, 'strength': 0.0},
            'level_6': {'corrections': 0, 'strength': 0.0}, 
            'level_9': {'corrections': 0, 'strength': 0.0},
            'overall_rate': 0.0
        }
        
        total_correctable = 0
        
        for i, (offbit, pos) in enumerate(zip(offbits, positions)):
            for level in self.error_correction_levels:
                # Extract coordinates for this level
                level_coords = pos[:level*3]  # 3, 6, or 9 dimensions
                level_distance = math.sqrt(sum(coord**2 for coord in level_coords))
                
                # Calculate correction strength using constant
                if level == 3:
                    correction_strength = constant / (level_distance + 1e-10)
                elif level == 6:
                    correction_strength = (constant * self.pi) / (level_distance + 1e-10)
                else:  # level == 9
                    correction_strength = (constant * self.e) / (level_distance + 1e-10)
                
                error_correction[f'level_{level}']['strength'] += correction_strength
                
                if correction_strength > 1.0:
                    error_correction[f'level_{level}']['corrections'] += 1
        
        # Calculate averages and overall rate
        for level in self.error_correction_levels:
            error_correction[f'level_{level}']['strength'] /= len(offbits)
            total_correctable += error_correction[f'level_{level}']['corrections']
        
        error_correction['overall_rate'] = total_correctable / (len(offbits) * 3)
        
        return error_correction
    
    def calculate_constant_operational_metrics(self, offbits: List[Dict], positions: List[Tuple], 
                                             constant: float, constant_key: str) -> Dict:
        """
        Calculate operational effectiveness metrics for the constant
        """
        if not offbits:
            return {}
        
        # Calculate operational stability
        operations = [offbit['total_operation'] for offbit in offbits]
        mean_operation = sum(operations) / len(operations)
        std_operation = math.sqrt(sum((op - mean_operation)**2 for op in operations) / len(operations))
        stability = 1.0 - (std_operation / (mean_operation + 1e-10))
        
        # Calculate geometric coherence
        if len(positions) >= 3:
            # Calculate distances between consecutive positions
            distances = []
            for i in range(1, len(positions)):
                dist = math.sqrt(sum((positions[i][j] - positions[i-1][j])**2 
                                   for j in range(self.leech_dimension)))
                distances.append(dist)
            
            mean_distance = sum(distances) / len(distances)
            distance_stability = 1.0 - (np.std(distances) / (mean_distance + 1e-10))
        else:
            distance_stability = 0.0
        
        # Calculate constant-specific metrics
        constant_resonance = self.calculate_constant_resonance(constant, operations)
        
        # Cross-constant coupling
        pi_coupling = abs(math.sin(constant * self.pi))
        phi_coupling = abs(math.cos(constant * self.phi))
        e_coupling = abs(math.sin(constant * self.e))
        
        metrics = {
            'operational_stability': stability,
            'geometric_coherence': distance_stability,
            'mean_operation_strength': mean_operation,
            'constant_resonance': constant_resonance,
            'cross_coupling': {
                'pi_coupling': pi_coupling,
                'phi_coupling': phi_coupling,
                'e_coupling': e_coupling,
                'total_coupling': pi_coupling + phi_coupling + e_coupling
            },
            'leech_integration': {
                'kissing_utilization': min(1.0, len(offbits) / self.kissing_number),
                'dimensional_coverage': 1.0  # All 24 dimensions used
            }
        }
        
        return metrics
    
    def calculate_constant_resonance(self, constant: float, operations: List[float]) -> float:
        """
        Calculate resonance frequency for the constant
        """
        if len(operations) < 2:
            return 0.0
        
        # Calculate how operations relate to the constant value
        resonance_sum = 0.0
        for i in range(1, len(operations)):
            ratio = operations[i] / (operations[i-1] + 1e-10)
            resonance = abs(math.sin(ratio * constant * self.pi))
            resonance_sum += resonance
        
        return resonance_sum / (len(operations) - 1)
    
    def calculate_constant_unified_contribution(self, metrics: Dict, constant: float) -> float:
        """
        Calculate unified invariant contribution of the constant
        """
        if not metrics:
            return 0.0
        
        # Weight different aspects of operational behavior
        stability_weight = metrics['operational_stability'] * 0.3
        coherence_weight = metrics['geometric_coherence'] * 0.2
        resonance_weight = metrics['constant_resonance'] * 0.2
        coupling_weight = metrics['cross_coupling']['total_coupling'] * 0.2
        integration_weight = metrics['leech_integration']['kissing_utilization'] * 0.1
        
        # Scale by constant value (normalized)
        constant_scale = min(10.0, constant) / 10.0
        
        unified_contribution = (
            stability_weight + coherence_weight + resonance_weight + 
            coupling_weight + integration_weight
        ) * constant_scale
        
        return unified_contribution
    
    def classify_constant_operation(self, metrics: Dict, unified_contribution: float) -> Dict:
        """
        Classify the operational behavior of the constant
        """
        # Determine operational strength
        if unified_contribution > 0.8:
            strength = "STRONG"
        elif unified_contribution > 0.5:
            strength = "MODERATE"
        elif unified_contribution > 0.2:
            strength = "WEAK"
        else:
            strength = "MINIMAL"
        
        # Determine operational type based on metrics
        if metrics['cross_coupling']['total_coupling'] > 2.0:
            op_type = "CORE_OPERATOR"
        elif metrics['geometric_coherence'] > 0.7:
            op_type = "GEOMETRIC_OPERATOR"
        elif metrics['constant_resonance'] > 0.5:
            op_type = "RESONANCE_OPERATOR"
        elif metrics['operational_stability'] > 0.7:
            op_type = "STABILITY_OPERATOR"
        else:
            op_type = "AUXILIARY_FUNCTION"
        
        # Determine computational role
        if unified_contribution > 0.6 and metrics['cross_coupling']['total_coupling'] > 1.5:
            role = "FUNDAMENTAL_COMPUTATIONAL_CONSTANT"
        elif unified_contribution > 0.4:
            role = "SPECIALIZED_COMPUTATIONAL_FUNCTION"
        elif unified_contribution > 0.2:
            role = "SUPPORTING_MATHEMATICAL_ELEMENT"
        else:
            role = "PASSIVE_MATHEMATICAL_VALUE"
        
        return {
            'operational_strength': strength,
            'operational_type': op_type,
            'computational_role': role,
            'unified_score': unified_contribution,
            'is_active_operator': unified_contribution > 0.3,
            'leech_lattice_compatible': metrics['leech_integration']['kissing_utilization'] > 0.001
        }
    
    def display_constant_results(self, results: Dict):
        """
        Display test results for a constant
        """
        info = results['constant_info']
        metrics = results['operational_metrics']
        classification = results['operational_classification']
        
        print(f"\n📊 OPERATIONAL ANALYSIS RESULTS:")
        print(f"Operational Strength:        {classification['operational_strength']}")
        print(f"Operational Type:            {classification['operational_type']}")
        print(f"Computational Role:          {classification['computational_role']}")
        print(f"Unified Score:               {classification['unified_score']:.6f}")
        print(f"Active Operator:             {'✓ YES' if classification['is_active_operator'] else '✗ NO'}")
        
        print(f"\n🔧 OPERATIONAL METRICS:")
        print(f"Operational Stability:       {metrics['operational_stability']:.6f}")
        print(f"Geometric Coherence:         {metrics['geometric_coherence']:.6f}")
        print(f"Constant Resonance:          {metrics['constant_resonance']:.6f}")
        print(f"Mean Operation Strength:     {metrics['mean_operation_strength']:.6f}")
        
        print(f"\n🔗 CROSS-CONSTANT COUPLING:")
        coupling = metrics['cross_coupling']
        print(f"π Coupling:                  {coupling['pi_coupling']:.6f}")
        print(f"φ Coupling:                  {coupling['phi_coupling']:.6f}")
        print(f"e Coupling:                  {coupling['e_coupling']:.6f}")
        print(f"Total Coupling:              {coupling['total_coupling']:.6f}")
        
        print(f"\n⚡ ERROR CORRECTION:")
        error_corr = results['error_correction']
        print(f"Level 3 Corrections:         {error_corr['level_3']['corrections']}")
        print(f"Level 6 Corrections:         {error_corr['level_6']['corrections']}")
        print(f"Level 9 Corrections:         {error_corr['level_9']['corrections']}")
        print(f"Overall Correction Rate:     {error_corr['overall_rate']*100:.2f}%")
        
        print(f"\n📈 COMPARISON TO CORE CONSTANTS:")
        comparison = results['comparison_to_core']
        print(f"Relative to π:               {comparison['vs_pi']:.3f}x")
        print(f"Relative to φ:               {comparison['vs_phi']:.3f}x")
        print(f"Relative to e:               {comparison['vs_e']:.3f}x")
    
    def run_comprehensive_test(self, n_terms: int = 30) -> Dict:
        """
        Run comprehensive test of all constants in catalog
        """
        print(f"\n{'='*80}")
        print(f"UBP CONSTANTS HANDBOOK - COMPREHENSIVE OPERATIONAL TESTING")
        print(f"{'='*80}")
        print(f"Testing {len(self.constants_catalog)} mathematical constants")
        print(f"Sequence length: {n_terms}")
        print(f"UBP Framework: {self.ubp_version}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Generate test sequence
        test_sequence = self.generate_test_sequence(n_terms)
        
        # Test all constants
        all_results = {}
        operational_constants = []
        
        for constant_key in self.constants_catalog:
            try:
                results = self.test_constant_as_operator(constant_key, test_sequence)
                all_results[constant_key] = results
                
                if results['operational_classification']['is_active_operator']:
                    operational_constants.append({
                        'key': constant_key,
                        'name': results['constant_info']['name'],
                        'score': results['operational_classification']['unified_score'],
                        'role': results['operational_classification']['computational_role']
                    })
            except Exception as e:
                print(f"Error testing {constant_key}: {e}")
                continue
        
        # Sort operational constants by score
        operational_constants.sort(key=lambda x: x['score'], reverse=True)
        
        # Generate summary
        summary = self.generate_handbook_summary(all_results, operational_constants)
        
        return {
            'test_parameters': {
                'n_terms': n_terms,
                'constants_tested': len(all_results),
                'timestamp': datetime.now().isoformat()
            },
            'individual_results': all_results,
            'operational_constants': operational_constants,
            'summary': summary
        }
    
    def generate_handbook_summary(self, all_results: Dict, operational_constants: List) -> Dict:
        """
        Generate comprehensive summary for the handbook
        """
        print(f"\n{'='*80}")
        print(f"HANDBOOK OF COMPUTATIONAL REALITY - SUMMARY")
        print(f"{'='*80}")
        
        # Categorize results
        core_operators = []
        specialized_functions = []
        supporting_elements = []
        passive_values = []
        
        for const in operational_constants:
            role = const['role']
            if 'FUNDAMENTAL' in role:
                core_operators.append(const)
            elif 'SPECIALIZED' in role:
                specialized_functions.append(const)
            elif 'SUPPORTING' in role:
                supporting_elements.append(const)
            else:
                passive_values.append(const)
        
        print(f"\n🔥 CORE COMPUTATIONAL OPERATORS ({len(core_operators)}):")
        for const in core_operators:
            print(f"  {const['name']}: {const['score']:.3f}")
        
        print(f"\n⚙️  SPECIALIZED COMPUTATIONAL FUNCTIONS ({len(specialized_functions)}):")
        for const in specialized_functions:
            print(f"  {const['name']}: {const['score']:.3f}")
        
        print(f"\n🔧 SUPPORTING MATHEMATICAL ELEMENTS ({len(supporting_elements)}):")
        for const in supporting_elements:
            print(f"  {const['name']}: {const['score']:.3f}")
        
        print(f"\n📊 TOTAL OPERATIONAL CONSTANTS: {len(operational_constants)}")
        print(f"📊 TOTAL TESTED: {len(all_results)}")
        print(f"📊 OPERATIONAL RATE: {len(operational_constants)/len(all_results)*100:.1f}%")
        
        return {
            'core_operators': core_operators,
            'specialized_functions': specialized_functions,
            'supporting_elements': supporting_elements,
            'passive_values': passive_values,
            'operational_rate': len(operational_constants)/len(all_results),
            'total_tested': len(all_results),
            'total_operational': len(operational_constants)
        }

def main():
    """Run the comprehensive constants handbook test"""
    tester = UBPConstantsHandbookTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test(n_terms=30)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ubp_constants_handbook_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Complete handbook saved to: {filename}")

if __name__ == "__main__":
    main()

