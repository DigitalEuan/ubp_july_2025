#!/usr/bin/env python3
"""
UBP Expanded Constants Investigator
Comprehensive investigation of mathematical constants and τ integration into UBP theory

Authors: Euan Craig (New Zealand) and Manus AI
Date: July 3, 2025
Purpose: Discover computational layers/hierarchies and analyze UBP with τ as core operator
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import math
from datetime import datetime
from typing import List, Tuple, Dict, Any

class UBPExpandedConstantsInvestigator:
    """
    Comprehensive investigator for mathematical constants and UBP theory evolution
    """
    
    def __init__(self):
        # Core validated constants (π, φ, e, τ)
        self.pi = math.pi
        self.phi = (1 + math.sqrt(5)) / 2
        self.e = math.e
        self.tau = 2 * math.pi  # Newly discovered operational constant
        
        # Expanded constants catalog for comprehensive testing
        self.expanded_constants = {
            # CATEGORY A: FUNDAMENTAL MATHEMATICAL CONSTANTS
            'sqrt_2': {'value': math.sqrt(2), 'name': 'Square Root of 2', 'symbol': '√2', 'type': 'algebraic'},
            'sqrt_3': {'value': math.sqrt(3), 'name': 'Square Root of 3', 'symbol': '√3', 'type': 'algebraic'},
            'sqrt_5': {'value': math.sqrt(5), 'name': 'Square Root of 5', 'symbol': '√5', 'type': 'algebraic'},
            'sqrt_6': {'value': math.sqrt(6), 'name': 'Square Root of 6', 'symbol': '√6', 'type': 'algebraic'},
            'sqrt_7': {'value': math.sqrt(7), 'name': 'Square Root of 7', 'symbol': '√7', 'type': 'algebraic'},
            'sqrt_8': {'value': math.sqrt(8), 'name': 'Square Root of 8', 'symbol': '√8', 'type': 'algebraic'},
            'cbrt_2': {'value': 2**(1/3), 'name': 'Cube Root of 2', 'symbol': '∛2', 'type': 'algebraic'},
            'cbrt_3': {'value': 3**(1/3), 'name': 'Cube Root of 3', 'symbol': '∛3', 'type': 'algebraic'},
            'cbrt_5': {'value': 5**(1/3), 'name': 'Cube Root of 5', 'symbol': '∛5', 'type': 'algebraic'},
            
            # CATEGORY B: TRANSCENDENTAL CONSTANTS
            'euler_gamma': {'value': 0.5772156649015329, 'name': 'Euler-Mascheroni Constant', 'symbol': 'γ', 'type': 'transcendental'},
            'ln_2': {'value': math.log(2), 'name': 'Natural Log of 2', 'symbol': 'ln(2)', 'type': 'transcendental'},
            'ln_3': {'value': math.log(3), 'name': 'Natural Log of 3', 'symbol': 'ln(3)', 'type': 'transcendental'},
            'ln_5': {'value': math.log(5), 'name': 'Natural Log of 5', 'symbol': 'ln(5)', 'type': 'transcendental'},
            'ln_10': {'value': math.log(10), 'name': 'Natural Log of 10', 'symbol': 'ln(10)', 'type': 'transcendental'},
            'log10_2': {'value': math.log10(2), 'name': 'Log Base 10 of 2', 'symbol': 'log₁₀(2)', 'type': 'transcendental'},
            'log10_e': {'value': math.log10(math.e), 'name': 'Log Base 10 of e', 'symbol': 'log₁₀(e)', 'type': 'transcendental'},
            
            # CATEGORY C: SPECIAL FUNCTION CONSTANTS
            'apery': {'value': 1.2020569031595942, 'name': "Apéry's Constant", 'symbol': 'ζ(3)', 'type': 'special_function'},
            'catalan': {'value': 0.9159655941772190, 'name': "Catalan's Constant", 'symbol': 'G', 'type': 'special_function'},
            'omega': {'value': 0.5671432904097838, 'name': 'Omega Constant', 'symbol': 'Ω', 'type': 'special_function'},
            'zeta_2': {'value': (math.pi**2)/6, 'name': 'Riemann Zeta(2)', 'symbol': 'ζ(2)', 'type': 'special_function'},
            'zeta_4': {'value': (math.pi**4)/90, 'name': 'Riemann Zeta(4)', 'symbol': 'ζ(4)', 'type': 'special_function'},
            'zeta_6': {'value': (math.pi**6)/945, 'name': 'Riemann Zeta(6)', 'symbol': 'ζ(6)', 'type': 'special_function'},
            
            # CATEGORY D: GEOMETRIC RATIOS
            'silver_ratio': {'value': 1 + math.sqrt(2), 'name': 'Silver Ratio', 'symbol': 'δ_S', 'type': 'geometric'},
            'plastic_number': {'value': 1.3247179572447460, 'name': 'Plastic Number', 'symbol': 'ρ', 'type': 'geometric'},
            'supergolden': {'value': 1.4655712318767680, 'name': 'Supergolden Ratio', 'symbol': 'ψ', 'type': 'geometric'},
            'bronze_ratio': {'value': (3 + math.sqrt(13))/2, 'name': 'Bronze Ratio', 'symbol': 'β', 'type': 'geometric'},
            'golden_angle': {'value': 2*math.pi/((1+math.sqrt(5))/2)**2, 'name': 'Golden Angle', 'symbol': 'φ_angle', 'type': 'geometric'},
            
            # CATEGORY E: COMPUTATIONAL CONSTANTS
            'feigenbaum_delta': {'value': 4.669201609102990, 'name': 'Feigenbaum Delta', 'symbol': 'δ', 'type': 'computational'},
            'feigenbaum_alpha': {'value': 2.502907875095892, 'name': 'Feigenbaum Alpha', 'symbol': 'α', 'type': 'computational'},
            'conway': {'value': 1.303577269034296, 'name': "Conway's Constant", 'symbol': 'λ', 'type': 'computational'},
            'khinchin': {'value': 2.685452001065306, 'name': "Khinchin's Constant", 'symbol': 'K₀', 'type': 'computational'},
            'levy': {'value': 3.275822918721811, 'name': "Lévy's Constant", 'symbol': 'γ_L', 'type': 'computational'},
            
            # CATEGORY F: EXOTIC/ADVANCED CONSTANTS
            'gelfond': {'value': math.e**math.pi, 'name': "Gelfond's Constant", 'symbol': 'e^π', 'type': 'exotic'},
            'gelfond_schneider': {'value': 2**math.sqrt(2), 'name': 'Gelfond-Schneider Constant', 'symbol': '2^√2', 'type': 'exotic'},
            'liouville': {'value': 0.11000100000000000000000001, 'name': "Liouville's Constant", 'symbol': 'L', 'type': 'exotic'},
            'champernowne': {'value': 0.12345678910111213141516, 'name': 'Champernowne Constant', 'symbol': 'C₁₀', 'type': 'exotic'},
            'copeland_erdos': {'value': 0.23571113171923293137, 'name': 'Copeland-Erdős Constant', 'symbol': 'C_p', 'type': 'exotic'},
            'cahen': {'value': 0.6434105462883802618, 'name': "Cahen's Constant", 'symbol': 'C', 'type': 'exotic'},
            
            # CATEGORY G: PHYSICAL/UNIVERSAL CONSTANTS (Mathematical aspects)
            'fine_structure_math': {'value': 1/137.035999084, 'name': 'Fine Structure (Math)', 'symbol': 'α_fs', 'type': 'physical'},
            'planck_reduced_math': {'value': 1.054571817e-34, 'name': 'Reduced Planck (Math)', 'symbol': 'ℏ_m', 'type': 'physical'},
            'light_speed_math': {'value': 299792458, 'name': 'Light Speed (Math)', 'symbol': 'c_m', 'type': 'physical'},
            
            # CATEGORY H: COMPUTING-RELATED CONSTANTS
            'binary_log_e': {'value': math.log2(math.e), 'name': 'Binary Log of e', 'symbol': 'log₂(e)', 'type': 'computing'},
            'binary_log_10': {'value': math.log2(10), 'name': 'Binary Log of 10', 'symbol': 'log₂(10)', 'type': 'computing'},
            'binary_log_pi': {'value': math.log2(math.pi), 'name': 'Binary Log of π', 'symbol': 'log₂(π)', 'type': 'computing'},
            'binary_log_phi': {'value': math.log2((1+math.sqrt(5))/2), 'name': 'Binary Log of φ', 'symbol': 'log₂(φ)', 'type': 'computing'},
            'e_to_pi': {'value': math.e**math.pi, 'name': 'e to the π', 'symbol': 'e^π', 'type': 'computing'},
            'pi_to_e': {'value': math.pi**math.e, 'name': 'π to the e', 'symbol': 'π^e', 'type': 'computing'},
            
            # CATEGORY I: COMBINATORIAL CONSTANTS
            'stirling_approx': {'value': math.sqrt(2*math.pi), 'name': 'Stirling Approximation', 'symbol': '√(2π)', 'type': 'combinatorial'},
            'ramanujan_hardy': {'value': 1729, 'name': 'Ramanujan-Hardy Number', 'symbol': '1729', 'type': 'combinatorial'},
            'euler_totient_avg': {'value': 6/(math.pi**2), 'name': 'Euler Totient Average', 'symbol': '6/π²', 'type': 'combinatorial'},
        }
        
        # Leech Lattice parameters
        self.leech_dimension = 24
        self.kissing_number = 196560
        self.error_correction_levels = [3, 6, 9]
        
        # UBP framework with τ integration
        self.ubp_version = "v23.0_TauIntegrated"
        
        # Core operational constants (now including τ)
        self.core_operators = {
            'pi': self.pi,
            'phi': self.phi, 
            'e': self.e,
            'tau': self.tau  # Newly integrated
        }
        
    def test_constant_comprehensive(self, constant_key: str, test_sequence: List[int]) -> Dict:
        """
        Comprehensive test of a mathematical constant as operational function
        """
        if constant_key not in self.expanded_constants:
            return {}
        
        constant_info = self.expanded_constants[constant_key]
        constant_value = constant_info['value']
        
        print(f"\n{'='*70}")
        print(f"TESTING: {constant_info['name']} ({constant_info['symbol']})")
        print(f"Value: {constant_value:.12f}")
        print(f"Type: {constant_info['type']}")
        print(f"{'='*70}")
        
        # Enhanced encoding with τ integration
        offbits = self.encode_tau_integrated_offbits(test_sequence, constant_value, constant_key)
        
        # Calculate 24D positions with τ as core operator
        positions = self.calculate_tau_integrated_positions(offbits, constant_value)
        
        # Analyze error correction with 4-constant framework (π, φ, e, τ)
        error_correction = self.analyze_four_constant_error_correction(offbits, positions, constant_value)
        
        # Calculate operational metrics with τ integration
        operational_metrics = self.calculate_tau_integrated_metrics(offbits, positions, constant_value, constant_key)
        
        # Calculate unified invariant with 4-constant framework
        unified_contribution = self.calculate_four_constant_unified_contribution(operational_metrics, constant_value)
        
        # Enhanced classification with computational layers
        operational_class = self.classify_computational_layer(operational_metrics, unified_contribution, constant_info['type'])
        
        # Cross-constant analysis with all 4 core operators
        cross_analysis = self.analyze_cross_constant_relationships(constant_value, operational_metrics)
        
        results = {
            'constant_info': constant_info,
            'test_parameters': {
                'sequence_length': len(test_sequence),
                'tau_integrated': True,
                'core_operators_count': 4
            },
            'offbits_created': len(offbits),
            'error_correction': error_correction,
            'operational_metrics': operational_metrics,
            'unified_contribution': unified_contribution,
            'operational_classification': operational_class,
            'cross_constant_analysis': cross_analysis,
            'computational_layer': self.determine_computational_layer(operational_class, constant_info['type'])
        }
        
        self.display_comprehensive_results(results)
        return results
    
    def encode_tau_integrated_offbits(self, sequence: List[int], constant: float, constant_key: str) -> List[Dict]:
        """
        Encode OffBits with τ integrated as 4th core operator
        """
        offbits = []
        
        for i, num in enumerate(sequence):
            binary_rep = format(num % (2**24), '024b')
            
            # Split into 4 layers (6 bits each) - now with τ integration
            layers = [
                binary_rep[0:6],    # Reality layer (π-based)
                binary_rep[6:12],   # Information layer (φ-based)
                binary_rep[12:18],  # Activation layer (e-based)
                binary_rep[18:24]   # Unactivated layer (τ-based) - NEW!
            ]
            
            # Apply 4-constant operational framework
            layer_operations = []
            for j, layer in enumerate(layers):
                layer_val = int(layer, 2)
                
                if j == 0:  # Reality layer - π operations
                    operation = (layer_val * self.pi * constant) / (64 * self.pi)
                elif j == 1:  # Information layer - φ operations
                    operation = (layer_val * self.phi * constant) / (64 * self.phi)
                elif j == 2:  # Activation layer - e operations
                    operation = (layer_val * self.e * constant) / (64 * self.e)
                else:  # Unactivated layer - τ operations (NEW!)
                    operation = (layer_val * self.tau * constant) / (64 * self.tau)
                
                # Apply constant-specific modulation
                if constant_key in ['sqrt_2', 'sqrt_3', 'sqrt_5', 'sqrt_6', 'sqrt_7', 'sqrt_8']:
                    operation *= math.sqrt(j + 1)
                elif constant_key in ['cbrt_2', 'cbrt_3', 'cbrt_5']:
                    operation *= (j + 1)**(1/3)
                elif constant_key in ['ln_2', 'ln_3', 'ln_5', 'ln_10']:
                    operation *= math.log(j + 2)
                elif constant_key in ['feigenbaum_delta', 'feigenbaum_alpha']:
                    operation *= math.sin(constant * (j + 1))
                
                layer_operations.append(operation)
            
            # Calculate τ-integrated total operation
            tau_integration_factor = math.cos(i * self.tau / len(sequence))
            total_operation = sum(layer_operations) * tau_integration_factor
            
            offbit = {
                'index': i,
                'sequence_value': num,
                'binary_representation': binary_rep,
                'layers': layers,
                'layer_operations': layer_operations,
                'total_operation': total_operation,
                'tau_integration_factor': tau_integration_factor,
                'constant_applied': constant
            }
            
            offbits.append(offbit)
        
        return offbits
    
    def calculate_tau_integrated_positions(self, offbits: List[Dict], constant: float) -> List[Tuple]:
        """
        Calculate 24D positions with τ as 4th core operator
        """
        positions = []
        
        for i, offbit in enumerate(offbits):
            coordinates = []
            
            for dim in range(self.leech_dimension):
                layer_idx = dim % 4
                operation_val = offbit['layer_operations'][layer_idx]
                
                # 4-constant geometric framework
                if dim < 6:  # Dimensions 0-5: π operations
                    coord = operation_val * math.cos(dim * self.pi / 6)
                elif dim < 12:  # Dimensions 6-11: φ operations
                    coord = operation_val * math.sin(dim * self.phi / 6)
                elif dim < 18:  # Dimensions 12-17: e operations
                    coord = operation_val * math.cos(dim * self.e / 6)
                else:  # Dimensions 18-23: τ operations (NEW!)
                    coord = operation_val * math.sin(dim * self.tau / 6)
                
                # Apply constant-specific geometric transformations
                coord *= (1 + constant / 10) * offbit['tau_integration_factor']
                
                coordinates.append(coord)
            
            positions.append(tuple(coordinates))
        
        return positions
    
    def analyze_four_constant_error_correction(self, offbits: List[Dict], positions: List[Tuple], constant: float) -> Dict:
        """
        Analyze error correction with 4-constant framework (π, φ, e, τ)
        """
        error_correction = {
            'level_3': {'corrections': 0, 'strength': 0.0, 'operator': 'φ'},
            'level_6': {'corrections': 0, 'strength': 0.0, 'operator': 'π'},
            'level_9': {'corrections': 0, 'strength': 0.0, 'operator': 'e'},
            'level_12': {'corrections': 0, 'strength': 0.0, 'operator': 'τ'},  # NEW!
            'overall_rate': 0.0
        }
        
        total_correctable = 0
        levels = [3, 6, 9, 12]  # Extended with τ level
        operators = [self.phi, self.pi, self.e, self.tau]
        
        for i, (offbit, pos) in enumerate(zip(offbits, positions)):
            for j, level in enumerate(levels):
                if level <= 24:  # Ensure we don't exceed 24D
                    level_coords = pos[:level]
                    level_distance = math.sqrt(sum(coord**2 for coord in level_coords))
                    
                    # Calculate correction strength using appropriate operator
                    correction_strength = (operators[j] * constant) / (level_distance + 1e-10)
                    
                    error_correction[f'level_{level}']['strength'] += correction_strength
                    
                    if correction_strength > 1.0:
                        error_correction[f'level_{level}']['corrections'] += 1
                        total_correctable += 1
        
        # Calculate averages
        for level in levels:
            if level <= 24:
                error_correction[f'level_{level}']['strength'] /= len(offbits)
        
        error_correction['overall_rate'] = total_correctable / (len(offbits) * len(levels))
        
        return error_correction
    
    def calculate_tau_integrated_metrics(self, offbits: List[Dict], positions: List[Tuple], 
                                       constant: float, constant_key: str) -> Dict:
        """
        Calculate operational metrics with τ integration
        """
        if not offbits:
            return {}
        
        # Enhanced operational stability with τ
        operations = [offbit['total_operation'] for offbit in offbits]
        tau_factors = [offbit['tau_integration_factor'] for offbit in offbits]
        
        mean_operation = sum(operations) / len(operations)
        mean_tau_factor = sum(tau_factors) / len(tau_factors)
        
        # τ-enhanced stability calculation
        tau_enhanced_stability = 1.0 - (np.std(operations) / (mean_operation + 1e-10)) * abs(mean_tau_factor)
        
        # 4-constant cross-coupling analysis
        pi_coupling = abs(math.sin(constant * self.pi))
        phi_coupling = abs(math.cos(constant * self.phi))
        e_coupling = abs(math.sin(constant * self.e))
        tau_coupling = abs(math.cos(constant * self.tau))  # NEW!
        
        # Enhanced resonance with τ
        tau_resonance = self.calculate_tau_resonance(constant, operations, tau_factors)
        
        # Computational layer analysis
        layer_analysis = self.analyze_computational_layer_behavior(constant, constant_key, operations)
        
        metrics = {
            'tau_enhanced_stability': tau_enhanced_stability,
            'mean_operation_strength': mean_operation,
            'mean_tau_integration': mean_tau_factor,
            'tau_resonance': tau_resonance,
            'four_constant_coupling': {
                'pi_coupling': pi_coupling,
                'phi_coupling': phi_coupling,
                'e_coupling': e_coupling,
                'tau_coupling': tau_coupling,  # NEW!
                'total_coupling': pi_coupling + phi_coupling + e_coupling + tau_coupling
            },
            'computational_layer_analysis': layer_analysis,
            'leech_integration': {
                'kissing_utilization': min(1.0, len(offbits) / self.kissing_number),
                'dimensional_coverage': 1.0,
                'tau_dimensional_coverage': 6/24  # τ operates on dimensions 18-23
            }
        }
        
        return metrics
    
    def calculate_tau_resonance(self, constant: float, operations: List[float], tau_factors: List[float]) -> float:
        """
        Calculate resonance frequency enhanced with τ
        """
        if len(operations) < 2:
            return 0.0
        
        resonance_sum = 0.0
        for i in range(1, len(operations)):
            ratio = operations[i] / (operations[i-1] + 1e-10)
            tau_modulation = tau_factors[i] * self.tau
            
            # τ-enhanced resonance calculation
            resonance = abs(math.sin(ratio * constant * self.pi)) * abs(math.cos(tau_modulation))
            resonance_sum += resonance
        
        return resonance_sum / (len(operations) - 1)
    
    def analyze_computational_layer_behavior(self, constant: float, constant_key: str, operations: List[float]) -> Dict:
        """
        Analyze which computational layer the constant operates in
        """
        constant_type = self.expanded_constants[constant_key]['type']
        
        # Analyze operational patterns
        operation_variance = np.var(operations) if len(operations) > 1 else 0
        operation_trend = (operations[-1] - operations[0]) / len(operations) if len(operations) > 1 else 0
        
        # Determine computational layer characteristics
        layer_characteristics = {
            'algebraic': {'stability': 0.8, 'predictability': 0.9, 'complexity': 0.3},
            'transcendental': {'stability': 0.6, 'predictability': 0.5, 'complexity': 0.8},
            'special_function': {'stability': 0.4, 'predictability': 0.3, 'complexity': 0.9},
            'geometric': {'stability': 0.7, 'predictability': 0.7, 'complexity': 0.5},
            'computational': {'stability': 0.3, 'predictability': 0.2, 'complexity': 1.0},
            'exotic': {'stability': 0.2, 'predictability': 0.1, 'complexity': 1.0},
            'physical': {'stability': 0.9, 'predictability': 0.8, 'complexity': 0.4},
            'computing': {'stability': 0.5, 'predictability': 0.6, 'complexity': 0.7},
            'combinatorial': {'stability': 0.6, 'predictability': 0.4, 'complexity': 0.6}
        }
        
        expected_chars = layer_characteristics.get(constant_type, {'stability': 0.5, 'predictability': 0.5, 'complexity': 0.5})
        
        # Calculate actual vs expected behavior
        actual_stability = 1.0 - min(1.0, operation_variance)
        actual_predictability = 1.0 - min(1.0, abs(operation_trend))
        actual_complexity = min(1.0, constant / 10)  # Normalized complexity measure
        
        layer_analysis = {
            'constant_type': constant_type,
            'expected_characteristics': expected_chars,
            'actual_characteristics': {
                'stability': actual_stability,
                'predictability': actual_predictability,
                'complexity': actual_complexity
            },
            'layer_alignment': {
                'stability_match': 1.0 - abs(actual_stability - expected_chars['stability']),
                'predictability_match': 1.0 - abs(actual_predictability - expected_chars['predictability']),
                'complexity_match': 1.0 - abs(actual_complexity - expected_chars['complexity'])
            }
        }
        
        # Overall layer alignment score
        layer_analysis['overall_alignment'] = (
            layer_analysis['layer_alignment']['stability_match'] +
            layer_analysis['layer_alignment']['predictability_match'] +
            layer_analysis['layer_alignment']['complexity_match']
        ) / 3
        
        return layer_analysis
    
    def calculate_four_constant_unified_contribution(self, metrics: Dict, constant: float) -> float:
        """
        Calculate unified invariant contribution with 4-constant framework
        """
        if not metrics:
            return 0.0
        
        # Enhanced weighting with τ integration
        stability_weight = metrics['tau_enhanced_stability'] * 0.25
        resonance_weight = metrics['tau_resonance'] * 0.25
        coupling_weight = metrics['four_constant_coupling']['total_coupling'] * 0.25
        layer_weight = metrics['computational_layer_analysis']['overall_alignment'] * 0.25
        
        # τ-enhanced scaling
        tau_scale = min(10.0, constant * self.tau / (self.pi * self.phi * self.e)) / 10.0
        
        unified_contribution = (
            stability_weight + resonance_weight + coupling_weight + layer_weight
        ) * tau_scale
        
        return unified_contribution
    
    def classify_computational_layer(self, metrics: Dict, unified_contribution: float, constant_type: str) -> Dict:
        """
        Enhanced classification with computational layers
        """
        # Determine operational strength with τ integration
        if unified_contribution > 1.0:
            strength = "CORE_OPERATIONAL"
        elif unified_contribution > 0.7:
            strength = "STRONG_OPERATIONAL"
        elif unified_contribution > 0.4:
            strength = "MODERATE_OPERATIONAL"
        elif unified_contribution > 0.2:
            strength = "WEAK_OPERATIONAL"
        else:
            strength = "NON_OPERATIONAL"
        
        # Determine computational layer
        layer_alignment = metrics['computational_layer_analysis']['overall_alignment']
        total_coupling = metrics['four_constant_coupling']['total_coupling']
        
        if total_coupling > 3.0 and unified_contribution > 0.8:
            comp_layer = "CORE_REALITY_LAYER"
        elif total_coupling > 2.0 and unified_contribution > 0.5:
            comp_layer = "OPERATIONAL_LAYER"
        elif layer_alignment > 0.7:
            comp_layer = f"SPECIALIZED_{constant_type.upper()}_LAYER"
        elif unified_contribution > 0.3:
            comp_layer = "AUXILIARY_COMPUTATIONAL_LAYER"
        else:
            comp_layer = "PASSIVE_MATHEMATICAL_LAYER"
        
        # Determine role in UBP with τ
        if unified_contribution > 0.8 and total_coupling > 3.0:
            ubp_role = "FUNDAMENTAL_UBP_OPERATOR"
        elif unified_contribution > 0.5:
            ubp_role = "SPECIALIZED_UBP_FUNCTION"
        elif unified_contribution > 0.3:
            ubp_role = "SUPPORTING_UBP_ELEMENT"
        else:
            ubp_role = "NON_UBP_MATHEMATICAL_VALUE"
        
        return {
            'operational_strength': strength,
            'computational_layer': comp_layer,
            'ubp_role': ubp_role,
            'unified_score': unified_contribution,
            'is_operational': unified_contribution > 0.3,
            'tau_compatible': metrics['four_constant_coupling']['tau_coupling'] > 0.5,
            'layer_type': constant_type,
            'layer_alignment_score': layer_alignment
        }
    
    def analyze_cross_constant_relationships(self, constant: float, metrics: Dict) -> Dict:
        """
        Analyze relationships with all 4 core operators
        """
        coupling = metrics['four_constant_coupling']
        
        # Calculate relationship strengths
        relationships = {
            'with_pi': {
                'coupling_strength': coupling['pi_coupling'],
                'ratio': constant / self.pi,
                'harmonic_relationship': abs(math.sin(constant * self.pi)) > 0.8,
                'geometric_relationship': abs(math.cos(constant / self.pi)) > 0.8
            },
            'with_phi': {
                'coupling_strength': coupling['phi_coupling'],
                'ratio': constant / self.phi,
                'golden_relationship': abs(constant - self.phi) < 0.1 or abs(constant - self.phi**2) < 0.1,
                'proportion_relationship': abs(math.sin(constant * self.phi)) > 0.8
            },
            'with_e': {
                'coupling_strength': coupling['e_coupling'],
                'ratio': constant / self.e,
                'exponential_relationship': abs(math.log(constant + 1) - self.e) < 0.5,
                'growth_relationship': abs(math.sin(constant * self.e)) > 0.8
            },
            'with_tau': {  # NEW!
                'coupling_strength': coupling['tau_coupling'],
                'ratio': constant / self.tau,
                'circular_relationship': abs(constant - self.tau) < 0.1 or abs(constant - self.tau/2) < 0.1,
                'periodic_relationship': abs(math.cos(constant * self.tau)) > 0.8
            }
        }
        
        # Determine strongest relationships
        strongest_coupling = max(coupling.values())
        strongest_operator = max(coupling, key=coupling.get).replace('_coupling', '')
        
        # Calculate overall integration with core operators
        total_integration = sum(rel['coupling_strength'] for rel in relationships.values())
        
        return {
            'individual_relationships': relationships,
            'strongest_coupling': strongest_coupling,
            'strongest_operator': strongest_operator,
            'total_core_integration': total_integration,
            'is_core_integrated': total_integration > 2.0,
            'tau_integration_strength': relationships['with_tau']['coupling_strength']
        }
    
    def determine_computational_layer(self, classification: Dict, constant_type: str) -> str:
        """
        Determine which computational layer the constant belongs to
        """
        strength = classification['operational_strength']
        layer = classification['computational_layer']
        
        # Define computational hierarchy
        if "CORE_REALITY" in layer:
            return "LAYER_0_CORE_REALITY"
        elif "OPERATIONAL" in layer and "STRONG" in strength:
            return "LAYER_1_PRIMARY_OPERATORS"
        elif "OPERATIONAL" in layer:
            return "LAYER_2_SECONDARY_OPERATORS"
        elif "SPECIALIZED" in layer:
            return f"LAYER_3_SPECIALIZED_{constant_type.upper()}"
        elif "AUXILIARY" in layer:
            return "LAYER_4_AUXILIARY_FUNCTIONS"
        else:
            return "LAYER_5_PASSIVE_VALUES"
    
    def display_comprehensive_results(self, results: Dict):
        """
        Display comprehensive test results
        """
        info = results['constant_info']
        metrics = results['operational_metrics']
        classification = results['operational_classification']
        cross_analysis = results['cross_constant_analysis']
        
        print(f"\n📊 COMPREHENSIVE OPERATIONAL ANALYSIS:")
        print(f"Operational Strength:        {classification['operational_strength']}")
        print(f"Computational Layer:         {classification['computational_layer']}")
        print(f"UBP Role:                    {classification['ubp_role']}")
        print(f"Unified Score:               {classification['unified_score']:.6f}")
        print(f"Operational:                 {'✓ YES' if classification['is_operational'] else '✗ NO'}")
        print(f"τ Compatible:                {'✓ YES' if classification['tau_compatible'] else '✗ NO'}")
        
        print(f"\n🔧 τ-ENHANCED METRICS:")
        print(f"τ-Enhanced Stability:        {metrics['tau_enhanced_stability']:.6f}")
        print(f"τ Resonance:                 {metrics['tau_resonance']:.6f}")
        print(f"Mean τ Integration:          {metrics['mean_tau_integration']:.6f}")
        
        print(f"\n🔗 4-CONSTANT COUPLING:")
        coupling = metrics['four_constant_coupling']
        print(f"π Coupling:                  {coupling['pi_coupling']:.6f}")
        print(f"φ Coupling:                  {coupling['phi_coupling']:.6f}")
        print(f"e Coupling:                  {coupling['e_coupling']:.6f}")
        print(f"τ Coupling:                  {coupling['tau_coupling']:.6f} ⭐")
        print(f"Total Coupling:              {coupling['total_coupling']:.6f}")
        
        print(f"\n🎯 CROSS-CONSTANT ANALYSIS:")
        print(f"Strongest Operator:          {cross_analysis['strongest_operator']}")
        print(f"Total Core Integration:      {cross_analysis['total_core_integration']:.6f}")
        print(f"Core Integrated:             {'✓ YES' if cross_analysis['is_core_integrated'] else '✗ NO'}")
        print(f"τ Integration Strength:      {cross_analysis['tau_integration_strength']:.6f}")
        
        print(f"\n📈 COMPUTATIONAL LAYER:")
        print(f"Layer Assignment:            {results['computational_layer']}")
        layer_analysis = metrics['computational_layer_analysis']
        print(f"Layer Alignment:             {layer_analysis['overall_alignment']:.6f}")
        print(f"Type Consistency:            {layer_analysis['constant_type']}")
    
    def run_expanded_investigation(self, n_terms: int = 30) -> Dict:
        """
        Run expanded investigation of all constants with τ integration
        """
        print(f"\n{'='*80}")
        print(f"UBP EXPANDED CONSTANTS INVESTIGATION - τ INTEGRATED")
        print(f"{'='*80}")
        print(f"Testing {len(self.expanded_constants)} mathematical constants")
        print(f"Core operators: π, φ, e, τ (4 total)")
        print(f"Sequence length: {n_terms}")
        print(f"UBP Framework: {self.ubp_version}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Generate test sequence
        test_sequence = self.generate_fibonacci_sequence(n_terms)
        
        # Test all constants
        all_results = {}
        operational_constants = []
        layer_distribution = {}
        
        for constant_key in self.expanded_constants:
            try:
                results = self.test_constant_comprehensive(constant_key, test_sequence)
                all_results[constant_key] = results
                
                # Collect operational constants
                if results['operational_classification']['is_operational']:
                    operational_constants.append({
                        'key': constant_key,
                        'name': results['constant_info']['name'],
                        'score': results['operational_classification']['unified_score'],
                        'layer': results['computational_layer'],
                        'tau_compatible': results['operational_classification']['tau_compatible']
                    })
                
                # Track layer distribution
                layer = results['computational_layer']
                layer_distribution[layer] = layer_distribution.get(layer, 0) + 1
                
            except Exception as e:
                print(f"Error testing {constant_key}: {e}")
                continue
        
        # Sort by score
        operational_constants.sort(key=lambda x: x['score'], reverse=True)
        
        # Generate comprehensive summary
        summary = self.generate_expanded_summary(all_results, operational_constants, layer_distribution)
        
        return {
            'test_parameters': {
                'n_terms': n_terms,
                'constants_tested': len(all_results),
                'tau_integrated': True,
                'core_operators': 4,
                'timestamp': datetime.now().isoformat()
            },
            'individual_results': all_results,
            'operational_constants': operational_constants,
            'layer_distribution': layer_distribution,
            'summary': summary
        }
    
    def generate_fibonacci_sequence(self, n_terms: int) -> List[int]:
        """Generate Fibonacci sequence"""
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
    
    def generate_expanded_summary(self, all_results: Dict, operational_constants: List, layer_distribution: Dict) -> Dict:
        """
        Generate comprehensive summary of expanded investigation
        """
        print(f"\n{'='*80}")
        print(f"EXPANDED INVESTIGATION SUMMARY - τ INTEGRATED UBP")
        print(f"{'='*80}")
        
        # Categorize by computational layers
        layer_0 = [c for c in operational_constants if 'LAYER_0' in c['layer']]
        layer_1 = [c for c in operational_constants if 'LAYER_1' in c['layer']]
        layer_2 = [c for c in operational_constants if 'LAYER_2' in c['layer']]
        layer_3 = [c for c in operational_constants if 'LAYER_3' in c['layer']]
        layer_4 = [c for c in operational_constants if 'LAYER_4' in c['layer']]
        
        print(f"\n🔥 LAYER 0 - CORE REALITY ({len(layer_0)}):")
        for const in layer_0:
            tau_indicator = "🌟" if const['tau_compatible'] else ""
            print(f"  {const['name']}: {const['score']:.3f} {tau_indicator}")
        
        print(f"\n⚡ LAYER 1 - PRIMARY OPERATORS ({len(layer_1)}):")
        for const in layer_1:
            tau_indicator = "🌟" if const['tau_compatible'] else ""
            print(f"  {const['name']}: {const['score']:.3f} {tau_indicator}")
        
        print(f"\n🔧 LAYER 2 - SECONDARY OPERATORS ({len(layer_2)}):")
        for const in layer_2:
            tau_indicator = "🌟" if const['tau_compatible'] else ""
            print(f"  {const['name']}: {const['score']:.3f} {tau_indicator}")
        
        print(f"\n⚙️  LAYER 3 - SPECIALIZED FUNCTIONS ({len(layer_3)}):")
        for const in layer_3:
            tau_indicator = "🌟" if const['tau_compatible'] else ""
            print(f"  {const['name']}: {const['score']:.3f} {tau_indicator}")
        
        print(f"\n🔩 LAYER 4 - AUXILIARY FUNCTIONS ({len(layer_4)}):")
        for const in layer_4:
            tau_indicator = "🌟" if const['tau_compatible'] else ""
            print(f"  {const['name']}: {const['score']:.3f} {tau_indicator}")
        
        # τ compatibility analysis
        tau_compatible = [c for c in operational_constants if c['tau_compatible']]
        
        print(f"\n🌟 τ INTEGRATION ANALYSIS:")
        print(f"Total Operational Constants:  {len(operational_constants)}")
        print(f"τ Compatible Constants:       {len(tau_compatible)}")
        print(f"τ Compatibility Rate:         {len(tau_compatible)/len(operational_constants)*100:.1f}%")
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"Total Constants Tested:       {len(all_results)}")
        print(f"Operational Constants:        {len(operational_constants)}")
        print(f"Operational Rate:             {len(operational_constants)/len(all_results)*100:.1f}%")
        print(f"Layer Distribution:           {len(layer_distribution)} layers identified")
        
        return {
            'layer_0_core_reality': layer_0,
            'layer_1_primary_operators': layer_1,
            'layer_2_secondary_operators': layer_2,
            'layer_3_specialized_functions': layer_3,
            'layer_4_auxiliary_functions': layer_4,
            'tau_compatible_constants': tau_compatible,
            'layer_distribution': layer_distribution,
            'operational_rate': len(operational_constants)/len(all_results),
            'tau_compatibility_rate': len(tau_compatible)/len(operational_constants) if operational_constants else 0,
            'total_tested': len(all_results),
            'total_operational': len(operational_constants)
        }

def main():
    """Run the expanded constants investigation"""
    investigator = UBPExpandedConstantsInvestigator()
    
    # Run comprehensive investigation
    results = investigator.run_expanded_investigation(n_terms=30)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ubp_expanded_investigation_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Complete expanded investigation saved to: {filename}")

if __name__ == "__main__":
    main()

