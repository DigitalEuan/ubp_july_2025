#!/usr/bin/env python3
"""
UBP Fibonacci Sequence Analysis Framework
Building on successful Collatz methodology to explore φ-π relationships

Authors: Euan Craig (New Zealand) and Manus AI
Date: July 3, 2025
Purpose: Foundational analysis of Fibonacci sequences through UBP framework
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any
import math

class UBPFibonacciAnalyzer:
    """
    UBP-based Fibonacci sequence analyzer
    Explores φ-π relationships through geometric invariant analysis
    """
    
    def __init__(self):
        # Mathematical constants
        self.pi = math.pi
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.e = math.e
        
        # UBP Framework constants
        self.ubp_version = "v22.0_Fibonacci"
        self.tgic_structure = "3-6-9"
        
        # Analysis parameters
        self.max_sequence_length = 1000
        self.offbit_layers = 4  # Reality, Information, Activation, Unactivated
        
    def generate_fibonacci_sequence(self, n_terms: int) -> List[int]:
        """Generate Fibonacci sequence up to n terms"""
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
    
    def encode_fibonacci_offbits(self, fib_sequence: List[int]) -> List[Dict]:
        """
        Encode Fibonacci numbers as 24-bit OffBits with ontological layers
        Each layer represents different aspects of the recursive relationship
        """
        offbits = []
        
        for i, fib_num in enumerate(fib_sequence):
            # Convert to binary representation
            binary_rep = format(fib_num, '024b')
            
            # Split into 4 ontological layers (6 bits each)
            reality_layer = binary_rep[0:6]      # Most significant - actual value
            info_layer = binary_rep[6:12]        # Recursive relationship info
            activation_layer = binary_rep[12:18] # Position in sequence
            unactivated_layer = binary_rep[18:24] # Potential relationships
            
            # Calculate layer weights using fundamental constants
            reality_weight = int(reality_layer, 2) * self.pi / 64
            info_weight = int(info_layer, 2) * self.phi / 64
            activation_weight = int(activation_layer, 2) * self.e / 64
            unactivated_weight = int(unactivated_layer, 2) / 64
            
            offbit = {
                'index': i,
                'fibonacci_number': fib_num,
                'binary_representation': binary_rep,
                'layers': {
                    'reality': reality_layer,
                    'information': info_layer,
                    'activation': activation_layer,
                    'unactivated': unactivated_layer
                },
                'weights': {
                    'reality': reality_weight,
                    'information': info_weight,
                    'activation': activation_weight,
                    'unactivated': unactivated_weight
                },
                'total_weight': reality_weight + info_weight + activation_weight + unactivated_weight
            }
            
            offbits.append(offbit)
        
        return offbits
    
    def calculate_3d_positions(self, offbits: List[Dict]) -> List[Tuple[float, float, float]]:
        """
        Calculate 3D spatial positions for OffBits using UBP geometric mapping
        Incorporates φ-based spiral geometry
        """
        positions = []
        
        for i, offbit in enumerate(offbits):
            # Use golden ratio for spiral positioning
            theta = i * 2 * self.pi / self.phi  # φ-based angular increment
            radius = math.sqrt(i + 1) * self.phi  # φ-scaled radial distance
            
            # 3D coordinates incorporating all layer weights
            x = radius * math.cos(theta) * offbit['weights']['reality']
            y = radius * math.sin(theta) * offbit['weights']['information']
            z = i * offbit['weights']['activation'] + offbit['weights']['unactivated']
            
            positions.append((x, y, z))
        
        return positions
    
    def form_fibonacci_glyphs(self, offbits: List[Dict], positions: List[Tuple]) -> List[Dict]:
        """
        Form Glyphs from OffBits using TGIC (3-6-9) principles
        Fibonacci-specific clustering based on golden ratio relationships
        """
        glyphs = []
        used_indices = set()
        
        for i in range(len(offbits)):
            if i in used_indices:
                continue
            
            # Find nearby OffBits for Glyph formation
            glyph_members = [i]
            base_pos = positions[i]
            
            # TGIC-based clustering: look for 3, 6, or 9 member groups
            for j in range(i + 1, len(offbits)):
                if j in used_indices:
                    continue
                
                # Calculate distance using φ-weighted metrics
                pos_j = positions[j]
                distance = math.sqrt(
                    (base_pos[0] - pos_j[0])**2 + 
                    (base_pos[1] - pos_j[1])**2 + 
                    (base_pos[2] - pos_j[2])**2
                )
                
                # φ-based clustering threshold
                threshold = self.phi * math.sqrt(len(glyph_members) + 1)
                
                if distance < threshold and len(glyph_members) < 9:  # TGIC max
                    glyph_members.append(j)
            
            # Only form Glyphs with TGIC-compliant sizes (3, 6, or 9)
            if len(glyph_members) in [3, 6, 9]:
                glyph_positions = [positions[idx] for idx in glyph_members]
                glyph_offbits = [offbits[idx] for idx in glyph_members]
                
                # Calculate Glyph coherence using φ-π relationships
                coherence = self.calculate_fibonacci_coherence(glyph_offbits, glyph_positions)
                
                glyph = {
                    'id': len(glyphs),
                    'members': glyph_members,
                    'size': len(glyph_members),
                    'positions': glyph_positions,
                    'coherence': coherence,
                    'tgic_compliant': True
                }
                
                glyphs.append(glyph)
                used_indices.update(glyph_members)
        
        return glyphs
    
    def calculate_fibonacci_coherence(self, glyph_offbits: List[Dict], positions: List[Tuple]) -> float:
        """
        Calculate Glyph coherence using Fibonacci-specific metrics
        Incorporates φ-π relationships
        """
        if len(positions) < 3:
            return 0.0
        
        # Calculate centroid
        centroid = (
            sum(pos[0] for pos in positions) / len(positions),
            sum(pos[1] for pos in positions) / len(positions),
            sum(pos[2] for pos in positions) / len(positions)
        )
        
        # Calculate distances from centroid
        distances = []
        for pos in positions:
            dist = math.sqrt(
                (pos[0] - centroid[0])**2 + 
                (pos[1] - centroid[1])**2 + 
                (pos[2] - centroid[2])**2
            )
            distances.append(dist)
        
        # Coherence based on φ-π relationship
        mean_distance = sum(distances) / len(distances)
        std_distance = math.sqrt(sum((d - mean_distance)**2 for d in distances) / len(distances))
        
        # φ-π coherence metric
        coherence = (mean_distance * self.phi) / (std_distance * self.pi + 1e-10)
        
        return coherence
    
    def calculate_s_phi_invariant(self, glyphs: List[Dict]) -> float:
        """
        Calculate S_φ invariant - the Fibonacci equivalent of S_π
        Should approach φ for valid UBP Fibonacci sequences
        """
        if not glyphs:
            return 0.0
        
        total_angles = 0.0
        angle_count = 0
        
        for glyph in glyphs:
            positions = glyph['positions']
            
            if len(positions) >= 3:
                # Calculate angles within the Glyph
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        for k in range(j + 1, len(positions)):
                            # Calculate angle at position j
                            p1, p2, p3 = positions[i], positions[j], positions[k]
                            
                            # Vectors from p2 to p1 and p2 to p3
                            v1 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
                            v2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])
                            
                            # Calculate angle using dot product
                            dot_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
                            mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
                            mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
                            
                            if mag_v1 > 0 and mag_v2 > 0:
                                cos_angle = dot_product / (mag_v1 * mag_v2)
                                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                                angle = math.acos(cos_angle)
                                
                                total_angles += angle
                                angle_count += 1
        
        if angle_count == 0:
            return 0.0
        
        # Calculate S_φ using φ-based geometric analysis
        mean_angle = total_angles / angle_count
        s_phi = mean_angle * self.phi / self.pi  # φ-π relationship
        
        return s_phi
    
    def calculate_phi_pi_resonance(self, offbits: List[Dict]) -> Dict:
        """
        Calculate resonance frequencies based on φ-π relationships
        """
        if len(offbits) < 2:
            return {'frequency': 0.0, 'phi_pi_ratio': 0.0}
        
        # Calculate sequence ratios (F_n+1 / F_n approaches φ)
        ratios = []
        for i in range(1, len(offbits)):
            if offbits[i-1]['fibonacci_number'] > 0:
                ratio = offbits[i]['fibonacci_number'] / offbits[i-1]['fibonacci_number']
                ratios.append(ratio)
        
        if not ratios:
            return {'frequency': 0.0, 'phi_pi_ratio': 0.0}
        
        # Calculate how close ratios are to φ
        phi_deviations = [abs(ratio - self.phi) for ratio in ratios]
        mean_deviation = sum(phi_deviations) / len(phi_deviations)
        
        # Resonance frequency based on φ-π relationship
        frequency = (self.phi / self.pi) / (mean_deviation + 1e-10)
        phi_pi_ratio = self.phi / self.pi
        
        return {
            'frequency': frequency,
            'phi_pi_ratio': phi_pi_ratio,
            'mean_ratio': sum(ratios) / len(ratios),
            'phi_convergence': 1.0 - (mean_deviation / self.phi)
        }
    
    def analyze_fibonacci_sequence(self, n_terms: int, save_results: bool = False) -> Dict:
        """
        Complete UBP analysis of Fibonacci sequence
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"UBP FIBONACCI SEQUENCE ANALYSIS")
        print(f"{'='*60}")
        print(f"Terms to analyze: {n_terms}")
        print(f"UBP Framework: {self.ubp_version}")
        print(f"TGIC Structure: {self.tgic_structure}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Generate Fibonacci sequence
        print(f"\nGenerating Fibonacci sequence...")
        fib_sequence = self.generate_fibonacci_sequence(n_terms)
        print(f"Sequence length: {len(fib_sequence)}")
        print(f"First 10 terms: {fib_sequence[:10]}")
        if len(fib_sequence) > 10:
            print(f"Last 5 terms: {fib_sequence[-5:]}")
        
        # Encode as OffBits
        print(f"\nCreating UBP OffBit sequence...")
        offbits = self.encode_fibonacci_offbits(fib_sequence)
        print(f"OffBits created: {len(offbits)}")
        
        # Calculate 3D positions
        print(f"Calculating φ-based geometric positions...")
        positions = self.calculate_3d_positions(offbits)
        
        # Form Glyphs
        print(f"Forming Glyphs using TGIC principles...")
        glyphs = self.form_fibonacci_glyphs(offbits, positions)
        print(f"Glyphs formed: {len(glyphs)}")
        
        # Calculate S_φ invariant
        print(f"Calculating S_φ invariant...")
        s_phi = self.calculate_s_phi_invariant(glyphs)
        
        # Calculate φ-π resonance
        print(f"Analyzing φ-π resonance...")
        resonance = self.calculate_phi_pi_resonance(offbits)
        
        computation_time = time.time() - start_time
        
        # Display results
        print(f"\n{'='*60}")
        print(f"UBP FIBONACCI ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"S_φ (Phi Invariant):     {s_phi:.6f}")
        print(f"Target (φ):              {self.phi:.6f}")
        print(f"Error:                   {abs(s_phi - self.phi):.6f}")
        print(f"Accuracy:                {(s_phi/self.phi)*100:.2f}%")
        print(f"")
        print(f"φ-π Resonance Analysis:")
        print(f"Resonance Frequency:     {resonance['frequency']:.6f} Hz")
        print(f"φ/π Ratio:               {resonance['phi_pi_ratio']:.6f}")
        print(f"Mean F_n+1/F_n Ratio:    {resonance['mean_ratio']:.6f}")
        print(f"φ Convergence:           {resonance['phi_convergence']*100:.2f}%")
        print(f"")
        print(f"UBP Framework Analysis:")
        print(f"OffBits Created:         {len(offbits)}")
        print(f"Glyphs Formed:           {len(glyphs)}")
        print(f"TGIC Structure:          {self.tgic_structure}")
        print(f"")
        print(f"Performance:")
        print(f"Computation Time:        {computation_time:.3f} seconds")
        print(f"Processing Rate:         {len(fib_sequence)/computation_time:.0f} terms/sec")
        
        # Prepare results dictionary
        results = {
            'input': {
                'n_terms': n_terms,
                'timestamp': datetime.now().isoformat(),
                'ubp_version': self.ubp_version
            },
            'fibonacci_sequence': {
                'length': len(fib_sequence),
                'first_10': fib_sequence[:10],
                'last_5': fib_sequence[-5:] if len(fib_sequence) > 5 else fib_sequence
            },
            'ubp_framework': {
                'offbits_created': len(offbits),
                'glyphs_formed': len(glyphs),
                'tgic_structure': self.tgic_structure,
                'computation_time': computation_time
            },
            'phi_analysis': {
                's_phi': s_phi,
                's_phi_target': self.phi,
                's_phi_error': abs(s_phi - self.phi),
                's_phi_accuracy': (s_phi/self.phi)*100 if self.phi != 0 else 0,
                'phi_invariant_valid': abs(s_phi - self.phi) < 0.1
            },
            'resonance_analysis': resonance,
            'validation': {
                'ubp_signature_valid': len(glyphs) > 0 and s_phi > 0,
                'phi_convergence_valid': resonance['phi_convergence'] > 0.8,
                'tgic_compliant': all(glyph['tgic_compliant'] for glyph in glyphs)
            }
        }
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ubp_fibonacci_results_{n_terms}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {filename}")
        
        return results

def main():
    """Test the UBP Fibonacci analyzer"""
    analyzer = UBPFibonacciAnalyzer()
    
    # Test with different sequence lengths
    test_cases = [10, 20, 30, 50]
    
    for n_terms in test_cases:
        results = analyzer.analyze_fibonacci_sequence(n_terms, save_results=True)
        print(f"\n" + "="*60)

if __name__ == "__main__":
    main()

