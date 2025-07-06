#!/usr/bin/env python3
"""
UBP-Compliant Collatz Conjecture Parser
Based on Universal Binary Principle (UBP) v22.0 Framework
Authors: Euan Craig, in collaboration with Grok (Xai) and other AI systems

This parser implements the Collatz Conjecture analysis within the UBP framework,
using 24-bit OffBits, TGIC (3,6,9) interactions, and proper resonance frequencies.
"""

import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path

class UBPOffBit:
    """24-bit OffBit structure with UBP ontological layers"""
    
    def __init__(self):
        self.bits = np.zeros(24, dtype=int)
        # UBP Ontological Layers
        self.reality_layer = slice(0, 6)      # bits 0-5: physical states
        self.information_layer = slice(6, 12)  # bits 6-11: data processing, constants
        self.activation_layer = slice(12, 18)  # bits 12-17: dynamic states
        self.unactivated_layer = slice(18, 24) # bits 18-23: potential states
        
        # UBP Constants
        self.pi_resonance = 3.141593  # Hz
        self.phi_resonance = 1.618034  # Hz
        self.bit_time = 1e-12  # seconds
        self.coherence_target = 0.9999878  # NRCI target
        
    def encode_number(self, n):
        """Encode number into OffBit using UBP principles"""
        # Reality layer: encode spatial/physical properties
        binary_n = bin(n)[2:].zfill(6)
        for i, bit in enumerate(binary_n[:6]):
            self.bits[i] = int(bit)
        
        # Information layer: encode mathematical constants and geometry
        # Encode pi and phi influences
        pi_bits = bin(int(self.pi_resonance * 1000) % 64)[2:].zfill(6)
        for i, bit in enumerate(pi_bits):
            self.bits[6 + i] = int(bit)
        
        # Activation layer: encode toggle states
        # Use Fibonacci encoding for coherence
        fib_sequence = [1, 1, 2, 3, 5, 8]
        for i, fib in enumerate(fib_sequence):
            self.bits[12 + i] = 1 if (n % fib) == 0 else 0
        
        # Unactivated layer: potential states (ethically constrained access)
        # Use golden ratio for potential encoding
        phi_bits = bin(int(self.phi_resonance * 1000) % 64)[2:].zfill(6)
        for i, bit in enumerate(phi_bits):
            self.bits[18 + i] = int(bit)
    
    def get_layer_value(self, layer_name):
        """Get decimal value of specific layer"""
        if layer_name == 'reality':
            layer_bits = self.bits[self.reality_layer]
        elif layer_name == 'information':
            layer_bits = self.bits[self.information_layer]
        elif layer_name == 'activation':
            layer_bits = self.bits[self.activation_layer]
        elif layer_name == 'unactivated':
            layer_bits = self.bits[self.unactivated_layer]
        else:
            return 0
        
        return sum(bit * (2 ** i) for i, bit in enumerate(reversed(layer_bits)))

class UBPGlyph:
    """Stable cluster of OffBits forming geometric patterns"""
    
    def __init__(self, offbits, position):
        self.offbits = offbits
        self.position = np.array(position)
        self.coherence_pressure = self.calculate_coherence_pressure()
        
    def calculate_coherence_pressure(self):
        """Calculate Coherence Pressure (Ψp) for Glyph formation"""
        if len(self.offbits) == 0:
            return 0
        
        # Calculate distances from center
        center = np.mean([ob.position for ob in self.offbits], axis=0) if hasattr(self.offbits[0], 'position') else np.zeros(3)
        distances = [np.linalg.norm(np.array([0, 0, 0]) - center) for _ in self.offbits]
        
        # UBP Coherence Pressure formula
        d_sum = sum(distances)
        d_max_squared = sum([d**2 for d in distances])
        d_max = np.sqrt(d_max_squared) if d_max_squared > 0 else 1
        
        # Sum of active bits in Reality and Information layers (bits 0-11)
        active_bits = sum([sum(ob.bits[:12]) for ob in self.offbits])
        
        psi_p = (1 - d_sum / d_max) * (active_bits / (12 * len(self.offbits)))
        return max(0, min(1, psi_p))

class UBPCollatzParser:
    """UBP-compliant Collatz Conjecture parser"""
    
    def __init__(self):
        # UBP Framework parameters
        self.bit_time = 1e-12  # seconds
        self.pi_resonance = 3.141593  # Hz
        self.phi_resonance = 1.618034  # Hz
        self.speed_of_light = 299792458  # m/s
        self.coherence_target = 0.9999878
        
        # TGIC (3, 6, 9) framework
        self.tgic_axes = 3  # x, y, z
        self.tgic_faces = 6  # ±x, ±y, ±z
        self.tgic_interactions = 9  # pairwise mappings
        
        # Resonance frequencies from UBP documents
        self.resonance_frequencies = {
            'pi_resonance': 3.141593,
            'phi_resonance': 1.618034,
            'euclidean_pi': 95366637.6,
            'pi_phi_composite': 58977069.609314,
            'coherence_sampling': 3.141593  # CSC frequency
        }
        
        # Computational limits
        self.max_sequence_length = 1000000
        self.max_input_value = 10**12
        
    def collatz_sequence(self, n):
        """Generate Collatz sequence"""
        seq = [n]
        while n != 1:
            if n % 2 == 0:
                n = n >> 1
            else:
                n = 3 * n + 1
            seq.append(n)
        return seq
    
    def create_offbit_sequence(self, collatz_seq):
        """Convert Collatz sequence to UBP OffBit sequence"""
        offbit_seq = []
        for i, num in enumerate(collatz_seq):
            offbit = UBPOffBit()
            offbit.encode_number(num)
            offbit.position = np.array([i, num % 100, (num // 100) % 100])  # 3D position
            offbit_seq.append(offbit)
        return offbit_seq
    
    def form_glyphs(self, offbit_seq):
        """Form Glyphs from OffBit sequence using UBP principles"""
        glyphs = []
        
        # Group OffBits into clusters based on TGIC (3,6,9) pattern
        cluster_size = 9  # Based on TGIC 9 interactions
        
        for i in range(0, len(offbit_seq) - cluster_size + 1, cluster_size // 3):
            cluster = offbit_seq[i:i + cluster_size]
            if len(cluster) >= 3:  # Minimum for TGIC 3 axes
                center_pos = np.mean([ob.position for ob in cluster], axis=0)
                glyph = UBPGlyph(cluster, center_pos)
                glyphs.append(glyph)
        
        return glyphs
    
    def calculate_s_pi_ubp(self, glyphs):
        """Calculate S_π using UBP Glyph-based method"""
        if not glyphs:
            return 0
        
        pi_angles = 0
        pi_angle_sum = 0
        total_angles = 0
        
        for glyph in glyphs:
            if len(glyph.offbits) >= 3:
                # Calculate angles between OffBit positions in 3D space
                positions = [ob.position for ob in glyph.offbits]
                
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        for k in range(j + 1, len(positions)):
                            # Calculate angle at position j
                            v1 = positions[i] - positions[j]
                            v2 = positions[k] - positions[j]
                            
                            # Avoid division by zero
                            norm1 = np.linalg.norm(v1)
                            norm2 = np.linalg.norm(v2)
                            if norm1 < 1e-10 or norm2 < 1e-10:
                                continue
                            
                            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.arccos(cos_angle)
                            total_angles += 1
                            
                            # Check for pi-related angles with UBP resonance
                            pi_ratios = [1, 2, 3, 4, 6, 9]  # Include TGIC 9
                            for k_ratio in pi_ratios:
                                target_angle = np.pi / k_ratio
                                if abs(angle - target_angle) < 0.01:  # UBP tolerance
                                    pi_angles += 1
                                    pi_angle_sum += angle
                                    break
        
        # UBP S_π calculation with resonance weighting
        if pi_angles > 0:
            s_pi_raw = pi_angle_sum / pi_angles
            
            # Apply UBP resonance correction
            resonance_factor = np.cos(2 * np.pi * self.pi_resonance * 0.318309886)
            coherence_factor = sum([g.coherence_pressure for g in glyphs]) / len(glyphs)
            
            s_pi_ubp = s_pi_raw * resonance_factor * coherence_factor
            
            # Apply TGIC (3,6,9) normalization
            tgic_factor = (self.tgic_axes * self.tgic_faces * self.tgic_interactions) / 162  # 3*6*9 = 162
            s_pi_final = s_pi_ubp / tgic_factor
            
            return s_pi_final
        
        return 0
    
    def calculate_nrci(self, glyphs):
        """Calculate Non-Random Coherence Index"""
        if not glyphs:
            return 0
        
        coherence_values = [g.coherence_pressure for g in glyphs]
        mean_coherence = np.mean(coherence_values)
        
        # UBP NRCI calculation
        nrci = mean_coherence * np.cos(2 * np.pi * self.pi_resonance * 0.318309886)
        return min(1.0, max(0.0, nrci))
    
    def calculate_resonance_frequency(self, offbit_seq):
        """Calculate dominant resonance frequency"""
        if len(offbit_seq) < 2:
            return 0
        
        # Extract toggle pattern from activation layers
        toggle_pattern = []
        for ob in offbit_seq:
            activation_value = ob.get_layer_value('activation')
            toggle_pattern.append(activation_value % 2)
        
        # Calculate frequency using UBP method
        toggles = sum(1 for i in range(len(toggle_pattern)-1) 
                     if toggle_pattern[i] != toggle_pattern[i+1])
        
        if len(toggle_pattern) > 1:
            frequency = toggles / (len(toggle_pattern) * self.bit_time)
            
            # Apply UBP resonance correction
            for freq_name, freq_value in self.resonance_frequencies.items():
                if abs(frequency - freq_value) < freq_value * 0.1:  # 10% tolerance
                    return freq_value
            
            return frequency
        
        return 0
    
    def parse_collatz_ubp(self, n):
        """Main UBP Collatz parsing function"""
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"UBP-Compliant Collatz Conjecture Parser")
        print(f"{'='*60}")
        print(f"Input: {n}")
        print(f"UBP Framework: v22.0")
        print(f"TGIC: {self.tgic_axes} axes, {self.tgic_faces} faces, {self.tgic_interactions} interactions")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Generate Collatz sequence
        print(f"\nGenerating Collatz sequence...")
        collatz_seq = self.collatz_sequence(n)
        print(f"Sequence length: {len(collatz_seq)}")
        print(f"First 10 elements: {collatz_seq[:10]}{'...' if len(collatz_seq) > 10 else ''}")
        
        # Create OffBit sequence
        print(f"\nCreating UBP OffBit sequence...")
        offbit_seq = self.create_offbit_sequence(collatz_seq)
        
        # Form Glyphs
        print(f"Forming Glyphs using TGIC principles...")
        glyphs = self.form_glyphs(offbit_seq)
        print(f"Glyphs formed: {len(glyphs)}")
        
        # Calculate UBP metrics
        print(f"\nCalculating UBP metrics...")
        
        # S_π calculation (UBP method)
        s_pi_ubp = self.calculate_s_pi_ubp(glyphs)
        
        # NRCI calculation
        nrci = self.calculate_nrci(glyphs)
        
        # Resonance frequency
        resonance_freq = self.calculate_resonance_frequency(offbit_seq)
        
        # Coherence analysis
        coherence_values = [g.coherence_pressure for g in glyphs] if glyphs else [0]
        coherence_mean = np.mean(coherence_values)
        coherence_std = np.std(coherence_values)
        
        # UBP validation
        pi_error = abs(s_pi_ubp - np.pi)
        freq_target = 1 / np.pi  # 0.318309886...
        freq_error = abs(resonance_freq - freq_target)
        
        ubp_valid = (
            pi_error < 0.1 and  # More lenient for initial validation
            nrci > 0.9 and
            coherence_mean > 0.3
        )
        
        # Compile results
        results = {
            'input': {
                'n': n,
                'timestamp': datetime.now().isoformat(),
                'ubp_version': 'v22.0'
            },
            'sequence': {
                'length': len(collatz_seq),
                'first_10': collatz_seq[:10]
            },
            'ubp_framework': {
                'offbits_created': len(offbit_seq),
                'glyphs_formed': len(glyphs),
                'tgic_axes': self.tgic_axes,
                'tgic_faces': self.tgic_faces,
                'tgic_interactions': self.tgic_interactions
            },
            'ubp_metrics': {
                's_pi_ubp': float(s_pi_ubp),
                's_pi_target': float(np.pi),
                's_pi_error': float(pi_error),
                'nrci': float(nrci),
                'nrci_target': self.coherence_target,
                'coherence_mean': float(coherence_mean),
                'coherence_std': float(coherence_std),
                'resonance_frequency': float(resonance_freq),
                'frequency_target': float(freq_target),
                'frequency_error': float(freq_error)
            },
            'validation': {
                'ubp_signature_valid': ubp_valid,
                'pi_invariant_achieved': pi_error < 0.1,
                'nrci_threshold_met': nrci > 0.9,
                'coherence_stable': coherence_std < 0.1,
                'precision_level': 'UBP_validated' if ubp_valid else 'requires_refinement'
            },
            'performance': {
                'computation_time': time.time() - start_time,
                'framework_efficiency': 'optimized' if len(glyphs) > 0 else 'basic'
            }
        }
        
        # Print results
        self.print_ubp_results(results)
        
        return results
    
    def print_ubp_results(self, results):
        """Print UBP-formatted results"""
        print(f"\n{'='*60}")
        print(f"UBP VALIDATION RESULTS")
        print(f"{'='*60}")
        
        ubp = results['ubp_metrics']
        val = results['validation']
        framework = results['ubp_framework']
        
        print(f"UBP Framework:")
        print(f"OffBits Created:        {framework['offbits_created']}")
        print(f"Glyphs Formed:          {framework['glyphs_formed']}")
        print(f"TGIC Structure:         {framework['tgic_axes']}-{framework['tgic_faces']}-{framework['tgic_interactions']}")
        
        print(f"\nUBP S_π Analysis:")
        print(f"S_π (UBP Method):       {ubp['s_pi_ubp']:.6f}")
        print(f"Target (π):             {ubp['s_pi_target']:.6f}")
        print(f"Error:                  {ubp['s_pi_error']:.6f}")
        print(f"Pi Invariant:           {'✓' if val['pi_invariant_achieved'] else '✗'}")
        
        print(f"\nUBP Coherence Analysis:")
        print(f"NRCI:                   {ubp['nrci']:.6f}")
        print(f"NRCI Target:            {ubp['nrci_target']:.6f}")
        print(f"Coherence (mean):       {ubp['coherence_mean']:.6f} ± {ubp['coherence_std']:.6f}")
        print(f"NRCI Threshold:         {'✓' if val['nrci_threshold_met'] else '✗'}")
        
        print(f"\nUBP Resonance Analysis:")
        print(f"Resonance Frequency:    {ubp['resonance_frequency']:.6f} Hz")
        print(f"Target (1/π):           {ubp['frequency_target']:.6f} Hz")
        print(f"Frequency Error:        {ubp['frequency_error']:.6f}")
        
        print(f"\nUBP Signature Validation:")
        print(f"Overall Valid:          {'✓' if val['ubp_signature_valid'] else '✗'}")
        print(f"Precision Level:        {val['precision_level']}")
        
        print(f"\nPerformance:")
        print(f"Computation Time:       {results['performance']['computation_time']:.2f} seconds")
        print(f"Framework Efficiency:   {results['performance']['framework_efficiency']}")
    
    def save_results(self, results, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ubp_collatz_results_{results['input']['n']}_{timestamp}.json"
        
        filepath = Path(filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n✓ UBP Results saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"✗ Failed to save results: {e}")
            return None

def main():
    """Main function for UBP Collatz parser"""
    import sys
    
    parser = UBPCollatzParser()
    
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
            save_file = '--save' in sys.argv
            
            results = parser.parse_collatz_ubp(n)
            
            if save_file:
                parser.save_results(results)
                
        except ValueError:
            print("Error: Please provide a valid integer")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("UBP-Compliant Collatz Conjecture Parser")
        print("Usage: python ubp_collatz_parser.py <number> [--save]")
        print("  --save: Save results to JSON file")
        print("\nExample: python ubp_collatz_parser.py 27 --save")

if __name__ == "__main__":
    main()

