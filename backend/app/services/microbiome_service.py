"""
Microbiome Service for Signal Processing
Provides service layer for microbiome data analysis
"""

import pandas as pd
from typing import Dict, List, Optional
from app.models.microbiome_model import (
    MicrobiomeDataLoader,
    MicrobiomeAnalyzer,
    load_microbiome_data,
    analyze_microbiome,
    estimate_patient
)


class MicrobiomeService:
    """Service for microbiome data operations."""
    
    def __init__(self):
        self.data_loader = MicrobiomeDataLoader()
        self.analyzer = MicrobiomeAnalyzer()
        self.current_data = None
    
    def load_data(self, n_samples: int = 100) -> Dict:
        """
        Load microbiome data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary with microbiome data
        """
        try:
            result = load_microbiome_data(n_samples)
            self.current_data = result
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def get_sample(self, sample_id: str) -> Dict:
        """Get a specific sample by ID."""
        if self.current_data is None:
            # Load default data
            self.load_data()
        
        for sample in self.current_data.get('data', []):
            if sample.get('sample_id') == sample_id:
                return sample
        
        return {'error': 'Sample not found'}
    
    def get_diversity_analysis(self, sample_id: Optional[str] = None) -> Dict:
        """
        Get diversity analysis for samples.
        
        Args:
            sample_id: Optional specific sample ID
            
        Returns:
            Diversity analysis results
        """
        if self.current_data is None:
            self.load_data()
        
        diversity = self.current_data.get('diversity', [])
        
        if sample_id:
            for d in diversity:
                if d.get('sample_id') == sample_id:
                    return d
            return {'error': 'Sample not found'}
        
        return {
            'samples': diversity,
            'average_shannon': sum(d.get('shannon_index', 0) for d in diversity) / len(diversity) if diversity else 0,
            'average_simpson': sum(d.get('simpson_index', 0) for d in diversity) / len(diversity) if diversity else 0
        }
    
    def analyze_sample(self, sample_data: Dict) -> Dict:
        """
        Analyze a microbiome sample.
        
        Args:
            sample_data: Sample data with bacterial abundances
            
        Returns:
            Analysis results
        """
        try:
            return analyze_microbiome(sample_data)
        except Exception as e:
            return {'error': str(e)}
    
    def estimate_patient(self, sample_data: Dict) -> Dict:
        """
        Estimate patient profile.
        
        Args:
            sample_data: Sample data with microbiome and metadata
            
        Returns:
            Patient profile estimation
        """
        try:
            return estimate_patient(sample_data)
        except Exception as e:
            return {'error': str(e)}
    
    def compare_samples(self, sample1_id: str, sample2_id: str) -> Dict:
        """
        Compare two samples.
        
        Args:
            sample1_id: First sample ID
            sample2_id: Second sample ID
            
        Returns:
            Comparison results
        """
        if self.current_data is None:
            self.load_data()
        
        sample1 = None
        sample2 = None
        
        for s in self.current_data.get('data', []):
            if s.get('sample_id') == sample1_id:
                sample1 = s
            if s.get('sample_id') == sample2_id:
                sample2 = s
        
        if sample1 is None or sample2 is None:
            return {'error': 'One or both samples not found'}
        
        try:
            return self.analyzer.compare_samples(sample1, sample2)
        except Exception as e:
            return {'error': str(e)}
    
    def get_disease_profiles(self) -> Dict:
        """Get available disease profiles."""
        return self.data_loader.DISEASE_PROFILES
    
    def get_bacterial_taxonomy(self) -> Dict:
        """Get bacterial taxonomy information."""
        return {
            'phyla': self.data_loader.COMMON_PHYLA,
            'genera': self.data_loader.COMMON_GENERA
        }
    
    def get_statistics(self) -> Dict:
        """Get overall statistics about the microbiome data."""
        if self.current_data is None:
            self.load_data()
        
        data = self.current_data.get('data', [])
        
        if not data:
            return {'error': 'No data available'}
        
        # Calculate statistics
        ages = [s.get('age', 0) for s in data]
        bmis = [s.get('bmi', 0) for s in data]
        
        # Disease distribution
        disease_counts = {}
        for s in data:
            disease = s.get('disease_status', 'Unknown')
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        return {
            'n_samples': len(data),
            'n_subjects': len(set(s.get('subject_id', '') for s in data)),
            'age_range': [min(ages), max(ages)] if ages else [0, 0],
            'age_mean': sum(ages) / len(ages) if ages else 0,
            'bmi_range': [min(bmis), max(bmis)] if bmis else [0, 0],
            'bmi_mean': sum(bmis) / len(bmis) if bmis else 0,
            'disease_distribution': disease_counts
        }


# Singleton instance
microbiome_service = MicrobiomeService()


def get_microbiome_service() -> MicrobiomeService:
    """Get the microbiome service instance."""
    return microbiome_service
