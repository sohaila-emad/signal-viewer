"""
Microbiome Models for Signal Processing
- iHMP/iPOP dataset handling
- Bacterial profiling
- Disease profiling
- Patient profile estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter
import math
import random


class MicrobiomeDataLoader:
    """Load and process microbiome data from iHMP/iPOP datasets."""
    
    # Common bacterial phyla in human microbiome
    COMMON_PHYLA = [
        'Firmicutes', 'Bacteroidetes', 'Proteobacteria', 'Actinobacteria',
        'Verrucomicrobia', 'Fusobacteria', 'Cyanobacteria', 'TM7'
    ]
    
    # Common bacterial genera
    COMMON_GENERA = [
        'Bacteroides', 'Prevotella', 'Faecalibacterium', 'Bifidobacterium',
        'Lactobacillus', 'Escherichia', 'Streptococcus', 'Clostridium',
        'Ruminococcus', 'Akkermansia', 'Blautia', 'Roseburia'
    ]
    
    # Disease-related microbiome profiles
    DISEASE_PROFILES = {
        'IBD': {
            'description': 'Inflammatory Bowel Disease',
            'decreased': ['Faecalibacterium', 'Roseburia', 'Bifidobacterium'],
            'increased': ['Escherichia', 'Streptococcus', 'Ruminococcus']
        },
        'Type2Diabetes': {
            'description': 'Type 2 Diabetes',
            'decreased': ['Akkermansia', 'Faecalibacterium'],
            'increased': ['Bacteroides', 'Ruminococcus']
        },
        'ColorectalCancer': {
            'description': 'Colorectal Cancer',
            'decreased': ['Lactobacillus', 'Bifidobacterium'],
            'increased': ['Escherichia', 'Fusobacterium']
        },
        'Obesity': {
            'description': 'Obesity',
            'decreased': ['Bacteroidetes', 'Akkermansia'],
            'increased': ['Firmicutes', 'Escherichia']
        },
        'CVD': {
            'description': 'Cardiovascular Disease',
            'decreased': ['Faecalibacterium', 'Roseburia'],
            'increased': ['Bacteroides', 'Clostridium']
        }
    }
    
    def __init__(self):
        self.data = None
        self.sample_metadata = None
    
    def generate_sample_data(self, n_samples: int = 100, 
                           include_disease: bool = False) -> pd.DataFrame:
        """
        Generate sample microbiome data for demonstration.
        
        Args:
            n_samples: Number of samples to generate
            include_disease: Whether to include disease labels
            
        Returns:
            DataFrame with microbiome profiles
        """
        np.random.seed(42)
        random.seed(42)
        
        data = {
            'sample_id': [f'Sample_{i:04d}' for i in range(n_samples)],
            'subject_id': [f'Subject_{i%20:03d}' for i in range(n_samples)],
            'age': np.random.randint(20, 80, n_samples),
            'bmi': np.random.uniform(18, 40, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples)
        }
        
        # Generate relative abundances for each phylum
        for phylum in self.COMMON_PHYLA:
            data[phylum] = np.random.dirichlet(np.ones(len(self.COMMON_PHYLA)) * 2, n_samples)[:, 
                                  self.COMMON_PHYLA.index(phylum)]
        
        # Generate relative abundances for each genus
        for genus in self.COMMON_GENERA:
            data[genus] = np.random.dirichlet(np.ones(len(self.COMMON_GENERA)) * 2, n_samples)[:, 
                                self.COMMON_GENERA.index(genus)]
        
        # Add disease labels if requested
        if include_disease:
            diseases = list(self.DISEASE_PROFILES.keys())
            data['disease_status'] = np.random.choice(['Healthy'] + diseases, n_samples, 
                                                        p=[0.6] + [0.4/len(diseases)] * len(diseases))
        
        df = pd.DataFrame(data)
        
        # Normalize abundances to sum to 1
        phylum_cols = self.COMMON_PHYLA
        genus_cols = self.COMMON_GENERA
        
        df[phylum_cols] = df[phylum_cols].div(df[phylum_cols].sum(axis=1), axis=0)
        df[genus_cols] = df[genus_cols].div(df[genus_cols].sum(axis=1), axis=0)
        
        return df
    
    def load_ihmp_data(self, data_type: str = 'sample') -> pd.DataFrame:
        """
        Load iHMP data (simulated for demo purposes).
        
        Args:
            data_type: Type of data ('sample', 'metagenomics', 'metatranscriptomics')
            
        Returns:
            DataFrame with microbiome data
        """
        # In a real implementation, this would load actual iHMP data
        # For now, generate sample data
        return self.generate_sample_data(n_samples=200, include_disease=True)
    
    def calculate_diversity_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate diversity indices for each sample.
        
        Args:
            df: DataFrame with microbiome abundance data
            
        Returns:
            DataFrame with diversity indices
        """
        results = []
        
        for idx, row in df.iterrows():
            # Get abundance values (non-phylum/genera columns)
            abundances = row[self.COMMON_GENERA].values
            
            # Filter out zeros
            abundances = abundances[abundances > 0]
            
            if len(abundances) == 0:
                results.append({
                    'sample_id': row['sample_id'],
                    'shannon_index': 0,
                    'simpson_index': 0,
                    'chao1': 0,
                    'observed_species': 0
                })
                continue
            
            # Shannon index
            shannon = -sum(a * math.log(a) for a in abundances if a > 0)
            
            # Simpson index
            simpson = sum(a ** 2 for a in abundances)
            
            # Chao1 (estimated species richness)
            n = sum(abundances > 0)
            f1 = sum(1 for a in abundances if 0 < a < 0.01)  # singletons
            f2 = sum(1 for a in abundances if 0.01 <= a < 0.02)  # doubletons
            chao1 = n + (f1 * (f1 - 1)) / (2 * (f2 + 1)) if f2 > 0 else n + f1 * (f1 - 1) / 2
            
            results.append({
                'sample_id': row['sample_id'],
                'shannon_index': round(shannon, 4),
                'simpson_index': round(1 - simpson, 4),
                'chao1': round(chao1, 2),
                'observed_species': n
            })
        
        return pd.DataFrame(results)


class MicrobiomeAnalyzer:
    """Analyze microbiome data for disease profiling and patient estimation."""
    
    def __init__(self):
        self.data_loader = MicrobiomeDataLoader()
        self.disease_profiles = self.data_loader.DISEASE_PROFILES
    
    def analyze_bacterial_profile(self, abundances: Dict[str, float]) -> Dict:
        """
        Analyze bacterial profile and compare to healthy reference.
        
        Args:
            abundances: Dictionary of bacterial genus abundances
            
        Returns:
            Analysis results
        """
        # Get top genera
        sorted_abundances = sorted(abundances.items(), key=lambda x: x[1], reverse=True)
        top_genera = sorted_abundances[:10]
        
        # Calculate profile characteristics
        Firmicutes_Bacteroidetes_ratio = (
            abundances.get('Firmicutes', 0) / max(abundances.get('Bacteroidetes', 0.001), 0.001)
        )
        
        return {
            'top_genera': [{'genus': g, 'abundance': round(a, 4)} for g, a in top_genera],
            'total_abundance': sum(abundances.values()),
            'F_B_ratio': round(Firmicutes_Bacteroidetes_ratio, 4),
            'diversity': self._calculate_diversity(abundances)
        }
    
    def _calculate_diversity(self, abundances: Dict[str, float]) -> float:
        """Calculate Shannon diversity index."""
        values = [v for v in abundances.values() if v > 0]
        if not values:
            return 0
        return -sum(v * math.log(v) for v in values)
    
    def detect_disease_risk(self, abundances: Dict[str, float]) -> Dict:
        """
        Detect potential disease risks based on microbiome profile.
        
        Args:
            abundances: Dictionary of bacterial genus abundances
            
        Returns:
            Risk assessment results
        """
        risks = {}
        
        for disease, profile in self.disease_profiles.items():
            score = 0
            details = []
            
            # Check for decreased beneficial bacteria
            for genus in profile['decreased']:
                if genus in abundances:
                    abundance = abundances[genus]
                    if abundance < 0.05:  # Threshold
                        score += 1
                        details.append(f'{genus} is decreased')
            
            # Check for increased potentially harmful bacteria
            for genus in profile['increased']:
                if genus in abundances:
                    abundance = abundances[genus]
                    if abundance > 0.1:  # Threshold
                        score += 1
                        details.append(f'{genus} is increased')
            
            # Calculate risk level
            if score >= 3:
                risk_level = 'High'
            elif score >= 1:
                risk_level = 'Moderate'
            else:
                risk_level = 'Low'
            
            risks[disease] = {
                'risk_level': risk_level,
                'score': score,
                'details': details,
                'description': profile['description']
            }
        
        # Determine overall risk
        high_risk = sum(1 for r in risks.values() if r['risk_level'] == 'High')
        moderate_risk = sum(1 for r in risks.values() if r['risk_level'] == 'Moderate')
        
        if high_risk > 0:
            overall_risk = 'High'
        elif moderate_risk > 0:
            overall_risk = 'Moderate'
        else:
            overall_risk = 'Low'
        
        return {
            'overall_risk': overall_risk,
            'diseases': risks
        }
    
    def estimate_patient_profile(self, sample_data: Dict) -> Dict:
        """
        Estimate patient profile based on microbiome and metadata.
        
        Args:
            sample_data: Dictionary with microbiome abundances and metadata
            
        Returns:
            Patient profile estimation
        """
        # Extract features
        abundances = {k: v for k, v in sample_data.items() 
                     if k in self.data_loader.COMMON_GENERA}
        
        age = sample_data.get('age', 50)
        bmi = sample_data.get('bmi', 25)
        
        # Analyze bacterial profile
        profile_analysis = self.analyze_bacterial_profile(abundances)
        
        # Detect disease risks
        disease_risks = self.detect_disease_risk(abundances)
        
        # Generate recommendations
        recommendations = []
        
        if profile_analysis.get('F_B_ratio', 1) > 3:
            recommendations.append('Consider increasing fiber intake to balance gut bacteria')
        
        if profile_analysis.get('diversity', 0) < 2:
            recommendations.append('Low diversity detected. Consider probiotic supplementation')
        
        for disease, risk in disease_risks['diseases'].items():
            if risk['risk_level'] == 'High':
                recommendations.append(f'High {risk["description"]} risk detected. Consult healthcare provider')
        
        if not recommendations:
            recommendations.append('Maintain current diet and lifestyle for gut health')
        
        return {
            'patient_profile': {
                'age_group': self._get_age_group(age),
                'bmi_category': self._get_bmi_category(bmi),
                'gut_health_score': round(profile_analysis['diversity'] / 4 * 100, 1),  # Normalize to 100
                'microbial_diversity': round(profile_analysis['diversity'], 2)
            },
            'disease_risks': disease_risks,
            'recommendations': recommendations,
            'bacterial_profile': profile_analysis
        }
    
    def _get_age_group(self, age: int) -> str:
        """Get age group category."""
        if age < 30:
            return 'Young Adult (20-29)'
        elif age < 45:
            return 'Middle Adult (30-44)'
        elif age < 60:
            return 'Senior Adult (45-59)'
        else:
            return 'Elderly (60+)'
    
    def _get_bmi_category(self, bmi: float) -> str:
        """Get BMI category."""
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    def compare_samples(self, sample1: Dict, sample2: Dict) -> Dict:
        """
        Compare two microbiome samples.
        
        Args:
            sample1: First sample data
            sample2: Second sample data
            
        Returns:
            Comparison results
        """
        genera = self.data_loader.COMMON_GENERA
        
        abundances1 = {k: sample1.get(k, 0) for k in genera}
        abundances2 = {k: sample2.get(k, 0) for k in genera}
        
        differences = {}
        for genus in genera:
            diff = abundances1.get(genus, 0) - abundances2.get(genus, 0)
            differences[genus] = round(diff, 4)
        
        # Sort by absolute difference
        sorted_diff = sorted(differences.items(), key=lambda x: abs(x[1]), reverse=True)
        
        diversity1 = self._calculate_diversity(abundances1)
        diversity2 = self._calculate_diversity(abundances2)
        
        return {
            'sample1_id': sample1.get('sample_id', 'Unknown'),
            'sample2_id': sample2.get('sample_id', 'Unknown'),
            'diversity_comparison': {
                'sample1_diversity': round(diversity1, 2),
                'sample2_diversity': round(diversity2, 2),
                'difference': round(diversity1 - diversity2, 2)
            },
            'top_differences': [{'genus': g, 'difference': d} for g, d in sorted_diff[:5]]
        }


def load_microbiome_data(n_samples: int = 100) -> dict:
    """Load microbiome data and return as dictionary."""
    loader = MicrobiomeDataLoader()
    df = loader.generate_sample_data(n_samples, include_disease=True)
    
    # Calculate diversity indices
    diversity_df = loader.calculate_diversity_indices(df)
    
    return {
        'data': df.to_dict(orient='records'),
        'diversity': diversity_df.to_dict(orient='records'),
        'columns': df.columns.tolist(),
        'n_samples': len(df)
    }


def analyze_microbiome(sample_data: dict) -> dict:
    """Analyze a microbiome sample."""
    analyzer = MicrobiomeAnalyzer()
    
    # Extract abundances
    abundances = {k: v for k, v in sample_data.items() 
                if k in MicrobiomeDataLoader.COMMON_GENERA}
    
    profile = analyzer.analyze_bacterial_profile(abundances)
    risks = analyzer.detect_disease_risk(abundances)
    
    return {
        'bacterial_profile': profile,
        'disease_risks': risks
    }


def estimate_patient(sample_data: dict) -> dict:
    """Estimate patient profile."""
    analyzer = MicrobiomeAnalyzer()
    return analyzer.estimate_patient_profile(sample_data)
