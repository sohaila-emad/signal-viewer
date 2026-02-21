"""
Microbiome Models for Signal Processing
- iHMP/iPOP dataset handling
- Bacterial profiling
- Disease profiling
- Patient profile estimation
- ML Classification Models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import math
import random
import warnings
warnings.filterwarnings('ignore')

# ML imports - with fallback for when sklearn is not available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


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
        'Ruminococcus', 'Akkermansia', 'Blautburia'
    ]
    
    # Disease-related microbiome profilesia', 'Rose
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
                           include_disease: bool = False,
                           disease_bias: bool = True) -> pd.DataFrame:
        """
        Generate sample microbiome data for demonstration.
        
        Args:
            n_samples: Number of samples to generate
            include_disease: Whether to include disease labels
            disease_bias: If True, create disease-specific microbiome patterns
            
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
        
        # Generate disease labels first if needed
        diseases = list(self.DISEASE_PROFILES.keys())
        if include_disease:
            disease_status = np.random.choice(['Healthy'] + diseases, n_samples, 
                                            p=[0.6] + [0.4/len(diseases)] * len(diseases))
        else:
            disease_status = ['Healthy'] * n_samples
        
        # Generate relative abundances with disease-specific patterns
        for i, disease in enumerate(disease_status):
            # Base abundances
            base_alphas = np.ones(len(self.COMMON_GENERA)) * 2
            
            # Apply disease-specific modifications
            if disease != 'Healthy' and disease_bias:
                profile = self.DISEASE_PROFILES.get(disease, {})
                
                # Increase disease-associated bacteria
                for increased in profile.get('increased', []):
                    if increased in self.COMMON_GENERA:
                        idx = self.COMMON_GENERA.index(increased)
                        base_alphas[idx] += 5
                
                # Decrease beneficial bacteria
                for decreased in profile.get('decreased', []):
                    if decreased in self.COMMON_GENERA:
                        idx = self.COMMON_GENERA.index(decreased)
                        base_alphas[idx] = max(0.5, base_alphas[idx] - 2)
            
            # Generate Dirichlet-distributed abundances
            abundances = np.random.dirichlet(base_alphas)
            for j, genus in enumerate(self.COMMON_GENERA):
                if genus not in data:
                    data[genus] = np.zeros(n_samples)
                data[genus][i] = abundances[j]
        
        # Add disease status
        if include_disease:
            data['disease_status'] = disease_status
        
        df = pd.DataFrame(data)
        
        # Generate phylum data based on genera (simplified mapping)
        phylum_mapping = {
            'Firmicutes': ['Faecalibacterium', 'Lactobacillus', 'Clostridium', 'Ruminococcus', 'Blautia', 'Roseburia'],
            'Bacteroidetes': ['Bacteroides', 'Prevotella'],
            'Proteobacteria': ['Escherichia'],
            'Actinobacteria': ['Bifidobacterium'],
            'Verrucomicrobia': ['Akkermansia'],
        }
        
        for phylum, genera_list in phylum_mapping.items():
            if phylum in self.COMMON_PHYLA:
                df[phylum] = sum(df.get(g, pd.Series([0]*n_samples)) for g in genera_list if g in df.columns)
        
        # Normalize abundances
        genus_cols = self.COMMON_GENERA
        df[genus_cols] = df[genus_cols].div(df[genus_cols].sum(axis=1), axis=0)
        
        return df
    
    def load_ihmp_data(self, data_type: str = 'sample') -> pd.DataFrame:
        """Load iHMP data (simulated for demo purposes)."""
        return self.generate_sample_data(n_samples=200, include_disease=True)
    
    def calculate_diversity_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate diversity indices for each sample."""
        results = []
        
        for idx, row in df.iterrows():
            abundances = row[self.COMMON_GENERA].values
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
            
            # Chao1
            n = sum(abundances > 0)
            f1 = sum(1 for a in abundances if 0 < a < 0.01)
            f2 = sum(1 for a in abundances if 0.01 <= a < 0.02)
            chao1 = n + (f1 * (f1 - 1)) / (2 * (f2 + 1)) if f2 > 0 else n + f1 * (f1 - 1) / 2
            
            results.append({
                'sample_id': row['sample_id'],
                'shannon_index': round(shannon, 4),
                'simpson_index': round(1 - simpson, 4),
                'chao1': round(chao1, 2),
                'observed_species': n
            })
        
        return pd.DataFrame(results)


class MicrobiomeMLClassifier:
    """
    ML Classifier for microbiome-based disease prediction.
    Uses Random Forest, Logistic Regression, and SVM.
    """
    
    def __init__(self):
        self.data_loader = MicrobiomeDataLoader()
        self.genera = self.data_loader.COMMON_GENERA
        self.diseases = list(self.data_loader.DISEASE_PROFILES.keys()) + ['Healthy']
        
        # Models
        self.rf_model = None
        self.lr_model = None
        self.svm_model = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = None
        
        # Training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.is_trained = False
    
    def prepare_data(self, n_samples: int = 500, test_size: float = 0.2) -> Dict:
        """Prepare training and test data."""
        # Generate data with disease patterns
        df = self.data_loader.generate_sample_data(n_samples, include_disease=True, disease_bias=True)
        
        # Extract features (bacterial abundances)
        X = df[self.genera].values
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['disease_status'].values)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return {
            'n_train': len(self.X_train),
            'n_test': len(self.X_test),
            'n_features': X.shape[1],
            'n_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_)
        }
    
    def train_models(self) -> Dict:
        """Train all classification models."""
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn is not available'}
        
        results = {}
        
        # Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        
        rf_pred = self.rf_model.predict(self.X_test_scaled)
        rf_acc = accuracy_score(self.y_test, rf_pred)
        rf_cv = cross_val_score(self.rf_model, self.X_train_scaled, self.y_train, cv=5).mean()
        
        results['random_forest'] = {
            'accuracy': round(rf_acc, 4),
            'cv_score': round(rf_cv, 4),
            'feature_importance': self._get_feature_importance()
        }
        
        # Logistic Regression
        self.lr_model = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            random_state=42
        )
        self.lr_model.fit(self.X_train_scaled, self.y_train)
        
        lr_pred = self.lr_model.predict(self.X_test_scaled)
        lr_acc = accuracy_score(self.y_test, lr_pred)
        lr_cv = cross_val_score(self.lr_model, self.X_train_scaled, self.y_train, cv=5).mean()
        
        results['logistic_regression'] = {
            'accuracy': round(lr_acc, 4),
            'cv_score': round(lr_cv, 4)
        }
        
        # SVM
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(self.X_train_scaled, self.y_train)
        
        svm_pred = self.svm_model.predict(self.X_test_scaled)
        svm_acc = accuracy_score(self.y_test, svm_pred)
        svm_cv = cross_val_score(self.svm_model, self.X_train_scaled, self.y_train, cv=5).mean()
        
        results['svm'] = {
            'accuracy': round(svm_acc, 4),
            'cv_score': round(svm_cv, 4)
        }
        
        self.is_trained = True
        
        # Apply PCA for visualization
        self.pca = PCA(n_components=2)
        self.pca_result = self.pca.fit_transform(self.X_train_scaled)
        
        results['pca'] = {
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'components': self.pca.components_.tolist()
        }
        
        return results
    
    def _get_feature_importance(self) -> List[Dict]:
        """Get feature importance from Random Forest."""
        if self.rf_model is None:
            return []
        
        importances = self.rf_model.feature_importances_
        return [
            {'feature': self.genera[i], 'importance': round(float(importances[i]), 4)}
            for i in np.argsort(importances)[::-1]
        ]
    
    def predict(self, abundances: Dict[str, float], model: str = 'rf') -> Dict:
        """Predict disease from microbiome profile."""
        if not self.is_trained:
            return {'error': 'Models not trained yet'}
        
        # Extract features in correct order
        features = np.array([[abundances.get(g, 0) for g in self.genera]])
        features_scaled = self.scaler.transform(features)
        
        if model == 'rf':
            model = self.rf_model
        elif model == 'lr':
            model = self.lr_model
        else:
            model = self.svm_model
        
        # Get prediction and probabilities
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        predicted_disease = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get class probabilities
        class_probs = {
            cls: round(prob, 4) 
            for cls, prob in zip(self.label_encoder.classes_, probabilities)
        }
        
        return {
            'predicted_disease': predicted_disease,
            'confidence': round(max(probabilities), 4),
            'probabilities': class_probs,
            'model_used': model.__class__.__name__
        }
    
    def get_pca_visualization_data(self) -> Dict:
        """Get PCA visualization data."""
        if self.pca is None:
            return {'error': 'PCA not computed'}
        
        return {
            'pca_coordinates': [
                {'x': float(x), 'y': float(y), 'label': self.label_encoder.inverse_transform([label])[0]}
                for x, y, label in zip(self.pca_result[:, 0], self.pca_result[:, 1], self.y_train)
            ],
            'explained_variance': self.pca.explained_variance_ratio_.tolist()
        }


class MicrobiomeAnalyzer:
    """Analyze microbiome data for disease profiling and patient estimation."""
    
    def __init__(self):
        self.data_loader = MicrobiomeDataLoader()
        self.disease_profiles = self.data_loader.DISEASE_PROFILES
        self.ml_classifier = None
    
    def analyze_bacterial_profile(self, abundances: Dict[str, float]) -> Dict:
        """Analyze bacterial profile and compare to healthy reference."""
        sorted_abundances = sorted(abundances.items(), key=lambda x: x[1], reverse=True)
        top_genera = sorted_abundances[:10]
        
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
        """Detect potential disease risks based on microbiome profile."""
        risks = {}
        
        for disease, profile in self.disease_profiles.items():
            score = 0
            details = []
            
            for genus in profile['decreased']:
                if genus in abundances:
                    abundance = abundances[genus]
                    if abundance < 0.05:
                        score += 1
                        details.append(f'{genus} is decreased')
            
            for genus in profile['increased']:
                if genus in abundances:
                    abundance = abundances[genus]
                    if abundance > 0.1:
                        score += 1
                        details.append(f'{genus} is increased')
            
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
        """Estimate patient profile based on microbiome and metadata."""
        abundances = {k: v for k, v in sample_data.items() 
                     if k in self.data_loader.COMMON_GENERA}
        
        age = sample_data.get('age', 50)
        bmi = sample_data.get('bmi', 25)
        
        profile_analysis = self.analyze_bacterial_profile(abundances)
        disease_risks = self.detect_disease_risk(abundances)
        
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
                'gut_health_score': round(profile_analysis['diversity'] / 4 * 100, 1),
                'microbial_diversity': round(profile_analysis['diversity'], 2)
            },
            'disease_risks': disease_risks,
            'recommendations': recommendations,
            'bacterial_profile': profile_analysis
        }
    
    def _get_age_group(self, age: int) -> str:
        if age < 30:
            return 'Young Adult (20-29)'
        elif age < 45:
            return 'Middle Adult (30-44)'
        elif age < 60:
            return 'Senior Adult (45-59)'
        else:
            return 'Elderly (60+)'
    
    def _get_bmi_category(self, bmi: float) -> str:
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    def compare_samples(self, sample1: Dict, sample2: Dict) -> Dict:
        """Compare two microbiome samples."""
        genera = self.data_loader.COMMON_GENERA
        
        abundances1 = {k: sample1.get(k, 0) for k in genera}
        abundances2 = {k: sample2.get(k, 0) for k in genera}
        
        differences = {}
        for genus in genera:
            diff = abundances1.get(genus, 0) - abundances2.get(genus, 0)
            differences[genus] = round(diff, 4)
        
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


# ML Model Functions
def train_ml_models(n_samples: int = 500) -> dict:
    """Train ML classification models."""
    if not SKLEARN_AVAILABLE:
        return {'error': 'scikit-learn is not available'}
    
    classifier = MicrobiomeMLClassifier()
    prep_result = classifier.prepare_data(n_samples=n_samples)
    train_result = classifier.train_models()
    
    return {
        'data_preparation': prep_result,
        'training_results': train_result
    }


def predict_disease(abundances: dict, model: str = 'rf') -> dict:
    """Predict disease from microbiome profile using ML."""
    if not SKLEARN_AVAILABLE:
        return {'error': 'scikit-learn is not available'}
    
    classifier = MicrobiomeMLClassifier()
    classifier.prepare_data(n_samples=200)
    classifier.train_models()
    
    return classifier.predict(abundances, model=model)
