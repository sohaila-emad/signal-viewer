#!/usr/bin/env python3
"""
Generate microbiome test data in standard bioinformatics formats:
- OTU table (TSV) - standard QIIME2 / mothur format
- Taxonomy table (TSV)
- Metadata mapping file (TSV)
"""

import numpy as np
import pandas as pd
import random
import json

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Common bacterial genera (OTUs)
COMMON_GENERA = [
    'Bacteroides', 'Prevotella', 'Faecalibacterium', 'Bifidobacterium',
    'Lactobacillus', 'Escherichia', 'Streptococcus', 'Clostridium',
    'Ruminococcus', 'Akkermansia', 'Blautia', 'Roseburia'
]

# Full taxonomy (domain > phylum > class > order > family > genus)
TAXONOMY = {
    'Bacteroides': 'd__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Bacteroides',
    'Prevotella': 'd__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__Prevotellaceae;g__Prevotella',
    'Faecalibacterium': 'd__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Ruminococcaceae;g__Faecalibacterium',
    'Bifidobacterium': 'd__Bacteria;p__Actinobacteria;c__Actinobacteria;o__Bifidobacteriales;f__Bifidobacteriaceae;g__Bifidobacterium',
    'Lactobacillus': 'd__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Lactobacillaceae;g__Lactobacillus',
    'Escherichia': 'd__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia',
    'Streptococcus': 'd__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Streptococcaceae;g__Streptococcus',
    'Clostridium': 'd__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Clostridiaceae;g__Clostridium',
    'Ruminococcus': 'd__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Ruminococcaceae;g__Ruminococcus',
    'Akkermansia': 'd__Bacteria;p__Verrucomicrobia;c__Verrucomicrobiae;o__Verrucomicrobiales;f__Akkermansiaceae;g__Akkermansia',
    'Blautia': 'd__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Lachnospiraceae;g__Blautia',
    'Roseburia': 'd__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Lachnospiraceae;g__Roseburia'
}

# Disease profiles
DISEASE_PROFILES = {
    'IBD': {'decreased': ['Faecalibacterium', 'Roseburia', 'Bifidobacterium'], 'increased': ['Escherichia', 'Streptococcus', 'Ruminococcus']},
    'Type2Diabetes': {'decreased': ['Akkermansia', 'Faecalibacterium'], 'increased': ['Bacteroides', 'Ruminococcus']},
    'ColorectalCancer': {'decreased': ['Lactobacillus', 'Bifidobacterium'], 'increased': ['Escherichia']},
    'Obesity': {'decreased': ['Bacteroidetes', 'Akkermansia'], 'increased': ['Firmicutes', 'Escherichia']},
    'CVD': {'decreased': ['Faecalibacterium', 'Roseburia'], 'increased': ['Bacteroides', 'Clostridium']}
}

diseases = list(DISEASE_PROFILES.keys())
n_samples = 50

# Generate sample IDs (QIIME2 format)
sample_ids = [f'SAMPLE_{i:04d}' for i in range(n_samples)]

# Generate OTU count matrix (using raw counts instead of relative abundances)
# Simulate sequencing depth of ~10,000 reads per sample
np.random.seed(42)
otu_matrix = np.zeros((n_samples, len(COMMON_GENERA)), dtype=int)

for i in range(n_samples):
    # Base counts
    base_counts = np.random.randint(100, 2000, len(COMMON_GENERA))
    
    # Add disease-specific modifications
    disease = random.choice(['Healthy'] + diseases)
    if disease != 'Healthy':
        profile = DISEASE_PROFILES.get(disease, {})
        for inc in profile.get('increased', []):
            if inc in COMMON_GENERA:
                idx = COMMON_GENERA.index(inc)
                base_counts[idx] += np.random.randint(500, 2000)
        for dec in profile.get('decreased', []):
            if dec in COMMON_GENERA:
                idx = COMMON_GENERA.index(dec)
                base_counts[idx] = max(10, base_counts[idx] - np.random.randint(100, 500))
    
    # Normalize to ~10,000 reads
    total = base_counts.sum()
    scaling_factor = 10000 / total
    otu_matrix[i] = (base_counts * scaling_factor).astype(int)

# Create OTU table (TSV format - QIIME2 style)
otu_df = pd.DataFrame(otu_matrix, index=sample_ids, columns=COMMON_GENERA)
otu_df.index.name = '#OTU ID'
otu_df.to_csv('data/microbiome_otu_table.tsv', sep='\t')
print(f'Created data/microbiome_otu_table.tsv with {n_samples} samples x {len(COMMON_GENERA)} OTUs')

# Create taxonomy table (TSV)
taxonomy_data = {'#OTU ID': COMMON_GENERA, 'Taxon': [TAXONOMY[g] for g in COMMON_GENERA], 'Confidence': [1.0] * len(COMMON_GENERA)}
taxonomy_df = pd.DataFrame(taxonomy_data)
taxonomy_df.to_csv('data/microbiome_taxonomy.tsv', sep='\t', index=False)
print('Created data/microbiome_taxonomy.tsv')

# Create sample metadata (mapping file - TSV)
metadata = {
    '#SampleID': sample_ids,
    'subject_id': [f'SUBJ_{i%15:03d}' for i in range(n_samples)],
    'age': np.random.randint(22, 75, n_samples),
    'bmi': np.round(np.random.uniform(18.5, 38.0, n_samples), 1),
    'gender': np.random.choice(['M', 'F'], n_samples),
    'disease_status': np.random.choice(['Healthy'] + diseases, n_samples, p=[0.5] + [0.5/len(diseases)] * len(diseases)),
    'collection_date': pd.date_range('2024-01-01', periods=n_samples, freq='7D').strftime('%Y-%m-%d').tolist()
}
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv('data/microbiome_metadata.tsv', sep='\t', index=False)
print('Created data/microbiome_metadata.tsv')

# Create BIOM-style JSON (compatible with QIIME2)
biom_data = {
    "id": "test-microbiome-dataset",
    "format": "Biological Observation Matrix 1.0.0",
    "format_url": "http://biom-format.org",
    "type": "OTU table",
    "generated_by": "signal-viewer microbiome generator",
    "date": "2024-01-01T00:00:00.000000",
    "rows": [{"id": otu, "metadata": {"taxonomy": TAXONOMY[otu]}} for otu in COMMON_GENERA],
    "columns": [{"id": sid, "metadata": {}} for sid in sample_ids],
    "matrix_type": "sparse",
    "matrix_element_type": "int",
    "data": []
}

# Convert to sparse triplet format
for i in range(n_samples):
    for j in range(len(COMMON_GENERA)):
        count = otu_matrix[i, j]
        if count > 0:
            biom_data["data"].append([i, j, count])

with open('data/microbiome_table.biom', 'w') as f:
    json.dump(biom_data, f, indent=2)
print('Created data/microbiome_table.biom')

# Print summary
print(f'\nDisease distribution:')
print(metadata_df['disease_status'].value_counts())

print(f'\nTotal reads per sample (mean): {otu_matrix.sum(axis=1).mean():.0f}')
print(f'OTU table shape: {otu_df.shape}')
