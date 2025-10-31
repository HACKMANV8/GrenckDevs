#update data gen
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Define genetic disorders and their characteristics
DISORDERS = {
    'Cystic Fibrosis': {
        'gene': 'CFTR',
        'chromosome': 7,
        'inheritance': 'autosomal_recessive',
        'snps': [
            {'id': 'rs113993960', 'position': 117559593},  # Common ΔF508 mutation
            {'id': 'rs75527207', 'position': 117587806}
        ],
        'mutation_rate': 0.004,
        'penetrance_range': (0.95, 1.0)
    },
    'Sickle Cell Anemia': {
        'gene': 'HBB',
        'chromosome': 11,
        'inheritance': 'autosomal_recessive',
        'snps': [
            {'id': 'rs334', 'position': 5227002},  # HbS mutation
        ],
        'mutation_rate': 0.003,
        'penetrance_range': (0.95, 1.0)
    },
    'Huntington Disease': {
        'gene': 'HTT',
        'chromosome': 4,
        'inheritance': 'autosomal_dominant',
        'snps': [
            {'id': 'rs362307', 'position': 3074877},
        ],
        'mutation_rate': 0.001,
        'penetrance_range': (0.95, 1.0)
    },
    'Duchenne Muscular Dystrophy': {
        'gene': 'DMD',
        'chromosome': 'X',
        'inheritance': 'x_linked_recessive',
        'snps': [
            {'id': 'rs28935907', 'position': 31117307},
            {'id': 'rs104894790', 'position': 31137345}
        ],
        'mutation_rate': 0.008,
        'penetrance_range': (0.98, 1.0)
    },
    'Tay-Sachs Disease': {
        'gene': 'HEXA',
        'chromosome': 15,
        'inheritance': 'autosomal_recessive',
        'snps': [
            {'id': 'rs121907954', 'position': 72637817},
        ],
        'mutation_rate': 0.002,
        'penetrance_range': (0.95, 1.0)
    },
    'Marfan Syndrome': {
        'gene': 'FBN1',
        'chromosome': 15,
        'inheritance': 'autosomal_dominant',
        'snps': [
            {'id': 'rs137854476', 'position': 48707833},
        ],
        'mutation_rate': 0.003,
        'penetrance_range': (0.90, 0.97)
    },
    'Hemophilia A': {
        'gene': 'F8',
        'chromosome': 'X',
        'inheritance': 'x_linked_recessive',
        'snps': [
            {'id': 'rs28935203', 'position': 154835788},
        ],
        'mutation_rate': 0.005,
        'penetrance_range': (0.95, 1.0)
    },
    'Familial Hypercholesterolemia': {
        'gene': 'LDLR',
        'chromosome': 19,
        'inheritance': 'autosomal_dominant',
        'snps': [
            {'id': 'rs121908025', 'position': 11200045},
        ],
        'mutation_rate': 0.002,
        'penetrance_range': (0.70, 0.90)
    },
    'BRCA-Related Breast Cancer': {
        'gene': 'BRCA1',
        'chromosome': 17,
        'inheritance': 'autosomal_dominant',
        'snps': [
            {'id': 'rs80357090', 'position': 43094464},
            {'id': 'rs28897696', 'position': 43094915}
        ],
        'mutation_rate': 0.006,
        'penetrance_range': (0.45, 0.85)
    },
    'Alpha-1 Antitrypsin Deficiency': {
        'gene': 'SERPINA1',
        'chromosome': 14,
        'inheritance': 'autosomal_recessive',
        'snps': [
            {'id': 'rs28929474', 'position': 94844947},
        ],
        'mutation_rate': 0.002,
        'penetrance_range': (0.60, 0.95)
    }
}

def generate_genotype_pair() -> Tuple[str, str]:
    """Generate random genotype pairs using C and T."""
    return np.random.choice(['CC', 'CT', 'TT']), np.random.choice(['CC', 'CT', 'TT'])

def calculate_offspring_probability(
    inheritance: str,
    parent1_genotype: str,
    parent2_genotype: str,
    parent1_sex: str,
    parent2_sex: str,
    penetrance: float
) -> float:
    """Calculate probability of offspring inheriting disorder based on inheritance pattern."""

    if inheritance == 'autosomal_recessive':
        # Both parents must pass mutant allele
        p1_mutant = 1 if 'T' in parent1_genotype else 0
        p2_mutant = 1 if 'T' in parent2_genotype else 0
        prob = (p1_mutant * p2_mutant * 0.25) * penetrance

    elif inheritance == 'autosomal_dominant':
        # One mutant allele is sufficient
        p1_mutant = 1 if 'T' in parent1_genotype else 0
        p2_mutant = 1 if 'T' in parent2_genotype else 0
        prob = (1 - (1-p1_mutant*0.5) * (1-p2_mutant*0.5)) * penetrance

    elif inheritance == 'x_linked_recessive':
        if parent1_sex == 'M':  # Father affected
            prob = 0.5 if parent1_genotype == 'T' else 0
        else:  # Mother carrier
            prob = 0.25 if 'T' in parent2_genotype else 0
        prob *= penetrance

    else:  # x_linked_dominant
        if parent1_sex == 'M':  # Father affected
            prob = 1.0 if 'T' in parent1_genotype else 0
        else:  # Mother carrier
            prob = 0.5 if 'T' in parent2_genotype else 0
        prob *= penetrance

    return round(prob, 4)

def generate_dataset(n_samples: int = 100000) -> pd.DataFrame:
    """Generate synthetic genetic disorder dataset."""

    data = []
    for _ in range(n_samples):
        # Select random disorder
        disorder_name = np.random.choice(list(DISORDERS.keys()))
        disorder = DISORDERS[disorder_name]

        # Select random SNP for this disorder
        snp = np.random.choice(disorder['snps'])
        snp_id, position = snp['id'], snp['position']

        # Generate parent characteristics
        parent1_sex = np.random.choice(['M', 'F'])
        parent2_sex = 'F' if parent1_sex == 'M' else np.random.choice(['M', 'F'])

        # Generate genotypes
        parent1_genotype, parent2_genotype = generate_genotype_pair()

        # Determine affected status based on genotype and inheritance
        parent1_affected = 1 if 'T' in parent1_genotype else 0
        parent2_affected = 1 if 'T' in parent2_genotype else 0

        # Generate penetrance
        penetrance = round(np.random.uniform(*disorder['penetrance_range']), 4)

        # Calculate offspring probability
        offspring_prob = calculate_offspring_probability(
            disorder['inheritance'],
            parent1_genotype,
            parent2_genotype,
            parent1_sex,
            parent2_sex,
            penetrance
        )

        data.append({
            'disorder': disorder_name,
            'chromosome': disorder['chromosome'],
            'snp_id': snp_id,
            'position': position,
            'parent1_sex': parent1_sex,
            'parent2_sex': parent2_sex,
            'parent1_genotype': parent1_genotype,
            'parent2_genotype': parent2_genotype,
            'parent1_affected': parent1_affected,
            'parent2_affected': parent2_affected,
            'inheritance': disorder['inheritance'],
            'penetrance': penetrance,
            'mutation_rate': disorder['mutation_rate'],
            'offspring_probability': offspring_prob
        })

    return pd.DataFrame(data)

# Generate and save dataset
df = generate_dataset(10000000)
df.to_csv('genetic_disorders_dataset.csv', index=False)