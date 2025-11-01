#!/usr/bin/env python3
"""
Enhanced Visualizations Module for GrenckDevs Genetic Analysis Platform
Includes interactive charts, protein structure visualization, and AlphaFold integration
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import requests
import json
import base64
from io import BytesIO
import py3Dmol
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings
warnings.filterwarnings('ignore')

# Disease-related protein mappings (common genetic disorders)
DISEASE_PROTEIN_MAP = {
    'Huntington Disease': 'P42858',  # Huntingtin
    'Cystic Fibrosis': 'P13569',     # CFTR
    'Sickle Cell Disease': 'P68871',  # Hemoglobin subunit beta
    'Tay-Sachs Disease': 'P06865',   # HEXA
    'Duchenne Muscular Dystrophy': 'P11532',  # Dystrophin
    'Hemophilia A': 'P00451',        # Factor VIII
    'Phenylketonuria': 'P00439',     # Phenylalanine hydroxylase
    'Marfan Syndrome': 'P35555',     # Fibrillin-1
    'Neurofibromatosis': 'P21359',   # Neurofibromin
    'Polycystic Kidney Disease': 'P98161',  # Polycystin-1
}

class EnhancedVisualizer:
    """Enhanced visualization class with AlphaFold integration"""
    
    def __init__(self):
        self.alphafold_base_url = "https://alphafold.ebi.ac.uk/api/prediction/"
        
    def create_probability_gauge(self, probability):
        """Create an interactive probability gauge chart"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Disease Risk Probability (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            font={'color': "darkblue", 'family': "Arial"},
            paper_bgcolor="white"
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="probability_gauge")
    
    def create_inheritance_pattern_chart(self, parent1_genotype, parent2_genotype):
        """Create inheritance pattern visualization"""
        # Generate all possible offspring combinations
        alleles1 = list(parent1_genotype)
        alleles2 = list(parent2_genotype)
        
        offspring = []
        for a1 in alleles1:
            for a2 in alleles2:
                offspring.append(a1 + a2)
        
        # Count frequencies
        from collections import Counter
        freq = Counter(offspring)
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(freq.keys()),
            values=list(freq.values()),
            hole=.3,
            textinfo='label+percent',
            textfont_size=12,
            marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        )])
        
        fig.update_layout(
            title="Offspring Genotype Distribution",
            annotations=[dict(text='Inheritance', x=0.5, y=0.5, font_size=16, showarrow=False)],
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="inheritance_chart")
    
    def create_risk_comparison_chart(self, disorder, probability):
        """Create risk comparison with population averages"""
        # Mock population data for comparison
        population_risks = {
            'Huntington Disease': 0.0001,
            'Cystic Fibrosis': 0.0004,
            'Sickle Cell Disease': 0.001,
            'Tay-Sachs Disease': 0.0003,
            'Duchenne Muscular Dystrophy': 0.0002,
            'Hemophilia A': 0.0001,
            'Phenylketonuria': 0.0001,
            'Marfan Syndrome': 0.0002,
            'Neurofibromatosis': 0.0003,
            'Polycystic Kidney Disease': 0.001,
        }
        
        pop_risk = population_risks.get(disorder, 0.0005) * 100
        your_risk = probability * 100
        
        fig = go.Figure(data=[
            go.Bar(name='Population Average', x=[disorder], y=[pop_risk], marker_color='lightblue'),
            go.Bar(name='Your Risk', x=[disorder], y=[your_risk], marker_color='red')
        ])
        
        fig.update_layout(
            title='Risk Comparison: You vs Population Average',
            yaxis_title='Risk Percentage (%)',
            barmode='group',
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="risk_comparison")
    
    def get_alphafold_protein_data(self, uniprot_id):
        """Fetch protein structure data from AlphaFold"""
        try:
            url = f"{self.alphafold_base_url}{uniprot_id}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data[0] if data else None
            else:
                print(f"AlphaFold API returned status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching AlphaFold data: {str(e)}")
            return None
    
    def create_protein_structure_viewer(self, disorder):
        """Create 3D protein structure viewer using AlphaFold data"""
        uniprot_id = DISEASE_PROTEIN_MAP.get(disorder)
        
        if not uniprot_id:
            return f"""
            <div id="protein_viewer" style="height: 400px; border: 1px solid #ccc; padding: 20px; text-align: center;">
                <h3>Protein Structure Viewer</h3>
                <p>Protein structure data not available for {disorder}</p>
                <p>Available disorders: {', '.join(DISEASE_PROTEIN_MAP.keys())}</p>
            </div>
            """
        
        # Get AlphaFold data
        protein_data = self.get_alphafold_protein_data(uniprot_id)
        
        if not protein_data:
            return f"""
            <div id="protein_viewer" style="height: 400px; border: 1px solid #ccc; padding: 20px; text-align: center;">
                <h3>Protein Structure Viewer</h3>
                <p>Unable to fetch protein structure for {disorder}</p>
                <p>UniProt ID: {uniprot_id}</p>
            </div>
            """
        
        pdb_url = protein_data.get('pdbUrl', '')
        confidence_score = protein_data.get('confidenceScore', 'N/A')
        
        # Create 3D viewer HTML
        viewer_html = f"""
        <div id="protein_viewer" style="height: 500px; border: 1px solid #ccc;">
            <h3 style="text-align: center; margin: 10px;">3D Protein Structure - {disorder}</h3>
            <div style="text-align: center; margin: 10px;">
                <strong>UniProt ID:</strong> {uniprot_id} | 
                <strong>Confidence Score:</strong> {confidence_score}
            </div>
            <div id="3dmol-container" style="height: 400px; width: 100%; position: relative;"></div>
        </div>
        
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <script>
            $(document).ready(function() {{
                let element = $('#3dmol-container');
                let config = {{ backgroundColor: 'white' }};
                let viewer = $3Dmol.createViewer(element, config);
                
                // Load protein structure from AlphaFold
                $3Dmol.download("pdb:{uniprot_id}", viewer, {{}}, function() {{
                    viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                    viewer.zoomTo();
                    viewer.render();
                }});
                
                // Fallback: try to load from AlphaFold directly
                if ("{pdb_url}") {{
                    fetch("{pdb_url}")
                        .then(response => response.text())
                        .then(data => {{
                            viewer.addModel(data, "pdb");
                            viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                            viewer.zoomTo();
                            viewer.render();
                        }})
                        .catch(error => {{
                            console.log("Error loading protein structure:", error);
                            element.html('<p style="text-align: center; padding: 50px;">Unable to load 3D structure</p>');
                        }});
                }}
            }});
        </script>
        """
        
        return viewer_html
    
    def create_genetic_timeline(self, disorder, generations=3):
        """Create a genetic inheritance timeline"""
        # Generate mock data for inheritance across generations
        gen_data = []
        risk_levels = [np.random.uniform(0.1, 0.9) for _ in range(generations)]
        
        for i in range(generations):
            gen_data.append({
                'Generation': f'Generation {i+1}',
                'Risk_Level': risk_levels[i] * 100,
                'Affected_Count': np.random.randint(1, 10),
                'Total_Count': np.random.randint(10, 50)
            })
        
        df = pd.DataFrame(gen_data)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Risk Level Across Generations', 'Affected vs Total Population'),
            vertical_spacing=0.1
        )
        
        # Risk level line chart
        fig.add_trace(
            go.Scatter(x=df['Generation'], y=df['Risk_Level'], 
                      mode='lines+markers', name='Risk Level (%)',
                      line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # Population bar chart
        fig.add_trace(
            go.Bar(x=df['Generation'], y=df['Total_Count'], 
                   name='Total Population', marker_color='lightblue'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df['Generation'], y=df['Affected_Count'], 
                   name='Affected', marker_color='red'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text=f"Genetic Timeline Analysis - {disorder}")
        
        return fig.to_html(include_plotlyjs='cdn', div_id="genetic_timeline")
    
    def create_comprehensive_dashboard(self, disorder, probability, parent1_genotype, parent2_genotype):
        """Create a comprehensive visualization dashboard"""
        dashboard_html = f"""
        <div class="visualization-dashboard" style="padding: 20px;">
            <h2 style="text-align: center; color: #2c3e50;">🧬 GrenckDevs Enhanced Genetic Analysis Dashboard</h2>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                <div class="chart-container">
                    {self.create_probability_gauge(probability)}
                </div>
                <div class="chart-container">
                    {self.create_inheritance_pattern_chart(parent1_genotype, parent2_genotype)}
                </div>
            </div>
            
            <div style="margin: 20px 0;">
                {self.create_risk_comparison_chart(disorder, probability)}
            </div>
            
            <div style="margin: 20px 0;">
                {self.create_protein_structure_viewer(disorder)}
            </div>
            
            <div style="margin: 20px 0;">
                {self.create_genetic_timeline(disorder)}
            </div>
            
            <div style="text-align: center; margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                <h3>🔬 Analysis Summary</h3>
                <p><strong>Disorder:</strong> {disorder}</p>
                <p><strong>Risk Probability:</strong> {probability*100:.2f}%</p>
                <p><strong>Parent Genotypes:</strong> {parent1_genotype} × {parent2_genotype}</p>
                <p><strong>Protein Involved:</strong> {DISEASE_PROTEIN_MAP.get(disorder, 'Unknown')}</p>
            </div>
        </div>
        
        <style>
            .visualization-dashboard {{
                font-family: 'Arial', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                color: white;
            }}
            .chart-container {{
                background: white;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
        </style>
        """
        
        return dashboard_html

# Create global instance
visualizer = EnhancedVisualizer()

def generate_enhanced_visualizations(disorder, probability, parent1_genotype, parent2_genotype):
    """Generate all enhanced visualizations"""
    return visualizer.create_comprehensive_dashboard(
        disorder, probability, parent1_genotype, parent2_genotype
    )