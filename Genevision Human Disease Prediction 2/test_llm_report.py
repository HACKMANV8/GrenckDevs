#!/usr/bin/env python3
"""
Test LLM report generation
"""

import ollama
import pandas as pd

def test_llm_report():
    try:
        print("🤖 Testing LLM report generation...")
        
        # Sample data
        input_data = {
            'disorder': 'Cystic Fibrosis',
            'parent1_sex': 'M',
            'parent2_sex': 'F',
            'parent1_genotype': 'CC',
            'parent2_genotype': 'CT',
            'parent1_affected': False,
            'parent2_affected': False,
            'generations': 1
        }
        
        df = pd.DataFrame([input_data])
        probability = 0.4567  # 45.67%
        
        # Exact query from gui3.py
        query = f"""You are a report generator working for GrenckDevs. Generate a professional medical report.

        Parent 1 Information:
        Sex (M/F): {df['parent1_sex'].values[0]}
        Genotype: {df['parent1_genotype'].values[0]}
        Affected: {df['parent1_affected'].values[0]}

        Parent 2 Information:
        Sex (M/F): {df['parent2_sex'].values[0]}
        Genotype: {df['parent2_genotype'].values[0]}
        Affected: {df['parent2_affected'].values[0]}

        Disorder: {input_data['disorder']}
        First Generation Probability: {probability*100:.2f}%
        
        Generate a brief professional genetic analysis report for GrenckDevs."""

        print("📝 Sending query to phi3:latest...")
        response = ollama.chat(model='phi3:latest', messages=[
            {"role": "user", "content": query}
        ])

        report = response['message']['content']
        
        print("✅ LLM Report Generated Successfully!")
        print("="*60)
        print(report)
        print("="*60)
        print(f"📊 Report length: {len(report)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_llm_report()