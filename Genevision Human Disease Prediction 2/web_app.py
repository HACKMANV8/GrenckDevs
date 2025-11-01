#!/usr/bin/env python3
"""
Flask Web Application for Genetic Analysis
Integrates with existing backend logic from main_app.py
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import json
import os
from main_app import GeneticModelTrainer
from gui3 import generate_report_text, create_pdf, validate_inputs
from web_punnett import generate_web_safe_punnett_image

app = Flask(__name__, static_folder='assets', static_url_path='/assets')
CORS(app)

# Global model variable
model = None

def load_model():
    """Load the genetic model on startup"""
    global model
    try:
        model = GeneticModelTrainer.load_saved_model(
            'models/genetic_model.h5',
            'models/preprocessors.pkl'
        )
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return False

@app.route('/')
def index():
    """Serve the main genetic analysis page"""
    return render_template('genetic_website.html')

@app.route('/genetic')
def genetic_page():
    """Alternative route for genetic analysis page"""
    return render_template('genetic_website.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_genetics():
    """API endpoint for genetic analysis"""
    try:
        data = request.get_json()
        
        # Extract data from request
        disorder = data.get('disorder')
        parent1_sex = data.get('parent1_sex')
        parent1_genotype = data.get('parent1_genotype', '').upper()
        parent1_affected = data.get('parent1_affected', False)
        parent2_sex = data.get('parent2_sex')
        parent2_genotype = data.get('parent2_genotype', '').upper()
        parent2_affected = data.get('parent2_affected', False)
        generations = int(data.get('generations', 1))
        
        # Validate inputs
        errors = validate_inputs(
            disorder, parent1_sex, parent1_genotype, parent1_affected,
            parent2_sex, parent2_genotype, parent2_affected, generations
        )
        
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Prepare input data
        input_data = {
            'disorder': disorder,
            'parent1_sex': 'M' if parent1_sex == "MALE" else 'F',
            'parent2_sex': 'M' if parent2_sex == "MALE" else 'F',
            'parent1_genotype': parent1_genotype,
            'parent2_genotype': parent2_genotype,
            'parent1_affected': parent1_affected,
            'parent2_affected': parent2_affected,
            'generations': generations
        }
        
        # Create DataFrame for model prediction
        df = pd.DataFrame([{
            'disorder': input_data['disorder'],
            'parent1_sex': input_data['parent1_sex'],
            'parent2_sex': input_data['parent2_sex'],
            'parent1_genotype': input_data['parent1_genotype'],
            'parent2_genotype': input_data['parent2_genotype'],
            'parent1_affected': input_data['parent1_affected'],
            'parent2_affected': input_data['parent2_affected'],
            'chromosome': model.label_encoders['chromosome'].classes_[0],
            'snp_id': model.label_encoders['snp_id'].classes_[0],
            'position': 0,
            'inheritance': model.label_encoders['inheritance'].classes_[0],
            'penetrance': 1.0,
            'mutation_rate': 0.001
        }])
        
        # Get prediction
        probability = model.predict_probability(df)
        
        # Generate Punnett square image
        punnett_img_bytes = generate_web_safe_punnett_image(parent1_genotype, parent2_genotype)
        punnett_img_base64 = base64.b64encode(punnett_img_bytes).decode('utf-8')
        
        # Generate report text
        report_text = generate_report_text(input_data, df, probability)
        
        # Store results in session (you might want to use a proper session store)
        session_data = {
            'input_data': input_data,
            'df': df.to_dict('records')[0],
            'probability': float(probability),
            'report_text': report_text,
            'punnett_img_bytes': punnett_img_bytes
        }
        
        # For now, we'll return everything in the response
        return jsonify({
            'success': True,
            'results': {
                'disorder': disorder,
                'probability': float(probability * 100),  # Convert to percentage
                'punnett_image': punnett_img_base64,
                'input_data': input_data,
                'session_id': 'temp_session'  # In production, use proper session management
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export-report', methods=['POST'])
def export_report():
    """API endpoint to generate and download PDF report"""
    try:
        data = request.get_json()
        
        # In a real application, you'd retrieve this from session storage
        # For now, we'll regenerate the data
        input_data = data.get('input_data')
        probability = data.get('probability') / 100  # Convert back from percentage
        
        # Recreate DataFrame
        df = pd.DataFrame([{
            'disorder': input_data['disorder'],
            'parent1_sex': input_data['parent1_sex'],
            'parent2_sex': input_data['parent2_sex'],
            'parent1_genotype': input_data['parent1_genotype'],
            'parent2_genotype': input_data['parent2_genotype'],
            'parent1_affected': input_data['parent1_affected'],
            'parent2_affected': input_data['parent2_affected'],
            'chromosome': model.label_encoders['chromosome'].classes_[0],
            'snp_id': model.label_encoders['snp_id'].classes_[0],
            'position': 0,
            'inheritance': model.label_encoders['inheritance'].classes_[0],
            'penetrance': 1.0,
            'mutation_rate': 0.001
        }])
        
        # Generate report and Punnett square
        report_text = generate_report_text(input_data, df, probability)
        punnett_img_bytes = generate_web_safe_punnett_image(
            input_data['parent1_genotype'], 
            input_data['parent2_genotype']
        )
        
        # Create PDF
        pdf_bytes = create_pdf(report_text, punnett_img_bytes)
        
        # Create a BytesIO object to serve the PDF
        pdf_buffer = BytesIO(pdf_bytes)
        pdf_buffer.seek(0)
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name='GrenckDevs_Genetic_Analysis_Report.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("Starting Flask web application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to start application: Model could not be loaded")