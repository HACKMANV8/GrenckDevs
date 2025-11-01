import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from io import BytesIO
import ollama
import re
from main_app import GeneticModelTrainer

# TensorFlow Configuration
import tensorflow as tf
import os
try:
    tf.config.set_visible_devices([], 'GPU')
    if tf.config.list_logical_devices('GPU'):
        st.warning("TensorFlow still sees GPUs after attempting to disable. Forcing CPU via environment variable.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        print("TensorFlow configured to not use GPUs.")
except Exception as e:
    st.warning(f"Could not configure TensorFlow to disable GPUs: {e}. TensorFlow will use its default. If issues persist, this might be related.")

# Page configuration
st.set_page_config(
    page_title="GrenckDevs Genetic Analysis",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def local_css():
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        /* Root Variables */
        :root {
            --primary-color: #6366f1;
            --secondary-color: #8b5cf6;
            --accent-color: #06b6d4;
            --success-color: #10b981;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --glass-bg: rgba(15, 23, 42, 0.7);
            --glass-border: rgba(148, 163, 184, 0.1);
        }
        
        /* Global Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        .block-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 2rem 4rem 2rem;
        }
        
        .main, .block-container, [data-testid="stAppViewContainer"] {
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }
        
        /* Animated Gradient Background */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 25%, #312e81 50%, #1e1b4b 75%, #0f172a 100%);
            background-size: 400% 400%;
            animation: gradientShift 20s ease infinite;
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }
        
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(139, 92, 246, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 40% 20%, rgba(6, 182, 212, 0.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        /* DNA Helix Animation Background */
        .stApp::after {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: 
                repeating-linear-gradient(
                    90deg,
                    transparent,
                    transparent 50px,
                    rgba(99, 102, 241, 0.03) 50px,
                    rgba(99, 102, 241, 0.03) 51px
                );
            pointer-events: none;
            z-index: 0;
        }
        
        /* Hide default elements */
        [data-testid="stHeader"],
        footer,
        #MainMenu {
            display: none !important;
        }
        
        /* Header Section */
        .header-container {
            position: relative;
            z-index: 10;
            margin-bottom: 3rem;
        }
        
        .logo-section {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .main-header {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 2.5rem 2rem;
            text-align: center;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.1), transparent);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            letter-spacing: 2px;
            margin: 0;
            text-transform: uppercase;
            filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.5));
        }
        
        .main-header .subtitle {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            font-weight: 300;
        }
        
        /* Glass Card Components */
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            z-index: 1;
        }
        
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 12px 40px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border-color: rgba(99, 102, 241, 0.3);
        }
        
        .glass-card h3 {
            color: var(--primary-color);
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .glass-card h3::before {
            content: '';
            width: 4px;
            height: 20px;
            background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
            border-radius: 2px;
        }
        
        /* Form Elements */
        label, .stSelectbox label, .stTextInput label, .stNumberInput label {
            color: var(--text-secondary) !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Select boxes and Inputs */
        div[data-baseweb="select"],
        div[data-baseweb="base-input"] {
            background: rgba(30, 41, 59, 0.6) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-baseweb="select"]:hover,
        div[data-baseweb="base-input"]:hover {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
        }
        
        div[data-baseweb="select"]:focus-within,
        div[data-baseweb="base-input"]:focus-within {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
        }
        
        div[data-baseweb="select"] input,
        div[data-baseweb="base-input"] input {
            color: var(--text-primary) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.95rem !important;
        }
        
        /* Radio buttons and checkboxes */
        .stRadio, .stCheckbox {
            padding: 0.5rem 0;
        }
        
        .stRadio label, .stCheckbox label {
            color: var(--text-primary) !important;
            font-weight: 400 !important;
            font-size: 0.95rem !important;
        }
        
        .stRadio > div {
            gap: 1rem;
        }
        
        /* Checkbox styling */
        .stCheckbox > label > div {
            background: rgba(30, 41, 59, 0.6) !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 6px !important;
        }
        
        .stCheckbox > label > div[data-checked="true"] {
            background: var(--primary-color) !important;
            border-color: var(--primary-color) !important;
        }
        
        /* Button Styles */
        .stButton > button {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            padding: 0.875rem 2rem !important;
            border: none !important;
            border-radius: 12px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            font-size: 0.875rem !important;
            position: relative !important;
            overflow: hidden !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.4) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0px) !important;
        }
        
        /* Button color variants */
        div[data-testid="column"]:nth-child(1) .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            color: white !important;
        }
        
        div[data-testid="column"]:nth-child(2) .stButton > button {
            background: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%) !important;
            color: white !important;
        }
        
        div[data-testid="column"]:nth-child(3) .stButton > button {
            background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%) !important;
            color: white !important;
        }
        
        /* Results Card */
        .results-card {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: 
                0 8px 32px rgba(99, 102, 241, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            animation: fadeInUp 0.5s ease;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .results-card h3 {
            color: var(--accent-color);
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .results-card p {
            color: var(--text-primary);
            font-size: 1rem;
            line-height: 1.8;
            margin-bottom: 0.75rem;
        }
        
        .results-card p strong {
            color: var(--primary-color);
            font-weight: 600;
        }
        
        /* Images */
        .stImage {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            margin: 1.5rem 0;
        }
        
        /* Warning/Error messages */
        .stWarning, .stError, .stSuccess {
            background: var(--glass-bg) !important;
            backdrop-filter: blur(20px);
            border-radius: 12px !important;
            border-left-width: 4px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        }
        
        /* Number input */
        .stNumberInput input {
            background: rgba(30, 41, 59, 0.6) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 10px !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, var(--secondary-color), var(--primary-color));
        }
        
        /* Floating particles effect */
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
        }
        
        /* Download button specific styling */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            color: white !important;
            width: 100%;
            margin-top: 1rem;
        }
        
        /* Terminate message */
        .terminate-message {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(6, 182, 212, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
            box-shadow: 0 4px 20px rgba(6, 182, 212, 0.2);
        }
        
        .terminate-message p {
            color: var(--accent-color);
            font-weight: 500;
            margin: 0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def validate_inputs(disorder, parent1_sex, parent1_genotype, parent1_affected, 
                    parent2_sex, parent2_genotype, parent2_affected, generations):
    """Validate all user inputs before analysis."""
    errors = []
    if disorder == "Select a disorder":
        errors.append("Please select a genetic disorder")
    if not parent1_sex:
        errors.append("Parent 1: Sex not selected")
    if not parent2_sex:
        errors.append("Parent 2: Sex not selected")
    if len(parent1_genotype) != 2 or not parent1_genotype.isalpha():
        errors.append("Parent 1: Genotype must be a 2-letter code (e.g., CC)")
    if len(parent2_genotype) != 2 or not parent2_genotype.isalpha():
        errors.append("Parent 2: Genotype must be a 2-letter code (e.g., CC)")
    if generations < 1:
        errors.append("Generations must be at least 1")
    return errors

def generate_report_text(input_data, df, probability):
    """Generate a detailed report using Ollama based on input data and prediction."""
    query = f"""As a professional genetic analyst at GrenckDevs, generate a detailed genetic analysis report based on the provided input data.

Format the report as a professional medical document, suitable for an A4 page, maximum 5 pages. Adopt a first-person perspective using "We at GrenckDevs...". Ensure the report is comprehensive and professional.

Group statistical data and present numerical data in a table format within the 'Statistical Data and Findings' section (V).

Base all recommendations *solely* on the provided input data.

Use the term "GrenckDevs" frequently throughout the report.

Follow this report template structure:

GrenckDevs Genetic Analysis Report

I. Executive Summary
II. Introduction
   - Background
   - Purpose
III. Methodology
   - Data Collection
   - Genetic Analysis
   - Statistical Computation
   - Quality Assurance
IV. Detailed Patient and Parental Information
   - Disorder Under Evaluation
   - Parent 1 Details
     - Sex
     - Genotype
     - Affected Status
   - Parent 2 Details
     - Sex
     - Genotype
     - Affected Status
   - Generational Prediction
V. Statistical Data and Findings
   - Present key statistics and numerical data in a table. Example table structure:
     | Parameter | Value |
     |---|---|
     | Disorder |  |
     | Probability (First Generation) |  |
     | Parent 1 Genotype |  |
     | Parent 1 Affected |  |
     | Parent 2 Genotype |  |
     | Parent 2 Affected |  |
     | Generations Predicted |  |
VI. In-Depth Analysis and Discussion
   A. Genetic Pattern Analysis
   B. Statistical Validity and Model Considerations
VII. Recommendations and Future Directions
   - Genetic Counseling
   - Additional Diagnostic Testing
   - Regular Monitoring
   - Multidisciplinary Consultation
VIII. Conclusion

Report Prepared By: GrenckDevs
Contact Information:
GrenckDevs
Email: grenck.devs@gmail.com
Phone: +91 1234567890

Disclaimer: This report is based solely on the provided data and GrenckDevs analytical models, intended for healthcare professionals.

Input Data:
Parent 1 Information:
Sex: {df['parent1_sex'].values[0]}
Genotype: {df['parent1_genotype'].values[0]}
Affected: {df['parent1_affected'].values[0]}

Parent 2 Information:
Sex: {df['parent2_sex'].values[0]}
Genotype: {df['parent2_genotype'].values[0]}
Affected: {df['parent2_affected'].values[0]}

Prediction Results:
Disorder: {input_data['disorder']}
Probability: {probability*100:.2f}%"""
    
    response = ollama.chat(model='gemma3:4b', messages=[
        {"role": "user", "content": query}
    ])
    report_text = response['message']['content']
    return report_text.split('>', 2)[-1] if '>' in report_text else report_text

def generate_punnett_image(father_genotype, mother_genotype):
    """Generate a Punnett square visualization as image bytes."""
    def get_color(genotype):
        if genotype == "CC":
            return "green"
        elif genotype == "CT":
            return "orange"
        elif genotype == "TT":
            return "red"
        return "grey"

    offspring = []
    father_alleles = list(father_genotype)
    mother_alleles = list(mother_genotype)
    for father_allele in father_alleles:
        for mother_allele in mother_alleles:
            offspring.append("".join(sorted(father_allele + mother_allele)))

    colors = [get_color(gen) for gen in offspring]
    father_color = get_color(father_genotype)
    mother_color = get_color(mother_genotype)

    graph = nx.DiGraph()
    graph.add_node("parent1", shape="square", label=f"Father ({father_genotype})", color=father_color)
    graph.add_node("parent2", shape="circle", label=f"Mother ({mother_genotype})", color=mother_color)

    offspring_labels = [f"Child {i+1}\n({gen})" for i, gen in enumerate(offspring)]
    for i, label in enumerate(offspring_labels):
        graph.add_node(label, shape="square", label=label, color=colors[i])
        graph.add_edge("parent1", label)
        graph.add_edge("parent2", label)

    pos = {
        "parent1": (0, 2), "parent2": (2, 2),
        offspring_labels[0]: (-1, 0), offspring_labels[1]: (1, 0),
        offspring_labels[2]: (3, 0), offspring_labels[3]: (5, 0)
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    node_colors = [graph.nodes[node]["color"] for node in graph.nodes]
    node_labels = {n: graph.nodes[n]["label"] for n in graph.nodes}

    nx.draw(graph, pos, with_labels=True, labels=node_labels, node_size=3000,
            node_color=node_colors, edge_color="black", font_size=10, font_weight="bold")

    plt.title("Punnett Square Family Tree")
    plt.legend(handles=[
        mpatches.Patch(color='green', label='Normal (CC)'),
        mpatches.Patch(color='orange', label='Carrier (CT)'),
        mpatches.Patch(color='red', label='Affected (TT)')
    ], loc='upper right')

    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300)
    img_buffer.seek(0)
    plt.close(fig)
    return img_buffer.getvalue()

def create_pdf(report_text, punnett_img_bytes):
    """Generate a PDF report in memory with report text and Punnett square image."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    story = []
    
    styles = getSampleStyleSheet()
    fixed_title_style = ParagraphStyle(
        'FixedTitle',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=30,
        spaceBefore=30,
        leading=30,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#87497d')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        alignment=TA_LEFT,
        spaceAfter=12,
        spaceBefore=24,
        textColor=colors.HexColor('#2C3E50')
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        leading=16
    )
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=body_style,
        leftIndent=20,
        spaceBefore=6,
        spaceAfter=6
    )

    story.append(Paragraph("GrenckDevs Genetic Analysis Report", fixed_title_style))
    story.append(Spacer(1, 20))

    img_buffer = BytesIO(punnett_img_bytes)
    story.append(Image(img_buffer, width=400, height=300))
    story.append(Spacer(1, 20))

    sections = report_text.split('---')
    current_list_items = []
    
    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.startswith('###'):
                if current_list_items:
                    story.append(ListFlowable(current_list_items, bulletType='bullet', leftIndent=20))
                    current_list_items = []
                story.append(Paragraph(line.replace('###', '').strip(), heading_style))
                i += 1
                continue
            if line.startswith('-'):
                text = line[1:].strip()
                text = re.sub(r'\*(.*?)\*', r'<b>\1</b>', text)
                current_list_items.append(ListItem(Paragraph(text, bullet_style)))
                i += 1
                continue
            if '|' in line and not any(marker in line for marker in ['-|-', '|-']):
                if current_list_items:
                    story.append(ListFlowable(current_list_items, bulletType='bullet', leftIndent=20))
                    current_list_items = []
                table_data = []
                header_row = None
                while i < len(lines) and '|' in lines[i]:
                    line = lines[i].strip()
                    if not any(marker in line for marker in ['-|-', '|-']):
                        cells = [cell.strip() for cell in line.split('|')[1:-1]]
                        if cells:
                            cells = [re.sub(r'\*(.*?)\*', r'<b>\1</b>', cell) for cell in cells]
                            row = [Paragraph(cell, body_style) for cell in cells]
                            if header_row is None:
                                header_row = row
                            else:
                                table_data.append(row)
                    i += 1
                if header_row and table_data:
                    table_data.insert(0, header_row)
                    col_count = len(header_row)
                    col_width = (doc.width - 40) / col_count
                    table = Table(table_data, colWidths=[col_width] * col_count)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('TOPPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                        ('PADDING', (0, 0), (-1, -1), 8)
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 12))
                continue
            text = re.sub(r'\*(.*?)\*', r'<b>\1</b>', line)
            story.append(Paragraph(text, body_style))
            i += 1
        if current_list_items:
            story.append(ListFlowable(current_list_items, bulletType='bullet', leftIndent=20))
            current_list_items = []
        story.append(Spacer(1, 16))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def main():
    local_css()
    
    # Load model
    @st.cache_resource
    def load_model():
        return GeneticModelTrainer.load_saved_model(
            'models/genetic_model.h5',
            'models/preprocessors.pkl'
        )
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return

    # Header section
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    
    # Logo section
    st.markdown('<div class="logo-section">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("logo.png", use_container_width=True)
        except:
            pass
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
        <div class="main-header">
            <h1>Genetic Analyzer</h1>
            <p class="subtitle">Advanced DNA Analysis Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Disorder selection
    st.markdown('<label style="font-size: 0.85rem; margin-bottom: 0.5rem; display: block;">Select Genetic Disorder</label>', unsafe_allow_html=True)
    disorder = st.selectbox(
        label="",
        options=[
            "Select a disorder", "Cystic Fibrosis", "Sickle Cell Anemia", "Huntington's Disease",
            "Duchenne Muscular Dystrophy", "Tay-Sachs Disease", "Marfan Syndrome", "Hemophilia A",
            "Familial Hypercholesterolemia", "BRCA-Related Breast Cancer", "Alpha-1 Antitrypsin Deficiency"
        ],
        label_visibility="collapsed"
    )
    
    # Parent information cards
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown('<div class="glass-card"><h3>🧬 Parent 1 Profile</h3>', unsafe_allow_html=True)
        st.markdown('<label>Biological Sex</label>', unsafe_allow_html=True)
        parent1_sex = st.radio(
            label="",
            options=["MALE", "FEMALE"],
            horizontal=True,
            key="parent1_sex",
            label_visibility="collapsed"
        )
        st.markdown('<label>Genotype</label>', unsafe_allow_html=True)
        parent1_genotype = st.text_input(
            label="",
            placeholder="e.g., CC, CT, TT",
            key="parent1_genotype",
            label_visibility="collapsed"
        )
        parent1_affected = st.checkbox("Affected Status", key="parent1_affected")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card"><h3>🧬 Parent 2 Profile</h3>', unsafe_allow_html=True)
        st.markdown('<label>Biological Sex</label>', unsafe_allow_html=True)
        parent2_sex = st.radio(
            label="",
            options=["MALE", "FEMALE"],
            horizontal=True,
            key="parent2_sex",
            label_visibility="collapsed"
        )
        st.markdown('<label>Genotype</label>', unsafe_allow_html=True)
        parent2_genotype = st.text_input(
            label="",
            placeholder="e.g., CC, CT, TT",
            key="parent2_genotype",
            label_visibility="collapsed"
        )
        parent2_affected = st.checkbox("Affected Status", key="parent2_affected")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Generations input
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<label>Generations to Simulate</label>', unsafe_allow_html=True)
    generations = st.number_input(
        label="",
        min_value=1,
        value=1,
        step=1,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        analyze_button = st.button("🔬 Analyze", key="analyze_button", use_container_width=True)
    
    with col2:
        export_button = st.button("📄 Export Report", key="export_button", use_container_width=True)
    
    with col3:
        terminate_button = st.button("🚪 Exit", key="terminate_button", use_container_width=True)
    
    # Analysis logic
    if analyze_button:
        errors = validate_inputs(
            disorder, parent1_sex, parent1_genotype, parent1_affected,
            parent2_sex, parent2_genotype, parent2_affected, generations
        )
        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                input_data = {
                    'disorder': disorder,
                    'parent1_sex': 'M' if parent1_sex == "MALE" else 'F',
                    'parent2_sex': 'M' if parent2_sex == "MALE" else 'F',
                    'parent1_genotype': parent1_genotype.upper(),
                    'parent2_genotype': parent2_genotype.upper(),
                    'parent1_affected': parent1_affected,
                    'parent2_affected': parent2_affected,
                    'generations': int(generations)
                }
                
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
                
                probability = model.predict_probability(df)
                report_text = generate_report_text(input_data, df, probability)
                punnett_img_bytes = generate_punnett_image(parent1_genotype, parent2_genotype)
                
                st.markdown(f"""
                    <div class="results-card">
                        <h3>📊 Analysis Results</h3>
                        <p><strong>Disorder:</strong> {disorder}</p>
                        <p><strong>First Generation Probability:</strong> {probability*100:.2f}%</p>
                        <p><strong>Parent 1:</strong> {parent1_genotype.upper()} ({parent1_sex})</p>
                        <p><strong>Parent 2:</strong> {parent2_genotype.upper()} ({parent2_sex})</p>
                        <p><strong>Generations Simulated:</strong> {generations}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.image(punnett_img_bytes, caption="Punnett Square Family Tree", use_container_width=True)
                
                st.session_state.report_text = report_text
                st.session_state.punnett_img_bytes = punnett_img_bytes
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    
    if export_button:
        if 'report_text' in st.session_state and 'punnett_img_bytes' in st.session_state:
            pdf_bytes = create_pdf(st.session_state.report_text, st.session_state.punnett_img_bytes)
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="GrenckDevs_Genetic_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning("⚠️ Please run the analysis first before exporting.")
    
    if terminate_button:
        st.session_state.clear()
        st.markdown("""
            <div class="terminate-message">
                <p>✓ Session terminated successfully. All data has been cleared from memory.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
