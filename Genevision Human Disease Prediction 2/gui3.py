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
from main_app import GeneticModelTrainer  # Assuming this is where the model is defined








import streamlit as st
import pandas as pd
# ... other standard library imports ...

# Add these lines to configure TensorFlow for CPU usage
import tensorflow as tf
import os
try:
    # Disable all GPUs
    tf.config.set_visible_devices([], 'GPU')
    if tf.config.list_logical_devices('GPU'):
        st.warning("TensorFlow still sees GPUs after attempting to disable. Forcing CPU via environment variable.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # For TensorFlow, -1 typically means use CPU
    else:
        print("TensorFlow configured to not use GPUs.")
except Exception as e:
    st.warning(f"Could not configure TensorFlow to disable GPUs: {e}. TensorFlow will use its default. If issues persist, this might be related.")


import matplotlib.pyplot as plt # Moved after TF config
import networkx as nx # Moved after TF config
# ... rest of your imports like reportlab, ollama ...
from main_app import GeneticModelTrainer # Ensure this is imported after TF configuration
# ... rest of your gui2.py code











# Page configuration with centered layout
st.set_page_config(
    page_title="GrenckDevs Genetic Analysis",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def local_css():
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');
        
        /* Global App Styling */
        .stApp {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 50%, #1a1a1a 100%);
            color: #f5f5f0;
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide default Streamlit elements */
        [data-testid="stHeader"] {
            display: none;
        }
        
        footer {
            display: none !important;
        }
        
        /* Main container styling */
        .block-container {
            max-width: 1200px;
            padding: 2rem 1rem;
            margin: 0 auto;
        }
        
        /* Hero Header Section */
        .hero-header {
            background: linear-gradient(135deg, #2c2c2c 0%, #1a1a1a 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 3rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            text-align: center;
            border: 1px solid rgba(139, 125, 107, 0.2);
        }
        
        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 3.5rem;
            font-weight: 600;
            color: #f5f5f0;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            line-height: 1.2;
        }
        
        .hero-subtitle {
            font-size: 1.2rem;
            color: #8b7d6b;
            font-weight: 300;
            margin-bottom: 0;
            letter-spacing: 0.5px;
        }
        
        /* Card-based sections */
        .medical-card {
            background: linear-gradient(135deg, #2a2a2a 0%, #1f1f1f 100%);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(139, 125, 107, 0.15);
            transition: all 0.3s ease;
        }
        
        .medical-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        }
        
        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.8rem;
            font-weight: 500;
            color: #f5f5f0;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #8b7d6b;
            display: inline-block;
        }
        
        .parent-section {
            background: linear-gradient(135deg, #2d2d2d 0%, #242424 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid #8b7d6b;
        }
        
        .parent-title {
            font-family: 'Inter', sans-serif;
            font-size: 1.3rem;
            font-weight: 600;
            color: #d4c4a8;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Form Elements */
        .stSelectbox label, .stTextInput label, .stNumberInput label, .stRadio label {
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            color: #d4c4a8 !important;
            font-size: 0.9rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Input field styling */
        div[data-baseweb="select"],
        div[data-baseweb="base-input"] {
            background-color: rgba(45, 45, 45, 0.8) !important;
            border: 1px solid rgba(139, 125, 107, 0.3) !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-baseweb="select"]:hover,
        div[data-baseweb="base-input"]:hover {
            border-color: rgba(139, 125, 107, 0.6) !important;
            box-shadow: 0 0 10px rgba(139, 125, 107, 0.2) !important;
        }
        
        div[data-baseweb="select"] input,
        div[data-baseweb="base-input"] input {
            color: #f5f5f0 !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 400 !important;
        }
        
        /* Radio button styling */
        .stRadio > div {
            background: rgba(45, 45, 45, 0.5);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(139, 125, 107, 0.2);
        }
        
        .stRadio label {
            color: #f5f5f0 !important;
        }
        
        /* Checkbox styling */
        .stCheckbox {
            background: rgba(45, 45, 45, 0.5);
            padding: 0.8rem;
            border-radius: 8px;
            border: 1px solid rgba(139, 125, 107, 0.2);
            margin-top: 1rem;
        }
        
        .stCheckbox label {
            color: #f5f5f0 !important;
            font-weight: 500 !important;
        }
        
        /* Button styling */
        .stButton > button {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            padding: 0.8rem 2rem !important;
            border: none !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
            cursor: pointer !important;
        }
        
        .primary-button button {
            background: linear-gradient(135deg, #8b7d6b 0%, #6b5d4a 100%) !important;
            color: #f5f5f0 !important;
        }
        
        .secondary-button button {
            background: linear-gradient(135deg, #d4c4a8 0%, #b8a888 100%) !important;
            color: #2a2a2a !important;
        }
        
        .danger-button button {
            background: linear-gradient(135deg, #8b6b6b 0%, #6b4a4a 100%) !important;
            color: #f5f5f0 !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Action buttons container */
        .action-buttons {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-top: 2rem;
            padding: 2rem 0;
        }
        
        /* Results section */
        .results-section {
            background: linear-gradient(135deg, #2a2a2a 0%, #1f1f1f 100%);
            border-radius: 16px;
            padding: 2rem;
            margin-top: 2rem;
            border: 1px solid rgba(139, 125, 107, 0.2);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        
        /* Success/Error messages */
        .stSuccess {
            background: rgba(107, 139, 107, 0.2) !important;
            border: 1px solid rgba(107, 139, 107, 0.4) !important;
            border-radius: 8px !important;
            color: #d4f5d4 !important;
        }
        
        .stError {
            background: rgba(139, 107, 107, 0.2) !important;
            border: 1px solid rgba(139, 107, 107, 0.4) !important;
            border-radius: 8px !important;
            color: #f5d4d4 !important;
        }
        
        /* Spacing utilities */
        .spacer-small {
            height: 1rem;
        }
        
        .spacer-medium {
            height: 2rem;
        }
        
        .spacer-large {
            height: 3rem;
        }
        
        /* Enhanced visual elements */
        .stSelectbox > div > div {
            background: rgba(45, 45, 45, 0.8) !important;
            border: 1px solid rgba(139, 125, 107, 0.3) !important;
        }
        
        .stNumberInput > div > div > input {
            background: rgba(45, 45, 45, 0.8) !important;
            border: 1px solid rgba(139, 125, 107, 0.3) !important;
            color: #f5f5f0 !important;
        }
        
        /* Info boxes */
        .stInfo {
            background: rgba(139, 125, 107, 0.1) !important;
            border: 1px solid rgba(139, 125, 107, 0.3) !important;
            border-radius: 8px !important;
            color: #d4c4a8 !important;
        }
        
        .stWarning {
            background: rgba(139, 139, 107, 0.1) !important;
            border: 1px solid rgba(139, 139, 107, 0.3) !important;
            border-radius: 8px !important;
            color: #d4d4a8 !important;
        }
        
        /* Download button special styling */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #8b7d6b 0%, #6b5d4a 100%) !important;
            color: #f5f5f0 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.8rem 2rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            transition: all 0.3s ease !important;
        }
        
        .stDownloadButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2.5rem;
            }
            
            .block-container {
                padding: 1rem 0.5rem;
            }
            
            .medical-card {
                padding: 1.5rem;
            }
            
            .action-buttons {
                flex-direction: column;
                align-items: center;
            }
            
            .parent-section {
                margin-bottom: 2rem;
            }
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
    # query = f"""You are a report generator working for GrenckDevs. Give the text generated in a neat and professional layout, like a medical report with the first person speaker being a professional genetic analyst at GrenckDevs (e.g., 'We at GrenckDevs have meticulously analysed your test results...'). Modify the layout of the text you generate for the given prompt to look like a professional medical report given by medical institutions. The statistical data must be grouped into one layout and generate a table-like layout for the numerical data. At the conclusion of the report, suggest possible solutions based solely on the given input, even if the input is not accurate. Keep the report comprehensive yet professional, with a maximum of 5 pages, matching the layout of an A4-sized sheet. Use the word 'GrenckDevs' wherever possible.

    # Template:
    # GrenckDevs Genetic Analysis Report

    # I. Executive Summary
    # - [Provide a brief summary of the analysis, key findings, and overall prediction.]

    # II. Introduction
    # - Background: [Describe the genetic disorder and its relevance.]
    # - Purpose: [Explain the objective of the report and what it aims to achieve.]

    # III. Methodology
    # - Data Collection: [Detail the data inputs such as disorder selection, parental information, and generational prediction.]
    # - Genetic Analysis: [Describe how parental genotypes and inheritance patterns were evaluated.]
    # - Statistical Computation: [Outline the computational models and probability calculations used.]
    # - Quality Assurance: [Explain the validation and review process.]

    # IV. Detailed Patient and Parental Information
    # - Disorder Under Evaluation: [Insert disorder name.]
    # - Parent 1 Details:
    #   - Sex: [Insert Sex]
    #   - Genotype: [Insert Genotype]
    #   - Affected Status: [Insert Yes/No]
    # - Parent 2 Details:
    #   - Sex: [Insert Sex]
    #   - Genotype: [Insert Genotype]
    #   - Affected Status: [Insert Yes/No]
    # - Generational Prediction: [Insert number of generations predicted.]

    # V. Statistical Data and Findings
    # - Table Format Example:
    #   | *Parameter*                     | *Value*               |
    #   |-----------------------------------|-------------------------|
    #   | *Disorder*                      | [Insert Disorder Name]  |
    #   | *First Generation Probability*  | [Insert Percentage]     |
    #   | *Parent 1 Genotype*             | [Insert Genotype]       |
    #   | *Parent 1 Affected Status*      | [Insert Yes/No]         |
    #   | *Parent 2 Genotype*             | [Insert Genotype]       |
    #   | *Parent 2 Affected Status*      | [Insert Yes/No]         |
    #   | *Generations Predicted*         | [Insert Number]         |
    # - [Add any additional statistical commentary.]

    # VI. In-Depth Analysis and Discussion
    # A. Genetic Pattern Analysis
    # - [Discuss the allele distribution and genetic transmission relevant to the disorder.]
    # B. Statistical Validity and Model Considerations
    # - [Explain the robustness of the statistical model, algorithm validation, and any model limitations.]

    # VII. Recommendations and Future Directions
    # - Genetic Counseling: [Recommendation for genetic counseling and further family evaluations.]
    # - Additional Diagnostic Testing: [Suggestions for any additional genetic tests.]
    # - Regular Monitoring: [Outline recommendations for ongoing monitoring and follow-up.]
    # - Multidisciplinary Consultation: [Advice on consulting with specialists as needed.]

    # VIII. Conclusion
    # - [Summarize the key findings and overall risk prediction, and reinforce the recommended next steps.]

    # Report Prepared By: GrenckDevs
    # Contact Information:
    # GrenckDevs
    # Email: grenck.devs@gmail.com
    # Phone: +91 1234567890

    # This report is intended to serve as a detailed reference document for healthcare professionals and genetic counselors. All conclusions are based solely on the provided data and the analytical models employed by GrenckDevs.

    # Use this data:
    # Parent 1 Information:
    # Sex: {df['parent1_sex'].values[0]}
    # Genotype: {df['parent1_genotype'].values[0]}
    # Affected: {df['parent1_affected'].values[0]}

    # Parent 2 Information:
    # Sex: {df['parent2_sex'].values[0]}
    # Genotype: {df['parent2_genotype'].values[0]}
    # Affected: {df['parent2_affected'].values[0]}

    # Prediction Results:
    # Disorder: {input_data['disorder']}
    # Probability: {probability*100:.2f}%
    # """


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
    










    
    response = ollama.chat(model='phi3:latest', messages=[
        {"role": "user", "content": query}
    ])
    report_text = response['message']['content']
    return report_text.split('>', 2)[-1] if '>' in report_text else report_text

def generate_punnett_image(father_genotype, mother_genotype):
    """Generate a Punnett square visualization as image bytes."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    from io import BytesIO
    
    def get_color(genotype):
        if genotype == "CC":
            return "green"   # Normal
        elif genotype == "CT":
            return "orange"  # Carrier
        elif genotype == "TT":
            return "red"     # Affected
        return "grey"       # Unknown

    # Generate offspring genotypes
    offspring = []
    father_alleles = list(father_genotype)
    mother_alleles = list(mother_genotype)
    for father_allele in father_alleles:
        for mother_allele in mother_alleles:
            offspring.append("".join(sorted(father_allele + mother_allele)))

    colors = [get_color(gen) for gen in offspring]
    father_color = get_color(father_genotype)
    mother_color = get_color(mother_genotype)

    # Create directed graph for family tree
    graph = nx.DiGraph()
    graph.add_node("parent1", shape="square", label=f"Father ({father_genotype})", color=father_color)
    graph.add_node("parent2", shape="circle", label=f"Mother ({mother_genotype})", color=mother_color)

    offspring_labels = [f"Child {i+1}\n({gen})" for i, gen in enumerate(offspring)]
    for i, label in enumerate(offspring_labels):
        graph.add_node(label, shape="square", label=label, color=colors[i])
        graph.add_edge("parent1", label)
        graph.add_edge("parent2", label)

    # Define node positions
    pos = {
        "parent1": (0, 2), "parent2": (2, 2),
        offspring_labels[0]: (-1, 0), offspring_labels[1]: (1, 0),
        offspring_labels[2]: (3, 0), offspring_labels[3]: (5, 0)
    }

    # Plot family tree with thread-safe backend
    fig, ax = plt.subplots(figsize=(8, 6))
    node_colors = [graph.nodes[node]["color"] for node in graph.nodes]
    node_labels = {n: graph.nodes[n]["label"] for n in graph.nodes}

    nx.draw(graph, pos, with_labels=True, labels=node_labels, node_size=3000,
            node_color=node_colors, edge_color="black", font_size=10, font_weight="bold", ax=ax)

    ax.set_title("Punnett Square Family Tree")
    legend_handles = [
        mpatches.Patch(color='green', label='Normal (CC)'),
        mpatches.Patch(color='orange', label='Carrier (CT)'),
        mpatches.Patch(color='red', label='Affected (TT)')
    ]
    ax.legend(handles=legend_handles, loc='upper right')

    # Save to BytesIO buffer
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)
    plt.clf()  # Clear the figure
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

    # Add Punnett square image
    img_buffer = BytesIO(punnett_img_bytes)
    story.append(Image(img_buffer, width=400, height=300))
    story.append(Spacer(1, 20))

    # Process report text
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
                # Corrected regex substitution here
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
                            # Corrected regex substitution for table cells
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
            # Corrected regex substitution for general text
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

    # Hero Header Section
    st.markdown("""
        <div class="hero-header">
            <h1 class="hero-title">GrenckDevs Genetic Analysis</h1>
            <p class="hero-subtitle">Advanced Genetic Disorder Prediction & Family Planning Insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Logo section (if available)
    try:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("logo.png", width=300)
    except:
        pass  # Logo not found, continue without it
    
    # Main Analysis Form
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Genetic Analysis Configuration</h2>', unsafe_allow_html=True)
    
    # Disorder Selection
    st.markdown('<div class="spacer-small"></div>', unsafe_allow_html=True)
    disorder = st.selectbox(
        label="Select Genetic Disorder",
        options=[
            "Select a disorder", "Cystic Fibrosis", "Sickle Cell Anemia", "Huntington's Disease",
            "Duchenne Muscular Dystrophy", "Tay-Sachs Disease", "Marfan Syndrome", "Hemophilia A",
            "Familial Hypercholesterolemia", "BRCA-Related Breast Cancer", "Alpha-1 Antitrypsin Deficiency"
        ]
    )
    
    st.markdown('<div class="spacer-medium"></div>', unsafe_allow_html=True)
    
    # Parent Information Section
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="parent-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="parent-title">Parent 1 Profile</h3>', unsafe_allow_html=True)
        
        parent1_sex = st.radio(
            label="Biological Sex",
            options=["MALE", "FEMALE"],
            horizontal=True,
            key="parent1_sex"
        )
        
        parent1_genotype = st.text_input(
            label="Genotype",
            placeholder="e.g., CC, CT, TT",
            key="parent1_genotype",
            help="Enter the genetic variant (allele combination)"
        )
        
        parent1_affected = st.checkbox("Currently Affected by Disorder", key="parent1_affected")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="parent-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="parent-title">Parent 2 Profile</h3>', unsafe_allow_html=True)
        
        parent2_sex = st.radio(
            label="Biological Sex",
            options=["MALE", "FEMALE"],
            horizontal=True,
            key="parent2_sex"
        )
        
        parent2_genotype = st.text_input(
            label="Genotype",
            placeholder="e.g., CC, CT, TT",
            key="parent2_genotype",
            help="Enter the genetic variant (allele combination)"
        )
        
        parent2_affected = st.checkbox("Currently Affected by Disorder", key="parent2_affected")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-medium"></div>', unsafe_allow_html=True)
    
    # Generations Input
    generations = st.number_input(
        label="Number of Generations to Simulate",
        min_value=1,
        max_value=10,
        value=1,
        help="Specify how many future generations to analyze"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
        
    # Action Buttons
    st.markdown('<div class="spacer-large"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            st.markdown('<div class="primary-button">', unsafe_allow_html=True)
            analyze_button = st.button("🧬 Run Analysis", key="analyze_button", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_btn2:
            st.markdown('<div class="secondary-button">', unsafe_allow_html=True)
            export_button = st.button("📄 Export Report", key="export_button", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_btn3:
            st.markdown('<div class="danger-button">', unsafe_allow_html=True)
            terminate_button = st.button("🚪 Exit", key="terminate_button", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
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
                    
                    # Results Section with elegant styling
                    st.markdown('<div class="spacer-large"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="results-section">', unsafe_allow_html=True)
                    st.markdown('<h2 class="section-title">Analysis Results</h2>', unsafe_allow_html=True)
                    
                    st.success("✅ Genetic analysis completed successfully!")
                    
                    # Results display in elegant format
                    col_res1, col_res2 = st.columns(2, gap="large")
                    
                    with col_res1:
                        st.markdown(f"""
                        **Genetic Disorder Analyzed:**  
                        `{disorder}`
                        
                        **First Generation Risk Probability:**  
                        `{probability*100:.2f}%`
                        """)
                    
                    with col_res2:
                        st.markdown(f"""
                        **Analysis Parameters:**  
                        • Parent 1: {input_data['parent1_sex']} ({input_data['parent1_genotype']})  
                        • Parent 2: {input_data['parent2_sex']} ({input_data['parent2_genotype']})  
                        • Generations Simulated: {input_data['generations']}
                        """)
                    
                    st.markdown('<div class="spacer-medium"></div>', unsafe_allow_html=True)
                    
                    # Display Punnett square with elegant styling
                    st.markdown("**Genetic Inheritance Visualization:**")
                    st.image(punnett_img_bytes, caption="Punnett Square Family Tree Analysis", use_column_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.session_state.report_text = report_text
                    st.session_state.punnett_img_bytes = punnett_img_bytes
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        
        if export_button:
            if 'report_text' in st.session_state and 'punnett_img_bytes' in st.session_state:
                pdf_bytes = create_pdf(st.session_state.report_text, st.session_state.punnett_img_bytes)
                st.markdown('<div class="spacer-medium"></div>', unsafe_allow_html=True)
                st.markdown('<div class="medical-card">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">📄 Report Export</h3>', unsafe_allow_html=True)
                st.success("Report generated successfully! Click below to download.")
                st.download_button(
                    label="📥 Download Comprehensive Report (PDF)",
                    data=pdf_bytes,
                    file_name="GrenckDevs_Genetic_Analysis_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please run the genetic analysis first before exporting the report.")
        
        if terminate_button:
            st.session_state.clear()
            st.markdown('<div class="spacer-medium"></div>', unsafe_allow_html=True)
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            st.info("🔒 Session terminated successfully. All sensitive genetic data has been securely cleared from memory.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer Section
    st.markdown('<div class="spacer-large"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="medical-card" style="text-align: center; background: linear-gradient(135deg, #1f1f1f 0%, #2a2a2a 100%);">
            <div style="color: #8b7d6b; font-size: 0.9rem; line-height: 1.6;">
                <p><strong>GrenckDevs Genetic Analysis Platform</strong></p>
                <p>Advanced computational genetics • Evidence-based predictions • Secure data processing</p>
                <p style="font-size: 0.8rem; margin-top: 1rem; opacity: 0.7;">
                    This tool provides educational insights based on genetic modeling. 
                    Always consult with qualified genetic counselors and healthcare professionals for medical decisions.
                </p>
                <p style="font-size: 0.8rem; opacity: 0.6;">
                    © 2024 GrenckDevs • Contact: grenck.devs@gmail.com
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()