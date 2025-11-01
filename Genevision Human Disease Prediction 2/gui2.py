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
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&display=swap');
        
        /* Limit container width for centered layout */
        .block-container {
            max-width: 800px;
            margin: 0 auto;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        
        .main, .block-container, [data-testid="stAppViewContainer"] {
            color: #e0e0ff !important;
            font-family: 'Orbitron', sans-serif !important;
        }
        
        /* Animated background: black, dark purple, black */
        .stApp {
            background: linear-gradient(45deg, #000000, #1F0D31, #951C1C);
            background-size: 400% 400%;
            animation: gradientAnimation 15s ease infinite;
            min-height: 100vh;
        }
        
        @keyframes gradientAnimation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        /* Hide default Streamlit header */
        [data-testid="stHeader"] {
            display: none;
        }
        
        .cyber-header {
            background: linear-gradient(90deg, #1a1a2e, #0f0f1a);
            border-bottom: 2px solid #00f0b5;
            padding: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px rgba(0, 240, 181, 0.3);
            margin-bottom: 2rem;
        }
        
        .cyber-header h1 {
            background: linear-gradient(to right, #00f0b5, #ff00a0);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 0 0 5px rgba(0, 240, 181, 0.5);
            letter-spacing: 1px;
            font-family: 'Orbitron', sans-serif;
        }
        
        /* Form styling */
        .stSelectbox, .stTextInput, .stNumberInput {
            margin-bottom: 1rem;
        }
        
        div[data-baseweb="select"],
        div[data-baseweb="base-input"] {
            background-color: rgba(10, 10, 18, 0.7) !important;
            border: 1px solid #00f0b5 !important;
            border-radius: 5px !important;
        }
        
        div[data-baseweb="select"] input,
        div[data-baseweb="base-input"] input {
            color: #e0e0ff !important;
            font-family: 'Orbitron', sans-serif !important;
        }
        
        /* Label styling */
        label, .stRadio label, .stCheckbox label {
            color: #00f0b5 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            font-weight: bold !important;
            font-family: 'Orbitron', sans-serif !important;
        }
        
        /* Parent card styling */
        .parent-card {
            background: linear-gradient(135deg, rgba(26, 26, 46, 0.9), rgba(15, 15, 25, 0.8));
            border: 1px solid rgba(0, 240, 181, 0.3);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 15px rgba(0, 240, 181, 0.2);
            position: relative;
            overflow: hidden;
        }
        
        .parent-card h3 {
            color: #ff00a0 !important;
            text-shadow: 0 0 5px rgba(255, 0, 160, 0.5);
            margin-bottom: 1.2rem;
            font-family: 'Orbitron', sans-serif !important;
        }
        
        /* Button styling */
        .stButton > button,
        [data-testid="baseButton-secondary"] {
            font-family: 'Orbitron', sans-serif !important;
            font-weight: bold !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            padding: 0.75rem 1.5rem !important;
            border: none !important;
            border-radius: 5px !important;
            transition: all 0.3s ease !important;
            color: #ffffff !important;
        }
        
        .cyan-button button {
            background-color: #00f0b5 !important;
            box-shadow: 0 0 15px rgba(0, 240, 181, 0.5) !important;
        }
        
        .magenta-button button {
            background-color: #ff00a0 !important;
            box-shadow: 0 0 15px rgba(255, 0, 160, 0.5) !important;
        }
        
        .purple-button button {
            background-color: #9900ff !important;
            box-shadow: 0 0 15px rgba(153, 0, 255, 0.5) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.03) !important;
        }
        
        /* Other elements */
        .css-10trblm {
            margin-bottom: 1rem;
            color: #00f0b5 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: bold;
        }
        
        /* Hide Streamlit footer */
        footer {
            display: none !important;
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
    query = f"""You are a report generator working for GrenckDevs. Give the text generated in a neat and professional layout, like a medical report with the first person speaker being a professional genetic analyst at GrenckDevs (e.g., 'We at GrenckDevs have meticulously analysed your test results...'). Modify the layout of the text you generate for the given prompt to look like a professional medical report given by medical institutions. The statistical data must be grouped into one layout and generate a table-like layout for the numerical data. At the conclusion of the report, suggest possible solutions based solely on the given input, even if the input is not accurate. Keep the report comprehensive yet professional, with a maximum of 5 pages, matching the layout of an A4-sized sheet. Use the word 'GrenckDevs' wherever possible.

    Template:
    GrenckDevs Genetic Analysis Report

    I. Executive Summary
    - [Provide a brief summary of the analysis, key findings, and overall prediction.]

    II. Introduction
    - Background: [Describe the genetic disorder and its relevance.]
    - Purpose: [Explain the objective of the report and what it aims to achieve.]

    III. Methodology
    - Data Collection: [Detail the data inputs such as disorder selection, parental information, and generational prediction.]
    - Genetic Analysis: [Describe how parental genotypes and inheritance patterns were evaluated.]
    - Statistical Computation: [Outline the computational models and probability calculations used.]
    - Quality Assurance: [Explain the validation and review process.]

    IV. Detailed Patient and Parental Information
    - Disorder Under Evaluation: [Insert disorder name.]
    - Parent 1 Details:
      - Sex: [Insert Sex]
      - Genotype: [Insert Genotype]
      - Affected Status: [Insert Yes/No]
    - Parent 2 Details:
      - Sex: [Insert Sex]
      - Genotype: [Insert Genotype]
      - Affected Status: [Insert Yes/No]
    - Generational Prediction: [Insert number of generations predicted.]

    V. Statistical Data and Findings
    - Table Format Example:
      | **Parameter**                     | **Value**               |
      |-----------------------------------|-------------------------|
      | **Disorder**                      | [Insert Disorder Name]  |
      | **First Generation Probability**  | [Insert Percentage]     |
      | **Parent 1 Genotype**             | [Insert Genotype]       |
      | **Parent 1 Affected Status**      | [Insert Yes/No]         |
      | **Parent 2 Genotype**             | [Insert Genotype]       |
      | **Parent 2 Affected Status**      | [Insert Yes/No]         |
      | **Generations Predicted**         | [Insert Number]         |
    - [Add any additional statistical commentary.]

    VI. In-Depth Analysis and Discussion
    A. Genetic Pattern Analysis
    - [Discuss the allele distribution and genetic transmission relevant to the disorder.]
    B. Statistical Validity and Model Considerations
    - [Explain the robustness of the statistical model, algorithm validation, and any model limitations.]

    VII. Recommendations and Future Directions
    - Genetic Counseling: [Recommendation for genetic counseling and further family evaluations.]
    - Additional Diagnostic Testing: [Suggestions for any additional genetic tests.]
    - Regular Monitoring: [Outline recommendations for ongoing monitoring and follow-up.]
    - Multidisciplinary Consultation: [Advice on consulting with specialists as needed.]

    VIII. Conclusion
    - [Summarize the key findings and overall risk prediction, and reinforce the recommended next steps.]

    Report Prepared By: GrenckDevs
    Contact Information:
    GrenckDevs
    Email: grenck.devs@gmail.com
    Phone: +91 1234567890

    *This report is intended to serve as a detailed reference document for healthcare professionals and genetic counselors. All conclusions are based solely on the provided data and the analytical models employed by GrenckDevs.*

    Use this data:
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
    Probability: {probability*100:.2f}%
    """
    response = ollama.chat(model='custom_finetuned', messages=[
        {"role": "user", "content": query}
    ])
    report_text = response['message']['content']
    return report_text.split('>', 2)[-1] if '>' in report_text else report_text

def generate_punnett_image(father_genotype, mother_genotype):
    """Generate a Punnett square visualization as image bytes."""
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

    # Plot family tree
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

    # Save to BytesIO buffer
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

    # Centered logo
    col1, col2, col3 = st.columns(3)
    with col2:
        try:
            st.image("logo.png", width=5000)
        except:
            st.warning("Logo image 'logo.png' not found. Place it in the same directory.")

    # Custom header
    st.markdown("""
        <div class="cyber-header">
            <h1>GRENCKDEVS GENETIC ANALYZER</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Form container
    with st.container():
        st.markdown('<label style="font-size: 16px;">SELECT GENETIC DISORDER:</label>', unsafe_allow_html=True)
        disorder = st.selectbox(
            label="",
            options=[
                "Select a disorder", "Cystic Fibrosis", "Sickle Cell Anemia", "Huntington's Disease",
                "Duchenne Muscular Dystrophy", "Tay-Sachs Disease", "Marfan Syndrome", "Hemophilia A",
                "Familial Hypercholesterolemia", "BRCA-Related Breast Cancer", "Alpha-1 Antitrypsin Deficiency"
            ],
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3>PARENT 1 DNA PROFILE</h3>', unsafe_allow_html=True)
            st.markdown('<label>SEX:</label>', unsafe_allow_html=True)
            parent1_sex = st.radio(
                label="",
                options=["MALE", "FEMALE"],
                horizontal=True,
                key="parent1_sex",
                label_visibility="collapsed"
            )
            st.markdown('<label>GENOTYPE:</label>', unsafe_allow_html=True)
            parent1_genotype = st.text_input(
                label="",
                placeholder="e.g., CC, CT, TT",
                key="parent1_genotype",
                label_visibility="collapsed"
            )
            parent1_affected = st.checkbox("AFFECTED", key="parent1_affected")
        
        with col2:
            st.markdown('<h3>PARENT 2 DNA PROFILE</h3>', unsafe_allow_html=True)
            st.markdown('<label>SEX:</label>', unsafe_allow_html=True)
            parent2_sex = st.radio(
                label="",
                options=["MALE", "FEMALE"],
                horizontal=True,
                key="parent2_sex",
                label_visibility="collapsed"
            )
            st.markdown('<label>GENOTYPE:</label>', unsafe_allow_html=True)
            parent2_genotype = st.text_input(
                label="",
                placeholder="e.g., CC, CT, TT",
                key="parent2_genotype",
                label_visibility="collapsed"
            )
            parent2_affected = st.checkbox("AFFECTED", key="parent2_affected")
        
        st.markdown('<label>GENERATIONS TO SIMULATE:</label>', unsafe_allow_html=True)
        generations = st.number_input(
            label="",
            min_value=1,
            placeholder="Enter number of generations",
            label_visibility="collapsed"
        )
        
        st.markdown('<div style="margin-top: 2.5rem; display: flex; gap: 1rem;">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write('<div class="cyan-button">', unsafe_allow_html=True)
            analyze_button = st.button("ANALYZE DNA", key="analyze_button")
            print("debug print")
            st.write('</div>', unsafe_allow_html=True)
        
        with col2:
            st.write('<div class="magenta-button">', unsafe_allow_html=True)
            export_button = st.button("EXPORT DATA", key="export_button")
            st.write('</div>', unsafe_allow_html=True)
        
        with col3:
            st.write('<div class="purple-button">', unsafe_allow_html=True)
            terminate_button = st.button("TERMINATE", key="terminate_button")
            st.write('</div>', unsafe_allow_html=True)
        
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
                    
                    st.markdown(f"""
                        <div style="margin-top: 2rem; padding: 1.5rem; border-radius: 10px; 
                                   background: linear-gradient(135deg, rgba(26, 26, 46, 0.9), rgba(15, 15, 25, 0.8));
                                   border: 1px solid rgba(0, 240, 181, 0.3); box-shadow: 0 0 15px rgba(0, 240, 181, 0.2);">
                            <h3 style="color: #00f0b5; margin-top: 0;">DNA ANALYSIS RESULTS</h3>
                            <p>Disorder: {disorder}</p>
                            <p>First Generation Probability: {probability*100:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(punnett_img_bytes, caption="Punnett Square Family Tree")
                    
                    st.session_state.report_text = report_text
                    st.session_state.punnett_img_bytes = punnett_img_bytes
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        
        if export_button:
            if 'report_text' in st.session_state and 'punnett_img_bytes' in st.session_state:
                pdf_bytes = create_pdf(st.session_state.report_text, st.session_state.punnett_img_bytes)
                st.download_button(
                    label="Download Report",
                    data=pdf_bytes,
                    file_name="Genetic_Report.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("Please run the analysis first.")
        
        if terminate_button:
            st.session_state.clear()
            st.markdown("""
                <div style="margin-top: 2rem; padding: 1rem; border-radius: 10px; 
                           background: rgba(153, 0, 255, 0.1); border: 1px solid #9900ff;">
                    <p style="margin: 0; color: #9900ff;">Session terminated. All data has been cleared from memory.</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()