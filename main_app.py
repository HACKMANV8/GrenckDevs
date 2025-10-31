#updated user
import ollama
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pickle
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.units import inch
import re


class GeneticModelTrainer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

        # Define categorical and numerical columns
        self.categorical_columns = [
            'disorder', 'chromosome', 'snp_id',
            'parent1_sex', 'parent2_sex',
            'parent1_genotype', 'parent2_genotype',
            'inheritance'
        ]
        self.numerical_columns = [
            'position', 'parent1_affected', 'parent2_affected',
            'penetrance', 'mutation_rate'
        ]

    def prepare_features(self, data: pd.DataFrame, training: bool = True) -> np.ndarray:
        """Prepare features for training or prediction."""
        # Create a copy to avoid modifying the original data
        processed_data = data.copy()

        # Encode categorical variables
        for col in self.categorical_columns:
            if col in processed_data.columns:
                if training:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    processed_data[col] = self.label_encoders[col].fit_transform(processed_data[col])
                else:
                    processed_data[col] = self.label_encoders[col].transform(processed_data[col])

        # Combine features
        features = processed_data[self.categorical_columns + self.numerical_columns].values

        # Scale features
        if training:
            return self.scaler.fit_transform(features)
        else:
            return self.scaler.transform(features)

    def create_model(self, input_shape: int) -> Sequential:
        """Create the neural network model."""
        model = Sequential([
            # Input layer
            Dense(256, input_shape=(input_shape,), activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            # Hidden layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),

            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),

            # Output layer
            Dense(1, activation='sigmoid')
        ])

        # Compile model with custom learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['mae', 'mse','accuracy']
        )

        return model

    def train_model(self, data: pd.DataFrame, validation_split: float = 0.2):
        """Train the genetic disorder prediction model."""
        # Prepare features and target
        X = self.prepare_features(data, training=True)
        y = data['offspring_probability'].values

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=data['inheritance']
        )

        # Create model
        self.model = self.create_model(X_train.shape[1])

        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                mode='min'
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # Load best model
        self.model = load_model('best_model.h5')

        # Evaluate model
        self.evaluate_model(X_val, y_val)

        return self.history

    def evaluate_model(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate the model and print metrics."""
        # Get predictions
        y_pred = self.model.predict(X_val)

        # Calculate metrics
        mse = np.mean((y_val - y_pred.flatten()) ** 2)
        mae = np.mean(np.abs(y_val - y_pred.flatten()))
        rmse = np.sqrt(mse)

        print("\nModel Evaluation Metrics:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")

        # Calculate R-squared
        ss_res = np.sum((y_val - y_pred.flatten()) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        print(f"R-squared Score: {r2:.4f}")

    def predict_probability(self, input_data: pd.DataFrame) -> float:
        """Predict probability for new input data."""
        X = self.prepare_features(input_data, training=False)
        return self.model.predict(X)[0][0]

    def save_model(self, model_path: str, encoders_path: str):
        """Save the model and preprocessing objects."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save model
        save_model(self.model, model_path)

        # Save encoders and scaler
        preprocessing_objects = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }

        with open(encoders_path, 'wb') as f:
            pickle.dump(preprocessing_objects, f)

    @classmethod
    def load_saved_model(cls, model_path: str, encoders_path: str):
        """Load a saved model and preprocessing objects."""
        trainer = cls()

        # Load model
        trainer.model = load_model(model_path)

        # Load preprocessing objects
        with open(encoders_path, 'rb') as f:
            preprocessing_objects = pickle.load(f)
            trainer.label_encoders = preprocessing_objects['label_encoders']
            trainer.scaler = preprocessing_objects['scaler']
            trainer.categorical_columns = preprocessing_objects['categorical_columns']
            trainer.numerical_columns = preprocessing_objects['numerical_columns']

        return trainer
def get_user_input():
    """Get input from user through command line."""
    print("\n=== Genetic Disorder Prediction System ===\n")

    # List available disorders
    disorders = [
        'Cystic Fibrosis', 'Sickle Cell Anemia', 'Huntington Disease',
        'Duchenne Muscular Dystrophy', 'Tay-Sachs Disease	', 'Marfan Syndrome', 'Hemophilia A',
        'Familial Hypercholesterolemia', 'BRCA-Related Breast Cancer', 'Alpha-1 Antitrypsin Deficiency'
    ]

    print("Available Disorders:")
    for i, disorder in enumerate(disorders, 1):
        print(f"{i}. {disorder.replace('_', ' ')}")
    
    # Get disorder choice
    while True:
        try:
            choice = int(input("\nEnter the number of the disorder (1-10): "))
            aluma=choice
            if 1 <= choice <= 10:
                disorder = disorders[choice-1]
                break
            print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")

    # Get parent information
    print("\nParent 1 Information:")
    parent1_sex = input("Sex (M/F): ").upper()
    parent1_genotype = input("Genotype (e.g., CC, CT, TT): ").upper()
    parent1_affected = input("Affected by disorder? (y/n): ").lower() == 'y'

    print("\nParent 2 Information:")
    parent2_sex = input("Sex (M/F): ").upper()
    parent2_genotype = input("Genotype (e.g., CC, CT, TT): ").upper()
    parent2_affected = input("Affected by disorder? (y/n): ").lower() == 'y'

    # Get number of generations
    while True:
        try:
            generations = int(input("\nEnter number of generations to predict (1-10): "))
            if 1 <= generations <= 10:
                break
            print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")

    return {
        'disorder': disorder,
        'parent1_sex': parent1_sex,
        'parent1_genotype': parent1_genotype,
        'parent1_affected': parent1_affected,
        'parent2_sex': parent2_sex,
        'parent2_genotype': parent2_genotype,
        'parent2_affected': parent2_affected,
        'generations': generations,
        
    }

def main():
    try:
        # Load the trained model
        print("Loading model...")
        model = GeneticModelTrainer.load_saved_model(
            'models/genetic_model.h5',
            'models/preprocessors.pkl'
        )

        # Get user input
        input_data = get_user_input()

        # Prepare input for model
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

        # Get prediction for first generation
        probability = model.predict_probability(df)

        # Display results
        print("\n=== Prediction Results ===")
        print(f"\nDisorder: {input_data['disorder'].replace('_', ' ')}")
        print(f"First Generation Probability: {probability*100:.2f}%")

        # Calculate for additional generations if requested
        if input_data['generations'] > 1:
            current_prob = probability
            for gen in range(2, input_data['generations'] + 1):
                # Simple population genetics calculation
                mutation_rate = 0.001
                selection_coeff = 0.1
                current_prob = current_prob * (1 - selection_coeff) + mutation_rate * (1 - current_prob)
                print(f"Generation {gen} Population Probability: {current_prob*100:.2f}%")
        #ADDING HERE
                # Single query to your custom model
        current_prob = probability
        query = f"""You are a report generator working for GrenckDevs. Give the text generated in a neat and professional layout, like a medical report with the first person speaker being a professional genetic analyst at GrenckDevs(ex. we at GrenckDevs have meticulously analysed your test results...). Modify the layout of the text you generate for given prompt to look like a professional medical report given by medical instituions. The statistical data must grouped into one layout and generate a table like layout for the numerical data. At the conclusion of the report you need to suggest the possible solution to the given prompt and only utilize the given input for the conclusion and analysis even if the input is not accurate. Avoid from prolonging the report too long, keep to a maximum of 5 pages and comprehensively yet professionally structure the text. Text layout should match the layout of an A4 sized sheet.use the word GrenckDevs wherever possible.
        make the report as long as possible. Use the context given to generate the report, dont deviate away from the provided assistant response.
        Generate a report in the format you are supposed to based on the following data:
        === Genetic Disorder Prediction system ===

        Parent 1 Information:
        Sex (M/F): {df['parent1_sex']}
        Genotype (e.g., CC, CT, TT): {df['parent1_genotype']}
        Affected by disorder? (y/n): {df['parent1_affected']}

        Parent 2 Information:
        Sex (M/F): {df['parent2_sex']}
        Genotype (e.g., CC, CT, TT): {df['parent2_genotype']}
        Affected by disorder? (y/n): {df['parent2_affected']}

        === Prediction Results ===

        Disorder: BRCA-Related Breast Cancer
        First Generation Probability: {current_prob*100:.2f}"""

        response = ollama.chat(model='custom_finetuned', messages=[
            {"role": "user", "content": query}
        ])


        sample_text=response['message']['content']
        smaller = 0
        new_text = ""

        for i, char in enumerate(sample_text):
            if char == ">":
                smaller += 1
                if smaller == 2:  # Stop at the second '>'
                    new_text = sample_text[i+1:]  # Keep only the part after the 2nd '>'
                    break

        # Store the modified string in `sample_text1`
        sample_text1 = new_text  


        def create_pdf(filename="report.pdf", report_text=""):
            doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
            story = []
            
            # Create styles
            styles = getSampleStyleSheet()

            # Fixed title style for GrenckDevs Genetic Analysis Report with specified color
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
            title_style = ParagraphStyle(  # ADDED MISSING STYLE
                'ContentTitle',
                parent=styles['Heading2'],
                fontSize=14,
                alignment=TA_CENTER,
                spaceAfter=18,
                textColor=colors.HexColor('#2C3E50'),
                fontName='Helvetica-Bold'
            )
            
            # Add fixed title with color and spacing
            story.append(Paragraph("GrenckDevs Genetic Analysis Report", fixed_title_style))
            story.append(Spacer(1, 20))

            
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
            
            # Process the text
            sections = report_text.split('---')
            current_list_items = []
            
            for section in sections:
                if not section.strip():
                    continue
                    
                lines = section.strip().split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Skip empty lines
                    if not line:
                        i += 1
                        continue
                    
                    # Handle title (first non-empty line of first section)
                    if i == 0 and section == sections[0]:
                        # Process all title lines until we hit a section marker
                        title_lines = []
                        while i < len(lines) and not lines[i].startswith('###'):
                            if lines[i].strip():
                                text = re.sub(r'\*(.*?)\*', r'<b>\1</b>', lines[i].strip())
                                title_lines.append(text)
                            i += 1
                        story.append(Paragraph('<br/>'.join(title_lines), title_style))
                        continue
                    
                    # Handle headings
                    if line.startswith('###'):
                        if current_list_items:
                            story.append(ListFlowable(current_list_items, bulletType='bullet', leftIndent=20, spaceBefore=6, spaceAfter=6))
                            current_list_items = []
                        story.append(Paragraph(line.replace('###', '').strip(), heading_style))
                        i += 1
                        continue
                    
                    # Handle bullet points
                    if line.startswith('-'):
                        text = line[1:].strip()
                        text = re.sub(r'\*(.*?)\*', r'<b>\1</b>', text)
                        current_list_items.append(ListItem(Paragraph(text, bullet_style)))
                        i += 1
                        continue
                    
                    # Handle tables
                    if '|' in line and not any(marker in line for marker in ['-|-', '|-']):
                        if current_list_items:
                            story.append(ListFlowable(current_list_items, bulletType='bullet', leftIndent=20))
                            current_list_items = []
                        
                        table_data = []
                        header_row = None
                        
                        # Collect table rows
                        while i < len(lines) and '|' in lines[i]:
                            line = lines[i].strip()
                            if not any(marker in line for marker in ['-|-', '|-']):  # Skip separator lines
                                cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty edges
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
                            # Calculate column widths - divide available space equally
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
                    
                    # Handle regular text
                    text = re.sub(r'\*(.*?)\*', r'<b>\1</b>', line)
                    story.append(Paragraph(text, body_style))
                    i += 1
                
                # Add any remaining list items
                if current_list_items:
                    story.append(ListFlowable(current_list_items, bulletType='bullet', leftIndent=20))
                    current_list_items = []
                
                # Add section spacing
                story.append(Spacer(1, 16))
            
            # Build the PDF
            try:
                doc.build(story)
                print(f"PDF saved as {filename}")
            except Exception as e:
                print(f"Error creating PDF: {str(e)}")

            # Sample text for testing

        create_pdf("test_report.pdf", sample_text1)
        # Punnett square function to calculate offspring genotypes
        def punnett_square(father_genotype, mother_genotype):
            offspring = []
            father_alleles = list(father_genotype)
            mother_alleles = list(mother_genotype)

            for father_allele in father_alleles:
                for mother_allele in mother_alleles:
                    offspring.append("".join(sorted(father_allele + mother_allele)))

            genotype_count = {"CC": 0, "CT": 0, "TT": 0}
            for genotype in offspring:
                genotype_count[genotype] += 1

            total_offspring = len(offspring)
            genotype_ratio = {genotype: count / total_offspring for genotype, count in genotype_count.items()}

            return genotype_count, genotype_ratio

        # Function to determine the color of the genotype
        def get_color(genotype):
            if genotype == "CC":
                return "green"   # Normal
            elif genotype == "CT":
                return "orange"  # Carrier
            elif genotype == "TT":
                return "red"  # Affected

        # Main function to integrate both parts: genetic disorder prediction and Punnett square

        try:
            # Get user input
            input_data = get_user_input()


            # Punnett square calculations
            father_genotype = input_data['parent1_genotype']
            mother_genotype = input_data['parent2_genotype']

            genotype_count, genotype_ratio = punnett_square(father_genotype, mother_genotype)
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

            # Create a directed graph for the family tree
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
            node_labels = {n: n for n in graph.nodes}

            nx.draw(graph, pos, with_labels=True, labels=node_labels, node_size=3000,
                    node_color=node_colors, edge_color="black", font_size=10, font_weight="bold")

            plt.title("Punnett Square Family Tree")
            plt.legend(handles=[
                mpatches.Patch(color='green', label='Normal (CC)')
                , mpatches.Patch(color='orange', label='Carrier (CT)')
                , mpatches.Patch(color='red', label='Affected (TT)')
            ], loc='upper right')

            plt.savefig("punnett_tree.png", dpi=300)
         

        except Exception as e:
            print(f"\nError: {str(e)}")
        punnett_square(df['parent1_genotype'], df['parent2_genotype'])



    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please make sure the model files exist and inputs are correct.")

if __name__ == "__main__":
    main()

#ENDING OF USER SCRIPT