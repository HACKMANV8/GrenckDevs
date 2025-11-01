import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pickle
import os

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
            metrics=['mae', 'mse']
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

if __name__ == "__main__":
    # Load the training data
    data = pd.read_csv('genetic_disorders_dataset.csv')

    # Initialize and train the model
    trainer = GeneticModelTrainer()
    history = trainer.train_model(data)

    # Save the model and preprocessing objects
    trainer.save_model('models/genetic_model.h5', 'models/preprocessors.pkl')

    print("\nModel training complete and saved successfully!")