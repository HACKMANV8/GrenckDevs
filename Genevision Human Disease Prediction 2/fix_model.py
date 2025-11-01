#!/usr/bin/env python3
"""
Model compatibility fix script for TensorFlow version issues.
This script recreates the model with compatible architecture.
"""

import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def create_compatible_model(input_shape: int) -> Sequential:
    """Create a TensorFlow 2.x compatible model."""
    model = Sequential([
        # Input layer - using input_shape instead of batch_shape
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

    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['mae', 'mse', 'accuracy']
    )

    return model

def fix_model():
    """Fix the model compatibility issue."""
    
    # Check if files exist
    model_path = 'models/genetic_model.h5'
    
    # Based on the original code, we know the input shape
    # categorical_columns: 8 features
    # numerical_columns: 5 features  
    # Total: 13 features
    input_shape = 13
    print(f"Using known input shape: {input_shape}")
    
    # Create new compatible model
    print("Creating compatible model...")
    new_model = create_compatible_model(input_shape)
    
    # Save the new model
    backup_path = 'models/genetic_model_backup.h5'
    new_model_path = 'models/genetic_model_fixed.h5'
    
    # Backup original if it exists
    if os.path.exists(model_path):
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f"Original model backed up to: {backup_path}")
    
    # Save new model
    new_model.save(new_model_path)
    print(f"New compatible model saved to: {new_model_path}")
    
    # Also save as the original name
    new_model.save(model_path)
    print(f"Compatible model saved as: {model_path}")
    
    print("\nModel fix completed successfully!")
    print("Note: This is a new model with random weights. You may want to retrain it for better accuracy.")
    
    return True

if __name__ == "__main__":
    fix_model()