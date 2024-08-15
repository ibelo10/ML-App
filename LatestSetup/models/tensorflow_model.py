import os
import time
import joblib
import logging
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from data.add_features import add_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Dense(512, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_data(data):
    data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    majority_class = data[data['target'] == 0]
    minority_class = data[data['target'] == 1]
    minority_class_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
    data_balanced = pd.concat([majority_class, minority_class_upsampled])
    
    # Exclude 'Date' and 'Adj Close' from features
    features = data_balanced.drop(columns=['Date', 'Adj Close', 'target'])
    target = data_balanced['target']
    
    # Log columns before scaling
    print(f"Columns before scaling: {features.columns.tolist()}")
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"Features shape after scaling: {features_scaled.shape}")
    return features_scaled, target

def train_model(data, logger):
    data = add_features(data)
    features, target = preprocess_data(data)
    print(f"Features after preprocessing: {features.shape[1]}")

    max_attempts = 3
    attempt = 0
    desired_accuracy = 0.7
    best_accuracy = 0
    best_model_path = 'tensormodel/best_model.keras'
    if not os.path.exists('tensormodel'):
        os.makedirs('tensormodel')
    while best_accuracy < desired_accuracy and attempt < max_attempts:
        logger.info(f"Training attempt {attempt + 1} of {max_attempts}.")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_index, test_index) in enumerate(skf.split(features, target)):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            model = create_model(input_shape=(X_train.shape[1],))
            model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, save_weights_only=False)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=128,
                validation_data=(X_test, y_test),
                callbacks=[model_checkpoint, early_stopping, reduce_lr]
            )
            test_loss, test_acc = model.evaluate(X_test, y_test)
            logger.info(f'Fold {fold + 1} Test accuracy: {test_acc}')
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                joblib.dump(model, best_model_path.replace('.keras', '.pkl'))
        attempt += 1
        logger.info(f'Best accuracy after attempt {attempt}: {best_accuracy}')
    return best_model_path

if __name__ == "__main__":
    from data.data_fetching import DataFetcher

    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_data()
    best_model_path = train_model(data, logger)
    print(f"Training completed. Best model saved at: {best_model_path}")
