import mne
from mne.preprocessing import ICA, create_eog_epochs
from mne.io import read_raw_eeglab
from mne import create_info
from mne.io import RawArray
from scipy.signal import resample
from scipy.signal import butter, sosfilt
from scipy.stats import iqr
from scipy.signal import butter, sosfiltfilt
from pathlib import Path
import matplotlib.pyplot as plt
import glob
import random
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the TSV file into a DataFrame
file_path = "/mmfs1/home/eor13/Final_Project/participants.tsv"
patient_df = pd.read_csv(file_path, sep='\t')

patient_df = patient_df.rename(columns={'participant_id': 'SubjectID'})

# View the first few rows to understand the structure
print(patient_df.head())

# Define the path to the directory containing the EEG data
main_dir = Path("/mmfs1/home/eor13/Final_Project/derivatives/") 

# Initialize an empty list to store each patient's data
data_list = []

# Search through subdirectories to load all .set files
file_paths = glob.glob(str(main_dir / 'sub-**' / 'eeg' / '*.set'), recursive = True)

for file_path in file_paths:
    
    # Load the EEG data file
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    
    # Extract data and channel names
    data, times = raw.get_data(return_times=True)
    channel_names = raw.ch_names
    
    # Create a DataFrame for this patient
    df = pd.DataFrame(data.T, columns=channel_names)
    df['Time'] = times  # Add a time column
    
    # Add patient ID from filename or folder structure
    subject_id = Path(file_path).stem
    df['SubjectID'] = subject_id
    
    # Append to the list
    data_list.append(df)

# Concatenate all dataframes
eeg_df = pd.concat(data_list, ignore_index=True)

# Display dataframe shape
print(f"DataFrame shape: {eeg_df.shape}")

# Display the final DataFrame
print(eeg_df.head())
print(eeg_df.tail())

# Extract just the subject ID (e.g., 'sub-001', 'sub-002')
eeg_df['SubjectID_clean'] = eeg_df['SubjectID'].str.split('_').str[0]

# Choose seed for repeatability
random.seed(42)

# Define the ranges of subjects for each group
group_A_subjects = [f'sub-{i:03}' for i in range(1, 37)]  # Subjects 1-36
group_C_subjects = [f'sub-{i:03}' for i in range(37, 66)]  # Subjects 37-65
group_F_subjects = [f'sub-{i:03}' for i in range(66, 89)]  # Subjects 66-88

# Randomly sample a specific number of subjects from each group (e.g., 10 subjects per group)
num_subjects_per_group = 13
selected_A_subjects = random.sample(group_A_subjects, num_subjects_per_group)
selected_C_subjects = random.sample(group_C_subjects, num_subjects_per_group)
selected_F_subjects = random.sample(group_F_subjects, num_subjects_per_group)

# Combine the selected subjects from all groups
selected_subjects = selected_A_subjects + selected_C_subjects + selected_F_subjects

# Filter the eeg_df_filtered dataframe to only include the selected subjects
balanced_subset = eeg_df[eeg_df['SubjectID_clean'].isin(selected_subjects)]

# Print the subset to confirm
#print(balanced_subset.head())
#print(balanced_subset.tail())

print("Selected Subjects: ", selected_subjects)

# Define target sampling rate
target_sampling_rate = 125  # Hz

# Extract unique subjects
unique_subjects = balanced_subset['SubjectID'].unique()

# Store resampled data
resampled_data_list = []

# Loop through each subject
for subject in unique_subjects:
    # Subset data for the subject
    subject_data = eeg_df[eeg_df['SubjectID'] == subject].copy()
    
    # Extract channel data and sampling frequency
    data = subject_data.iloc[:, :-2].to_numpy().T  # Exclude 'Time' and 'SubjectID'
    original_sampling_rate = 500  # Replace with your dataset's original rate
    
    # Resample data
    num_samples = int((len(subject_data) / original_sampling_rate) * target_sampling_rate)
    resampled_data = resample(data, num_samples, axis=1)
    
    # Create a new time array for the resampled data
    new_times = np.linspace(0, subject_data['Time'].iloc[-1], num_samples)
    
    # Convert resampled data back to DataFrame
    resampled_df = pd.DataFrame(resampled_data.T, columns=subject_data.columns[:-2])
    resampled_df['Time'] = new_times
    resampled_df['SubjectID'] = subject
    
    # Append to list
    resampled_data_list.append(resampled_df)

# Concatenate all resampled data
eeg_df_resampled = pd.concat(resampled_data_list, ignore_index=True)

# Display results
print(f"Resampled DataFrame shape: {eeg_df_resampled.shape}")
print(eeg_df_resampled.head())
print(eeg_df_resampled.tail())

# Define Butterworth filter function
def butter_bandpass(lowcut, highcut, fs, order=5):
    """Creates a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype='band', output='sos')

def apply_filter(data, lowcut=0.5, highcut=48, fs=250, order=5):
    """Applies the Butterworth bandpass filter."""
    sos = butter_bandpass(lowcut, highcut, fs, order)
    return sosfilt(sos, data, axis=0)

# Apply filter to each subject's data
filtered_data_list = []

for subject in eeg_df_resampled['SubjectID'].unique():
    # Select data for the subject
    subject_data = eeg_df_resampled[eeg_df_resampled['SubjectID'] == subject].copy()
    
    # Apply filter to EEG channels (all columns except 'Time' and 'SubjectID')
    eeg_channels = subject_data.columns[:-2]  # Assuming last two columns are 'Time' and 'SubjectID'
    subject_data[eeg_channels] = apply_filter(subject_data[eeg_channels].to_numpy(), fs=target_sampling_rate)
    
    # Append filtered data
    filtered_data_list.append(subject_data)

# Concatenate all filtered data
eeg_df_filtered = pd.concat(filtered_data_list, ignore_index=True)

# Display results
print(f"Filtered DataFrame shape: {eeg_df_filtered.shape}")
print(eeg_df_filtered.head())
print(eeg_df_filtered.tail())

def extract_epochs(data, epoch_length=2, overlap=1, sampling_rate=125):
    """
    Extract overlapping epochs from EEG data.

    Parameters:
    - data: DataFrame containing EEG data.
    - epoch_length: Length of each epoch in seconds (default: 5).
    - overlap: Overlap between epochs in seconds (default: 2.5).
    - sampling_rate: Sampling rate of the data in Hz (default: 250).

    Returns:
    - A list of DataFrames, each containing epochs for a single subject.
    """
    epochs_list = []
    samples_per_epoch = int(epoch_length * sampling_rate)
    overlap_samples = int(overlap * sampling_rate)
    
    for subject in data['SubjectID'].unique():
        subject_data = data[data['SubjectID'] == subject].reset_index(drop=True)
        channels = subject_data.columns[:-2]  # Exclude 'Time' and 'SubjectID'
        time = subject_data['Time']
        
        # Sliding window to extract epochs
        for start_idx in range(0, len(subject_data) - samples_per_epoch + 1, samples_per_epoch - overlap_samples):
            end_idx = start_idx + samples_per_epoch
            epoch = subject_data.iloc[start_idx:end_idx].copy()
            epoch['EpochID'] = f"{subject}_epoch_{start_idx // (samples_per_epoch - overlap_samples)}"
            epochs_list.append(epoch)
    
    return pd.concat(epochs_list, ignore_index=True)

# Apply epoch extraction
eeg_epochs = extract_epochs(eeg_df_filtered)

# Display results
print(f"Total number of epochs: {len(eeg_epochs['EpochID'].unique())}")
print(eeg_epochs.head())
print(eeg_epochs.tail())

# Precompute band-filtered signals for all bands
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 25),
    "gamma": (25, 48)
}

# Dictionary to store filtered signals
filtered_signals = {}

for band_name, (low, high) in bands.items():
    print(f"Filtering for band: {band_name}")
    filtered_signals[band_name] = apply_filter(
        eeg_df_filtered.iloc[:, :-2].to_numpy(), low, high, fs=125
    )

def compute_features(epoch_data, bands):
    """
    Compute features from an epoch.

    Parameters:
    - epoch_data: DataFrame containing an epoch.
    - bands: Dictionary with frequency bands.

    Returns:
    - Dictionary of computed features.
    """
    features = {}
    for band_name, (low, high) in bands.items():
        # Band-filtered signal
        filtered_signal = apply_filter(epoch_data.iloc[:, :-3].to_numpy(), low, high, fs=125)
        features[f"{band_name}_mean"] = np.mean(filtered_signal, axis=0)
        features[f"{band_name}_std"] = np.std(filtered_signal, axis=0)
    
    # Statistical features
    features["variance"] = np.var(epoch_data.iloc[:, :-3].to_numpy(), axis=0).mean()
    features["iqr"] = iqr(epoch_data.iloc[:, :-3].to_numpy(), axis=0).mean()

    return features

# Iterate over epochs and compute features
features_list = []
for epoch_id in eeg_epochs['EpochID'].unique():
    epoch_data = eeg_epochs[eeg_epochs['EpochID'] == epoch_id]
    subject_id = epoch_data['SubjectID'].iloc[0]
    features = compute_features(epoch_data, bands)
    features["SubjectID"] = subject_id
    features["EpochID"] = epoch_id
    features_list.append(features)

# Combine features into a DataFrame
features_df = pd.DataFrame(features_list)
print(features_df.head())

print(patient_df.head())

# Ensure subject IDs match format in patient_df
features_df['SubjectID'] = features_df['SubjectID'].str.extract(r"(sub-\d+)")

# Merge with metadata
combined_df = pd.merge(features_df, patient_df[['SubjectID', 'Group']], on='SubjectID', how='inner')

print(combined_df.head())
print(combined_df.tail())

# Check the column types to ensure they are all numeric
print(combined_df.dtypes)

def compute_mean(value):
    # Check if the value is a numpy array and compute the mean
    return np.mean(value) if isinstance(value, np.ndarray) else np.nan

# Columns to process
columns_to_process = [
    'delta_mean', 'delta_std', 'theta_mean', 'theta_std',
    'alpha_mean', 'alpha_std', 'beta_mean', 'beta_std',
    'gamma_mean', 'gamma_std'
]

# Apply the computation
for col in columns_to_process:
    combined_df[col] = combined_df[col].apply(compute_mean)

# Check for NaN values after processing
print(combined_df.isna().sum())

# Drop non-numeric columns (SubjectID and EpochID) before applying transformations
combined_df_numeric = combined_df.drop(columns=['SubjectID', 'EpochID','Group'])

# Convert list-like columns to numeric (mean of the list)
for col in ['delta_mean', 'delta_std', 'theta_mean', 'theta_std', 'alpha_mean', 'alpha_std', 
            'beta_mean', 'beta_std', 'gamma_mean', 'gamma_std']:
    combined_df_numeric[col] = combined_df_numeric[col].apply(lambda x: np.mean(eval(x)) if isinstance(x, str) else np.mean(x))

# Now, check for NaN values in the numeric dataframe
print(combined_df_numeric.isna().sum())

# Fill NaN values with the mean of each column
combined_df_numeric.fillna(combined_df_numeric.mean(), inplace=True)

# Check the column types to ensure they are all numeric
print(combined_df_numeric.dtypes)

# Reattach non-numeric columns to the numeric dataframe
combined_df_final = pd.concat([combined_df[['SubjectID', 'EpochID','Group']], combined_df_numeric], axis=1)

# Check the result
print(combined_df_final.head())

# Separate features and labels
X = combined_df.drop(columns=['SubjectID', 'EpochID', 'Group'])
y = combined_df['Group']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # This converts 'A', 'C', 'F' into 0, 1, 2

# Split data into train, validation, and test sets (60%, 20%, 20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)


X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_train = X_train.astype('float64')
print(X_train.dtypes)

X_test = X_test.apply(pd.to_numeric, errors='coerce')
X_test = X_test.astype('float64')
print(X_test.dtypes)

# Check shapes
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Step 3: Evaluate on the validation and test set
y_val_pred = rf_model.predict(X_val)
y_test_pred = rf_model.predict(X_test)

# Print classification report for both validation and test sets
print("Random Forest - Validation Set Classification Report:\n")
print(classification_report(y_val, y_val_pred))
print("\nRandom Forest - Test Set Classification Report:\n")
print(classification_report(y_test, y_test_pred))

logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train, y_train)

y_val_pred = logreg_model.predict(X_val)
y_test_pred = logreg_model.predict(X_test)

# Print classification report for both validation and test sets
print("Logistic Regression - Validation Set Classification Report:\n")
print(classification_report(y_val, y_val_pred))
print("\nLogistic Regression - Test Set Classification Report:\n")
print(classification_report(y_test, y_test_pred))

svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

y_val_pred = svm_model.predict(X_val)
y_test_pred = svm_model.predict(X_test)

# Print classification report for both validation and test sets
print("SVM - Validation Set Classification Report:\n")
print(classification_report(y_val, y_val_pred))
print("\nSVM - Test Set Classification Report:\n")
print(classification_report(y_test, y_test_pred))

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

y_val_pred = knn_model.predict(X_val)
y_test_pred = knn_model.predict(X_test)

# Print classification report for both validation and test sets
print("KNN - Validation Set Classification Report:\n")
print(classification_report(y_val, y_val_pred))
print("\nKNN - Test Set Classification Report:\n")
print(classification_report(y_test, y_test_pred))

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

y_val_pred = gb_model.predict(X_val)
y_test_pred = gb_model.predict(X_test)

# Print classification report for both validation and test sets
print("Gradient Boosting - Validation Set Classification Report:\n")
print(classification_report(y_val, y_val_pred))
print("\nGradient Boosting - Test Set Classification Report:\n")
print(classification_report(y_test, y_test_pred))

mlp_model = MLPClassifier(max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)

y_val_pred = mlp_model.predict(X_val)
y_test_pred = mlp_model.predict(X_test)

# Print classification report for both validation and test sets
print("MLP - Validation Set Classification Report:\n")
print(classification_report(y_val, y_val_pred))
print("\nMLP - Test Set Classification Report:\n")
print(classification_report(y_test, y_test_pred))

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

y_val_pred = xgb_model.predict(X_val)
y_test_pred = xgb_model.predict(X_test)

# Print classification report for both validation and test sets
print("XGBoost - Validation Set Classification Report:\n")
print(classification_report(y_val, y_val_pred))
print("\nXGBoost - Test Set Classification Report:\n")
print(classification_report(y_test, y_test_pred))

# Models and their hyperparameter grids
models_and_params = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
        },
    },
    'SVM': {
        'model': SVC(random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
        },
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
        },
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
        },
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        },
    },
    'MLP': {
        'model': MLPClassifier(random_state=42),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'learning_rate_init': [0.001, 0.01],
        },
    },
}

# Loop through each model
best_models = {}

for name, model_info in models_and_params.items():
    print(f"Starting hyperparameter tuning for {name}...")
    
    # Define GridSearchCV
    grid_search = GridSearchCV(
        estimator=model_info['model'],
        param_grid=model_info['params'],
        scoring='accuracy',
        cv=5,  # 5-fold cross-validation
        verbose=2,
        n_jobs=-1  # Use all processors
    )
    
    # Perform the grid search
    grid_search.fit(X_train, y_train)
    
    # Store the best model and parameters
    best_models[name] = {
        'best_estimator': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
    }

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validation score for {name}: {grid_search.best_score_}")

# Evaluate each model
for name, model_info in best_models.items():
    print(f"Evaluating {name} on test data...")
    
    # Get the best model
    best_model = model_info['best_estimator']
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Print classification report
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))

# Initialize a DataFrame to store metrics
metrics_summary = []

# Extract performance metrics from classification reports
for name, model_info in best_models.items():
    # Get the best model
    best_model = model_info['best_estimator']
    
    # Predict on test data
    y_pred = best_model.predict(X_test)
    
    # Generate classification report as a dictionary
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Extract metrics for the 'weighted avg' row (or another specific class if desired)
    weighted_avg = report['weighted avg']
    metrics_summary.append({
        'Model': name,
        'Precision': weighted_avg['precision'],
        'Recall': weighted_avg['recall'],
        'F1-Score': weighted_avg['f1-score'],
        'Accuracy': report['accuracy'],  # Overall accuracy
    })

# Create a DataFrame
metrics_df = pd.DataFrame(metrics_summary)
print(metrics_df)

# Save classification reports to a .txt file
with open('classification_reports.txt', 'w') as f:
    for name, model_info in best_models.items():
        # Get the best model
        best_model = model_info['best_estimator']
        
        # Predict on test data
        y_pred = best_model.predict(X_test)
        
        # Generate classification report
        report = classification_report(y_test, y_pred)
        
        # Write to file
        f.write(f"Classification Report for {name}:\n")
        f.write(report)
        f.write("\n" + "="*80 + "\n")

# Plot the metrics
metrics_df.set_index('Model', inplace=True)
metrics_df.plot(kind='bar', figsize=(10, 6), rot=45, colormap='viridis')

# Add title and labels
plt.title('Model Performance Comparison', fontsize=16)
plt.ylabel('Score', fontsize=14)
plt.xlabel('Model', fontsize=14)
plt.ylim(0, 1)  # Since metrics are between 0 and 1
plt.legend(loc='lower right')
plt.tight_layout()

# Save the plot as a PDF
plt.savefig('model_comparison.pdf')
plt.show()
