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
from matplotlib.backends.backend_pdf import PdfPages
import glob
import random
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


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
    subject_data = balanced_subset[balanced_subset['SubjectID'] == subject].copy()
    
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

# Define group ranges
group_A_subjects = [f'sub-{i:03}' for i in range(1, 37)]  # Subjects 1-36
group_C_subjects = [f'sub-{i:03}' for i in range(37, 66)]  # Subjects 37-65
group_F_subjects = [f'sub-{i:03}' for i in range(66, 89)]  # Subjects 66-88

# Map subjects to groups
subject_to_group = {sub: 'A' for sub in group_A_subjects}
subject_to_group.update({sub: 'C' for sub in group_C_subjects})
subject_to_group.update({sub: 'F' for sub in group_F_subjects})

# Extract clean SubjectID from eeg_epochs
eeg_epochs['SubjectID_clean'] = eeg_epochs['SubjectID'].str.extract(r'(sub-\d{3})')

# Map the group to each SubjectID
eeg_epochs['Group'] = eeg_epochs['SubjectID_clean'].map(subject_to_group)

# Verify the mapping
print(eeg_epochs[['SubjectID', 'SubjectID_clean', 'Group']].head())

print(eeg_epochs.head())
print(eeg_epochs.tail())


# Ensure the data is sorted by Time within each SubjectID
eeg_epochs = eeg_epochs.sort_values(by=['SubjectID', 'Time'])

# Group data by SubjectID
grouped_data = eeg_epochs.groupby('SubjectID')

# Prepare for storing scaled data
scaled_data = []

# Apply scaling for each subject group
for name, group in grouped_data:
    # Initialize scaler
    scaler = StandardScaler()
    
    # Extract EEG channels for scaling
    eeg_features = group[['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                          'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']].values  
    
    # Scale the features
    scaled_features = scaler.fit_transform(eeg_features)
    
    # Replace original features with scaled ones
    group[['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
           'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']] = scaled_features
    
    # Append to scaled data
    scaled_data.append(group)

# Combine all scaled groups back into one DataFrame
eeg_epochs = pd.concat(scaled_data)

print(eeg_epochs.head())

# Ensure the data is sorted by Time within each SubjectID
eeg_epochs = eeg_epochs.sort_values(by=['SubjectID', 'Time'])

# Group data by SubjectID
grouped_data = eeg_epochs.groupby('SubjectID')

# Sliding window parameters
window_size = 100  # Number of timesteps in each window
stride = 50        # Overlap between consecutive windows

# Function to create sliding windows
def create_sliding_windows(sequence, window_size, stride):
    return [sequence[i:i + window_size] for i in range(0, len(sequence) - window_size + 1, stride)]

# Prepare sequences and targets
sequences = []
labels = []

for name, group in grouped_data:
    # Extract the 19-channel EEG data as features
    sequence = group[['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                      'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']].values  

    # Create sliding windows for the sequence
    windows = create_sliding_windows(sequence, window_size, stride)
    
    # Add each window and its corresponding label
    for window in windows:
        sequences.append(window)
        # Use Group (classification) or MMSE (regression) as the target
        label = group['Group'].iloc[0]  # Classification
        # label = group['MMSE'].iloc[0]  # Uncomment for regression
        labels.append(label)

# Pad sequences to ensure consistent shape (optional, but useful for variable window sizes)
X = pad_sequences(sequences, maxlen=window_size, padding='post', dtype='float32')

# Encode classification labels (for regression, skip this step)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)  # Use labels as-is for regression

# Check shapes of prepared data
print("Feature Shape (X):", X.shape)  # Should be (#samples, window_size, #channels)
print("Labels Shape (y):", y.shape)  # Should be (#samples,)

# Encode group labels to integers
label_encoder = LabelEncoder()
eeg_epochs['Group_encoded'] = label_encoder.fit_transform(eeg_epochs['Group'])

# Print mapping for reference
print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Get unique subject IDs
unique_subjects = eeg_epochs['SubjectID_clean'].unique()

# Get the labels for each subject
subject_labels = eeg_epochs.groupby('SubjectID_clean')['Group'].first()

# Stratified split
train_subjects, temp_subjects = train_test_split(
    unique_subjects, test_size=0.4, stratify=subject_labels[unique_subjects], random_state=42
)
val_subjects, test_subjects = train_test_split(
    temp_subjects, test_size=0.5, stratify=subject_labels[temp_subjects], random_state=42
)

train_data = eeg_epochs[eeg_epochs['SubjectID_clean'].isin(train_subjects)]
val_data = eeg_epochs[eeg_epochs['SubjectID_clean'].isin(val_subjects)]
test_data = eeg_epochs[eeg_epochs['SubjectID_clean'].isin(test_subjects)]

def process_split(data):
    grouped_data = data.groupby('SubjectID')
    sequences = []
    labels = []
    
    for name, group in grouped_data:
        # Extract the EEG data as features
        sequence = group[['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                          'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']].values
        
        # Create sliding windows for the sequence
        windows = create_sliding_windows(sequence, window_size, stride)
        
        for window in windows:
            sequences.append(window)
            labels.append(group['Group_encoded'].iloc[0])  # Use encoded label
    
    return np.array(sequences), np.array(labels)

# Generate sliding windows
X_train, y_train = process_split(train_data)
X_val, y_val = process_split(val_data)
X_test, y_test = process_split(test_data)

# Check shapes
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

# Check class distributions
print("Training labels distribution:", np.bincount(y_train))
print("Validation labels distribution:", np.bincount(y_val))
print("Test labels distribution:", np.bincount(y_test))

# Split into training and temp (test + validation) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Further split temp into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Print dataset shapes
print("Training Set:", X_train.shape, y_train.shape)
print("Validation Set:", X_val.shape, y_val.shape)
print("Test Set:", X_test.shape, y_test.shape)

# Define the sequential model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Define EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                               patience=5,          # Stop if no improvement for 5 epochs
                               restore_best_weights=True)  # Restore best weights

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,                    # You can adjust the number of epochs if needed
    batch_size=32,                # You can adjust batch size as well
    class_weight=class_weights_dict,  # Apply class weights during training
    callbacks=[early_stopping],   # Add the EarlyStopping callback
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

# Print test performance
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict on the test set
y_pred = model.predict(X_test)

# Convert predictions from one-hot encoding (if necessary)
y_pred_classes = np.argmax(y_pred, axis=1)  # For classification tasks

# Generate a classification report
print(classification_report(y_test, y_pred_classes))

# Optionally, plot the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Assuming y_test and y_pred are your true labels and predictions
report = classification_report(y_test, y_pred_classes)

# Save the classification report to a text file
with open('sequential_classification_report.txt', 'w') as file:
    file.write(report)

print("Classification report saved to 'sequential_classification_report.txt'")

# Assuming y_test and y_pred are your true labels and predictions
cm = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Save the confusion matrix to a PDF
plt.savefig('sequential_confusion_matrix.pdf', format='pdf')

print("Confusion matrix saved to 'sequential_confusion_matrix.pdf'")