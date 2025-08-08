import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Root directory of the dataset
root_dir = '/kaggle/input/lip-reading/cropped_mouth_mp4_phrase'

# List to store tuples of (label, video_path)
dataset = []

# Iterate over each label folder
for label_folder in os.listdir(root_dir):
    label_folder_path = os.path.join(root_dir, label_folder)

    # Ensure it's a directory
    if os.path.isdir(label_folder_path) and not label_folder.isdigit():
        # Iterate over each subfolder
        for subfolder in os.listdir(label_folder_path):
            subfolder_path = os.path.join(label_folder_path, subfolder)

            # Ensure it's a directory
            if os.path.isdir(subfolder_path):
                # Iterate over each video file
                for video_file in os.listdir(subfolder_path):
                    if video_file.endswith('.mp4'):
                        video_path = os.path.join(subfolder_path, video_file)
                        # Append tuple (label, video_path) to the dataset list
                        dataset.append((label_folder, video_path))

# Numeric labels array
labels_array = np.array([3, 8, 2, 5, 9, 7, 1, 10, 4, 6])

# Adjust the labels_array if it contains 0
if np.min(labels_array) == 0:
    labels_array += 1

# Split dataset into train, test, and validation sets
train_data, test_val_data = train_test_split(dataset, test_size=0.2, random_state=42)
test_data, val_data = train_test_split(test_val_data, test_size=0.5, random_state=42)

# Function to convert labels to numeric numpy array
def convert_labels_to_array(data, unique_labels):
    labels_array = np.array([np.where(unique_labels == label)[0][0] + 1 for label, _ in data])
    return labels_array

# Convert labels to numeric numpy array
unique_labels = np.array(['Excuse me', 'Goodbye', 'Have a good time', 'Hello', 'How are you',
                          'I am sorry', 'Nice to meet you', 'See you', 'Thank you', 'You are welcome'])
train_labels = convert_labels_to_array(train_data, unique_labels)
test_labels = convert_labels_to_array(test_data, unique_labels)
val_labels = convert_labels_to_array(val_data, unique_labels)

# Print the train and test labels arrays
# print("Train Labels Array:")
# print(train_labels)
# print("\nTest Labels Array:")
# print(test_labels)
# print("\nValidation Labels Array:")
# print(val_labels)

# Function to arrange data into folders based on labels and split sets
def arrange_data(data, root_dir, labels):
    for (label, video_path), label_index in zip(data, labels):
        set_dir = os.path.join(root_dir, str(label_index))
        if not os.path.exists(set_dir):
            os.makedirs(set_dir)
        subfolder_dir = os.path.join(set_dir, os.path.basename(os.path.dirname(video_path)))
        if not os.path.exists(subfolder_dir):
            os.makedirs(subfolder_dir)
        shutil.copy(video_path, subfolder_dir)

# Arrange train data into folders
arrange_data(train_data, '/kaggle/working/train', train_labels)

# Arrange test data into folders
arrange_data(test_data, '/kaggle/working/test', test_labels)

# Arrange validation data into folders
arrange_data(val_data, '/kaggle/working/validation', val_labels)

print("Data has been arranged into folders based on labels and replicated the dataset structure.")
