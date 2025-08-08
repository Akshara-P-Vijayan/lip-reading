import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Root directory of the dataset
root_dir = '/kaggle/input/lip-reading/cropped_mouth_mp4_phrase'

# List to store tuples of (label, video_path)
dataset = []
labels = set()

# Iterate over each label folder
for label_folder in os.listdir(root_dir):
    label_folder_path = os.path.join(root_dir, label_folder)

    # Ensure it's a directory
    if os.path.isdir(label_folder_path):
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
                        labels.add(label_folder)  # Add label to set

# Convert labels to numpy array
labels_array = np.array(list(labels))

# Use LabelEncoder to convert string labels to corresponding number format starting from 1
label_encoder = LabelEncoder()
label_encoder.fit(labels_array)
encoded_labels = label_encoder.transform(labels_array) + 1  # Adding 1 to start labels from 1

# Print the collected dataset with labels
print("Collected Dataset:")
for label, video_path in dataset:
    print(f"Label: {label_encoder.transform([label])[0] + 1}, Video Path: {video_path}")

# Print the numpy array containing the unique labels starting from 1
print("Unique labels:")
print(encoded_labels)

# Print the number of samples per class
for label, encoded_label in zip(labels_array, encoded_labels):
    num_samples = sum(1 for l, _ in dataset if l == label)
    print(f"Label: {encoded_label}, Class: {label}, Num Samples: {num_samples}")
