import numpy as np

# Assuming all_train_frames_final_array and all_train_labels_final_array are already defined

# Reshape all_train_frames_final_array to flatten the frames
all_train_frames_final_flat = np.concatenate(all_train_frames_final_array)

# Calculate the total number of frames
total_frames = len(all_train_frames_final_flat)

# Repeat each label in all_train_labels_final_array according to the number of frames
all_train_labels_final_matched = np.repeat(all_train_labels_final_array, total_frames // len(all_train_labels_final_array))

# Ensure the lengths match
assert len(all_train_frames_final_flat) == len(all_train_labels_final_matched), "Number of frames and labels don't match"

# Reshape all_train_frames_final_flat into a 4D array with shape (number_of_samples, height, width, channels)
all_train_frames_final_matched = all_train_frames_final_flat.reshape(-1, 64, 64, 1)

# Check the shapes
print("Shape of all_train_frames_final_matched:", all_train_frames_final_matched.shape)
print("Shape of all_train_labels_final_matched:", all_train_labels_final_matched.shape)
