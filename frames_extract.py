import os
import cv2
import numpy as np

# Root directory of the training dataset
root_dir = '/kaggle/working/train'

# Destination directory for the converted frames
dest_dir = '/kaggle/working/train_frames'

# Function to extract frames from video and save them as grayscale images
def extract_frames(video_path, dest_path, label, num_frames=29):
    frames = []

    # Create directory for frames
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames / num_frames
    frame_count = 0

    # Read until video is completed
    while cap.isOpened() and frame_count < num_frames:
        frame_index = int(frame_count * frame_interval)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize frame to 64x64
        resized_frame = cv2.resize(gray_frame, (64, 64))

        # Save frame as JPEG image
        frame_path = os.path.join(dest_path, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, resized_frame)

        # Append frame to frames list
        frames.append(resized_frame)
        frame_count += 1

    # Release video capture object
    cap.release()

    return frames

# Initialize lists to store all training frames and their corresponding labels
all_train_frames_final = []
all_train_labels_final = []

# Iterate over each label folder in the training set
for label_folder in os.listdir(root_dir):
    label_folder_path = os.path.join(root_dir, label_folder)

    # Ensure it's a directory
    if os.path.isdir(label_folder_path):
        # Iterate over each subfolder (video)
        for video_folder in os.listdir(label_folder_path):
            video_folder_path = os.path.join(label_folder_path, video_folder)

            # Ensure it's a directory
            if os.path.isdir(video_folder_path):
                # Create destination folder for the video frames
                dest_video_folder_path = os.path.join(dest_dir, label_folder, video_folder)
                if not os.path.exists(dest_video_folder_path):
                    os.makedirs(dest_video_folder_path)

                # Extract frames from video
                video_files = [file for file in os.listdir(video_folder_path) if file.endswith('.mp4')]
                for video_file in video_files:
                    video_path = os.path.join(video_folder_path, video_file)

                    # Extract frames from video
                    frames = extract_frames(video_path, os.path.join(dest_video_folder_path, video_file[:-4]), label_folder)

                    # Extend the all_train_frames list
                    all_train_frames_final.append(frames)

                    # Append label of the video to the labels list
                    all_train_labels_final.append(label_folder)

                    # Print the number of frames extracted
                    print(f"Label: {label_folder}, Video: {video_folder}, Frames extracted: {len(frames)}")

# Convert the list of frames to a numpy array
all_train_frames_final_array = np.array(all_train_frames_final)

# Convert the list of labels to a numpy array
all_train_labels_final_array = np.array(all_train_labels_final)

# Ensure the number of training videos matches the number of training labels
assert len(all_train_frames_final_array) == len(all_train_labels_final_array), "Number of training videos and labels don't match"

print("All training videos have been processed and their labels have been assigned.")
print("Total training videos shape:", all_train_frames_final_array.shape)
print("Total training labels shape:", all_train_labels_final_array.shape)
