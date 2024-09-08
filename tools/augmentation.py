import pandas as pd
import cv2
import numpy as np
import random
import os
from imgaug import augmenters as iaa

# Function to load the CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)

# Function to apply augmentations to a video
def augment_video(video_path, aug_type="appearance"):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    
    # Choose augmentation type
    if aug_type == "appearance":
        aug = random.choice([
            iaa.Multiply((0.5, 1.5)),  # Change brightness
            iaa.LinearContrast((0.75, 1.5)),  # Change contrast
            iaa.AddToHueAndSaturation((-20, 20))  # Change hue/saturation
        ])
    elif aug_type == "geometry":
        aug = random.choice([
            iaa.Fliplr(1.0),  # Flip horizontally
            iaa.Flipud(1.0),  # Flip vertically
            iaa.Affine(rotate=(-25, 25))  # Rotate
        ])

    augmented_frames = aug.augment_images(frames)
    
    return augmented_frames

# Function to save augmented video
def save_augmented_video(frames, output_path, fps=30):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Saving in .mp4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    
    out.release()

# Function to randomly select videos and apply augmentations
def augment_class_videos(csv_file, target_class, num_videos, output_dir, new_csv_file):
    # Load the CSV
    df = load_csv(csv_file)
    
    # Filter rows based on the target class
    class_videos = df[df.iloc[:]['Emotion'] == target_class]
    
    # Randomly select videos
    selected_videos = class_videos.sample(n=num_videos, random_state=42)
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List to store new rows (video paths and class labels)
    new_rows = []
    
    # Iterate over the selected videos
    for idx, row in selected_videos.iterrows():
        video_path = os.path.join('D:/uzh_project/emotion_recognition/dataset',row[0][2:])  # Assuming the second column is the video path
        print(video_path)
        video_name = os.path.basename(video_path)
        
        # Randomly choose augmentation type
        aug_type = random.choice(["appearance", "geometry"])
        
        # Augment the video
        augmented_frames = augment_video(video_path, aug_type)
        
        # Save the augmented video
        aug_output_path = os.path.join(output_dir, f"aug_{aug_type}_{video_name}")
        save_augmented_video(augmented_frames, aug_output_path)
        
        print(f"Augmented video saved: {aug_output_path}")

        new_rows.append([aug_output_path, None, None, None, target_class, row['Sentiment'], None, None, None, None, None, None])

    # Create a DataFrame for the new rows with all required columns
    new_df = pd.DataFrame(new_rows, columns=['file_path', 'Sr No.', 'Utterance', 'Speaker', 'Emotion', 'Sentiment', 'Dialogue_ID', 'Utterance_ID', 'Season', 'Episode', 'StartTime', 'EndTime'])
    
    # Append the new rows to the original dataframe
    updated_df = pd.concat([df, new_df], ignore_index=True)
    
    # Save the updated dataframe to a new CSV file
    updated_df.to_csv(new_csv_file, index=False)
    print(f"CSV file updated and saved as {new_csv_file}")

# Example usage
augment_class_videos('./data/train.csv', 'anger', 10, './data/augmented_videos', './data/updated_train.csv')
