import os
import json
import pandas as pd

csv_path = 'D:/uzh_project/emotion_recognition/tools/test_sent_emo.csv'
emotion_lookup_save_path = 'D:/uzh_project/emotion_recognition/tools/data/emotion_lookup_table.json'  # Ensure this is a file path
sentiment_lookup_save_path = 'D:/uzh_project/emotion_recognition/tools/data/sentiment_lookup_table.json'

df = pd.read_csv(csv_path)

# Create a lookup table (vocabulary)
def create_lookup_table(classes, file_path):
    unique_classes = sorted(set(classes))
    lookup_table = {cls: idx for idx, cls in enumerate(unique_classes)}
    with open(file_path, 'w') as file:
        json.dump(lookup_table, file)
    return lookup_table

emotion_classes = df.iloc[:, 3]
sentiment_classes = df.iloc[:, 4]

# Generate the lookup table
emotion_lookup_table = create_lookup_table(emotion_classes, emotion_lookup_save_path)
sentiment_lookup_table = create_lookup_table(sentiment_classes, sentiment_lookup_save_path)
print("Emotion Lookup Table:", emotion_lookup_table)
print("Emotion Lookup Table:", sentiment_lookup_table)
