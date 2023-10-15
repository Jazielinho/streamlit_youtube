

import pandas as pd
import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

WIDTH = 1000
HEIGHT = 600


directory = '/media/jahaziel/Datos/proyectos/Smarketing/streamlit_youtube/'

# youtube_videos = os.listdir(directory)
youtube_videos = [
    '2023_FIDE_World_Championship_Match_FINAL_RECAP',
]

for _youtube_vide in youtube_videos:
    image_best_images_per_topic_df = pd.read_csv(os.path.join(directory, _youtube_vide, 'Analysis', 'image_best_images_per_topic.csv'))
    face_best_images_per_topic_df = pd.read_csv(os.path.join(directory, _youtube_vide, 'Analysis', 'face_best_images_per_topic.csv'))

    valid_images_filenames = image_best_images_per_topic_df['image_filename'].values
    valid_images_filenames = [x.split('/')[-1].split('.')[0] for x in valid_images_filenames]

    valid_faces_filenames = face_best_images_per_topic_df['image_filename'].values
    valid_faces_filenames = [x.split('/')[-1].split('.')[0] for x in valid_faces_filenames]

    all_images = os.listdir(os.path.join(directory, _youtube_vide, 'frames'))
    all_faces = os.listdir(os.path.join(directory, _youtube_vide, 'Analysis', 'face'))

    for image in all_images:
        if image.split('.')[0] not in valid_images_filenames:
            os.remove(os.path.join(directory, _youtube_vide, 'frames', image))

    for face in all_faces:
        if face.split('.')[0] not in valid_faces_filenames:
            os.remove(os.path.join(directory, _youtube_vide, 'Analysis', 'face', face))

