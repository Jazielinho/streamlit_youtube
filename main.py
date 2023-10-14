
import pandas as pd
import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.preprocessing import normalize
import numpy as np
from streamlit_plotly_events import plotly_events


directory = __file__.split('main.py')[0]
directory = '/media/jahaziel/Datos/proyectos/Smarketing/streamlit_youtube/'