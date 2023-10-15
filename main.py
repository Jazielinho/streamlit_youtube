
import pandas as pd
import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

WIDTH = 1000
HEIGHT = 600


directory = __file__.split('main.py')[0]
# directory = '/media/jahaziel/Datos/proyectos/Smarketing/streamlit_youtube/'

# youtube_videos = os.listdir(directory)
youtube_videos = [
    # '2023_FIDE_World_Championship_Match_FINAL_RECAP',
    "Ding_Doesn't_Know_Chinese?",
    'Final_4_minutes_of_the_Chess_World_Championship_Match_2023!',
    'The_Feeling_Of_Winning_A_World_Chess_Championship',
    'The_moment_Ding_Liren_became_World_Chess_Champion',
    'The_Shortest_Game_of_2023_FIDE_World_Championship_Ends_In_100_Minutes',
    'Why_Is_Everyone_Supporting_Ding_Liren?'
]


def get_video_statistics(video_dir):
    summary_df = pd.read_csv(os.path.join(directory, video_dir, 'Analysis', 'summary.csv'))

    image_report_df = pd.read_csv(os.path.join(directory, video_dir, 'Analysis', 'image_report.csv'))
    image_topic_info_df = pd.read_csv(os.path.join(directory, video_dir, 'Analysis', 'image_topic_info.csv'))
    image_probabilities_df = pd.read_csv(os.path.join(directory, video_dir, 'Analysis', 'image_probabilities.csv'))
    image_best_images_per_topic_df = pd.read_csv(os.path.join(directory, video_dir, 'Analysis', 'image_best_images_per_topic.csv'))
    # image_topics_over_seconds_df = pd.read_csv(os.path.join(directory, video_dir, 'Analysis', 'image_topics_over_seconds.csv'))

    face_report_df = pd.read_csv(os.path.join(directory, video_dir, 'Analysis', 'face_report.csv'))
    face_probabilities_df = pd.read_csv(os.path.join(directory, video_dir, 'Analysis', 'face_probabilities.csv'))
    face_best_images_per_topic_df = pd.read_csv(os.path.join(directory, video_dir, 'Analysis', 'face_best_images_per_topic.csv'))
    # face_topics_over_seconds_df = pd.read_csv(os.path.join(directory, video_dir, 'Analysis', 'face_topics_over_seconds.csv'))

    image_report_df['name'] = image_report_df['name'].apply(lambda x: os.path.join(directory, video_dir, 'frames', x.split('/')[-1]))
    image_probabilities_df['Unnamed: 0'] = image_probabilities_df['Unnamed: 0'].apply(lambda x: os.path.join(directory, video_dir, 'frames', x.split('/')[-1]))
    image_best_images_per_topic_df['image_filename'] = image_best_images_per_topic_df['image_filename'].apply(lambda x: os.path.join(directory, video_dir, 'frames', x.split('/')[-1]))

    face_report_df['name'] = face_report_df['name'].apply(lambda x: os.path.join(directory, video_dir, 'frames', x.split('/')[-1] + '.png'))
    face_probabilities_df['Unnamed: 0'] = face_probabilities_df['Unnamed: 0'].apply(lambda x: os.path.join(directory, video_dir, 'frames', x.split('/')[-1] + '.png'))
    face_best_images_per_topic_df['image_filename'] = face_best_images_per_topic_df['image_filename'].apply(lambda x: os.path.join(directory, video_dir, 'Analysis', 'face', x.split('/')[-1] + '.png'))

    face_report_df['negative'] = face_report_df[['disgust', 'anger', 'sadness', 'fear']].sum(axis=1)
    face_report_df['positive'] = face_report_df[['happy', 'surprise']].sum(axis=1)
    face_report_df['neutral'] = face_report_df[['neutral', 'contempt']].sum(axis=1)

    image_report_df['second_grouped'] = image_report_df['second'].apply(lambda x: int(x / 10) * 10)
    image_topics_over_seconds_df = image_report_df.groupby(['second_grouped', 'cluster']).size().reset_index()
    image_topics_over_seconds_df.columns = ['second', 'cluster', 'size']

    face_report_df['second_grouped'] = face_report_df['second'].apply(lambda x: int(x / 10) * 10)
    face_topics_over_seconds_df = face_report_df.groupby(['second_grouped', 'cluster']).size().reset_index()
    face_topics_over_seconds_df.columns = ['second', 'cluster', 'size']

    return {
        'summary_df': summary_df,
        'image_report_df': image_report_df,
        'image_topic_info_df': image_topic_info_df,
        'image_probabilities_df': image_probabilities_df,
        'image_best_images_per_topic_df': image_best_images_per_topic_df,
        'image_topics_over_seconds_df': image_topics_over_seconds_df,
        'face_report_df': face_report_df,
        'face_probabilities_df': face_probabilities_df,
        'face_best_images_per_topic_df': face_best_images_per_topic_df,
        'face_topics_over_seconds_df': face_topics_over_seconds_df,
    }


if 'statistics' not in st.session_state:
    st.session_state['statistics'] = {}

    for video_dir in youtube_videos:
        print(video_dir)
        st.session_state['statistics'][video_dir] = get_video_statistics(video_dir)


video_url = {
    '2023_FIDE_World_Championship_Match_FINAL_RECAP': 'https://www.youtube.com/watch?v=81ys7zw9VaI',
    "Ding_Doesn't_Know_Chinese?": 'https://www.youtube.com/watch?v=7zCF_xaUjEw',
    'Final_4_minutes_of_the_Chess_World_Championship_Match_2023!': 'https://www.youtube.com/watch?v=MBFzhd-ti0k',
    'The_Feeling_Of_Winning_A_World_Chess_Championship': 'https://www.youtube.com/watch?v=wChD-ZCZlcc',
    'The_moment_Ding_Liren_became_World_Chess_Champion': 'https://www.youtube.com/watch?v=03v2-VKVFnc',
    'The_Praggnanandhaa_interview_after_he_won_the_silver_medal_at_the_FIDE_World_Cup_2023': 'https://www.youtube.com/watch?v=ZC1wYLcnrG0',
    'The_Shortest_Game_of_2023_FIDE_World_Championship_Ends_In_100_Minutes': 'https://www.youtube.com/watch?v=p1coQJNezH0',
    'Why_Is_Everyone_Supporting_Ding_Liren?': 'https://www.youtube.com/watch?v=vjn23Yl9QCM',
}


st.set_page_config(page_title='Prueba Instagram', page_icon=':chess_pawn:', layout='wide')


st.title(f'''YouTube Video analysis''')

video_select = st.selectbox('Select video', youtube_videos)
if st.button('Analysis'):

    tab1, tab2, tab3, tab4 = st.tabs(['General', 'Images', 'Faces', 'Subtitles'])

    colors = {
        'negative': 'red',
        'neutral': 'gray',
        'positive': 'green'
    }

    labels_order = ['negative', 'neutral', 'positive']

    def reorder_data_for_plotting(data):
        return [data[label] for label in labels_order]


    with tab1:
        st.markdown(f'''## Video''')
        width = 30
        side = max((100 - width) / 2, 0.01)
        _, container, _ = st.columns([side, width, side])
        container.video(data=video_url[video_select])

        st.markdown(f'''## Sentiment from subtitles''')
        mean_sentiment_subtitles = st.session_state['statistics'][video_select]['summary_df'][['negative', 'neutral', 'positive']].mean()
        fig = go.Figure(data=[go.Pie(labels=['negative', 'neutral', 'positive'], values=reorder_data_for_plotting(mean_sentiment_subtitles), marker_colors=[colors[key] for key in labels_order])])
        st.plotly_chart(fig)

        st.markdown(f'''## Sentiment from images''')
        mean_sentiment_images = st.session_state['statistics'][video_select]['image_report_df'][['negative', 'neutral', 'positive']].mean()
        fig = go.Figure(data=[go.Pie(labels=['negative', 'neutral', 'positive'], values=reorder_data_for_plotting(mean_sentiment_subtitles), marker_colors=[colors[key] for key in labels_order])])
        st.plotly_chart(fig)

        count_sentiment_df = st.session_state['statistics'][video_select]['image_report_df'].groupby('second_grouped')[['negative', 'neutral', 'positive']].mean().reset_index()
        count_sentiment_df['size'] = st.session_state['statistics'][video_select]['image_report_df'].groupby('second_grouped').size().values
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['size'], name='Cantidad de imágenes', mode='lines+markers'), secondary_y=True)
        fig.add_trace(go.Bar(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['negative'], name='Sentimiento negativo'), secondary_y=False)
        fig.add_trace(go.Bar(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['neutral'], name='Sentimiento neutral'), secondary_y=False)
        fig.add_trace(go.Bar(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['positive'], name='Sentimiento positivo'), secondary_y=False)
        fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos en imágenes')
        fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)
        st.plotly_chart(fig, use_container_width=True)


        st.markdown(f'''## Sentiment from faces''')
        mean_sentiment_faces = st.session_state['statistics'][video_select]['face_report_df'][['negative', 'neutral', 'positive']].mean()
        fig = go.Figure(data=[go.Pie(labels=['negative', 'neutral', 'positive'], values=reorder_data_for_plotting(mean_sentiment_subtitles), marker_colors=[colors[key] for key in labels_order])])
        st.plotly_chart(fig)

        count_sentiment_df = st.session_state['statistics'][video_select]['face_report_df'].groupby('second_grouped')[['negative', 'neutral', 'positive']].mean().reset_index()
        count_sentiment_df['size'] = st.session_state['statistics'][video_select]['face_report_df'].groupby('second_grouped').size().values
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['size'], name='Cantidad de imágenes', mode='lines+markers'), secondary_y=True)
        fig.add_trace(go.Bar(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['negative'], name='Sentimiento negativo'), secondary_y=False)
        fig.add_trace(go.Bar(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['neutral'], name='Sentimiento neutral'), secondary_y=False)
        fig.add_trace(go.Bar(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['positive'], name='Sentimiento positivo'), secondary_y=False)
        fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos en Faces')
        fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown(f'''## Topics from images''')

        topics_name = st.session_state['statistics'][video_select]['image_topic_info_df'][['label', 'description']].drop_duplicates().set_index('label').to_dict()['description']

        text_images_info = st.session_state['statistics'][video_select]['image_report_df']

        fig = go.Figure()
        _df = text_images_info[text_images_info['cluster'] == -1]
        fig.add_trace(go.Scatter(x=_df['x'], y=_df['y'], hoverinfo='text', mode='markers+text', name='Sin tópico', marker=dict(color='#CFD8DC', size=5, opacity=0.5), showlegend=False))
        all_topics = sorted(text_images_info['cluster'].unique())
        for topic in all_topics:
            if int(topic) == -1:
                continue
            selection = text_images_info[text_images_info['cluster'] == topic]
            label_name = topics_name[topic]
            fig.add_trace(go.Scatter(x=selection['x'], y=selection['y'], hoverinfo='text', mode='markers+text', name=label_name, marker=dict(size=5, opacity=0.5)))
        x_range = [text_images_info['x'].min() - abs(text_images_info['x'].min() * 0.15), text_images_info['x'].max() + abs(text_images_info['x'].max() * 0.15)]
        y_range = [text_images_info['y'].min() - abs(text_images_info['y'].min() * 0.15), text_images_info['y'].max() + abs(text_images_info['y'].max() * 0.15)]
        fig.add_shape(type="rect", x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1], line=dict(color="#CFD8DC", width=2))
        fig.add_shape(type="rect", x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2, line=dict(color="#CFD8DC", width=2))
        fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
        fig.add_annotation(x=sum(x_range) / 2, y=y_range[1], text="D2", showarrow=False, xshift=10)
        fig.update_layout(template='simple_white', title={'text': "<b>", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22, color='Black')})
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(width=WIDTH * 1.5, height=HEIGHT * 0.8)

        st.plotly_chart(fig)

        st.markdown(f'''## Size and sentiment of topics''')
        topic_count_sentiment_df = st.session_state['statistics'][video_select]['image_report_df'].groupby('cluster')[['negative', 'neutral', 'positive']].mean().reset_index()
        topic_count_sentiment_df['size'] = st.session_state['statistics'][video_select]['image_report_df'].groupby('cluster').size().values
        topic_count_sentiment_df = topic_count_sentiment_df[topic_count_sentiment_df['cluster'] != -1].sort_values('cluster')
        topic_count_sentiment_df['topic_label'] = [topics_name[x] for x in topic_count_sentiment_df['cluster']]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['size'], name='Cantidad de imágenes', mode='lines+markers'), secondary_y=True)
        fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['negative'], name='Sentimiento negativo'), secondary_y=False)
        fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['neutral'], name='Sentimiento neutral'), secondary_y=False)
        fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['positive'], name='Sentimiento positivo'), secondary_y=False)
        fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos por tópico')
        fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)
        st.plotly_chart(fig, use_container_width=True)

        _all_topics_sorted = topic_count_sentiment_df.sort_values('size', ascending=False)['cluster'].values


        st.markdown(f'''## Topics over time''')
        fig = go.Figure()
        for topic in _all_topics_sorted:
            if int(topic) == -1:
                continue
            selection = st.session_state['statistics'][video_select]['image_topics_over_seconds_df'][st.session_state['statistics'][video_select]['image_topics_over_seconds_df']['cluster'] == topic]
            label_name = topics_name[topic]
            fig.add_trace(go.Scatter(x=selection['second'], y=selection['size'], hoverinfo='text', mode='lines+markers', name=label_name, marker=dict(size=5, opacity=0.5)))
        fig.update_layout(template='simple_white', title={'text': "<b>", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22, color='Black')})
        fig.update_layout(width=WIDTH * 1.5, height=HEIGHT)
        st.plotly_chart(fig)

        st.markdown(f'''## Best images per topic''')
        topic_best_images_df = st.session_state['statistics'][video_select]['image_best_images_per_topic_df']
        topic_best_images_df['topic_label'] = [topics_name[x] for x in topic_best_images_df['label']]

        _text_images_info = st.session_state['statistics'][video_select]['image_report_df']

        for topic in _all_topics_sorted:
            if int(topic) == -1:
                continue

            st.markdown(f'''### {topics_name[topic]}''')
            _best_images = topic_best_images_df[topic_best_images_df['label'] == topic].sort_values('prob', ascending=False).head(10)
            _best_images_pil = [Image.open(x) for x in _best_images['image_filename'].values]
            st.image(_best_images_pil, width=800, caption=_best_images['description'].values)

            _text_images_info_topic = _text_images_info[_text_images_info['cluster'] == topic]

            count_sentiment_df = _text_images_info_topic.groupby('second_grouped')[['negative', 'neutral', 'positive']].mean().reset_index()
            count_sentiment_df['size'] = _text_images_info_topic.groupby('second_grouped').size().values
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['size'], name='Cantidad de imágenes', mode='lines+markers'), secondary_y=True)
            fig.add_trace(go.Bar(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['negative'], name='Sentimiento negativo'), secondary_y=False)
            fig.add_trace(go.Bar(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['neutral'], name='Sentimiento neutral'), secondary_y=False)
            fig.add_trace(go.Bar(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['positive'], name='Sentimiento positivo'), secondary_y=False)
            fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos en imágenes, tópico: {}'.format(topics_name[topic]))
            fig.update_layout(width=WIDTH * 0.4, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)


    with tab3:
        st.markdown(f'''## Topics from faces''')

        text_images_info = st.session_state['statistics'][video_select]['face_report_df']

        fig = go.Figure()
        _df = text_images_info[text_images_info['cluster'] == -1]
        fig.add_trace(go.Scatter(x=_df['x'], y=_df['y'], hoverinfo='text', mode='markers+text', name='Sin tópico', marker=dict(color='#CFD8DC', size=5, opacity=0.5), showlegend=False))
        all_topics = sorted(text_images_info['cluster'].unique())
        for topic in all_topics:
            if int(topic) == -1:
                continue
            selection = text_images_info[text_images_info['cluster'] == topic]
            label_name = str(topic)
            fig.add_trace(go.Scatter(x=selection['x'], y=selection['y'], hoverinfo='text', mode='markers+text', name=label_name, marker=dict(size=5, opacity=0.5)))
        x_range = [text_images_info['x'].min() - abs(text_images_info['x'].min() * 0.15), text_images_info['x'].max() + abs(text_images_info['x'].max() * 0.15)]
        y_range = [text_images_info['y'].min() - abs(text_images_info['y'].min() * 0.15), text_images_info['y'].max() + abs(text_images_info['y'].max() * 0.15)]
        fig.add_shape(type="rect", x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1], line=dict(color="#CFD8DC", width=2))
        fig.add_shape(type="rect", x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2, line=dict(color="#CFD8DC", width=2))
        fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
        fig.add_annotation(x=sum(x_range) / 2, y=y_range[1], text="D2", showarrow=False, xshift=10)
        fig.update_layout(template='simple_white', title={'text': "<b>", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22, color='Black')})
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(width=WIDTH * 1.5, height=HEIGHT * 0.8)

        st.plotly_chart(fig)

        st.markdown(f'''## Size and sentiment of topics''')
        topic_count_sentiment_df = st.session_state['statistics'][video_select]['face_report_df'].groupby('cluster')[['negative', 'neutral', 'positive']].mean().reset_index()
        topic_count_sentiment_df['size'] = st.session_state['statistics'][video_select]['face_report_df'].groupby('cluster').size().values
        topic_count_sentiment_df = topic_count_sentiment_df[topic_count_sentiment_df['cluster'] != -1].sort_values('cluster')
        topic_count_sentiment_df['topic_label'] = [str(x) for x in topic_count_sentiment_df['cluster']]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['size'], name='Cantidad de imágenes', mode='lines+markers'), secondary_y=True)
        fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['negative'], name='Sentimiento negativo'), secondary_y=False)
        fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['neutral'], name='Sentimiento neutral'), secondary_y=False)
        fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['positive'], name='Sentimiento positivo'), secondary_y=False)
        fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos por tópico')
        fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)
        st.plotly_chart(fig, use_container_width=True)

        _all_topics_sorted = topic_count_sentiment_df.sort_values('size', ascending=False)['cluster'].values


        st.markdown(f'''## Topics over time''')
        fig = go.Figure()
        for topic in _all_topics_sorted:
            if int(topic) == -1:
                continue
            selection = st.session_state['statistics'][video_select]['face_topics_over_seconds_df'][st.session_state['statistics'][video_select]['face_topics_over_seconds_df']['cluster'] == topic]
            label_name = str(topic)
            fig.add_trace(go.Scatter(x=selection['second'], y=selection['size'], hoverinfo='text', mode='lines+markers', name=label_name, marker=dict(size=5, opacity=0.5)))
        fig.update_layout(template='simple_white', title={'text': "<b>", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22, color='Black')})
        fig.update_layout(width=WIDTH * 1.5, height=HEIGHT)
        st.plotly_chart(fig)


        st.markdown(f'''## Best images per topic''')
        topic_best_images_df = st.session_state['statistics'][video_select]['face_best_images_per_topic_df']
        topic_best_images_df['topic_label'] = [str(x) for x in topic_best_images_df['label']]

        for topic in _all_topics_sorted:
            if int(topic) == -1:
                continue

            st.markdown(f'''### {str(topic)}''')
            _best_images = topic_best_images_df[topic_best_images_df['label'] == topic].sort_values('prob', ascending=False).head(10)
            _best_images_pil = [Image.open(x) for x in _best_images['image_filename'].values]
            st.image(_best_images_pil, width=600, caption=_best_images['prob'].values)


            _text_images_info_topic = text_images_info[text_images_info['cluster'] == topic]

            count_sentiment_df = _text_images_info_topic.groupby('second_grouped')[['negative', 'neutral', 'positive']].mean().reset_index()
            count_sentiment_df['size'] = _text_images_info_topic.groupby('second_grouped').size().values
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['size'], name='Cantidad de imágenes', mode='lines+markers'), secondary_y=True)
            fig.add_trace(go.Bar(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['negative'], name='Sentimiento negativo'), secondary_y=False)
            fig.add_trace(go.Bar(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['neutral'], name='Sentimiento neutral'), secondary_y=False)
            fig.add_trace(go.Bar(x=count_sentiment_df['second_grouped'], y=count_sentiment_df['positive'], name='Sentimiento positivo'), secondary_y=False)
            fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos en Faces, tópico: {}'.format(str(topic)))
            fig.update_layout(width=WIDTH * 0.4, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)


    with tab4:
        st.markdown(f'''## Subtitles''')
        subtitles_df = st.session_state['statistics'][video_select]['summary_df']
        st.dataframe(subtitles_df)
