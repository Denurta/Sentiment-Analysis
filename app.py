import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import os
import pandas as pd
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import time
from pathlib import Path
from PIL import Image

# Get the current directory
current_dir = Path(__file__).parent if '__file__' in locals() else Path.cwd()

# Print the current directory to verify
st.write(f"Current directory: {current_dir}")

# Set NLTK data path to the local directory
nltk_data_path = current_dir / 'nltk_data'
nltk.data.path.append(str(nltk_data_path))

# Load Indonesian stopwords
stop_words_id = set(stopwords.words('indonesian'))

# Function to clean text
def clean_text(text):
    text = ' '.join([word.lower() for word in text.split() if word.isalpha() and word.lower() not in stop_words_id])
    return text

# Load model and vectorizer
model_path = current_dir / 'esg_sentiment_model.sav'
vectorizer_path = current_dir / 'esg_vectorizer.sav'

try:
    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    st.error('Model or vectorizer not found. Please ensure "esg_sentiment_model.sav" and "esg_vectorizer.sav" are in the correct directory.')
except EOFError:
    st.error('EOFError: The model file appears to be corrupted. Please check the file and try again.')

# Function to predict sentiment using the loaded model
def predict_sentiment(text):
    text_vector = loaded_vectorizer.transform([text])
    text_vector_dense = text_vector.toarray()
    sentiment = loaded_model.predict(text_vector_dense)[0]
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to predict sentiment using TextBlob
def predict_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Initialize session state for history if not already done
if 'history' not in st.session_state:
    st.session_state.history = []

# Set page configuration
st.set_page_config(
    page_title="Sentiment Prediction",
    layout="wide",
    page_icon="📊"
)

# Load the image
image_path = current_dir / 'image.png'

try:
    image = Image.open(image_path)
except FileNotFoundError:
    st.error('Image file not found. Please ensure "image.png" is in the correct directory.')

# Sidebar for navigation
with st.sidebar:
    if 'image' in locals():
        st.image(image, width=50)  # Adjust width as needed
    selected = option_menu(
        'Sentiment Classifier',
        [
            'Single Analysis',
            'Batch Analysis',
            'Analysis History'
        ],
        icons=['chat', 'list-task', 'clock-history'],
        default_index=0
    )

# Single Analysis Page
if selected == 'Single Analysis':
    st.title('Single Sentiment Analysis')

    input_text = st.text_area('Enter text for sentiment analysis:', height=200)

    # Button to process prediction
    if st.button('Analyze Sentiment'):
        if input_text:
            start_time = time.time()
            cleaned_text = clean_text(input_text)
            sentiment_result_model = predict_sentiment(cleaned_text)
            sentiment_result_textblob = predict_sentiment_textblob(input_text)
            end_time = time.time()
            prediction_time = end_time - start_time

            st.write(f"#### Time taken for prediction: {prediction_time:.4f} seconds")

            st.write("#### Model-based Sentiment Analysis")
            if sentiment_result_model == 'Positive':
                st.success(f'The text has a {sentiment_result_model} sentiment.')
            elif sentiment_result_model == 'Negative':
                st.error(f'The text has a {sentiment_result_model} sentiment.')
            else:
                st.warning(f'The text has a {sentiment_result_model} sentiment.')

            st.write("#### TextBlob Sentiment Analysis")
            if sentiment_result_textblob == 'Positive':
                st.success(f'The text has a {sentiment_result_textblob} sentiment.')
            elif sentiment_result_textblob == 'Negative':
                st.error(f'The text has a {sentiment_result_textblob} sentiment.')
            else:
                st.warning(f'The text has a {sentiment_result_textblob} sentiment.')

            # Save to history
            st.session_state.history.append({
                'text': input_text,
                'model_sentiment': sentiment_result_model,
                'textblob_sentiment': sentiment_result_textblob,
                'prediction_time': prediction_time
            })

# Batch Analysis Page
if selected == 'Batch Analysis':
    st.title('Batch Sentiment Analysis')

    st.markdown("""
    <style>
    .result-table {
        margin-top: 20px;
    }
    .result-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
        padding: 10px;
        border: 1px solid #ccc;
    }
    .result-row .text {
        flex: 3;
        margin-right: 10px;
    }
    .result-row .sentiment {
        flex: 1;
    }
    .btn {
        padding: 5px 10px;
        border: none;
        color: white;
        cursor: pointer.
    }
    .btn.positive {
        background-color: green.
    }
    .btn.negative {
        background-color: red.
    }
    .btn.neutral {
        background-color: yellow.
    }
    </style>
    """, unsafe_allow_html=True)

    upl = st.file_uploader('Choose File', type='xlsx')

    if upl:
        df = pd.read_excel(upl)
        if 'Text' not in df.columns:
            st.error('The uploaded file must contain a "Text" column.')
        else:
            start_time = time.time()
            df['Analysis_Model'] = df['Text'].apply(lambda x: predict_sentiment(clean_text(x)))
            df['Analysis_TextBlob'] = df['Text'].apply(predict_sentiment_textblob)
            end_time = time.time()
            prediction_time = end_time - start_time

            st.write(f"#### Time taken for batch prediction: {prediction_time:.4f} seconds")
            st.write(df[['Text', 'Analysis_Model', 'Analysis_TextBlob']])

            # Save to history
            for index, row in df.iterrows():
                st.session_state.history.append({
                    'text': row['Text'],
                    'model_sentiment': row['Analysis_Model'],
                    'textblob_sentiment': row['Analysis_TextBlob'],
                    'prediction_time': prediction_time
                })

# Analysis History Page
if selected == 'Analysis History':
    st.title('Analysis History')

    if st.button("Delete All Records"):
        st.session_state.history.clear()
        st.experimental_rerun()

    # Display the last 20 records
    if st.session_state.history:
        # Convert history to DataFrame for displaying in table format
        history_df = pd.DataFrame(st.session_state.history[-20:][::-1])  # Get last 20 records in reverse order

        # Display each record with a delete button
        for i in range(len(history_df)):
            st.write(f"**Text:** {history_df.iloc[i]['text']}")
            st.markdown(f"**Model Sentiment:** {history_df.iloc[i]['model_sentiment']}")
            st.markdown(f"**TextBlob Sentiment:** {history_df.iloc[i]['textblob_sentiment']}")
            prediction_time = history_df.iloc[i].get('prediction_time', 'N/A')
            st.markdown(f"**Prediction Time:** {prediction_time:.4f} seconds" if prediction_time != 'N/A' else "**Prediction Time:** N/A")
            if st.button(f"Delete Record {i + 1}", key=f"history_delete_{i}"):
                st.session_state.history.pop(-i - 1)  # Remove from history in reverse order
                st.experimental_rerun()
    else:
        st.write("No history records found.")
