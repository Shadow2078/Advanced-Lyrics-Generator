import os
import streamlit as st
from openai import OpenAI
import dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
from nltk.util import ngrams
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

dotenv.load_dotenv()

client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Streamlit App Title and Description
st.title("Advanced Rap Lyrics Generator for Rap Music Enthusiasts in Nepal")
st.write("Generate rap lyrics based on your prompts and visualize the results with advanced analytics.")

# User input for the prompt
prompt = st.text_input("What shall I Rap about ? :", "About Kathmandu ? ")

# Function to generate rap lyrics using OpenAI API
def generate_rap_lyrics(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
        {
            "role": "system",
            "content": """You are a talented rapper known for your clever wordplay, intricate rhymes, and profound storytelling like a J Cole. Your goal is to create rap lyrics that are both meaningful and engaging, showcasing advanced lyricism. The lyrics should include:
                - Complex rhyme schemes and multi-syllabic rhymes
                - Vivid imagery and descriptive language
                - A strong narrative or theme
                - Clever metaphors and similes
                - A smooth flow and rhythm

                Please write rap lyrics"""
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
            max_tokens=300,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0.1,
            presence_penalty=0.1,

        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None
    
# Function to perform sentiment analysis with positive, negative, and neutral classifications
def get_sentiment_category(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'




# Generate lyrics on button click
if st.button("Generate Lyrics"):
    with st.spinner("Generating lyrics..."):
        generated_lyrics = generate_rap_lyrics(prompt)
        if generated_lyrics:
            st.subheader("Generated Lyrics")
            st.write(generated_lyrics)
            
            # Word Cloud Visualization
            st.subheader("Word Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(generated_lyrics)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
              # Explanation of Word Cloud
            st.write("The Word Cloud above visualizes the most frequent words in the generated rap lyrics. "
                     "Larger words appear more frequently, giving an insight into the key themes and vocabulary used.")
            
            
            # Sentiment Analysis
            st.subheader("Sentiment Analysis")
            def get_sentiment(text):
                blob = TextBlob(text)
                return blob.sentiment

            sentiment = get_sentiment(generated_lyrics)
            st.write("Polarity:", sentiment.polarity)
            st.write("Subjectivity:", sentiment.subjectivity)
            labels = ['Polarity', 'Subjectivity']
            sizes = [sentiment.polarity, sentiment.subjectivity]
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#FF5252'])
            ax.axis('equal')
            st.pyplot(fig)

           

            # Part-of-Speech (POS) Tagging Analysis
            st.subheader("Part-of-Speech (POS) Tagging Analysis")
            tokens = word_tokenize(generated_lyrics)
            pos_tags = pos_tag(tokens)
            pos_counts = Counter(tag for word, tag in pos_tags)
            pos_df = pd.DataFrame(pos_counts.items(), columns=['POS', 'Count']).sort_values(by='Count', ascending=False)
            st.write(pos_df)
            plt.figure(figsize=(10, 5))
            sns.barplot(x='Count', y='POS', data=pos_df)
            plt.title('Part-of-Speech Tagging in Generated Lyrics')
            st.pyplot(plt)

# Optional: Save and Load Generated Lyrics
st.sidebar.header("Save and Load Lyrics")
if 'generated_lyrics' not in st.session_state:
    st.session_state['generated_lyrics'] = []

if st.button("Save Lyrics"):
    st.session_state['generated_lyrics'].append(generate_rap_lyrics)
    st.sidebar.write("Lyrics Saved!")

if st.sidebar.button("Load Saved Lyrics"):
    if st.session_state['generated_lyrics']:
        st.sidebar.subheader("Saved Lyrics")
        for idx, lyrics in enumerate(st.session_state['generated_lyrics']):
            st.sidebar.write(f"Lyrics {idx+1}:\n{lyrics}\n")
    else:
        st.sidebar.write("No saved lyrics found.")
