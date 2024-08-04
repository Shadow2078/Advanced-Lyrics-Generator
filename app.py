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
from textblob.download_corpora import download_all
import bcrypt
import sqlite3
from PIL import Image
import io

# Load environment variables
dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
download_all()

# Database setup
def get_db_connection():
    conn = sqlite3.connect('app.db')
    return conn

def initialize_database():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, name TEXT, password TEXT, profile_pic BLOB)''')
        c.execute('''CREATE TABLE IF NOT EXISTS admins (email TEXT PRIMARY KEY, name TEXT, password TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS lyrics (id INTEGER PRIMARY KEY, email TEXT, prompt TEXT, lyrics TEXT, rating INTEGER, feedback TEXT)''')
        conn.commit()

initialize_database()

# Insert initial admin user
def insert_initial_admin():
    admin_email = "admin@example.com"
    admin_name = "Admin"
    admin_password = bcrypt.hashpw("admin_password".encode(), bcrypt.gensalt()).decode()
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM admins WHERE email=?', (admin_email,))
        if not c.fetchone():
            c.execute('INSERT INTO admins (email, name, password) VALUES (?, ?, ?)', (admin_email, admin_name, admin_password))
        conn.commit()

insert_initial_admin()

# Authentication function
def authenticate(email, password, login_type):
    table = 'admins' if login_type == 'Admin' else 'users'
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute(f'SELECT * FROM {table} WHERE email=?', (email,))
        user = c.fetchone()
    if user:
        if bcrypt.checkpw(password.encode(), user[2].encode()):
            return user
    return None

# User signup function
def signup(email, name, password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE email=?', (email,))
            if c.fetchone():
                return False, "Email already exists"
            c.execute('INSERT INTO users (email, name, password) VALUES (?, ?, ?)', (email, name, hashed_password))
            conn.commit()
            return True, "Signup successful! Please login."
    except Exception as e:
        return False, f"Error during signup: {e}"

# Update user profile
def update_profile(email, name, password=None, profile_pic=None):
    with get_db_connection() as conn:
        c = conn.cursor()
        if password:
            hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            c.execute('UPDATE users SET name=?, password=?, profile_pic=? WHERE email=?', (name, hashed_password, profile_pic, email))
        else:
            c.execute('UPDATE users SET name=?, profile_pic=? WHERE email=?', (name, profile_pic, email))
        conn.commit()

# Load user profile picture
def load_profile_pic(email):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT profile_pic FROM users WHERE email=?', (email,))
        profile_pic = c.fetchone()
    if profile_pic and profile_pic[0]:
        return Image.open(io.BytesIO(profile_pic[0]))
    return None

# Save prompt, generated lyrics, rating, and feedback to the database
def save_prompt_and_lyrics(email, prompt, lyrics, rating=None, feedback=None):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('INSERT INTO lyrics (email, prompt, lyrics, rating, feedback) VALUES (?, ?, ?, ?, ?)', (email, prompt, lyrics, rating, feedback))
        conn.commit()
        # Ensure only the latest 5 entries are kept
        c.execute('DELETE FROM lyrics WHERE id NOT IN (SELECT id FROM lyrics WHERE email=? ORDER BY id DESC LIMIT 5)', (email,))
        conn.commit()

# Word Cloud Visualization
def generate_word_cloud(lyrics):
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(lyrics)
    plt.figure(figsize=(6, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Evaluation Metrics
# 1. Phonetic Quality Analysis
def phonetic_quality_analysis(lyrics):
    words = word_tokenize(lyrics)
    freq_dist = Counter(words)
    rhyme_density = sum([freq_dist[word] for word in freq_dist if word.endswith(('ee', 'ay', 'ow', 'ine', 'ing'))])
    return rhyme_density

# 2. Thematic Depth Analysis
def thematic_depth_analysis(lyrics):
    blob = TextBlob(lyrics)
    themes = blob.noun_phrases
    return themes

# 3. Technical Accuracy Analysis
def technical_accuracy_analysis(lyrics):
    blob = TextBlob(lyrics)
    grammar_errors = len(blob.correct().words) - len(blob.words)
    return grammar_errors

# 4. Sentiment Analysis
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment

# Streamlit App Title
st.set_page_config(page_title="Rap Lyrics Generator", page_icon="🎤", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
        .stApp {
            background-color: #1E1E1E;
            color: white;
        }
        .stTextInput, .stSlider, .stTextArea, .stButton, .stSelectbox, .stMarkdown {
            background-color: #333333;
            color: white;
        }
        .stButton button {
            background-color: #FF4B4B;
            color: white;
        }
        .stTextInput input, .stTextArea textarea, .stSelectbox select, .stSlider .stSliderValue span {
            background-color: #333333;
            color: white;
        }
        .stHeader {
            font-size: 24px;
            font-weight: bold;
        }
        .stSubheader {
            font-size: 20px;
            font-weight: bold;
        }
        .stTable {
            background-color: #333333;
            color: white;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #FF4B4B;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎤 Advanced Rap Lyrics Generator for Rap Music Enthusiasts in Nepal")

# User authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    auth_action = st.selectbox("Choose Action", ["Login", "Signup"], index=0)

    if auth_action == "Login":
        login_type = st.selectbox("Login as", ["User", "Admin"], index=0)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = authenticate(email, password, login_type)
            if user:
                if login_type == "Admin" and email != "admin@example.com":
                    st.error("Invalid admin credentials")
                else:
                    st.session_state['authenticated'] = True
                    st.session_state['email'] = email
                    st.session_state['name'] = user[1]
                    st.session_state['is_admin'] = login_type == "Admin"
                    st.sidebar.success(f"Welcome {user[1]}")
            else:
                st.error("Invalid email or password")
    elif auth_action == "Signup":
        new_email = st.text_input("Enter your Email")
        new_name = st.text_input("Enter your Name")
        new_password = st.text_input("Choose a Password (min 8 characters)", type="password")
        if st.button("Signup"):
            success, message = signup(new_email, new_name, new_password)
            if success:
                st.success(message)
            else:
                st.error(message)
else:
    st.sidebar.success(f"Welcome {st.session_state['name']}")
    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.session_state['is_admin'] = False

    # Navigation
    if st.session_state.get('is_admin', False):
        page = st.sidebar.selectbox("Navigate", ["Admin Dashboard"])
    else:
        page = st.sidebar.selectbox("Navigate", ["Generate Lyrics", "Analysis", "Profile", "History"])

    # Function to generate rap lyrics using OpenAI API
    def generate_rap_lyrics(prompt, style=None):
        system_content = """You are a talented rapper known for your clever wordplay, intricate rhymes, and profound storytelling like a J Cole, Tupac, Nas. Use more slangs. Your goal is to create rap lyrics that are both meaningful and engaging, showcasing advanced lyricism. The lyrics should include:
            - Complex rhyme schemes and multi-syllabic rhymes
            - Vivid imagery and descriptive language
            - A strong narrative or theme
            - Clever metaphors and similes
            - A smooth flow and rhythm
        """
        if style:
            system_content += f"\n\nThe lyrics should mimic the style of {style}."
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_content
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

    if page == "Generate Lyrics":
        st.header("Generate Rap Lyrics")
        # User input for the prompt
        prompt = st.text_input("What shall I Rap about? :", "About Kathmandu?")
        style = st.selectbox("Choose a rap style or artist influence:", ["None", "J Cole", "Tupac", "Nas"])
        if st.button("Generate Lyrics"):
            with st.spinner("Generating lyrics..."):
                generated_lyrics = generate_rap_lyrics(prompt, style if style != "None" else None)
                if generated_lyrics:
                    st.subheader("Generated Lyrics")
                    st.write(generated_lyrics)
                    save_prompt_and_lyrics(st.session_state['email'], prompt, generated_lyrics)
                    st.subheader("Rate the Lyrics and Provide Feedback")
                    rating = st.slider("Rating", 1, 5)
                    feedback = st.text_area("Feedback")
                    if st.button("Submit Feedback"):
                        save_prompt_and_lyrics(st.session_state['email'], prompt, generated_lyrics, rating, feedback)
                        st.success("Feedback submitted successfully")

    elif page == "Analysis":
        st.header("Analysis of Generated Lyrics")
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT id, lyrics FROM lyrics WHERE email=?', (st.session_state['email'],))
            user_lyrics = c.fetchall()
        if user_lyrics:
            lyrics_dict = {id: lyrics for id, lyrics in user_lyrics}
            selected_lyrics_id = st.selectbox("Select lyrics to analyze", list(lyrics_dict.keys()))
            selected_lyrics = lyrics_dict[selected_lyrics_id]

            if selected_lyrics:
                st.subheader("Selected Lyrics")
                st.write(selected_lyrics)

                # Word Cloud Visualization
                st.subheader("Word Cloud")
                generate_word_cloud(selected_lyrics)
                st.write("The Word Cloud above visualizes the most frequent words in the generated rap lyrics. Larger words appear more frequently, giving an insight into the key themes and vocabulary used.")
                
                # Sentiment Analysis
                st.subheader("Sentiment Analysis")
                sentiment = get_sentiment(selected_lyrics)
                st.write("Polarity:", sentiment.polarity)
                st.write("Subjectivity:", sentiment.subjectivity)
                labels = ['Polarity', 'Subjectivity']
                sizes = [max(sentiment.polarity, 0), max(sentiment.subjectivity, 0)]
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#FF5252'])
                ax.axis('equal')
                st.pyplot(fig)
                
                # Part-of-Speech (POS) Tagging Analysis
                st.subheader("Part-of-Speech (POS) Tagging Analysis")
                tokens = word_tokenize(selected_lyrics)
                pos_tags = pos_tag(tokens)
                pos_counts = Counter(tag for word, tag in pos_tags)
                pos_df = pd.DataFrame(pos_counts.items(), columns=['POS', 'Count']).sort_values(by='Count', ascending=False)
                st.write(pos_df)
                plt.figure(figsize=(8, 4))
                sns.barplot(x='Count', y='POS', data=pos_df, palette='viridis')
                plt.title('Part-of-Speech Tagging in Generated Lyrics')
                st.pyplot(plt)
                
                # Phonetic Quality Analysis
                st.subheader("Phonetic Quality Analysis")
                rhyme_density = phonetic_quality_analysis(selected_lyrics)
                st.write("Rhyme Density:", rhyme_density)
                
                # Thematic Depth Analysis
                st.subheader("Thematic Depth Analysis")
                themes = thematic_depth_analysis(selected_lyrics)
                st.write("Identified Themes:", ', '.join(themes))
                
                # Technical Accuracy Analysis
                st.subheader("Technical Accuracy Analysis")
                grammar_errors = technical_accuracy_analysis(selected_lyrics)
                st.write("Grammar Errors:", grammar_errors)
        else:
            st.warning("No lyrics available for analysis. Please generate and save lyrics first.")

    elif page == "Profile":
        st.header("Profile")
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT name FROM users WHERE email=?', (st.session_state['email'],))
            name = c.fetchone()[0]
        new_name = st.text_input("Name", value=name)
        new_password = st.text_input("New Password", type="password")
        profile_pic = st.file_uploader("Upload Profile Picture", type=["jpg", "jpeg", "png"])
        if st.button("Update Profile"):
            profile_pic_data = profile_pic.read() if profile_pic else None
            update_profile(st.session_state['email'], new_name, new_password if new_password else None, profile_pic_data)
            st.success("Profile updated successfully")
        st.subheader("Profile Picture")
        if profile_pic_data:
            st.image(profile_pic_data, caption='Profile Picture')
        else:
            st.image(load_profile_pic(st.session_state['email']), caption='Profile Picture')

    elif page == "History":
        st.header("Your History")
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT id, prompt, lyrics, rating, feedback FROM lyrics WHERE email=?', (st.session_state['email'],))
            history = c.fetchall()
        if history:
            for entry in history:
                st.write(f"**Prompt:** {entry[1]}")
                st.write(f"**Lyrics:** {entry[2]}")
                if entry[3]:
                    st.write(f"**Rating:** {entry[3]}")
                if entry[4]:
                    st.write(f"**Feedback:** {entry[4]}")
                st.write("---")
            if st.button("Clear History"):
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute('DELETE FROM lyrics WHERE email=?', (st.session_state['email'],))
                    conn.commit()
                st.success("History cleared successfully")
        else:
            st.warning("No history available")

    elif page == "Admin Dashboard" and st.session_state['email'] == "admin@example.com":
        st.header("Admin Dashboard")
        st.write("Manage users and view usage statistics.")
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT email, name FROM users')
            users = c.fetchall()
        df_users = pd.DataFrame(users, columns=['Email', 'Name'])
        st.write("### User List")
        st.table(df_users)
        user_to_delete = st.text_input("Enter email to delete")
        if st.button("Delete User"):
            with get_db_connection() as conn:
                c = c.cursor()
                c.execute('DELETE FROM users WHERE email=?', (user_to_delete,))
                conn.commit()
            st.success("User deleted successfully")
        st.write("### Prompts and Lyrics")
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT email, prompt, lyrics, rating, feedback FROM lyrics')
            prompts_and_lyrics = c.fetchall()
        for entry in prompts_and_lyrics:
            st.write(f"**User:** {entry[0]}")
            st.write(f"**Prompt:** {entry[1]}")
            st.write(f"**Lyrics:** {entry[2]}")
            if entry[3]:
                st.write(f"**Rating:** {entry[3]}")
            if entry[4]:
                st.write(f"**Feedback:** {entry[4]}")
            st.write("---")
        else:
            st.warning("Please enter your email and password.")
