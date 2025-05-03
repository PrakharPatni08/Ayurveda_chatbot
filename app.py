import streamlit as st
from chatbot_engine import load_data, create_vector_index, load_vector_index, get_recommendation
from streamlit_extras.stylable_container import stylable_container
from streamlit_lottie import st_lottie
import json
import os

# --- Load Lottie animation ---
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"⚠️ Could not load Lottie animation: {e}")
        return None

st.set_page_config(page_title="🌿 HerbSphere Remedy Chatbot", layout="centered", page_icon="🌿")

# Animation section
lottie_path = os.path.join("lottie_herb_animation.json")
lottie_herb = load_lottiefile(lottie_path)
if lottie_herb:
    st_lottie(lottie_herb, height=250, speed=1)

# Title & Input
st.title("🌿 HerbSphere Remedy Chatbot")
st.markdown("Type your **disease** or **symptoms**, and discover Ayurvedic remedies tailored for you! ✨")

query = st.text_input("🔍 Enter your symptoms or disease name", placeholder="E.g., headache, arthritis, obesity...")

# Setup chatbot
@st.cache_resource(show_spinner="Setting up your herbal assistant...")
def setup_chatbot():
    if not os.path.exists("vectorstore"):
        df = load_data()
        return create_vector_index(df)
    else:
        return load_vector_index()

vectorstore = setup_chatbot()

# Process input
if query:
    with st.spinner('🧘‍♀️ Finding the perfect remedy for you...'):
        results = get_recommendation(query, vectorstore)

    if results:
        st.success("🌼 Here are some recommendations for you:")
        for res in results:
            with stylable_container(
                key=f"card_{hash(res)}",
                css_styles="""
                    {
                        border: 1px solid #cde0d6;
                        border-radius: 12px;
                        padding: 1rem;
                        margin-top: 1rem;
                        background-color: #ffffff;
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
                        font-size: 1.1rem;
                        color: #333333;
                    }
                """,
            ):
                st.markdown(f"🧾 {res}")
    else:
        st.warning("🧪 This disease's remedy is still under research. We're working to find the best Ayurvedic solutions soon!")

# Footer
st.markdown(
    """
    <hr style="margin-top: 3rem; margin-bottom: 1rem;">
    <div style="text-align: center; font-size: 0.9rem; color: #555;">
        Crafted with care and expertise by <b>Team HerbSphere</b>.<br>
        Empowering holistic wellness through Ayurvedic wisdom. 🌿
    </div>
    """,
    unsafe_allow_html=True,
)
