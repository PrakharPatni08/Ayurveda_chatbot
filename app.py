import streamlit as st
from chatbot_engine import load_data, create_vector_index, get_recommendation
from streamlit_extras.stylable_container import stylable_container
from streamlit_lottie import st_lottie
import json

# --- Functions ---
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# --- Page config ---
st.set_page_config(page_title="🌿 HerbSphere Remedy Chatbot", layout="centered", page_icon="🌿")

# --- Load Animation ---
try:
    lottie_herb = load_lottiefile("C:/Users/admin/OneDrive/Desktop/ayurveda_chatbot/lottie_herb_animation.json")  # Path to your Lottie file
except Exception :
    lottie_herb = None

# --- Top Animation ---
if lottie_herb:
    st_lottie(lottie_herb, height=250, speed=1)

# --- Title and description inside a container ---
st.title("🌿 HerbSphere Remedy Chatbot")
st.markdown("Type your **disease** or **symptoms**, and discover Ayurvedic remedies tailored for you! ✨")

query = st.text_input(
    "🔍 Enter your symptoms or disease name",
    placeholder="E.g., headache, arthritis, obesity...",
    help="Type a symptom or disease and get remedies."
)

# --- Cache chatbot setup ---
@st.cache_resource(show_spinner="Setting up your herbal assistant...")
def setup_chatbot():
    df = load_data()
    st.write(f"📄 Loaded {len(df)} remedies from the dataset.")
    vectorstore = create_vector_index(df)
    print(f"Vectorstore has {vectorstore.index.ntotal} documents.")  # Updated line
    return vectorstore


# --- Initialize chatbot ---
vectorstore = setup_chatbot()

# --- Handle the query ---
if query:
    with st.spinner('🧘‍♀️ Finding the perfect remedy for you...'):
        results = get_recommendation(query, vectorstore)

    if results:
        st.success("🌼 Here are some recommendations for you:")
        for res in results:
            with stylable_container(
                key=f"card_{res}",
                css_styles="""
                    {
                        border: 1px solid #cde0d6;
                        border-radius: 12px;
                        padding: 1rem;
                        margin-top: 1rem;
                        background-color: #ffffff;
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
                        font-size: 1rem;
                        color: #333333;
                        font-size: 1.1rem;
                    }
                """,
            ):
                st.markdown(f"🧾 {res}")
    else:
        st.warning("🧪 This disease's remedy is still under research. We're working to find the best Ayurvedic solutions soon!")

# --- Footer ---
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

