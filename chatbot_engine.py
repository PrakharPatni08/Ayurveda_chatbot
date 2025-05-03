import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

known_terms = set()
INDEX_PATH = "vectorstore"

# List of diseases with no remedies
no_remedy_diseases = {
    "cancer", "covid-19", "pneumonia", "chronic kidney disease", "hiv/aids", "heart attack", 
    "stroke", "chronic obstructive pulmonary disease", "cystic fibrosis", "sickle cell anemia", 
    "amyotrophic lateral sclerosis", "hepatitis c", "pancreatitis", "polycystic kidney disease", 
    "glaucoma", "end-stage liver disease", "chronic fatigue syndrome", "hemophilia", "liver abscess", 
    "chronic hepatitis b", "chronic bronchitis", "emphysema", "pneumothorax", "kidney stones", 
    "tetanus", "sars", "meningitis", "tuberculosis", "chikungunya virus", "malaria", "dengue", "leukemia"
}

def load_data(path="data/herbal_remedies.csv"):
    df = pd.read_csv(path)
    df.fillna("", inplace=True)

    global known_terms
    known_diseases = df['Disease'].str.lower().tolist()
    known_symptoms = df['Symptoms'].str.lower().tolist()
    known_terms = set(known_diseases + known_symptoms)

    df['content'] = (
        "Disease: " + df['Disease'] +
        "\nSymptoms: " + df['Symptoms'] +
        "\nRemedy: " + df['Remedy'] +
        "\nUsage: " + df['Usage']
    )
    return df[['content']]

def create_vector_index(df):
    loader = DataFrameLoader(df, page_content_column="content")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs_split, embeddings)

    # Save index for future reuse
    vectorstore.save_local(INDEX_PATH)
    print("✅ FAISS index created and saved.")

    return vectorstore

def load_vector_index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings)
    print("✅ FAISS index loaded from disk.")
    return vectorstore

def get_recommendation(query, vectorstore, k=3):
    query_lower = query.lower()

    if any(disease in query_lower for disease in no_remedy_diseases):
        return ["🌱 We don’t have a remedy for this disease yet, but the search for Ayurvedic solutions is still underway."]

    try:
        results = vectorstore.similarity_search(query_lower, k=k)
    except Exception as e:
        print(f"❌ Error during similarity search: {e}")
        return ["🌱 There was an error processing your request. Please try again later."]

    if not results:
        return ["🌱 We don’t have a remedy for this disease yet, but the search for Ayurvedic solutions is still underway."]

    return [r.page_content for r in results]
