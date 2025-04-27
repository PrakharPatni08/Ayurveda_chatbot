#import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DataFrameLoader

# 🔐 Load environment variables
load_dotenv()

# Global variables to store known disease and symptom keywords
known_terms = set()

# 📂 Load CSV into DataFrame
def load_data(path="data/herbal_remedies.csv"):
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    
    # Collect known diseases and symptoms
    global known_terms
    known_diseases = df['Disease'].str.lower().tolist()
    known_symptoms = df['Symptoms'].str.lower().tolist()
    known_terms = set(known_diseases + known_symptoms)
    
    print(f"Known terms: {known_terms}")  # Debugging line
    df['content'] = (
        "Disease: " + df['Disease'] +
        "\nSymptoms: " + df['Symptoms'] +
        "\nRemedy: " + df['Remedy'] +
        "\nUsage: " + df['Usage']
    )
    return df[['content']]

# 🧠 Create FAISS index for searching
def create_vector_index(df):
    loader = DataFrameLoader(df, page_content_column="content")
    docs = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Local model
    vectorstore = FAISS.from_documents(docs_split, embeddings)
    
    print(f"Vectorstore has {vectorstore.index.ntotal} documents.")  # Corrected line

    return vectorstore

# 🔍 Recommendation with keyword check
def get_recommendation(query, vectorstore, k=3):
    # Check if query contains known disease/symptom
    query_lower = query.lower()
    print(f"Checking for known terms in query: {query_lower}")  # Debugging line
    if not any(term in query_lower for term in known_terms):
        print("No known terms found, returning default message.")  # Debugging line
        return ["🌱 We don’t have a remedy for this disease yet, but the search for Ayurvedic solutions is still underway."]

    # Proceed with vector search
    print("Query contains known terms, proceeding with vector search.")  # Debugging line
    results = vectorstore.similarity_search(query, k=k)
    
    print(f"Similarity search results: {results}")  # Debugging line
    if not results:
        print("No results found in similarity search.")  # Debugging line
        return ["🌱 We don’t have a remedy for this disease yet, but the search for Ayurvedic solutions is still underway."]
    
    return [r.page_content for r in results]

# Main execution
if __name__ == "__main__":
    # Load the data
    df = load_data()
    
    # Create vectorstore (FAISS index)
    vectorstore = create_vector_index(df)
    
    # Example usage of the get_recommendation function
    query = "headache"  # Example query
    recommendations = get_recommendation(query, vectorstore, k=3)
    
    for recommendation in recommendations:
        print(recommendation)
