import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DataFrameLoader

# Global variables to store known disease and symptom keywords
known_terms = set()

# 📂 Load CSV into DataFrame
def load_data(path="data/herbal_remedies.csv"):
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    
    # Collect known diseases and symptoms, ensuring case insensitivity
    global known_terms
    known_diseases = df['Disease'].str.lower().tolist()
    known_symptoms = df['Symptoms'].str.lower().tolist()
    known_terms = set(known_diseases + known_symptoms)
    
    print(f"Known terms after loading data: {known_terms}")  # Debugging line
    
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
    
    print(f"Vectorstore has {vectorstore.index.ntotal} documents.")  # Debugging line

    return vectorstore

# 🔍 Recommendation using Langchain's similarity search
# List of diseases that should return no remedies found
no_remedy_diseases = {
    "cancer", "covid-19", "pneumonia", "chronic kidney disease", "hiv/aids", "heart attack", 
    "stroke", "chronic obstructive pulmonary disease", "cystic fibrosis", "sickle cell anemia", 
    "amyotrophic lateral sclerosis", "hepatitis c", "pancreatitis", "polycystic kidney disease", 
    "glaucoma", "end-stage liver disease", "chronic fatigue syndrome", "hemophilia", "liver abscess", 
    "chronic hepatitis b", "chronic bronchitis", "emphysema", "pneumothorax", "kidney stones", 
    "tetanus", "sars", "meningitis", "tuberculosis", "chikungunya virus", "malaria", "dengue", "leukemia"
}

def get_recommendation(query, vectorstore, k=3):
    query_lower = query.lower()
    
    # Debugging: Print the query being processed
    print(f"Query: {query_lower}")
    print(f"Known terms: {known_terms}")
    
    # Check if the query matches any disease from the no_remedy_diseases list
    if any(disease in query_lower for disease in no_remedy_diseases):
        print("Query matches a disease with no remedy.")
        return ["🌱 We don’t have a remedy for this disease yet, but the search for Ayurvedic solutions is still underway."]
    
    # Perform similarity search using the original query
    try:
        results = vectorstore.similarity_search(query_lower, k=k)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return ["🌱 There was an error processing your request. Please try again later."]
    
    # Debugging: Output the results of the similarity search
    print(f"Similarity search results: {results}")
    
    if not results:
        print("No results found in similarity search.")
        return ["🌱 We don’t have a remedy for this disease yet, but the search for Ayurvedic solutions is still underway."]
    
    return [r.page_content for r in results]


# Main execution
if __name__ == "__main__":
    # Load the data
    df = load_data()
    
    # Create vectorstore (FAISS index)
    vectorstore = create_vector_index(df)
    
    # Example usage of the get_recommendation function
    query = "i have cold and cough"  # Example query
    recommendations = get_recommendation(query, vectorstore, k=3)
    
    # Output recommendations
    for recommendation in recommendations:
        print(recommendation)
