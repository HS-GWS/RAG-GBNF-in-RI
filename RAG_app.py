import os
import re
from langchain_chroma import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from llama_cpp import Llama

# Regular expression to remove HTML tags
CLEANR = re.compile('<.*?>')

def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext

# Cache models to avoid reloading
class ModelCache:
    def __init__(self):
        self.embed_model = None
        self.llm_model = None
        self.db = None

    def get_embed_model(self):
        if self.embed_model is None:
            # Load embedding model with caching
            self.embed_model = GPT4AllEmbeddings(model_path=r"C:\Academic\RAGPlayground\Models_gguf\all-MiniLM-L6-v2-Q5_K_M.gguf", embedding=True, show_progress_bar=True)
        return self.embed_model

    def get_vector_store(self):
        if self.db is None:
            embed_model = self.get_embed_model()
            # Initialize Chroma vectorstore and persist data for caching
            self.db = Chroma(persist_directory="Files/Vectorstore/chroma_db", embedding_function=embed_model)
        return self.db

    def get_llm_model(self):
        if self.llm_model is None:
            # Load the Llama model once and reuse (caching)
            self.llm_model = Llama(model_path=r"C:\Academic\RAGPlayground\Models_gguf\Hermes-3-Llama-3.1-8B.Q8_0.gguf", gpu_layers=28, n_ctx=4196, show_progress_bar=True)
        return self.llm_model

# Initialize the model cache (for caching models)
model_cache = ModelCache()

# Load LLM model
def load_llm():
    return model_cache.get_llm_model()

# Retrieve relevant documents from the vector store based on user_input
def get_documents_from_question(question):
    db = model_cache.get_vector_store()
    docs = db.similarity_search(question, k=3)
    return docs

# Process documents to extract arguments using the LLM
def get_arguments_from_question(question, documents, llm):
    arguments = []
    for i in documents:
        prompt = f"""
        Bitte schreibe keine Hinleitung oder abschließendes Kommentar. In Hinsicht auf die Frage: {question}, 
        finde in folgendem Regest alle Argumentationen und gebe diese als Liste aus, 
        die Argumentationen in der Liste sollen jeweils aus einem aussagekräftigen Titel als Überschrift und ein Beschreibung der Argumentation als Fließtext bestehen: {cleanhtml(i.page_content)}
        """
        output = llm(prompt=prompt, max_tokens=500)
        arguments.append(f"[{i.metadata['Identifier']}]({i.metadata['URI']})\n\n{output['choices'][0]['text']}")
    return arguments

# Save the extracted arguments to files
def save_strings_to_files(strings_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, string in enumerate(strings_list):
        file_name = f"file_{i + 1}.txt"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(string)
        print(f"Saved string to {file_path}")

# Main function to run the app
def main():
    question = 'Warum werden Privilegien wiederufen bzw. entzogen?'
    
    # Get relevant documents
    documents = get_documents_from_question(question)
    
    # Load the cached LLM model
    llm = load_llm()
    
    # Extract arguments from the documents
    arguments = get_arguments_from_question(question, documents, llm)
    
    # Save the arguments to files
    save_strings_to_files(arguments, 'Files/Arguments')

# Run the app
if __name__ == "__main__":
    main()
