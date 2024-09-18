import os
import re
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from llama_cpp import Llama

# Regular expression to remove HTML tags
CLEANR = re.compile('<.*?>')

def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    print('cleaning html')
    return cleantext

def load_llm():
    llm = Llama(model_path=r"C:\Academic\RAGPlayground\Models_gguf\qwen2-7b-instruct-q2_k.gguf", n_ctx=4196)
    print('model loaded')
    return llm

# Retrieve relevant documents from the vector store based on the question
def get_documents_from_question(question):
    embed_model = GPT4AllEmbeddings(model_path=r"C:\Academic\RAGPlayground\Models_gguf\all-MiniLM-L6-v2-Q5_K_M.gguf", embedding=True, show_progress_bar=True)
    db = Chroma(persist_directory=r"C:\Academic\RAGPlayground\Second_Draft\Files\Vectorstore", embedding_function=embed_model) 
    docs = db.similarity_search(question, k=3)
    print('loaded vectorbase')
    return docs

# Process documents to extract arguments using the LLM
def get_arguments_from_question(question, documents, llm):
    arguments = []
    n = 0
    print(n)
    print(documents)
    for i in documents:
        n+=1
        prompt = f"""
        Bitte schreibe keine Hinleitung oder abschließendes Kommentar. In Hinsicht auf die Frage: {question}, 
        finde in folgendem Regest alle Argumentationen und gebe diese als Liste aus, 
        die Argumentationen in der Liste sollen jeweils aus einem aussagekräftigen Titel als Überschrift und ein Beschreibung der Argumentation als Fließtext bestehen: {cleanhtml(i.page_content)}
        """
        output = llm(prompt=prompt, temperature=0.5, max_tokens=400)
        arguments.append(f"[{str(i.metadata['Identifier']).strip()}]({str(i.metadata['URI']).strip()})\n\n{output['choices'][0]['text']}")
        print('added arguments_'+'n')
    return arguments

# Save the extracted arguments to files
def save_strings_to_files(strings_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, string in enumerate(strings_list):
        file_name = f"file_{i + 1}.txt"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(string)
        st.write(f"Saved string to {file_path}")

# Streamlit App Interface
def main():
    st.title("Argument Mining in den RI")
    
    # Input question from user
    question = st.text_input("Geben Sie eine Frage ein:", "")
    
    if st.button("Argumente generieren"):
        if question:
            # Get relevant documents
            documents = get_documents_from_question(question)
            
            # Load the cached LLM model
            llm = load_llm()
            
            # Extract arguments from the documents
            arguments = get_arguments_from_question(question, documents, llm)
            print(arguments)            
            # Display arguments in the Streamlit app
            st.write("### Gefundene Argumente:")
            for i, arg in enumerate(arguments):
                st.write(f"**Argument {i + 1}:**\n{arg}\n")
            
            # Save the arguments to files
            save_strings_to_files(arguments, 'Files/Arguments')
        else:
            st.warning("Please enter a question.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
