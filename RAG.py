
import os
import re
from langchain_chroma import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from llama_cpp import llama_cpp
from llama_cpp import Llama

CLEANR = re.compile('<.*?>') 

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

embed_model = GPT4AllEmbeddings(model_path=r"C:\Academic\RAGPlayground\Models_gguf\all-MiniLM-L6-v2-Q5_K_M.gguf", embedding=True, show_progress_bar=True)

db = Chroma(persist_directory="Files/Vectorstore/chroma_db", embedding_function=embed_model)

def load_llm():
    llm = Llama(model_path=r"C:\Academic\RAGPlayground\Models_gguf\Hermes-3-Llama-3.1-8B.Q8_0.gguf", gpu_layers=28, n_ctx=4196, show_progress_bar=True)
    return llm

##Takes a users question to get relevant documents from a vectorstore
def getDocuments_from_question(question):
    docs = db.similarity_search("Was ist ein Privileg?", k=3)
    return docs

##Takes in a question asked by a user, documents and a loaded llm,
##and then uses argument mining to find arguments and then returns a list
def getArguments_from_question(question, documents, llm):
    arguments = []
    for i in documents:
        prompt = f"""
            Bitte schreibe keine Hinleitung oder abschließendes Kommentar. In Hinsicht auf die Frage:" {question}", 
            finde in folgendem Regest alle Argumentationen und gebe diese als Liste aus, 
            die Argumentationen in der Liste sollen jeweils aus einem aussagekräftigen Titel und ein Beschreibung der Argumentation bestehen: {cleanhtml(i.page_content)}
        """
        output = llm(prompt=prompt, max_tokens=300)
        arguments.append(output['choices'][0]['text'])
    return arguments

question = 'Warum werden Privilegien wiederufen bzw. entzogen?'
documents = getDocuments_from_question(question)
llm = load_llm()
arguments = getArguments_from_question(question, documents, llm)

def save_strings_to_files(strings_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for i, string in enumerate(strings_list):
        # Define the file name
        file_name = f"file_{i + 1}.txt"
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(string)
        print(f"Saved string to {file_path}")

save_strings_to_files(arguments, 'Files\Arguments')






