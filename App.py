import os
import re
import streamlit as st
import subprocess
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from llama_cpp import Llama, LlamaGrammar

# Regular expression to remove HTML tags
CLEANR = re.compile('<.*?>')

my_grammar = LlamaGrammar.from_string(r"""
#Dies ist ein grammar das ein Argdown Dokument erzeugen soll. Argdown ist eine Auszeichnungssprache die größtenteils dem Markdown-Syntax folgt

root ::= statement nl (relation statement nl)+   

statement ::= "\["[a-zA-Z ]+"\]"

relation ::= (confirmation | contradiction)

confirmation ::= "\n\t+ "

contradiction ::= "\n\t- "

nl ::= "\n"                                 
""")


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

# Function to run Argdown CLI and get DOT output
def generate_argument_map(argdown_input_path):
    try:
        # Run Argdown CLI and get the DOT output
        argdown_path = r'C:\Users\hanna\AppData\Roaming\npm\argdown.cmd'
        result = subprocess.run(
            [argdown_path, 'web-component', argdown_input_path, 'current.html'],
            text=True, check=True
        )
        # Return the DOT output
        return True
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

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
    
    if st.button("Argument Map generieren"):
        if question:
            # Get relevant documents
            documents = get_documents_from_question(question)
            
            # Load the cached LLM model
            llm = load_llm()
            
            # Extract arguments from the documents
            arguments = get_arguments_from_question(question, documents, llm)
            ArgdownBeispiel = open(r'C:\Academic\RAGPlayground\ArgdownSandboxBeispiel.argdown', 'r').read()
            prompt = f"Argdown ist eine Auszeichnungssprache die größtenteils dem Markdownsyntax folgt. Hier ein Beispiel für den Argdown Syntax:{ArgdownBeispiel} Argdown ist spacing-sensitiv. Argumente in Argdown bestehen aus einem Argumenttitel in spitzen Klammern der den Inhalt des Arguments in einem kurzen Satz zusammenfasst und einzigartig sein muss, einer Leerzeile, drei nummerierten Prämissen sie sollten kurz und prägnant zusammengefasst sein, eine durch zwei dashes -- oben und unten abgregrenzte logisch gültige Schlussregel  bzw. Modus Ponens in dem Beschrieben wird warum die Prämissen die Konklusion implizieren und einer Konklusion, die ebenfalls nummeriert is und in der die Schlussfolgerung in einem Satz beschrieben wird. Statements in Argdown sind entweder Behauptungen oder Resultate von Argumenten, sie beschreiben allgemeine Aussagen die entweder unterstützt oder widerlegt werden können. Sie können als Resultat eines Arguments an den Beginn der Konlusion gesetzt werden. Bitte suche in folgendem Text Argumente und gib das Argument oder die Argumente dann im Ardownsyntax aus.{' '.join(arguments)}"

            argdown_input = llm(prompt, max_tokens=1000, temperature=0.5, grammar=my_grammar)

            with open('user.argdown', 'w') as wf:
                wf.write(str(argdown_input))           
            generate_argument_map('user.argdown') 
            with open('current.html\\user.component.html','r',) as rf:
                map = rf.read()
        
            if "Error" in map:
                st.error(map)
            else:
                # Display the argument map
                st.components.v1.html(map, width=None, height=800, scrolling=True)
        else:
            st.warning("Please enter a question.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
