import csv
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document



# 1. CSV-Datei einlesen und jede Zeile als Dokument behandeln
def load_csv_as_documents(file_path):
    documents = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        header = next(reader)  
        n=0
        for row in reader:
            n+=1
            content = f"Das ist der Regestentext: {row[4]}. Dazu kommentiert der Regestenmitarbeiter, der das Regest verfasst hat: {row[7]}"
            documents.append(Document(page_content=content, metadata={"source": {row[0]}}))
    return documents

# CSV-Dateipfad - Hier wird der relative Pfad zur csv-Datei eingesetzt
csv_file_path = 'RI_05.csv'

# 2. Dokumente aus der CSV-Datei laden
documents = load_csv_as_documents(csv_file_path)

# 3. Erstellen des TextSplitter und Splitten der Dokumente
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_documents = text_splitter.split_documents(documents)


# 4. FAISS Vektorstore erstellen
embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
db = FAISS.from_documents(documents, embeddings)

# 5. FAISS Vektorstore speichern
db.save_local(folder_path='RI_05_FAISS')

print("Vektorstore erfolgreich erstellt und gespeichert.")
