
import csv
from torch import cuda
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import GPT4AllEmbeddings


embed_model = GPT4AllEmbeddings(model_path=r"C:\Academic\RAGPlayground\Models_gguf\all-MiniLM-L6-v2-Q5_K_M.gguf", embedding=True, show_progress_bar=True)


def load_csv_as_documents(file_path):
    documents = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        header = next(reader)  # Falls es eine Header-Zeile gibt, die überspringen
        for row in reader:
            content = f"{str(row[4])}"
            documents.append(Document(page_content=content, metadata={"Identifier": str({row[0]}), "URI": " http://www.regesta-imperii.de/id/"+str({row[30]})}))
    return documents


documents = load_csv_as_documents("C:\Academic\RAGPlayground\First_Draft\RI_06.csv")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_splits = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model, persist_directory="Files/Vectorstore/chroma_db")
