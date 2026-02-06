from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class DataIngestion:
    def __init__(self):
        print("self.")

    def load_docs(self):
        try:
          base_dir = os.path.dirname(os.path.abspath(__file__))
          pdf_path = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "data"))
          docs_load = DirectoryLoader(
                      path=pdf_path,          
                      glob="**/*.pdf",        
                      loader_cls=PyPDFLoader
                  )
          res = docs_load.load()
          return res
        except Exception as e:
            print(f"Failed to load documents: {e}")
            return None

    def loaded_docs_chunks(self, doc_chunks):
        try:
          docs_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(doc_chunks)
          return docs_split
        except Exception as e:
            print(f"Failed to loaded docs: {e}")
            return None