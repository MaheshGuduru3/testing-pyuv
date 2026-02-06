from langchain_community.vectorstores import FAISS

class VectorStore:
    def __init__(self):
        pass
    def store_data_with_emdeddings(self , docs_chunks, embeddings):
        retrieve = FAISS.from_documents(docs_chunks, embeddings)
        return retrieve.as_retriever()

    def retriever_data(self, retrieverdata):
        return retrieverdata
