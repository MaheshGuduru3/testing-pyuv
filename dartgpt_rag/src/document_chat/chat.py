from dartgpt_rag.src.document_ingestion.ingestion import DataIngestion
from dartgpt_rag.src.docments_emdeddings.emdedding import DataEmbedding
from dartgpt_rag.vectorstore.store import VectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dartgpt_rag.prompts.prompt import Prompt
from dartgpt_rag.model.chat_model.chat_model import ChatModel

parser = StrOutputParser()

class Chat:
    def chat_with_docs(self, query):
        pdf_load = DataIngestion()
        r = pdf_load.load_docs()
        print(r)
        chunks = pdf_load.loaded_docs_chunks(r)
        emded = DataEmbedding().load_emdeddings()
        store = VectorStore()
        vec_store = store.store_data_with_emdeddings(chunks, emded)
        prompt1 = Prompt().load_prompt()
        llm_model = ChatModel().load_chat_model()
        retriever = store.retriever_data(vec_store)
        res = RunnableParallel({"context": retriever, "question": RunnablePassthrough() })
        chains = res | prompt1 | llm_model | parser
        results = chains.invoke(query)
        return results


