from dartgpt_rag.model.embedding_model.embeddings_model import EmbeddingModel

class DataEmbedding:
    def load_emdeddings(self):
            print("Load to emdeddings.")
            return EmbeddingModel().model_emded()

            