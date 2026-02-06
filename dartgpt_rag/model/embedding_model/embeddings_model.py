from langchain_ollama import OllamaEmbeddings

    # model: "mxbai-embed-large",
    #   baseUrl: "http://127.0.0.1:11434",

class EmbeddingModel:
    def model_emded(self):
        return OllamaEmbeddings(
                     model="mxbai-embed-large",
                     base_url="http://127.0.0.1:11434"
                    )
        