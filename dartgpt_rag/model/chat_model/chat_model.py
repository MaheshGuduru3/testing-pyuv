from langchain_groq import ChatGroq


class  ChatModel:
    def load_chat_model(self):
        llm_model = ChatGroq(model="llama-3.3-70b-versatile", 
                             temperature=0, 
                             api_key=""
                             )
        return llm_model