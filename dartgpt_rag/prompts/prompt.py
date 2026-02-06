from langchain_core.prompts import PromptTemplate


class Prompt:
    def load_prompt(self):
        prompt_temp =  "Hi, here im giving {context} and {question} give me a answer."
        chat_prompt = PromptTemplate(template=prompt_temp)
        return chat_prompt

