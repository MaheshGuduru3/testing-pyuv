from typing import Literal
from pydantic import BaseModel


class State(BaseModel):
    message:str
    choice:Literal['RAG', 'WEB']




