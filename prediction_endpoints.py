from fastapi import FastAPI, Depends
from pydantic import BaseModel
import contextlib
from gensim.models.doc2vec import Doc2Vec
from src.utils.summarize_utils import get_summary
from src.utils.utils import calc_sim, extract_simular_wine
import pandas as pd
from collections.abc import Sequence


class TextInput(BaseModel):
    text: str


class SummaryOutput(BaseModel):
    summary: str


class SimilarOutput(BaseModel):
    title: str
    points: int
    price: int
    description: str


class NLPModel:
    def __init__(self):
        self.model = None
        self.data = None

    def load_model(self):
        model_file = "model/doc2vec_model"
        model = Doc2Vec.load(model_file)
        self.model = model

    def load_data(self):
        data_file = "data/wine_reviews.csv"
        data = pd.read_csv(data_file, usecols=['title', 'points', 'price', 'description'], encoding='utf-8')
        self.data = data

    def summary(self, input: TextInput) -> SummaryOutput:
        summary_text = get_summary(input.text)
        return SummaryOutput(summary=summary_text)

    def similar(self, input: TextInput) -> Sequence[SimilarOutput]:
        similar_wines = calc_sim(self.model, input.text)
        result = extract_simular_wine(self.data, similar_wines)
        return [SimilarOutput(**wine) for wine in result]


nlp_model = NLPModel()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    nlp_model.load_model()
    nlp_model.load_data()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/summary", response_model=SummaryOutput)
async def get_summary_wine(input: TextInput) -> SummaryOutput:
    output = nlp_model.summary(input)
    return output


@app.post("/similar", response_model=list[SimilarOutput])
async def get_similar_wine(input: TextInput) -> Sequence[SimilarOutput]:
    output = nlp_model.similar(input)
    return output
