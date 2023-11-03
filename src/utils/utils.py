import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from src.utils.preprocess_utils import cleaning_text


def calc_sim(model: Doc2Vec, query: str) -> list[int]:
    clean_query = cleaning_text(query)
    vector_query = model.infer_vector([clean_query][0])
    similar_sentences = model.dv.most_similar(positive=[vector_query])
    return [int(similar_sentences[i][0]) for i in range(0, 5)]


def extract_simular_wine(data: pd.DataFrame, sim_idx: list) -> list[dict[str, any]]:
    result = []
    for idx in sim_idx:
        wine_data = {
            "title": data['title'][idx],
            "points": data['points'][idx],
            "price": int(data['price'][idx]) if not pd.isna(data['price'][idx]) else 0,
            "description": data['description'][idx]
        }
        result.append(wine_data)
    return result
