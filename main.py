from typing import Optional
from sim_utils.SimTFIDF import SimTFIDF
from fastapi import FastAPI

app = FastAPI()

sim = SimTFIDF(candidate_path='data/零件名.txt', queryset_path='data/车名.txt')


@app.get("/query/{query}")
def get_sim_query(query: str, q: Optional[str] = None, topk: int = 5, similarity_min = 0.6):
    return {"query": query, "q": q, "result":sim.search_query(query, top_k=topk, similarity_min=float(similarity_min))}
    