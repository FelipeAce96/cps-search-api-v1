from fastapi import FastAPI, HTTPException
from gensim.models import Word2Vec as word2vec 
import pandas as pd
from typing import Optional
from searcher import SEARCHER
import time


# LOAD ALL INPUTS


#Load Word2vec
w2v = word2vec.load('MODEL/W2V_MODEL1.w2v')
#Load Embeddings
EMBEDDINGS = pd.read_parquet('DATA/PD_EMBEDDINGS.parquet.gzip')
# READ ATC
ATC = pd.read_parquet('DATA/ATC.parquet.gzip').rename(columns={'COUNT':'ATC_COUNT'})
# TOP QUERIES
TOP_QUERIES = pd.read_parquet('DATA/TOP_QUERIES.parquet.gzip')

EMBEDDINGS = EMBEDDINGS.merge(ATC, on=['BRAND','PRODUCT_ID'], how='left')
EMBEDDINGS['ATC_COUNT'] = EMBEDDINGS['ATC_COUNT'].fillna(0)

available_brands = list(EMBEDDINGS['BRAND'].unique())


searcher = SEARCHER(
    model = w2v, 
    TOP_QUERIES=TOP_QUERIES,
    EMBEDDINGS=EMBEDDINGS,
    BRANDS = available_brands,
    k=3,
    n_filter = 300,
)





# APP & ROUTES

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/api/v1/top-queries')
async def getTopQueries():
    return {"results": TOP_QUERIES['RAW_KW'].to_list() }


@app.get('/api/v1/avilable-brands')
async def getAvailableBrands():
    return {"results": available_brands }


@app.get('/api/v1/search/general')
async def GeneralSearch(raw_query : str, topn:Optional[int] = 100):
    t0 = time.time()
    res = searcher.search(raw_query=raw_query, brand=None, topn=topn)
    t1 = time.time()
    print(f'Execution Time: {t1-t0:.3f} seconds')
    if res['error'] is not None: return HTTPException(400, res)
    return res

@app.get('/api/v1/search/brand')
async def GeneralSearch(raw_query : str, brand: str, topn:Optional[int] = 10):
    if brand not in available_brands: return  HTTPException(400, {"results":[], "error":"Not Available Brand"})
    t0 = time.time()
    res = searcher.search(raw_query=raw_query, brand=brand, topn=topn)
    t1 = time.time()
    print(f'Execution Time: {t1-t0:.3f} seconds')
    if res['error'] is not None: return HTTPException(400, res)
    return res