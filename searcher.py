from matcher import NameMatcher
import numpy as np
from utils import clean_name
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer





def CalcSubIndicies(length,min_len=1,max_len=np.inf):
    #global M
    A,B = np.arange(length+1), np.arange(length+1)
    M = []
    for a in A:
        for b in B:
            M.append([a,b])
    M = np.array(M)
    difs = M[:,1]-M[:,0] 
    tf = (difs>=min_len) & (difs<=max_len) 
    return M[tf] 


class SEARCHER():
    def __init__(self, model, TOP_QUERIES, EMBEDDINGS, BRANDS,  k = 10, n_filter = 250 ) -> None:
        self.model = model
        self.TOP_QUERIES = TOP_QUERIES
        self.EMBEDDINGS = EMBEDDINGS
        self.embeddings_columns = [f'V{i+1}' for i in range(15)]

        self.k = k
        self.brands = BRANDS
        self.n_filter = n_filter

        IBatch = {0:[]}
        for length in range(1,100):
            IBatch[length] = CalcSubIndicies(length,1,15)

        self.IBatch = IBatch

        #Query matcher
        query_matcher = NameMatcher(IBatch) 
        query_matcher.register(TOP_QUERIES['RAW_KW'].head(5000).to_list()) 
        self.query_matcher = query_matcher

    def getCosSim(self, vec_target,vec_matrix):
        return self.model.wv.cosine_similarities(vec_target,vec_matrix) 
    
    def search (self, raw_query, brand, topn = 10):
        raw_query = self.query_matcher.Clean2(raw_query)
        r = self.query_matcher.match(raw_query) 
        n_matches=2

        TopKWDs = []
        if r[0][0]==1.0:
            top_kwd = (r[0][1]).lower()
            TopKWDs = [top_kwd]  
        else:
            TopKWDs = [a[1] for a in r[:n_matches]] 


        TopKWDs = ["Q_"+kwd for kwd in TopKWDs]
        query = clean_name(raw_query)

        print(f'RAW QUERY: {raw_query}')
        print(f'COMPUTED QUERY: {query}')
        print(f'TopKWDS: {TopKWDs}')

        # Run word 2 vec
        preds = self.model.predict_output_word(context_words_list=TopKWDs, topn=self.k)
        if preds is None: return {"results":[], "error":"Not results for {raw_query}", "raw_query": raw_query, "computed_query":query, "top_kwds":TopKWDs}
        products = [int(prod) for prod, value in preds if len(prod)>1 or 'Q_' not in prod ]
        if products == []: return {"results":[], "error":"Not results for {raw_query}", "raw_query": raw_query, "computed_query":query, "top_kwds":TopKWDs}

        #EMBEDDINGS FOR MODEL RESULTS
        temp = self.EMBEDDINGS.loc[self.EMBEDDINGS['PRODUCT_ID'].isin(products)]
        print(temp['PRODUCT_NAME'].head(5))
        w_embeddings = self.EMBEDDINGS.loc[self.EMBEDDINGS['PRODUCT_ID'].isin(products), self.embeddings_columns].to_numpy()
        w_embeddings = w_embeddings.mean(axis=0)

        # FILTER BY BRAND
        if brand is None:
            products = self.EMBEDDINGS.copy()
        else:
            products = self.EMBEDDINGS.loc[self.EMBEDDINGS['BRAND']==brand].reset_index(drop=True)
        #Products Embeddings
        products_embeddings = products[self.embeddings_columns].to_numpy()
        products_embeddings.shape

        #COS SIMILARITY
        
        sims = self.getCosSim(w_embeddings,products_embeddings)
        products['COS_SIM'] = sims

        #Filter worst cosime similarities
        indx = np.argsort(sims)[::-1] 

        if brand is None:
            n_filter = 3000
        else:
            n_filter = self.n_filter

        top_indx = indx[:n_filter] 

        results = products.loc[top_indx]

        #NAME SIMILARITY

        vectorizer = CountVectorizer(analyzer = 'char_wb',ngram_range=(2, 2)) #bi-gram
        q = vectorizer.fit_transform([query])
        # vectorizer.get_feature_names_out()
        X = vectorizer.transform(results['PRODUCT_NAME'])
        results['TEXT_SIM']=cosine_similarity(q, X).T[:,0]
        # ATC 

        # results['ATC_MO_RATE'] = results['ATC_COUNT'].divide(results['ATC_COUNT'].mean())
        results['ATC_MO_RATE'] = results['ATC_COUNT'].apply(np.log2)
        results['IS_FAVORITE'] = 0

        #RANKER
        results['VEC_SIM_EFFECT']   = (0.02 + results['COS_SIM'].apply(lambda x: max(0, x))) **1.5  
        results['TEXT_SIM_EFFECT']  = (1.00 + results['TEXT_SIM'].apply(lambda x: x if x>=.8 else 0)) **1.0     
        results['POPULAR_EFFECT']   = (1.00 + (results['ATC_MO_RATE'])/20) **0.5 
        results['FAVORITE_EFFECT']  = (0.50 + results['IS_FAVORITE']) **1.0   
        
        results['SEARCH_SCORE'] = results['VEC_SIM_EFFECT'] * results['TEXT_SIM_EFFECT'] * results['POPULAR_EFFECT'] * results['FAVORITE_EFFECT']

        # results['SEARCH_SCORE'] = results['COS_SIM'] + results['TEXT_SIM'].apply(lambda x: x if x>=.8 else 0) + (results['ATC_COUNT'].apply(np.log2)/20)

        #FINAL
        final = results [['BRAND','PRODUCT_ID','MASTER_PRODUCT_ID','PRODUCT_NAME','L3_CATEGORY_NAME', 'IMAGE_URL',
         "ATC_COUNT",'TEXT_SIM','IS_FAVORITE',"SEARCH_SCORE", 
         ]].sort_values(by='SEARCH_SCORE', ascending=False)
        
        if brand is None: final = final.drop_duplicates(subset='MASTER_PRODUCT_ID')

        final = final.astype(str)

        print(final['PRODUCT_NAME'].head(5))
        return {"results": final.head(topn).to_dict('records'), "error":None, "raw_query": raw_query, "computed_query":query, "top_kwds":TopKWDs}
    