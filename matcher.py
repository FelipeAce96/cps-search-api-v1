## NAME MATCHER
import numpy as np



class NameMatcher: 
    def __init__(self, IBatch):
        self.unique = set() 
        self.IBatch = IBatch
        
    def Clean2(self, thing):
        return str(thing).replace('NaN','').upper().replace('  ',' ').strip()
    
    def SubStringList(self, word):
        return [word[r[0]:r[1]] for r in self.IBatch[len(word)]]
    
    def SubStringSet(self, word): 
        ss_set = set() 
        for r in self.IBatch[len(word)]:
            ss_set.add(word[r[0]:r[1]])
        return ss_set   
    
    def register(self,name_records):
        if type(name_records[0])==str:
            name_records = [[a] for a in name_records] 
        self.recs = {} 
        self.sets = {} 
        self.lens = {} 
        for r in name_records: 
            name = r[0]
            name2 = self.Clean2(name)[:99] 
            if not name2: continue
            self.recs[name2] = list(r) 
            self.sets[name2] = self.SubStringSet(name2)    
            self.lens[name2] = len(self.sets[name2]) 
        self.names = sorted(self.recs) 
            
    def match(self,target="Greycroft"):
        tgt = self.Clean2(target)[:99]  
        tgt_set = self.SubStringSet(tgt)  
        tgt_len = len(tgt_set) 
        
        results = []
        N = 0
        for name in self.names:
            N += 1 
            inter = len(tgt_set.intersection(self.sets[name]))
            score1 = inter/tgt_len
            score2 = inter/self.lens[name] 
            score  = (score1**0.5)*(score2**0.5)  
            results.append([score,name]) 
            
            if N%1000==0:
                results.sort(reverse=True)
                results = results[:100] 

        results.sort(reverse=True)
        results = results[:100] 
        return results