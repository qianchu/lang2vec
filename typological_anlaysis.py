import pandas as pd
import lang2vec.lang2vec as l2v


class LexicalSimilarity():
    def __init__(self, lg_list):
        self.store_vocabs(lg_list)

    def store_vocabs(self, lg_list):
        vocab_all = {}
        for lg in lg_list:
            vocab_all[lg.lower()] = self.calculate_vocab(lg.lower())
        self.vocab_all = vocab_all

    def calculate_vocab(self, lg):
        fname = f'/mnt/hdd/ql261/wikicorpora/wikiextracted/{lg}/contexts_raw_orig2.wic'
        vocab = set()
        for line in open(fname).readlines()[:10000]:
            for w in line.split('\t')[0].strip().split():
                vocab.add(w)
        return vocab

    def compute_lex_similarity(self, src_lg, tgt_lg):
        src_vocab, tgt_vocab = self.vocab_all[src_lg], self.vocab_all[tgt_lg]
        common_w = src_vocab.intersection(tgt_vocab)
        all_w = src_vocab.union(tgt_vocab)
        return float(len(common_w))/(len(all_w))

if __name__ == '__main__':

    data = 'data/amico-zero-shot-bertm.csv'
    df = pd.read_csv(data, index_col=0)
    items = []
    lgs = ['DE', 'RU', 'JA', "ZH", "AR", 'FI',	'TR',	'ID', 
        	'EU',	'KA',	'BN',	'KK',	'UR', 'KO']
    lg2traindata= {'DE': 50000, 'RU': 28286, 'JA': 16142, 'ZH': 13154, 'AR': 9622}
    lexEngine = LexicalSimilarity(lg_list= lgs)

    for src_lg in ['DE','RU', 'JA', 'ZH', 'AR']:
        for tgt_lg in lgs:
            if src_lg == tgt_lg:
                continue
            print (src_lg, tgt_lg)
            gen, geo, syn = l2v.distance(["genetic", "geographic", "syntactic"], 
                    l2v.LETTER_CODES[src_lg.lower()], 
                    l2v.LETTER_CODES[tgt_lg.lower()])
            lex = lexEngine.compute_lex_similarity(src_lg.lower(), tgt_lg.lower())
            items.append({'geo':geo, 'syn':syn, 
                'gen': gen, 'datasize':lg2traindata[src_lg], 
                'score': df.loc[src_lg][tgt_lg],
                'lex': lex
                })
    df_results = pd.DataFrame(items)

    import statsmodels.api as sm
    import numpy as np
    X = np.column_stack((df_results['geo'], df_results['gen'],
    df_results['syn'], 
    df_results['lex'], 
    df_results['datasize']))
    y = df_results['score']
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())
