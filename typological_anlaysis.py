import pandas as pd
import lang2vec.lang2vec as l2v


class LexicalSimilarity():
    def __init__(self, lg_list):
        self.store_vocabs(lg_list)

    def store_vocabs(self, lg_list):
        vocab_all = {}
        for lg in lg_list:
            vocab_all[lg] = self.calculate_vocab(lg)
        self.vocab_all = vocab_all

    def calculate_vocab(self, lg, vocab):
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
    Y = []
    Geo = []
    Genetics = []
    Syntax = []
    Lex = []
    lgs = ['DE', 'RU', 'JA', "ZH", "AR", 'FI',	'TR',	'ID', 
        	'EU',	'KA',	'BN',	'KK',	'UR']
    lexEngine = LexicalSimilarity(lg_list= lgs)

    for src_lg in ['DE', 'RU', 'JA', "ZH", 'AR']:
        for tgt_lg in lgs:
            print (src_lg, tgt_lg)
            if src_lg == tgt_lg:
                continue
            Y.append(df.loc[src_lg][tgt_lg])
            gen, geo, syn = l2v.distance(["genetic", "geographic", "syntactic"], 
                    l2v.LETTER_CODES[src_lg.lower()], 
                    l2v.LETTER_CODES[tgt_lg.lower()])
            lex = lexEngine.compute_lex_similarity(src_lg.lower(), tgt_lg.lower())
            Geo.append(geo)
            Syntax.append(syn)
            Genetics.append(gen)
            Lex.append(lex)
    print (Lex)
     
