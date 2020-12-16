import numpy as np
import scipy.sparse as sp
import cPickle
from collections import defaultdict
import sys, re, os, logging, argparse, ast
import pandas as pd
from scipy import spatial

import gensim
from gensim.models.doc2vec import LabeledSentence, Doc2Vec

logger = logging.getLogger("qa.sent.selection")

def build_data(trainfname, devfname, testfname, qcol, acol, lcol, extract_feature=0):
    """
    Loads and process data.
    """
    if extract_feature > 0:
        feature_extractor = Feature(trainfname, devfname, testfname, qcol, "../data/short-stopwords.txt")

    revs = []
    vocab = defaultdict(float)

    for i, fname in enumerate([trainfname, devfname, testfname]):
        ins_idx = 0
        qid, aid = 0, 0
        pre_ques = ""
        split = i + 1
        feat_stat = defaultdict(float)
        label_cnt = {0:0., 1:0.}
        with open(fname, "rb") as f:
            for line in f:  
                parts = line.strip().split("\t")
                question = clean_str(parts[qcol])
                if question != pre_ques: 
                    qid += 1
                    aid = 0
                    pre_ques = question
                answer = clean_str(parts[acol])
                label = int(parts[lcol])
                
                feat = []
                if extract_feature == 1:
                    feat = feature_extractor.count_feature_extractor(question, answer)
                elif extract_feature == 2:
                    feat = feature_extractor.cali_feature_extractor(question, answer)

                for fidx, fval in enumerate(feat):
                    feat_stat[str(fidx)+"-"+str(label)] += fval
                label_cnt[label] += 1
                
                words = set(question.split())
                words.update(set(answer.split()))
                for word in words: 
                    vocab[word] += 1  # only conside answer for df here? 
                datum  = {"y": label, 
                          "qid": qid,
                          "aid": aid,
                          "question": question,
                          "answer": answer,
                          "features": feat,
                          "num_words_q": len(question.split()),
                          "num_words_a": len(answer.split()),
                          "split": split}
                revs.append(datum)
                aid += 1
                ins_idx += 1
        logger.info("processed dataset %d with %d question-answer pairs " %(split, ins_idx))
        for fkey in feat_stat:
            fkeyl = int(fkey.split("-")[1])
            logger.info("feature %s = %.4f" %(fkey, feat_stat[fkey] / label_cnt[fkeyl]))

    max_l = max(np.max(pd.DataFrame(revs)["num_words_q"]), np.max(pd.DataFrame(revs)["num_words_a"]))

    logger.info("vocab size: %d, max question/answer length: %d" %(len(vocab), max_l))
    return revs, vocab, max_l
    

def clean_str(string):
    """
    Tokenization/string cleaning
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


class Feature(object):
    """
    extract features for Q-A pairs
    """
    def __init__(self, trainfname, devfname, testfname, qcol, stopfname):
        stopwords = set()
        with open(stopfname, "rb") as f:
            for line in f:
                stopwords.add(line.strip().lower())
        self.stopwords = stopwords
        idf = defaultdict(float)
        n = 0
        ques_list, qtype_list = [], []
        for fname in [trainfname, devfname, testfname]:
            with open(fname, "rb") as f:
                for line in f:  
                    n += 1
                    parts = line.strip().split("\t")
                    question = clean_str(parts[qcol])
                    ques_list.append(question)
                    words = set(question.split())
                    for word in words:
                        if word in stopwords: continue
                        idf[word] += 1
        for word in idf:
            idf[word] = np.log(n / idf[word])
        self.idf = idf
        for fname in [trainfname, devfname, testfname]:
            with open(fname[:fname.rfind(".")+1] + "qtype", "rb") as f:
                for line in f:  
                    parts = line.strip().split(":")
                    qtype_list.append(parts[0])
        qtype_map, qtype_invmap = {}, {}
        for q, qt in zip(ques_list, qtype_list):
            qtype_map[q] = qt
            if qt not in qtype_invmap:
                qtype_invmap[qt] = len(qtype_invmap)
        self.qtype_map, self.qtype_invmap = qtype_map, qtype_invmap

    def count_feature_extractor(self, question, answer):
        qset, aset = set(question.split()), set(answer.split())
        count, weighted_count = 0.0, 0.0
        for word in qset:
            if word not in self.stopwords and word in aset:
                count += 1.0
                weighted_count += self.idf[word]
        return [count, weighted_count]

    def cali_feature_extractor(self, question, answer):
        feat = self.count_feature_extractor(question, answer)
        qtoks, atoks = question.split(), answer.split()
        feat.append(len(qtoks))
        feat.append(len(atoks))
        count, idf_sum = 0.0, 0.0
        for word in qtoks:
            if word not in self.stopwords:
                count += 1.0
                idf_sum += self.idf[word]
        feat.append(idf_sum / count)
        count, idf_sum = 0.0, 0.0
        for word in atoks:
            if word not in self.stopwords:
                count += 1.0
                idf_sum += self.idf[word]
        feat.append(idf_sum / count)
        qtype_vec = np.zeros(len(self.qtype_invmap))
        qtype_vec[self.qtype_invmap[self.qtype_map[question]]] = 1.0
        feat += qtype_vec.tolist()
        return feat


class WordVecs(object):
    """
    precompute embeddings for word/feature/tweet etc.
    """
    def __init__(self, fname, vocab, binary=1, has_header=False):
        if binary == 1:
            word_vecs = self.load_bin_vec(fname, vocab)
        else:
            word_vecs = self.load_txt_vec(fname, vocab, has_header)
        self.k = len(word_vecs.values()[0])
        self.add_unknown_words(word_vecs, vocab, k=self.k)
        self.W, self.word_idx_map = self.get_W(word_vecs, k=self.k)

    def get_W(self, word_vecs, k=300):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, k))            
        W[0] = np.zeros(k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def load_bin_vec(self, fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word in vocab:
                   word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                else:
                    f.read(binary_len)
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs
    
    def load_txt_vec(self, fname, vocab, has_header=False):
        """
        Loads 50x1 word vecs from sentiment word embeddings (Tang et al., 2014)
        """
        word_vecs = {}
        pos = 0
        with open(fname, "rb") as f:
            if has_header: header = f.readline()
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                   word_vecs[word] = np.asarray(map(float, parts[1:]))
                pos += 1
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs

    def add_unknown_words(self, word_vecs, vocab, min_df=1, k=300):
        """
        For words that occur in at least min_df documents, create a separate word vector.    
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                #print word
                word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
    

if __name__=="__main__":    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    parser = argparse.ArgumentParser()
    parser.add_argument("train_fname", help="train file")
    parser.add_argument("dev_fname", help="development file")
    parser.add_argument("test_fname", help="test file")
    parser.add_argument("--ques_col", type=int, default=0, help="column of question, default is 0")
    parser.add_argument("--ans_col", type=int, default=1, help="column of answer, default is 1")
    parser.add_argument("--lab_col", type=int, default=2, help="column of label, default is 2")
    parser.add_argument("--extract_feat", type=int, default=0, help="extract features or not, default is 0")
    parser.add_argument("--w2v_fname", help="path/name of pretrained word embeddings file")
    parser.add_argument("outfname", help="name of output pickle file")
    args = parser.parse_args()

    revs, vocab, max_l = build_data(args.train_fname, args.dev_fname, args.test_fname, args.ques_col, args.ans_col, args.lab_col, extract_feature=args.extract_feat)

    wordvecs = None
    if args.w2v_fname is not None: # use word embeddings for CNN
        logger.info("loading and processing pretrained word vectors")
        wordvecs = WordVecs(args.w2v_fname, vocab, binary=1, has_header=False)

    cPickle.dump([revs, wordvecs, max_l], open(args.outfname, "wb"))
    logger.info("dataset created!")
    logger.info("end logging")
    
