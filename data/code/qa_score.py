import numpy as np
import scipy.sparse as sp
from collections import defaultdict, OrderedDict
import sys, re, cPickle, random, logging, argparse
from sklearn import linear_model

from process_data import WordVecs
from qa_cnn import make_cnn_data, train_qacnn

logger = logging.getLogger("qa.sent.score")

def tuning_cnn(revs, wordvecs, max_l=40, dev_refname=None, test_refname=None, dev_ofname=None, test_ofname=None, cnn_cnt=False):
    """
    training CNN representations for questions and answer sentences
    """
    # tuning parameters
    n_epoch = 5
    n_feature_maps = 50
    filter_hs = [2]
    filter_h = max(filter_hs)
    lam = 0.0

    datasets = make_cnn_data(revs, wordvecs.word_idx_map, max_l=max_l, filter_h=filter_h)
    train_preds_epos, dev_preds_epos, test_preds_epos = train_qacnn(datasets, U=wordvecs.W, filter_hs=filter_hs, hidden_units=[n_feature_maps,2], shuffle_batch=False, n_epochs=n_epoch, lam=lam, batch_size=20, lr_decay = 0.95, sqr_norm_lim=9)

    results, trigger_results = [], []
    best_dev_map = .0
    best_dev_preds, best_test_preds = defaultdict(list), defaultdict(list)
    for i, (dev_preds, test_preds) in enumerate(zip(dev_preds_epos, test_preds_epos)):
        dev_map, dev_mrr = calc_mean_avg_prec(dev_preds), calc_mean_reciprocal_rank(dev_preds)
        test_map, test_mrr = calc_mean_avg_prec(test_preds), calc_mean_reciprocal_rank(test_preds)
        result = ("epo: %d" %(i+1), dev_map, dev_mrr, test_map, test_mrr)
        results.append(result)
        for thre in range(5, 15):
            thre = thre / 100.0
            dev_prec, dev_reca, dev_f1 = calc_trigger_fscore(dev_preds, thre=thre)
            test_prec, test_reca, test_f1 = calc_trigger_fscore(test_preds, thre=thre)
            result = ("epo: %d, thre=%.2f" %(i+1, thre), dev_prec, dev_reca, dev_f1, test_prec, test_reca, test_f1)
            trigger_results.append(result)
            if dev_map > best_dev_map:
                best_dev_preds = dev_preds
                best_test_preds = test_preds
                best_dev_map = dev_map

    sorted_results = sorted(results, key=lambda res: res[1], reverse=True) # according to dev MAP
    for result in sorted_results[:5]:
        print "%s\tdev_MAP=%.4f, dev_MRR=%.4f, test_MAP=%.4f, test_MRR=%.4f" %result

    sorted_trigger_results = sorted(trigger_results, key=lambda res: res[3], reverse=True) # according to dev F1
    for result in sorted_trigger_results[:5]:
        print "%s\tdev_precision=%.4f, dev_recall=%.4f, dev_F1=%.4f, test_precision=%.4f, test_recall=%.4f, test_F1=%.4f" %result

    if not cnn_cnt:
        if dev_refname is not None and dev_ofname is not None:
            create_pred(best_dev_preds, dev_refname, dev_ofname)
        if test_refname is not None and test_ofname is not None:
            create_pred(best_test_preds, test_refname, test_ofname)
        return

    # logistic regression
    results, trigger_results = [], []
    best_dev_map = .0
    best_dev_preds, best_test_preds = defaultdict(list), defaultdict(list)
    for i, (train_preds, dev_preds, test_preds) in enumerate(zip(train_preds_epos, dev_preds_epos, test_preds_epos)):
        datasets = make_lr_data(revs, train_preds, dev_preds, test_preds)
        new_dev_preds, new_test_preds = train_logistic_regression(datasets)
        dev_map, dev_mrr = calc_mean_avg_prec(new_dev_preds), calc_mean_reciprocal_rank(new_dev_preds)
        test_map, test_mrr = calc_mean_avg_prec(new_test_preds), calc_mean_reciprocal_rank(new_test_preds)
        result = ("epo: %d" %(i+1), dev_map, dev_mrr, test_map, test_mrr)
        results.append(result)
        if dev_map > best_dev_map:
            best_dev_preds = new_dev_preds
            best_test_preds = new_test_preds
            best_dev_map = dev_map
        for thre in range(5, 15):
            thre = thre / 100.0
            dev_prec, dev_reca, dev_f1 = calc_trigger_fscore(new_dev_preds, thre=thre)
            test_prec, test_reca, test_f1 = calc_trigger_fscore(new_test_preds, thre=thre)
            result = ("epo: %d, thre=%.2f" %(i+1, thre), dev_prec, dev_reca, dev_f1, test_prec, test_reca, test_f1)
            trigger_results.append(result)

    sorted_results = sorted(results, key=lambda res: res[1], reverse=True) # according to dev MAP
    for result in sorted_results[:5]:
        print "Logistic Regression -> %s\tdev_MAP=%.4f, dev_MRR=%.4f, test_MAP=%.4f, test_MRR=%.4f" %result
    sorted_trigger_results = sorted(trigger_results, key=lambda res: res[3], reverse=True) # according to dev F1
    for result in sorted_trigger_results[:5]:
        print "Logistic Regression -> %s\tdev_precision=%.4f, dev_recall=%.4f, dev_F1=%.4f, test_precision=%.4f, test_recall=%.4f, test_F1=%.4f" %result

    if dev_refname is not None and dev_ofname is not None:
        create_pred(best_dev_preds, dev_refname, dev_ofname)
    if test_refname is not None and test_ofname is not None:
        create_pred(best_test_preds, test_refname, test_ofname)


def train_logistic_regression(datasets, reg=0.01):
    """
    train logistic regression model
    """
    train, dev, test, train_y, dev_y, test_y, train_id, dev_id, test_id = datasets
    clf = linear_model.LogisticRegression(C=reg, solver='lbfgs')
    clf = clf.fit(train, train_y)
    dev_preds, test_preds = defaultdict(list), defaultdict(list)
    ypred = clf.predict_proba(dev)
    for i, pr in enumerate(ypred):
        qid, aid, y = dev_id[i][0], dev_id[i][1], dev_y[i]
        dev_preds[qid].append((aid, y, pr[1]))
    ypred = clf.predict_proba(test)
    for i, pr in enumerate(ypred):
        qid, aid, y = test_id[i][0], test_id[i][1], test_y[i]
        test_preds[qid].append((aid, y, pr[1]))
    return dev_preds, test_preds
    

def make_lr_data(revs, train_preds, dev_preds, test_preds):
    """
    concatenate features of revs and preds
    """
    feat_dict = defaultdict(list)
    for rev in revs:
        split, qid, aid = rev["split"], rev["qid"], rev["aid"]
        feat_dict[(split, qid, aid)] = rev["features"]
    train, dev, test = [], [], []
    train_y, dev_y, test_y = [], [], []
    train_id, dev_id, test_id = [], [], []
    for qid in train_preds:
        for atri in train_preds[qid]:
            feat = feat_dict[(1, qid, atri[0])]
            feat.append(atri[2])
            train.append(feat)
            train_y.append(atri[1])
            train_id.append((qid, atri[0]))
    for qid in dev_preds:
        for atri in dev_preds[qid]:
            feat = feat_dict[(2, qid, atri[0])]
            feat.append(atri[2])
            dev.append(feat)
            dev_y.append(atri[1])
            dev_id.append((qid, atri[0]))
    for qid in test_preds:
        for atri in test_preds[qid]:
            feat = feat_dict[(3, qid, atri[0])]
            feat.append(atri[2])
            test.append(feat)
            test_y.append(atri[1])
            test_id.append((qid, atri[0]))
    train = np.array(train,dtype="float")
    dev = np.array(dev,dtype="float")
    test = np.array(test,dtype="float")
    train_y = np.array(train_y,dtype="int")
    dev_y = np.array(dev_y,dtype="int")
    test_y = np.array(test_y,dtype="int")
    return [train, dev, test, train_y, dev_y, test_y, train_id, dev_id, test_id]

def calc_mean_avg_prec(preds):
    """
    skip all questions w/o correct answers
    and all questions w/ only correct answers
    """
    mean_avg_prec, relQ = 0.0, 0.0
    for pred in preds.values():
        cnt = 0
        for tri in pred: cnt += tri[1]
        if cnt == 0 or cnt == len(pred): continue
        sorted_pred = sorted(pred, key=lambda res: res[1])
        sorted_pred = sorted(sorted_pred, key=lambda res: res[2], reverse=True)
        avg_prec, rel = 0.0, 0.0
        for i, tri in enumerate(sorted_pred):
            if tri[1] == 1:
                rel += 1.0
                avg_prec += rel / (i + 1)
        avg_prec /= rel
        mean_avg_prec += avg_prec
        relQ += 1.0
    mean_avg_prec /= relQ
    return mean_avg_prec

def calc_mean_reciprocal_rank(preds):
    """
    skip all questions w/o correct answers
    and all questions w/ only correct answers
    """
    mean_reciprocal_rank, relQ = 0.0, 0.0
    for pred in preds.values():
        cnt = 0
        for tri in pred: cnt += tri[1]
        if cnt == 0 or cnt == len(pred): continue
        sorted_pred = sorted(pred, key=lambda res: res[1])
        sorted_pred = sorted(sorted_pred, key=lambda res: res[2], reverse=True)
        reciprocal_rank, rel = 0.0, 0.0
        for i, tri in enumerate(sorted_pred):
            if tri[1] == 1:
                rel += 1.0
                reciprocal_rank += rel / (i + 1)
                break
        relQ += 1.0
        mean_reciprocal_rank += reciprocal_rank
    mean_reciprocal_rank /= relQ
    return mean_reciprocal_rank

def calc_trigger_fscore(preds, thre=0.1):
    """
    precision, recall, fmeasure for the task of answering triggering
    """
    gt_cnt, pred_cnt, match_cnt = 0.0, 0.0, 0.0
    for pred in preds.values():
        sorted_pred = sorted(pred, key=lambda res: res[2], reverse=True)
        if sorted_pred[0][2] > thre: 
            pred_cnt += 1.0
            if sorted_pred[0][1] == 1:
                match_cnt += 1.0
        sorted_gt = sorted(pred, key=lambda res: res[1], reverse=True)
        if sorted_gt[0][1] == 1:
            gt_cnt += 1.0
    prec, reca = match_cnt / pred_cnt, match_cnt / gt_cnt
    return prec, reca, 2*prec*reca / (prec+reca)


def create_pred(preds, alfname, ofname, qcol=0, acol=2):
    """
    create prediction file for trec eval
    alfname: input file for alignment, containing quetion id column and answer id column
    """
    pscrs = []
    for i in xrange(len(preds)):
        lst = preds[i+1]
        for tri in lst:
            pscrs.append(tri[2])
    f = open(alfname, "rb")
    allines = f.readlines()
    f.close()

    of = open(ofname, "w")
    for alline, pscr in zip(allines, pscrs):
        parts = alline.strip().split()
        qid, aid = parts[qcol], parts[acol]
        of.write("%s 0 %s 0 %s 0\n"%(qid, aid, pscr))
    of.close()


if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="pkl file for dataset")
    parser.add_argument("--dev_refname", help="reference fname for dev set")
    parser.add_argument("--test_refname", help="reference fname for test set")
    parser.add_argument("--dev_ofname", help="output prediction for dev set")
    parser.add_argument("--test_ofname", help="output prediction for test set")
    parser.add_argument("--cnn_cnt", type=int, default=0, help="CNN-Cnt model or not, default is 0")
    args = parser.parse_args()
    
    print "loading data...",
    x = cPickle.load(open(args.dataset,"rb"))
    revs, wordvecs, max_l = x[0], x[1], x[2]
    max_l = 40
    print "data loaded!"
    if args.cnn_cnt == 0:
        cnn_cnt = False
    else:
        cnn_cnt = True
    tuning_cnn(revs, wordvecs, max_l, args.dev_refname, args.test_refname, args.dev_ofname, args.test_ofname, cnn_cnt)
    
    logger.info('end logging')
