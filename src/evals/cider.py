import copy
from collections import defaultdict
import numpy as np
import pdb
import math


class CiderScorer(object):
    def copy(self):
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0, score_mode="max"):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None
        self.score_mode = score_mode

    def cook_append(self, test, refs):
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test)) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        if type(other) is tuple:
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self

    def compute_doc_freq(self):
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self):
        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val

        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))
        if len(self.crefs) == 1:
            self.ref_len = 1
        scs = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            scores = [np.array([0.0 for _ in range(self.n)]) for _ in range(len(refs))]
            for ki, ref in enumerate(refs):
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                scores[ki] += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            scores_avg = [np.mean(score) for score in scores]
            score_avg = sum(scores_avg) / len(refs)
            score_avg *= 10.0
            score_max = max(scores_avg)
            score_max *= 10.0
            scores_all = [float(s) * 10.0 for s in scores_avg]
            # append score of an image to the score list
            if self.score_mode == "avg":
                scs.append(score_avg)
            elif self.score_mode == "max":
                scs.append(score_max)
            elif self.score_mode == "all":
                scs.append(scores_all)
        return scs

    def compute_score(self, option=None, verbose=0):
        # compute idf
        self.compute_doc_freq()
        # assert to check document frequency
        assert(len(self.ctest) >= max(self.document_frequency.values()))
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        try:
            return np.mean(np.array(score)), np.array(score)
        except:
            return 0, score


class Cider():
    def __init__(self, n=4, sigma=6.0, score_mode="max"):
        self._n = n
        self._sigma = sigma
        self._score_mode = score_mode

    def compute_score(self, info_dict, pred_name, targets_name):
        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma, score_mode=self._score_mode)
        names = sorted(list(info_dict.keys()))
        for name in names:
            info = info_dict[name]
            pred_sent = info[pred_name]
            target_sents = info[targets_name]
            cider_scorer += (pred_sent, target_sents)
        (score_avg, scores) = cider_scorer.compute_score()
        score_dict = {name: scores[i] for i, name in enumerate(names)}
        return score_avg, score_dict


def precook(s, n=4, out=False):
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    return precook(test, n, True)

