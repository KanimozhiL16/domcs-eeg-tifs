import numpy as np
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.cluster import KMeans


def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = sk_auc(fpr, tpr)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2 * 100
    return eer, roc_auc


def build_prototypes(E_train, Y_train, k=3):
    protos = {}
    for s in np.unique(Y_train):
        E = E_train[Y_train == s]
        k_use = min(k, len(E))
        km = KMeans(n_clusters=k_use, random_state=0, n_init=10).fit(E)
        p = km.cluster_centers_
        protos[s] = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    return protos


def evaluate_b2t(E_train, Y_train, E_test, Y_test, k=3):
    E_train = E_train/(np.linalg.norm(E_train,axis=1,keepdims=True)+1e-8)
    E_test  = E_test /(np.linalg.norm(E_test, axis=1,keepdims=True)+1e-8)
    protos  = build_prototypes(E_train, Y_train, k)
    subjs   = sorted(np.unique(Y_train))
    gallery = np.vstack([protos[s] for s in subjs])
    sids    = np.concatenate([[s]*len(protos[s]) for s in subjs])

    correct = 0
    for i in range(0, len(E_test), 1024):
        Eb  = E_test[i:i+1024]
        sim = Eb @ gallery.T
        spj = np.zeros((len(Eb), len(subjs)))
        for j, s in enumerate(subjs):
            spj[:,j] = sim[:,sids==s].max(axis=1)
        pred = np.array([subjs[p] for p in spj.argmax(axis=1)])
        correct += (pred == Y_test[i:i+1024]).sum()
    crr = correct/len(E_test)*100

    scores, labels = [], []
    for s in subjs:
        pk = protos[s]
        for s2 in subjs:
            m = (Y_test==s2)
            if m.sum()==0: continue
            sim=(E_test[m]@pk.T).max(axis=1)
            scores.extend(sim.tolist())
            labels.extend([int(s2==s)]*m.sum())
    eer, auc = compute_eer(np.array(scores), np.array(labels))
    return {"eer": eer, "auc": auc, "crr": crr}
