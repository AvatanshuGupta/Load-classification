import numpy as np
import pandas as pd

from model import RandomForest
from score import accuracy,macro_f1,to_matrix,confusion

import joblib
from pathlib import Path

import time

import argparse

CIR_CLASS=['RC', 'RL', 'LC', 'L', 'C', 'R', 'RLC']
BEHAV_CLASS=['CZ', 'CP', 'CC']



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trees", type=int, default=25)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--min-samples-split", type=int, default=10)
    ap.add_argument(
        "--n-feats",
        type=int,
        default=None,
        help="features tried per split"
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    X_train = pd.DataFrame(joblib.load("artifacts/X_train.pkl"))
    X_test = pd.DataFrame(joblib.load("artifacts/X_test.pkl"))

    y_cir_train = pd.Series(joblib.load("artifacts/y_cir_train.pkl"))
    y_cir_test = pd.Series(joblib.load("artifacts/y_cir_test.pkl"))

    y_behav_train = pd.Series(joblib.load("artifacts/y_behav_train.pkl"))
    y_behav_test = pd.Series(joblib.load("artifacts/y_behav_test.pkl"))

    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    clf_behav=RandomForest(n_trees=args.n_trees,max_depth=args.max_depth,min_samples_split=args.min_samples_split,n_feature=args.n_feats)
    start1=time.time()
    print("started")
    clf_behav.fit(X_train,y_behav_train)
    print("time taken for behaviour classification = ",time.time()-start1)


    clf_cir=RandomForest(n_trees=args.n_trees,max_depth=args.max_depth,min_samples_split=args.min_samples_split,n_feature=args.n_feats)
    start2=time.time()
    clf_cir.fit(X_train,y_cir_train)
    print("time taken for circuit classification = ",time.time()-start2)



    behav_pred = clf_behav.predict(X_test)
    cir_pred = clf_cir.predict(X_test)

    print("Behaviour  classification")
    print("accuracy:", accuracy(y_behav_test, behav_pred))
    print("Macro F1:", macro_f1(y_behav_test, behav_pred,y_behav_test.nunique()))
    print("Confusion matrix:")
    print(confusion(y_behav_test, behav_pred,BEHAV_CLASS))

    print("circuit Classification")
    print("Accuracy:", accuracy(y_cir_test, cir_pred))
    print("Macro f1:", macro_f1(y_cir_test, cir_pred,y_cir_test.nunique()))
    print("confusion Matrix:")
    print(confusion(y_cir_test, cir_pred,CIR_CLASS))

if __name__=="__main__":
    main()