import numpy as np
import pandas as pd

from model import RandomForest
from score import accuracy,macro_f1,to_matrix,confusion

import joblib
from pathlib import Path
import json

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
    print("starting training....")
    clf_behav.fit(X_train,y_behav_train)
    time_behav = time.time() - start1
    print("time taken for behaviour classification = ",time_behav)


    clf_cir=RandomForest(n_trees=args.n_trees,max_depth=args.max_depth,min_samples_split=args.min_samples_split,n_feature=args.n_feats)
    start2=time.time()
    clf_cir.fit(X_train,y_cir_train)
    time_cir = time.time() - start2
    print("time taken for circuit classification = ",time_cir)

    behav_model_path = (out_dir /f"rf_behaviour-t{args.n_trees}-d{args.max_depth}.pkl")

    cir_model_path = (out_dir /f"rf_circuit-t{args.n_trees}-d{args.max_depth}.pkl")

    joblib.dump(clf_behav, behav_model_path)
    joblib.dump(clf_cir, cir_model_path)



    behav_pred = clf_behav.predict(X_test)
    cir_pred = clf_cir.predict(X_test)

    behav_accuracy = accuracy(y_behav_test, behav_pred)
    behav_f1 = macro_f1(y_behav_test, behav_pred, y_behav_test.nunique())
    behav_cm = confusion(y_behav_test, behav_pred, BEHAV_CLASS)

    cir_accuracy = accuracy(y_cir_test, cir_pred)
    cir_f1 = macro_f1(y_cir_test, cir_pred, y_cir_test.nunique())
    cir_cm = confusion(y_cir_test, cir_pred, CIR_CLASS)


    
    print("\nBehaviour Classification")
    print("Accuracy:", behav_accuracy)
    print("Macro F1:", behav_f1)
    print("Confusion Matrix:")
    print(behav_cm)

    print("\nCircuit Classification")
    print("Accuracy:", cir_accuracy)
    print("Macro F1:", cir_f1)
    print("Confusion Matrix:")
    print(cir_cm)


    
    metrics = {
        "configuration": {
            "n_trees": args.n_trees,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "n_feats": args.n_feats,
            "seed": args.seed
        },

        "behaviour": {
            "accuracy": float(behav_accuracy),
            "macro_f1": float(behav_f1),
            "confusion_matrix": np.asarray(behav_cm).tolist(),
            "classes": BEHAV_CLASS,
            "training_time_seconds": time_behav
        },

        "circuit": {
            "accuracy": float(cir_accuracy),
            "macro_f1": float(cir_f1),
            "confusion_matrix": np.asarray(cir_cm).tolist(),
            "classes": CIR_CLASS,
            "training_time_seconds": time_cir
        }
    }

    with open(out_dir / "performance.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nPerformance metrics saved to:", out_dir / "performance.json")

if __name__=="__main__":
    main()