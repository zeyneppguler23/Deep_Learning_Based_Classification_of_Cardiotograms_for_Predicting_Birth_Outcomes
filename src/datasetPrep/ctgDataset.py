''' 
Step 1: preprocessing.py that can cut signals intostep1-step4 windows. 
Step 2: labelling.ipynb that can calculate majority vote labels for each step.
Step 3: ctgDataset.py that can create a Dataset class to load the preprocessed data and their corresponding labels for training a model.

Take last 30 minutes of second stage if available, else pad/truncate to 1800
This makes it plug-and-play with current architecture.

Load labels per step: labels_step4 = load_step_labels("../ExpertAnnotations/labels_step4.csv")
'''



# datasets/ctg_dataset.py

import numpy as np
import pandas as pd

UNINTERPRETABLE = "Uninterpretable (Filtered)"

MAP_3CLASS = {
    "No Hypoxia (Normal)": 0,
    "Mild Hypoxia (Suspicious)": 1,
    "Severe Hypoxia (Pathological)": 2,
}

def load_labels_csv(path):
    df = pd.read_csv(path)
    df["rec_id"] = df["rec_id"].astype(str)

    # assumes your CSV has Clinical_Label column
    df["y_interp"] = (df["Clinical_Label"] == UNINTERPRETABLE).astype(int)

    def map_state(lbl):
        if lbl == UNINTERPRETABLE:
            return -1
        return MAP_3CLASS.get(lbl, -1)

    df["y_state"] = df["Clinical_Label"].apply(map_state).astype(int)
    return df[["rec_id", "y_state", "y_interp"]]


def align_step(outputs, step_name, labels_df):
    rec_ids = [str(r) for r in outputs["rec_ids"]]
    X_list = outputs[step_name]
    valid = outputs["valid_mask"][step_name]

    lab = labels_df.set_index("rec_id")

    X, y_state, y_interp, kept = [], [], [], []
    for rid, x, ok in zip(rec_ids, X_list, valid):
        if not ok or x is None:
            continue
        if rid not in lab.index:
            continue

        X.append(x)
        y_state.append(int(lab.loc[rid, "y_state"]))
        y_interp.append(int(lab.loc[rid, "y_interp"]))
        kept.append(rid)

    return (
        np.stack(X).astype(np.float32),
        np.array(y_state, dtype=np.int32),
        np.array(y_interp, dtype=np.int32),
        kept
    )
