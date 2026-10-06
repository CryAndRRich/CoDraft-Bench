import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error

from model.models.BiEncoder import get_model_bi_encoder_baseline, BasicBiEncoderClassifier
from preprocess.data_loader import PairSiameseDataset, collate_siamese

from config import *

def get_preds_ml(model, X_test, y_test):
    test_preds = model.predict(X_test)
    return (test_preds, y_test)

def get_preds_multi(trainer, test_ds, df_test, save_logits=None, return_logits=False):
    """Predict with the multi-task model.

    create_patterns duplicates every row (one copy masks side 1, one masks side 2),
    so the logits are reshaped to (-1, 2, n_classes) and averaged. This requires the
    test order to be preserved, i.e. no shuffling.

    save_logits: path to write the averaged logits to (.npy). Saving them means the
    calibration and risk-coverage analyses can be run later without re-predicting.
    return_logits: append avg_logits to the returned tuple.
    """
    test_output = trainer.predict(test_ds)
    predictions = test_output.predictions

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    reshaped_logits = predictions.reshape(-1, 2, predictions.shape[-1])
    avg_logits = reshaped_logits.mean(axis=1)

    if len(predictions) != 2 * len(df_test):
        raise ValueError(
            f"got {len(predictions)} prediction rows for {len(df_test)} test pairs; "
            f"expected exactly twice that, since create_patterns writes two masked "
            f"copies of every pair. A shuffled or padded eval loader breaks the "
            f"pairing that the (-1, 2, n_classes) reshape relies on."
        )
    if len(avg_logits) != len(df_test):
        raise ValueError(
            f"averaged logits ({len(avg_logits)}) do not line up with df_test "
            f"({len(df_test)}); the test order was probably not preserved"
        )

    test_preds = np.argmax(avg_logits, axis=-1)
    test_true = df_test["label_score"].values

    if save_logits is not None:
        os.makedirs(os.path.dirname(save_logits) or ".", exist_ok=True)
        np.save(save_logits, avg_logits)

    if return_logits:
        return (test_preds, test_true, avg_logits)
    return (test_preds, test_true)


def build_result_df(df_test, test_true, test_preds):
    """Prediction frame that carries Pair ID when the split provides one.

    Without an identifier two runs cannot be compared pair by pair, which is why
    the first round's two ablation files could not be matched up.
    """
    out = {"label": test_true, "pred": test_preds}
    for key in ("Pair ID", "Class 1", "Class 2"):
        if key in df_test.columns:
            out[key] = df_test[key].values
    cols = [c for c in ("Pair ID", "label", "pred", "Class 1", "Class 2") if c in out]
    return pd.DataFrame(out)[cols]
def get_preds_cross_encoder(model, df_test):
    test_inputs = [
        [str(row['input_text_1']), str(row['input_text_2'])]
        for i, row in df_test.iterrows()
    ]
    test_output = model.predict(test_inputs)
    test_preds = np.argmax(test_output, axis=1)
    test_true = df_test["label_score"].values
    return (test_preds, test_true)

def get_preds_siamese(test_df, model_path,model_name, device):
    _, test_preds = _predict_probabilities(model_path,model_name, test_df, device)
    return (test_preds, test_df["label_score"])


def compute_metrics(eval_pred):
    """Validation metrics during training.

    Note these are computed per augmented row. create_patterns makes two masked copies
    of every pair, and unlike get_preds_multi this does not average them back together,
    so the validation figures sit slightly below the final test figures. Kept as-is
    because it is only used to pick the best checkpoint.
    """
    logits, labels = eval_pred

    if isinstance(labels, tuple):
        labels = labels[0]

    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)

    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    mae = mean_absolute_error(labels, preds)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "qwk": float(qwk),
        "mae": float(mae)
    }
def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def get_stats(df, fig_prefix="confusion_matrix", return_metrics=False, num_classes=5):
    """Per-class and overall metrics for a (label, pred) frame.

    fig_prefix: figure stem, so each run writes its own file instead of
    overwriting the previous one.
    return_metrics: return every metric as a dict, including the macro-F1 and the
    per-class table, which used to be printed and then discarded.
    num_classes: 5 for the ordinal levels, 2 for the binary target, so the macro
    averages run over the classes that exist.
    """
    y_true = df["label"].to_numpy()
    y_pred = df["pred"].to_numpy()

    labels = list(range(num_classes))
    label_names = (['Dissimilar (0)', 'Low (1)', 'Similar (2)', 'High (3)', 'Identical (4)']
                   if num_classes == 5 else ['Dissimilar (0)', 'Similar (1)'])
    K = len(labels)
    idx = {c: i for i, c in enumerate(labels)}

    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    per_class = []
    for i, c in enumerate(labels):
        support = cm[i, :].sum()
        precision = safe_div(TP[i], TP[i] + FP[i])
        recall = safe_div(TP[i], TP[i] + FN[i])
        f1 = safe_div(2 * precision * recall, precision + recall)
        acc = safe_div(TP[i] + TN[i], TP[i] + TN[i] + FP[i] + FN[i])
        per_class.append({
            "class": c,
            "support": int(support),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    TP_micro = TP.sum()
    FP_micro = FP.sum()
    FN_micro = FN.sum()

    micro_precision = safe_div(TP_micro, TP_micro + FP_micro)
    micro_recall = safe_div(TP_micro, TP_micro + FN_micro)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    macro_precision = np.mean([r["precision"] for r in per_class])
    macro_recall = np.mean([r["recall"] for r in per_class])
    macro_f1 = np.mean([r["f1"] for r in per_class])

    overall_acc = safe_div((y_true == y_pred).sum(), len(y_true))

    mae = np.mean(np.abs(y_true - y_pred))

    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    print("Per-class metrics (class | support | accuracy | precision | recall | f1):")
    for r in per_class:
        print(f"{r['class']:>2} | {r['support']:>6} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f}")

    print("\nOverall accuracy:", f"{overall_acc:.4f}")
    print("Micro Precision | Recall | F1:", f"{micro_precision:.4f}", f"{micro_recall:.4f}", f"{micro_f1:.4f}")
    print("Macro Precision | Recall | F1:", f"{macro_precision:.4f}", f"{macro_recall:.4f}", f"{macro_f1:.4f}")

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print("\nConfusion Matrix (Hàng = Thực tế, Cột = Dự đoán):")
    header = "    " + "".join([f" P{c:>3}" for c in labels])
    print(header)
    print("   " + "-" * len(header))
    for i, row in enumerate(cm):
        print(f"T{labels[i]} |" + "".join([f"{val:>4}" for val in row]))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)

    plt.savefig(f'{fig_prefix}.pdf', format='pdf', bbox_inches='tight')

    plt.savefig(f'{fig_prefix}.png', format='png', dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

    if return_metrics:
        err = np.abs(y_true - y_pred)
        return {
            "n": int(len(y_true)),
            "accuracy": overall_acc,
            "mae": mae,
            "qwk": qwk,
            "f1_macro": macro_f1,
            "precision_macro": macro_precision,
            "recall_macro": macro_recall,
            "f1_micro": micro_f1,
            "adjacent_acc": float((err <= 1).mean()),
            "severe_rate": float((err >= 2).mean()),
            "per_class": per_class,
            "confusion_matrix": cm,
        }
    return overall_acc, mae, qwk



def save_model(trainer, tokenizer, model_name, save_path):
    print(f"Saving: {save_path} ...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    shutil.make_archive(model_name, 'zip', save_path)
    print("Success!")

def zip_model_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Cannot find: {folder_path}")
        return
    base_name = folder_path.rstrip('/')
    
    print(f"Compressing folder: {folder_path} ...")
    
    try:
        shutil.make_archive(
            base_name=base_name,  
            format='zip',         
            root_dir=folder_path  
        )
        print(f"Sucessfully compressed: {base_name}.zip")
    except Exception as e:
        print(f"Error : {e}")

def _predict_probabilities(model_path,model_name, test_df, device):
    model = get_model_bi_encoder_baseline(
        input_model_path=model_name,
        num_classes=CONFIG_MODEL.NUM_CLASSES
    )
    
    state_dict_path = os.path.join(model_path, "siamese_state.pth")
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    tokenizer = model.encoder.tokenizer
    
    test_ds = PairSiameseDataset(test_df, tokenizer, max_len=CONFIG_MODEL.MAX_LEN) 
    
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG_MODEL.MODEL_CONFIG['siamese']['batch_size'],
        shuffle=False,
        collate_fn=collate_siamese,
    )
    
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting Baseline"):
            ids1 = batch["ids1"].to(device)
            mask1 = batch["mask1"].to(device)
            ids2 = batch["ids2"].to(device)
            mask2 = batch["mask2"].to(device)
            
            logits = model(ids1, mask1, ids2, mask2)
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    return np.array(all_probs), np.array(all_preds)