import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.utils.multiclass import unique_labels

from superphot_plus.utils import calc_accuracy, f1_score


def plot_cm(full_df, norm, purity):
    y_pred = full_df['pred_class'].to_numpy()
    y_true = full_df['true_class'].to_numpy()
    acc = calc_accuracy(y_pred, y_true)
    f1_avg = f1_score(y_pred, y_true, class_average=True)
    recall_scr = recall_score(y_true, y_pred, average='micro')
    precision_scr = precision_score(y_true, y_pred, average='micro')
    recall_scr_a = recall_score(y_true, y_pred, labels=['SN Ia'], average='micro')
    precision_scr_a = precision_score(y_true, y_pred, labels=['SN Ia'], average='micro')
    f1 = f1_score(y_true, y_pred)
    print("recall: ", recall_scr)
    print("precision: ", precision_scr)
    print("recall_ia: ", recall_scr_a)
    print("precision_ia", precision_scr_a)
    print("f1_score: ", f1_avg)
    print("f1_mine: ", f1)

    #purity
    plt.rcParams.update({'font.size':12})
    fig, ax = plt.subplots(figsize=(7,7))
    cm_vals = confusion_matrix(y_true, y_pred, normalize=norm)
    if purity:
        title = "Purity\n"+rf"$N = {len(y_pred)}, A = {acc:.2f}, F_1 = {f1_avg:.2f}$"
    else: 
        title = "Completeness\n"+rf"$N = {len(y_pred)}, A = {acc:.2f}, F_1 = {f1_avg:.2f}$"
    classes = unique_labels(y_true, y_pred)

    _ = ax.imshow(cm_vals, interpolation="nearest", vmin=0.0, vmax=1.0, cmap="Purples")

    ax.set(
        xticks=np.arange(cm_vals.shape[1]),
        yticks=np.arange(cm_vals.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="Spectroscopic Classification",
        xlabel="Photometric Classification",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_vals.max() / 1.5

    for i in range(cm_vals.shape[0]):
        for j in range(cm_vals.shape[1]):
            class_i = classes[i]
            class_j = classes[j]
            num_in_cell = len(y_pred[(y_pred == class_j) & (y_true == class_i)])
            ax.text(
                j,
                i,
                rf"${cm_vals[i, j]:.2f}$"+ "\n" + rf"$({num_in_cell})$",
                ha="center",
                va="center",
                color="white" if cm_vals[i, j] > thresh else "black",
            )
    ax.set_xlim(-0.5, len(classes) - 0.5)
    ax.set_ylim(len(classes) - 0.5, -0.5)

    plt.show()