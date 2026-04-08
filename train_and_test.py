import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold

from data_load import load_graph_data, infer_feature_dims
from model import DyHTSAGCN


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def find_best_threshold(y_true, y_probs, max_thresh=0.9):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1 = f1[:-1]
    valid = thresholds <= max_thresh

    if not valid.any():
        print("⚠️ All thresholds are too large. Return the default of 0.5")
        return 0.5

    best_idx = np.argmax(f1[valid])
    best_thresh = thresholds[valid][best_idx]
    best_f1 = f1[valid][best_idx]
    print(f"Best threshold (F1): {best_thresh:.4f}, F1: {best_f1:.4f}")
    return best_thresh


def train_model(model, data, criterion, optimizer, num_epochs, val_mask, scheduler, args):
    best_val_auc = 0.0
    best_model_dict = None
    patience = args.early_stop_patience
    patience_counter = 0
    best_val_threshold = 0.5

    lambda_adv = args.lambda_adv
    epsilon = args.epsilon

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        h = model.extract_embeddings(data)
        h.requires_grad_()
        h.retain_grad()

        out = model.classifier_1(h).squeeze(dim=1)
        train_outputs = out[data[0].train_mask]
        train_labels = data[0].y[data[0].train_mask].float()
        loss = criterion(train_outputs, train_labels)

        grad = torch.autograd.grad(loss, h, retain_graph=True)[0]
        norm = grad.norm(p=2, dim=1, keepdim=True)
        perturbation = epsilon * grad / (norm + 1e-8)
        h_adv = h + perturbation.detach()

        out_adv = model.classifier_1(h_adv).squeeze(dim=1)
        loss_adv = criterion(out_adv[data[0].train_mask], train_labels)

        total_loss = loss + lambda_adv * loss_adv
        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            h_val = model.extract_embeddings(data)
            val_outputs = model.classifier_1(h_val).squeeze(dim=1)
            val_labels = data[0].y[val_mask].cpu().numpy()
            val_probs = torch.sigmoid(val_outputs[val_mask]).cpu().numpy()
            val_auc = roc_auc_score(val_labels, val_probs)

        print(f"\nEpoch {epoch + 1}")
        print(f"  Original Loss     : {loss.item():.4f}")
        print(f"  Disturbance Loss  : {loss_adv.item():.4f}")
        print(f"  Total Loss        : {total_loss.item():.4f}")
        print(f"  Val AUC           : {val_auc:.4f}")

        if val_auc > best_val_auc:
            print("The current model is the best and is being saved...")
            best_val_auc = val_auc
            patience_counter = 0
            best_model_dict = model.state_dict()
            best_val_threshold = find_best_threshold(val_labels, val_probs)
        else:
            print(f"No boost, EarlyStopping count: {patience_counter + 1}/{patience}")
            patience_counter += 1

        scheduler.step(val_auc)

        if patience_counter >= patience:
            print("Early Stopping!")
            break

    return best_model_dict, best_val_threshold, epsilon


def evaluate_model(model, data, best_threshold=0.5):
    model.eval()
    with torch.no_grad():
        out = model(data)
        test_outputs = out[data[0].test_mask]
        test_labels = data[0].y[data[0].test_mask].float()
        test_probs = torch.sigmoid(test_outputs).cpu().numpy()
        test_labels_np = test_labels.cpu().numpy()
        test_preds = (test_probs >= best_threshold).astype(int)

        auc = roc_auc_score(test_labels_np, test_probs)
        acc = accuracy_score(test_labels_np, test_preds)
        aupr = average_precision_score(test_labels_np, test_probs)

        tn, fp, fn, tp = confusion_matrix(test_labels_np, test_preds).ravel()
        sn = tp / (tp + fn) if tp + fn > 0 else 0
        sp = tn / (tn + fp) if tn + fp > 0 else 0
        ppv = tp / (tp + fp) if tp + fp > 0 else 0
        npv = tn / (tn + fn) if tn + fn > 0 else 0
        f1 = f1_score(test_labels_np, test_preds)
        precision = precision_score(test_labels_np, test_preds)
        recall = recall_score(test_labels_np, test_preds)
        fpr, tpr, _ = roc_curve(test_labels_np, test_probs)
        prec_curve, rec_curve, _ = precision_recall_curve(test_labels_np, test_probs)

        return (
            auc, acc, aupr, sn, sp, ppv, npv, f1, precision, recall,
            test_labels_np, test_probs, fpr, tpr, rec_curve, prec_curve
        )


def train_and_test(args):
    setup_seed(args.seed)

    args.gene_layer_input, args.sub_layer_input, args.orth_layer_input = infer_feature_dims(
        args.species, args.dataset
    )

    print(
        f"Auto inferred feature dims -> "
        f"gene: {args.gene_layer_input}, "
        f"sub: {args.sub_layer_input}, "
        f"orth: {args.orth_layer_input}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    species = args.species
    dataset = args.dataset
    k_value = args.k_value
    q_th = args.q_th
    k_folds = args.k_folds

    save_dir = f"datasets/{species}/PPI/{dataset}/data_and_model/"
    os.makedirs(save_dir, exist_ok=True)

    graph_data = load_graph_data(args)
    graph_data = [g.to(device) for g in graph_data]

    labels = graph_data[0].y.cpu().numpy()
    indices = np.arange(len(labels))
    train_idx_all, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=args.seed
    )
    train_labels_all = labels[train_idx_all]

    for g in graph_data:
        g.train_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
        g.test_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
        g.test_mask[test_idx] = True

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.seed)

    all_probs = []
    all_labels = []
    fpr_list, tpr_list, rec_list, prec_list = [], [], [], []
    aucs = []
    accs = []
    auprs = []
    sns = []
    sps = []
    ppvs = []
    npvs = []
    f1s = []
    precisions = []
    recalls = []
    best_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_idx_all, train_labels_all)):
        print(f"\n🔹 Fold {fold + 1}/{k_folds} Start training...")

        train_ids = train_idx_all[train_idx]
        val_ids = train_idx_all[val_idx]

        for g in graph_data:
            g.train_mask[:] = False
            g.train_mask[train_ids] = True

        val_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
        val_mask[val_ids] = True

        model = DyHTSAGCN(args).to(device)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([args.weight_positive]).to(device)
        )
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            verbose=True
        )

        best_model_dict, best_threshold, epsilon = train_model(
            model=model,
            data=graph_data,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.epochs_num,
            val_mask=val_mask,
            scheduler=scheduler,
            args=args
        )
        best_thresholds.append(best_threshold)

        if best_model_dict is not None:
            model.load_state_dict(best_model_dict)
            model_save_path = os.path.join(
                save_dir,
                f"best_model_fold_{fold + 1}_k={k_value}_p={q_th}.pth"
            )
            torch.save(best_model_dict, model_save_path)
            print(f"{fold + 1} Best model saved")

        auc, acc, aupr, sn, sp, ppv, npv, f1, precision, recall, lbls, probs, fpr, tpr, rec, prec = evaluate_model(
            model, graph_data, best_threshold
        )

        print(
            f"Fold {fold + 1} - "
            f"AUC: {auc:.4f}, ACC: {acc:.4f}, AUPR: {aupr:.4f}, "
            f"SN: {sn:.4f}, SP: {sp:.4f}, PPV: {ppv:.4f}, NPV: {npv:.4f}, "
            f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )

        aucs.append(auc)
        accs.append(acc)
        auprs.append(aupr)
        sns.append(sn)
        sps.append(sp)
        ppvs.append(ppv)
        npvs.append(npv)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

        all_probs.extend(probs)
        all_labels.extend(lbls)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        rec_list.append(rec)
        prec_list.append(prec)

    data_to_save = {
        "all_probs": all_probs,
        "all_labels": all_labels,
        "fpr_list": fpr_list,
        "tpr_list": tpr_list,
        "rec_list": rec_list,
        "prec_list": prec_list,
        "best_thresholds": best_thresholds,
        "args": vars(args)
    }

    with open(os.path.join(save_dir, "evaluation_results.pkl"), "wb") as f:
        pickle.dump(data_to_save, f)

    print("\nAverage indicator:")

    def report(name, values):
        print(f"{name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print(f"*** species: {species}, dataset: {dataset} ***")
    print(f"*** k_value = {k_value}, q_th = {q_th} ***")
    report("AUC", aucs)
    report("ACC", accs)
    report("AUPR", auprs)
    report("SN", sns)
    report("SP", sps)
    report("PPV", ppvs)
    report("NPV", npvs)
    report("F1", f1s)
    report("Precision", precisions)
    report("Recall", recalls)

    print(
        f"*** auto dims -> gene: {args.gene_layer_input}, "
        f"sub: {args.sub_layer_input}, orth: {args.orth_layer_input} ***"
    )
    print(
        f"*** mlp_hidden: {args.mlp_hidden}, "
        f"classifier_hidden_1: {args.classifier_hidden_1}, "
        f"classifier_hidden_2: {args.classifier_hidden_2}, "
        f"drop_classifier: {args.drop_classifier}, T = {args.T} ***"
    )
    print(
        f"*** learning_rate: {args.lr}, weight_decay: {args.weight_decay}, "
        f"drop_mlp: {args.drop_mlp}, drop_conv: {args.drop_conv}, "
        f"conv_hidden1: {args.conv_hidden1}, conv_hidden2: {args.conv_hidden2} ***"
    )
    print(f"*** epsilon: {epsilon}, lambda_adv: {args.lambda_adv} ***")


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Train and evaluate DyHTSAGCN for essential protein prediction.")
        parser.add_argument("--species", type=str, default="S.cerevisiae",
                            help="Species name, for example: yeast, coli, melanogaster.")
        parser.add_argument("--dataset", type=str, default="BIOGRID",
                            help="PPI dataset name, for example: BIOGRID or DIP.")
        parser.add_argument("--k_value", type=float, default=3.00,
                            help="Threshold coefficient used when constructing the dynamic network from gene expression.")
        parser.add_argument("--q_th", type=int, default=30,
                            help="Ortholog score threshold used when constructing the orthologous network.")
        parser.add_argument("--k_folds", type=int, default=10,
                            help="Number of folds for stratified cross-validation on the training split.")
        parser.add_argument("--epochs_num", type=int, default=70,
                            help="Maximum number of training epochs for each fold.")
        parser.add_argument("--weight_positive", type=float, default=5.59,
                            help="Positive class weight for BCEWithLogitsLoss to reduce class imbalance.")
        parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for the Adam optimizer.")
        parser.add_argument("--weight_decay", type=float, default=2e-4,
                            help="Weight decay (L2 regularization) for the optimizer.")
        parser.add_argument("--seed", type=int, default=45, help="Random seed for reproducibility.")
        parser.add_argument("--drop_mlp", type=float, default=0.2,
                            help="Dropout rate used in the MLP feature projection branch.")
        parser.add_argument("--drop_conv", type=float, default=0.2,
                            help="Dropout rate used after graph convolution layers.")
        parser.add_argument("--drop_classifier", type=float, default=0.2,
                            help="Dropout rate used in the final classifier.")
        parser.add_argument("--mlp_hidden", type=int, default=256,
                            help="Hidden dimension of the MLP used to project concatenated raw features.")
        parser.add_argument("--classifier_hidden_1", type=int, default=64,
                            help="Hidden dimension of the first classifier layer.")
        parser.add_argument("--classifier_hidden_2", type=int, default=32,
                            help="Hidden dimension of the second classifier layer.")
        parser.add_argument("--conv_hidden1", type=int, default=64,
                            help="Output dimension of the first GCN layer in each graph branch.")
        parser.add_argument("--conv_hidden2", type=int, default=32,
                            help="Output dimension of the second GCN layer in each graph branch and final branch embedding size.")
        parser.add_argument("--T", type=int, default=5,
                            help="Temporal window size used when aggregating dynamic graph snapshots.")
        parser.add_argument("--lambda_adv", type=float, default=0.1,
                            help="Weight of adversarial loss in the total training loss.")
        parser.add_argument("--epsilon", type=float, default=0.1,
                            help="Perturbation strength used in adversarial training.")
        parser.add_argument("--early_stop_patience", type=int, default=5,
                            help="Number of epochs with no validation AUC improvement before early stopping.")
        args = parser.parse_args()
        train_and_test(args)