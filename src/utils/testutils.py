from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.tree import export_text

def full_report(y_true, y_pred):
    return classification_report(y_true, y_pred, zero_division=1)

def print_tree_details(clf):
    """Prints tree details: depth, total nodes, leaf nodes, and features used."""
    print("-----TREE DETAILS-----")
    print(f"Tree Depth: {clf.get_depth()}")
    print(f"Total Nodes: {clf.tree_.node_count}")
    print(f"Leaf Nodes: {sum(clf.tree_.children_left == -1)}")
    print(f"Number of Features Used: {clf.tree_.n_features}")

def metric_results(y_true, y_pred, y_score, metric_lst):
    """
    Return the scores of the classification results by metrics listed
    Params:
        y_true: ground truth
        y_pred: predicted class
        y_scores: model-assigned score (logits)
    """
    if "full" in metric_lst:
        return full_report(y_true, y_pred)
    
    score_lst = []
    for metric in metric_lst:
        if metric == "auc_score":
            score = roc_auc_score(y_true, y_score, multi_class="ovr")
        else:
            score = eval(f"{metric}(y_true, y_pred)") 
        score_lst.append(score)
    return score_lst
    