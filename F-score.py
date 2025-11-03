import numpy as np


def f_score(y_true, y_pred, beta):
    # 计算混淆矩阵的元素
    tp = np.sum((y_true == 1) & (y_pred == 1))  # 真正例
    fp = np.sum((y_true == 0) & (y_pred == 1))  # 假正例
    fn = np.sum((y_true == 1) & (y_pred == 0))  # 假反例
    tn = np.sum((y_true == 0) & (y_pred == 0))  # 真反例
    
    # 计算precision和recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 计算F-score
    beta_squared = beta ** 2
    if precision == 0 and recall == 0:
        score = 0
    else:
        score = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
    
    return round(score, 3)


if __name__ == "__main__":
    y_true = np.array(eval(input()))
    y_pred = np.array(eval(input()))
    beta = float(input())
    print(f"{f_score(y_true, y_pred, beta):.3f}")