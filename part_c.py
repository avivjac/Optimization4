import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# ————— Numerically stable sigmoid —————
def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out

# ————— Logistic‐regression objective, gradient, Hessian —————
def logistic_loss(w, b, X, y, lam=0.0):
    z = X.dot(w) + b
    loss_terms = np.logaddexp(0, z) - y * z
    f = loss_terms.sum()
    if lam > 0:
        f += 0.5 * lam * np.dot(w, w)
    return f

def logistic_grad(w, b, X, y, lam=0.0):
    z = X.dot(w) + b
    p = sigmoid(z)
    diff = p - y
    grad_w = X.T.dot(diff)
    grad_b = diff.sum()
    if lam > 0:
        grad_w += lam * w
    return grad_w, grad_b

def logistic_hess(w, b, X, y, lam=0.0):
    n, d = X.shape
    z = X.dot(w) + b
    p = sigmoid(z)
    D = p * (1 - p)
    Xd = X * D[:, None]
    H_ww = X.T.dot(Xd)
    H_wb = X.T.dot(D)
    H_bb = D.sum()
    if lam > 0:
        H_ww += lam * np.eye(d)
    H = np.zeros((d+1, d+1), float)
    H[:d, :d] = H_ww
    H[:d,  d] = H_wb
    H[ d, :d] = H_wb
    H[ d,  d] = H_bb
    return H

# ————— Data loader for decompressed IDX files —————
class MnistDataloader:
    def __init__(self, train_img, train_lbl, test_img, test_lbl):
        self.train_img = train_img
        self.train_lbl = train_lbl
        self.test_img  = test_img
        self.test_lbl  = test_lbl

    def read_images_labels(self, img_path, lbl_path):
        with open(lbl_path,'rb') as f:
            magic, n = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"Bad magic {magic} in {lbl_path}")
            labels = np.frombuffer(f.read(n), dtype=np.uint8)
        with open(img_path,'rb') as f:
            magic, n2, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051 or n2 != n:
                raise ValueError(f"Bad header in {img_path}")
            data = np.frombuffer(f.read(rows*cols*n), dtype=np.uint8)
        images = data.reshape(n, rows*cols).astype(np.float32) / 255.0
        return images, labels

    def load_data(self):
        X_train, y_train = self.read_images_labels(self.train_img, self.train_lbl)
        X_test,  y_test  = self.read_images_labels(self.test_img,  self.test_lbl)
        return (X_train, y_train), (X_test, y_test)

# ————— Armijo backtracking & solvers —————
def backtracking_line_search(w, b, X, y, lam, gw, gb, dw, db,
                             alpha0=1.0, rho=0.5, c=1e-4):
    f0 = logistic_loss(w, b, X, y, lam)
    dg = gw.dot(dw) + gb * db
    alpha = alpha0
    while True:
        w_new = w + alpha * dw
        b_new = b + alpha * db
        if logistic_loss(w_new, b_new, X, y, lam) <= f0 + c*alpha*dg:
            return alpha
        alpha *= rho

def gradient_descent(X, y, lam=1e-3, max_iters=100, tol=1e-3):
    w = np.zeros(X.shape[1]); b = 0.0
    hist = {'loss': [], 'grad_norm': [], 'w': [], 'b': []}
    for _ in range(max_iters):
        gw, gb = logistic_grad(w, b, X, y, lam)
        gn = np.linalg.norm(np.append(gw, gb))
        hist['loss'].append(logistic_loss(w, b, X, y, lam))
        hist['grad_norm'].append(gn)
        hist['w'].append(w.copy())
        hist['b'].append(b)
        if gn < tol:
            break
        dw, db = -gw, -gb
        alpha = backtracking_line_search(w, b, X, y, lam, gw, gb, dw, db)
        w += alpha * dw;  b += alpha * db
    return w, b, hist

def newton_method(X, y, lam=1e-3, max_iters=100, tol=1e-3):
    w = np.zeros(X.shape[1]); b = 0.0
    hist = {'loss': [], 'grad_norm': [], 'w': [], 'b': []}
    for _ in range(max_iters):
        gw, gb = logistic_grad(w, b, X, y, lam)
        gn = np.linalg.norm(np.append(gw, gb))
        hist['loss'].append(logistic_loss(w, b, X, y, lam))
        hist['grad_norm'].append(gn)
        hist['w'].append(w.copy())
        hist['b'].append(b)
        if gn < tol:
            break
        H    = logistic_hess(w, b, X, y, lam)
        H   += 1e-6 * np.eye(H.shape[0])     # ← add tiny ridge for stability
        grad = np.append(gw, gb)
        delta = np.linalg.solve(H, -grad)
        dw, db = delta[:-1], delta[-1]
        alpha = backtracking_line_search(w, b, X, y, lam, gw, gb, dw, db)
        w += alpha * dw;  b += alpha * db
    return w, b, hist

# ————— Main: load data, PCA, solve 0 vs 1 & 8 vs 9, plot —————
if __name__ == "__main__":
    cwd = os.getcwd()
    ti = os.path.join(cwd, "train-images.idx3-ubyte")
    tl = os.path.join(cwd, "train-labels.idx1-ubyte")
    vi = os.path.join(cwd, "t10k-images.idx3-ubyte")
    vl = os.path.join(cwd, "t10k-labels.idx1-ubyte")

    lam = 1e-3

    loader = MnistDataloader(ti, tl, vi, vl)
    (X_train, y_train), (X_test, y_test) = loader.load_data()
    print("Train:", X_train.shape, y_train.shape)
    print("Test: ", X_test.shape,  y_test.shape)

    # PCA → top-50 components
    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    C  = (Xc.T @ Xc) / Xc.shape[0]
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    U50 = eigvecs[:, idx[:50]]
    Z_train = Xc @ U50
    Z_test  = (X_test - mu) @ U50
    print("PCA ->", Z_train.shape, Z_test.shape)

    # 0 vs 1
    mask01 = np.isin(y_train, [0,1])
    Z01, y01 = Z_train[mask01], y_train[mask01]
    w_gd, b_gd, hist_gd = gradient_descent(Z01, y01, lam=lam)
    w_nt, b_nt, hist_nt = newton_method(Z01, y01, lam=lam)
    print("0 vs 1 -> GD iters:", len(hist_gd['loss']),
          "Newton iters:", len(hist_nt['loss']))

    # 8 vs 9
    mask89 = np.isin(y_train, [8,9])
    Z89, y89 = Z_train[mask89], y_train[mask89]
    y89 = (y89 == 9).astype(np.uint8)
    w_gd_89, b_gd_89, hist_gd_89 = gradient_descent(Z89, y89, lam=lam)
    w_nt_89, b_nt_89, hist_nt_89 = newton_method(Z89, y89, lam=lam)
    print("8 vs 9 -> GD iters:", len(hist_gd_89['loss']),
          "Newton iters:", len(hist_nt_89['loss']))

    # build validation histories
    def make_val_history(hist, Z_val, y_val, lam):
        L_val, G_val = [], []
        for w_i, b_i in zip(hist['w'], hist['b']):
            L_val.append(logistic_loss(w_i, b_i, Z_val, y_val, lam))
            gw_i, gb_i = logistic_grad(w_i, b_i, Z_val, y_val, lam)
            G_val.append(np.linalg.norm(np.append(gw_i, gb_i)))
        return L_val, G_val

    mask01_test = np.isin(y_test, [0,1])
    Z01_t, y01_t = Z_test[mask01_test], y_test[mask01_test]
    mask89_test = np.isin(y_test, [8,9])
    Z89_t, y89_t = Z_test[mask89_test], y_test[mask89_test]
    y89_t       = (y89_t == 9).astype(np.uint8)

    L_gd_01, G_gd_01 = make_val_history(hist_gd,      Z01_t, y01_t, lam)
    L_nt_01, G_nt_01 = make_val_history(hist_nt,      Z01_t, y01_t, lam)
    L_gd_89, G_gd_89 = make_val_history(hist_gd_89,   Z89_t, y89_t, lam)
    L_nt_89, G_nt_89 = make_val_history(hist_nt_89,   Z89_t, y89_t, lam)

    # helper to plot
    def plot_curves(train, val, title, ylabel, semilogy=False):
        plt.figure()
        iters = range(len(train))
        plt.plot(iters, train, label='train')
        plt.plot(iters, val,   linestyle='--', label='val')
        if semilogy:
            plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()

    # 0 vs 1: loss & grad-norm
    plot_curves(hist_gd['loss'], L_gd_01,  '0 vs 1: GD Loss',     'Loss')
    plot_curves(hist_nt['loss'], L_nt_01,  '0 vs 1: Newton Loss', 'Loss')
    plot_curves(hist_gd['grad_norm'], G_gd_01, '0 vs 1: GD grad-norm',     'grad-norm', semilogy=True)
    plot_curves(hist_nt['grad_norm'], G_nt_01, '0 vs 1: Newton grad-norm', 'grad-norm', semilogy=True)

    # 8 vs 9: loss & grad-norm
    plot_curves(hist_gd_89['loss'], L_gd_89,  '8 vs 9: GD Loss',     'Loss')
    plot_curves(hist_nt_89['loss'], L_nt_89,  '8 vs 9: Newton Loss', 'Loss')
    plot_curves(hist_gd_89['grad_norm'], G_gd_89, '8 vs 9: GD grad-norm',     'grad-norm', semilogy=True)
    plot_curves(hist_nt_89['grad_norm'], G_nt_89, '8 vs 9: Newton grad-norm', 'grad-norm', semilogy=True)

    plt.show()
    
    
    def predict_accuracy(w, b, Z, y):
        probs = sigmoid(Z.dot(w) + b)
        preds = (probs >= 0.5).astype(int)
        return accuracy_score(y, preds)

    # 0 vs 1 test accuracy
    mask01_test = np.isin(y_test, [0,1])
    Z01_t, y01_t = Z_test[mask01_test], y_test[mask01_test]
    acc_gd_01 = predict_accuracy(w_gd, b_gd, Z01_t, (y01_t==1).astype(int))
    acc_nt_01 = predict_accuracy(w_nt, b_nt, Z01_t, (y01_t==1).astype(int))

    # 8 vs 9 test accuracy
    mask89_test = np.isin(y_test, [8,9])
    Z89_t, y89_t = Z_test[mask89_test], y_test[mask89_test]
    y89_t_bin   = (y89_t == 9).astype(int)
    acc_gd_89 = predict_accuracy(w_gd_89, b_gd_89, Z89_t, y89_t_bin)
    acc_nt_89 = predict_accuracy(w_nt_89, b_nt_89, Z89_t, y89_t_bin)

    print(f"0 vs 1 test acc: GD={acc_gd_01:.4f}, Newton={acc_nt_01:.4f}")
    print(f"8 vs 9 test acc: GD={acc_gd_89:.4f}, Newton={acc_nt_89:.4f}")
