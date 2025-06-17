import numpy as np
import matplotlib.pyplot as plt
from loadMNIST import MnistDataloader
from os.path import join
import os

# ----------A-------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss(w, b, X, y, lam):
    """
    Logistic regression loss with L2 regularization
    """
    m = X.shape[1]
    z = w @ X + b       # shape (m,)
    s = sigmoid(z)      # shape (m,)

    # Avoid log(0)
    eps = 1e-15
    s = np.clip(s, eps, 1 - eps)

    loss = (-1 / m) * np.sum(y * np.log(s) + (1 - y) * np.log(1 - s))
    loss += (lam / 2) * np.sum(w ** 2)
    return loss

def logistic_grad(w, b, X, y, lam):
    """
    Gradient of logistic loss w.r.t w and b
    """
    m = X.shape[1]
    z = w @ X + b
    s = sigmoid(z)

    dw = (1 / m) * (X @ (s - y).T) + lam * w   # shape (n,)
    db = (1 / m) * np.sum(s - y)
    return dw, db

def logistic_hessian(w, b, X, y, lam):
    """
    Hessian of logistic loss w.r.t w (and optionally b)
    Returns: (n+1)x(n+1) matrix for [w; b]
    """
    m = X.shape[1]
    z = w @ X + b
    s = sigmoid(z)
    D = s * (1 - s)  # shape (m,)
    
    # build X~ by adding row of ones for bias
    X_tilde = np.vstack([X, np.ones((1, m))])  # shape (n+1, m)

    # compute Hessian using D
    H = (1 / m) * (X_tilde * D) @ X_tilde.T
    
    # add lambda to w-w block (not bias)
    H[:-1, :-1] += lam * np.eye(w.shape[0])

    return H

# ----------B-------------

def gradient_check(w, b, X, y, lam, d_w=None, d_b=None, epsilons=None):
    """
    Checks the gradient implementation of logistic loss.
    """
    if epsilons is None:
        epsilons = [10**(-i) for i in range(1, 9)]

    if d_w is None:
        d_w = np.random.randn(*w.shape)
    if d_b is None:
        d_b = np.random.randn()

    loss0 = logistic_loss(w, b, X, y, lam)
    grad_w, grad_b = logistic_grad(w, b, X, y, lam)
    g_dot_d = grad_w @ d_w + grad_b * d_b

    print("Gradient Check:")
    for eps in epsilons:
        w_eps = w + eps * d_w
        b_eps = b + eps * d_b
        loss_eps = logistic_loss(w_eps, b_eps, X, y, lam)
        num_approx = (loss_eps - loss0) / eps
        rel_err = abs(num_approx - g_dot_d) / max(1e-8, abs(num_approx) + abs(g_dot_d))
        print(f"ε={eps:.0e}:  approx={num_approx:.6f},  grad·d={g_dot_d:.6f},  rel_error={rel_err:.2e}")

def hessian_check(w, b, X, y, lam, d_w=None, d_b=None, epsilons=None):
    """
    Checks the Hessian implementation of logistic loss.
    """
    if epsilons is None:
        epsilons = [10**(-i) for i in range(1, 9)]

    if d_w is None:
        d_w = np.random.randn(*w.shape)
    if d_b is None:
        d_b = np.random.randn()

    grad0_w, grad0_b = logistic_grad(w, b, X, y, lam)
    grad0 = np.concatenate([grad0_w, [grad0_b]])

    H = logistic_hessian(w, b, X, y, lam)
    d = np.concatenate([d_w, [d_b]])
    H_d = H @ d

    print("\nHessian Check:")
    for eps in epsilons:
        w_eps = w + eps * d_w
        b_eps = b + eps * d_b
        grad_eps_w, grad_eps_b = logistic_grad(w_eps, b_eps, X, y, lam)
        grad_eps = np.concatenate([grad_eps_w, [grad_eps_b]])
        diff = (grad_eps - grad0) / eps
        rel_err = np.linalg.norm(diff - H_d) / max(1e-8, np.linalg.norm(diff) + np.linalg.norm(H_d))
        print(f"ε={eps:.0e}:  rel_error={rel_err:.2e}")

# checking derivatives for logistic regression
def check_derivatives(w, b, X, y, lam, d_w=None, d_b=None, epsilons=None):
    if epsilons is None:
        epsilons = [10**(-i) for i in range(1, 9)]

    if d_w is None:
        d_w = np.random.randn(*w.shape)
    if d_b is None:
        d_b = np.random.randn()

    # === GRADIENT CHECK ===
    loss0 = logistic_loss(w, b, X, y, lam)
    grad_w, grad_b = logistic_grad(w, b, X, y, lam)
    g_dot_d = grad_w @ d_w + grad_b * d_b

    grad_errors = []
    for eps in epsilons:
        w_eps = w + eps * d_w
        b_eps = b + eps * d_b
        loss_eps = logistic_loss(w_eps, b_eps, X, y, lam)
        num_approx = (loss_eps - loss0) / eps
        rel_err = abs(num_approx - g_dot_d) / max(1e-8, abs(num_approx) + abs(g_dot_d))
        grad_errors.append(rel_err)

    # === HESSIAN CHECK ===
    grad0 = np.concatenate([grad_w, [grad_b]])
    H = logistic_hessian(w, b, X, y, lam)
    d = np.concatenate([d_w, [d_b]])
    H_d = H @ d

    hess_errors = []
    for eps in epsilons:
        w_eps = w + eps * d_w
        b_eps = b + eps * d_b
        grad_eps_w, grad_eps_b = logistic_grad(w_eps, b_eps, X, y, lam)
        grad_eps = np.concatenate([grad_eps_w, [grad_eps_b]])
        diff = (grad_eps - grad0) / eps
        rel_err = np.linalg.norm(diff - H_d) / max(1e-8, np.linalg.norm(diff) + np.linalg.norm(H_d))
        hess_errors.append(rel_err)

    # === PLOT RESULTS ===
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].semilogx(epsilons, grad_errors, marker='o', label='Gradient check')
    ax[0].set_title('Relative error - Gradient')
    ax[0].set_xlabel('ε')
    ax[0].set_ylabel('Relative error')
    ax[0].grid(True)
    
    ax[1].semilogx(epsilons, hess_errors, marker='o', color='orange', label='Hessian check')
    ax[1].set_title('Relative error - Hessian')
    ax[1].set_xlabel('ε')
    ax[1].set_ylabel('Relative error')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    n, m = 5, 10
    X = np.random.randn(n, m)
    y = np.random.randint(0, 2, size=m)
    w = np.random.randn(n)
    b = np.random.randn()
    lam = 0.1

    # Check gradients and Hessians
    check_derivatives(w, b, X, y, lam)
 
# ----------C-------------

# --- שלב 1: הגדרת נתיבים לטעינת MNIST ---
# cwd = os.getcwd()
# input_path = cwd + '\MNIST'

from loadMNIST import MnistDataloader, show_images

# קבע את הנתיבים הנכונים
training_images_filepath = r'HW4/train-images.idx3-ubyte'
training_labels_filepath = r'HW4/train-labels.idx1-ubyte'
test_images_filepath = r'HW4/t10k-images.idx3-ubyte'
test_labels_filepath = r'HW4/t10k-labels.idx1-ubyte'

# טען את הנתונים
mnist = MnistDataloader(training_images_filepath, training_labels_filepath,
                        test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# === חלק 2: סינון ספרות והמרה למטריצה X ∈ R^(784 x m) ===
def filter_digits(images, labels, digits=(0,1), max_samples=30000):
    images = np.array(images)
    labels = np.array(labels)
    mask = np.isin(labels, digits)
    images = images[mask]
    labels = labels[mask]
    if images.shape[0] > max_samples:
        images = images[:max_samples]
        labels = labels[:max_samples]

    # flatten to shape (784, m), normalize to [-0.5, 0.5]
    X = images.reshape(images.shape[0], -1).T / 255.0 - 0.5
    y = (labels == digits[1]).astype(np.float64)
    return X, y

X, y = filter_digits(x_train_raw, y_train_raw, digits=(0,1), max_samples=30000)
m = X.shape[1]

# === חלק 3: חישוב מטריצת קווריאנציה Θ ∈ R^(784 x 784) ===
X_mean = np.mean(X, axis=1, keepdims=True)
X_centered = X - X_mean
Theta = (1 / m) * (X_centered @ X_centered.T)

# === חלק 4: Eigendecomposition של Θ ===
eigvals, eigvecs = np.linalg.eigh(Theta)  # מתאים כי Θ סימטרית

# === חלק 5: מיון הערכים העצמיים והעמודות של U בסדר יורד ===
idx = np.argsort(eigvals)[::-1]  # יורד
eigvals_sorted = eigvals[idx]
U = eigvecs[:, idx]              # גם את העמודות של U נמיין בהתאם

# === חלק 6: הצגת ערכי Σ (שורש של eigenvalues) ===
singular_values = np.sqrt(np.maximum(eigvals_sorted, 0))  # רק חיוביים
plt.figure(figsize=(8, 4))
plt.plot(singular_values, marker='o')
plt.title("Singular values (√eigenvalues of Θ)")
plt.xlabel("Index")
plt.ylabel("Singular value")
plt.grid(True)
plt.show()

# === חלק 7: הקרנה ל־p=50 ממדים (Z ∈ R^(50 x m)) ===
p = 50
Up = U[:, :p]        # בסיס ה־PCA
Z = Up.T @ X_centered

# === חלק 8: שחזור תמונה ובדיקת איכות ההפחתה ===
i = 0  # ניקח לדוגמה את הדוגמה הראשונה
zi = Z[:, i]                             # הווקטור במימד 50
x_rec = Up @ zi + X_mean[:, 0]          # שחזור + החזרת ממוצע
x_orig = X[:, i] + X_mean[:, 0]         # המקורית אחרי centering

# === ציור: המקורית מול המשוחזרת ===
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(x_orig.reshape(28, 28), cmap='gray')
plt.title("Original image")

plt.subplot(1, 2, 2)
plt.imshow(x_rec.reshape(28, 28), cmap='gray')
plt.title("Reconstructed from p=50")
plt.tight_layout()
plt.show()