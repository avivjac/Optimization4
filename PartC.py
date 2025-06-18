import numpy as np

#---------1--------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_regression_objective(X, y, w, b, reg_lambda: float = 0.0):
    m = X.shape[1]
    z = w.T @ X + b
    h = sigmoid(z)
    eps = 1e-12                       
    cost = -(np.sum(y * np.log(h + eps) +
                    (1 - y) * np.log(1 - h + eps))) / m
    cost += (reg_lambda / 2) * np.sum(w ** 2)
    return cost

def logistic_regression_gradient(X, y, w, b, reg_lambda: float = 0.0):
    m = X.shape[1]
    h = sigmoid(w.T @ X + b)
    diff = h - y

    grad_w = (X @ diff) / m + reg_lambda * w
    grad_b = np.sum(diff) / m
    return grad_w, grad_b

def logistic_regression_hessian(X, y, w, b, reg_lambda: float = 0.0):
    m = X.shape[1]
    h = sigmoid(w.T @ X + b)
    S = h * (1 - h)                    
    XS = X * S                         
    H_ww = (XS @ X.T) / m + reg_lambda * np.eye(X.shape[0])
    return H_ww

#---------2---------

def directional_gradient_test_wb(X, y, w, b, epsilon=1e-5, num_checks=5):
    print("Directional Gradient Test :")
    n = w.size
    for _ in range(num_checks):
        d = np.random.randn(n + 1)
        d /= np.linalg.norm(d)
        dw, db = d[:-1], d[-1]

        f0 = logistic_regression_objective(X, y, w,                b)
        f1 = logistic_regression_objective(X, y, w + epsilon * dw, b + epsilon * db)

        grad_w, grad_b = logistic_regression_gradient(X, y, w, b)
        gTd = grad_w @ dw + grad_b * db
        err = abs((f1 - f0) / epsilon - gTd)
        print(f"  error: {err:.5e}")

def directional_hessian_test_wb(X, y, w, b, epsilon=1e-5, num_checks=5):
    print("Directional Hessian Test :")
    n, m = X.shape

    # חישוב block-wise
    z  = w.T @ X + b
    h  = sigmoid(z)
    S  = h * (1 - h)

    XS   = X * S
    H_ww = XS @ X.T / m
    H_wb = np.sum(XS, axis=1) / m
    H_bb = np.sum(S) / m

    H = np.zeros((n + 1, n + 1))
    H[:n, :n]  = H_ww
    H[:n, -1]  = H_wb
    H[-1, :n]  = H_wb
    H[-1, -1]  = H_bb

    for _ in range(num_checks):
        d = np.random.randn(n + 1)
        d /= np.linalg.norm(d)
        dw, db = d[:-1], d[-1]

        g0_w, g0_b = logistic_regression_gradient(X, y, w,                b)
        g1_w, g1_b = logistic_regression_gradient(X, y, w + epsilon*dw,   b + epsilon*db)
        g_diff = np.concatenate([(g1_w - g0_w), [g1_b - g0_b]])

        Hd   = H @ d
        err  = np.linalg.norm(g_diff / epsilon - Hd)
        print(f"  error: {err:.5e}")

if __name__ == "__main__":
    np.random.seed(0)
    n, m = 5, 20
    X = np.random.randn(n, m)
    y = np.random.randint(0, 2, m).astype(float)

    w = np.random.randn(n)
    b = np.random.randn()

    directional_gradient_test_wb(X, y, w, b)
    print("")
    directional_hessian_test_wb(X, y, w, b)

#-------3---------
import matplotlib.pyplot as plt
import numpy.linalg as la

def show_images(images, titles, cols=5, figsize=(12, 6)):
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=figsize)
    for idx, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(rows, cols, idx)
        plt.imshow(img, cmap="gray")
        plt.title(title, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

x_train_n = x_train / 255.0 - 0.5
x_test_n  = x_test  / 255.0 - 0.5

x_train_flat = x_train_n.reshape(x_train.shape[0], -1).T  # (784 , m_train)
x_test_flat  = x_test_n.reshape(x_test.shape[0],  -1).T    # (784 , m_test)

# 2) Covariance matrix  Θ = (1/m) X Xᵀ   →  SVD
m = x_train_flat.shape[1]
theta = (x_train_flat @ x_train_flat.T) / m          # 784 × 784

U, S, _ = la.svd(theta)                              # Θ = U diag(S) Uᵀ
eigens = np.sqrt(S)                                  # √ eigenvalues

# 3) Plot eigenvalues
plt.figure(figsize=(6,4))
plt.plot(eigens, lw=1.5)
plt.title("Eigenvalues of covariance matrix")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.show()

# 4) Dimensionality reduction to p = 50
p = 50
U_p = U[:, :p]                        # (784 × 50)
Z_train = U_p.T @ x_train_flat        # (50 × m_train)
Z_test  = U_p.T @ x_test_flat         # (50 × m_test)

# 5) Reconstruct a few samples and compare
np.random.seed(20)
indices = np.random.choice(x_train.shape[0], 10, replace=False)

# original & reconstructed (shift back +0.5 only for display)
original_imgs      = [x_train_n[i] + 0.5 for i in indices]
reconstructed_flat = (U_p @ Z_train).T.reshape(-1, 28, 28)
reconstructed_imgs = [reconstructed_flat[i] + 0.5 for i in indices]

show_images(original_imgs,      [f"Original {i}"     for i in indices])
show_images(reconstructed_imgs, [f"Reconstructed {i}" for i in indices])