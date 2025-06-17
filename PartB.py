import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#------------5---------------------

# ------------------------------------------------------------------
# 1. נתונים: x_i = t_i  ,  y_obs = x_i / 1000
# ------------------------------------------------------------------
x = np.loadtxt(Path("Covid-19-USA.txt"), dtype=np.float64)
y_obs = x / 1000.0
m = x.size

# ------------------------------------------------------------------
# 2. פונקציית שאריות  r(x)  וג'ייקוביאן  J(x)   (x = [x1, x2])
# ------------------------------------------------------------------
def safe_exp(z):
    return np.exp(np.clip(z, -700, 700))

def residual(v):
    x1, x2 = v
    return safe_exp(x1 * x) - (1 + x2 * x)          # סעיף 5: אין y_obs

def jacobian(v):
    x1, _ = v
    exp_term = safe_exp(x1 * x)
    J = np.empty((m, 2))
    J[:, 0] = x * exp_term       # ∂r/∂x1
    J[:, 1] = -x                 # ∂r/∂x2
    return J

# ------------------------------------------------------------------
# 3. Steepest-Descent  (עם back-tracking)
# ------------------------------------------------------------------
def steepest_descent(x0, alpha0=1.0, c=1e-4, rho=0.5,
                     tol=1e-12, max_iter=500):
    xk = x0.astype(float).copy()
    f_hist, g_hist = [], []

    for k in range(max_iter):
        r = residual(xk)
        J = jacobian(xk)
        g = J.T @ r
        f = 0.5 * r.dot(r)

        # save history
        if k == 0 or g_hist[-1] >= tol:
            f_hist.append(f)
            g_hist.append(np.linalg.norm(g))
        if g_hist[-1] < tol:
            break

        # back-tracking Armijo
        alpha = alpha0
        while True:
            x_new = xk - alpha * g
            f_new = 0.5 * residual(x_new).dot(residual(x_new))
            if f_new <= f - c * alpha * g.dot(g):
                xk = x_new
                break
            alpha *= rho
    return xk, np.array(f_hist), np.array(g_hist)

# ------------------------------------------------------------------
# 4. Gauss–Newton  עם  Line-Search  (Armijo)
# ------------------------------------------------------------------
def gauss_newton_ls(x0, c=1e-4, rho=0.5,
                    tol=1e-12, max_iter=500):
    xk = x0.astype(float).copy()
    f_hist, g_hist = [], []

    for k in range(max_iter):
        r = residual(xk)
        J = jacobian(xk)
        g = J.T @ r
        f = 0.5 * r.dot(r)

        # save history
        if k == 0 or g_hist[-1] >= tol:
            f_hist.append(f)
            g_hist.append(np.linalg.norm(g))
        if g_hist[-1] < tol:
            break

        H = J.T @ J
        dx = -np.linalg.solve(H, g)

        # back-tracking
        alpha = 1.0
        while True:
            x_new = xk + alpha * dx
            f_new = 0.5 * residual(x_new).dot(residual(x_new))
            if f_new <= f - c * alpha * g.dot(dx):
                xk = x_new
                break
            alpha *= rho
    return xk, np.array(f_hist), np.array(g_hist)

# ------------------------------------------------------------------
# 5. Levenberg–Marquardt
# ------------------------------------------------------------------
def levenberg_marquardt(x0, mu0=1e-3, tol=1e-12, max_iter=500):
    xk = x0.astype(float).copy()
    mu, I = mu0, np.eye(2)
    f_hist, g_hist = [], []

    for k in range(max_iter):
        r = residual(xk)
        J = jacobian(xk)
        g = J.T @ r
        f = 0.5 * r.dot(r)

        if k == 0 or g_hist[-1] >= tol:
            f_hist.append(f)
            g_hist.append(np.linalg.norm(g))
        if g_hist[-1] < tol:
            break

        H = J.T @ J
        dx = -np.linalg.solve(H + mu * I, g)
        x_new = xk + dx
        f_new = 0.5 * residual(x_new).dot(residual(x_new))

        if f_new < f:
            xk = x_new
            mu *= 0.7
        else:
            mu *= 2.0
    return xk, np.array(f_hist), np.array(g_hist)

# ------------------------------------------------------------------
# 6. הרצה מ-x0  (ניחוש שלא גורם overflow)
# ------------------------------------------------------------------
x0 = np.array([-1e-4, 0.5])

x_sd, f_sd, g_sd = steepest_descent(x0)
x_gn, f_gn, g_gn = gauss_newton_ls(x0)
x_lm, f_lm, g_lm = levenberg_marquardt(x0)

print("--- Solutions ---")
print("SD :", x_sd, "f*=", f_sd[-1], "iter=", len(f_sd))
print("GN :", x_gn, "f*=", f_gn[-1], "iter=", len(f_gn))
print("LM :", x_lm, "f*=", f_lm[-1], "iter=", len(f_lm))

# ------------------------------------------------------------------
# 7. גרפים
# ------------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(f_sd, 'g.-', label="SD")
plt.plot(f_gn, 'b.-', label="GN-LS")
plt.plot(f_lm, 'C1.-', label="LM")
plt.xlabel("iteration"); plt.ylabel("f(x^k)")
plt.title("Objective value"); plt.grid(); plt.legend()

plt.subplot(1,2,2)
plt.semilogy(np.maximum(g_sd,1e-16), 'g.-', label="SD")
plt.semilogy(np.maximum(g_gn,1e-16), 'b.-', label="GN-LS")
plt.semilogy(np.maximum(g_lm,1e-16), 'C1.-', label="LM")
plt.xlabel("iteration"); plt.ylabel("||∇f||")
plt.title("Gradient norm (semilog)"); plt.grid(); plt.legend()

plt.tight_layout(); plt.show()

#------------6---------------------

x = np.loadtxt(Path("Covid-19-USA.txt"), dtype=np.float64)   # shape (m,)
y_obs = x / 1000.0
m = x.size


def safe_exp(z):
    return np.exp(np.clip(z, -700.0, 700.0))

def residual(theta):
    theta1, theta2, theta3 = theta
    delta = x - 110.0 * theta3
    E = safe_exp(-0.001 * theta2 * delta**2)
    return theta1 * E - y_obs          

def jacobian(theta):
    theta1, theta2, theta3 = theta
    delta = x - 110.0 * theta3
    E = safe_exp(-0.001 * theta2 * delta**2)

    J = np.empty((m, 3))
    J[:, 0] = E
    J[:, 1] = -0.001 * theta1 * E * delta**2
    J[:, 2] = 0.22 * theta1 * theta2 * E * delta    
    return J


def steepest_descent(theta0, alpha0=1.0, tol=1e-8,
                     c=1e-4, rho=0.5, max_iter=500):
    theta = theta0.astype(float).copy()
    f_hist, g_hist = [], []

    for k in range(max_iter):
        r = residual(theta)
        J = jacobian(theta)
        g = J.T @ r
        f_val = 0.5 * np.dot(r, r)

        f_hist.append(f_val)
        g_hist.append(np.linalg.norm(g))
        if g_hist[-1] < tol:
            break

        # Armijo back-tracking
        alpha = alpha0
        while True:
            theta_new = theta - alpha * g
            if 0.5 * np.dot(residual(theta_new), residual(theta_new)) \
               <= f_val - c * alpha * np.dot(g, g):
                break
            alpha *= rho
        theta = theta_new
    return theta, np.array(f_hist), np.array(g_hist)


def gauss_newton(theta0, tol=1e-8, max_iter=500):
    theta = theta0.astype(float).copy()
    f_hist, g_hist = [], []
    for k in range(max_iter):
        r = residual(theta)
        J = jacobian(theta)
        g = J.T @ r
        f_val = 0.5 * np.dot(r, r)

        f_hist.append(f_val)
        g_hist.append(np.linalg.norm(g))
        if g_hist[-1] < tol:
            break

        H = J.T @ J
        dtheta = -np.linalg.solve(H, g)
        theta += dtheta
    return theta, np.array(f_hist), np.array(g_hist)


def levenberg_marquardt(theta0, mu0=1e-3, tol=1e-8, max_iter=500):
    theta = theta0.astype(float).copy()
    mu = mu0
    I = np.eye(3)
    f_hist, g_hist = [], []
    for k in range(max_iter):
        r = residual(theta)
        J = jacobian(theta)
        g = J.T @ r
        f_val = 0.5 * np.dot(r, r)

        f_hist.append(f_val)
        g_hist.append(np.linalg.norm(g))
        if g_hist[-1] < tol:
            break

        H = J.T @ J
        dtheta = -np.linalg.solve(H + mu * I, g)
        theta_new = theta + dtheta
        f_new = 0.5 * np.dot(residual(theta_new), residual(theta_new))

        if f_new < f_val:        
            theta = theta_new
            mu *= 0.7
        else:                    
            mu *= 2.0
    return theta, np.array(f_hist), np.array(g_hist)


theta0 = np.array([1.0, 1.0, 1.0])

th_sd, f_sd, g_sd = steepest_descent(theta0)
th_gn, f_gn, g_gn = gauss_newton(theta0)
th_lm, f_lm, g_lm = levenberg_marquardt(theta0)

print("\n--- converged parameters ---")
print("SD : ", th_sd, "  f*=", f_sd[-1], "  it=", len(f_sd))
print("GN : ", th_gn, "  f*=", f_gn[-1], "  it=", len(f_gn))
print("LM : ", th_lm, "  f*=", f_lm[-1], "  it=", len(f_lm))

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(f_sd, label="SD")
plt.plot(f_gn, label="GN")
plt.plot(f_lm, label="LM")
plt.xlabel("iteration"); plt.ylabel("f"); plt.grid(); plt.legend()
plt.title("Objective value")

plt.subplot(1,2,2)
plt.semilogy(g_sd, label="SD")
plt.semilogy(g_gn, label="GN")
plt.semilogy(g_lm, label="LM")
plt.xlabel("iteration"); plt.ylabel("||∇f||"); plt.grid(); plt.legend()
plt.title("Gradient norm (semilog)")

plt.tight_layout(); plt.show()