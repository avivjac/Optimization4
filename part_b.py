import numpy as np
import matplotlib.pyplot as plt

#---------5---------
data = np.loadtxt("Covid-19-USA.txt")        
y_obs = data / 1e6                           
x = np.arange(1, 100)                        
theta_0 = np.array([1.0, 0.001, 110.0])      

def gaussian_model(theta, x):
    """ θ₁ · exp( -θ₂·(x-θ₃)² ) """
    return theta[0] * np.exp(-theta[1] * (x - theta[2]) ** 2)

def residual(theta, x, y):
    return gaussian_model(theta, x) - y

def objective_function(theta, x, y):
    return 0.5 * np.linalg.norm(gaussian_model(theta, x) - y) ** 2

def gradient(theta, x, y):
    r = residual(theta, x, y)
    J = np.zeros((x.size, 3))

    J[:, 0] = np.exp(-theta[1] * (x - theta[2]) ** 2)
    J[:, 1] = -theta[0] * (x - theta[2]) ** 2 * 2 * np.exp(-theta[1] * (x - theta[2]) ** 2)
    J[:, 2] = 2 * theta[0] * theta[1] * (x - theta[2]) * np.exp(-theta[1] * (x - theta[2]) ** 2)

    return J.T @ r                            

def armijo_line_search(f, grad, theta_k, dk,
                       alpha0=1.0, beta=0.5, c=1e-4):
    alpha = alpha0
    while f(theta_k + alpha * dk) > f(theta_k) + c * alpha * grad(theta_k).T @ dk:
        alpha *= beta
    return alpha

def steepest_descent(theta0, x, y, max_iter=500, epsilon=1e-3):
    theta = theta0.copy()
    obj_values, grad_norms = [], []
    grad0 = gradient(theta, x, y)

    for i in range(max_iter):
        grad = gradient(theta, x, y)
        d = -grad
        f = lambda t: objective_function(t, x, y)
        g = lambda t: gradient(t, x, y)
        alpha = armijo_line_search(f, g, theta, d, alpha0=0.25)
        theta += alpha * d

        obj_values.append(objective_function(theta, x, y))
        grad_norms.append(np.linalg.norm(grad))

        if np.linalg.norm(grad) / np.linalg.norm(grad0) < epsilon:
            break

    return theta, np.array(obj_values), np.array(grad_norms)

def gauss_newton(theta0, x, y, max_iter=500, epsilon=1e-3, mu=1e-4):
    theta = theta0.copy()
    obj_values, grad_norms = [], []
    grad0 = gradient(theta, x, y)

    for i in range(max_iter):
        r = residual(theta, x, y)
        J = np.zeros((x.size, 3))
        
        J[:, 0] = np.exp(-theta[1] * (x - theta[2]) ** 2)
        J[:, 1] = -theta[0] * (x - theta[2]) ** 2 * 2 * np.exp(-theta[1] * (x - theta[2]) ** 2)
        J[:, 2] = 2 * theta[0] * theta[1] * (x - theta[2]) * np.exp(-theta[1] * (x - theta[2]) ** 2)

        grad = J.T @ r
        d = np.linalg.solve(J.T @ J + mu * np.eye(3), -grad)

        f = lambda t: objective_function(t, x, y)
        g = lambda t: gradient(t, x, y)
        alpha = armijo_line_search(f, g, theta, d, alpha0=1.0)
        theta += alpha * d

        obj_values.append(objective_function(theta, x, y))
        grad_norms.append(np.linalg.norm(grad))

        if np.linalg.norm(grad) / np.linalg.norm(grad0) < epsilon:
            break

    return theta, np.array(obj_values), np.array(grad_norms)

theta_SD, obj_SD, grad_SD = steepest_descent(theta_0, x, y_obs)
theta_GN, obj_GN, grad_GN = gauss_newton(theta_0, x, y_obs)

plt.figure(figsize=(8, 4))
plt.plot(obj_SD, label='F(θ) - SD', linewidth=2)
plt.title('Steepest Descent - Objective Value')
plt.xlabel('Iteration')
plt.ylabel('F(θ)')
plt.grid(True)
plt.legend()

plt.figure(figsize=(8, 4))
plt.semilogy(grad_SD, label='||∇F(θ)|| - SD', linewidth=2)
plt.title('Steepest Descent - Gradient Norm')
plt.xlabel('Iteration')
plt.ylabel('||∇F(θ)|| (log scale)')
plt.grid(True)
plt.legend()

plt.figure(figsize=(8, 4))
plt.plot(obj_GN, label='F(θ) - GN', linewidth=2)
plt.title('Gauss Newton - Objective Value')
plt.xlabel('Iteration')
plt.ylabel('F(θ)')
plt.grid(True)
plt.legend()

plt.figure(figsize=(8, 4))
plt.semilogy(grad_GN, label='||∇F(θ)|| - GN', linewidth=2)
plt.title('Gauss Newton - Gradient Norm')
plt.xlabel('Iteration')
plt.ylabel('||∇F(θ)|| (log scale)')
plt.grid(True)
plt.legend()

plt.show()

#---------6---------
data   = np.loadtxt("Covid-19-USA.txt")   
y_obs  = data / 1e6                       
x      = np.arange(1, 100)               
theta_0 = np.array([1.0, 1.0, 1.0])       

def scaled_gaussian_model(theta, x):
    """    θ₁ · exp( -0.001·θ₂·(x−110·θ₃)² )    """
    return theta[0] * np.exp(-0.001 * theta[1] * (x - 110 * theta[2]) ** 2)

def scaled_residual(theta, x, y):
    return scaled_gaussian_model(theta, x) - y

def scaled_objective_function(theta, x, y):
    return 0.5 * np.linalg.norm(scaled_gaussian_model(theta, x) - y) ** 2

def scaled_gradient(theta, x, y):
    r    = scaled_residual(theta, x, y)
    diff = x - 110 * theta[2]
    exp_term = np.exp(-0.001 * theta[1] * diff ** 2)

    J = np.zeros((x.size, 3))
    J[:, 0] = exp_term
    J[:, 1] = -0.001 * theta[0] * diff ** 2 * 2 * exp_term
    J[:, 2] = 2 * 0.001 * 110 * theta[0] * theta[1] * diff * exp_term
    return J.T @ r                       # ∇f = Jᵀ r

def armijo_line_search(f, grad, theta_k, dk,
                       alpha0=1.0, beta=0.5, c=1e-4):
    alpha = alpha0
    while f(theta_k + alpha * dk) > f(theta_k) + c * alpha * grad(theta_k).T @ dk:
        alpha *= beta
    return alpha

def steepest_descent(theta0, x, y, max_iter=500, epsilon=1e-3):
    theta       = theta0.copy()
    obj_values  = []
    grad_norms  = []
    grad0_norm  = np.linalg.norm(scaled_gradient(theta, x, y))

    for _ in range(max_iter):
        grad = scaled_gradient(theta, x, y)
        d    = -grad
        # Armijo step
        f = lambda t: scaled_objective_function(t, x, y)
        g = lambda t: scaled_gradient(t, x, y)
        alpha = armijo_line_search(f, g, theta, d, alpha0=0.25)
        theta += alpha * d

        obj_values.append(scaled_objective_function(theta, x, y))
        grad_norms.append(np.linalg.norm(grad))

        if grad_norms[-1] / grad0_norm < epsilon:
            break

    return theta, np.array(obj_values), np.array(grad_norms)

def gauss_newton(theta0, x, y, max_iter=500, epsilon=1e-3, mu=1e-4):
    theta       = theta0.copy()
    obj_values  = []
    grad_norms  = []
    grad0_norm  = np.linalg.norm(scaled_gradient(theta, x, y))

    for _ in range(max_iter):
        diff     = x - 110 * theta[2]
        exp_term = np.exp(-0.001 * theta[1] * diff ** 2)

        J = np.zeros((x.size, 3))
        J[:, 0] = exp_term
        J[:, 1] = -0.001 * theta[0] * diff ** 2 * 2 * exp_term
        J[:, 2] = 2 * 0.001 * 110 * theta[0] * theta[1] * diff * exp_term

        r    = scaled_residual(theta, x, y)
        grad = J.T @ r
        d    = np.linalg.solve(J.T @ J + mu * np.eye(3), -grad)

        f = lambda t: scaled_objective_function(t, x, y)
        g = lambda t: scaled_gradient(t, x, y)
        alpha = armijo_line_search(f, g, theta, d, alpha0=1.0)
        theta += alpha * d

        obj_values.append(f(theta))
        grad_norms.append(np.linalg.norm(grad))

        if grad_norms[-1] / grad0_norm < epsilon:
            break

    return theta, np.array(obj_values), np.array(grad_norms)

theta_SD, obj_SD, grad_SD = steepest_descent(theta_0, x, y_obs)
theta_GN, obj_GN, grad_GN = gauss_newton(theta_0, x, y_obs)

plt.figure(figsize=(8,4))
plt.plot(obj_SD, label='F(θ) – SD', lw=2)
plt.title('Steepest Descent – Objective')
plt.xlabel('Iteration'); plt.ylabel('F(θ)'); plt.grid(); plt.legend()

plt.figure(figsize=(8,4))
plt.semilogy(grad_SD, label='‖∇F(θ)‖ – SD', lw=2)
plt.title('Steepest Descent – Gradient norm')
plt.xlabel('Iteration'); plt.ylabel('‖∇F(θ)‖ (log)'); plt.grid(); plt.legend()

plt.figure(figsize=(8,4))
plt.plot(obj_GN, label='F(θ) – GN', lw=2)
plt.title('Gauss–Newton – Objective')
plt.xlabel('Iteration'); plt.ylabel('F(θ)'); plt.grid(); plt.legend()

plt.figure(figsize=(8,4))
plt.semilogy(grad_GN, label='‖∇F(θ)‖ – GN', lw=2)
plt.title('Gauss–Newton – Gradient norm')
plt.xlabel('Iteration'); plt.ylabel('‖∇F(θ)‖ (log)'); plt.grid(); plt.legend()

plt.show()