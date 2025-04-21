import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 定义初始值
initial_theta = 0.5  # 假设初始值为 0.5

# 更新后的目标损失函数
def updated_loss_function(theta):
    return np.exp(theta) - np.log(theta)

# 优化求最小值
updated_result = minimize(updated_loss_function, x0=initial_theta, bounds=[(1e-5, None)])

# 可视化更新后的损失函数
theta_vals = np.linspace(0.1, 5, 500)
updated_loss_vals = updated_loss_function(theta_vals)

plt.plot(theta_vals, updated_loss_vals, label=r"$g(\theta) = e^{\theta} - \log(\theta)$")
plt.axvline(updated_result.x[0], color='r', linestyle='--', label=f"Minimum at θ ≈ {updated_result.x[0]:.4f}")
plt.xlabel("θ")
plt.ylabel("g(θ)")
plt.title("Optimization of g(θ) = exp(θ) - log(θ)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

updated_result.x[0], updated_result.fun
