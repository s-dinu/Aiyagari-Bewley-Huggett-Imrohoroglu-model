# Project: Aiyagari-Bewley-Huggett-Imrohoroglu model simulation
# Author: Sergiu Dinu
# Date: September 2024

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

# Functions
def discretize_assets(a_min, a_max, n_a):
    """
    Discretizes the asset space.
    """
    # Maximum u_bar (uniform grid) corresponding to maximum a_max (asset grid)
    u_bar = np.log(1 + np.log(1 + a_max - a_min)) 
    
    # Uniform grid
    u_grid = np.linspace(0, u_bar, n_a) 
    
    # Double-exponentiate uniform grid from a_min to a_max
    return a_min + np.exp(np.exp(u_grid) - 1) - 1

@njit
def rouwenhorst_M(N, p):
    """
    Constructs the Markov transition matrix using the Rouwenhorst method.
    """
    # 2 x 2 matrix
    M = np.array([[p, 1 - p], [1 - p, p]])
    
    # n x n matrix
    for n in range(3, N + 1):
        M_old = M
        M = np.zeros((n, n))
        M[:-1, :-1] += p * M_old
        M[:-1, 1:] += (1 - p) * M_old
        M[1:, :-1] += (1 - p) * M_old
        M[1:, 1:] += p * M_old
        M[1:-1, :] /= 2
        
    return M

def stationary_M(M, tol=1E-6):
    """
    Finds the stationary distribution of the Markov transition matrix.
    """
    # Initial uniform distribution over all states
    n = M.shape[0]
    m = np.full(n, 1/n)
    
    # Update distribution using M for maximum 100,000 iterations
    for _ in range(100_000):
        m_new = M.T @ m
        if np.max(np.abs(m_new - m)) < tol:
            return m_new
        m = m_new
    raise ValueError("Stationary distribution did not converge")

def discretize_income(rho, sigma, n_y):
    """
    Discretizes the income process using the Rouwenhorst method.
    """
    # Inner-switching probability corresponding to persistence
    p = (1+rho)/2
    
    # Scale the income log by alpha to match standard deviation
    alpha = 2*sigma/np.sqrt(n_y-1)
    log_y = alpha*np.arange(n_y)
    
    # Stationary distribution of Markov transition matrix
    M = rouwenhorst_M(n_y, p)
    m = stationary_M(M)
    
    # Recover income and scale such that mean is 1
    y = np.exp(log_y)
    y /= np.vdot(m, y)
    
    return y, m, M

@njit
def utility(c):
    """
    Calculates the utility of consumption.
    """
    if c > 0:
        return np.log(c)
    else:
        return -1e10
    
def meanvar_plots(assets, consumption, income):
    """
    Plots mean of assets, consumption, and income for each time period.
    """
    mean_assets = np.mean(assets, axis=0)
    var_assets = np.var(assets, axis=0)
    mean_consumption = np.mean(consumption, axis=0)
    var_consumption = np.var(consumption, axis=0)
    mean_income = np.mean(income, axis=0)
    var_income = np.var(income, axis=0)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(mean_assets, label='Assets')
    plt.plot(mean_consumption, label='Consumption')
    plt.plot(mean_income, label='Income')
    plt.xlabel('Time')
    plt.legend()
    plt.title('Mean of variables')
    plt.subplot(1, 2, 2)
    plt.plot(var_assets, label='Assets')
    plt.plot(var_consumption, label='Consumption')
    plt.plot(var_income, label='Income')
    plt.xlabel('Time')
    plt.legend()
    plt.title('Variance of variables')
    plt.tight_layout()
    plt.show()

# Start the timing of code execution
start_time = time.time() 

# Parameters
r = 0.05
beta = 0.95
T = 40
a_min_values = [0]
n_a = 50
rho, sigma, n_y = 0.9, 0.1, 20
n_h = 100000  # Number of households
y_grid, m_y, M_y = discretize_income(rho, sigma, n_y)

for a_min in a_min_values:
    # Discretize the asset space
    a_grid = discretize_assets(a_min, a_min + n_a, n_a)

    # Initialize value function and policy function
    V = np.zeros((n_y, n_a, T + 1))
    policy = np.zeros((n_y, n_a, T))

    # Terminal period value function
    for j in range(n_y):
        for i in range(n_a):
            V[j, i, T-1] = utility((1 + r) * a_grid[i] + y_grid[j])

    # Backward induction based on the Bellman equation
    for t in range(T - 2, -1, -1):
        for j in range(n_y):
            for i in range(n_a):
                current_max = -1e10
                current_policy = 0
                for k in range(n_a):
                    c = (1 + r) * a_grid[i] + y_grid[j] - a_grid[k]
                    if c > 0:
                        expected_value = 0
                        for l in range(n_y):
                            expected_value += M_y[j, l] * V[l, k, t + 1]
                        value = utility(c) + beta * expected_value
                        if value > current_max:
                            current_max = value
                            current_policy = a_grid[k]
                V[j, i, t] = current_max
                policy[j, i, t] = current_policy

    # Function to get policy for given state
    def get_policy(a, y, t):
        """
        Retrieves the policy corresponding to assets a and income y in period t.
        """
        a_index = np.argmin(np.abs(a_grid - a))
        y_index = np.argmin(np.abs(y_grid - y))
        return policy[y_index, a_index, t]

    # Initialize assets, consumption, and income
    assets = np.zeros((n_h, T + 1))
    consumption = np.zeros((n_h, T))
    income = np.zeros((n_h, T))

    # Initial conditions
    assets[:, 0] = 0
    initial_income_index = np.argmin(np.abs(y_grid - 1))
    income[:, 0] = y_grid[initial_income_index]

    # Simulate income process
    for t in range(1, T):
        for h in range(n_h):
            current_income_index = np.argmin(np.abs(y_grid - income[h, t-1]))
            income_prob = M_y[current_income_index]
            income[h, t] = np.random.choice(y_grid, p=income_prob)

    # Simulate asset and consumption choices
    for t in range(T):
        for h in range(n_h):
            a_current = assets[h, t]
            y_current = income[h, t]
            a_next = get_policy(a_current, y_current, t)
            assets[h, t + 1] = a_next
            consumption[h, t] = (1 + r) * a_current + y_current - a_next
            if consumption[h, t] < 0:
                consumption[h, t] = 1e-10

    # Plot the evolution of distributions of the variables
    meanvar_plots(assets, consumption, income)

# Display the code runtime
end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")