# Aiyagari-Bewley-Huggett-Imrohoroglu-model

Households aim to maximize their expected lifetime utility, which is represented in this model by the discounted sum of logarithmic utility from consumption. The choice variables are consumption today $c_t$ and assets tomorrow $a_{t+1}$. As such, the optimization problem can be formulated as:

$$
\max_{c_t, \hspace{0.05cm} a_{t+1}} \mathbb{E} \sum_{t=1}^{T} \beta^t \ln(c_t), \hspace{0.25cm} \text{subject to:} \hspace{0.25cm} c_t + a_{t+1} = (1 + r) a_t + y_t, \hspace{0.25cm} a_t \geq \underline{a},
$$

where $\beta$ is the discount factor, $r$ is the interest rate on assets, and $\underline{a}$ is the minimum asset level that agents can possess. The optimization reflects the trade-off between current consumption and future savings, as households aim to smooth consumption over time despite income fluctuations.  

The problem can be redefined in terms of a value function and using the binding budget constraint:

$$
V_t(a_t, y_t) = \max_{a_{t+1}} \left( \ln((1 + r) a_t + y_t - a_{t+1}) + \beta \mathbb{E} [V_{t+1}(a_{t+1}, y_{t+1}) \mid y_t] \right).
$$

Due to the finite nature of the model, backward induction can be used to solve for the policy functions, starting from the terminal period $T$ and working the way backward to the initial period $0$. In period $T$, households consume all their available resources since there are no future periods to save for. Thus, the value function at time $T$ is:

$$
V_T(a_T, y_T) = \ln ((1 + r) a_T + y_T),
$$

where $a_T$ and $y_T$ are the asset and income level respectively at time $T$. For each preceding period $t = \{T-1, T-2, ..., 0\}$, the value function $V_t(a_t, y_t)$ can be determined with the help of the Bellman equation. The corresponding policy function, which dictates the optimal choice of $a_{t+1}$ given $a_t$ and $y_t$, is:

$$
a_{t+1}^*(a_t, y_t) = \arg \max_{a_{t+1}} \left( \ln((1 + r) a_t + y_t - a_{t+1}) + \beta \mathbb{E} [V_{t+1}(a_{t+1}, y_{t+1}) \mid y_t] \right),
$$

Let $a_i \text{ for } i=1,2,\dots,n_a$ be the grid points for assets and $y_j \text{ for } j=1,2,\dots,n_y$
 the grid points for income. The value function and policy function are then computed on these grids. The backward induction proceeds as follows:

1. Initialize the value function for period $T$ for each $(a_i, y_j)$: $V_T(a_i, y_j) = \ln ((1 + r) a_i + y_j)$.
2. For each period $t = \{T-1, T-2, ..., 0\}$ and for each $(a_i, y_j)$:
    1. For each possible future asset level $a_k$, compute the consumption $c$: $c = (1 + r) a_i + y_j - a_k$
    2. If $c > 0$, compute the expected value:
        $\mathbb{E} [V_{t+1}(a_k, y_{t+1}) \mid y_t = y_j] = \sum_{l=1}^{n_y} \pi_{jl} V_{t+1}(a_k, y_l)$,
        where $\pi_{jl}$ is the transition probability from income state $y_j$ to $y_l$.
    3. Compute the total value:
        $\tilde{V}(a_k) = u(c) + \beta \mathbb{E} [V_{t+1}(a_k, y_{t+1}) \mid y_t = y_j]$
    4. Choose the future asset level $a_{t+1}$ that maximizes $\tilde{V}(a_k)$.
3. Update the value function and policy function: $V_t(a_i, y_j) = \max_k \tilde{V}(a_k)$ and $a_{t+1}^*(a_i, y_j) = \arg \max_k \tilde{V}(a_k)$.

Given initial conditions for assets and income, it is possible to find the optimal paths for savings and consumption and simulate the model while ensuring that the terminal condition is also satisfied.
