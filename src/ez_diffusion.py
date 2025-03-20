#Prompted ChatGPT to use equations from EZ Diffusion Model 

import numpy as np

class EZDiffusionModel:
    def __init__(self, a, v, t):
        """Initialize with boundary separation (α), drift rate (ν), and nondecision time (τ)."""
        self.a = a  # Boundary separation (α)
        self.v = v  # Drift rate (ν)
        self.t = t  # Nondecision time (τ)

    def forward_equations(self):
        """Step 2: Compute predicted summary statistics (R_pred, M_pred, V_pred)."""
        y = np.exp(-self.v * self.a)  # Exponential term

        R_pred = 1 / (1 + y)  # Eq. (1)
        M_pred = self.t + (self.a / (2 * self.v)) * ((1 - y) / (1 + y))  # Eq. (2)
        V_pred = (self.a / (2 * self.v**3)) * ((1 - 2 * self.v * self.a * y - y**2) / (1 + y)**2)  # Eq. (3)

        return R_pred, M_pred, V_pred

    @staticmethod
    def simulate_observed_data(R_pred, M_pred, V_pred, N):
        """Step 3: Simulate observed summary statistics using sampling distributions."""
        if N < 1:
            raise ValueError("N must be at least 1.")
        if N == 1:
            R_obs = R_pred
            M_obs = M_pred
            V_obs = V_pred  # Avoid division by zero
        else:
            R_obs = np.random.binomial(N, R_pred) / N # Eq. (7)
            M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N)) # Eq. (8)
            V_obs = np.random.gamma((N - 1) / 2, 2 * V_pred / (N - 1))  # Eq. (9)
        
        return R_obs, M_obs, V_obs

    @staticmethod
    def inverse_equations(R_obs, M_obs, V_obs):
        """Recover estimated parameters (ν_est, α_est, τ_est) while avoiding instability issues."""
        if R_obs <= 0 or R_obs >= 1:
            return None  # Prevent log(0) errors

        L = np.log(R_obs / (1 - R_obs))

        # Prevent instability for small variance values
        V_obs = max(V_obs, 1e-5)  # Set a minimum variance threshold

        v_est = np.sign(R_obs - 0.5) * (max(0, L * (R_obs**2 * L - R_obs * L + R_obs - 0.5)) / V_obs) ** (1/4)

        if v_est == 0 or np.isnan(v_est):
            return None  

        a_est = L / v_est  
        t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))

        return a_est, v_est, t_est



    @staticmethod
    def compute_metrics(true_params, recovered_params):
        """Compute bias and squared error for parameter recovery."""
        if recovered_params is None:  # Handle cases where recovery is invalid
            return None, None  

        bias = np.array(recovered_params) - np.array(true_params)  # Bias = true - estimated
        squared_error = bias ** 2  # Squared error

        return bias, squared_error
