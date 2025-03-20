#Prompted ChatGPT to simulate model

import numpy as np
from ez_diffusion import EZDiffusionModel

def run_simulation(N, iterations=1000):
    """Run multiple simulations and analyze parameter recovery."""
    biases, squared_errors = [], []

    for _ in range(iterations):
        # Step 1: Select random true parameters
        a, v, t = np.random.uniform(0.5, 2), np.random.uniform(0.5, 2), np.random.uniform(0.1, 0.5)
        model = EZDiffusionModel(a, v, t)

        # Step 2: Compute predicted statistics
        R_pred, M_pred, V_pred = model.forward_equations()

        # Step 3: Simulate noisy observed data
        R_obs, M_obs, V_obs = model.simulate_observed_data(R_pred, M_pred, V_pred, N)

        # Step 4: Recover parameters from observed data
        recovered_params = EZDiffusionModel.inverse_equations(R_obs, M_obs, V_obs)

        # Compute bias and squared error
        metrics = EZDiffusionModel.compute_metrics((a, v, t), recovered_params)

        if metrics[0] is not None:  # Ignore cases where recovery failed
            biases.append(metrics[0])
            squared_errors.append(metrics[1])

    return np.mean(biases, axis=0), np.mean(squared_errors, axis=0)

if __name__ == "__main__":
    Ns = [10, 40, 4000]
    for N in Ns:
        bias, squared_error = run_simulation(N)
        print(f'N={N}, Bias: {bias}, Squared Error: {squared_error}')
