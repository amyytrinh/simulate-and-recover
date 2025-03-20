import unittest
import numpy as np
from src.ez_diffusion import EZDiffusionModel

class TestEZDiffusionModel(unittest.TestCase):

    def test_forward_equations_valid(self):
        """Test if forward equations produce valid predicted statistics."""
        model = EZDiffusionModel(1.0, 1.0, 0.2)
        R_pred, M_pred, V_pred = model.forward_equations()
        
        self.assertTrue(0 < R_pred < 1, "R_pred should be between 0 and 1")
        self.assertTrue(M_pred > 0, "M_pred should be positive")
        self.assertTrue(V_pred > 0, "V_pred should be positive")

    def test_forward_equations_extreme_values(self):
        """Test forward equations with extreme parameter values."""
        model = EZDiffusionModel(2.0, 0.5, 0.5)  # Edge values
        R_pred, M_pred, V_pred = model.forward_equations()

        self.assertTrue(0 < R_pred < 1, "R_pred should be between 0 and 1 even for extreme values")
        self.assertTrue(M_pred > 0, "M_pred should be positive")
        self.assertTrue(V_pred > 0, "V_pred should be positive")

    def test_simulate_observed_data_small_N(self):
        """Test observed data simulation with N=1 (smallest sample size)."""
        R_pred, M_pred, V_pred = 0.7, 0.4, 0.02
        R_obs, M_obs, V_obs = EZDiffusionModel.simulate_observed_data(R_pred, M_pred, V_pred, 1)

        self.assertTrue(0 <= R_obs <= 1, "R_obs should be between 0 and 1")
        self.assertTrue(np.isfinite(M_obs), "M_obs should be finite")
        self.assertTrue(np.isfinite(V_obs), "V_obs should be finite")

    def test_simulate_observed_data_large_N(self):
        """Test observed data simulation with a very large N."""
        R_pred, M_pred, V_pred = 0.7, 0.4, 0.02
        R_obs, M_obs, V_obs = EZDiffusionModel.simulate_observed_data(R_pred, M_pred, V_pred, 10**6)

        self.assertTrue(0 <= R_obs <= 1, "R_obs should be between 0 and 1")
        self.assertTrue(np.isfinite(M_obs), "M_obs should be finite")
        self.assertTrue(np.isfinite(V_obs), "V_obs should be finite")

    def test_inverse_equations_valid(self):
        """Test inverse equations for valid recovery."""
        R_obs, M_obs, V_obs = 0.7, 0.4, 0.02
        recovered_params = EZDiffusionModel.inverse_equations(R_obs, M_obs, V_obs)

        self.assertIsNotNone(recovered_params, "Recovered parameters should not be None")
        self.assertTrue(all(np.isfinite(recovered_params)), "All recovered parameters should be finite")

    def test_inverse_equations_edge_case_Robs_0_1(self):
        """Test inverse equations for edge cases where R_obs is exactly 0 or 1."""
        recovered_params_0 = EZDiffusionModel.inverse_equations(0, 0.4, 0.02)
        recovered_params_1 = EZDiffusionModel.inverse_equations(1, 0.4, 0.02)

        self.assertIsNone(recovered_params_0, "Recovery should return None for R_obs = 0")
        self.assertIsNone(recovered_params_1, "Recovery should return None for R_obs = 1")

    def test_inverse_equations_zero_variance(self):
        """Test inverse equations when variance V_obs is near zero."""
        recovered_params = EZDiffusionModel.inverse_equations(0.7, 0.4, 1e-10)

        self.assertIsNotNone(recovered_params, "Recovered parameters should not be None even for small variance")
        self.assertTrue(all(np.isfinite(recovered_params)), "All recovered parameters should be finite")

    def test_compute_metrics_valid(self):
        """Test if compute_metrics correctly calculates bias and squared error."""
        true_params = (1.2, 0.8, 0.3)
        recovered_params = (1.1, 0.9, 0.25)

        bias, squared_error = EZDiffusionModel.compute_metrics(true_params, recovered_params)

        self.assertTrue(all(np.isfinite(bias)), "Bias values should be finite")
        self.assertTrue(all(np.isfinite(squared_error)), "Squared error values should be finite")
        self.assertTrue(all(squared_error >= 0), "Squared error values should be non-negative")

    def test_compute_metrics_invalid(self):
        """Test compute_metrics when recovered parameters are None."""
        true_params = (1.2, 0.8, 0.3)
        bias, squared_error = EZDiffusionModel.compute_metrics(true_params, None)

        self.assertIsNone(bias, "Bias should be None when recovered parameters are invalid")
        self.assertIsNone(squared_error, "Squared error should be None when recovered parameters are invalid")

    def test_simulate_observed_data_N_zero(self):
        """Test that N=0 raises an error."""
        with self.assertRaises(ValueError):
            EZDiffusionModel.simulate_observed_data(0.5, 0, 1, 0)

    def test_simulate_observed_data_R_pred_limits(self):
        """Test that R_pred values at 0 and 1 don't break."""
        R_obs_0, _, _ = EZDiffusionModel.simulate_observed_data(0, 0, 1, 10)
        R_obs_1, _, _ = EZDiffusionModel.simulate_observed_data(1, 0, 1, 10)
        self.assertIn(R_obs_0, [0, 0.1])  # Because N=10
        self.assertIn(R_obs_1, [0.9, 1])  # Because N=10

    def test_simulate_observed_data_negative_variance(self):
        """Test that negative variance raises an error."""
        with self.assertRaises(ValueError):
            EZDiffusionModel.simulate_observed_data(0.5, 0, -1, 10)

    def test_simulate_observed_data_large_N(self):
        """Test that a large N runs without crashing."""
        R_obs, M_obs, V_obs = EZDiffusionModel.simulate_observed_data(0.5, 1, 1, 10000)
        self.assertIsInstance(R_obs, float)
        self.assertIsInstance(M_obs, float)
        self.assertIsInstance(V_obs, float)

    def test_zero_bias_when_obs_equals_pred(self):
        """Test that bias is zero when (R_obs, M_obs, V_obs) = (R_pred, M_pred, V_pred)."""
        # Generate a set of true parameters
        true_a, true_v, true_t = 1.2, 0.8, 0.3
        model = EZDiffusionModel(true_a, true_v, true_t)

        R_pred, M_pred, V_pred = model.forward_equations()
        R_obs, M_obs, V_obs = R_pred, M_pred, V_pred

        recovered_params = EZDiffusionModel.inverse_equations(R_obs, M_obs, V_obs)
        bias, _ = EZDiffusionModel.compute_metrics((true_a, true_v, true_t), recovered_params)

        # Assert that bias is effectively zero for all parameters
        np.testing.assert_almost_equal(bias, [0, 0, 0], decimal=6, err_msg="Bias should be zero when (R_obs, M_obs, V_obs) = (R_pred, M_pred, V_pred)")

if __name__ == "__main__":
    unittest.main()
