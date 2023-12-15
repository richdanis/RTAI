import unittest
import torch
import torch.nn as nn
from transforms import *


class TestTransformers(unittest.TestCase):
    def test_transform_linear(self):
        # Create a simple linear layer
        layer = nn.Linear(5, 3)

        # Define an input tensor and its bounds
        init_lb = torch.zeros(5)
        init_ub = torch.ones(5)

        # Define the relative bounds
        lb_rel = torch.eye(5)
        ub_rel = torch.eye(5)

        # Transform the bounds
        lb_rel_transformed, ub_rel_transformed = transform_linear(lb_rel, ub_rel, layer.weight, layer.bias)

        # Check that the transformed bounds have the correct shape
        self.assertEqual(lb_rel_transformed.shape, (3, 6))
        self.assertEqual(ub_rel_transformed.shape, (3, 6))

        # Check that the transformed bounds are correct
        # (This will depend on the specific values of the layer's weights and bias,
        #  so you'll need to replace this with the correct checks for your specific case)
        # self.assertTrue(torch.allclose(lb_rel_transformed, expected_lb_rel_transformed))
        # self.assertTrue(torch.allclose(ub_rel_transformed, expected_ub_rel_transformed))

if __name__ == '__main__':
    unittest.main()