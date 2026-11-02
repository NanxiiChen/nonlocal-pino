"""The same physical stream-envelope contracts at the explicit height scale."""
import unittest
from unittest.mock import patch

from tests import test_channel_stream_lifting as contracts

inlet_config=contracts.config


def height_config():
    c=inlet_config();c.stream_lifting.kind='normalized_height'
    return c


class ChannelStreamHeightTests(unittest.TestCase):
    def test_height_constraints_flux_divergence_and_legacy_parity(self):
        with patch('tests.test_channel_stream_lifting.config',height_config):
            contracts.ChannelStreamLiftingTests().test_constraints_flux_divergence_and_original_branch_parity()

    def test_height_physical_gradient_and_strict_schema(self):
        with patch('tests.test_channel_stream_lifting.config',height_config):
            contract=contracts.ChannelStreamLiftingTests()
            contract.test_full_physical_residual_parameter_gradient()
            contract.test_strict_schema_and_matched_jvp_comparison()


if __name__=='__main__':unittest.main()
