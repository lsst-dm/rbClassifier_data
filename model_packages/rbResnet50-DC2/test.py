# This file is part of meas_transiNet.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest
import torch
import numpy as np

from lsst.meas.transiNet import RBTransiNetInterface, CutoutInputs
from lsst.meas.transiNet.modelPackages import NNModelPackage


class TestWithModelPackage(unittest.TestCase):
    def setUp(self):
        self.model_package_name = 'rbResnet50-DC2'

    def test_load(self):
        """Test that this model package can be loaded
        """
        model_package = NNModelPackage(self.model_package_name, 'neighbor')
        model = model_package.load(device='cpu')

        weights = next(model.parameters())

        # test shape of loaded weights
        self.assertTupleEqual(weights.shape, (64, 3, 7, 7))

        # test weight values
        torch.testing.assert_close(weights[0][0][0],
                                   torch.tensor([-0.06167374, 0.0780527, 0.01648014, -0.05989857,
                                                 0.01510609, 0.06968862, 0.05509452]),
                                   rtol=1e-8, atol=1e-8)


class TestWithTransiNetInterface(unittest.TestCase):
    """Test the RBTransiNetInterface using this model package.
    """

    def setUp(self):
        model_package_name = 'rbResnet50-DC2'
        self.interface = RBTransiNetInterface(model_package_name, 'neighbor')

    def test_infer_empty(self):
        """Test running infer on images containing all zeros.
        """
        data = np.zeros((256, 256), dtype=np.single)
        inputs = CutoutInputs(science=data, difference=data, template=data)
        result = self.interface.infer([inputs])
        self.assertTupleEqual(result.shape, (1,))
        self.assertAlmostEqual(result[0], 0.00042127)  # Empirical meaningless value spit by this very model
