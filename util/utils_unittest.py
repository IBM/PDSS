import unittest

from util.activation_extractor import (
    PytorchActivationExtractor,
    KerasActivationExtractor,
)
from util.custom_functions import CustomFunction
from util.metrics import Metrics
from util.model_loader import ModelLoader
from util.pvalranges_calculator import PvalueCalculator
from util.sampler import Sampler
from util.scoring_functions import ScoringFunctions

import numpy as np


# python3 -m unittest util/utils_unittest.py
class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_scoring_functions(self):
        n_alpha = np.array([7.0, 13.0, 18.0])
        no_records = np.array(
            [
                55.0,
                110.0,
                165.0,
            ]
        )
        alpha = np.array([0.00332226, 0.00332226, 0.00332226])

        scores = ScoringFunctions.get_score_bj_fast(n_alpha, no_records, alpha)
        scores = np.around(scores, 8)
        assert np.array_equal(scores, [19.14519838, 34.55369211, 46.35644406])
        scores = ScoringFunctions.get_score_hc_fast(n_alpha, no_records, alpha)
        scores = np.around(scores, 8)
        assert np.array_equal(scores, [15.97479615, 20.93482312, 23.61047593])
        scores = ScoringFunctions.get_score_ks_fast(n_alpha, no_records, alpha)
        scores = np.around(scores, 8)
        assert np.array_equal(scores, [0.91924127, 1.20465721, 1.3586229])

    def test_sampler(self):
        clean = np.random.rand(15, 10, 5, 2)
        anom = np.random.rand(15, 10, 5, 2)

        samples, indices = Sampler.sample(clean, anom, 2, 2, 2, conditional=True)

        assert len(samples) == 2
        assert len(samples[0]) == 4

        assert indices is not None

        clean = np.random.rand(15, 10, 2)
        anom = np.random.rand(15, 10, 2)

        samples, indices = Sampler.sample(clean, anom, 1, 0, 2)

        assert indices is None
