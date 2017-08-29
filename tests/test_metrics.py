# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from unittest import TestCase

from polyaxon_schemas.metrics import (
    StreamingTruePositivesConfig,
    StreamingTrueNegativesConfig,
    StreamingFalseNegativesConfig,
    StreamingFalsePositivesConfig,
    StreamingMeanConfig,
    StreamingMeanTensorConfig,
    StreamingAccuracyConfig,
    StreamingPrecisionConfig,
    StreamingRecallConfig,
    StreamingAUCConfig,
    StreamingSpecificityAtSensitivityConfig,
    StreamingSensitivityAtSpecificityConfig,
    StreamingPrecisionAtThresholdsConfig,
    StreamingRecallAtThresholdsConfig,
    StreamingSparsePrecisionAtKConfig,
    StreamingMeanAbsoluteErrorConfig,
    StreamingMeanRelativeErrorConfig,
    StreamingMeanSquaredErrorConfig,
    StreamingRootMeanSquaredErrorConfig,
    StreamingCovarianceConfig,
    StreamingPearsonCorrelationConfig,
    StreamingMeanCosineDistanceConfig,
    StreamingPercentageLessConfig,
    StreamingMeanIOUConfig,
)
from tests.utils import assert_tensors, assert_equal_dict


class TestMetricConfigs(TestCase):
    @staticmethod
    def assert_equal_metrics(m1, m2):
        assert_tensors(m1.pop('input_layer', None), m2.pop('input_layer', None))
        assert_tensors(m1.pop('output_layer', None), m2.pop('output_layer', None))

        assert_equal_dict(m1, m2)

    def test_base_metrics_config(self):
        config_dict = {
            'input_layer': 'images',
            'output_layer': 'relu_1',
            'weights': None,
            'name': 'm'
        }

        config_classes = [
            StreamingTruePositivesConfig,
            StreamingTrueNegativesConfig,
            StreamingFalsePositivesConfig,
            StreamingFalseNegativesConfig,
            StreamingMeanConfig,
            StreamingAccuracyConfig,
            StreamingPrecisionConfig,
            StreamingRecallConfig,
            StreamingAUCConfig,
            StreamingMeanAbsoluteErrorConfig,
            StreamingMeanSquaredErrorConfig,
            StreamingRootMeanSquaredErrorConfig,
            StreamingCovarianceConfig,
            StreamingPearsonCorrelationConfig,
        ]

        for config_class in config_classes:
            config = config_class.from_dict(config_dict)
            self.assert_equal_metrics(config.to_dict(), config_dict)

    def test_mean_tensor_metric_config(self):
        config_dicts = [
            {
                'tensor': ['images', 0, 0],
                'weights': None,
                'name': 'm'
            },
            {
                'tensor': ['images', 0, 0],
                'weights': None,
                'name': 'm'
            }
        ]

        for config_dict in config_dicts:
            config = StreamingMeanTensorConfig.from_dict(config_dict)
            self.assert_equal_metrics(config.to_dict(), config_dict)

    def test_specificity_at_sensitivity_metric_config(self):
        config_dict = {
            'input_layer': 'images',
            'output_layer': 'relu_1',
            'sensitivity': 0.1,
            'num_thresholds': 300,
            'weights': None,
            'name': 'm'
        }
        config = StreamingSpecificityAtSensitivityConfig.from_dict(config_dict)
        self.assert_equal_metrics(config.to_dict(), config_dict)

    def test_sensitivity_at_specificity_metric_config(self):
        config_dict = {
            'input_layer': 'images',
            'output_layer': 'relu_1',
            'specificity': 0.1,
            'num_thresholds': 300,
            'weights': None,
            'name': 'm'
        }
        config = StreamingSensitivityAtSpecificityConfig.from_dict(config_dict)
        self.assert_equal_metrics(config.to_dict(), config_dict)

    def test_metric_at_thresholds_config(self):
        config_dict = {
            'input_layer': 'images',
            'output_layer': 'relu_1',
            'thresholds': [0.1, 0.1, 0.1],
            'weights': None,
            'name': 'm'
        }
        config_classes = [StreamingPrecisionAtThresholdsConfig, StreamingRecallAtThresholdsConfig]
        for config_class in config_classes:
            config = config_class.from_dict(config_dict)
            self.assert_equal_metrics(config.to_dict(), config_dict)

    def test_sparse_precistion_at_k_metric_config(self):
        config_dict = {
            'input_layer': 'images',
            'output_layer': 'relu_1',
            'k': 1,
            'class_id': None,
            'weights': None,
            'name': 'm'
        }
        config = StreamingSparsePrecisionAtKConfig.from_dict(config_dict)
        self.assert_equal_metrics(config.to_dict(), config_dict)

    def test_mean_relative_error_config(self):
        config_dict = {
            'input_layer': 'images',
            'output_layer': 'relu_1',
            'normalizer': 'tensor_normalizer',
            'weights': None,
            'name': 'm'
        }
        config = StreamingMeanRelativeErrorConfig.from_dict(config_dict)
        self.assert_equal_metrics(config.to_dict(), config_dict)

    def test_mean_consine_distance_metric_config(self):
        config_dict = {
            'input_layer': 'images',
            'output_layer': 'relu_1',
            'dim': 1,
            'weights': None,
            'name': 'm'
        }
        config = StreamingMeanCosineDistanceConfig.from_dict(config_dict)
        self.assert_equal_metrics(config.to_dict(), config_dict)

    def test_percentage_less_metric_config(self):
        config_dict = {
            'input_layer': 'images',
            'output_layer': 'relu_1',
            'tensor': 'tensor',
            'threshold': 0.4,
            'weights': None,
            'name': 'm'
        }
        config = StreamingPercentageLessConfig.from_dict(config_dict)
        self.assert_equal_metrics(config.to_dict(), config_dict)

    def test_mean_iou_config(self):
        config_dict = {
            'input_layer': 'images',
            'output_layer': 'relu_1',
            'num_classes': 4,
            'weights': None,
            'name': 'm'
        }
        config = StreamingMeanIOUConfig.from_dict(config_dict)
        self.assert_equal_metrics(config.to_dict(), config_dict)
