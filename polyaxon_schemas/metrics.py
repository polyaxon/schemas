# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from marshmallow import Schema, fields, validate, post_load

from polyaxon_schemas.base import BaseConfig, BaseMultiSchema
from polyaxon_schemas.utils import Tensor


class BaseMetricSchema(Schema):
    input_layer = Tensor(allow_none=True)
    output_layer = Tensor(allow_none=True)
    weights = fields.Float(allow_none=True)
    name = fields.Str(allow_none=True)


class BaseMetricConfig(BaseConfig):
    def __init__(self, input_layer=None, output_layer=None, weights=None, name=None):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.weights = weights
        self.name = name


class StreamingTruePositivesSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingTruePositivesConfig(**data)


class StreamingTruePositivesConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingTruePositives'
    SCHEMA = StreamingTruePositivesSchema


class StreamingTrueNegativesSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingTrueNegativesConfig(**data)


class StreamingTrueNegativesConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingTrueNegatives'
    SCHEMA = StreamingTrueNegativesSchema


class StreamingFalsePositivesSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingFalsePositivesConfig(**data)


class StreamingFalsePositivesConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingFalsePositives'
    SCHEMA = StreamingFalsePositivesSchema


class StreamingFalseNegativesSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingFalseNegativesConfig(**data)


class StreamingFalseNegativesConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingFalseNegatives'
    SCHEMA = StreamingFalseNegativesSchema


class StreamingMeanSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingMeanConfig(**data)


class StreamingMeanConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingMean'
    SCHEMA = StreamingMeanSchema


class StreamingMeanTensorSchema(Schema):
    tensor = Tensor()
    weights = fields.Float(allow_none=True)
    name = fields.Str(allow_none=True)

    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingMeanTensorConfig(**data)


class StreamingMeanTensorConfig(BaseConfig):
    IDENTIFIER = 'StreamingMeanTensor'
    SCHEMA = StreamingMeanTensorSchema

    def __init__(self, tensor, weights=None, name=None):
        self.tensor = tensor
        self.weights = weights
        self.name = name


class StreamingAccuracySchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingAccuracyConfig(**data)


class StreamingAccuracyConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingAccuracy'
    SCHEMA = StreamingAccuracySchema


class StreamingPrecisionSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingPrecisionConfig(**data)


class StreamingPrecisionConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingPrecision'
    SCHEMA = StreamingPrecisionSchema


class StreamingRecallSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingRecallConfig(**data)


class StreamingRecallConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingRecall'
    SCHEMA = StreamingRecallSchema


class StreamingAUCSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingAUCConfig(**data)


class StreamingAUCConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingAUC'
    SCHEMA = StreamingAUCSchema


class StreamingSpecificityAtSensitivitySchema(BaseMetricSchema):
    sensitivity = fields.Float(validate=validate.Range(min=0., max=1.))
    num_thresholds = fields.Int(allow_none=True)

    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingSpecificityAtSensitivityConfig(**data)


class StreamingSpecificityAtSensitivityConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingSpecificityAtSensitivity'
    SCHEMA = StreamingSpecificityAtSensitivitySchema

    def __init__(self,
                 sensitivity,
                 num_thresholds=200,
                 input_layer=None,
                 output_layer=None,
                 weights=None,
                 name=None):
        self.sensitivity = sensitivity
        self.num_thresholds = num_thresholds
        super(StreamingSpecificityAtSensitivityConfig, self).__init__(
            input_layer, output_layer, weights, name)


class StreamingSensitivityAtSpecificitySchema(BaseMetricSchema):
    specificity = fields.Float(validate=validate.Range(min=0., max=1.))
    num_thresholds = fields.Int(allow_none=True)

    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingSensitivityAtSpecificityConfig(**data)


class StreamingSensitivityAtSpecificityConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingSensitivityAtSpecificity'
    SCHEMA = StreamingSensitivityAtSpecificitySchema

    def __init__(self,
                 specificity,
                 num_thresholds=200,
                 input_layer=None,
                 output_layer=None,
                 weights=None,
                 name=None):
        self.specificity = specificity
        self.num_thresholds = num_thresholds
        super(StreamingSensitivityAtSpecificityConfig, self).__init__(
            input_layer, output_layer, weights, name)


class StreamingPrecisionAtThresholdsSchema(BaseMetricSchema):
    thresholds = fields.List(fields.Float(validate=validate.Range(min=0., max=1.)))

    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingPrecisionAtThresholdsConfig(**data)


class StreamingPrecisionAtThresholdsConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingPrecisionAtThresholds'
    SCHEMA = StreamingPrecisionAtThresholdsSchema

    def __init__(self,
                 thresholds,
                 input_layer=None,
                 output_layer=None,
                 weights=None,
                 name=None):
        self.thresholds = thresholds
        super(StreamingPrecisionAtThresholdsConfig, self).__init__(
            input_layer, output_layer, weights, name)


class StreamingRecallAtThresholdsSchema(BaseMetricSchema):
    thresholds = fields.List(fields.Float(validate=validate.Range(min=0., max=1.)))

    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingRecallAtThresholdsConfig(**data)


class StreamingRecallAtThresholdsConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingRecallAtThresholds'
    SCHEMA = StreamingRecallAtThresholdsSchema

    def __init__(self,
                 thresholds,
                 input_layer=None,
                 output_layer=None,
                 weights=None,
                 name=None):
        self.thresholds = thresholds
        super(StreamingRecallAtThresholdsConfig, self).__init__(
            input_layer, output_layer, weights, name)


class StreamingSparseRecallAtKSchema(BaseMetricSchema):
    k = fields.Int()
    class_id = fields.Int(allow_none=True)

    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingSparseRecallAtKConfig(**data)


class StreamingSparseRecallAtKConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingSparseRecallAtK'
    SCHEMA = StreamingSparseRecallAtKSchema

    def __init__(self,
                 k,
                 class_id,
                 input_layer=None,
                 output_layer=None,
                 weights=None,
                 name=None):
        self.k = k
        self.class_id = class_id
        super(StreamingSparseRecallAtKConfig, self).__init__(
            input_layer, output_layer, weights, name)


class StreamingSparsePrecisionAtKSchema(BaseMetricSchema):
    k = fields.Int()
    class_id = fields.Int(allow_none=True)

    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingSparsePrecisionAtKConfig(**data)


class StreamingSparsePrecisionAtKConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingSparsePrecisionAtK'
    SCHEMA = StreamingSparsePrecisionAtKSchema

    def __init__(self,
                 k,
                 class_id,
                 input_layer=None,
                 output_layer=None,
                 weights=None,
                 name=None):
        self.k = k
        self.class_id = class_id
        super(StreamingSparsePrecisionAtKConfig, self).__init__(
            input_layer, output_layer, weights, name)


class StreamingMeanAbsoluteErrorSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingMeanAbsoluteErrorConfig(**data)


class StreamingMeanAbsoluteErrorConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingMeanAbsoluteError'
    SCHEMA = StreamingMeanAbsoluteErrorSchema


class StreamingMeanRelativeErrorSchema(BaseMetricSchema):
    normalizer = fields.Str()  # name of the normalizer tensor

    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingMeanRelativeErrorConfig(**data)


class StreamingMeanRelativeErrorConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingMeanRelativeError'
    SCHEMA = StreamingMeanRelativeErrorSchema

    def __init__(self,
                 normalizer,
                 input_layer=None,
                 output_layer=None,
                 weights=None,
                 name=None):
        self.normalizer = normalizer
        super(StreamingMeanRelativeErrorConfig, self).__init__(
            input_layer, output_layer, weights, name)


class StreamingMeanSquaredErrorSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingMeanSquaredErrorConfig(**data)


class StreamingMeanSquaredErrorConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingMeanSquaredError'
    SCHEMA = StreamingMeanSquaredErrorSchema


class StreamingRootMeanSquaredErrorSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingRootMeanSquaredErrorConfig(**data)


class StreamingRootMeanSquaredErrorConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingRootMeanSquaredError'
    SCHEMA = StreamingRootMeanSquaredErrorSchema


class StreamingCovarianceSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingCovarianceConfig(**data)


class StreamingCovarianceConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingCovariance'
    SCHEMA = StreamingCovarianceSchema


class StreamingPearsonCorrelationSchema(BaseMetricSchema):
    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingPearsonCorrelationConfig(**data)


class StreamingPearsonCorrelationConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingPearsonCorrelation'
    SCHEMA = StreamingPearsonCorrelationSchema


class StreamingMeanCosineDistanceSchema(BaseMetricSchema):
    dim = fields.Int()

    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingMeanCosineDistanceConfig(**data)


class StreamingMeanCosineDistanceConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingMeanCosineDistance'
    SCHEMA = StreamingMeanCosineDistanceSchema

    def __init__(self,
                 dim,
                 input_layer=None,
                 output_layer=None,
                 weights=None,
                 name=None):
        self.dim = dim
        super(StreamingMeanCosineDistanceConfig, self).__init__(
            input_layer, output_layer, weights, name)


class StreamingPercentageLessSchema(BaseMetricSchema):
    tensor = fields.Str()
    threshold = fields.Float()

    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingPercentageLessConfig(**data)


class StreamingPercentageLessConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingPercentageLess'
    SCHEMA = StreamingPercentageLessSchema

    def __init__(self,
                 tensor,
                 threshold,
                 input_layer=None,
                 output_layer=None,
                 weights=None,
                 name=None):
        self.tensor = tensor
        self.threshold = threshold
        super(StreamingPercentageLessConfig, self).__init__(
            input_layer, output_layer, weights, name)


class StreamingMeanIOUSchema(BaseMetricSchema):
    num_classes = fields.Int()

    class Meta:
        ordered = True

    @post_load
    def make_load(self, data):
        return StreamingMeanIOUConfig(**data)


class StreamingMeanIOUConfig(BaseMetricConfig):
    IDENTIFIER = 'StreamingMeanIOU'
    SCHEMA = StreamingMeanIOUSchema

    def __init__(self,
                 num_classes,
                 input_layer=None,
                 output_layer=None,
                 weights=None,
                 name=None):
        self.num_classes = num_classes
        super(StreamingMeanIOUConfig, self).__init__(
            input_layer, output_layer, weights, name)


class MetricSchema(BaseMultiSchema):
    __multi_schema_name__ = 'Metric'
    __configs__ = {
        StreamingTruePositivesConfig.IDENTIFIER: StreamingTruePositivesConfig,
        StreamingTrueNegativesConfig.IDENTIFIER: StreamingTrueNegativesConfig,
        StreamingFalsePositivesConfig.IDENTIFIER: StreamingFalsePositivesConfig,
        StreamingFalseNegativesConfig.IDENTIFIER: StreamingFalseNegativesConfig,
        StreamingMeanConfig.IDENTIFIER: StreamingMeanConfig,
        StreamingMeanTensorConfig.IDENTIFIER: StreamingMeanTensorConfig,
        StreamingAccuracyConfig.IDENTIFIER: StreamingAccuracyConfig,
        StreamingPrecisionConfig.IDENTIFIER: StreamingPrecisionConfig,
        StreamingRecallConfig.IDENTIFIER: StreamingRecallConfig,
        StreamingAUCConfig.IDENTIFIER: StreamingAUCConfig,
        StreamingSpecificityAtSensitivityConfig.IDENTIFIER: StreamingSpecificityAtSensitivityConfig,
        StreamingSensitivityAtSpecificityConfig.IDENTIFIER: StreamingSensitivityAtSpecificityConfig,
        StreamingPrecisionAtThresholdsConfig.IDENTIFIER: StreamingPrecisionAtThresholdsConfig,
        StreamingRecallAtThresholdsConfig.IDENTIFIER: StreamingRecallAtThresholdsConfig,
        StreamingSparseRecallAtKConfig.IDENTIFIER: StreamingSparseRecallAtKConfig,
        StreamingSparsePrecisionAtKConfig.IDENTIFIER: StreamingSparsePrecisionAtKConfig,
        StreamingMeanAbsoluteErrorConfig.IDENTIFIER: StreamingMeanAbsoluteErrorConfig,
        StreamingMeanRelativeErrorConfig.IDENTIFIER: StreamingMeanRelativeErrorConfig,
        StreamingMeanSquaredErrorConfig.IDENTIFIER: StreamingMeanSquaredErrorConfig,
        StreamingRootMeanSquaredErrorConfig.IDENTIFIER: StreamingRootMeanSquaredErrorConfig,
        StreamingCovarianceConfig.IDENTIFIER: StreamingCovarianceConfig,
        StreamingPearsonCorrelationConfig.IDENTIFIER: StreamingPearsonCorrelationConfig,
        StreamingMeanCosineDistanceConfig.IDENTIFIER: StreamingMeanCosineDistanceConfig,
        StreamingPercentageLessConfig.IDENTIFIER: StreamingPercentageLessConfig,
        StreamingMeanIOUConfig.IDENTIFIER: StreamingMeanIOUConfig,
    }
