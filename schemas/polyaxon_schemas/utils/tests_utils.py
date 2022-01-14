#!/usr/bin/python
#
# Copyright 2018-2021 Polyaxon, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from collections.abc import Mapping


def assert_equal_dict(dict1, dict2):
    for k, v in dict1.items():
        if v is None:
            continue
        if isinstance(v, Mapping):
            assert_equal_dict(v, dict2[k])
        else:
            assert v == dict2[k]


def assert_equal_feature_processors(fp1, fp2):
    # Check that they have same features
    assert list(fp1.keys()) == list(fp2.key())

    # Check that all features have the same graph
    for feature in fp1:
        assert_equal_graphs(fp2[feature], fp1[feature])


def assert_tensors(tensor1, tensor2):
    if isinstance(tensor1, str):
        tensor1 = [tensor1, 0, 0]

    if isinstance(tensor2, str):
        tensor2 = [tensor2, 0, 0]

    assert tensor1 == tensor2


def assert_equal_graphs(result_graph, expected_graph):
    for i, input_layer in enumerate(expected_graph["input_layers"]):
        assert_tensors(input_layer, result_graph["input_layers"][i])

    for i, output_layer in enumerate(expected_graph["output_layers"]):
        assert_tensors(output_layer, result_graph["output_layers"][i])

    for layer_i, layer in enumerate(result_graph["layers"]):
        layer_name, layer_data = list(layer.items())[0]
        assert layer_name in expected_graph["layers"][layer_i]
        for k, v in layer_data.items():
            assert v == expected_graph["layers"][layer_i][layer_name][k]


def assert_equal_layers(config, expected_layer):
    result_layer = config.to_dict()
    for k, v in expected_layer.items():
        if v is not None or k not in config.REDUCED_ATTRIBUTES:
            assert v == result_layer[k]
        else:
            assert k not in result_layer


def tensor_np(shape, dtype=float):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)
