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

import pytest

from unittest import TestCase, mock

from polyaxon_schemas.utils.requests_utils import Bar, create_progress_callback


@pytest.mark.utils_mark
class TestRequestsUtils(TestCase):
    def test_create_progress_callback(self):
        encoder = mock.MagicMock()
        encoder.configure_mock(len=10)
        _, progress_bar = create_progress_callback(encoder)
        assert isinstance(progress_bar, Bar)
