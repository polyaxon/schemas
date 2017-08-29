# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from polyaxon_schemas.polyaxonfile import validator
from polyaxon_schemas.polyaxonfile import reader
from polyaxon_schemas.polyaxonfile.parser import Parser


class PolyaxonFile(object):
    """Parses Polyaxonfiles, and validate that it respects the current file specification"""

    def __init__(self, filepath):
        self._filepath = filepath
        self._data = None
        self._parsed_data = None
        self._schema = None

        self._data = reader.read(self._filepath)
        self._parsed_data = Parser.parse(self._data)
        self._validated_data = validator.validate(self._parsed_data)
