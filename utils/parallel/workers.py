"""collections of commonly used parallel worker functions."""

import numpy as np

def energy_content(source, level, field, mx)
"""Compute the energy content of the field

-'source' the data source
-'level' the time level to read from
-'field' the name of field to read
- 'mx' mass matrix

Equivalent of u^T x M x u, where M is the mass matrix and U is the coefficient vector.
"""
coeffs = source.coefficient	