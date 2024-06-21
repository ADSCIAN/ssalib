"""Functions to load vassal datasets."""

from .data_loader import (
    load_mortality,
    load_sst,
    load_sunspots
)

__all__ = ['load_mortality', 'load_sst', 'load_sunspots']