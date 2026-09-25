# -*- coding: utf-8 -*-
"""POLISY lab: Politics, Organizations, Leadership, Strategy & Innovation.

A reusable research laboratory on top of the POLISY_DA pipeline: a registry of open datasets
(sources.py), adapters that turn each into canonical tables (adapters/), panels that link them
(link.py), analyses that register findings and the data behind every chart (analyses/), and
a static site that renders them (site.py). Start with run_all().
"""
from .core import LAB, LAB_ROOT, RESULTS, __version__
from .run import run_all, STAGES

__all__ = ["run_all", "STAGES", "LAB", "LAB_ROOT", "RESULTS", "__version__"]
