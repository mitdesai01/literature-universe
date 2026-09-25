# -*- coding: utf-8 -*-
"""Adapters: one function per source, each writes canonical tables and returns True when it did.

To add a source: write adapt_<name>() in a module here, register it below, and add its
SOURCES entry in sources.py. Order matters only where an adapter reads another's output
(geography reads AIGE county names written by aioe).
"""
from .ai import adapt_aioe, adapt_dynamic_aioe, adapt_btos
from .politics import adapt_vrscores, adapt_elections, adapt_cspp
from .innovation import adapt_patentsview
from .geo import adapt_geo, adapt_irs_migration

ADAPTERS = {
    "aioe": adapt_aioe,
    "dynamic_aioe": adapt_dynamic_aioe,
    "btos": adapt_btos,
    "vrscores": adapt_vrscores,
    "elections": adapt_elections,
    "cspp": adapt_cspp,
    "geography": adapt_geo,
    "irs_migration": adapt_irs_migration,
    "patentsview": adapt_patentsview,
}
