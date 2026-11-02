# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Result organization for ArtiBench evaluations.

``dimension_report`` turns raw per-sample metric dicts into the three-dimension
report (``results.json`` / ``results.jsonl`` / ``analysis.json``) that every
entry point writes. Import it directly:

    from artibench.reporting.dimension_report import write_dimension_report
"""
