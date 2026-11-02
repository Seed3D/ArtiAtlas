# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Part matching via Hungarian assignment."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_match_bbox(
    centers_a: List[np.ndarray],
    centers_b: List[np.ndarray],
) -> np.ndarray:
    """Hungarian assignment on bbox center distances.

    Returns:
        mapping: (N_a, 2) array where mapping[i] = [matched_j, cost].
        Unmatched parts have matched_j = -1.
    """
    na, nb = len(centers_a), len(centers_b)
    INF = 1e9

    cost = np.full((na, nb), INF, dtype=np.float32)
    for i in range(na):
        for j in range(nb):
            cost[i, j] = float(np.linalg.norm(centers_a[i] - centers_b[j]))

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = np.full((na, 2), -1, dtype=np.float32)
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < INF:
            mapping[r, 0] = c
            mapping[r, 1] = cost[r, c]
    return mapping
