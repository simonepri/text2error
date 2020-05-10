from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import random


def non_adjacent_sample(
    n: int, m: int, rng: Optional[random.Random] = None
) -> List[int]:
    """
    Generates a uniform random sample from a given 1-D array such that
    elements chosen are not adjacent (i.e. they are at distance >= 1 in the
    array).
    """
    # pylint: disable=invalid-name
    # TODO: The current implementation is not always uniform yet.

    if rng is None:
        # pylint: disable=protected-access
        rng = random._inst  # type: ignore

    odd = n % 2 == 1
    max_m = (n + 1) // 2
    if n < 0 or m < 0:
        raise ValueError("n and m must be non negative.")
    if m > max_m:
        raise ValueError("m must be at most (n + 1) // 2.")
    if n == 0 or m == 0:
        return []
    if odd:
        n += 1

    max_b = n // 2
    if m == max_b:
        pairs = list(range(max_b))
    else:
        pairs = rng.sample(range(max_b), k=m)
        pairs.sort()

    # Compute a DP array for the pairs, each with two entry (can, cant), which is
    # equal to the number of ways of taking one element per pair, in the two cases:
    #     # 1. No restriction
    #     # 2. The first element cannot be taken
    #  Those values are computed as follows:
    #     for the last element, (2, 1)
    #     for the previous ones,
    #          this.cant := next.id == this.id + 1 if next.cant else next.can
    #          this.can := next.can + this.cant
    dp = [[0, 0] for _ in range(m)]
    dp[m - 1] = [2, 1]
    for i in range(m - 2, -1, -1):
        dp_c, dp_n = dp[i], dp[i + 1]
        dp_c[1] = dp_n[int(pairs[i + 1] == pairs[i] + 1)]
        dp_c[0] = dp_n[0] + dp_c[1]

    # Now select the elements in a uniform way using the dp array.
    # More precisely, when the choice is forced
    # (i.e. pairs[current - 1] == pairs[current] - 1) and we took the second,
    # we're forced to take the second element again.
    # Otherwise we take the second with probability cant/can.
    ans = [0 for i in range(m)]
    first_forbidden = 0 if odd else -1
    for i in range(m):
        element = pairs[i] * 2  # first element of the pair
        if element == first_forbidden or rng.random() < (dp[i][1] / dp[i][0]):
            element += 1
        first_forbidden = element + 1
        ans[i] = element
        if odd:
            ans[i] -= 1

    return ans
