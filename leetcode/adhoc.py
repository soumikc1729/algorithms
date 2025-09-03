from typing import List


# https://leetcode.com/problems/jump-game-ix
def max_value(nums: List[int]) -> List[int]:
    """
    >>> max_value([2, 1, 3])
    [2, 2, 3]
    >>> max_value([2, 3, 1])
    [3, 3, 3]
    >>> max_value([9, 30, 16, 6, 36, 9])
    [36, 36, 36, 36, 36, 36]
    """
    ans = [0] * len(nums)
    sorted_idx_nums = list(sorted(enumerate(nums), key=lambda x: -x[1]))
    last_max = last_min = sorted_idx_nums[0][1]
    upto = len(nums)
    for i, n in sorted_idx_nums:
        if i > upto:  # already covered
            continue
        if n > last_min:  # can use last_min as stepping stone
            for j in range(i, upto):
                last_min = min(last_min, nums[j])
                ans[j] = last_max
            upto = i
            continue
        # n <= last_min and n is the highest from left still not covered
        ans[i] = n
        for j in range(i + 1, upto):
            last_min = min(last_min, nums[j])
            ans[j] = n
        last_max, upto = n, i
        upto = i
    return ans


if __name__ == "__main__":
    import doctest

    doctest.testmod()
