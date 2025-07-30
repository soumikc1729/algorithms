# https://leetcode.com/problems/maximum-subarray
def max_sub_array(nums):
    """
    >>> max_sub_array([-2,1,-3,4,-1,2,1,-5,4])
    6
    >>> max_sub_array([1])
    1
    >>> max_sub_array([5,4,-1,7,8])
    23
    >>> max_sub_array([-1])
    -1
    """
    MIN_POSSIBLE_NUM = -10**4 - 1
    max_sum_ending_here = max_sum_so_far = MIN_POSSIBLE_NUM
    for num in nums:
        max_sum_ending_here = max(max_sum_ending_here + num, num)
        max_sum_so_far = max(max_sum_so_far, max_sum_ending_here)
    return max_sum_so_far


if __name__ == "__main__":
    import doctest
    doctest.testmod()
