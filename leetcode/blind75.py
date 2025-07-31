# https://leetcode.com/problems/invert-binary-tree
def invert_tree(root):
    """
    >>> from datastructures.binary_tree import BinaryTree
    >>> BinaryTree.serialize(invert_tree(BinaryTree.deserialize([4, 2, 7, 1, 3, 6, 9])))
    [4, 7, 2, 9, 6, 3, 1]
    >>> BinaryTree.serialize(invert_tree(BinaryTree.deserialize([2, 1, 3])))
    [2, 3, 1]
    >>> BinaryTree.serialize(invert_tree(BinaryTree.deserialize([])))
    []
    """
    if not root:
        return root

    left = invert_tree(root.left)
    right = invert_tree(root.right)

    root.left, root.right = right, left

    return root


# https://leetcode.com/problems/contains-duplicate
def contains_duplicate(nums):
    """
    >>> contains_duplicate([1, 2, 3, 1])
    True
    >>> contains_duplicate([1, 2, 3, 4])
    False
    >>> contains_duplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2])
    True
    """
    num_set = set()
    for num in nums:
        if num in num_set:
            return True
        num_set.add(num)
    return False


# https://leetcode.com/problems/jump-game
def can_jump(nums):
    """
    >>> can_jump([2, 3, 1, 1, 4])
    True
    >>> can_jump([3, 2, 1, 0, 4])
    False
    """
    can_jump_upto = 0
    for i, num in enumerate(nums):
        if i > can_jump_upto:
            return False
        can_jump_upto = max(can_jump_upto, i + num)
    return True


# https://leetcode.com/problems/maximum-subarray
def max_sub_array(nums):
    """
    >>> max_sub_array([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    6
    >>> max_sub_array([1])
    1
    >>> max_sub_array([5, 4, -1, 7, 8])
    23
    >>> max_sub_array([-1])
    -1
    """
    MIN_POSSIBLE_NUM = -(10**4) - 1
    max_sum_ending_here = max_sum_so_far = MIN_POSSIBLE_NUM
    for num in nums:
        max_sum_ending_here = max(max_sum_ending_here + num, num)
        max_sum_so_far = max(max_sum_so_far, max_sum_ending_here)
    return max_sum_so_far


if __name__ == "__main__":
    import doctest

    doctest.testmod()
