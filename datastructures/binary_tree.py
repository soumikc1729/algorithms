from collections import deque


class BinaryTreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def create(val):
        return BinaryTreeNode(val) if val else None


def deserialize_binary_tree(vals):
    if not vals:
        return []

    root = BinaryTreeNode.create(vals[0])

    q = deque()
    q.append(root)

    i = 0
    while True:
        node = q.popleft()
        i += 1
        if i == len(vals):
            break
        node.left = BinaryTreeNode.create(vals[i])
        if node.left:
            q.append(node.left)
        i += 1
        if i == len(vals):
            break
        node.right = BinaryTreeNode.create(vals[i])
        if node.right:
            q.append(node.right)

    return root


def serialize_binary_tree(root):
    if not root:
        return []

    q = deque()
    q.append(root)

    arr = [root.val]

    def append(node):
        if node:
            q.append(node)
            arr.append(node.val)
        else:
            arr.append(None)

    while q:
        node = q.popleft()
        append(node.left)
        append(node.right)

    i = len(arr) - 1
    while i >= 0 and not arr[i]:
        i -= 1

    return arr[: i + 1]
