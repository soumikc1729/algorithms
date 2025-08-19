class UnionFind:
    def __init__(self, keys):
        if len(keys) <= 0:
            raise Exception("length of keys cannot be <= 0")
        self.num_components = len(keys)
        self.component_sizes = {k: 1 for k in keys}
        self.id = {k: k for k in keys}

    def find(self, p):
        root = p
        while root != self.id[root]:
            root = self.id[root]

        while p != root:
            next = self.id[p]
            self.id[p] = root
            p = next

        return root

    def union(self, p, q):
        root1 = self.find(p)
        root2 = self.find(q)

        if root1 == root2:
            return

        if self.component_sizes[root1] < self.component_sizes[root2]:
            self.component_sizes[root2] += self.component_sizes[root1]
            self.id[root1] = root2
            self.component_sizes[root1] = 0
        else:
            self.component_sizes[root1] += self.component_sizes[root2]
            self.id[root2] = root1
            self.component_sizes[root2] = 0

        self.num_components -= 1
