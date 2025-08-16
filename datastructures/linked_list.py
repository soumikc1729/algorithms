class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def deserialize_linked_list(vals):
    if not vals:
        return None

    head = ListNode(vals[0])
    ptr = head
    for val in vals[1:]:
        ptr.next = ListNode(val)
        ptr = ptr.next

    return head


def serialize_linked_list(head):
    if not head:
        return []

    arr = []
    ptr = head
    while ptr:
        arr.append(ptr.val)
        ptr = ptr.next

    return arr
