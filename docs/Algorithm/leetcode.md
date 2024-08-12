# LeetCode

## 1. Two Sum

!!! Note "算法"

    哈希

从 $O(n^2)$ 优化到 $O(n)$

利用 Hash Map 的插入, 检索的时间复杂度 $O(1)$ 以及数学中的 Complement 来优化算法.

1. 创建 Hash Map 存放 `{value: index}`
2. 计算 `complement = target - value`
3. 如果 `complement` 存在于 Hash Table 中, 则返回 `[hash_map[complement], i]`

上面的算法可以实现仅对输入的数组进行一次迭代即可完成.

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 创建哈希表
        hash_map = {}
        # 迭代数组
        for i, v in enumerate(nums):
            # 计算差值. 也叫 "补数"
            complement = target - v
            # 补数存在于哈希表中, 表示找到了结果
            if complement in hash_map:
                # 返回结果
                return [hash_map[complement], i]
            # 补数不存在于哈希表, 则创建
            hash_map[v] = i
```

## 26. Remove Duplicates from Sorted Array

!!! Note "算法"

    双指针

首先需要理解 **non-decreasing order** 的含义, 这是一个数学和计算机科学的专业术语, **非严格递增排序** 指的是数据从小到大排序且允许重复值的出现, 例如: `[0, 0, 1, 2, 2, 3]`.

双指针, 一个 slow point, 一个 fast point, fast pint 用于迭代数组元素, slow point 用于跟踪需要覆盖的数组索引.

```python
class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        # 慢指针
        slow = 0
        # 迭代
        for i in range(1, len(nums)):
            # 当前值不等于前一个值
            if nums[i] != nums[i - 1]:
                # 慢指针向前移动 1 个位置
                slow += 1
                # 当前值写入慢指针位置
                nums[slow] = nums[i]

        # 返回慢指针位置 +1
        # +1 表示移动了几次, 因为慢指针从 0 开始, 也就是有多少个重复值
        return slow + 1
```

## 141. Linked List Cycle

!!! Note "算法"

    双指针. 慢指针(一次一步), 快指针(一次两步).

最直观的想法就是沿着链表迭代每个节点, 然后将节点值放入 `Set` 中, 利用其哈希特性实现 $O(1)$ 的查找效率, 从而是的整个事件复杂度变为 $O(1)$. 但是, 此种方法会导致空间复杂度为 $O(n)$, 因此这并不是一个好的解决方案.

通常此类问题属于链表环检测, 在数学中这属于 **同余原理** 的问题, 简单来说, 在一个环中走的快的一定会在走的慢的的一个周期内相遇. 比如: 钟表, 时针一定会在12小时(一圈)内与分钟相遇一次. 因此这道题就采用双指针来进行解答.

```python

class ListNode:
    def __init__(self, x: int) -> None:
        self.val = x
        self.next: ListNode | None = None


class Solution:
    def hasCycle(self, head: ListNode | None) -> bool:
        if head is None:
            return False

        # 初始化慢指针, 指向链表头
        slow = head
        # 初始化快指针, 指向链表头
        fast = head

        # 迭代链表, 因为快指针快, 所以使用快指针作为判断依据
        while fast and fast.next:
            # 慢指针走一步
            slow = slow.next
            # 快指针走两步
            fast = fast.next.next
            # 快指针等于慢指针
            if slow == fast:
                # 表示有环出现, 返回
                return True

        # 遍历链表发现快指针或快指针的 next 指针为空, 表示没有环出现
        return False
```

## 206. Reverse Linked List

!!! Note "算法"

    双指针. 一个指向上前驱节点, 一个指向当前节点

```python
class ListNode:
    def __init__(self, x: int) -> None:
        self.val = x
        self.next: ListNode | None = None


class Solution:
    def reverseList(self, head: ListNode | None) -> ListNode | None:
        # 边界检查
        if head is None or head.next is None:
            return head

        # 初始化. 前驱节点指针
        prev = None
        # 初始化. 当前节点指针
        current = head

        # 遍历链表
        while current:
            # 存储. 后驱节点指针
            next_node = current.next
            # 反转当前节点
            current.next = prev
            # 移动前驱节点指针到当前节点
            prev = current
            # 移动当前节点指针到后驱节点
            current = next_node

        # 返回
        return prev
```
