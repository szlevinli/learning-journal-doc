# LeetCode

## 1. Two Sum

!!! Tip

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

!!! Tip

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

!!! Tip

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

!!! Tip

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

## 21. Merge Two Sorted Lists

!!! Tip

    双指针合并算法 (Two-Pointer Merge Algorithm).

!!! Quote 一开始的思路

    - 处理边界.
        - list1 or list2 is None
        - list1.next or list2.next is None
    - 初始化两个指针分别指向两个链表的头部. p1, p2
    - 遍历 list1
        - p2 is None, 表示 list2 已到底, 将 list1 append 到 list2 之后, 终止遍历 返回 list2
        - p1.val >= p2.val, 删除 list1 当前节点并将该节点插入 list2 当前节点之后
        - p1.val < p2.val, 删除 list1 当前节点并将该节点插入 list2 当前节点之前
    - 返回 list2

**上面的思路有问题, 重新梳理后的**

- 边界处理
  - list1 is None, return list2
  - list2 is None, return list1
- 初始化
  - 创建一个虚拟节点 dummy
  - 创建一个 current 指针指向 dummy
- 合并链表
  - 使用 `while` 循环遍历两个链表, 直到其中一个到达尾部
  - 比较两个链表当前节点值, 将较小值的节点接到 current 之后, 并将相应链表的指针前进一位
  - 将 current 指针移动到新添加的节点
- 处理剩余节点
  - 循环结束后, 如果还有未处理的节点, 直接将这些节点接到 current 之后
- 返回合并后的链表:
  - 返回 `dummy.next`, 因为 `dummy` 是一个虚拟头节点.

```python
class ListNode:
    def __init__(self, val=0, next=None) -> None:
        self.val = val
        self.next: ListNode | None = next


class Solution:
    def mergeTwoList(
        self, list1: ListNode | None, list2: ListNode | None
    ) -> ListNode | None:
        # 创建虚拟节点
        dummy = ListNode()
        # 创建 "当前" 指针指向虚拟节点
        current = dummy

        # 遍历 list 和 list2 直到其中任意一个到达尾部
        while list1 and list2:
            # list1 当前节点值 小于等于 list2 当前节点值
            if list1.val <= list2.val:
                # list1 接到 current 之后
                current.next = list1
                # list1 向前一步
                list1 = list1.next
            else:
                # list2 接到 current 之后
                current.next = list2
                # list2 向前一步
                list2 = list2.next
            # current 移动到新添加的节点上
            current = current.next

        # 处理剩余节点
        current.next = list1 if list1 is not None else list2

        # 返回
        return dummy.next
```

## 234. Palindrome Linked List

!!! Tip

    双指针. 快慢指针. 快速定位链表中心点.

这里首先要说明, 在 LeetCode 的官方题解中它使用将链表值转存到数组中然后反转数组进行比较. 但是, 这种解法虽然简单, 但是空间复杂度非常不理想, 所以下面将采用双指针的快慢指针来找到链表的中心点, 并对后半部分链表进行原地反转, 最后前后两部分链表节点一一比对. 这种双指针的算法空间复杂度可以降至 $O(1)$, 时间复杂度仍然保持 $O(n)$.

双指针的快慢指针有一个核心点, 可以在 $O(\frac{2}{n})$ 时间内找到中心点, 快指针一次两步, 慢指针一次一步, 当快指针抵达链表尾部时, 慢指针所在位置即为链表的中心点. 其中奇数链表慢指针所指节点为中心点, 偶数链表慢指针所指节点为上半部链表的尾节点.

同时, 我们还能根据快指针 (`fast`) 来判断原链表的奇偶性, `fast is None` 表示链表是偶数链表, `fast.next is None` 表示链表是奇数链表.

!!! Warning 注意

    实际上原链表的奇偶性并不影响整个代码, 换句话说, 无需考虑奇偶性.

    例如: 
    
    - `[1, 2, 3, 4, 5]` 后半部分的链表是 `[5, 4, 3]`, 前半部分链表是 `[1, 2, 3, 4, 5]`
    - `[1, 2, 3, 4]` 后半部分的链表是 `[4, 3]`, 前半部分链表是 `[1, 2, 3, 4]`

    前半部分链表实际是原链表

```python
class ListNode:
    def __init__(self, val=0, next=None) -> None:
        self.val = val
        self.next: ListNode | None = next


class Solution:
    def isPalindrome(self, head: ListNode | None) -> bool:
        # None 也是回文
        if head is None:
            return True

        # 使用快慢指针找到链表中心点
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # 反转后半部分链表
        prev = None
        while slow:
            next_node = slow.next
            slow.next = prev
            prev = slow
            slow = next_node

        # 比较两部分链表
        left = head
        right = prev
        # 遍历后半部分.
        while right:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next

        return True
```
