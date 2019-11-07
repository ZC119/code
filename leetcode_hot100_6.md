
## 19. 删除链表的倒数第N个节点（中等）双指针

给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。

示例：

    给定一个链表: 1->2->3->4->5, 和 n = 2.

    当删除了倒数第二个节点后，链表变为 1->2->3->5.
说明：

给定的 n 保证是有效的。

进阶：

你能尝试使用一趟扫描实现吗？


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        cur = dummy

        for i in range(n+1):
            cur = cur.next

        cur2 = dummy
        pre = None
        while cur:
            cur = cur.next
            pre = cur2
            cur2 = cur2.next

        if cur2 and cur2.next:
            cur2.next = cur2.next.next

        return dummy.next
```

## 21. 合并两个有序链表（简单）

将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

示例：

    输入：1->2->4, 1->3->4
    输出：1->1->2->3->4->4


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        cur = dummy
        p1 = l1
        p2 = l2

        while p1 and p2:
            if p1.val <= p2.val:
                cur.next = p1
                p1 = p1.next
            else:
                cur.next = p2
                p2 = p2.next
            cur = cur.next
        
        if p1:
            cur.next = p1
        if p2:
            cur.next = p2

        return dummy.next
```

## 23. 合并K个排序链表（困难）优先队列

合并 k 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。

示例:

    输入:
    [
      1->4->5,
      1->3->4,
      2->6
    ]
    输出: 1->1->2->3->4->4->5->6


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        import heapq
        dummy = ListNode(0)
        p = dummy
        head = []

        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(head, (lists[i].val, i))
                lists[i] = lists[i].next
        
        while head:
            val, idx = heapq.heappop(head)
            p.next = ListNode(val)
            p = p.next

            if lists[idx]:
                heapq.heappush(head, (lists[idx].val, idx))
                lists[idx] = lists[idx].next
        
        return dummy.next
```

## 148. 排序链表（中等）归并排序

在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序。

示例 1:

    输入: 4->2->1->3
    输出: 1->2->3->4
示例 2:

    输入: -1->5->3->4->0
    输出: -1->0->3->4->5

由于题目要求空间复杂度是 O(1)，因此不能使用递归。因此这里使用 bottom-to-up 的算法来解决。

bottom-to-up 的归并思路是这样的：先两个两个的 merge，完成一趟后，再 4 个4个的 merge，直到结束。举个简单的例子：[4,3,1,7,8,9,2,11,5,6].

    step=1: (3->4)->(1->7)->(8->9)->(2->11)->(5->6)
    step=2: (1->3->4->7)->(2->8->9->11)->(5->6)
    step=4: (1->2->3->4->7->8->9->11)->5->6
    step=8: (1->2->3->4->5->6->7->8->9->11)

- merge(l1, l2)，双路归并，我相信这个操作大家已经非常熟练的，就不做介绍了。
- cut(l, n)，可能有些同学没有听说过，它其实就是一种 split 操作，即断链操作。不过我感觉使用 cut 更准确一些，它表示，将链表 l 切掉前 n 个节点，并返回后半部分的链表头。
- 额外再补充一个 dummyHead 大法

伪代码

```cpp
current = dummy.next;
tail = dummy;
for (step = 1; step < length; step *= 2) {
	while (current) {
		// left->@->@->@->@->@->@->null
		left = current;

		// left->@->@->null   right->@->@->@->@->null
		right = cut(current, step); // 将 current 切掉前 step 个头切下来。

		// left->@->@->null   right->@->@->null   current->@->@->null
		current = cut(right, step); // 将 right 切掉前 step 个头切下来。
		
		// dummy.next -> @->@->@->@->null，最后一个节点是 tail，始终记录
		//                        ^
		//                        tail
		tail.next = merge(left, right);
		while (tail->next) tail = tail->next; // 保持 tail 为尾部
	}
}
```


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def cut(self, head, step):
        for i in range(step-1):
            if not head:
                break
            head = head.next

        if not head:
            return None
        newhead = head.next
        head.next = None
        return newhead
    
    def merge(self, left, right):
        p1 = left
        p2 = right
        dummy = ListNode(0)
        cur = dummy

        while p1 and p2:
            if p1.val <= p2.val:
                cur.next = p1
                p1 = p1.next
            else:
                cur.next = p2
                p2 = p2.next
            cur = cur.next

        if p1:
            cur.next = p1
        if p2:
            cur.next = p2

        return dummy.next
            

    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head

        length = 0
        
        temp = head
        while temp:
            length += 1
            temp = temp.next
        
        step = 1
        while step < length:
            current = dummy.next
            tail = dummy
            
            while current:
            
                left = current

                right = self.cut(left, step)

                current = self.cut(right, step)

                tail.next = self.merge(left, right)

                while tail.next:
                    tail = tail.next
            
            step *= 2
        return dummy.next
```

## 160. 相交链表（简单）双指针

编写一个程序，找到两个单链表相交的起始节点。
 

示例 1：



    输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
    输出：Reference of the node with value = 8
    输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
 

示例 2：



    输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
    输出：Reference of the node with value = 2
    输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。


示例 3：



    输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
    输出：null
    输入解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
    解释：这两个链表不相交，因此返回 null。
 

注意：

    如果两个链表没有交点，返回 null.
    在返回结果后，两个链表仍须保持原有的结构。
    可假定整个链表结构中没有循环。
    程序尽量满足 O(n) 时间复杂度，且仅用 O(1) 内存。


创建两个指针 pA 和 pB，分别初始化为链表 A 和 B 的头结点。然后让它们向后逐结点遍历。
当 pA 到达链表的尾部时，将它重定位到链表 B 的头结点 (你没看错，就是链表 B); 类似的，当 pB 到达链表的尾部时，将它重定位到链表 A 的头结点。
若在某一时刻 pA 和 pB 相遇，则 pA/pB 为相交结点。


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        p1 = headA
        p2 = headB

        while p1 != p2:
            if p1:
                p1 = p1.next
            else:
                p1 = headB
            
            if p2:
                p2 = p2.next
            else:
                p2 = headA
        
        return p1
```

## 94. 二叉树的中序遍历（中等）栈

给定一个二叉树，返回它的中序 遍历。

示例:

    输入: [1,null,2,3]
       1
        \
         2
        /
       3

    输出: [1,3,2]
进阶: 递归算法很简单，你可以通过迭代算法完成吗？


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        stack = []
        cur = root
        result = []

        while cur or stack:
            while cur:
                stack.append(cur)
                cur = cur.left
            
            cur = stack.pop()
            result.append(cur.val)
            cur = cur.right

        return result
```

## 98. 验证二叉搜索树（中等）先序遍历

给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：

节点的左子树只包含小于当前节点的数。
节点的右子树只包含大于当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。  
示例 1:

    输入:
        2
       / \
      1   3
    输出: true
示例 2:

    输入:
        5
       / \
      1   4
         / \
        3   6
    输出: false
    解释: 输入为: [5,1,4,null,null,3,6]。
         根节点的值为 5 ，但是其右子节点值为 4 。


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def __init__(self):
        self.result = []

    def preOrder(self, root):
        if not root:
            return True

        left = self.preOrder(root.left)

        if self.result:
            if self.result[-1] >= root.val:
                return False

        self.result.append(root.val)

        right = self.preOrder(root.right)

        return left and right
 

    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.preOrder(root)
```

## 101. 对称二叉树（简单）dfs

给定一个二叉树，检查它是否是镜像对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

        1
       / \
      2   2
     / \ / \
    3  4 4  3
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

        1
       / \
      2   2
       \   \
       3    3
说明:

如果你可以运用递归和迭代两种方法解决这个问题，会很加分。


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isMirror(self, t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False

        return t1.val == t2.val and self.isMirror(t1.left, t2.right) and self.isMirror(t1.right, t2.left)

    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.isMirror(root, root)
```

## 102. 二叉树的层次遍历（中等）bfs

给定一个二叉树，返回其按层次遍历的节点值。 （即逐层地，从左到右访问所有节点）。

例如:
给定二叉树: [3,9,20,null,null,15,7],

        3
       / \
      9  20
        /  \
       15   7
返回其层次遍历结果：

    [
      [3],
      [9,20],
      [15,7]
    ]


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []

        result = [[]]

        stack = [(root, 0)]

        while stack:
            cur, hight = stack.pop(0)
            if cur.left:
                stack.append((cur.left, hight+1))
            if cur.right:
                stack.append((cur.right, hight+1))

            if len(result) <= hight:
                result.append([])
            result[hight].append(cur.val)

        return result
```

## 104. 二叉树的最大深度（简单）

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

示例：
给定二叉树 [3,9,20,null,null,15,7]，

        3
       / \
      9  20
        /  \
       15   7
    返回它的最大深度 3 。



```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0

        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

## 105. 从前序与中序遍历序列构造二叉树（中等）

根据一棵树的前序遍历与中序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

例如，给出

前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：

        3
       / \
      9  20
        /  \
       15   7


```python

```

## 114. 二叉树展开为链表（中等）先序遍历

给定一个二叉树，原地将它展开为链表。

例如，给定二叉树

        1
       / \
      2   5
     / \   \
    3   4   6
将其展开为：

    1
     \
      2
       \
        3
         \
          4
           \
            5
             \
              6
              
可以发现展开的顺序其实就是二叉树的先序遍历。算法和 94 题 中序遍历的 Morris 算法有些神似，我们需要两步完成这道题。

    将左子树插入到右子树的地方
    将原来的右子树接到左子树的最右边节点
    考虑新的右子树的根节点，一直重复上边的过程，直到新的右子树为 null


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        cur = root
        while cur:
            tmp = cur.right
            cur.right = cur.left
            cur.left = None
            #cur.right最右子节点
            rightnode = cur
            curright = cur
            while curright.right:
                rightnode = curright.right
                curright = curright.right
            rightnode.right = tmp
            cur = cur.right
```

## 11. 盛最多水的容器（中等）双指针

给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

说明：你不能倾斜容器，且 n 的值至少为 2。



图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。


示例:

输入: [1,8,6,2,5,4,8,3,7]
输出: 49


```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        i = 0
        j = len(height) - 1

        result = 0

        while i < j:
            result = max((j-i)*min(height[i], height[j]), result)
            if height[i] <= height[j]:
                i += 1
            else:
                j -= 1
        
        return result
```

## 17. 电话号码的字母组合（中等）

给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。



示例:

    输入："23"
    输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].



```python
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []

        result = ['']

        dic = {'2':'abc', '3':'def', '4':'ghi', '5':'jkl', '6':'mno', '7':'pqrs', '8':'tuv', '9':'wxyz'}

        for digit in digits:
            string = dic[digit]
            newres = []

            for s in string:
                for res in result:
                    newres.append(res+s)
            result = newres

        return result
            
```

## 10. 正则表达式匹配（困难）

给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素
所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。

说明:

s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 . 和 *。
示例 1:

    输入:
    s = "aa"
    p = "a"
    输出: false
    解释: "a" 无法匹配 "aa" 整个字符串。
示例 2:

    输入:
    s = "aa"
    p = "a*"
    输出: true
    解释: 因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
示例 3:

    输入:
    s = "ab"
    p = ".*"
    输出: true
    解释: ".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
示例 4:

    输入:
    s = "aab"
    p = "c*a*b"
    输出: true
    解释: 因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
示例 5:

    输入:
    s = "mississippi"
    p = "mis*is*p*."
    输出: false



```python

```

## 136. 只出现一次的数字（简单）

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

示例 1:

    输入: [2,2,1]
    输出: 1
示例 2:

    输入: [4,1,2,1,2]
    输出: 4



```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        result = nums[0]

        for num in nums[1:]:
            result ^= num
        
        return result
```

## 79. 单词搜索（中等）dfs 类似200

给定一个二维网格和一个单词，找出该单词是否存在于网格中。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

示例:

    board =
    [
      ['A','B','C','E'],
      ['S','F','C','S'],
      ['A','D','E','E']
    ]

    给定 word = "ABCCED", 返回 true.
    给定 word = "SEE", 返回 true.
    给定 word = "ABCB", 返回 false.


```python
class Solution(object):
    def help(self, board, word, i, j, m, n, idx, marked):
        if idx == len(word) - 1:
            return word[idx] == board[i][j]

        direction = [[-1, 0], [1, 0], [0, 1], [0, -1]]

        if word[idx] == board[i][j]:
            marked[i][j] = True
            for direc in direction:
                new_i = i + direc[0]
                new_j = j + direc[1]

                if 0 <= new_i < m and 0 <= new_j < n and not marked[new_i][new_j] and self.help(board, word, new_i, new_j, m, n, idx+1, marked):
                    return True
            marked[i][j] = False

        return False

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        m = len(board)
        if m == 0:
            return False
        n = len(board[0])

        marked = [[False for _ in range(n)] for _ in range(m)]

        idx = 0
        for i in range(m):
            for j in range(n):
                if self.help(board, word, i, j, m, n, idx, marked):
                    return True
        return False
```

## 78. 子集（中等）递归+回溯

给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

说明：解集不能包含重复的子集。

示例:

    输入: nums = [1,2,3]
    输出:
    [
      [3],
      [1],
      [2],
      [1,2,3],
      [1,3],
      [2,3],
      [1,2],
      []
    ]


```python
class Solution(object):
    
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        n = len(nums)

        def help(temp, i):
            result.append(temp)

            for j in range(i, n):
                help(temp+[nums[j]], j+1)

        help([], 0)
        return result
```

## 75. 颜色分类（中等）

给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

注意:
不能使用代码库中的排序函数来解决这道题。

示例:

    输入: [2,0,2,1,1,0]
    输出: [0,0,1,1,2,2]
进阶：

一个直观的解决方案是使用计数排序的两趟扫描算法。
首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。
你能想出一个仅使用常数空间的一趟扫描算法吗？

本解法的思路是沿着数组移动 curr 指针，若nums[curr] = 0，则将其与 nums[p0]互换；若 nums[curr] = 2 ，则与 nums[p2]互换。

算法

    初始化0的最右边界：p0 = 0。在整个算法执行过程中 nums[idx < p0] = 0.

    初始化2的最左边界 ：p2 = n - 1。在整个算法执行过程中 nums[idx > p2] = 2.

    初始化当前考虑的元素序号 ：curr = 0.

    While curr <= p2 :

    若 nums[curr] = 0 ：交换第 curr个 和 第p0个 元素，并将指针都向右移。

    若 nums[curr] = 2 ：交换第 curr个和第 p2个元素，并将 p2指针左移 。

    若 nums[curr] = 1 ：将指针curr右移。
    
因为curr左边的值已经扫描过了，所以curr要++继续扫描下一位，而与p2交换的值，curr未扫描，要停下来扫描一下，所以curr不用++。


```python
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        p0 = 0
        p2 = len(nums) - 1
        curr = 0

        while curr <= p2:
            if nums[curr] == 0:
                nums[p0], nums[curr] = nums[curr], nums[p0]
                curr += 1
                p0 += 1
            elif nums[curr] == 2:
                nums[p2], nums[curr] = nums[curr], nums[p2]
                p2 -= 1
            elif nums[curr] == 1:
                curr += 1
```

## 56. 合并区间（中等）

给出一个区间的集合，请合并所有重叠的区间。

示例 1:

    输入: [[1,3],[2,6],[8,10],[15,18]]
    输出: [[1,6],[8,10],[15,18]]
    解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
示例 2:

    输入: [[1,4],[4,5]]
    输出: [[1,5]]
    解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。

## 55. 跳跃游戏（中等）贪心法

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个位置。

示例 1:

    输入: [2,3,1,1,4]
    输出: true
    解释: 从位置 0 到 1 跳 1 步, 然后跳 3 步到达最后一个位置。
示例 2:

    输入: [3,2,1,0,4]
    输出: false
    解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。
    
### 动态规划

```python
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        dp = [True] + [False] * (len(nums) - 1)

        for i in range(1, len(nums)):
            for j in range(1, i+1):
                if dp[i-j] and nums[i-j] >= j:
                    dp[i] = True
                    break

        return dp[-1]
```

O(n^2)，超时

### 贪心法

下面我们使用贪心的思路看下这个问题，我们记录一个的坐标代表当前可达的最后节点，这个坐标初始等于nums.length-1，
然后我们每判断完是否可达，都向前移动这个坐标，直到遍历结束。

如果这个坐标等于0，那么认为可达，否则不可达。


```python
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        lastposition = len(nums) - 1

        for i in range(len(nums)-1, -1, -1):
            if nums[i] + i >= lastposition:
                lastposition = i
        
        return lastposition == 0
```

## 49. 字母异位词分组（中等）哈希表

给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

示例:

    输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
    输出:
    [
      ["ate","eat","tea"],
      ["nat","tan"],
      ["bat"]
    ]
说明：

所有输入均为小写字母。
不考虑答案输出的顺序。


```python
from collections import Counter

class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        word = []
        result = []
        index = 0

        for s in strs:
            if Counter(s) in word:              
                result[word.index(Counter(s))].append(s)
            else:
                word.append(Counter(s))
                result.append([s])

        return result
```

## 46. 全排列（中等）回溯算法

给定一个没有重复数字的序列，返回其所有可能的全排列。

示例:

    输入: [1,2,3]
    输出:
    [
      [1,2,3],
      [1,3,2],
      [2,1,3],
      [2,3,1],
      [3,1,2],
      [3,2,1]
    ]
    
    
方法：“回溯搜索”算法即“深度优先遍历 + 状态重置 + 剪枝”（这道题没有剪枝）

这里我们介绍什么是“状态”。

在递归树里，辅助数组 used 记录的情况和当前已经选出数组成的一个排序，我们统称为当前的“状态”。

下面解释“状态重置”。

在程序执行到上面这棵树的叶子结点的时候，此时递归到底，当前根结点到叶子结点走过的路径就构成一个全排列，把它加入结果集，我把这一步称之为“结算”。此时递归方法要返回了，对于方法返回以后，要做两件事情：

（1）释放对最后一个数的占用；
（2）将最后一个数从当前选取的排列中弹出。

事实上在每一层的方法执行完毕，即将要返回的时候都需要这么做。这棵树上的每一个结点都会被访问 2 次，绕一圈回到第 1 次来到的那个结点，第 2 次回到结点的“状态”要和第 1 次来到这个结点时候的“状态”相同，这种程序员赋予程序的操作叫做“状态重置”。

“状态重置”是“回溯”的重要操作，“回溯搜索”是有方向的搜索，否则我们要写多重循环，代码量不可控。



```python
class Solution(object):
    def __dfs(self, nums, index, pre, used, res):
        if index == len(nums):
            res.append(pre[:])
            return 
        
        for i in range(len(nums)):
            if not used[i]:
                pre.append(nums[i])
                used[i] = True

                self.__dfs(nums, index+1, pre, used, res)

                used[i] = False
                pre.pop()



    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) == 0:
            return []


        used = [False] * len(nums)

        pre = []

        res = []

        self.__dfs(nums, 0, pre, used, res)

        return res
```

## 39. 组合总和（中等）回溯

给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的数字可以无限制重复被选取。

说明：

所有数字（包括 target）都是正整数。
解集不能包含重复的组合。 
示例 1:

    输入: candidates = [2,3,6,7], target = 7,
    所求解集为:
    [
      [7],
      [2,2,3]
    ]
示例 2:

    输入: candidates = [2,3,5], target = 8,
    所求解集为:
    [
      [2,2,2,2],
      [2,3,3],
      [3,5]
    ]


```python
class Solution(object):
    def __dfs(self, candidates, index, pre, target, res):
        for i, candidate in enumerate(candidates):
            if i < index:
                continue
            if candidate > target:
                return
            elif candidate == target:
                res.append(pre[:]+[candidate])
                return 
            else:
                pre.append(candidate)
                self.__dfs(candidates, i, pre, target-candidate, res)
                pre.pop()



    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        pre = []
        index = 0

        self.__dfs(sorted(candidates), index, pre, target, res)

        return res
```

## 34. 在排序数组中查找元素的第一个和最后一个位置（中等）二分查找

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

你的算法时间复杂度必须是 O(log n) 级别。

如果数组中不存在目标值，返回 [-1, -1]。

示例 1:

    输入: nums = [5,7,7,8,8,10], target = 8
    输出: [3,4]
示例 2:

    输入: nums = [5,7,7,8,8,10], target = 6
    输出: [-1,-1]


```python
class Solution(object):
    def searchLeft(self, nums, target):

        left = 0
        right = len(nums)

        while left < right:
            mid = (left + right) // 2
            if nums[mid] > target:
                right = mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        if left == len(nums):
            return -1
        if nums[left] == target:
            return left
        else:
            return -1

    def searchRight(self, nums, target):

        left = 0
        right = len(nums)

        while left < right:
            mid = (left + right) // 2
            if nums[mid] > target:
                right = mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                left = mid + 1

        if left == 0:
            return -1
        if nums[left-1] == target:
            return left-1
        else:
            return -1

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        return [self.searchLeft(nums, target), self.searchRight(nums, target)]
```

## 33. 搜索旋转排序数组（中等）

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

你可以假设数组中不存在重复的元素。

你的算法时间复杂度必须是 O(log n) 级别。

示例 1:

    输入: nums = [4,5,6,7,0,1,2], target = 0
    输出: 4
示例 2:

    输入: nums = [4,5,6,7,0,1,2], target = 3
    输出: -1


```python

```

## 31. 下一个排列（中等）

实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须原地修改，只允许使用额外常数空间。

以下是一些例子，输入位于左侧列，其相应输出位于右侧列。

    1,2,3 → 1,3,2
    3,2,1 → 1,2,3
    1,1,5 → 1,5,1


```python
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        find = False
        maxk = 0
        for k in range(len(nums)-1):
            if nums[k] < nums[k+1]:
                find = True
                maxk = k

        if not find:
            nums.reverse()
            return 

        maxl = 0
        for l in range(maxk+1, len(nums)):
            if nums[l] > nums[maxk]:
                maxl = l

        nums[maxk], nums[maxl] = nums[maxl], nums[maxk]

        i = maxk+1
        j = len(nums) - 1
        while i <= j:
            nums[j], nums[i] = nums[i], nums[j]
            i += 1
            j -= 1
```

## 22. 括号生成（中等）回溯

给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。

例如，给出 n = 3，生成结果为：

    [
      "((()))",
      "(()())",
      "(())()",
      "()(())",
      "()()()"
    ]


```python
class Solution(object):
    def __init__(self):
        self.pre = ''

    def __dfs(self, left, right, n, res, idx):
        if left == right and left == n:
            res.append(self.pre)
            return 
        
        if left < right:
            return 

        if idx == 1:
            # n-left, n-right 剩下可生成数
            for l in range(1, n-left+1):
                self.pre = self.pre+'('*l
                self.__dfs(left+l, right, n, res, 2)
                self.pre = self.pre[:-l]
        elif idx == 2:
            for r in range(1, n-right+1):
                self.pre = self.pre+')'*r
                self.__dfs(left, right+r, n, res, 1)
                self.pre = self.pre[:-r]

        
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        left = 0
        right = 0
        res = []
        idx = 1

        self.__dfs(left, right, n, res, idx)

        return res
```

## 20. 有效的括号（简单）

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。

示例 1:

    输入: "()"
    输出: true
示例 2:

    输入: "()[]{}"
    输出: true
示例 3:

    输入: "(]"
    输出: false


```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        for l in s:
            if l in '([{':
                stack.append(l)
            else:
                if not stack:
                    return False
                if l == '}' and stack[-1] == '{' or l == ']' and stack[-1] == '[' or l == ')' and stack[-1] == '(':
                    stack.pop()
                else:
                    return False
        
        if not stack:
            return True
        
        return False
```

## 155. 最小栈（简单）

设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。

    push(x) -- 将元素 x 推入栈中。
    pop() -- 删除栈顶的元素。
    top() -- 获取栈顶元素。
    getMin() -- 检索栈中的最小元素。
示例:

    MinStack minStack = new MinStack();
    minStack.push(-2);
    minStack.push(0);
    minStack.push(-3);
    minStack.getMin();   --> 返回 -3.
    minStack.pop();
    minStack.top();      --> 返回 0.
    minStack.getMin();   --> 返回 -2.


```python
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.minstack = []
        

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.stack.append(x)
        if self.minstack:
            if x > self.minstack[-1]:
                self.minstack.append(self.minstack[-1])
            else:
                self.minstack.append(x)
        else:
            self.minstack.append(x)
        
        

    def pop(self):
        """
        :rtype: None
        """
        self.stack.pop()
        self.minstack.pop()
        

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        if not self.minstack:
            return None
        return self.minstack[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

## 206. 反转链表（简单）

反转一个单链表。

示例:

    输入: 1->2->3->4->5->NULL
    输出: 5->4->3->2->1->NULL
进阶:
你可以迭代或递归地反转链表。你能否用两种方法解决这道题？


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        cur = head
        pre = None

        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp

        return pre
```

## 146. LRU缓存机制（中等）哈希链表

运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制。它应该支持以下操作： 获取数据 get 和 写入数据 put 。

获取数据 get(key) - 如果密钥 (key) 存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。
写入数据 put(key, value) - 如果密钥不存在，则写入其数据值。当缓存容量达到上限时，它应该在写入新数据之前删除最近最少使用的数据值，从而为新的数据值留出空间。

进阶:

你是否可以在 O(1) 时间复杂度内完成这两种操作？

示例:

    LRUCache cache = new LRUCache( 2 /* 缓存容量 */ );

    cache.put(1, 1);
    cache.put(2, 2);
    cache.get(1);       // 返回  1
    cache.put(3, 3);    // 该操作会使得密钥 2 作废
    cache.get(2);       // 返回 -1 (未找到)
    cache.put(4, 4);    // 该操作会使得密钥 1 作废
    cache.get(1);       // 返回 -1 (未找到)
    cache.get(3);       // 返回  3
    cache.get(4);       // 返回  4


```python

```

## 5. 最长回文子串（中等）动态规划

给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

示例 1：

    输入: "babad"
    输出: "bab"
    注意: "aba" 也是一个有效答案。
示例 2：

    输入: "cbbd"
    输出: "bb"
    
1、定义 “状态”；  
2、找到 “状态转移方程”。

记号说明： 下文中，使用记号 s[l, r] 表示原始字符串的一个子串，l、r 分别是区间的左右边界的索引值，使用左闭、右闭区间表示左右边界可以取到。举个例子，当 s = 'babad' 时，s[0, 1] = 'ba' ，s[2, 4] = 'bad'。

1、定义 “状态”，这里 “状态”数组是二维数组。

dp[l][r] 表示子串 s[l, r]（包括区间左右端点）是否构成回文串，是一个二维布尔型数组。即如果子串 s[l, r] 是回文串，那么 dp[l][r] = true。

2、找到 “状态转移方程”。

首先，我们很清楚一个事实：

1、当子串只包含 1 个字符，它一定是回文子串；

2、当子串包含 2 个以上字符的时候：如果 s[l, r] 是一个回文串，例如 “abccba”，那么这个回文串两边各往里面收缩一个字符（如果可以的话）的子串 s[l + 1, r - 1] 也一定是回文串，即：如果 dp[l][r] == true 成立，一定有 dp[l + 1][r - 1] = true 成立。
根据这一点，我们可以知道，给出一个子串 s[l, r] ，如果 s[l] != s[r]，那么这个子串就一定不是回文串。如果 s[l] == s[r] 成立，就接着判断 s[l + 1] 与 s[r - 1]，这很像中心扩散法的逆方法。

事实上，当 s[l] == s[r] 成立的时候，dp[l][r] 的值由 dp[l + 1][r - l] 决定，这一点也不难思考：当左右边界字符串相等的时候，整个字符串是否是回文就完全由“原字符串去掉左右边界”的子串是否回文决定。但是这里还需要再多考虑一点点：“原字符串去掉左右边界”的子串的边界情况。

1、当原字符串的元素个数为 3 个的时候，如果左右边界相等，那么去掉它们以后，只剩下 1 个字符，它一定是回文串，故原字符串也一定是回文串；

2、当原字符串的元素个数为 2 个的时候，如果左右边界相等，那么去掉它们以后，只剩下 0 个字符，显然原字符串也一定是回文串。
把上面两点归纳一下，只要 s[l + 1, r - 1] 至少包含两个元素，就有必要继续做判断，否则直接根据左右边界是否相等就能得到原字符串的回文性。而“s[l + 1, r - 1] 至少包含两个元素”等价于 l + 1 < r - 1，整理得 l - r < -2，或者 r - l > 2。

综上，如果一个字符串的左右边界相等，以下二者之一成立即可：
1、去掉左右边界以后的字符串不构成区间，即“ s[l + 1, r - 1] 至少包含两个元素”的反面，即 l - r >= -2，或者 r - l <= 2；
2、去掉左右边界以后的字符串是回文串，具体说，它的回文性决定了原字符串的回文性。

于是整理成“状态转移方程”：

dp[l, r] = (s[l] == s[r] and (l - r >= -2 or dp[l + 1, r - 1]))
或者

dp[l, r] = (s[l] == s[r] and (r - l <= 2 or dp[l + 1, r - 1]))


```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(s) <= 1:
            return s
        result = s[0]
        n = len(s)
        # dp[i][j] 保存到i到j是否为回文
        dp = [[False for _ in range(n)] for _ in range(n)]

        for r in range(1, len(s)):
            for l in range(r):
                if s[l] == s[r] and (r-l <= 2 or dp[l+1][r-1]):
                    dp[l][r] = True
                    if len(result) < r-l+1:
                        result = s[l:r+1]

        return result
```


```python

```
