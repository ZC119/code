## 739. 每日温度（中等）
根据每日 气温 列表，请重新生成一个列表，对应位置的输入是你需要再等待多久温度才会升高超过该日的天数。如果之后都不会升高，请在该位置用 0 来代替。

例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。

提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。

idea：**递减栈**。栈中只存放递减序列。每次入栈，如果当前温度大于栈顶温度，表明栈顶元素第一次遇到了温度大于它的时候，则栈顶元素出栈，计算相差天数即为栈顶温度对应的结果，如果当前温度小于栈顶温度，则入栈。


```python
class Solution(object):
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        stack = [] # 存储日期id
        
        result = []
        
        # 从右向左遍历
        T.reverse()
        
        for i, val in enumerate(T):
            while stack and val >= T[stack[-1]]:
                stack.pop()
            if stack:
                result.append(i-stack[-1])
            else:
                result.append(0)
            stack.append(i)
            
        result.reverse()
        return result
```

## 647. 回文子串（中等）
给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被计为是不同的子串。

> 输入: "abc"  
输出: 3  
解释: 三个回文子串: "a", "b", "c".

> 输入: "aaa"  
输出: 6  
说明: 6个回文子串: "a", "a", "a", "aa", "aa", "aaa".

> 输入的字符串长度不会超过1000。

动态规划数组dp[j]记录从j位置到当前遍历的字符位置i的子字符串是否为回文子串。若是记为1，否则记为0。


```python
class Solution(object):
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        length = len(s)
        result = 0
        
        dp = [0] * length
        
        for i in range(length):
            dp[i] = 1
            result += 1
            
            for j in range(i):
                if s[j] == s[i] and dp[j+1] == 1:
                    dp[j] = 1
                    result += 1
                else:
                    dp[j] = 0
        return result  
```

## 621. 任务调度器（中等）
给定一个用字符数组表示的 CPU 需要执行的任务列表。其中包含使用大写的 A - Z 字母表示的26 种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。CPU 在任何一个单位时间内都可以执行一个任务，或者在待命状态。

然而，两个相同种类的任务之间必须有长度为 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。

你需要计算完成所有任务所需要的最短时间。

> 输入: tasks = ["A","A","A","B","B","B"], n = 2  
输出: 8  
执行顺序: A -> B -> (待命) -> A -> B -> (待命) -> A -> B.

> 任务的总个数为 [1, 10000]。  
n 的取值范围为 [0, 100]。

完成所有任务的最短时间取决于出现次数最多的任务数量

上例中，先安排任务A

A -> () -> () -> A -> () -> () ->A

再安排B

A -> B -> (待命) -> A -> B -> (待命) -> A -> B.

观察得，A之后是否还有任务取决于是否存在和A相同次数的任务

结果如下计算

(任务 A 出现的次数 - 1) * (n + 1) + (出现次数为 3 的任务个数)，即：

(3 - 1) * (2 + 1) + 2 = 8

公式算出的值可能会比数组的长度小，如["A","A","B","B"]，n = 0，此时要取数组的长度


```python
class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        task_dic = {}
        max_task = tasks[0]
        
        for task in tasks:
            if task in task_dic:
                task_dic[task] += 1
            else:
                task_dic[task] = 1
            if task_dic[task] > task_dic[max_task]:
                max_task = task
        
        replica = 0
        
        for key, values in task_dic.items():
            if values == task_dic[max_task]:
                replica += 1
        
        result = max((task_dic[max_task] - 1) * (n + 1) + replica, len(tasks))
        
        return result
```

## 617. 合并二叉树（简单）

给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。

你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

示例 1:

输入:  
    Tree 1                                     
          1                                                   
         / \                                              
        3   2                                            
       /                                            
      5                                    
    Tree 2  
         2    
      / \    
      1   3   
       \   \   
        4   7    
输出: 
合并后的树:  
	     3  
	    / \  
	   4   5  
	  / \   \   
	 5   4   7


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if not t1:
            return t2
        if not t2:
            return t1
        
        t1.val += t2.val
        
        t1.left = self.mergeTrees(t1.left, t2.left)
        t1.right = self.mergeTrees(t1.right, t2.right)
        
        return t1
```

## 581. 最短无序连续子数组（简单）

给定一个整数数组，你需要寻找一个连续的子数组，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。

你找到的子数组应是最短的，请输出它的长度。

示例 1:


>输入: [2, 6, 4, 8, 10, 9, 15]  
>输出: 5  
>解释: 你只需要对 [6, 4, 8, 10, 9] 进行升序排序，那么整个表都会变为升序排序。

说明

> 1. 输入的数组长度范围在 [1, 10,000]。
> 2. 输入的数组可能包含重复元素 ，所以升序的意思是<=。

如果最右端的一部分已经排好序，这部分的每个数都比它左边的最大值要大，同理，如果最左端的一部分排好序，这每个数都比它右边的最小值小。所以我们从左往右遍历，如果i位置上的数比它左边部分最大值小，则这个数肯定要排序， 就这样找到右端不用排序的部分，同理找到左端不用排序的部分，它们之间就是需要排序的部分


```python
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        
        start = -1
        end = -2
        
        max_ = nums[0]
        min_ = nums[-1]
        
        for i in range(length):
            pos = length - 1 - i
            
            max_ = max(max_, nums[i])
            if max_ > nums[i]:
                end = i
            
            min_ = min(min_, nums[pos])
            if min_ < nums[pos]:
                start = pos
                
        return end - start + 1
```

## 560. 和为K的子数组（中等）

给定一个整数数组和一个整数 k，你需要找到该数组中和为 k 的连续的子数组的个数。

> 示例1:
> 输入:nums = [1,1,1], k = 2  
输出: 2 , [1,1] 与 [1,1] 为两种不同的情况。

>说明：
>1. 数组的长度为 [1, 20,000]。  
>2. 数组中元素的范围是 [-1000, 1000] ，且整数 k 的范围是 [-1e7, 1e7]。

哈希表，遍历数组，计算从第0元素到当前位置的sum。用哈希表保存从第0元素到当前位置的和为sum出现的次数。如果sum-k在哈希表中出现过。则表示从当前位置往前有和为k的连续子数组。初始化dic[0]=1。


```python
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        dic = {}
        
        dic[0] = 1
        
        sum_ = 0
        
        result = 0
        
        for num in nums:
            sum_ += num
            
            if sum_ - k in dic:
                result += dic[sum_ - k]
                
            if sum_ in dic:
                dic[sum_] += 1
            else:
                dic[sum_] = 1
            
        return result
```

## 494. 目标和（中等）

给定一个非负整数数组，a1, a2, ..., an, 和一个目标数，S。现在你有两个符号 + 和 -。对于数组中的任意一个整数，你都可以从 + 或 -中选择一个符号添加在前面。

返回可以使最终数组和为目标数 S 的所有添加符号的方法数。

示例1:
> 输入: nums: [1, 1, 1, 1, 1], S: 3  
输出: 5   
解释:   
-1+1+1+1+1 = 3  
+1-1+1+1+1 = 3  
+1+1-1+1+1 = 3  
+1+1+1-1+1 = 3  
+1+1+1+1-1 = 3  
一共有5种方法让最终目标和为3。

注意：

> 1. 数组的长度不会超过20，并且数组中的值全为正数。  
> 2. 初始的数组的和不会超过1000。  
> 3. 保证返回的最终结果为32位整数。  

原问题等同于： 找到nums一个正子集和一个负子集，使得总和等于target

我们假设P是正子集，N是负子集 例如： 假设nums = [1, 2, 3, 4, 5]，target = 3，一个可能的解决方案是+1-2+3-4+5 = 3 这里正子集P = [1, 3, 5]和负子集N = [2, 4]

那么让我们看看如何将其转换为子集求和问题：

sum(P) - sum(N) = target  
sum(P) + sum(N) + sum(P) - sum(N) = target + sum(P) + sum(N)  
2sum(P) = target + sum(nums)

寻找子集P，使得sum(P) == (target + sum(nums)) / 2

子集和问题，0-1背包问题，采用动态规划算法求解。

### 解法步骤

1. 新建数组dp，长度为P+1
2. dp的第x项表示和为x有多少种方法。dp[0] = 1
3. 返回dp[P]

如何更新dp数组

- 遍历nums，遍历的数记为num
 - 逆序从P遍历到num，遍历的数记做j
   - 更新dp[j]=dp[j-num]+dp[j]
   
- 这样遍历的含义是，对每一个在nums数组中的数num而言，dp在从num到P的这些区间里，都可以加上一个num，来到达想要达成的P

举例，对于[1, 2, 3, 4, 5], target=4, 设置数组dp[0]到dp[4]  

- 假如选择了数字2,那么dp[2:5]（也就是2到4）都可以通过加上数字2有所改变，而dp[0:2]（也就是0到1）加上这个2很明显就超了，就不管它。
- 以前没有考虑过数字2,考虑了会怎么样呢？就要更新dp[2:5]，比如说当我们在更新dp[3]的时候，就相当于dp[3] = dp[3] + dp[1],即本来有多少种方法，加上去掉了2以后有多少种方法。因为以前没有考虑过2, 现在知道, 只要整到了1, 就一定可以整到3。

**为什么以这个顺序来遍历呢？**  
假如给定nums = [num1,num2,num3]，我们现在可以理解dp[j] = dp[j-num1] + dp[j-num2] + dp[j-num3]。

但是如何避免不会重复计算或者少算？要知道，我们的nums并不是排序的，我们的遍历也不是从小到大的。

我们不妨跟着流程走一遍

第一次num1，仅仅更新了dp[num1] = 1，其他都是0+0都是0啊都是0  
第二次num2，更新了dp[num2] = 1和dp[num1+num2] = dp[num1+num2] + dp[num1] = 1,先更新后者。  
第三次num3，更新了dp[num3] = 1和dp[num1+num3] += 1和dp[num2+num3] += 1和dp[num1+num2+num3] += 1。按下标从大到小顺序来更新。  
......  
由此可见，这种顺序遍历能得到最后的答案。这里可以跟着IDE的debug功能走一遍，加深理解。


```python
class Solution(object):
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        if sum(nums) < S or (sum(nums) + S) % 2 == 1:
            return 0
        
        P = (sum(nums) + S) // 2
        dp = [1] + [0 for _ in range(P)]
        
        for num in nums:
            for j in range(P, num-1, -1):
                dp[j] = dp[j-num] + dp[j]
        
        return dp[P]
```

## 416. 分割等和子集（中等）

给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

1. 每个数组中的元素不会超过 100
2. 数组的大小不会超过 200

示例1:

> 输入: [1, 5, 11, 5]  
> 输出: true  
> 解释: 数组可以分割成 [1, 5, 5] 和 [11].  

示例2:

>输入: [1, 2, 3, 5]  
>输出: false  
>解释: 数组不能分割成两个元素和相等的子集.

同样采用动态规划算法求解

0-1背包问题。

dp[i][j]：表示能否从数组的 [0, i] 这个子区间内挑选一些正整数，每个数只能用一次，使得这些数的和等于 j。

根据我们学习的 0-1 背包问题的状态转移推导过程，新来一个数，例如是 nums[i]，根据这个数可能选择也可能不被选择：

- 如果不选择 nums[i]，在 [0, i - 1] 这个子区间内已经有一部分元素，使得它们的和为 j ，那么 dp[i][j] = true；
- 如果选择num[i]，在 [0, i - 1] 这个子区间内就得找到一部分元素，使得它们的和为 j - nums[i] ，我既然这样写出来了，你就应该知道，这里讨论的前提条件是 nums[i] <= j。

得到状态转移方程

dp[i][j] = dp[i-1][j] or dp[i - 1][j - nums[i]], (nums[i] <= j)


```python
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        sums = sum(nums)
        if sums % 2 == 1:
            return False
        
        target = sums // 2
        
        dp = [[False for _ in range(target + 1)] for _ in range(len(nums))]
        
        # 第一行，看第0个数能否填满给定和j
        for j in range(target + 1):
            dp[0][j] = True if nums[0] == j else False
        
        
        for i in range(1, len(nums)):
            for j in range(target + 1):
                if j >= nums[i]:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i]]
                else:
                    dp[i][j] = dp[i-1][j]
                
        return dp[-1][-1]
```

## 406. 根据身高重建队列 （中等）

假设有打乱顺序的一群人站成一个队列。 每个人由一个整数对(h, k)表示，其中h是这个人的身高，k是排在这个人前面且身高大于或等于h的人数。 编写一个算法来重建这个队列。

注意：
总人数少于1100人。

示例

> 输入:  
> [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]  
> 输出:  
> [[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]

排序，然后插入。

假设候选队列为 A，已经站好队的队列为 B.

从 A 里挑身高最高的人 x 出来，插入到 B. 因为 B 中每个人的身高都比 x 要高，因此 x 插入的位置，就是看 x 前面应该有多少人就行了。比如 x 前面有 5 个人，那 x 就插入到队列 B 的第 5 个位置。


```python
class Solution(object):
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        people.sort(key=lambda x: x[0], reverse=True)
        
        result = []
        
        for peo in people:
            idx = peo[1]
            result.insert(idx, peo)
            
        return result
```

## 399. 除法求值 （中等）DFS/BFS

给出方程式 A / B = k, 其中 A 和 B 均为代表字符串的变量， k 是一个浮点型数字。根据已知方程式求解问题，并返回计算结果。如果结果不存在，则返回 -1.0。

示例
> 给定 a / b = 2.0, b / c = 3.0  
问题: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?   
返回 [6.0, 0.5, -1.0, 1.0, -1.0 ]

输入为: vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries(方程式，方程式结果，问题方程式)， 其中 equations.size() == values.size()，即方程式的长度与方程式结果长度相等（程式与结果一一对应），并且结果值均为正数。以上为方程式的描述。 返回vector<double>类型。

基于上述例子，输入如下:

>equations(方程式) = [ ["a", "b"], ["b", "c"] ],  
values(方程式结果) = [2.0, 3.0],  
queries(问题方程式) = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ]. 

输入总是有效的。你可以假设除法运算中不会出现除数为0的情况，且不存在任何矛盾的结果。

思路：

建图，a->b权重为2.0，b->c权重为3.0。a->c权重为2*3

建图后采用dfs或bfs搜索路径，路径权重乘积记为返回结果，找不到则返回-1


```python
class Solution(object):
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        # 建图
        graph = {} # {equ[0] : [equ[1], ...], ...}
        weight = {}
        
        for idx, equ in enumerate(equations):
            if equ[0] not in graph:
                graph[equ[0]] = [equ[1]]
            else:
                graph[equ[0]].append(equ[1])
            
            if equ[1] not in graph:
                graph[equ[1]] = [equ[0]]
            else:
                graph[equ[1]].append(equ[0])
                
            weight[(equ[0], equ[1])] = values[idx]
            weight[(equ[1], equ[0])] = 1.0 / values[idx]
            
        # DFS
        def dfs(start, end, visited):
            # 如果图中有此边，直接输出
            if (start, end) in weight:
                return weight[(start, end)]
            # 如果不存在结点
            if start not in graph or end not in graph:
                return 0
            # 已经访问过此节点
            if start in visited:
                return 0
            
            visited.append(start)
            res = 0
            for neigh in graph[start]:
                res = dfs(neigh, end, visited) * weight[(start, neigh)]
                # 遍历到不为0的解跳出
                if res != 0:
                    # 添加此边，之后访问节省时间
                    weight[(start, end)] = res
                    break
            visited.remove(start)
            return res
        
        # 逐个求解
        result = []
        for que in queries:
            res = dfs(que[0], que[1], list())
            if res == 0:
                res = -1
            result.append(res)
        return result
```
