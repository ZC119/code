
## 204. 计数质数（简单）埃氏筛法

统计所有小于非负整数 n 的质数的数量。

示例:

    输入: 10
    输出: 4
    解释: 小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。


```python
class Solution(object):
    def isPrim(self, n):
        i = 2
        while i * i <= n:
            if n % i == 0:
                return False
            i += 1

        return True

    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        isPrimes = [True] * n

        i = 2
        while i * i < n:
            if self.isPrim(i):
                j = i * i
                while j < n:
                    isPrimes[j] = False
                    j += i
            i += 1

        count = 0
        for i in range(2, n):
            if isPrimes[i]:
                count += 1

        return count
```

## 202. 快乐数（简单）

编写一个算法来判断一个数是不是“快乐数”。

一个“快乐数”定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是无限循环但始终变不到 1。如果可以变为 1，那么这个数就是快乐数。

示例: 

    输入: 19
    输出: true
    解释: 
    1^2 + 9^2 = 82
    8^2 + 2^2 = 68
    6^2 + 8^2 = 100
    1^2 + 0^2 + 0^2 = 1


```python
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        repeat = set()

        pre = n
        while True:
            repeat.add(pre)
            newnum = 0
            for s in str(pre):
                newnum += int(s)**2
            if newnum == 1:
                return True
            if newnum in repeat:
                return False

            pre = newnum
```

## 191. 位1的个数（简单）

编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’ 的个数（也被称为汉明重量）。

 

示例 1：

    输入：00000000000000000000000000001011
    输出：3
    解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
示例 2：

    输入：00000000000000000000000010000000
    输出：1
    解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
示例 3：

    输入：11111111111111111111111111111101
    输出：31
    解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。


```python
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        n = bin(n)
        count = 0
        for s in n:
            if s == '1':
                count += 1

        return count
```

## 189. 旋转数组（简单）

给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。

示例 1:

    输入: [1,2,3,4,5,6,7] 和 k = 3
    输出: [5,6,7,1,2,3,4]
    解释:
    向右旋转 1 步: [7,1,2,3,4,5,6]
    向右旋转 2 步: [6,7,1,2,3,4,5]
    向右旋转 3 步: [5,6,7,1,2,3,4]
示例 2:

    输入: [-1,-100,3,99] 和 k = 2
    输出: [3,99,-1,-100]
    解释: 
    向右旋转 1 步: [99,-1,-100,3]
    向右旋转 2 步: [3,99,-1,-100]
说明:

尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题。
要求使用空间复杂度为 O(1) 的 原地 算法。


```python
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n
        nums[:] = nums[-k:] + nums[:-k]
```

## 179. 最大数（中等）排序

给定一组非负整数，重新排列它们的顺序使之组成一个最大的整数。

示例 1:

    输入: [10,2]
    输出: 210
示例 2:

    输入: [3,30,34,5,9]
    输出: 9534330
    说明: 输出结果可能非常大，所以你需要返回一个字符串而不是整数。


```python
class largerNum(str):
    def __lt__(x, y):
        return x + y > y + x

class Solution(object):
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        
        
        result = ''.join(sorted(map(str, nums), key=largerNum))
        return '0' if result[0] == '0' else result
```

## 172. 阶乘后的零（简单）

给定一个整数 n，返回 n! 结果尾数中零的数量。

示例 1:

    输入: 3
    输出: 0
    解释: 3! = 6, 尾数中没有零。
示例 2:

    输入: 5
    输出: 1
    解释: 5! = 120, 尾数中有 1 个零.
说明: 你算法的时间复杂度应为 O(log n) 。


```python
class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        count = 0

        while n >= 5:
            count += n // 5
            n //= 5

        return count
```

## 171. Excel表列序号（简单）

给定一个Excel表格中的列名称，返回其相应的列序号。

例如，

    A -> 1
    B -> 2
    C -> 3
    ...
    Z -> 26
    AA -> 27
    AB -> 28 
    ...
示例 1:

    输入: "A"
    输出: 1
示例 2:

    输入: "AB"
    输出: 28


```python
class Solution(object):
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        result = 0

        for string in s:
            str2num = ord(string) - 64
            result = result * 26 + str2num

        return result
```

## 166. 分数到小数（中等）

给定两个整数，分别表示分数的分子 numerator 和分母 denominator，以字符串形式返回小数。

如果小数部分为循环小数，则将循环的部分括在括号内。

示例 1:

    输入: numerator = 1, denominator = 2
    输出: "0.5"
示例 2:

    输入: numerator = 2, denominator = 1
    输出: "2"
示例 3:

    输入: numerator = 2, denominator = 3
    输出: "0.(6)"


```python

```

## 162. 寻找峰值（中等）二分法

峰值元素是指其值大于左右相邻值的元素。

给定一个输入数组 nums，其中 nums[i] ≠ nums[i+1]，找到峰值元素并返回其索引。

数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可。

你可以假设 nums[-1] = nums[n] = -∞。

示例 1:

    输入: nums = [1,2,3,1]
    输出: 2
    解释: 3 是峰值元素，你的函数应该返回其索引 2。
示例 2:

    输入: nums = [1,2,1,3,5,6,4]
    输出: 1 或 5 
    解释: 你的函数可以返回索引 1，其峰值元素为 2；
         或者返回索引 5， 其峰值元素为 6。
说明:

你的解法应该是 O(logN) 时间复杂度的。

    首先要注意题目条件，在题目描述中出现了 nums[-1] = nums[n] = -∞，这就代表着 只要数组中存在一个元素比相邻元素大，那么沿着它一定可以找到一个峰值
    根据上述结论，我们就可以使用二分查找找到峰值
    查找时，左指针 l，右指针 r，以其保持左右顺序为循环条件
    根据左右指针计算中间位置 m，并比较 m 与 m+1 的值，如果 m 较大，则左侧存在峰值，r = m，如果 m + 1 较大，则右侧存在峰值，l = m + 1


```python
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        right = len(nums) - 1

        while left < right:
            mid = (left + right) // 2
            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            else:
                right = mid
        
        return left
```

## 130. 被围绕的区域（中等）

给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。

找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。

示例:

    X X X X
    X O O X
    X X O X
    X O X X
    运行你的函数后，矩阵变为：

    X X X X
    X X X X
    X X X X
    X O X X
解释:

被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。


```python
class Solution(object):
    def helper(self, board, i, j, m, n):
        if i < 0 or i > m-1 or j < 0 or j > n-1 or board[i][j] == 'X' or board[i][j] == '#':
            return
        # 如果i, j为'O'，用'#'替换
        board[i][j] = '#'

        self.helper(board, i-1, j, m, n)
        self.helper(board, i+1, j, m, n)
        self.helper(board, i, j-1, m, n)
        self.helper(board, i, j+1, m, n)



    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        if not board:
            return
        m = len(board)
        n = len(board[0])

        # 遍历边界
        i = 0
        for j in range(n):
            self.helper(board, i, j, m, n)

        i = m-1
        for j in range(n):
            self.helper(board, i, j, m, n)
        
        j = 0
        for i in range(1, m-1):
            self.helper(board, i, j, m, n)

        j = n-1
        for i in range(1, m-1):
            self.helper(board, i, j, m, n)
        
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == '#':
                    board[i][j] = 'O'
```

## 329. 矩阵中的最长递增路径（困难）

给定一个整数矩阵，找出最长递增路径的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。 你不能在对角线方向上移动或移动到边界外（即不允许环绕）。

示例 1:

    输入: nums = 
    [
      [9,9,4],
      [6,6,8],
      [2,1,1]
    ] 
    输出: 4 
    解释: 最长递增路径为 [1, 2, 6, 9]。
示例 2:

    输入: nums = 
    [
      [3,4,5],
      [3,2,6],
      [2,2,1]
    ] 
    输出: 4 
    解释: 最长递增路径是 [3, 4, 5, 6]。注意不允许在对角线方向上移动。


```python

```

## 315. 计算右侧小于当前元素的个数（困难）归并排序+索引数组

给定一个整数数组 nums，按要求返回一个新数组 counts。数组 counts 有该性质： counts[i] 的值是  nums[i] 右侧小于 nums[i] 的元素的数量。

示例:

    输入: [5,2,6,1]
    输出: [2,1,1,0] 
    解释:
    5 的右侧有 2 个更小的元素 (2 和 1).
    2 的右侧仅有 1 个更小的元素 (1).
    6 的右侧有 1 个更小的元素 (1).
    1 的右侧有 0 个更小的元素.

先考虑归并排序操作

求解 “逆序对” 的关键在于：当其中一个数字放进最终归并以后的有序数组中的时候，这个数字与之前看过的数字个数（或者是未看过的数字个数）可以直接统计出来，而不必一个一个数”。


```python

```
