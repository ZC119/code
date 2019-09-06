'''
你有3个需要完成的任务，完成这3个任务是需要付出代价的。
首先，你可以不花任何代价的完成一个任务；然后，在完成了第i个任务之后，你可以花费|Ai - Aj|的代价完成第j个任务。|x|代表x的绝对值。
计算出完成所有任务的最小代价。

输入：一行3个整数A1,A2,A3，每个数字之间用一个空格分隔。所有数字都是整数，并且在[1,100]范围内。

输入例子1:
1 6 3
输出例子1:
5

输入例子2:
10 10 10
输出例子2:
0
'''

# import sys
# for line in sys.stdin:
#     a = line.split()
#     a = list(map(int, a))
#     print(max(a)-min(a))




'''
小易准备去拜访他的朋友，他的家在0点，但是他的朋友的家在x点(x > 0)，
均在一条坐标轴上。小易每一次可以向前走1，2，3，4或者5步。
问小易最少走多少次可以到达他的朋友的家。

输入：一行包含一个数字x(1 <= x <= 1000000)，代表朋友家的位置。
输出：一个整数，最少的步数。

输入例子：4
输出例子：1
输入例子：10
输出例子：2
'''

# import sys

# for line in sys.stdin:
#     a = int(line.split()[0])

#     step = 0
#     if a <= 5:
#         print(1)
#     else:
#         for i in range(5, 0, -1):
#             delta = a // i
#             a = a - delta * i
#             step += delta
#             if a <= 5 and a > 0:
#                 step += 1
#                 break
#             else:
#                 break
#         print(step)

'''
给定一个N*M的矩阵，在矩阵中每一块有一张牌，我们假定刚开始的时候所有牌的牌面向上。
现在对于每个块进行如下操作：
> 翻转某个块中的牌，并且与之相邻的其余八张牌也会被翻转。
XXX
XXX
XXX
如上矩阵所示，翻转中间那块时，这九块中的牌都会被翻转一次。
请输出在对矩阵中每一块进行如上操作以后，牌面向下的块的个数。

输入描述：
输入的第一行为测试用例数t(1 <= t <= 100000),
接下来t行，每行包含两个整数N,M(1 <= N, M <= 1,000,000,000)

输出描述：
对于每个用例输出包含一行，输出牌面向下的块的个数

输入例子：
5 1 1 1 2 3 1 4 1 2 2

输出例子：
1 0 1 2 0
'''

## TODO: 通过率90%。大数乘法不通过。
# import sys

# start = True
# for line in sys.stdin:
#     a = line.split()
#     a = list(map(int, a))

#     if start:
#         start = False
#         t = a[0]
#         continue

#     N = a[0]
#     M = a[1]
    
#     # 求n*m矩阵的结果, 观察可知，统计周围九宫格的个数，若为奇数，则牌面向下
#     if N*M == 1:
#         print(1)
#         continue

#     if N == 1:
#         print(M-2)
#         continue
    
#     if M == 1:
#         print(N-2)
#         continue

#     print((M-2)*(N-2))


    




