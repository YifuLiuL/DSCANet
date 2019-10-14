import numpy as np

a = np.array([0.2,0.56,1.23])
print(type(a))
print(a.dtype)
b = a.astype(np.uint8)
print(a,b)
# def minfd():
#     a, b = 1, 2
#     return a, b   #自动封装成元组
# cc = minfd()
# print(type(cc))

# if isinstance((1,1), tuple):
#     print('true')

#闭包函数
def func1():
    print('func1')
    def func2():
        print('func2')
    func2()
func1()