代码结构：

# api.py
flask后端运行程序
+ def topics(): #得到top-k 子领域并以json文件返回
+ def main(): #运行

# mlp.py
生成mlp_model.h5


api功能：对于给定的area，返回该area的top k 的父领域和子领域
api参数意义：
area: 给定的领域，比如artificial intelligence, 中间用下划线或空格连接都可以，默认是[machine learning]
context: 背景，默认是[computer science]，加上指定背景的准确率会好一些，当然也可以不加，就是&context= 就可以
k: 返回的子领域的个数, 默认为10
weighting_mode：0或1，0表示简化版的算法，效率高，结果也不错。1表示原始算法，计算速度稍慢。默认是0。
compute_mode: 0，1，2，3，4， 默认是0。0表示的是考虑不同的初始的树。 0表示wiki+acm+mag的综合结果，1, 2, 4是只考虑wiki，mag，acm的结果。3是我算法内部使用的得到父领域的参数。
method：计算子领域的方法，目前可以是origin(默认）, mlp, rnn。
confidence： 只返回与area相似度大于confidence的结果。默认是0.0。 这个只在method=origin下有意义
has_parents: 0或1， 是否计算并返回area的父领域。因为原始api是只计算子领域的，所以这里加了一个参数。默认是0，即不返回父领域
has_children: 0或1，是否计算并返回area的父领域。默认是1，即返回子领域

kp: 表示每层父领域的个数， 默认为3
depth: 表示子领域的树的深度， 默认为1
depth_p: 表示父领域的树的深度， 默认为1

上述参数可以由用户调节的应该就是 area, context, k, kp, has_parents, has_children, confidence, depth, depth_p
其中， k, kp 如果超过30则置为30， depth, depth_p如果超过5，则赋为5

api示例：
http://0.0.0.0:5098/topics?area=machine learning&method=origin&has_parent=1&confidence=0.4
