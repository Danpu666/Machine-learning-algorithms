# Logistic Regression

#思路

>* 梯度下降求解
边界为线性的
边界为曲线的（unfinished）
>* 函数库求解

#梯度下降求解
##边界为线性时
###1、导入数据
###2、对数据图形展示
###3、代价函数

![](http://latex.codecogs.com/gif.latex?\\ J(\theta)=-\frac{1}{m}\sum_{i=1}^m
[y^{(i)}log(h_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)})])
$$
$$

    h=sigmoid(X.dot(theta))
    J=(-1.0/m)*(transpose(y).dot(log(h))+transpose(1.0-y).dot(log(1.0-h)))

###4、梯度下降函数
$$
\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^m
(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j
$$
 

    h=sigmoid(X.dot(theta))
    grad=np.dot(np.transpose(X),h-y)/m
    theta=theta-alpha*grad

###5、训练
注意 $\theta $ 的初始化（$ \theta=0$）、学习率的设定（$\alpha=0.001$）、迭代次数的选择（iters=100000）
！！！参数的选择非常重要，因为有时候报错是因为参数选择不对
###6、图形展示及正确率的计算
###7、改进
>* 将数据划分为训练集和测试集

##边界为曲线时（unfinished）
###1、数据处理
需要把数据$X_1$、$X_2$映射成$X_1$、$X_2$、$X_1X_2$、$X_1^2$、$X_2^2$等形式

  

    mapped_fea = ones(shape=(x1[:,0].size,1))   
    for i in range(1, degree +1):  
        for j in range(i +1):  
            r =(x1 **(i - j))*(x2 ** j)  
            mapped_fea = append(mapped_fea,r, axis=1)
###2、代价函数
$$ J(\theta)=-\frac{1}{m}\sum_{i=1}^m
[y^{(i)}log(h_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)})]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2
$$

注意$j$是重1开始的，因为$\theta(0)$为一个常数项，$X$中最前面一列会加上1列1，所以乘积还是$\theta(0)$,与feature没有关系，没有必要正则化

    thetaR=zeros([theta.shape[0]-1,1])
    for i in range(theta.shape[0]-1):
        thetaR[i]=theta[i+1]
此代码块等同于`theta=theta[1:,0]`

但是在python中，会自动把`(m,1)`二维数组处理为`(m,)`一维数组，从而报错，故此麻烦进行迭代
  

 如下是代价函数（含有惩罚项）：

    J =(1.0/ m)*((-y.T.dot(log(h)))-((1- y.T).dot(log(1.0- h))))+(l /(2.0* m))*(thetaR.T.dot(thetaR)) 

3、梯度下降函数

    h = sigmoid(X.dot(theta))
    
    thetaR=zeros([theta.shape[0]-1,1])
    for i in range(theta.shape[0]-1):
        thetaR[i]=theta[i+1]
    #thetaR =theta[1:,0].reshape(theta.shape[0]-1,1)
    
    delta = h - y  
    sum_delta = delta.T.dot(X[:,1])  
    grad1 =(1.0/ m)* sum_delta  
    
    XR = X[:,1:X.shape[1]]  
    sum_delta = XR.T.dot(delta) 
    grad =(1.0/ m)*(sum_delta + l * thetaR)  
    
    out = zeros(shape=(grad.shape[0]+1, grad.shape[1]))  
    out[0,]= grad1  
    out[1:,]= grad 

4、优化
梯度下降使用`scipy`中`optimize`的`fmin_bfgs`函数

     import scipy as sp
     from scipy.optimize import fmin_bfgs

最终报错告终

    ValueError: shapes (6,118) and (6,118) not aligned: 118 (dim 1) != 6 (dim 0)



#函数库求解
采用`scikit-learn`库中的逻辑回归模型实现
值得注意的时，需要根据不同的数据集选择不同的`solver`

    solver=‘newton-cg’ or ‘liblinear’
