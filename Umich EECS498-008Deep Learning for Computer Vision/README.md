Course refer to [EECS 498.008 / 598.008 Deep Learning for Computer Vision](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html)

# L3 Linear Classifiers

socres for classification $s = f(x_i, W)$  

raw classifier scores as probabilities as $P(Y = k | X = x_i) = \frac{\exp(s_k)}{\sum_j{\exp(s_j)}}$


Use Cross-Entropy Loss : $L_i = - \log P(Y = y_i | X = x_i)$
