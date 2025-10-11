# L3 Linear Classifiers
`Do Assignment 1 here`
## **Tips:**  
- Socres for classification $s = f(x_i, W)$  

- raw classifier scores as probabilities as $P(Y = k | X = x_i) = \frac{\exp(s_k)}{\sum_j{\exp(s_j)}}$

- Use Cross-Entropy Loss : $L_i = - \log P(Y = y_i | X = x_i)$

## Content 
- Linear classifiers
- Three ways to think about linear classifiers
  - Algebraic Viewpoint
  - Visual Viewpoint
  - Geometric Viewpoint
- Loss
  - Cross-Entropy (softamx) : $L_i = - log( \frac{exp(s_{y_i})}{\sum_j exp(s_{j})} )$
  - SVM Loss with margin : $L_i = \sum_{j\neq y_i} max(0, s_j - s_{y_i} +1 )$



# L4 Regularization + Optimization
`After this lecture, can do A2 linear classifiers section`


# L5
`After Lecture 5, can do fully-connected networks`