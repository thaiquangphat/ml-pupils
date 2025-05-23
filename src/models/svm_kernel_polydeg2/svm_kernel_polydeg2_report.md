#### 1. Initialization
The model is initialized with a regularization parameter C:
$$
\text{Initialize with } C = 1.0 \text{ (default)}
$$

#### 2. Polynomial Kernel Function
The polynomial kernel of degree 2 is implemented as:
$$
K(x, y) = (x \cdot y + 1)^2
$$
where:
- $x$ and $y$ are feature vectors
- $x \cdot y$ is the dot product
- The kernel maps the input space to a higher-dimensional feature space

#### 3. Training Process (Dual Problem)
The SVM training solves the following quadratic programming problem:

$$
\min_{\alpha} \frac{1}{2}\alpha^T P \alpha + q^T \alpha
$$

Subject to:
$$
G \alpha \leq h
$$
$$
A \alpha = b
$$

Where:
- $P = (y_i y_j K(x_i, x_j))_{i,j}$ is the kernel matrix
- $q = -1$ (vector of ones)
- $G = \begin{bmatrix} -I \\ I \end{bmatrix}$ (constraints matrix)
- $h = \begin{bmatrix} 0 \\ C \end{bmatrix}$ (constraints vector)
- $A = y^T$ (equality constraint)
- $b = 0$ (equality constraint)

#### 4. Support Vector Selection
After solving the QP problem, support vectors are selected where:
$$
\alpha_i > 1e^{-5}
$$

#### 5. Bias Calculation
The bias term is calculated as:
$$
b = \frac{1}{n_{sv}} \sum_{k=1}^{n_{sv}} (y_k - \sum_{i=1}^{n_{sv}} \alpha_i y_i K(x_k, x_i))
$$
where:
- $n_{sv}$ is the number of support vectors
- $x_k, y_k$ are support vectors and their labels
- $\alpha_i$ are the Lagrange multipliers
- $K(x_k, x_i)$ is the kernel function

#### 6. Decision Function
For prediction, the decision function is:
$$
f(x) = \text{sign}(\sum_{i=1}^{n_{sv}} \alpha_i y_i K(x, x_i) + b)
$$

#### 7. Multiclass Extension (One-vs-Rest)
For multiclass classification, the model uses One-vs-Rest approach:
- For each class $c$:
  - Create binary labels: $y_i = 1$ if $x_i \in c$, else $y_i = -1$
  - Train a binary SVM classifier
  - Store the decision function $f_c(x)$
- Final prediction: $\arg\max_c f_c(x)$ 