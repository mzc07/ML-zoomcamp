# Regression

## What is Regression?

In supervised machine learning, problems are divided into two main categories:
- **Classification:** Predicts a discrete class or category label.
- **Regression:** Predicts a continuous numerical value.

Imagine a sheet of paper scattered with dots, where each dot represents the relationship between independent variables and a target outcome. The goal of regression is to model and describe this underlying relationship as accurately as possible.

## Why Do We Use It?
Regression serves three main objectives:
- **Prediction:** Estimating a future or unseen numerical value for a new observation based on its features or attributes.
- **Inference:** Identifying which features impact the target variable and measuring their relative importance.
- **Description:** Condensing the relationships among multiple complex variables within a dataset.

# Mathematical Basis
We assume that the *observed value* $y$ comes from an unknown true function $f(x)$ plus an error term or random noise:
$$
y = f(x) + \epsilon
$$

The theoretical objective is to estimate the regression function $r(x)$, which represents the conditional expectation:
$$
r(x) = \mathbb{E}[Y \mid X = x]
$$

To find the optimal line or surface, the model measures the deviation between its predictions $\hat{y}$ and the true values $y$ using a **loss function** (such as squared error $(y-\hat{y})^2$). Minimizing the residual sum of squares or the mean squared error guarantees finding the maximum likelihood estimate when the noise follows a normal/Gaussian distribution.

Regression projects the observation vector $y \in \mathbb{R}^N$ orthogonally onto the vector subspace spanned by the feature columns $X$. This projection ensures that the error vector is completely perpendicular to the predictor variables; therefore, no remaining information or pattern is left in the errors.