**Variational inference using Julia**
================
Nicoló Foppa Pedretti
3/10/23

### Introduction

Variational inference is a type of probabilistic inference algorithm
that approximates complex probability distributions using simpler,
tractable distributions. It is commonly used in machine learning and
statistical modeling to estimate the parameters of a model or to compute
the posterior distribution of a set of latent variables given some
observed data.

The basic idea behind variational inference is to define a family of
tractable probability distributions, called the variational family, that
approximates the true, but often intractable, posterior distribution.
The goal is to find the member of the variational family that is closest
to the true posterior distribution in terms of a distance metric such as
the Kullback-Leibler (KL) divergence:

![KL(P\|\|Q)=\sum\_{x}P(x)\log(\frac{P(x)}{Q(x)})](https://latex.codecogs.com/svg.latex?KL%28P%7C%7CQ%29%3D%5Csum_%7Bx%7DP%28x%29%5Clog%28%5Cfrac%7BP%28x%29%7D%7BQ%28x%29%7D%29 "KL(P||Q)=\sum_{x}P(x)\log(\frac{P(x)}{Q(x)})")

The algorithm proceeds by optimizing the parameters of the variational
distribution to minimize the KL divergence between the variational
distribution and the true posterior. This is done by formulating the
problem as an optimization problem, where the objective function is the
KL divergence between the true posterior and the variational
distribution, and the variables to be optimized are the parameters of
the variational distribution.

The optimization problem is typically solved using gradient descent or
other numerical optimization techniques. At each iteration, the
algorithm updates the parameters of the variational distribution based
on the gradients of the objective function with respect to the
parameters.

Once the optimization is complete, the resulting variational
distribution can be used to approximate the posterior distribution,
which can be used for inference or other downstream tasks. The quality
of the approximation depends on the choice of the variational family and
the optimization algorithm used.

Variational inference is a powerful and flexible method for
probabilistic inference, and has been applied to a wide range of
problems in machine learning and statistics, including Bayesian
modeling, deep learning, and reinforcement learning.

Load the libraries used in the script:

``` julia
using Turing, Plots, LinearAlgebra, Statistics, Parquet
using HTTP, JSON, DataFrames, Optim, Zygote, ReverseDiff
```

### Linear bayesian regression

``` julia
# Bayesian linear regression.
@model function linear_regression(x, y)
    # Set variance prior.
    σ₂ ~ truncated(Normal(0, 100), 0, Inf)

    # Set intercept prior.
    intercept ~ Normal(0, sqrt(3))

    # Set the priors on our coefficients.
    nfeatures = size(x, 2)
    coefficients ~ MvNormal(nfeatures, sqrt(10))

    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    return y ~ MvNormal(mu, sqrt(σ₂))
end;
```

Variational inference can be used for statistical variable selection,
which involves identifying a subset of relevant variables that best
explain the data. This is an important problem in many areas of science
and engineering, as it can help to improve the accuracy and
interpretability of statistical models.

In the context of variable selection, variational inference can be used
to estimate the posterior probability of each variable being included in
the model, given some observed data. This can be achieved by defining a
prior distribution over the set of possible models, and then using
variational inference to compute the posterior distribution over this
set.

The prior distribution over models can be formulated as a product of
prior distributions over individual variables, with each prior
distribution specifying the prior probability of including or excluding
that variable from the model. The likelihood function can be expressed
as a product of conditional distributions, where each distribution
depends on a subset of the variables included in the model.

The variational distribution is then defined over the set of possible
models, with the parameters of the variational distribution specifying
the probabilities of including or excluding each variable from the
model. The objective function for the optimization problem is the KL
divergence between the true posterior distribution over models and the
variational distribution.

The resulting variational distribution can be used to estimate the
posterior probability of each variable being included in the model,
based on the probabilities assigned by the variational distribution.
Variables with high posterior probabilities are likely to be relevant,
while those with low probabilities can be excluded from the model.

Variational inference for variable selection can be extended to handle
more complex models, such as hierarchical models and models with
interactions between variables. It is a powerful and flexible method
that can be used to address a wide range of statistical inference
problems.

``` julia
# Instantiate model
m = linear_regression(x,y)
```

``` julia
# ADVI
advi = ADVI(10, 1000)
q = vi(m, advi);
```
