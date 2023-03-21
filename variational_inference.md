**Variational inference using Julia**
================
Nicoló Foppa Pedretti

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
using Turing, Plots, LinearAlgebra, Statistics, Parquet, CSV
using DataFrames, Optim, Zygote, ReverseDiff, Arrow
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
```

    true

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
table = Arrow.Table("C:/Users/nicol/Documents/Github_projects/EnvBRAN/clean_dt.arrow")
df = DataFrame(table);
```

``` julia
Xdt = Matrix{Float64}(df[:,vcat(collect(1:2),6,collect(16:72))])
Ydt = Array{Float64}(df.bw_zscore);
```

``` julia
# Instantiate model
m = linear_regression(Xdt,Ydt)
```

    DynamicPPL.Model{typeof(linear_regression), (:x, :y), (), (), Tuple{Matrix{Float64}, Vector{Float64}}, Tuple{}, DynamicPPL.DefaultContext}(linear_regression, (x = [25.5102043151855 17.0 … -0.53671336807613 0.2880726323194142; 26.4915084838867 18.0 … 1.078749679027982 3.1954783677431906; … ; 27.3796653747559 20.0 … -0.5158381392316987 0.33732010720086447; 20.7967300415039 20.0 … 2.8622580221540206 2.9194911007009714], y = [0.4521301802575446, 0.6817836051502657, 0.1902644116339792, 0.9375193793664165, -0.18199614453022778, -0.644749766556538, -0.6727508181580242, 0.7978075164781839, 0.45836965210587055, 0.8876797791920156  …  -2.5858993789239384, -1.2539995980339271, 0.16827521450578153, -1.5179237735655002, 0.5135426034748191, 0.39201541151351876, -1.6064729645602154, -1.3679101117174621, 0.3254248423856985, 0.2851462985582067]), NamedTuple(), DynamicPPL.DefaultContext())

``` julia
# ADVI
advi = ADVI(30, 1000)
q = vi(m, advi);
advi_sample = rand(q, 1000);
```

    ┌ Info: [ADVI] Should only be seen once: optimizer created for θ
    └   objectid(θ) = 0x4081033f2069627f
    [ADVI] Optimizing... 100% Time: 0:02:04

``` julia
results = Matrix(undef,62,4);
```

``` julia
for i in 1:62
    results[i,1] = vcat("sigma","Intercept",names(df)[vcat(collect(1:2),6,collect(16:72))])[i]
    results[i,2] = mean(advi_sample[i,:])
    results[i,3] = quantile(advi_sample[i,:],0.025)
    results[i,4] = quantile(advi_sample[i,:],0.975)
end
```

``` julia
tmp = DataFrame(results,:auto)
rename!(tmp,["parameters","advi_mean","advi_2.5%","advi_97.5%"])
show(tmp)
```

    62×4 DataFrame
     Row │ parameters              advi_mean  advi_2.5%   advi_97.5% 
         │ Any                     Any        Any         Any        
    ─────┼───────────────────────────────────────────────────────────
       1 │ sigma                   46.6214    24.6448     80.3045
       2 │ Intercept               1.80514    0.00187761  3.60444
       3 │ h_mbmi_None             0.531899   0.266752    0.809667
       4 │ hs_wgtgain_None         0.106517   -0.301507   0.468765
       5 │ h_age_None              0.673467   0.519783    0.815241
       6 │ h_abs_ratio_preg_Log    0.352907   -0.933692   1.67205
       7 │ h_no2_ratio_preg_Log    1.69146    0.328859    3.08612
       8 │ h_pm10_ratio_preg_None  1.07501    0.900069    1.25294
       9 │ h_pm25_ratio_preg_None  1.66871    1.02198     2.35674
      10 │ hs_as_m_Log2            0.026667   -1.39355    1.31836
      11 │ hs_cd_m_Log2            0.09976    -0.413523   0.622815
      ⋮  │           ⋮                 ⋮          ⋮           ⋮
      53 │ hs_meohp_madj_Log2      -0.893147  -1.91758    0.169287
      54 │ hs_mep_madj_Log2        0.987388   0.655608    1.30409
      55 │ hs_mibp_madj_Log2       0.856288   0.00265975  1.69291
      56 │ hs_mnbp_madj_Log2       0.344048   -0.194063   0.825632
      57 │ hs_ohminp_madj_Log2     0.441602   -0.0879747  0.948027
      58 │ hs_oxominp_madj_Log2    -0.426741  -0.662123   -0.195485
      59 │ hs_sumDEHP_madj_Log2    1.74771    1.24623     2.22577
      60 │ h_bro_preg_Log          -0.815799  -1.76905    0.0914334
      61 │ h_clf_preg_Log          0.791142   0.534681    1.05081
      62 │ h_thm_preg_Log          -0.623156  -2.4198     1.02868
                                                      41 rows omitted
