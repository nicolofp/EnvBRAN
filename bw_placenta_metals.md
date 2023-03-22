Birth weight and placenta metals
================
Nicoló Foppa Pedretti

``` julia
using DataFrames, Statistics, LinearAlgebra, Plots, Clustering, TSne, DecisionTree, MLDataUtils
using Distributions, StatsPlots, StatsBase, FillArrays, Arrow, UMAP, Distances, MLBase
```

``` julia
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps()) #std(A, dims=dims) #
```

    rescale (generic function with 1 method)

``` julia
df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets/GSE71678/Data/processed_data/GSE71678_cov_original.arrow"));
```

``` julia
df = df[completecases(df[:,["gestational age","infant gender","birth weight (grams)"]]),:];
```

``` julia
df.gestage_cat = ceil.(df[:,"gestational age"]);
```

``` julia
df[df.gestage_cat .< 37,:gestage_cat] = 36*ones(11)
df[df.gestage_cat .> 41,:gestage_cat] = 42*ones(40);
```

``` julia
countmap(df[:,"infant gender"])
histogram(df[:,"gestational age"])
countmap(df[:,"gestage_cat"])
```

    Dict{Float64, Int64} with 7 entries:
      39.0 => 52
      41.0 => 75
      36.0 => 11
      38.0 => 26
      42.0 => 40
      37.0 => 9
      40.0 => 126

``` julia
zscore_table = combine(groupby(df,["gestage_cat","infant gender"]), ["birth weight (grams)"] .=> (mean,std));
```

``` julia
df = innerjoin(df, zscore_table, 
               on = ["gestage_cat","infant gender"]);
```

``` julia
df.bw_zscore = (df[:,"birth weight (grams)"] .- df[:,"birth weight (grams)_mean"])./df[:,"birth weight (grams)_std"];
```

``` julia
df = df[completecases(df[:,["bw_zscore","mom_education","parity","ever_smoker","gestational_diabetes","maternal age",
                "placenta_al","placenta_cr","placenta_ni","placenta_se","placenta_co","placenta_pb"]]),:];
```

``` julia
countmap(df[:,"gestational_diabetes"])
df.gest_diabetis = df[:,"gestational_diabetes"] .== "Yes";
countmap(df[:,"gest_diabetis"])
```

    Dict{Bool, Int64} with 2 entries:
      0 => 280
      1 => 31

``` julia
countmap(df[:,"ever_smoker"])
df.smoking = df[:,"ever_smoker"] .== "Yes";
countmap(df[:,"smoking"])
```

    Dict{Bool, Int64} with 2 entries:
      0 => 270
      1 => 41

``` julia
countmap(df[:,"parity"])
df.parity_bin = df[:,"parity"] .> 0;
countmap(df[:,"parity_bin"])
```

    Dict{Bool, Int64} with 2 entries:
      0 => 133
      1 => 178

``` julia
countmap(df[:,"maternal race"])
```

    Dict{Union{Missing, String}, Int64} with 2 entries:
      "Other/Unknown" => 9
      "White"         => 302

``` julia
countmap(df[:,"mom_education"])
df.low_education = [df[i,"mom_education"] ∈ ["<11th grade","HS grad","Jr College/technical school"] for i in 1:size(df,1)];
countmap(df[:,"low_education"])
```

    Dict{Bool, Int64} with 2 entries:
      0 => 207
      1 => 104

``` julia
dft = df[:,["bw_zscore","low_education","parity_bin","smoking","gest_diabetis","maternal age",
            "placenta_al","placenta_cr","placenta_ni","placenta_se","placenta_co","placenta_pb"]];
show(dft)
```

    311×12 DataFrame
     Row │ bw_zscore   low_education  parity_bin  smoking  gest_diabetis  maternal ⋯
         │ Float64     Bool           Bool        Bool     Bool           Float64? ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
       1 │ -0.454284           false        true    false          false       35. ⋯
       2 │ -0.0715015          false       false    false          false       30.
       3 │ -0.186338            true        true    false          false       24.
       4 │  1.37646             true        true    false          false       23.
       5 │  2.17201            false        true    false          false       27. ⋯
       6 │  1.02761            false        true    false           true       35.
       7 │ -1.03438             true       false     true          false       32.
       8 │  0.3278             false        true    false          false       36.
       9 │  0.819531            true       false    false          false       28. ⋯
      10 │ -0.358588            true        true    false          false       26.
      11 │ -0.468257            true       false    false           true       29.
      ⋮  │     ⋮             ⋮            ⋮          ⋮           ⋮             ⋮   ⋱
     302 │  1.32336             true        true     true           true       33.
     303 │  0.774269           false        true    false          false       38. ⋯
     304 │  0.183687           false       false    false          false       37.
     305 │ -0.495196           false       false    false          false       30.
     306 │  0.440867           false        true    false           true       36.
     307 │ -0.371664            true       false    false          false       30. ⋯
     308 │ -0.248184           false        true    false          false       29.
     309 │ -0.0379639          false       false    false          false       31.
     310 │ -0.671646           false        true    false          false       33.
     311 │  1.18438             true        true    false          false       36. ⋯
                                                      7 columns and 290 rows omitted

## Linear Regression

Linear regression is a statistical model that is used to establish a
relationship between a dependent variable and one or more independent
variables. In simple linear regression, the model takes the form of a
straight line, while in multiple linear regression, the model is a
linear combination of the independent variables.

The basic idea behind linear regression is to find the best fitting line
or curve that summarizes the relationship between the dependent variable
and the independent variables. This is done by minimizing the sum of the
squared differences between the actual values of the dependent variable
and the predicted values from the model. The resulting line or curve is
called the regression line or regression curve.

The equation for the simple linear regression model is
$$y = \beta_0 + \beta_1 x + \epsilon$$

Where, y is the dependent variable, x is the independent variable,
$\beta_0$ is the intercept, $\beta_1$ is the slope coefficient, and
$\epsilon$ is the error term. The slope coefficient, $\beta_1$,
represents the change in the dependent variable for a one-unit change in
the independent variable. The intercept, $\beta_0$, represents the value
of the dependent variable when the independent variable is equal to
zero.

The goal of linear regression is to estimate the values of the
coefficients, $\beta_0$ and $\beta_1$, based on the data, so that we can
make predictions about the dependent variable for new values of the
independent variable.

``` julia
y = Array{Float64}(dft[:,"bw_zscore"])
z = hcat(ones(size(dft,1)),Matrix{Float64}(dft[:,2:size(dft,2)]));
```

``` julia
tmp = vcat("Intercept",names(dft)[2:size(dft,2)])
results = Array{Float64}(undef,size(dft,2), 6);
results = hcat(tmp,results);
```

``` julia
β = z\y
σ² = sum((y - z*β).^2)/(size(z,1)-size(z,2))
Σ = σ²*inv(z'*z)
std_coeff = sqrt.(diag(Σ))

results[:,2] = β
results[:,3] = std_coeff
results[:,4] = β./std_coeff
results[:,5] = [cdf(TDist(size(z,1)-size(z,2)), -abs(β[i]./std_coeff[i])) for i in 1:size(z,2)]
results[:,6] = β .- quantile(TDist(size(z,1)-size(z,2)), 0.975) .* std_coeff
results[:,7] = β .+ quantile(TDist(size(z,1)-size(z,2)), 0.975) .* std_coeff;
```

``` julia
results = DataFrame(results,:auto)
rename!(results, ["variable","beta","std_error","t_value","p_value","CI0025","CI0975"])
show(results, allcols = true)
```

    12×7 DataFrame
     Row │ variable       beta          std_error    t_value     p_value      CI0025        CI0975      
         │ Any            Any           Any          Any         Any          Any           Any         
    ─────┼──────────────────────────────────────────────────────────────────────────────────────────────
       1 │ Intercept      1.24821       0.530494     2.35291     0.00963745   0.20423       2.29218
       2 │ low_education  -0.0995283    0.127718     -0.779281   0.218215     -0.350868     0.151812
       3 │ parity_bin     0.729197      0.113385     6.43118     2.50351e-10  0.506064      0.952331
       4 │ smoking        -0.155676     0.170039     -0.915532   0.180325     -0.4903       0.178948
       5 │ gest_diabetis  0.14671       0.178734     0.820827    0.206199     -0.205026     0.498445
       6 │ maternal age   -0.039798     0.0122965    -3.23652    0.000672909  -0.0639967    -0.0155993
       7 │ placenta_al    -0.000577837  0.000270393  -2.13703    0.0167034    -0.00110995   -4.57235e-5
       8 │ placenta_cr    0.000517445   0.000518296  0.998359    0.159456     -0.000502524  0.00153741
       9 │ placenta_ni    -0.000957211  0.0115395    -0.0829505  0.466973     -0.0236662    0.0217518
      10 │ placenta_se    -0.00132207   0.00137142   -0.964013   0.167909     -0.00402094   0.0013768
      11 │ placenta_co    -0.000977154  0.00497016   -0.196604   0.422135     -0.0107581    0.00880377
      12 │ placenta_pb    0.0161262     0.0145591    1.10764     0.134454     -0.012525     0.0447774

``` julia
ssres = sum((y - z*β).^2)
sstot = sum((y .- mean(y)).^2)
R² = 1 .- (ssres/sstot)
AR² = 1 .- ((ssres/(size(dft,1)-size(dft,2)))/(sstot/(size(dft,1)-1)))
mse = (1.0 ./size(dft,1)).*ssres
println("R²     - ", round(R²; digits = 4))
println("Adj.R² - ", round(AR²; digits = 4))
println("MSE    - ", round(mse; digits = 4))
```

    R²     - 0.1505
    Adj.R² - 0.1193
    MSE    - 0.8257

## Random Forest Regression

A random forest is a machine learning algorithm that is commonly used
for classification and regression tasks. It belongs to the family of
ensemble methods, which combine multiple models to improve predictive
performance.

The basic idea of a random forest is to create a large number of
decision trees, each of which is trained on a random subset of the
available data and features. Then, the predictions of all the trees are
aggregated to make a final prediction.

More formally, let
$\mathcal{D} = {(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}$ be a dataset
of $n$ instances, where $x_i$ is a vector of features and $y_i$ is the
corresponding label. The random forest algorithm can be summarized as
follows:

For $k=1$ to $K$ (the number of trees to grow): a. Sample a random
subset of size $m$ from the dataset, with replacement. This is known as
bootstrap sampling. b. Randomly select a subset of $p$ features to use
in the tree. c. Grow a decision tree from the bootstrap sample using the
selected features. To make a prediction for a new instance $x$, apply
each of the $K$ trees to $x$ and obtain a prediction. For classification
tasks, the final prediction is typically the mode of the predictions,
and for regression tasks, it is the mean. The key parameters of a random
forest are the number of trees $K$, the size of the bootstrap sample
$m$, and the number of features to select $p$. These parameters can be
tuned using cross-validation to optimize performance on a given task.

The prediction of a random forest can be formulated mathematically as
follows:

$$\hat{y} = \text{RF}(x) = \text{agg}(f_1(x), f_2(x), ..., f_K(x))$$

where $\hat{y}$ is the predicted label for the input vector $x$,
$\text{RF}$ denotes the random forest function, $\text{agg}$ is the
aggregation function (e.g., mode or mean), and $f_k(x)$ is the
prediction of the $k$-th decision tree on $x$.

``` julia
y = Array{Float64}(dft[:,"bw_zscore"])
x = Matrix{Float64}(dft[:,2:size(dft,2)]);

# train regression forest
# set of classification parameters and respective default values
# n_subfeatures: number of features to consider at random per split (default: -1, sqrt(# features))
# n_trees: number of trees to train (default: 10)
# partial_sampling: fraction of samples to train each tree on (default: 0.7)
# max_depth: maximum depth of the decision trees (default: no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
# keyword rng: the random number generator or seed to use (default Random.GLOBAL_RNG)
#              multi-threaded forests must be seeded with an `Int`
model_rf = build_forest(y, x, -1, 1000, 0.7, -1, 3, 2, 0.0)
```

    Ensemble of Decision Trees
    Trees:      1000
    Avg Leaves: 59.028
    Avg Depth:  12.444

``` julia
y_pred = apply_forest(model_rf,x);
ssres = sum((y - y_pred).^2)
sstot = sum((y .- mean(y)).^2)
R² = 1 .- (ssres/sstot)
AR² = 1 .- ((ssres/(size(dft,1)-size(dft,2)))/(sstot/(size(dft,1)-1)))
mse = (1.0 ./size(dft,1)).*ssres
println("R²     - ", round(R²; digits = 4))
println("Adj.R² - ", round(AR²; digits = 4))
println("MSE    - ", round(mse; digits = 4))
```

    R²     - 0.5946
    Adj.R² - 0.5797
    MSE    - 0.3941
