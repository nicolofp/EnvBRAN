GWAS - mRNA
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
df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets/GSE216998/GSE216275/GSE216275_cov_original.arrow"));
```

``` julia
df = filter([:replicate, :pregnancy_status_2] => (x,y) -> ismissing(x) & ismissing(y),df)
df = df[:,[1,2,3,4]];
```

``` julia
countmap(df.pregnancy_status_1)
```

    Dict{Union{Missing, String}, Int64} with 3 entries:
      "Gestational diabetes mellitus" => 37
      "Other pregnancy complications" => 40
      "Normal"                        => 286

``` julia
mrna = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets/GSE216998/GSE216275/GSE216275_mRNA_original.arrow"));
```

``` julia
mrna = mrna[:,df.ID1];
```

``` julia
x = Matrix{Float64}(mrna);
tx = Matrix(x');
```

``` julia
mrna_stats = Matrix{Float64}(hcat(mean(x,dims = 2),
        std(x,dims = 2),
        [minimum(x[i,:]) for i in 1:2170],
        [maximum(x[i,:]) for i in 1:2170],
        [length(countmap(x[i,:])) for i in 1:2170],
        [length(countmap(x[i,:])) for i in 1:2170]/363));
```

``` julia
sum(mrna_stats[:,6] .> 0.5)
```

    236

``` julia
x = x[mrna_stats[:,6] .> 0.5,:];
```

``` julia
y = convert(Array,  map(element -> element == "Normal" ? 1 : 2, df[:,:pregnancy_status_1]));
```

``` julia
# randomize the cancer data rows so when we split the data, we don't only pull one cancer classification
# also we need to transpose the x data, 
# because it needs to align properly with the output dimensions
# Xs, Ys = shuffleobs((x', y))
# split the data now
(X_train1, y_train1), (X_test1, y_test1) = splitobs((x, y); at = 0.67)
    
# transpose the x data back to its original dimensionality
    x_train = Array(transpose(X_train1))
    y_train = Array(y_train1)
    x_test = Array(transpose(X_test1))
    y_test = Array(y_test1);
```

``` julia
model_rf = RandomForestClassifier(n_subfeatures = 100, 
                                  n_trees = 1000, 
                                  min_samples_leaf = 5,
                                  partial_sampling = 0.90)

DecisionTree.fit!(model_rf,x_train,y_train)
```

    RandomForestClassifier
    n_trees:             1000
    n_subfeatures:       100
    partial_sampling:    0.9
    max_depth:           -1
    min_samples_leaf:    5
    min_samples_split:   2
    min_purity_increase: 0.0
    classes:             [1, 2]
    ensemble:            Ensemble of Decision Trees
    Trees:      1000
    Avg Leaves: 12.491
    Avg Depth:  6.435

``` julia
rf_prediction = convert(Array,DecisionTree.predict(model_rf,x_test))
errorrate(rf_prediction,y_test)
```

    0.2833333333333333

``` julia
confusmat(2, y_test, rf_prediction)
```

    2×2 Matrix{Int64}:
     86  0
     34  0
