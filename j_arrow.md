Arrow with Julia
================
Nicoló Foppa Pedretti

``` julia
using DataFrames, Statistics, LinearAlgebra, Plots, CategoricalArrays, UMAP 
using Distributions, StatsPlots, StatsBase, TSne, CSV, Optim, Arrow, Parquet
```

``` julia
df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/GSE71678/Data/processed_data/GSE71678_cov_original.arrow"))
```
