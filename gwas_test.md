GWAS - cpgs
================
Nicoló Foppa Pedretti

``` julia
using DataFrames, Statistics, LinearAlgebra, Plots, Clustering, TSne, DecisionTree, MLDataUtils
using Distributions, StatsPlots, StatsBase, FillArrays, Arrow, UMAP, Distances, MLBase, GigaSOM
```

Self-organizing maps (SOM) are a type of artificial neural network that
use unsupervised learning to create a low-dimensional representation of
high-dimensional data. The algorithm involves iteratively adjusting the
weights of neurons in the network to create a topological map of the
input data. The formula for SOM can be expressed as follows:

Let there be a set of input vectors X = {x1, x2, …, xn} where each
vector x has m features x = (x1, x2, …, xm). The goal of SOM is to
create a map of k neurons arranged in a grid with topological structure.
Each neuron j in the map has a weight vector wj = (w1j, w2j, …, wmj)
with the same dimension as the input vectors.

The algorithm works as follows:

1.  Initialize the weights of the neurons randomly.

2.  For each input vector xi, find the neuron j with the closest weight
    vector to xi. This is known as the Best Matching Unit (BMU) and can
    be calculated as follows:
    $$ BMU = argmin_{j} \Vert{x_i - w_j}\Vert $$ where \|\|.\|\| denotes
    the Euclidean distance.

3.  Update the weights of the BMU and its neighbors in the map to move
    them closer to xi. This is done using the following formula:
    $$ w_{j}(t+1) = w_{j}(t) + \alpha(t)h_{j,i}(t)(x_i - w_{j}(t)) $$
    where t is the iteration number, α(t) is the learning rate which
    decreases over time, and h\_{j,i}(t) is the neighborhood function
    which determines the extent to which neighboring neurons are
    updated. It is defined as:
    $$ h_{j,i}(t) = exp\Bigg(-\frac{\Vert{r_i - r_j}\Vert^2}{2\sigma^2(t)}\Bigg) $$
    where r_i and r_j are the positions of neurons i and j in the grid,
    and σ(t) is the neighborhood size which also decreases over time.

4.  Repeat steps 2 and 3 for a fixed number of iterations or until
    convergence. The resulting SOM can be used to visualize the
    high-dimensional data in a low-dimensional space and identify
    clusters or patterns in the data.

``` julia
df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets/GSE71678/Data/processed_data/GSE71678_cov_original.arrow"));
```

``` julia
cpgs = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets/GSE71678/Data/processed_data/GSE71678_cpgs_original.arrow"));
```

``` julia
cpgs = cpgs[:,vcat("ID_REF",df.ID)];
```

``` julia
x = Matrix{Float64}(cpgs[:,2:size(cpgs,2)]);
tx = Matrix(x');
```

``` julia
cpgs_stats = Matrix{Float64}(hcat(mean(x,dims = 2),
        std(x,dims = 2),
        [minimum(x[i,:]) for i in 1:size(x,1)],
        [maximum(x[i,:]) for i in 1:size(x,1)],
        [length(countmap(x[i,:])) for i in 1:size(x,1)],
        [length(countmap(x[i,:])) for i in 1:size(x,1)]/size(x,2)));
```

``` julia
size(x)
```

    (344348, 343)

``` julia
println("cg   - ",sum(contains.(cpgs.ID_REF,"cg")))
println("ch.1 - ",sum(contains.(cpgs.ID_REF,"ch.1")))
println("ch.2 - ",sum(contains.(cpgs.ID_REF,"ch.2")))
println("ch.3 - ",sum(contains.(cpgs.ID_REF,"ch.3")))
println("ch.4 - ",sum(contains.(cpgs.ID_REF,"ch.4")))
println("ch.5 - ",sum(contains.(cpgs.ID_REF,"ch.5")))
println("ch.6 - ",sum(contains.(cpgs.ID_REF,"ch.6")))
println("ch.7 - ",sum(contains.(cpgs.ID_REF,"ch.7")))
println("ch.8 - ",sum(contains.(cpgs.ID_REF,"ch.8")))
println("ch.9 - ",sum(contains.(cpgs.ID_REF,"ch.9")))
```

    cg   - 342088
    ch.1 - 1037
    ch.2 - 374
    ch.3 - 119
    ch.4 - 142
    ch.5 - 123
    ch.6 - 133
    ch.7 - 127
    ch.8 - 117
    ch.9 - 88

``` julia
# R = pairwise(Euclidean(), x[contains.(cpgs.ID_REF,"ch.9"),:], dims=1);
# hc = hclust(R, linkage=:average)
# plot(hc)
# countmap(cutree(hc; k = 4))
```

``` julia
txc1 = Matrix{Float64}(x[contains.(cpgs.ID_REF,"ch.2"),:]);
```

``` julia
som = initGigaSOM(txc1, 20, 20)
som = trainGigaSOM(som, txc1)
```

    Som([0.019482827133333335 0.018569973799999997 … 0.013485004399999998 0.031320994666666664; 0.02588981066666667 0.028449281 … 0.026306426999999997 0.04085090333333333; … ; 0.504460595 0.356209895 … 0.413164381 0.484063041; 0.6750458443333333 0.5910151936666667 … 0.672976798 0.656845044], 20, 20, 400, [0.0 0.0; 1.0 0.0; … ; 18.0 19.0; 19.0 19.0])

``` julia
mapToGigaSOM(som, txc1);
```

``` julia
e = embedGigaSOM(som, txc1);
```

``` julia
#scatter(e[:,1],e[:,2], label = "", title = "SOM plot")
```
