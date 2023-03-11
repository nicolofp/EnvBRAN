pPCA with Turing
================
Nicoló Foppa Pedretti

``` julia
using DataFrames, Statistics, LinearAlgebra, Plots, CategoricalArrays
using Distributions, StatsPlots, StatsBase, FillArrays
using Turing
using Turing.RandomMeasures
using ReverseDiff
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
```

    true

``` julia
n_genes = 90 # D
n_cells = 600 # N

# create a diagonal block like expression matrix, with some non-informative genes;
# not all features/genes are informative, some might just not differ very much between cells)
mat_exp = randn(n_genes, n_cells)
mat_exp[1:(n_genes ÷ 3), 1:(n_cells ÷ 2)] .+= 10
mat_exp[(2 * (n_genes ÷ 3) + 1):end, (n_cells ÷ 2 + 1):end] .+= 10
```

    30×300 view(::Matrix{Float64}, 61:90, 301:600) with eltype Float64:
     10.4372   11.1014   11.5422    9.5042   …  10.7548    9.63808   9.32686
     10.5018    8.72323   9.20409   9.85832      9.5771    9.92837   9.52497
     10.6838    9.27582   9.44461  10.848        8.77913   8.35863  11.4777
      9.85805   9.93355   9.26867   9.9755      10.724     9.7402    8.64637
     10.1188   10.1114    9.01299  10.5917      11.4272    9.44002  11.526
     10.2604   10.8684    9.82632  10.2331   …  11.014    10.4732   12.2661
      9.542     8.8815    8.95899  10.855        9.67726  11.798    11.2646
      9.94663   9.28141   8.47277   9.81985     11.0609    9.29915   9.07441
      8.87878  11.0274    8.5358    9.02455     10.6759   11.3455    9.63063
      9.83308  10.3702    9.28024   9.15991     10.0571    8.50058   9.29979
     10.8429   10.5215    9.71645   9.88254  …  10.9902   10.5415    9.22076
      9.89647  10.9806    9.76403   9.33033     12.1684   10.3105   10.6301
     11.0198    9.41047  10.4953   10.0184      10.55     10.4854    9.08032
      ⋮                                      ⋱                      
      8.24289  10.9547   10.3634   11.3795      10.2155   12.4285   11.9429
     11.4568    9.55312  11.2371   12.8296       9.92853  10.4461    9.7588
     10.7421    8.15343   9.49805  11.0752   …   7.80246   7.68998   9.75765
     13.0041    9.84122  10.8109    9.01021      8.75896   9.94514  10.2952
      9.53593   9.55516   9.99518  10.4065       9.98954   9.99749   8.77517
     11.2628   10.0339    9.95404  10.0441       9.32931  11.0174   10.0015
     10.9189    8.92746   9.29106  11.5351       9.40364  11.2907   10.2804
      9.96828   9.38394   8.55461  10.9156   …  10.5855    9.39128   8.82218
      9.05718   6.66317  11.208    12.6037       7.8324    7.48477   9.81401
     10.4341   11.1426   10.2583   11.26        11.6794    8.20182   9.91996
      9.34408   9.50967  10.3277    7.28627      9.17858  10.2203    9.06836
      9.48676   9.40314   8.56982   8.97879      8.73077  10.269    10.1814

``` julia
@model function pPCA(X::AbstractMatrix{<:Real}, k::Int)
    # retrieve the dimension of input matrix X.
    N, D = size(X)

    # weights/loadings W
    W ~ filldist(Normal(), D, k)

    # latent variable z
    Z ~ filldist(Normal(), k, N)

    # mean offset
    μ ~ MvNormal(Eye(D))
    genes_mean = W * Z .+ reshape(μ, n_genes, 1)
    return X ~ arraydist([MvNormal(m, Eye(N)) for m in eachcol(genes_mean')])
end;
```

``` julia
@model function pPCA_ARD(X)
    # Dimensionality of the problem.
    N, D = size(X)

    # latent variable Z
    Z ~ filldist(Normal(1.0, 0.0), D, N)

    # weights/loadings w with Automatic Relevance Determination part
    α ~ filldist(Gamma(1.0, 1.0), D)
    W ~ filldist(MvNormal(zeros(D), 1.0 ./ sqrt.(α)), D)

    mu = (W' * Z)'

    tau ~ Gamma(1.0, 1.0)
    return X ~ arraydist([MvNormal(m, 1.0 / sqrt(tau)) for m in eachcol(mu)])
end;
```

``` julia
k = 2 # k is the dimension of the projected space, i.e. the number of principal components/axes of choice
ppca = pPCA(mat_exp', k) # instantiate the probabilistic model
chain_ppca = sample(ppca, NUTS(), 1000);
```

    ┌ Info: Found initial step size
    └   ϵ = 0.05
    Sampling: 100%|█████████████████████████████████████████| Time: 1:14:31
