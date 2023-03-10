Infinite GMM with Turing
================
Nicoló Foppa Pedretti

``` julia
using DataFrames, Statistics, LinearAlgebra, Plots, CategoricalArrays
using Distributions, StatsPlots, StatsBase, FillArrays
using Turing
using Turing.RandomMeasures
```

``` julia
# Define Gaussian mixture model.
w = [0.33, 0.33, 0.34]
#μ = [-3.5, 0.5, 10.0, 4.8]
μ = [[5.0, 5.0], [10.0, 0.0], [0.0, 10.0]]
#mixturemodel = MixtureModel([MvNormal(Fill(μₖ, 2), 0.2*I) for μₖ in μ], w)
mixturemodel = MixtureModel([MvNormal(μₖ, 0.1*I) for μₖ in μ], w)

# We draw the data points.
N = 80
xa = rand(mixturemodel, N);
```

``` julia
@model function gaussian_mixture_model4(x)
    # Draw the parameters for each of the K=2 clusters from a standard normal distribution.
    K = 4
    μ ~ filldist(MvNormal(Zeros(2), 5.0*I),K)
    #μ ~ Normal(0.0, 1.0)
    #μ₂ ~ MvNormal(Zeros(K), I)

    # Draw the weights for the K clusters from a Dirichlet distribution with parameters αₖ = 1.
    w ~ Dirichlet(K, 2.0)
    # Alternatively, one could use a fixed set of weights.
    # w = fill(1/K, K)

    # Construct categorical distribution of assignments.
    distribution_assignments = Categorical(w)

    # Construct multivariate normal distributions of each cluster.
    D, N = size(x)
    #distribution_clusters = [MvNormal(Fill(μₖ, D), I) for μₖ in μ]
    distribution_clusters = [MvNormal(m, 5.0*I) for m in eachcol(μ)]

    # Draw assignments for each datum and generate it from the multivariate normal distribution.
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ distribution_assignments
        x[:, i] ~ distribution_clusters[k[i]]
    end

    return k
end
```

    gaussian_mixture_model4 (generic function with 2 methods)

``` julia
model = gaussian_mixture_model4(xa);
```

``` julia
D, N = size(xa)
println(D)
println(N)
```

    2
    80

``` julia
sampler = Gibbs(PG(100, :k), HMC(0.05, 10, :μ, :w))
nsamples = 100
nchains = 3
#@time chains = sample(model, SMC(), nsamples);
#@time chains = sample(model, sampler, MCMCThreads(), nsamples, nchains);
@time chains = sample(model, sampler, nsamples);
```

    Sampling: 100%|█████████████████████████████████████████| Time: 0:25:43

    1561.291814 seconds (9.46 G allocations: 704.862 GiB, 12.76% gc time, 0.11% compilation time)

``` julia
chains;
```

``` julia
mdn = zeros(N)
for i in 1:N
    mdn[i] = median(chains[:,i+12,:])
end
```

``` julia
countmap(mdn)
```

    Dict{Float64, Int64} with 2 entries:
      4.0 => 51
      1.0 => 29
