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
n_genes = 9 # D
n_cells = 60 # N

# create a diagonal block like expression matrix, with some non-informative genes;
# not all features/genes are informative, some might just not differ very much between cells)
mat_exp = randn(n_genes, n_cells)
mat_exp[1:(n_genes ÷ 3), 1:(n_cells ÷ 2)] .+= 10
mat_exp[(2 * (n_genes ÷ 3) + 1):end, (n_cells ÷ 2 + 1):end] .+= 10
```

    3×30 view(::Matrix{Float64}, 7:9, 31:60) with eltype Float64:
     10.1927  10.0972   11.5869   10.2483   …  10.8207    8.24761  10.2786
     10.4158  10.6889   10.4264   11.6159       9.52243  11.3454   10.3558
     10.5863   9.33311   9.27212   9.80815      9.84673   9.62128   9.17467

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
chain_ppca = sample(ppca, NUTS(), 500);
```

    ┌ Info: Found initial step size
    └   ϵ = 0.4
    Sampling: 100%|█████████████████████████████████████████| Time: 0:00:09

``` julia
chain_ppca
```

    Chains MCMC chain (500×159×1 Array{Float64, 3}):

    Iterations        = 251:1:750
    Number of chains  = 1
    Samples per chain = 500
    Wall duration     = 40.26 seconds
    Compute duration  = 40.26 seconds
    parameters        = W[1,1], W[2,1], W[3,1], W[4,1], W[5,1], W[6,1], W[7,1], W[8,1], W[9,1], W[1,2], W[2,2], W[3,2], W[4,2], W[5,2], W[6,2], W[7,2], W[8,2], W[9,2], Z[1,1], Z[2,1], Z[1,2], Z[2,2], Z[1,3], Z[2,3], Z[1,4], Z[2,4], Z[1,5], Z[2,5], Z[1,6], Z[2,6], Z[1,7], Z[2,7], Z[1,8], Z[2,8], Z[1,9], Z[2,9], Z[1,10], Z[2,10], Z[1,11], Z[2,11], Z[1,12], Z[2,12], Z[1,13], Z[2,13], Z[1,14], Z[2,14], Z[1,15], Z[2,15], Z[1,16], Z[2,16], Z[1,17], Z[2,17], Z[1,18], Z[2,18], Z[1,19], Z[2,19], Z[1,20], Z[2,20], Z[1,21], Z[2,21], Z[1,22], Z[2,22], Z[1,23], Z[2,23], Z[1,24], Z[2,24], Z[1,25], Z[2,25], Z[1,26], Z[2,26], Z[1,27], Z[2,27], Z[1,28], Z[2,28], Z[1,29], Z[2,29], Z[1,30], Z[2,30], Z[1,31], Z[2,31], Z[1,32], Z[2,32], Z[1,33], Z[2,33], Z[1,34], Z[2,34], Z[1,35], Z[2,35], Z[1,36], Z[2,36], Z[1,37], Z[2,37], Z[1,38], Z[2,38], Z[1,39], Z[2,39], Z[1,40], Z[2,40], Z[1,41], Z[2,41], Z[1,42], Z[2,42], Z[1,43], Z[2,43], Z[1,44], Z[2,44], Z[1,45], Z[2,45], Z[1,46], Z[2,46], Z[1,47], Z[2,47], Z[1,48], Z[2,48], Z[1,49], Z[2,49], Z[1,50], Z[2,50], Z[1,51], Z[2,51], Z[1,52], Z[2,52], Z[1,53], Z[2,53], Z[1,54], Z[2,54], Z[1,55], Z[2,55], Z[1,56], Z[2,56], Z[1,57], Z[2,57], Z[1,58], Z[2,58], Z[1,59], Z[2,59], Z[1,60], Z[2,60], μ[1], μ[2], μ[3], μ[4], μ[5], μ[6], μ[7], μ[8], μ[9]
    internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

    Summary Statistics
      parameters      mean       std   naive_se      mcse       ess      rhat   es ⋯
          Symbol   Float64   Float64    Float64   Float64   Float64   Float64      ⋯

          W[1,1]   -2.1074    2.1080     0.0943    0.4552    2.1185    1.4488      ⋯
          W[2,1]   -1.9512    2.0068     0.0897    0.4324    2.1245    1.4476      ⋯
          W[3,1]   -2.0456    2.0916     0.0935    0.4512    2.1121    1.4521      ⋯
          W[4,1]   -0.1756    0.2925     0.0131    0.0468   22.0908    1.0126      ⋯
          W[5,1]   -0.0124    0.2334     0.0104    0.0355    4.8480    1.1568      ⋯
          W[6,1]    0.0799    0.2585     0.0116    0.0415   11.8043    1.0847      ⋯
          W[7,1]    2.0639    2.1912     0.0980    0.4720    2.0632    1.4717      ⋯
          W[8,1]    2.0246    2.0665     0.0924    0.4453    2.1149    1.4488      ⋯
          W[9,1]    2.0992    2.1554     0.0964    0.4657    2.0828    1.4623      ⋯
          W[1,2]    1.1811    1.5574     0.0696    0.3239    1.7014    1.5955      ⋯
          W[2,2]    0.9944    1.5200     0.0680    0.3172    1.5903    1.6837      ⋯
          W[3,2]    1.0875    1.5778     0.0706    0.3287    1.6424    1.6392      ⋯
          W[4,2]    0.2972    0.3932     0.0176    0.0695    3.2025    1.3106      ⋯
          W[5,2]   -0.1148    0.3087     0.0138    0.0526    2.2282    1.4318      ⋯
          W[6,2]   -0.2233    0.3446     0.0154    0.0629    2.1960    1.4761      ⋯
          W[7,2]   -0.9670    1.7202     0.0769    0.3621    1.4868    1.8030      ⋯
          W[8,2]   -1.0949    1.5595     0.0697    0.3253    1.6561    1.6283      ⋯
          W[9,2]   -1.0888    1.6325     0.0730    0.3428    1.5739    1.7015      ⋯
          Z[1,1]   -0.8813    0.9646     0.0431    0.1948    2.6473    1.3262      ⋯
          Z[2,1]    0.5031    0.9001     0.0403    0.1362    3.5970    1.1939      ⋯
          Z[1,2]   -0.7771    0.9936     0.0444    0.1941    2.4518    1.3579      ⋯
          Z[2,2]    0.2648    0.9975     0.0446    0.1646    2.0072    1.4405      ⋯
          Z[1,3]   -0.8874    1.0276     0.0460    0.2031    2.5275    1.3376      ⋯
          ⋮           ⋮         ⋮         ⋮          ⋮         ⋮         ⋮         ⋱
                                                       1 column and 124 rows omitted

    Quantiles
      parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
          Symbol   Float64   Float64   Float64   Float64   Float64 

          W[1,1]   -3.8851   -3.5198   -3.2458   -0.8118    2.2914
          W[2,1]   -3.6044   -3.3060   -3.0351   -0.6162    2.2408
          W[3,1]   -3.8045   -3.4452   -3.1906   -0.6107    2.3279
          W[4,1]   -0.6635   -0.3600   -0.2058   -0.0359    0.5908
          W[5,1]   -0.4048   -0.1670   -0.0489    0.1258    0.5429
          W[6,1]   -0.4303   -0.0795    0.0721    0.2233    0.6027
          W[7,1]   -2.6143    0.2415    3.3149    3.5473    3.8524
          W[8,1]   -2.2832    0.7726    3.1348    3.4027    3.8038
          W[9,1]   -2.3402    0.3613    3.2864    3.5488    3.9013
          W[1,2]   -2.0235    0.1264    0.9544    2.7609    3.5692
          W[2,2]   -2.0904   -0.1081    0.8045    2.5279    3.3640
          W[3,2]   -2.1420   -0.0845    0.9360    2.6041    3.5122
          W[4,2]   -0.6605    0.1041    0.3787    0.5673    0.8559
          W[5,2]   -0.6137   -0.3504   -0.1525    0.1041    0.5090
          W[6,2]   -0.7435   -0.4749   -0.2984    0.0038    0.5124
          W[7,2]   -3.6303   -2.6535   -0.7616    0.4179    2.3846
          W[8,2]   -3.4593   -2.6669   -0.9517    0.0641    2.2177
          W[9,2]   -3.6408   -2.7284   -0.8597    0.1278    2.2162
          Z[1,1]   -1.8670   -1.5174   -1.3249   -0.5799    1.3125
          Z[2,1]   -1.4857   -0.0927    0.6414    1.2671    1.8295
          Z[1,2]   -1.7960   -1.4389   -1.2222   -0.3511    1.5766
          Z[2,2]   -1.7131   -0.4813    0.3116    1.1254    1.7360
          Z[1,3]   -1.9252   -1.5411   -1.3358   -0.5012    1.6305
          ⋮           ⋮         ⋮         ⋮         ⋮         ⋮
                                                    124 rows omitted
