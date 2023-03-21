**Bayesian ARD regression using Julia**
================
Nicoló Foppa Pedretti

### Introduction

Load the libraries used in the script:

``` julia
using Turing, Plots, LinearAlgebra, Statistics, Parquet, CSV
using DataFrames, Optim, Zygote, ReverseDiff, Arrow
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
```

    true

### Automatic Relevance Determination linear bayesian regression

``` julia
# Bayesian linear regression with variable selection
@model function linear_regression_ard(x, y)
    # Set variance prior.
    σ ~ Gamma(1.0,1.0)

    # Set intercept prior.
    intercept ~ Normal(0, sqrt(3))

    # Set the priors on our coefficients.
    nfeatures = size(x, 2)
    alpha ~ filldist(Gamma(2.0,2.0),nfeatures)
    coefficients ~ MvNormal(ones(nfeatures), 1.0 ./ sqrt.(alpha))

    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    y ~ MvNormal(mu, σ)
end
```

    linear_regression_ard (generic function with 2 methods)

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
m = linear_regression_ard(Xdt,Ydt)
```

    DynamicPPL.Model{typeof(linear_regression_ard), (:x, :y), (), (), Tuple{Matrix{Float64}, Vector{Float64}}, Tuple{}, DynamicPPL.DefaultContext}(linear_regression_ard, (x = [25.5102043151855 17.0 … -0.53671336807613 0.2880726323194142; 26.4915084838867 18.0 … 1.078749679027982 3.1954783677431906; … ; 27.3796653747559 20.0 … -0.5158381392316987 0.33732010720086447; 20.7967300415039 20.0 … 2.8622580221540206 2.9194911007009714], y = [0.4521301802575446, 0.6817836051502657, 0.1902644116339792, 0.9375193793664165, -0.18199614453022778, -0.644749766556538, -0.6727508181580242, 0.7978075164781839, 0.45836965210587055, 0.8876797791920156  …  -2.5858993789239384, -1.2539995980339271, 0.16827521450578153, -1.5179237735655002, 0.5135426034748191, 0.39201541151351876, -1.6064729645602154, -1.3679101117174621, 0.3254248423856985, 0.2851462985582067]), NamedTuple(), DynamicPPL.DefaultContext())

``` julia
reg_ard = sample(m, NUTS(), 1000);
```

    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\iWHPQ\src\hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\iWHPQ\src\hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\iWHPQ\src\hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\iWHPQ\src\hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\iWHPQ\src\hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\iWHPQ\src\hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\iWHPQ\src\hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\iWHPQ\src\hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, true, false, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\iWHPQ\src\hamiltonian.jl:47
    ┌ Info: Found initial step size
    └   ϵ = 2.44140625e-5
    Sampling: 100%|█████████████████████████████████████████| Time: 0:03:58

``` julia
results = Matrix(undef,122,10);
```

``` julia
for i in 1:122
    results[i,1] = string.(reg_ard.name_map.parameters[i])
    results[i,2] = mean(reg_ard.value[1:1000,i,1])
    results[i,3] = std(reg_ard.value[1:1000,i,1])
    results[i,4] = quantile(reg_ard.value[1:1000,i,1],0.025)
    results[i,5] = quantile(reg_ard.value[1:1000,i,1],0.1)
    results[i,6] = quantile(reg_ard.value[1:1000,i,1],0.5)
    results[i,7] = quantile(reg_ard.value[1:1000,i,1],0.8)
    results[i,8] = quantile(reg_ard.value[1:1000,i,1],0.975)
    results[i,9] = summarystats(reg_ard)[:,6][i]
    results[i,10] = summarystats(reg_ard)[:,7][i]
end
```

``` julia
tmp = DataFrame(results,:auto)
rename!(tmp,["parameters","mean","sd","2.5%","10%","50%","90%","97.5%","ess","rhat"]);
```

``` julia
# bar(1:60,tmp[occursin.("alpha", tmp.parameters),:mean])
coeffs = tmp[occursin.("coefficients", tmp.parameters),:]
coeffs.parameters = names(df)[vcat(collect(1:2),6,collect(16:72))] 
coeffs.significant = sign.(coeffs[:,"2.5%"]) .== sign.(coeffs[:,"97.5%"])
coeffs[coeffs.significant .== true,:]
```
