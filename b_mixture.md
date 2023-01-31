Bayesian mixture
================
Nicoló Foppa Pedretti

Import clean dataset and analyze the mixture of the chemicals. Let’s
upload the packages that we’ll use in the code

``` julia
using DataFrames, Statistics, Turing, LinearAlgebra, Plots, CategoricalArrays 
using JLD2, Distributions, Parquet, StatsPlots, StatsBase, Optim, Zygote, ReverseDiff, Memoization
```

Load the bayesian models in the external files. We are using `Turing.jl`
a Julia library for bayesian inference.

``` julia
include("bayes_lib.jl")
```

    bwqs_adv (generic function with 2 methods)

Import the clean dataset to start the analysis

``` julia
DT = load_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/clead_dt.jld2");
```

Let’s select metals and covariates for the analysis: the metals selected
are **As**,c **Cd**, **Co**, **Cs**, **Cu**, **Hg**, **Mn**, **Mo**,
**Pb**. The covariates are **Maternal BMI pre-pregnancy**,**Weight
gained during pregnancy**,**Maternal age**. The outcome is the birth
weight z-score.

``` julia
metals = names(DT)[contains.(names(DT),"m_Log2")][1:9];
covar = names(DT)[[1,2,6]];
outcome = "bw_zscore";
```

``` julia
mx = Matrix{Float64}(DT[:, metals]);
cx = Matrix{Float64}(DT[:, covar]); 
y = Vector{Float64}(DT[:,outcome]);
```

``` julia
for i in 1:size(metals)[1]
    mx[:,i] = ecdf(mx[:,i])(mx[:,i])*10
end
```

``` julia
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
```

    true

``` julia
model = bwqs(cx, mx, y);
model_adv = bwqs_adv(cx, mx, y);
```

``` julia
# chain = sample(model, NUTS(0.65), 5000, thinning = 3);
```

``` julia
#chain_adv = sample(model_adv, NUTS(0.65), 5000, thinning = 3);
```

``` julia
# save_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/mix_chain.jld2", chain)
# save_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/mix_adv_chain.jld2", chain_adv)
```

``` julia
chain = load_object(":/Users/nicol/Documents/Github_projects/EnvBRAN/mix_chain.jld2");
chain_adv = load_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/mix_adv_chain.jld2");
```

    ┌ Warning: Opening file with JLD2.MmapIO failed, falling back to IOStream
    └ @ JLD2 C:\Users\nicol\.julia\packages\JLD2\1YVED\src\JLD2.jl:286

    LoadError: SystemError: opening file ":/Users/nicol/Documents/Github_projects/EnvBRAN/mix_chain.jld2": Invalid argument

``` julia
hcat(quantile(chain_adv.value[:,3,1][:],[0.025,0.5,0.975]),
     quantile(chain.value[:,3,1][:],[0.025,0.5,0.975]))
```

    LoadError: UndefVarError: chain_adv not defined

``` julia
DataFrame(Metals = metals,
     w = mean(Matrix(chain.value[:,7:15,1]), dims = 1)'[:],
     w_adv = mean(Matrix(chain_adv.value[:,16:24,1]), dims = 1)'[:])  
```

    LoadError: UndefVarError: chain not defined