Bayesian mixture
================
Nicoló Foppa Pedretti

Import clean dataset and analyze the mixture of the chemicals. Let’s
upload the packages that we’ll use in the code

``` julia
using DataFrames, Statistics, LinearAlgebra, Plots, CategoricalArrays 
using JLD2, Distributions, Parquet, StatsPlots, StatsBase
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
are **As**, **Cd**, **Co**, **Cs**, **Cu**, **Hg**, **Mn**, **Mo**,
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
# save_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/mix_adv_chain.jld2", chain_adv)
# save_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/mix_sim_chain.jld2", chain)
```

``` julia
chain = load_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/mix_sim_chain.jld2");
```

``` julia
chain_adv = load_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/mix_adv_chain.jld2");
```

``` julia
#vcat(quantile(chain_adv.value[:,3,1][:],[0.025,0.5,0.975])',
#     quantile(chain.value[:,3,1][:],[0.025,0.5,0.975])')
print("bwqs: ",round(quantile(chain.value[:,3,1][:],0.5), digits = 4),
      " (",round(quantile(chain.value[:,3,1][:],0.025), digits = 4),"; ",
        round(quantile(chain.value[:,3,1][:],0.975), digits = 4),")\n")
print("bwqs_adv: ",round(quantile(chain_adv.value[:,3,1][:],0.5), digits = 4),
      " (",round(quantile(chain_adv.value[:,3,1][:],0.025), digits = 4),"; ",
        round(quantile(chain_adv.value[:,3,1][:],0.975), digits = 4),")")
```

    bwqs: -0.0137 (-0.0676; 0.0422)
    bwqs_adv: 0.0007 (-0.0421; 0.04)

``` julia
show(DataFrame(Metals = metals,
     w = mean(Matrix(chain.value[:,7:15,1]), dims = 1)'[:],
     w_adv = mean(Matrix(chain_adv.value[:,16:24,1]), dims = 1)'[:]))  
```

    9×3 DataFrame
     Row │ Metals        w          w_adv      
         │ String        Float64    Float64    
    ─────┼─────────────────────────────────────
       1 │ hs_as_m_Log2  0.0994072  0.00266723
       2 │ hs_cd_m_Log2  0.118124   0.0
       3 │ hs_co_m_Log2  0.103951   0.0559402
       4 │ hs_cs_m_Log2  0.137375   0.0156049
       5 │ hs_cu_m_Log2  0.109595   0.0837208
       6 │ hs_hg_m_Log2  0.0991375  0.0950911
       7 │ hs_mn_m_Log2  0.102253   0.0533021
       8 │ hs_mo_m_Log2  0.0995249  0.375829
       9 │ hs_pb_m_Log2  0.130633   0.317845
