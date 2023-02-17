RxInfer for bayesian analysis
================
Nicoló Foppa Pedretti

``` julia
using RxInfer, DataFrames, Statistics, LinearAlgebra, Plots
using JLD2, Distributions, Parquet, StatsBase, StableRNGs
```

``` julia
DT = load_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/clead_dt.jld2");
```

``` julia
enames = names(DT)[vcat([1,2,6],17:73)]
Exposure = Matrix{Float64}(DT[:,enames])
bw = Vector{Float64}(DT.bw_zscore);
```

``` julia
n = size(Exposure)[1]
m = size(Exposure)[2]

@model function multivariate_linear_regression(n,m)
    a ~ MvNormalMeanCovariance(zeros(m), diagm(ones(m)))
    b ~ NormalMeanVariance(0.0,1.0)
    W ~ InverseWishart(n+2, diageye(n))
    c ~ ones(n)*b
    x = datavar(Matrix{Float64})
    y = datavar(Vector{Float64})
    z ~ x*a+c
    y ~ MvNormalMeanCovariance(z, W)

end
```

``` julia
results = inference(
    model = multivariate_linear_regression(n,m),
    data  = (y = bw, x = Exposure),
    initmarginals = (W = InverseWishart(n+2, diageye(n)), ),
    returnvars   = (a = KeepLast(), b = KeepLast(), W = KeepLast()),
    free_energy = true,
    iterations   = 20,
    constraints = MeanField()
)
```

    Inference results:
      Posteriors       | available for (a, b, W)
      Free Energy:     | Real[-1.46026e5, -1.67359e5, -1.71006e5, -1.71635e5, -1.71745e5, -171764.0, -1.71768e5, -1.71768e5, -1.71769e5, -1.71769e5, -1.71769e5, -1.71769e5, -1.71769e5, -1.71769e5, -1.71769e5, -1.71769e5, -1.71769e5, -1.71769e5, -1.71769e5, -1.71769e5]

``` julia
# truncate the init step
# f = plot(results.free_energy[2:end], title ="Bethe Free Energy convergence", label = nothing) 
# plot(f, size = (400, 300))
```

``` julia
# [[quantile(as[i,:],0.025),mean(as[i,:]),quantile(as[i,:],0.975)] for i in 1:60]
as = rand(results.posteriors[:a], 10000)
Mean = [mean(as[i,:]) for i in 1:60]
LCI = [quantile(as[i,:],0.025) for i in 1:60]
UCI = [quantile(as[i,:],0.975) for i in 1:60]
DT_results = DataFrame(Variable = enames, Mean = Mean, CI0025 = LCI, CI0975 = UCI)
DT_results.sig = sign.(DT_results.CI0025) .== sign.(DT_results.CI0975)
show(DT_results[DT_results.sig .== 1,:])
```

    5×5 DataFrame
     Row │ Variable           Mean        CI0025      CI0975       sig  
         │ String             Float64     Float64     Float64      Bool 
    ─────┼──────────────────────────────────────────────────────────────
       1 │ h_mbmi_None         0.0221921   0.0074149   0.0370993   true
       2 │ hs_wgtgain_None     0.0215143   0.0112933   0.0317935   true
       3 │ hs_cs_m_Log2       -0.143476   -0.283603   -0.00454701  true
       4 │ hs_dep_madj_Log2   -0.0459427  -0.0872681  -0.00477563  true
       5 │ hs_dmtp_madj_Log2   0.0512991   0.0267493   0.0752538   true

``` julia
betas = vcat(mean(results.posteriors[:b]),mean(results.posteriors[:a]))
stds = vcat(std(results.posteriors[:b]),diag(std(results.posteriors[:a])))

DT_bayes = DataFrame(Names = vcat("intercept",enames), beta = betas, std = stds, 
                   p_value = cdf.(TDist(size(Exposure,1)-size(Exposure,2)), -abs.(betas./stds)))
show(DT_bayes[DT_bayes.p_value .< 0.05,:])
```

    13×4 DataFrame
     Row │ Names                    beta        std         p_value    
         │ String                   Float64     Float64     Float64    
    ─────┼─────────────────────────────────────────────────────────────
       1 │ h_mbmi_None               0.0221632  0.0076396   0.00190246
       2 │ hs_wgtgain_None           0.0214829  0.0051794   1.82997e-5
       3 │ hs_cs_m_Log2             -0.142373   0.0696021   0.0205396
       4 │ h_temperature_preg_None   0.0170185  0.00991685  0.043234
       5 │ hs_dde_madj_Log2         -0.0336264  0.0188159   0.037119
       6 │ hs_pcb170_madj_Log2       0.0736856  0.0378712   0.0259942
       7 │ hs_dep_madj_Log2         -0.0461144  0.0208658   0.0136707
       8 │ hs_detp_madj_Log2        -0.01432    0.00816437  0.0398809
       9 │ hs_dmtp_madj_Log2         0.0513992  0.0106629   8.34541e-7
      10 │ hs_pfhxs_m_Log2           0.0524051  0.0267013   0.0249905
      11 │ hs_pfos_m_Log2           -0.068099   0.0331466   0.0201023
      12 │ hs_mepa_madj_Log2        -0.0372125  0.0192635   0.026844
      13 │ hs_sumDEHP_madj_Log2     -0.0691889  0.0293367   0.00927737

``` julia
X_tmp = Matrix{Float32}(hcat(ones(size(DT)[1]),DT[:,enames]))
Y_tmp = convert.(Float32, DT[:,:bw_zscore])

β = X_tmp\Y_tmp
σ² = sum((Y_tmp - X_tmp*β).^2)/(size(X_tmp,1)-size(X_tmp,2))
Σ = σ²*inv(X_tmp'*X_tmp)
std_coeff = sqrt.(diag(Σ))

DT_inf = DataFrame(Names = vcat("intercept",enames), beta = β, std = std_coeff, 
                   p_value = cdf.(TDist(size(X_tmp,1)-size(X_tmp,2)), -abs.(β./std_coeff)))
show(DT_inf[DT_inf.p_value .< 0.05,:])
```

    10×4 DataFrame
     Row │ Names                beta        std         p_value    
         │ String               Float32     Float32     Float64    
    ─────┼─────────────────────────────────────────────────────────
       1 │ h_mbmi_None           0.0220775  0.00728344  0.00125125
       2 │ hs_wgtgain_None       0.0216013  0.0049429   6.8968e-6
       3 │ hs_cs_m_Log2         -0.15808    0.0692241   0.0113085
       4 │ hs_dde_madj_Log2     -0.0342088  0.0190084   0.0361162
       5 │ hs_pcb170_madj_Log2   0.0786626  0.0423215   0.0316912
       6 │ hs_dep_madj_Log2     -0.0468959  0.0202755   0.0104703
       7 │ hs_detp_madj_Log2    -0.0144531  0.0082087   0.0393059
       8 │ hs_dmtp_madj_Log2     0.0517077  0.0117527   6.03949e-6
       9 │ hs_pfhxs_m_Log2       0.0540166  0.0263145   0.020187
      10 │ hs_mepa_madj_Log2    -0.0380767  0.0192342   0.0240174
