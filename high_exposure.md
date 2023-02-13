High dimensional environmental exposure
================
Nicoló Foppa Pedretti

``` julia
using DataFrames, Statistics, LinearAlgebra, Plots, CategoricalArrays, Flux.Data, UMAP 
using JLD2, Distributions, Parquet, StatsPlots, StatsBase, TSne, Flux, Flux.Optimise, CSV
```

``` julia
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ std(A, dims=dims)
```

    rescale (generic function with 1 method)

``` julia
DT = load_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/clead_dt.jld2");
```

``` julia
genexpr = CSV.read("C:/Users/nicol/Documents/Metabolomics/data/tables_format/genexpr_num.csv",DataFrame);
genexpr_cov = DataFrame(read_parquet("C:/Users/nicol/Documents/Metabolomics/data/tables_format/genexpr_cov.parquet"));
```

``` julia
genexpr_t = permutedims(genexpr,1);
rename!(genexpr_t,:rn => :ID);
```

``` julia
DT = innerjoin(genexpr_cov,DT, on = :ID);
DT = innerjoin(DT, genexpr_t, on = :ID);
DT = DT[completecases(DT),:];
```

``` julia
exposure_label = names(DT)[28:84];
```

``` julia
M_exp = Matrix{Float32}(DT[:,exposure_label]);
M_exp = rescale(M_exp);
```

``` julia
length(names(DT)[17:73])^3 + length(names(DT)[17:73])^2 + length(names(DT)[17:73]) 
```

    188499

``` julia
w3 = reshape([[x,y,z]  for x=1:57, y=1:57, z=1:57],57*57*57)
tmp3 = mapreduce(permutedims, vcat, w3);
E3 = fill(NaN, size(DT)[1],size(tmp3)[1])
Threads.@threads for i in 1:size(tmp3)[1]
    E3[:,i] = M_exp[:,tmp3[i,1]] .* M_exp[:,tmp3[i,2]] .* M_exp[:,tmp3[i,3]]
end
```

``` julia
w2 = reshape([[x,y]  for x=1:57, y=1:57],57*57)
tmp2 = mapreduce(permutedims, vcat, w2);
E2 = fill(NaN, size(DT)[1],size(tmp2)[1])
Threads.@threads for i in 1:size(tmp2)[1]
    E2[:,i] = M_exp[:,tmp2[i,1]] .* M_exp[:,tmp2[i,2]]
end
```

``` julia
EM = hcat(M_exp,E2,E3);
```

``` julia
RExp = vcat(exposure_label,
            reshape([exposure_label[i] * string(" × ") * exposure_label[j]  for i=1:57, j=1:57],57*57,1),
            reshape([exposure_label[i] * string(" × ") * exposure_label[j] * string(" × ") * exposure_label[k]
                    for i=1:57, j=1:57, k=1:57],57*57*57,1)
    );
RExp = hcat(RExp,fill(NaN,length(RExp),6));
```

``` julia
Threads.@threads for i in 1:size(RExp)[1]
    X_tmp = Array{Float32, 2}(hcat(ones(size(DT)[1]),
        EM[:,i],
        Matrix{Float64}(DT[:,names(DT)[[3,5,6,7,8,9,10,11,12]]])))
    Y_tmp = convert.(Float32, DT[:,"TC01000001.hg.1"])

    β = X_tmp\Y_tmp
    σ² = sum((Y_tmp - X_tmp*β).^2)/(size(X_tmp,1)-size(X_tmp,2))
    Σ = σ²*inv(X_tmp'*X_tmp)
    std_coeff = sqrt.(diag(Σ))
    
    RExp[i,2] = β[2]
    RExp[i,3] = std_coeff[2]
    RExp[i,4] = β[2]/std_coeff[2]
    RExp[i,5] = cdf(TDist(size(X_tmp,1)-size(X_tmp,2)), -abs(β[2]/std_coeff[2]))
    RExp[i,6] = β[2] - quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
    RExp[i,7] = β[2] + quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
end
```

``` julia
ewas_results = DataFrame(RExp,:auto)
rename!(ewas_results, ["variable","beta","std_error","t_value","p_value","CI0025","CI0975"]);
ewas_results.p_value_bonf = (ewas_results.p_value .< 0.05/size(RExp)[1]);
```

``` julia
ewas_results
```

``` julia
filter([:p_value_bonf] => (x) -> x == 1,ewas_results)
```
