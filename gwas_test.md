GWAS - cpgs
================
Nicoló Foppa Pedretti

``` julia
using DataFrames, Statistics, LinearAlgebra, Plots, Clustering, TSne, DecisionTree, MLDataUtils
using Distributions, StatsPlots, StatsBase, FillArrays, Arrow, UMAP, Distances, MLBase
```

``` julia
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps()) #std(A, dims=dims) #
```

    rescale (generic function with 1 method)

``` julia
df = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets/GSE71678/Data/processed_data/GSE71678_cov_original.arrow"));
```

``` julia
df = df[completecases(df[:,["gestational age","infant gender","birth weight (grams)"]]),:];
```

``` julia
df.gestage_cat = ceil.(df[:,"gestational age"]);
```

``` julia
df[df.gestage_cat .< 37,:gestage_cat] = 36*ones(11)
df[df.gestage_cat .> 41,:gestage_cat] = 42*ones(40);
```

``` julia
countmap(df[:,"infant gender"])
histogram(df[:,"gestational age"])
countmap(df[:,"gestage_cat"])
```

    Dict{Float64, Int64} with 7 entries:
      39.0 => 52
      41.0 => 75
      36.0 => 11
      38.0 => 26
      42.0 => 40
      37.0 => 9
      40.0 => 126

``` julia
zscore_table = combine(groupby(df,["gestage_cat","infant gender"]), ["birth weight (grams)"] .=> (mean,std));
```

``` julia
df = innerjoin(df, zscore_table, 
               on = ["gestage_cat","infant gender"]);
```

``` julia
df.bw_zscore = (df[:,"birth weight (grams)"] .- df[:,"birth weight (grams)_mean"])./df[:,"birth weight (grams)_std"];
```

``` julia
cpgs = DataFrame(Arrow.Table("C:/Users/nicol/Documents/Datasets/GSE71678/Data/processed_data/GSE71678_cpgs_original.arrow"));
```

``` julia
cpgs = cpgs[:,df.ID];
```

``` julia
x = Matrix{Float64}(cpgs);
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
df = df[completecases(df[:,["bw_zscore","mom_education","parity","ever_smoker","gestational_diabetes","maternal age",
                "placenta_al","placenta_cr","placenta_ni","placenta_se","placenta_co","placenta_pb"]]),:];
```

``` julia
countmap(df[:,"gestational_diabetes"])
df.gest_diabetis = df[:,"gestational_diabetes"] .== "Yes";
```

``` julia
countmap(df[:,"ever_smoker"])
df.smoking = df[:,"ever_smoker"] .== "Yes";
```

``` julia
countmap(df[:,"parity"])
df.parity_bin = df[:,"parity"] .> 0;
```

``` julia
countmap(df[:,"maternal race"])
```

    Dict{Union{Missing, String}, Int64} with 2 entries:
      "Other/Unknown" => 9
      "White"         => 302

``` julia
countmap(df[:,"mom_education"])
df.low_education = [df[i,"mom_education"] ∈ ["<11th grade","HS grad","Jr College/technical school"] for i in 1:size(df,1)];
```

``` julia
dft = df[:,["bw_zscore","low_education","parity_bin","smoking","gest_diabetis","maternal age",
            "placenta_al","placenta_cr","placenta_ni","placenta_se","placenta_co","placenta_pb"]];
```

``` julia
y = Array{Float64}(dft[:,"bw_zscore"])
z = hcat(ones(size(dft,1)),Matrix{Float64}(dft[:,2:size(dft,2)]));
```

``` julia
tmp = vcat("Intercept",names(dft)[2:size(dft,2)])
results = Array{Float64}(undef,size(dft,2), 6);
results = hcat(tmp,results);
```

``` julia
β = z\y
σ² = sum((y - z*β).^2)/(size(z,1)-size(z,2))
Σ = σ²*inv(z'*z)
std_coeff = sqrt.(diag(Σ))

results[:,2] = β
results[:,3] = std_coeff
results[:,4] = β./std_coeff
results[:,5] = [cdf(TDist(size(z,1)-size(z,2)), -abs(β[i]./std_coeff[i])) for i in 1:size(z,2)]
results[:,6] = β .- quantile(TDist(size(z,1)-size(z,2)), 0.975) .* std_coeff
results[:,7] = β .+ quantile(TDist(size(z,1)-size(z,2)), 0.975) .* std_coeff;
```

``` julia
results = DataFrame(results,:auto)
rename!(results, ["variable","beta","std_error","t_value","p_value","CI0025","CI0975"])
```

``` julia
ssres = sum((y - z*β).^2)
sstot = sum((y .- mean(y)).^2)
R² = 1 .- (ssres/sstot)
```

    0.1505206725697913

``` julia
#Fast linear regression code (can be parallelized to make it even faster)
@time for i in 1:N
    X_tmp = Array{Float64, 2}(hcat(ones(size(DT)[1]),
        DT[:,ewas_results[i,2]],
        Matrix{Float64}(DT[:,[:ethn_PC1, :ethn_PC2, :NK_6, :Bcell_6, 
                    :CD4T_6, :CD8T_6, :Gran_6, :Mono_6]])))
    Y_tmp = convert.(Float64, DT[:,ewas_results[i,1]])

    β = X_tmp\Y_tmp
    σ² = sum((Y_tmp - X_tmp*β).^2)/(size(X_tmp,1)-size(X_tmp,2))
    Σ = σ²*inv(X_tmp'*X_tmp)
    std_coeff = sqrt.(diag(Σ))
    
    ewas_results[i,3] = β[2]
    ewas_results[i,4] = std_coeff[2]
    ewas_results[i,5] = β[2]/std_coeff[2]
    ewas_results[i,6] = cdf(TDist(size(X_tmp,1)-size(X_tmp,2)), -abs(β[2]/std_coeff[2]))
    ewas_results[i,7] = β[2] - quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
    ewas_results[i,8] = β[2] + quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
end
```
