Encoder - Dimensionality reduction
================
Nicoló Foppa Pedretti

``` julia
using DataFrames, Statistics, LinearAlgebra, Plots, CategoricalArrays, Flux.Data 
using JLD2, Distributions, Parquet, StatsPlots, StatsBase, TSne, Flux, Flux.Optimise
```

``` julia
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ std(A, dims=dims) #max.(std(A, dims=dims), eps())
```

    rescale (generic function with 1 method)

``` julia
DT = load_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/clead_dt.jld2");
```

``` julia
M_exp = Matrix{Float32}(DT[:,names(DT)[17:73]]);
M_exp = rescale(M_exp; dims = 1)
dl = Flux.Data.DataLoader(M_exp', batchsize=250, shuffle=true)
```

    5-element DataLoader(adjoint(::Matrix{Float32}), shuffle=true, batchsize=250)
      with first element:
      57×250 Matrix{Float32}

``` julia
r = size(M_exp)[2]
enc1 = Dense(r,20, leakyrelu)
enc2 = Dense(20,2, leakyrelu)
dec3 = Dense(2,20, leakyrelu)
dec4 = Dense(20,r)

model = Chain(enc1,enc2,dec3,dec4)
```

    Chain(
      Dense(57 => 20, leakyrelu),           # 1_160 parameters
      Dense(20 => 2, leakyrelu),            # 42 parameters
      Dense(2 => 20, leakyrelu),            # 60 parameters
      Dense(20 => 57),                      # 1_197 parameters
    )                   # Total: 8 arrays, 2_459 parameters, 10.105 KiB.

``` julia
loss(x) = Flux.Losses.mse(model(x), x)
loss(dl.data)
```

    1.0005879f0

``` julia
opt = ADAM()
```

    Adam(0.001, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}())

``` julia
parameters = Flux.params(model)
```

    Params([Float32[0.23030272 -0.19457112 … -0.059511267 0.05095589; -0.053012062 0.01915321 … 0.18896392 0.18043637; … ; 0.09016018 0.14863215 … -0.058375202 0.21529618; 0.12397997 -0.26794305 … 0.1849825 0.20901667], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Float32[-0.045471936 -0.47560918 … -0.20609686 -0.30379018; -0.40033686 -0.258891 … -0.5103229 0.18049872], Float32[0.0, 0.0], Float32[-0.25311062 0.38690624; -0.31018925 -0.4572478; … ; 0.4702428 0.101805136; -0.0010852295 0.1485091], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Float32[0.15617491 -0.05652222 … -0.053592674 0.05895641; -0.26171365 -0.075197175 … 0.022582108 0.20539138; … ; 0.05914915 -0.26010248 … -0.19959816 0.061366513; 0.09128494 -0.0963006 … 0.2082069 -0.13272396], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

``` julia
data = dl
```

    5-element DataLoader(adjoint(::Matrix{Float32}), shuffle=true, batchsize=250)
      with first element:
      57×250 Matrix{Float32}

``` julia
for epoch in 1:2000
    train!(loss, parameters, data, opt)
end
```

``` julia
loss(data.data)
```

    0.61525863f0

``` julia
model_encoder = Chain(model.layers[1],model.layers[2],model.layers[3])
```

    Chain(
      Dense(57 => 20, leakyrelu),           # 1_160 parameters
      Dense(20 => 2, leakyrelu),            # 42 parameters
      Dense(2 => 20, leakyrelu),            # 60 parameters
    )                   # Total: 6 arrays, 1_262 parameters, 5.305 KiB.

``` julia
dt_red = model_encoder(dl.data);
```

``` julia
x_spec = recode(unwrap.(DT.h_cohort), "1"=>1::Int64, "2"=>2::Int64, "3"=>3::Int64, 
                                      "4"=>4::Int64, "5"=>5::Int64, "6"=>6::Int64);
colors=[:coral, :yellow2, :lime, :turquoise2, :magenta, :dimgray]
scatter(dt_red[1,:],dt_red[2,:], mc=colors[x_spec], title = "Cluster Encoder", labels = "")
```

![](dim_red_aut_files/figure-commonmark/cell-15-output-1.svg)

``` julia
y_tsne = tsne(M_exp, 2, 57, 1000, 30, pca_init = true, progress = false);   
```

``` julia
scatter(y_tsne[:,1], y_tsne[:,2], mc=colors[x_spec], title = "Clusters T-Sne", labels = "")
```

![](dim_red_aut_files/figure-commonmark/cell-17-output-1.svg)
