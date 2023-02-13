Encoder - Dimensionality reduction
================
Nicoló Foppa Pedretti

``` julia
using DataFrames, Statistics, LinearAlgebra, Plots, CategoricalArrays, Flux.Data, UMAP 
using JLD2, Distributions, Parquet, StatsPlots, StatsBase, TSne, Flux, Flux.Optimise, CSV
```

``` julia
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps()) #std(A, dims=dims) #
```

    rescale (generic function with 1 method)

``` julia
DT = load_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/clead_dt.jld2");
```

``` julia
M_exp = Matrix{Float32}(DT[:,names(DT)[17:73]]);
M_exp = rescale(M_exp; dims = 1)
dl = Flux.Data.DataLoader(M_exp', batchsize=100, shuffle=true)
```

    5-element DataLoader(adjoint(::Matrix{Float64}), shuffle=true, batchsize=250)
      with first element:
      57×250 Matrix{Float64}

``` julia
r = size(M_exp)[2]
enc1 = Dense(r,150, leakyrelu)
enc2 = Dense(150,70, leakyrelu)
enc3 = Dense(70,20, leakyrelu)
enc4 = Dense(20,2, leakyrelu)
dec5 = Dense(2,20, leakyrelu)
dec6 = Dense(20,70, leakyrelu)
dec7 = Dense(70,150, leakyrelu)
dec8 = Dense(150,r)

model = Chain(enc1,enc2,
              enc3,enc4,
              dec5,dec6,
              dec7,dec8)
```

    Chain(
      Dense(57 => 150, leakyrelu),          # 8_700 parameters
      Dense(150 => 70, leakyrelu),          # 10_570 parameters
      Dense(70 => 20, leakyrelu),           # 1_420 parameters
      Dense(20 => 2, leakyrelu),            # 42 parameters
      Dense(2 => 20, leakyrelu),            # 60 parameters
      Dense(20 => 70, leakyrelu),           # 1_470 parameters
      Dense(70 => 150, leakyrelu),          # 10_650 parameters
      Dense(150 => 57),                     # 8_607 parameters
    )                   # Total: 16 arrays, 41_519 parameters, 163.184 KiB.

``` julia
loss(x) = Flux.Losses.mse(model(x), x)
loss(dl.data)
```

    0.9994493647321179

``` julia
opt = ADAM()
```

    Adam(0.001, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}())

``` julia
parameters = Flux.params(model);
```

``` julia
data = dl
```

    5-element DataLoader(adjoint(::Matrix{Float64}), shuffle=true, batchsize=250)
      with first element:
      57×250 Matrix{Float64}

``` julia
for epoch in 1:1000
    train!(loss, parameters, data, opt)
end
```

``` julia
loss(data.data)
```

    0.5291053318336782

``` julia
model_encoder = Chain(model.layers[1],model.layers[2],model.layers[3],model.layers[4])
```

    Chain(
      Dense(57 => 150, leakyrelu),          # 8_700 parameters
      Dense(150 => 70, leakyrelu),          # 10_570 parameters
      Dense(70 => 20, leakyrelu),           # 1_420 parameters
      Dense(20 => 2, leakyrelu),            # 42 parameters
    )                   # Total: 8 arrays, 20_732 parameters, 81.484 KiB.

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

``` julia
umap_model = umap(M_exp', 2)
```

    2×1004 Matrix{Float64}:
     -8.70174  -8.02622  -2.03233  -2.80472  …  9.62943  -8.8494   -1.48497
      1.10527   0.96715  -7.35212   5.71927     3.43952   0.93184  -3.13333

``` julia
scatter(umap_model[1,:], umap_model[2,:], mc=colors[x_spec], title = "Clusters UMAP", labels = "")
```

![](dim_red_aut_files/figure-commonmark/cell-19-output-1.svg)

# Variational autoencoder

``` julia
encoder = Chain(Dense(r, 20, leakyrelu), Parallel(tuple, Dense(20, 2), Dense(20, 2))) 
decoder = Chain(Dense(2, 20, leakyrelu), Dense(20, r))
encoder, decoder
```

    (Chain(Dense(57 => 20, leakyrelu), Parallel(tuple, Dense(20 => 2), Dense(20 => 2))), Chain(Dense(2 => 20, leakyrelu), Dense(20 => 57)))

``` julia
function reconstuct(x)
    μ, logσ = encoder(x) # decode posterior parameters
    ϵ = randn(Float32, size(logσ)) # sample from N(0,I)
    z = μ + ϵ .* exp.(logσ) # reparametrization
    μ, logσ, decoder(z)
end
```

    reconstuct (generic function with 1 method)

``` julia
Flux.Zygote.@nograd Flux.params
function vae_loss(λ, x)
    len = size(x)[end]
    μ, logσ, decoder_z = reconstuct(x)

    # D_KL(q(z|x) || p(z|x))
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 - 1f0 - 2f0 * logσ)) / len # from (10) in [1]

    # E[log p(x|z)]
    logp_x_z =  -Flux.Losses.logitbinarycrossentropy(decoder_z, x, agg=sum)/len

    # regularization
    reg = λ * sum(x->sum(x.^2), Flux.params(decoder))

    -logp_x_z + kl_q_p + reg
end
```

    vae_loss (generic function with 1 method)

``` julia
λ = 0.005f0
loss(x) = vae_loss(λ, x)
loss(dl.data)
```

    635.9869598273332

``` julia
η = 0.001
opt = ADAM(η) # optimizer
ps = Flux.params(encoder, decoder) # parameters
for epoch in 1:50
    train!(loss, ps, data, opt)
end
```

``` julia
loss(dl.data)
```

    -46.17380210114906

``` julia
new_enc = Chain(encoder.layers[1],encoder.layers[2])
```

    Chain(
      Dense(57 => 20, leakyrelu),           # 1_160 parameters
      Parallel(
        tuple,
        Dense(20 => 2),                     # 42 parameters
        Dense(20 => 2),                     # 42 parameters
      ),
    )                   # Total: 6 arrays, 1_244 parameters, 5.297 KiB.

``` julia
data_reducted = new_enc(dl.data);
```

    ([-4.722849514064804 -5.440047934160062 … -3.6785966685918665 -0.9226008834584022; -7.012163118383132 -7.314345771766137 … -6.0385474126273735 -1.850419474813118], [-1.6195187695575093 -2.9083639416863987 … -2.0284213250480434 0.8177029985403393; -0.12931096307856982 0.9038571583814822 … 0.6024681543392055 -0.24978452666746548])

``` julia
x_spec = recode(unwrap.(DT.h_cohort), "1"=>1::Int64, "2"=>2::Int64, "3"=>3::Int64, 
                                      "4"=>4::Int64, "5"=>5::Int64, "6"=>6::Int64);
colors=[:coral, :yellow2, :lime, :turquoise2, :magenta, :dimgray]
scatter(data_reducted[1][1,:],data_reducted[1][2,:], mc=colors[x_spec], title = "Cluster Variational Encoder", labels = "")
```

![](dim_red_aut_files/figure-commonmark/cell-28-output-1.svg)
