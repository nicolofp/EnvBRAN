High dimensional elastic-net regression
================
Nicoló Foppa Pedretti

``` julia
using DataFrames, Statistics, LinearAlgebra, Plots, CategoricalArrays, Distances
using Distributions, JSON, HTTP, StatsPlots, StatsBase, Parquet, CSV, Convex, SCS 
```

``` julia
codebook = CSV.read(download("https://envbran.s3.us-east-2.amazonaws.com/codebook.tsv"), DataFrame);
phenotype = CSV.read(download("https://envbran.s3.us-east-2.amazonaws.com/phenotype.tsv"), DataFrame);
met_urine = CSV.read(download("https://envbran.s3.us-east-2.amazonaws.com/met_urine.tsv"), DataFrame);
exposure = CSV.read(download("https://envbran.s3.us-east-2.amazonaws.com/covariates.tsv"), DataFrame);
```

``` julia
tmp = filter([:period, :var_type, :domain, :period_postnatal, :family, :subfamily] => (x,y,z,w,p,r) -> x == "Postnatal" && 
        y == "numeric" && z != "Covariates" && z!= "Phenotype" && (w == "NA" || contains.(w, "Year")) && 
       !(contains.(p,"Traffic") || contains.(p,"Social and economic capital") || contains.(p,"Built environment") || 
        contains.(p,"Indoor air") || contains.(p,"Lifestyle") || contains.(p,"Natural Spaces")),
        codebook);
```

``` julia
var_name = vcat("ID",tmp.variable_name)
environmental = exposure[:,var_name];
```

``` julia
met_urine = permutedims(met_urine[:,vcat(1,8:1199)],1)
names(met_urine)[2:45];
```

``` julia
met_urine.rn = parse.(Int64, met_urine.rn);
```

``` julia
DT = innerjoin(environmental, met_urine, on = :ID => :rn);
names(DT);
```

``` julia
M = cor(Matrix{Float64}(DT[:,2:size(DT)[2]]))       # correlation matrix
MC = Matrix{Float64}(M[1:(length(var_name)-1),length(var_name):(length(var_name)+43)])
# PLOT
var_name_ = tmp.labels
met_urine_ = names(met_urine)[2:45]
(n,m) = size(MC)
heatmap(MC, fc=cgrad(:plasma); #[:white,:dodgerblue4]); 
        yticks=(1:n,var_name_), xrot=90, 
        xticks=(1:m,met_urine_), size = (700,500),tickfontsize = 6)
```

![](umet_exposure_files/figure-commonmark/cell-9-output-1.svg)

Lasso implementation:

![\sum\_{i=1}^{n} \left( y_i - \beta_0 - \sum\_{j=1}^{p} \beta_j x\_{ij}    \right) ^ 2 + \lambda\|\|\beta\|\|\_2^2 + \nu \|\|\beta\|\|\_1](https://latex.codecogs.com/svg.latex?%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cleft%28%20y_i%20-%20%5Cbeta_0%20-%20%5Csum_%7Bj%3D1%7D%5E%7Bp%7D%20%5Cbeta_j%20x_%7Bij%7D%20%20%20%20%5Cright%29%20%5E%202%20%2B%20%5Clambda%7C%7C%5Cbeta%7C%7C_2%5E2%20%2B%20%5Cnu%20%7C%7C%5Cbeta%7C%7C_1 "\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij}    \right) ^ 2 + \lambda||\beta||_2^2 + \nu ||\beta||_1")

``` julia
"""
    LassoEN(Y,X,γ,λ)

Do Lasso (set γ>0,λ=0), ridge (set γ=0,λ>0) or elastic net regression (set γ>0,λ>0).


# Input
- `Y::Vector`:     T-vector with the response (dependent) variable
- `X::VecOrMat`:   TxK matrix of covariates (regressors)
- `γ::Number`:     penalty on sum(abs.(b))
- `λ::Number`:     penalty on sum(b.^2)

"""
function LassoEN(Y, X, γ, λ = 0)
    (T, K) = (size(X, 1), size(X, 2))

    Q = X'X / T
    c = X'Y / T                      #c'b = Y'X*b

    b = Variable(K)              #define variables to optimize over
    L1 = quadform(b, Q)            #b'Q*b
    L2 = dot(c, b)                 #c'b
    L3 = norm(b, 1)                #sum(|b|)
    L4 = sumsquares(b)            #sum(b^2)

    if λ > 0
        Sol = minimize(L1 - 2 * L2 + γ * L3 + λ * L4)      #u'u/T + γ*sum(|b|) + λ*sum(b^2), where u = Y-Xb
    else
        Sol = minimize(L1 - 2 * L2 + γ * L3)               #u'u/T + γ*sum(|b|) where u = Y-Xb
    end
    solve!(Sol, SCS.Optimizer; silent_solver = true)
    Sol.status == Convex.MOI.OPTIMAL ? b_i = vec(b.value) : b_i = NaN

    return b_i
end
```

    LassoEN

``` julia
Y = Array{Float64}(DT.metab_25)
X = Matrix{Float64}(DT[:,tmp.variable_name]);
```

``` julia
nγ = 30
nλ = 20
γM = range(0.0; stop = 1.5, length = nγ)
λM = range(0.0; stop = 3.5, length = nλ)
w = reshape([[x,y]  for x=γM, y=λM],nγ*nλ);
tmp = mapreduce(permutedims, vcat, w)
EN_opt = fill(NaN, nγ*nλ, 2)
EN_opt = hcat(tmp,EN_opt);
```

``` julia
Threads.@threads for i in 1:size(EN_opt)[1]
    sol = LassoEN(Y, X, EN_opt[i,1], EN_opt[i,2])
    RMSE = sqrt((1/size(X)[1])*(sum((Y - X * sol).^2)))
    MAE = (1/size(X)[1])*(sum(abs.(Y - X * sol)))
    EN_opt[i,3] = RMSE 
    EN_opt[i,4] = MAE 
end
```

``` julia
EN_opt[EN_opt[:,4] .== minimum(EN_opt[:,4]),:]  
```

    1×4 Matrix{Float64}:
     0.0517241  0.0  1.00564  0.754852

``` julia
γ = 0.052 
λ = 0
a = collect(1:size(X)[1])
betas = fill(NaN, 5000, 51)
Threads.@threads for i in 1:5000
    sp = sample(a, 835; replace=false, ordered=false)
    sol = LassoEN(Y[sp], X[sp,:], γ, λ)
    betas[i,:] = sol  
end
```

``` julia
tmp_beta = fill(NaN, 51, 3)
for i in 1:51
    tmp_beta[i,:] = quantile(betas[:,i],[0.025,0.5,0.975])
end
```

``` julia
LR = DataFrame(tmp_beta)
LR[:,"Names"] = var_name_
LR = LR[:,["Names","x1","x2","x3"]]
rename!(LR,[:Names,:q025,:median,:q975])
LR[:,"Sig"] = ifelse.(sign.(LR.q025) .== sign.(LR.q975),1,0);
```

``` julia
show(filter([:Sig] => x -> x == 1,LR))
```

    15×5 DataFrame
     Row │ Names        q025          median      q975         Sig   
         │ String       Float64       Float64     Float64      Int64 
    ─────┼───────────────────────────────────────────────────────────
       1 │ PM10(year)   -0.0520548    -0.0431788  -0.0342674       1
       2 │ PM2.5(year)   0.00037055    0.0232613   0.0455563       1
       3 │ Cu            0.90013       0.951731    0.998698        1
       4 │ Hg           -0.0578969    -0.0285746  -1.40006e-5      1
       5 │ HCB           0.104084      0.189385    0.271106        1
       6 │ DETP          0.024561      0.0365623   0.048388        1
       7 │ DMP           0.00233798    0.0122899   0.0225109       1
       8 │ DMDTP         0.0161392     0.0306119   0.0464814       1
       9 │ BPA           0.000618293   0.0295107   0.0568622       1
      10 │ BUPA          3.95319e-6    0.0194226   0.0427671       1
      11 │ PRPA         -0.0324774    -0.0195943  -0.00695834      1
      12 │ TRCS         -0.0462504    -0.0251367  -0.00383008      1
      13 │ MEHP         -0.120548     -0.0799429  -0.0366083       1
      14 │ MIBP          0.0381508     0.0809527   0.122412        1
      15 │ MNBP          1.03944e-7    0.0411545   0.0900464       1
