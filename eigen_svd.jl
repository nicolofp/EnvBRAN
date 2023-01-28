using HTTP, JSON, LinearAlgebra, DataFrames, Statistics, Plots

resp = HTTP.get("https://provanik.s3.us-east-2.amazonaws.com/data_serum.json")
str = String(resp.body)
jobj = JSON.Parser.parse(str)
serum_sample = vcat(DataFrame.(jobj["serum_sample"])...,cols = :union);
X_tmp = Matrix{Float64}(serum_sample[:,1:1198]);
X = X_tmp'

# Here we manually create the cov matrix (cov(X))
X = X .- mean(X,dims=1)
Y = (X'*X)./1197

eig_Y = eigen(Y);
eig_expl = sort(eig_Y.values/sum(eig_Y.values), rev = true)
p = bar(1:177,eig_expl)
ylabel!("% var explained")
xlabel!("Components")
title!("Eigen serum metabolites")
savefig(p,"eigen_svd.png")