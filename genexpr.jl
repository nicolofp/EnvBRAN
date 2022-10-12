using DataFrames, Plots, LinearAlgebra, Statistics, Distributions, CSV, Parquet, TSne

codebook = DataFrame(read_parquet("C:/Users/nicol/Documents/Metabolomics/data/tables_format/codebook.parquet"));
phenotype = DataFrame(read_parquet("C:/Users/nicol/Documents/Metabolomics/data/tables_format/phenotype.parquet"));
exposure = DataFrame(read_parquet("C:/Users/nicol/Documents/Metabolomics/data/tables_format/covariates.parquet"));

genexpr = CSV.read("C:/Users/nicol/Documents/Metabolomics/data/tables_format/genexpr_num.csv",DataFrame);

genexpr_cov = DataFrame(read_parquet("C:/Users/nicol/Documents/Metabolomics/data/tables_format/genexpr_cov.parquet"));

# first(genexpr, 10)

# tmp1 = Array{Float64}(genexpr[1,2:size(genexpr)[2]])
# tmp2 = Array{Float64}(genexpr[2,2:size(genexpr)[2]])
# tmp3 = Array{Float64}(genexpr[3,2:size(genexpr)[2]])
# p1 = histogram(tmp1, bins = 50, title = "TC01000001.hg.1")
# p2 = histogram(tmp2, bins = 50, title = "TC01000002.hg.1")
# p3 = histogram(tmp3, bins = 50, title = "TC01000003.hg.1")
# plot(p1, p2, p3, layout = (1, 3), legend = false, size = (750,250))

N,D = size(genexpr)
MV = zeros(N,5)
for i in 1:N
    tmp1 = Array{Float64}(genexpr[i,2:D])
    MV[i,1] = mean(tmp1)
    MV[i,2] = std(tmp1)
    MV[i,3] = quantile(tmp1,0.25)
    MV[i,4] = quantile(tmp1,0.5)
    MV[i,5] = quantile(tmp1,0.75)
end

X = Matrix{Float64}(Matrix{Float64}(genexpr[:,2:D])');

svd_gene = svd(X);

y_tsne = tsne(X, 2, N, 1000, 15, pca_init = true, progress = false); 

(svd_gene.S).^2/sum((svd_gene.S).^2)
PCA_gene = svd_gene.U * (diagm(svd_gene.S)/(D-1));

# z = ifelse.(genexpr_cov.e3_sex .== "male","orange","lime")
# p1 = scatter(PCA_gene[:,1],PCA_gene[:,2],size=(400,300), legend = false, color = z)
# p2 = scatter(y_tsne[:,1],y_tsne[:,2],size=(400,300), legend = false, color = z)
# plot(p1, p2, layout = (1, 2), legend = false, size = (500,250))

outcome = names(phenotype)[4:5]
gene_names = string.(genexpr.rn)
w = reshape([[x,y]  for x=outcome, y=gene_names],length(outcome)*length(gene_names));
tmp = mapreduce(permutedims, vcat, w);
tmp = tmp[[1:2:(2*N);],:]; # we select the even index only (BMI z-score)
genexpr_t = permutedims(genexpr,1);
rename!(genexpr_t,:rn => :ID);

ewas_results = Array{Float64}(undef,N, 6);
ewas_results = hcat(tmp,ewas_results);

phenotype.ID = string.(phenotype.ID);
DT = innerjoin(genexpr_cov,phenotype[:,[:ID,:hs_zbmi_who]], on = :ID);

DT = innerjoin(DT, genexpr_t, on = :ID);

DT = DT[completecases(DT),:];

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

ewas_results = DataFrame(ewas_results,:auto)
rename!(ewas_results, ["outcome","variable","beta","std_error","t_value","p_value","CI0025","CI0975"]);

ewas_results.p_value_bonf = (ewas_results.p_value .< 0.05/N);

# ewas_results
# ewas2 = ewas_results[abs.(ewas_results.beta) .< 10.0,:];
# z = ifelse.(ewas2.p_value_bonf .== 1,"aqua","coral2")
# scatter(ewas2[:,3], - log10.(ewas2[:,6]),
#         color = z,size=(400,300), label = "")
