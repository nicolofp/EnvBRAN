using DataFrames, Statistics, LinearAlgebra, Plots, CategoricalArrays, Distances, SyntheticDatasets
using JLD, Distributions, JSON, HTTP, StatsPlots, StatsBase, Parquet, CSV, MultivariateStats

codebook = CSV.read(download("https://envbran.s3.us-east-2.amazonaws.com/codebook.tsv"), DataFrame);
phenotype = CSV.read(download("https://envbran.s3.us-east-2.amazonaws.com/phenotype.tsv"), DataFrame);
exposure = CSV.read(download("https://envbran.s3.us-east-2.amazonaws.com/covariates.tsv"), DataFrame);

#chemicals = filter([:period, :var_type, :domain, :period_postnatal, :family, :subfamily, 
#        :variable_name] => (x,y,z,w,p,r,k) -> x == "Postnatal" && 
#        y == "numeric" && z != "Covariates" && z!= "Phenotype" && (w == "NA" || contains.(w, "Year")) && 
#       !(contains.(p,"Traffic") || contains.(p,"Social and economic capital") || contains.(p,"Built environment") || 
#        contains.(p,"Indoor air") || contains.(p,"Lifestyle") || contains.(p,"Natural Spaces") || contains.(k,"sum")),
#        codebook);

exposure_list = filter([:period, :var_type, :domain, :period_postnatal, :family, :subfamily, 
        :variable_name] => (x,y,z,w,p,r,k) -> x == "Pregnancy" && 
        y == "numeric" && z != "Covariates" && z!= "Phenotype" && !(contains.(k,"sum")),
        codebook)

exposure_list.family
id = 1:15
map_dict = Dict((unique(exposure_list.family)[i] => id[i] for i=1:15));

color_chemicals = [map_dict[exposure_list.family[i]] for i in 1:size(exposure_list.family)[1]];

environmental = exposure[:,exposure_list.variable_name];
#describe(environmental)

kernel = SyntheticDatasets.make_halfkernel( n_samples = 1000, 
                                            minx = -20,
                                            r1 = 1, 
                                            r2 = 5,
                                            noise = 1.0, 
                                            ratio = 0.6);

kernel = SyntheticDatasets.make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2);

z = ifelse.(kernel.label .== 1,"aqua","coral2")
scatter(kernel[:,1], kernel[:,2],
        color = z, size=(400,300), label = "")

# g = 100
# f_kernel(x,y) = exp( - g * sum((x - y).^2))

# a = 0.5
# b = 3
# d = 4
# p_kernel(x,y) = (a*x'*y + b).^d

# N = 1000
# K1 = zeros(N,N)

# X = Matrix(Matrix{Float64}(kernel[:,[1, 2]])');
# for j = 1:N
#  for i = 1:N
#    K1[i,j] = p_kernel(X[:,i],X[:,j])
#  end
# end

g = 0.01
#X = Matrix(Matrix{Float64}(kernel[:,[1, 2]]));
#X = X .- mean(X,dims=1) ./ std(X,dims=1)
X = Matrix(Matrix{Float64}(environmental)');
dX = pairwise(sqeuclidean, X, dims=1)
K1 = exp.(-g * dX)

N = size(X)[1]
ncomp = 2
one_n = ones(N,N)/N
# recenter K
Kp = Symmetric(K1 - one_n * K1 - K1 * one_n + one_n * K1 * one_n)

# eigendecomposition with the ncomp lagest eigenvalues
eigvals_n, eigvecs_n = eigen(Kp);

eigvecs_n = reverse(eigvecs_n,dims=2)
eigvals_n = reverse(eigvals_n)

eigvals_n[eigvals_n .< 0] .= 0

lambda = eigvals_n/N

amplitudes = Diagonal(sqrt.(eigvals_n)) * eigvecs_n';

z = ifelse.(exposure.h_cohort .== "1","aqua", 
        ifelse.(exposure.h_cohort .== "2","coral2",
            ifelse.(exposure.h_cohort .== "3","cyan4",
                ifelse.(exposure.h_cohort .== "4","chartreuse3",
                    ifelse.(exposure.h_cohort .== "5","goldenrod1",
                        ifelse.(exposure.h_cohort .== "6","magenta1","slateblue1"))))))

scatter(amplitudes[1,:],amplitudes[2,:],
        color = color_chemicals,
        size=(400,300), label = "")

scatter(amplitudes[1,:], zeros(1000),
        #amplitudes[1,:],
        color = z,
        size=(400,300), label = "")

M = fit(KernelPCA, X'; kernel = (x,y) -> exp(-15*norm(x-y)^2), maxoutdim=2, inverse=true)
Yte = MultivariateStats.transform(M, X')

scatter(Yte[1,:],Yte[2,:],color = z,
        size=(400,300), label = "")

using UMAP
using TSne

E = Matrix(Matrix{Float64}(environmental));

embedding = umap(E);

rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())
EI = rescale(E)

Y = tsne(EI', 2, 50, 10000, 5.0);

p = scatter(Y[:,1],Y[:,2],
            color = color_chemicals, 
            xlabel = "Component 1", ylabel = "Component 2",
            size=(400,300), label = "", title = "T-SNE")
q = scatter(embedding[1,:],embedding[2,:],
            color = color_chemicals,
            xlabel = "Component 1", ylabel = "Component 2",
            size=(400,300), label = "", title = "UMAP")
plot(p, q, layout = (1, 2), legend = false, size = (850, 300))
