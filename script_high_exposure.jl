using DataFrames, Statistics, LinearAlgebra, JLD2, DelimitedFiles
using Distributions, Parquet, StatsPlots, StatsBase, CSV, Combinatorics

DT = load_object("C:/Users/nicol/Documents/Github_projects/EnvBRAN/clead_dt.jld2");

rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ std(A, dims=dims)

genexpr = CSV.read("C:/Users/nicol/Documents/Metabolomics/data/tables_format/genexpr_num.csv",DataFrame);
genexpr_cov = DataFrame(read_parquet("C:/Users/nicol/Documents/Metabolomics/data/tables_format/genexpr_cov.parquet"));

genexpr_t = permutedims(genexpr,1);
rename!(genexpr_t,:rn => :ID);

DT = innerjoin(genexpr_cov,DT, on = :ID);
DT = innerjoin(DT, genexpr_t, on = :ID);
DT = DT[completecases(DT),:];

exposure_label = names(DT)[28:84];

M_exp = Matrix{Float32}(DT[:,exposure_label]);
M_exp = rescale(M_exp);

# Create the vector with all combinations
lab_inter = collect(combinations(exposure_label,2));
lab_inter = mapreduce(permutedims, vcat, lab_inter);
ind_inter = collect(combinations(1:57,2));
ind_inter = mapreduce(permutedims, vcat, ind_inter);

# Create the squared terms 
lab_square = hcat(exposure_label,exposure_label);
ind_square = hcat(1:57,1:57);

labs = vcat(lab_square,lab_inter);
inds = vcat(ind_square,ind_inter);

LBS = hcat(labs,inds)

# w3 = reshape([[x,y,z]  for x=1:57, y=1:57, z=1:57],57*57*57)
# tmp3 = mapreduce(permutedims, vcat, w3);
# E3 = fill(NaN, size(DT)[1],size(tmp3)[1])
# Threads.@threads for i in 1:size(tmp3)[1]
#     E3[:,i] = M_exp[:,tmp3[i,1]] .* M_exp[:,tmp3[i,2]] .* M_exp[:,tmp3[i,3]]
# end

E2 = fill(NaN, size(DT)[1],size(LBS)[1])
Threads.@threads for i in 1:size(LBS)[1]
    E2[:,i] = M_exp[:,LBS[i,3]] .* M_exp[:,LBS[i,4]]
end

EM = hcat(M_exp,E2);

LBS = hcat(LBS,LBS[:,1] .* " × " .* LBS[:,2])
RExp = hcat(fill(NaN,size(LBS)[1],1),LBS[:,5],fill(NaN,size(LBS)[1],6));

for j in 1:length(genexpr.rn)
    println("Analyze #",j,": ", genexpr.rn[j])
    Threads.@threads for i in 1:size(RExp)[1]
        X_tmp = Array{Float32, 2}(hcat(ones(size(DT)[1]),
            EM[:,i],
            Matrix{Float32}(DT[:,names(DT)[[3,5,6,7,8,9,10,11,12]]])))
        Y_tmp = convert.(Float32, DT[:,genexpr.rn[j]])

        β = X_tmp\Y_tmp
        σ² = sum((Y_tmp - X_tmp*β).^2)/(size(X_tmp,1)-size(X_tmp,2))
        Σ = σ²*inv(X_tmp'*X_tmp)
        std_coeff = sqrt.(diag(Σ))
        
        RExp[i,1] = genexpr.rn[j]
        RExp[i,3] = β[2]
        RExp[i,4] = std_coeff[2]
        RExp[i,5] = β[2]/std_coeff[2]
        RExp[i,6] = cdf(TDist(size(X_tmp,1)-size(X_tmp,2)), -abs(β[2]/std_coeff[2]))
        RExp[i,7] = β[2] - quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
        RExp[i,8] = β[2] + quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975)*std_coeff[2]
    end
	tmp1 = RExp[:,6] .< (0.05/size(RExp)[1])
    ewas_results = RExp[tmp1,:] 
	if(sum(tmp1) != 0) writedlm("C:/Users/nicol/Documents/Github_projects/EnvBRAN/gexpr_results2/" * genexpr.rn[j] * ".txt", 
								ewas_results)
	end
end
