Experiments with Julia
================
Nicoló Foppa Pedretti

Import data from **HELIX project** and analyze the mixture of the
chemicals

``` julia
using DataFrames, Statistics, Turing, LinearAlgebra, Plots, CategoricalArrays, Distributions, Parquet, StatsPlots, StatsBase
using MLDataUtils: shuffleobs, splitobs, rescale!
```

``` julia
include("bayes_lib.jl")
```

    bwqs_adv (generic function with 2 methods)

``` julia
path = "C:\\Users\\nicol\\Documents\\Metabolomics\\data\\tables_format\\"
covariates = DataFrame(read_parquet(string(path,"covariates.parquet")));
phenotype = DataFrame(read_parquet(string(path,"phenotype.parquet")));
```

``` julia
codebook = DataFrame(read_parquet(string(path,"codebook.parquet")));
show(codebook[:,[:variable_name, :domain, :family, :subfamily, :period, 
                 :var_type, :labelsshort]],
     allcols = true)
```

    241×7 DataFrame
     Row │ variable_name           domain             family         subfamily                  period     var_type  labelsshort 
         │ String?                 String?            String?        String?                    String?    String?   String?     
    ─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       1 │ h_abs_ratio_preg_Log    Outdoor exposures  Air Pollution  PMAbsorbance               Pregnancy  numeric   PMabs
       2 │ h_no2_ratio_preg_Log    Outdoor exposures  Air Pollution  NO2                        Pregnancy  numeric   NO2
       3 │ h_pm10_ratio_preg_None  Outdoor exposures  Air Pollution  PM10                       Pregnancy  numeric   PM10
       4 │ h_pm25_ratio_preg_None  Outdoor exposures  Air Pollution  PM2.5                      Pregnancy  numeric   PM2.5
       5 │ hs_no2_dy_hs_h_Log      Outdoor exposures  Air Pollution  NO2                        Postnatal  numeric   NO2(day)
       6 │ hs_no2_wk_hs_h_Log      Outdoor exposures  Air Pollution  NO2                        Postnatal  numeric   NO2(week)
       7 │ hs_no2_yr_hs_h_Log      Outdoor exposures  Air Pollution  NO2                        Postnatal  numeric   NO2(year)
       8 │ hs_pm10_dy_hs_h_None    Outdoor exposures  Air Pollution  PM10                       Postnatal  numeric   PM10(day)
       9 │ hs_pm10_wk_hs_h_None    Outdoor exposures  Air Pollution  PM10                       Postnatal  numeric   PM10(week)
      10 │ hs_pm10_yr_hs_h_None    Outdoor exposures  Air Pollution  PM10                       Postnatal  numeric   PM10(year)
      11 │ hs_pm25_dy_hs_h_None    Outdoor exposures  Air Pollution  PM2.5                      Postnatal  numeric   PM2.5(day)
      ⋮  │           ⋮                     ⋮                ⋮                    ⋮                  ⋮         ⋮           ⋮
     232 │ h_edumc_None            Covariates         Covariates     Maternal covariate         Pregnancy  factor    mEducation
     233 │ h_native_None           Covariates         Covariates     Child covariate            Pregnancy  factor    Native
     234 │ h_parity_None           Covariates         Covariates     Maternal covariate         Pregnancy  factor    Parity
     235 │ hs_child_age_None       Covariates         Covariates     Child covariate            Postnatal  numeric   cAge
     236 │ e3_bw                   Phenotype          Phenotype      Outcome at birth           Pregnancy  numeric   BW
     237 │ hs_asthma               Phenotype          Phenotype      Outcome at 6-11 years old  Postnatal  factor    Asthma
     238 │ hs_zbmi_who             Phenotype          Phenotype      Outcome at 6-11 years old  Postnatal  numeric   zBMI
     239 │ hs_correct_raven        Phenotype          Phenotype      Outcome at 6-11 years old  Postnatal  numeric   IQ
     240 │ hs_Gen_Tot              Phenotype          Phenotype      Outcome at 6-11 years old  Postnatal  numeric   Behavior
     241 │ hs_bmi_c_cat            Phenotype          Phenotype      Outcome at 6-11 years old  Postnatal  factor    BMI_cat
                                                                                                                 220 rows omitted

``` julia
cod1 = filter([:period, :var_type, :domain] => (x,y,z) -> x == "Pregnancy" && y == "numeric" &&
       z != "Covariates" && z!= "Phenotype",codebook);
show(cod1[:,[:variable_name, :domain, :family, :subfamily, :period, 
                 :var_type, :labelsshort]],
     allcols = true)
```

    72×7 DataFrame
     Row │ variable_name                 domain             family             subfamily         period     var_type  labelsshort 
         │ String?                       String?            String?            String?           String?    String?   String?     
    ─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       1 │ h_abs_ratio_preg_Log          Outdoor exposures  Air Pollution      PMAbsorbance      Pregnancy  numeric   PMabs
       2 │ h_no2_ratio_preg_Log          Outdoor exposures  Air Pollution      NO2               Pregnancy  numeric   NO2
       3 │ h_pm10_ratio_preg_None        Outdoor exposures  Air Pollution      PM10              Pregnancy  numeric   PM10
       4 │ h_pm25_ratio_preg_None        Outdoor exposures  Air Pollution      PM2.5             Pregnancy  numeric   PM2.5
       5 │ h_accesslines300_preg_dic0    Outdoor exposures  Built environment  Access            Pregnancy  numeric   BPTLine
       6 │ h_accesspoints300_preg_Log    Outdoor exposures  Built environment  Access            Pregnancy  numeric   BPTStop
       7 │ h_builtdens300_preg_Sqrt      Outdoor exposures  Built environment  Building density  Pregnancy  numeric   BuildDens
       8 │ h_connind300_preg_Sqrt        Outdoor exposures  Built environment  Connectivity      Pregnancy  numeric   Connec
       9 │ h_fdensity300_preg_Log        Outdoor exposures  Built environment  Facility          Pregnancy  numeric   FacDens
      10 │ h_frichness300_preg_None      Outdoor exposures  Built environment  Facility          Pregnancy  numeric   FacRich
      11 │ h_landuseshan300_preg_None    Outdoor exposures  Built environment  Land use          Pregnancy  numeric   Land use
      ⋮  │              ⋮                        ⋮                  ⋮                 ⋮              ⋮         ⋮           ⋮
      63 │ hs_ohminp_madj_Log2           Chemicals          Phthalates         OHMiNP            Pregnancy  numeric   OHMiNP
      64 │ hs_oxominp_madj_Log2          Chemicals          Phthalates         OXOMINP           Pregnancy  numeric   OXOMINP
      65 │ hs_sumDEHP_madj_Log2          Chemicals          Phthalates         DEHP              Pregnancy  numeric   SumDEHP
      66 │ e3_asmokcigd_p_None           Chemicals          Tobacco Smoke      Tobacco Smoke     Pregnancy  numeric   Cigarette
      67 │ h_distinvnear1_preg_Log       Outdoor exposures  Traffic            Traffic           Pregnancy  numeric   DistRoad
      68 │ h_trafload_preg_pow1over3     Outdoor exposures  Traffic            Traffic           Pregnancy  numeric   Traffic
      69 │ h_trafnear_preg_pow1over3     Outdoor exposures  Traffic            Traffic           Pregnancy  numeric   TrafDens
      70 │ h_bro_preg_Log                Outdoor exposures  Water DBPs         Water DBPs        Pregnancy  numeric   Brom
      71 │ h_clf_preg_Log                Outdoor exposures  Water DBPs         Water DBPs        Pregnancy  numeric   Chloroform
      72 │ h_thm_preg_Log                Outdoor exposures  Water DBPs         Water DBPs        Pregnancy  numeric   THMs
                                                                                                                   51 rows omitted

``` julia
phthalates = cod1[cod1.family .== "Metals",:variable_name];
```

``` julia
z = covariates[completecases(covariates),:]
covart = Matrix{Float64}(z[:, phthalates]) 
target = Vector{Float64}(z[:,:e3_gac_None]);
```

``` julia
for i in 1:size(phthalates)[1]
    covart[:,i] = ecdf(covart[:,i])(covart[:,i])*10
end
```

``` julia
model = bwqs(covart, target)
model_adv = bwqs_adv(covart, target);
```

``` julia
chain_adv = sample(model_adv, NUTS(0.65), 3000, thinning = 3)
chain = sample(model, NUTS(0.65), 3000, thinning = 3);
```

    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\51xgc\src\hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\51xgc\src\hamiltonian.jl:47
    ┌ Info: Found initial step size
    │   ϵ = 0.00625
    └ @ Turing.Inference C:\Users\nicol\.julia\packages\Turing\Oczpc\src\inference\hmc.jl:188
    Sampling: 100%|█████████████████████████████████████████| Time: 1:04:50
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, true, true, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\51xgc\src\hamiltonian.jl:47
    ┌ Warning: The current proposal will be rejected due to numerical error(s).
    │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, true, true, false)
    └ @ AdvancedHMC C:\Users\nicol\.julia\packages\AdvancedHMC\51xgc\src\hamiltonian.jl:47
    ┌ Info: Found initial step size
    │   ϵ = 0.00625
    └ @ Turing.Inference C:\Users\nicol\.julia\packages\Turing\Oczpc\src\inference\hmc.jl:188
    Sampling: 100%|█████████████████████████████████████████| Time: 0:04:52

``` julia
hcat(quantile(chain_adv.value[:,3,1][:],[0.025,0.5,0.975]),
     quantile(chain.value[:,3,1][:],[0.025,0.5,0.975]))
```

    3×2 Matrix{Float64}:
     -0.191766   -0.220385
     -0.123412   -0.140153
     -0.0681323  -0.0538109

``` julia
DataFrame(Metals = phthalates,
     w = mean(Matrix(chain.value[:,4:12,1]), dims = 1)'[:],
     w_adv = mean(Matrix(chain_adv.value[:,13:21,1]), dims = 1)'[:])  
```
