Experiments with Julia
================
Nicoló Foppa Pedretti

Import data from **HELIX project** and analyze the mixture of the
chemicals. Let’s upload the packages that we’ll use in the code

``` julia
using DataFrames, Statistics, Turing, LinearAlgebra, Plots, CategoricalArrays, Distributions, Parquet, StatsPlots, StatsBase
using MLDataUtils: shuffleobs, splitobs, rescale!
```

Load the bayesian models in the external files. We are using *Turing.jl*
a Julia library for bayesian inference.

``` julia
include("bayes_lib.jl")
```

    bwqs_adv (generic function with 2 methods)

Load the data from “.parquet” file format

``` julia
path = "C:\\Users\\nicol\\Documents\\Metabolomics\\data\\tables_format\\"
covariates = DataFrame(read_parquet(string(path,"covariates.parquet")));
phenotype = DataFrame(read_parquet(string(path,"phenotype.parquet")));
```

``` julia
codebook = DataFrame(read_parquet(string(path,"codebook.parquet")));
```

Codebook with all the exposure considered:

``` julia
exposure = filter([:period, :var_type, :domain, :family] => (x,y,z,w) -> x == "Pregnancy" && y == "numeric" &&
                   z != "Covariates" && z!= "Phenotype" && w ∉ ["Built environment","Natural Spaces",
                                                                "Noise","Traffic","Tobacco Smoke"],codebook);

show(exposure[:,[:variable_name, :domain, :family, :subfamily, :period, 
                 :var_type, :labelsshort]],
     allcols = true)
```

    57×7 DataFrame
     Row │ variable_name           domain             family          subfamily     period     var_type  labelsshort 
         │ String?                 String?            String?         String?       String?    String?   String?     
    ─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────
       1 │ h_abs_ratio_preg_Log    Outdoor exposures  Air Pollution   PMAbsorbance  Pregnancy  numeric   PMabs
       2 │ h_no2_ratio_preg_Log    Outdoor exposures  Air Pollution   NO2           Pregnancy  numeric   NO2
       3 │ h_pm10_ratio_preg_None  Outdoor exposures  Air Pollution   PM10          Pregnancy  numeric   PM10
       4 │ h_pm25_ratio_preg_None  Outdoor exposures  Air Pollution   PM2.5         Pregnancy  numeric   PM2.5
       5 │ hs_as_m_Log2            Chemicals          Metals          As            Pregnancy  numeric   As
       6 │ hs_cd_m_Log2            Chemicals          Metals          Cd            Pregnancy  numeric   Cd
       7 │ hs_co_m_Log2            Chemicals          Metals          Co            Pregnancy  numeric   Co
       8 │ hs_cs_m_Log2            Chemicals          Metals          Cs            Pregnancy  numeric   Cs
       9 │ hs_cu_m_Log2            Chemicals          Metals          Cu            Pregnancy  numeric   Cu
      10 │ hs_hg_m_Log2            Chemicals          Metals          Hg            Pregnancy  numeric   Hg
      11 │ hs_mn_m_Log2            Chemicals          Metals          Mn            Pregnancy  numeric   Mn
      ⋮  │           ⋮                     ⋮                ⋮              ⋮            ⋮         ⋮           ⋮
      47 │ hs_mehp_madj_Log2       Chemicals          Phthalates      MEHP          Pregnancy  numeric   MEHP
      48 │ hs_meohp_madj_Log2      Chemicals          Phthalates      MEOHP         Pregnancy  numeric   MEOHP
      49 │ hs_mep_madj_Log2        Chemicals          Phthalates      MEP           Pregnancy  numeric   MEP
      50 │ hs_mibp_madj_Log2       Chemicals          Phthalates      MIBP          Pregnancy  numeric   MIBP
      51 │ hs_mnbp_madj_Log2       Chemicals          Phthalates      MNBP          Pregnancy  numeric   MNBP
      52 │ hs_ohminp_madj_Log2     Chemicals          Phthalates      OHMiNP        Pregnancy  numeric   OHMiNP
      53 │ hs_oxominp_madj_Log2    Chemicals          Phthalates      OXOMINP       Pregnancy  numeric   OXOMINP
      54 │ hs_sumDEHP_madj_Log2    Chemicals          Phthalates      DEHP          Pregnancy  numeric   SumDEHP
      55 │ h_bro_preg_Log          Outdoor exposures  Water DBPs      Water DBPs    Pregnancy  numeric   Brom
      56 │ h_clf_preg_Log          Outdoor exposures  Water DBPs      Water DBPs    Pregnancy  numeric   Chloroform
      57 │ h_thm_preg_Log          Outdoor exposures  Water DBPs      Water DBPs    Pregnancy  numeric   THMs
                                                                                                      35 rows omitted

let’s consider all the phenotype and covariates for the analysis

``` julia
cov_pheno = filter([:domain, :period] => (z,w) -> w == "Pregnancy" &&
                   (z == "Covariates" || z== "Phenotype"),codebook)
show(cov_pheno[:,[:variable_name, :domain, :family, :subfamily, :period, 
                 :var_type, :labelsshort]],
     allcols = true)
```

    11×7 DataFrame
     Row │ variable_name    domain      family      subfamily           period     var_type  labelsshort 
         │ String?          String?     String?     String?             String?    String?   String?     
    ─────┼───────────────────────────────────────────────────────────────────────────────────────────────
       1 │ h_mbmi_None      Covariates  Covariates  Maternal covariate  Pregnancy  numeric   mBMI
       2 │ hs_wgtgain_None  Covariates  Covariates  Maternal covariate  Pregnancy  numeric   Weightgain
       3 │ e3_gac_None      Covariates  Covariates  Child covariate     Pregnancy  numeric   GestAge
       4 │ e3_sex_None      Covariates  Covariates  Child covariate     Pregnancy  factor    Sex
       5 │ e3_yearbir_None  Covariates  Covariates  Child covariate     Pregnancy  factor    YearBirth
       6 │ h_age_None       Covariates  Covariates  Maternal covariate  Pregnancy  numeric   mAge
       7 │ h_cohort         Covariates  Covariates  Maternal covariate  Pregnancy  factor    Cohort
       8 │ h_edumc_None     Covariates  Covariates  Maternal covariate  Pregnancy  factor    mEducation
       9 │ h_native_None    Covariates  Covariates  Child covariate     Pregnancy  factor    Native
      10 │ h_parity_None    Covariates  Covariates  Maternal covariate  Pregnancy  factor    Parity
      11 │ e3_bw            Phenotype   Phenotype   Outcome at birth    Pregnancy  numeric   BW

Create a unique dataset which contains all the covariates and phenotype.
Now we have to use those information to compute the **birth weight
z-score** according to *sex*, *gestational age* and *cohort* (which are
not homogenous). We are using only complete observations for
non-premature born.

``` julia
dt_covar = covariates[:,push!(cov_pheno.variable_name[1:10],"ID")];
dt_pheno = phenotype[:,["ID","e3_bw"]]
dt_pheno.ID = string.(dt_pheno.ID);
```

``` julia
DT = innerjoin(dt_covar, dt_pheno, 
               on = :ID => :ID)
DT = dropmissing(DT)
DT = DT[DT.e3_gac_None .>= 37.0,:]
show(DT[:,["ID","h_cohort","e3_bw","h_mbmi_None","hs_wgtgain_None",
           "e3_gac_None","e3_sex_None","h_age_None"]], allcols = true)
```

    1004×8 DataFrame
      Row │ ID      h_cohort  e3_bw  h_mbmi_None  hs_wgtgain_None  e3_gac_None  e3_sex_None  h_age_None 
          │ String  String    Int32  Float64      Float64          Float64      String       Float64    
    ──────┼─────────────────────────────────────────────────────────────────────────────────────────────
        1 │ 1       4          4100      25.5102             17.0      41.0     male            28.0
        2 │ 2       4          4158      26.4915             18.0      41.0     male            22.8416
        3 │ 4       2          3270      21.048              21.0      39.2857  female          32.7255
        4 │ 5       3          3950      22.151              20.0      43.0     male            20.8652
        5 │ 6       1          2900      24.9036             30.0      38.8571  female          27.0
        6 │ 7       2          3350      27.5172             20.0      40.0     male            39.6003
        7 │ 9       2          3000      23.029              12.0      40.8571  female          30.3874
        8 │ 12      3          3660      23.7387             18.0      39.0     male            37.5989
        9 │ 13      2          3390      21.6441              5.0      40.4286  female          34.2368
       10 │ 15      4          4210      43.0277             12.0      41.4286  male            30.6749
       11 │ 17      4          3510      17.9089             10.0      39.0     male            29.0
      ⋮   │   ⋮        ⋮        ⋮         ⋮              ⋮              ⋮            ⋮           ⋮
      994 │ 1288    1          3320      21.1677             10.0      40.0     female          26.0
      995 │ 1290    5          2560      27.6361             14.0      40.1429  female          33.0
      996 │ 1291    2          2800      19.5069             20.0      37.4286  female          30.601
      997 │ 1292    3          3185      20.202              10.0      38.4286  male            30.1875
      998 │ 1293    6          2540      22.4914              8.0      38.0     female          30.0961
      999 │ 1294    4          3843      29.7056             27.0      40.0     male            28.6162
     1000 │ 1295    3          3690      26.8136             30.0      41.4286  male            32.2081
     1001 │ 1296    5          2810      25.3069             14.0      39.5714  female          24.6858
     1002 │ 1297    6          2900      33.564              17.0      40.4286  female          32.1506
     1003 │ 1299    4          4068      27.3797             20.0      41.2857  male            22.0313
     1004 │ 1300    5          4000      20.7967             20.0      41.5714  male            36.0
                                                                                        982 rows omitted

Select the category for the gestational age:

``` julia
countmap(cut(DT.e3_gac_None,[37,39,40,41,45]))
```

    Dict{CategoricalValue{String, UInt32}, Int64} with 4 entries:
      "[39, 40)" => 197
      "[37, 39)" => 238
      "[40, 41)" => 330
      "[41, 45)" => 239

``` julia
DT.e3_gac_Cat = cut(DT.e3_gac_None,[37,39,40,41,45]);
```

``` julia
#combine(groupby(DT,[:h_cohort,:e3_gac_Cat,:e3_sex_None]), nrow => :count)
zscore_table = combine(groupby(DT,[:h_cohort,:e3_gac_Cat,:e3_sex_None]), [:e3_bw] .=> (mean,std));
```

``` julia
DT = innerjoin(DT, zscore_table, 
               on = [:h_cohort,:e3_gac_Cat,:e3_sex_None]);
```

``` julia
DT.bw_zscore = (DT.e3_bw .- DT.e3_bw_mean)./DT.e3_bw_std;
```

Join the dataset with the phenotype and covariates with the dataset of
exposure

``` julia
DT = innerjoin(DT, covariates[:,push!(exposure.variable_name,"ID")], 
               on = :ID)
DT = dropmissing(DT)
```

Here a preview of the final dataset with some variable:

``` julia
show(DT[:,["ID","h_cohort","e3_sex_None","bw_zscore","h_pm10_ratio_preg_None","hs_hg_m_Log2"]],allcols = true)
```

    1004×6 DataFrame
      Row │ ID      h_cohort  e3_sex_None  bw_zscore  h_pm10_ratio_preg_None  hs_hg_m_Log2 
          │ String  String    String       Float64    Float64                 Float64      
    ──────┼────────────────────────────────────────────────────────────────────────────────
        1 │ 1       4         male          0.45213                  25.9485     -3.12029
        2 │ 2       4         male          0.681784                 25.8977     -1.02327
        3 │ 4       2         female        0.190264                 14.9914      2.21101
        4 │ 5       3         male          0.937519                 35.1973      3.23447
        5 │ 6       1         female       -0.181996                 19.767       0.933573
        6 │ 7       2         male         -0.64475                  23.1212      0.575312
        7 │ 9       2         female       -0.672751                 21.8908     -0.26708
        8 │ 12      3         male          0.797808                 31.6698      5.14136
        9 │ 13      2         female        0.45837                  14.7818     -1.27579
       10 │ 15      4         male          0.88768                  28.6259     -0.483985
       11 │ 17      4         male         -0.402741                 24.1683      1.02148
      ⋮   │   ⋮        ⋮           ⋮           ⋮                ⋮                  ⋮
      994 │ 1288    1         female       -0.188357                 21.4659     -1.33279
      995 │ 1290    5         female       -2.5859                   10.4021      2.17632
      996 │ 1291    2         female       -1.254                    23.1424      1.926
      997 │ 1292    3         male          0.168275                 40.2789     -1.27579
      998 │ 1293    6         female       -1.51792                  37.1661      1.5008
      999 │ 1294    4         male          0.513543                 23.5744     -1.16812
     1000 │ 1295    3         male          0.392015                 26.7961     -0.26708
     1001 │ 1296    5         female       -1.60647                  13.2033      1.72247
     1002 │ 1297    6         female       -1.36791                  35.828       1.68257
     1003 │ 1299    4         male          0.325425                 24.9508     -2.40354
     1004 │ 1300    5         male          0.285146                 18.258       0.925999
                                                                           982 rows omitted
