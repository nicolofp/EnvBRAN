using Turing, Distributions, Optim, Zygote, ReverseDiff, Memoization, AdvancedVI
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# Unsupervised pPCA (ARD correction)
@model function pPCA_ARD(x,k, ::Type{TV}=Array{Float64}) where {TV}
    # Dimensionality of the problem.
    N, D = size(x)
    K = Integer(k)

    # latent variable z
    z ~ filldist(Normal(), K, N)

    # weights/loadings w with Automatic Relevance Determination part
    # alpha ~ filldist(Gamma(1.0, 1.0), D)
    # w ~ filldist(MvNormal(zeros(D), 1.0 ./ sqrt.(alpha)), K)
    
    alpha ~ filldist(Gamma(1.0, 1.0), K)
    w ~ filldist(MvNormal(zeros(K), 1.0 ./ alpha), D)

    mu = (w' * z)'

    tau ~ Gamma(1.0, 1.0)
    for n in 1:N
        x[n, :] ~ MvNormal(mu[n, :], 1.0 / tau)
    end
end;

# Supervised pPCA (ARD correction)
@model function spPCA_ARD(x,y,m, ::Type{TV}=Array{Float64}) where {TV}
    # Dimensionality of the problem.
    N, D = size(x)
    M = Integer(m)

    # latent variable z
    z ~ filldist(Normal(), M, N)

    # weights/loadings W
    alpha_x ~ filldist(Gamma(1.0, 1.0), M)
    alpha_y ~ Gamma(1.0, 1.0)
    w_x ~ filldist(MvNormal(zeros(M), sqrt.(alpha_x)),D)
    w_y ~ filldist(Normal(0.0, sqrt(alpha_y)), M)
    
    tau ~ Gamma(1.0, 1.0)
    # mean offset
    # m ~ MvNormal(ones(D))
    # mu_x = (w_x * z)'
    # mu_y = (w_y' * z)'
    for n in 1:N
        x[n, :] ~ MvNormal(w_x' * z[:,n], tau)
        y[n] ~ Normal(w_y' * z[:,n], tau)
    end
end;

# Supervised pPCA
@model function spPCA(x,y,k, ::Type{TV}=Array{Float64}) where {TV}
    # Dimensionality of the problem.
    N, D = size(x)
    K = Integer(k)

    # latent variable z
    z ~ filldist(Normal(), K, N)

    # weights/loadings W
    w_x ~ filldist(Normal(), D, K)
    w_y ~ filldist(Normal(), K)

    # mean offset
    # m ~ MvNormal(ones(D))
    mu_x = (w_x * z)'
    mu_y = (w_y' * z)'
    for n in 1:N
        x[n, :] ~ MvNormal(mu_x[n, :], ones(D))
        y[n] ~ Normal(mu_y[n], 1.0)
    end
end;

# Bayesian BWQS regression.
@model function bwqs(cx, mx, y)
    # Set variance prior.
    σ₂ ~ Gamma(2.0, 2.0)

    # Set intercept prior.
    intercept ~ Normal(0, 20)
    beta ~ Normal(0, 20)
    ncovariates = size(cx, 2)
    delta ~ filldist(Normal(0, 20), ncovariates)

    # Set the priors on our coefficients.
    nfeatures = size(mx, 2)
    w ~ Dirichlet(nfeatures, 1) 

    # Calculate all the mu terms.
    mu = intercept .+ beta * (mx * w) .+ cx * delta
    return y ~ MvNormal(mu, sqrt(σ₂))
end

# Bayesian BWQS advanced regression.
@model function bwqs_adv(cx, mx, y)
    
    # Set variance prior.
    σ₂ ~ Gamma(2.0, 2.0)

    # Set intercept prior.
    intercept ~ Normal(0, 20)
    beta ~ Normal(0, 20)
    ncovariates = size(cx, 2)
    delta ~ filldist(Normal(0, 20), ncovariates)

    # Set the priors on our coefficients.
    nfeatures = size(mx, 2)
    alpha ~ filldist(Gamma(1.0, 1.0), nfeatures)
    w ~ Dirichlet(alpha) 

    # Calculate all the mu terms.
    mu = intercept .+ beta * (mx * w) .+ cx * delta
    return y ~ MvNormal(mu, sqrt(σ₂))
end
