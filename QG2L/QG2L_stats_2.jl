
function compute_local_GPD_params_zonsym_mean_residual_life(hist_filenames::Vector{String}, obs_fun_xshiftable::Function, lead_time::Int64, follow_time::Int64, Nxshifts::Int64, xstride::Int64, resultdir::String, obs_label)
    # should return a scalar 
    @show Nxshifts, xstride
    obs_fun_allshifts(f::JLD2.JLDFile) = reduce(hcat, (obs_fun_xshiftable(f, xshift) for xshift=range(0,Nxshifts-1,step=1).*xstride))
    obs_val_allshifts = reduce(vcat, compute_observable_ensemble(hist_filenames, obs_fun_allshifts))
    println("computed all shifts")
    @show size(obs_val_allshifts)
    # ---------------- Visualize mean residual life plot ------------
    # uncdertainty quantification by looking at different longitudes separatel
    Nthresh = 15
    m1 = maximum(obs_val_allshifts)
    m2 = maximum(filter(x->x<m1, obs_val_allshifts))
    threshes = collect(range(SB.mean(obs_val_allshifts), m2; length=Nthresh))
    edges = vcat(threshes, [2*threshes[Nthresh]-threshes[Nthresh-1]])
    mrls,mrlslopes,ccdfs = (zeros(Float64, (Nthresh, Nxshifts)) for _=1:3)
    best_threshes,meansquare_nonlinearities = (zeros(Float64, Nxshifts) for _=1:2)
    for i_shift = 1:Nxshifts
        mrl_diags = compute_mean_residual_life(obs_val_allshifts[:,i_shift], threshes; lead_time=lead_time, follow_time=follow_time)
        mrlslopes[:,i_shift] .= mrl_diags[1]
        mrls[:,i_shift] .= mrl_diags[2]
        best_threshes[i_shift] = mrl_diags[3]
        meansquare_nonlinearities[i_shift] = mrl_diags[4]
        hg = SB.fit(SB.Histogram, obs_val_allshifts[:,i_shift], edges)
        ccdfs[:,i_shift] .= reverse(cumsum(reverse(hg.weights)))
        ccdfs[:,i_shift] ./= length(obs_val_allshifts[:,i_shift])
    end
    fig = Figure()
    lout = fig[1:3,1] = GridLayout()
    ax1 = Axis(lout[1,1]; xlabel=L"$%$(obs_label)^\theta$", title=L"$\mathrm{MRL}(%$(obs_label)^\theta)=\mathbb{E}[%$(obs_label)^*-%$(obs_label)^\theta|%$(obs_label)^*>%$(obs_label)^\theta]$", xgridvisible=false, ygridvisible=false, xlabelvisible=false, xticklabelsvisible=false)
    ax2 = Axis(lout[2,1]; xlabel=L"$%$(obs_label)^\theta$", title=L"$d\mathrm{(MRL)}/d%$(obs_label)^\theta$", xgridvisible=false, ygridvisible=false, xlabelvisible=false, xticklabelsvisible=false)
    ax3 = Axis(lout[3,1]; xlabel=L"$%$(obs_label)$", title=L"CCDF$(%$(obs_label))$", yscale=log10, xgridvisible=false, ygridvisible=false)
    for i_shift = 1:Nxshifts
        kwargs = Dict(:color=>i_shift, :colorrange=>(1,Nxshifts), :colormap=>:cyclic_protanopic_deuteranopic_wywb_55_96_c33_n256)
        scatterlines!(ax1, threshes, mrls[:,i_shift]; kwargs...)
        scatterlines!(ax2, threshes, mrlslopes[:,i_shift]; kwargs...)
        scatterlines!(ax3, threshes, ccdfs[:,i_shift]; kwargs...)
        vlines!(ax1,best_threshes[i_shift]; kwargs...)
    end
    scatterlines!(ax2, threshes, vec(SB.mean(mrlslopes; dims=2)); color=:black, linestyle=:solid)
    for sgn = [-1,1]
        scatterlines!(ax2, threshes, vec(SB.mean(mrlslopes; dims=2) .+ sgn.*SB.std(mrlslopes; dims=2)); color=:black, linestyle=:dash)
    end
    thresh_hi_med = SB.median(best_threshes) #[argmin(meansquare_nonlinearities)]
    std_slopes = replace(vec(SB.std(mrlslopes; dims=2)), NaN=>Inf)
    thresh_hi_minvar = threshes[argmin(std_slopes)]
    vlines!(ax1, thresh_hi_med; color=:black, linestyle=:dash)
    vlines!(ax2, thresh_hi_minvar; color=:black, linestyle=:dash)
    # ------------- GIANT DECISION: how to choose thresh_hi? -------------
    thresh_hi = (thresh_hi_med + thresh_hi_minvar)/2
    #thresh_hi = maximum(best_threshes) #(thresh_hi_med + thresh_hi_minvar)/2
    # --------------------------------------------------------------------
    vlines!(ax3, thresh_hi; color=:black, linestyle=:dash)
    @show thresh_hi_med,thresh_hi_minvar,thresh_hi

    # Fit the GPD distribution
    scales,shapes = (zeros(Float64, Nxshifts) for _=1:2)
    logccdfs_gpd = zeros(Float64, (Nthresh,Nxshifts))
    ccdf_at_thresh = SB.mean(obs_val_allshifts.> thresh_hi)
    @show ccdf_at_thresh
    for i_shift = 1:Nxshifts
        peak_vals,peak_tidx,upcross_tidx,downcross_tidx = peaks_over_threshold(obs_val_allshifts[:,i_shift], thresh_hi)
        scales[i_shift],shapes[i_shift] = compute_GPD_params(peak_vals, thresh_hi)
        D = Dists.GeneralizedPareto(thresh_hi, scales[i_shift], shapes[i_shift])
        logccdfs_gpd[:,i_shift] .= Dists.logccdf.(D, threshes) .+ log(ccdf_at_thresh)
    end
    logccdf_gpd_mean = vec(SB.mean(logccdfs_gpd; dims=2))
    logccdf_gpd_lo = vec(mapslices(a->SB.quantile(a,0.25), logccdfs_gpd; dims=2))
    logccdf_gpd_hi = vec(mapslices(a->SB.quantile(a,0.75), logccdfs_gpd; dims=2))
    # Now compute parameters from the whole concatenated sequence
    hg = SB.fit(SB.Histogram, obs_val_allshifts[:], edges)
    ccdf = zeros(Float64, Nthresh)
    ccdf .= reverse(cumsum(reverse(hg.weights)))
    ccdf ./= length(obs_val_allshifts)
    peak_vals,peak_tidx,upcross_tidx,downcross_tidx = peaks_over_threshold(obs_val_allshifts[:], thresh_hi)
    scale,shape = compute_GPD_params(peak_vals, thresh_hi)
    D = Dists.GeneralizedPareto(thresh_hi, scale, shape)
    logccdf_gpd = Dists.logccdf(D, threshes) .+ log(ccdf_at_thresh)

    mustr = @sprintf("%.2f", thresh_hi)
    sigstr = @sprintf("%.2f", scale)
    xistr = @sprintf("%.2f", shape)
    scatterlines!(ax3, threshes, ccdf; color=:black, linewidth=3, linestyle=:solid, label=L"All lons$$")
    scatterlines!(ax3, threshes, exp.(logccdf_gpd); color=:red, linestyle=:solid, label=L"GPD($%$(obs_label)^\theta=%$(mustr),\sigma=%$(sigstr),\xi=%$(xistr)$")
    scatterlines!(ax3, threshes, exp.(logccdf_gpd_lo); color=:red, linestyle=:dash)
    scatterlines!(ax3, threshes, exp.(logccdf_gpd_hi); color=:red, linestyle=:dash)
    axislegend(ax3; position=:lb, framevisible=false)
    ylims!(ax3, minimum(filter(x->x>0, ccdfs)), 1.0)
    linkxaxes!(ax1, ax2, ax3)
    save(joinpath(resultdir,"pot_mean_residual_life_plot.png"), fig)
    # TODO select threshold based on the plot
    # find the threshold 

    # ---------------------------------------------------------------
    return (thresh_hi,scale,shape,peak_vals,peak_tidx,obs_val_allshifts[:])
end



function compute_local_GPD_params(hist_filenames::Vector{String}, obs_fun::Function, thresh_hi_map::Array{Float64,3}, prebuffer::Int64, postbuffer::Int64, initbuffer::Int64)
    @show size(thresh_hi_map),size(thresh_lo_map)
    (Nx,Ny,Nz) = size(thresh_hi_map)
    peak_vals = reshape([Vector{Float64}([]) for _=1:(Nx*Ny*2)], (Nx,Ny,2))
    for filename = hist_filenames
        obs_val = histfile2obs(filename, obs_fun)
        @show size(obs_val)
        #obs_val_peaks = get_peaks_func_3d(obs_val) #get_peaks_func_2d)
        for ix=1:Nx
            for iy=1:Ny
                for iz=1:Nz
                    #append!(peak_vals[ix,iy,iz], obs_val_peaks[ix,iy,iz])
                    new_peaks = peaks_over_threshold(vec(obs_val[ix,iy,iz,:]), thresh_hi_map[ix,iy,iz], prebuffer, postbuffer, initbuffer)[1]
                    append!(peak_vals[ix,iy,iz], new_peaks)
                end
            end
        end
    end

    @show size(peak_vals),size(thresh_hi_map),size(thresh_lo_map)
    scale,shape = (zeros(Float64, (Nx,Ny,Nz)) for _=1:2)
    for ix=1:Nx
        for iy=1:Ny
            for iz=1:Nz
                scale[ix,iy,iz],shape[ix,iy,iz] = compute_GPD_params(peak_vals[ix,iy,iz], thresh_hi_map[ix,iy,iz])
            end
        end
    end
    @show size(scale),size(shape)
    return scale,shape
end

function compute_layerwise_histograms(hist_filenames::Vector{String},obs_fun_local::Function, sdm::SpaceDomain, bin_width::Float64)
    obs_fun_layer1(args...) = vec(obs_fun_local(args...)[:,:,1,:])
    obs_fun_layer2(args...) = vec(obs_fun_local(args...)[:,:,2,:])
    bins1,counts1 = compute_observable_histogram(hist_filenames, obs_fun_layer1, bin_width)
    bins2,counts2 = compute_observable_histogram(hist_filenames, obs_fun_layer2, bin_width)
    return bins1,counts1,bins2,counts2
end

function compute_GPD_params_from_histogram(thresh::Float64, bin_edges::Vector{Float64}, bin_weights::Vector{Float64}; method::String="MM")
    # taken from Hosking & Wallis 
    ws = bin_weights.*(bin_edges[2:end] .>= thresh)
    if (minimum(ws) < 0) || (maximum(ws) <= 0)
        return (NaN,NaN)
    end
    ws ./= sum(ws)
    scale = 0
    shape = 0
    if method == "MM"
        xs = (bin_edges[1:end-1] .+ bin_edges[2:end])./2 .- thresh
        m1 = sum(ws .* xs) 
        m2 = sum(ws .* xs.^2) 
        s2 = m2 - m1^2 # variance estimate 
        scale = 0.5 * m1 * (m1^2/s2 + 1)
        shape = -0.5 * (m1^2/s2 - 1)
    elseif method == "PWM"
        xs = bin_edges[2:end] .- thresh
        Fs = cumsum(ws) 
        a0 = sum(ws .* xs)
        a1 = sum(ws .* xs .* (1 .- Fs))
        scale = 2*a0*a1/(a0 - 2*a1)
        shape = 2 - a0/(a0 - 2*a1) 
    else
        error()
    end
    if !((scale > 0) && isfinite(scale) && isfinite(shape))
        return (NaN,NaN)
        #@show a0, a1, scale, shape
        #@show thresh, bin_edges[1], minimum(diff(bin_edges))
        #error()
    end
    return scale,shape
end

function compute_GPD_params_from_truncated_gaussian_mixture(means::Vector{Float64}, stds::Vector{Float64}, lolim::Float64)
    # use highest of the 0.001 quantiles for the upper limit
    cquant_hi = 1e-3
    uplim = lolim
    for (m,s) in zip(means,stds)
        D = Dists.Normal(m,s)
        ccdf_lolim = Dists.ccdf(D, lolim)
        if ccdf_lolim > 1e-10
            uplim = max(uplim, Dists.cquantile(D, ccdf_lolim*cquant_hi))
        end
        if !isfinite(uplim)
            @show ccdf_lolim
            @show m,s
            @show lolim
            @show uplim
            error()
        end

    end
    Nbins = 100
    Ncomponents = length(means)
    bin_edges = collect(range(lolim, uplim, Nbins+1))
    bin_weights = zeros(Float64, Nbins)
    for (m,s) in zip(means,stds)
        D = Dists.Normal(m,s)
        cdf_component = Dists.cdf.(D, bin_edges)
        bin_weights .+= diff(cdf_component)/Ncomponents
    end
    scale,shape = compute_GPD_params_from_histogram(bin_edges, bin_weights; method="MM")
    @show scale,shape
    return (scale,shape)
end


function quadratic_model_2d(X::Matrix{Float64}, coefs::Vector{Float64})
    @assert size(X,2) == 2 # each column is a feature 
    Y = (
         coefs[1]
         .+ coefs[2] .* X[:,1] .+ coefs[3] .* X[:,2]
         .+ coefs[4] .* X[:,1].^2 .+ coefs[5] .* X[:,1] .* X[:,2] .+ coefs[6] .* X[:,2].^2
        )
    return Y
end
function quadratic_model_2d_zero_intercept(X::Matrix{Float64}, coefs::Vector{Float64})
    @assert size(X,2) == 2 # each column is a feature 
    Y = (
         coefs[1] .* X[:,1] .+ coefs[2] .* X[:,2]
         .+ coefs[3] .* X[:,1].^2 .+ coefs[4] .* X[:,1] .* X[:,2] .+ coefs[5] .* X[:,2].^2
        )
    return Y
end

function quadratic_regression_2d(X::Matrix{Float64}, Y::Vector{Float64}; intercept::Union{Nothing,Float64}=nothing)
    if isnothing(intercept)
        regression_fun = quadratic_model_2d
        coefs_init_guess = zeros(Float64, 6)
    else
        regression_fun = ((x,c) -> intercept .+ quadratic_model_2d_zero_intercept(x,c))
        coefs_init_guess = zeros(Float64, 5)
    end
    fit = LsqFit.curve_fit(regression_fun, X, Y, coefs_init_guess)
    coefs = LsqFit.coef(fit)
    if !isnothing(intercept)
        pushfirst!(coefs, intercept)
    end
    mse = SB.mean(fit.resid.^2)
    rsquared = 1 - mse/SB.var(Y, corrected=false)
    resid_range = [extrema(fit.resid)...]
    return coefs,mse,rsquared,resid_range
end

function linear_model_2d(X::Matrix{Float64}, coefs::Vector{Float64})
    @assert size(X,2) == 2 # each column is a feature 
    # TODO enforce it goes through zero 
    Y = (
         coefs[1]
         .+ coefs[2] .* X[:,1] .+ coefs[3] .* X[:,2]
        )
    return Y
end

function linear_model_2d_zero_intercept(X::Matrix{Float64}, coefs::Vector{Float64})
    @assert size(X,2) == 2 # each column is a feature 
    # TODO enforce it goes through zero 
    Y = (
         coefs[1] .* X[:,1] .+ coefs[2] .* X[:,2]
        )
    return Y
end

function linear_regression_2d(X::Matrix{Float64}, Y::Vector{Float64}; intercept::Union{Nothing,Float64} = nothing)
    if isnothing(intercept)
        regression_fun = linear_model_2d
        coefs_init_guess = zeros(Float64, 3)
    else
        regression_fun = ((x, c) -> intercept .+ linear_model_2d_zero_intercept(x, c))
        coefs_init_guess = zeros(Float64, 2)
    end
    fit = LsqFit.curve_fit(regression_fun, X, Y, coefs_init_guess)
    coefs = LsqFit.coef(fit)
    if !isnothing(intercept)
        pushfirst!(coefs, intercept)
    end
    mse = SB.mean(fit.resid.^2)
    rsquared = 1 - mse/SB.var(Y, corrected=false)
    resid_range = [extrema(fit.resid)...]
    return coefs,mse,rsquared,resid_range
end

function quadratic_regression_2d_eigs(coefs::Vector{Float64})
    H = reshape([coefs[4],coefs[5]/2,coefs[5]/2,coefs[6]], (2,2))
    E = LA.eigen(H)
    order = sortperm(abs.(E.values); rev=true)
    # TODO maximize the quadratic on the bounded domain
    return (E.values[order],E.vectors[:,order])
end


# ---------------- functions to propagate uncertainty from the input distribution (on perturbatiosn) to the output distribution (on scores) 
function regression2distn_linear_gaussian(coefs::Vector{Float64}, input_std::Float64, levels::Vector{Float64})
    return regression2distn_linear_gaussian(coefs, 0.0, input_std, levels) 
end

function regression2distn_linear_gaussian(coefs::Vector{Float64}, residmse::Float64, input_std::Float64, levels::Vector{Float64})
    # Gaussian input means Gaussian output
    output_mean = coefs[1]
    output_var = input_std^2 * sum(coefs[3:4].^2) + residmse
    D = Dists.Normal(output_mean, sqrt(output_var))
    ccdf = Dists.ccdf.(D, levels)
    @assert maximum(diff(ccdf)) <= 0
    @assert minimum(ccdf) >= 0
    pdf = Dists.pdf.(D, 0.5 .* (levels[1:end-1] .+ levels[2:end]))
    return ccdf,pdf
end

function regression2distn_linear_uniform(coefs::Vector{Float64}, input_radius::Float64, levels::Vector{Float64})
    output_mean = coefs[1]
    # In the following, subtract the mean
    Fmax = input_radius * sqrt(sum(coefs[2:3].^2))
    ccdf_fun(u) = begin
        if u >= Fmax
            return 0.0
        elseif u <= -Fmax
            return 1.0
        end
        frac = u/Fmax
        ccdf = 1/pi * (acos(frac) - frac*sqrt(1-frac^2))
        return ccdf
    end
    pdf_fun(u) = begin
        if u >= Fmax
            return 0.0
        elseif u <= -Fmax
            return 0.0
        end
        frac = u/Fmax
        pdf = 2/(pi*Fmax) * sqrt(1-frac^2)
        return pdf
    end
    ccdf = ccdf_fun.(levels .- output_mean)
    @assert maximum(diff(ccdf)) <= 0
    @assert minimum(ccdf) >= 0
    pdf = pdf_fun.(0.5 .* (levels[1:end-1] .+ levels[2:end]) .- output_mean)
    return ccdf,pdf
end

function regression2distn_linear_uniform(coefs::Vector{Float64}, residmse::Float64, input_radius::Float64, levels::Vector{Float64}, U::Matrix{Float64})
    Nsamp = size(U,1) #10000
    seed = 89281
    rng = Random.MersenneTwister(seed)
    #U = Random.rand(rng, Float64, (Nsamp,2))
    radius = sqrt.(U[:,1])
    angle = 2pi.*U[:,2]
    X = radius .* cos.(angle)
    Y = radius .* sin.(angle)
    F = vec(linear_model_2d(hcat(X,Y), coefs))
    if residmse > 0
        F .+= sqrt(residmse).*Random.randn(rng, Float64, (Nsamp,))
    end
    ccdf = vec(sum(F .> levels'; dims=1))./Nsamp
    @assert maximum(diff(ccdf)) <= 0
    @assert minimum(ccdf) >= 0
    pdf = -diff(ccdf) ./ diff(levels)
    return ccdf,pdf
end

function regression2distn_linear_uniform(coefs::Vector{Float64}, resid_range::Vector{Float64}, input_radius::Float64, levels::Vector{Float64}, U::Matrix{Float64})
    Nsamp = size(U,1) #10000
    seed = 89281
    rng = Random.MersenneTwister(seed)
    #U = Random.rand(rng, Float64, (Nsamp,2))
    radius = input_radius .* sqrt.(U[:,1])
    angle = 2pi.*U[:,2]
    X = radius .* cos.(angle)
    Y = radius .* sin.(angle)
    F = vec(linear_model_2d(hcat(X,Y)))
    if resid_range[2] > resid_range[1]
        F .+= resid_range[1] .+ (resid_range[2]-resid_range[1]).*Random.rand(rng, Float64, (Nsamp,))
    end
    ccdf = vec(sum(F .> levels'; dims=1))./Nsamp
    @assert maximum(diff(ccdf)) <= 0
    @assert minimum(ccdf) >= 0
    pdf = -diff(ccdf) ./ diff(levels)
    return ccdf,pdf
end

function regression2distn_quadratic_uniform(coefs::Vector{Float64}, input_radius::Float64, levels::Vector{Float64}, U::Matrix{Float64})
    return regression2distn_quadratic_uniform(coefs, 0.0, input_radius, levels, U)
end

function regression2distn_quadratic_uniform(coefs::Vector{Float64}, residmse::Float64, input_radius::Float64, levels::Vector{Float64}, U::Matrix{Float64})
    # exhaustively sample the disc

    #Nsamp = 40000
    Nsamp = size(U,1)
    radius = input_radius .* sqrt.(U[:,1])
    angle = 2pi.*U[:,2]
    X = radius .* cos.(angle)
    Y = radius .* sin.(angle)
    F = vec(quadratic_model_2d(hcat(X,Y), coefs))
    if residmse > 0
        seed = 89281
        rng = Random.MersenneTwister(seed)
        F .+= sqrt(residmse).*Random.randn(rng, Float64, (Nsamp,))
    end
    ccdf = vec(sum(F .> levels'; dims=1))./Nsamp
    pdf = -diff(ccdf) ./ diff(levels)
    return ccdf,pdf
end


function regression2distn_quadratic_uniform(coefs::Vector{Float64}, resid_range::Vector{Float64}, input_radius::Float64, levels::Vector{Float64}, U::Matrix{Float64})
    # exhaustively sample the disc

    Nsamp = size(U,1) #10000
    radius = input_radius .* sqrt.(U[:,1])
    angle = 2pi.*U[:,2]
    X = radius .* cos.(angle)
    Y = radius .* sin.(angle)
    F = vec(quadratic_model_2d(hcat(X,Y), coefs))
    if resid_range[2] > resid_range[1]
        seed = 89281
        rng = Random.MersenneTwister(seed)
        F .+= resid_range[1] .+ (resid_range[2]-resid_range[1]).*Random.rand(rng, Float64, (Nsamp,))
    end
    ccdf = vec(sum(F .> levels'; dims=1))./Nsamp
    pdf = -diff(ccdf) ./ diff(levels)
    return ccdf,pdf
end

function bump_density(U::AbstractMatrix{Float64}, scale::Float64, support_radius::Float64)
    Nsamp = size(U,1) #1024 
    R2 = (support_radius^2) .* U[:,1]
    radius = sqrt.(R2)
    angle = 2pi.*U[:,2]
    X = radius .* cos.(angle)
    Y = radius .* sin.(angle)
    W = exp.(-0.5*(R2./scale^2) ./ (1 .- R2./(support_radius^2)))
    return W
end

function plot_bump_densities_2d(scales::Vector{Float64}, support_radius::Float64, pert_seq::Matrix{Float64}, figdir::String)
    Nx = 50
    xs = collect(range(0, 1; length=Nx))
    X = xs * ones(Nx)'
    U = hcat(vec(X), vec(X'))
    Nscale = length(scales)
    W = zeros(Float64, (Nx^2,Nscale))
    for (i_scl,scl) in enumerate(scales)
        W[:,i_scl] .= bump_density(U, scl, support_radius)
    end
    Zs = SB.mean(W; dims=1)[1,:]
    loglevels2plot = log.(collect(range(exp(-4), exp(-0.01); length=20))) #collect(range(-4.0, 0.0; length=20)[1:end-1])
    fig = Figure(size=(500,400))
    lout = fig[1,1] = GridLayout()
    axcont = Axis(lout[1,1], xlabel="Re{ω}", ylabel="Im{ω}", xgridvisible=false, ygridvisible=false, title=@sprintf("𝑝(ω; 𝑠, 𝑊)\nfor scales 𝑠 ∈ {%.2f,%.2f,...,%.2f}, support 𝑊 = %.2f", scales[1], scales[2], scales[end], support_radius), titlefont=:regular)
    axslice = Axis(lout[1,2]; title="Transect\nRe{ω}=0", ylabel="Im{ω}", ylabelvisible=false, yticklabelsvisible=false, xscale=identity, xgridvisible=false, ygridvisible=false, titlefont=:regular, xticklabelrotation=-pi/2)
    pslice = zeros(Float64, Nx)
    imomegas = collect(range(-support_radius, support_radius; length=50))
    for (i_scl,scl) in enumerate(scales)
        # Draw an arc in a small sector of the circle
        thetas = collect(range((i_scl-1)/Nscale, i_scl/Nscale; length=10)).*2pi
        rs = 1 ./ sqrt.(1/support_radius^2 .- 1 ./ (2*scl^2 .* loglevels2plot))
        colargs = Dict(:color=>scl, :colormap=>:RdYlBu_4, :colorrange=>(scales[1],scales[end]))
        for r = rs
            lines!(axcont, r.*cos.(thetas), r.*sin.(thetas); colargs...)
        end
        lines!(axcont, [0, support_radius*cos(thetas[1])], [0, support_radius*sin(thetas[1])]; color=:gray, alpha=0.25)
        pslice .= exp.(-0.5 .* (imomegas./scl).^2 ./ (1 .- (imomegas./support_radius).^2)) ./ Zs[i_scl]
        lines!(axslice, pslice, imomegas; colargs..., )
    end
    thetas = collect(range(0, 2pi; length=100))
    lines!(axcont, support_radius.*cos.(thetas), support_radius.*sin.(thetas); color=:gray, alpha=0.25)
    Npert = size(pert_seq, 2)
    # Convert given perturbations into the plane
    R2 = support_radius^2 .* pert_seq[1,:]
    R = sqrt.(R2)
    thetas = 2pi.*pert_seq[2,:]
    xperts,yperts = R.*cos.(thetas), R.*sin.(thetas)
    scatter!(axcont, xperts, yperts; color=:black)
    for i_pert = 1:Npert
        text!(axcont, string(i_pert), position=(xperts[i_pert]+support_radius/30,yperts[i_pert]+support_radius*0), align=(:left,:center), color=:black, fontsize=10, font=:bold)
    end
    colsize!(lout, 1, Relative(4/5))
    for ax = (axcont,axslice)
        ylims!(ax, -support_radius*1.01, support_radius*1.01)
    end
    xlims!(axslice, 1e-10, 1/Zs[1,1])
    save(joinpath(figdir,"bumps.png"), fig)
end

function regression2distn_empirical_bump(scores::Vector{Float64}, scale::Float64, support_radius::Float64, levels::Vector{Float64}, U::Matrix{Float64})
    Nsamp = size(U,1) #1024 
    R2 = (support_radius^2) .* U[:,1]
    radius = sqrt.(R2)
    angle = 2pi.*U[:,2]
    X = radius .* cos.(angle)
    Y = radius .* sin.(angle)
    W = exp.(-0.5*(R2./scale^2) ./ (1 .- R2./(support_radius^2)))
    ccdf = vec(sum(W .* (scores .> levels'); dims=1))./sum(W)
    @assert all(isfinite.(ccdf))
    pdf = -diff(ccdf) ./ diff(levels)
    return ccdf,pdf
end

function regression2distn_quadratic_bump(coefs::Vector{Float64}, scale::Float64, support_radius::Float64, levels::Vector{Float64}, U::Matrix{Float64})
    return regression2distn_quadratic_bump(coefs, 0.0, scale, support_radius, levels, U)
end

function regression2distn_quadratic_bump(coefs::Vector{Float64}, residmse::Float64, scale::Float64, support_radius::Float64, levels::Vector{Float64}, U::Matrix{Float64})
    # exhaustively sample the disc

    Nsamp = size(U,1) #1024 
    R2 = (support_radius^2) .* U[:,1]
    radius = sqrt.(R2)
    angle = 2pi.*U[:,2]
    X = radius .* cos.(angle)
    Y = radius .* sin.(angle)
    F = vec(quadratic_model_2d(hcat(X,Y),coefs))
    if residmse > 0
        seed = 89281
        rng = Random.MersenneTwister(seed)
        F .+= sqrt(residmse).*Random.randn(rng, Float64, (Nsamp,))
    end
    W = exp.(-0.5*(R2./scale^2) ./ (1 .- R2./(support_radius^2)))
    ccdf = vec(sum(W .* (F .> levels'); dims=1))./sum(W)
    @assert all(isfinite.(ccdf))
    pdf = -diff(ccdf) ./ diff(levels)
    return ccdf,pdf
end

function regression2distn_linear_bump(coefs::Vector{Float64}, scale::Float64, support_radius::Float64, levels::Vector{Float64}, U::Matrix{Float64})
    return regression2distn_linear_bump(coefs, 0.0, scale, support_radius, levels, U)

end
function regression2distn_linear_bump(coefs::Vector{Float64}, residmse::Float64, scale::Float64, support_radius::Float64, levels::Vector{Float64}, U::Matrix{Float64})
    # exhaustively sample the disc

    Nsamp = size(U,1) #40000
    radius = support_radius .* sqrt.(U[:,1])
    angle = 2pi.*U[:,2]
    X = radius .* cos.(angle)
    Y = radius .* sin.(angle)
    F = vec(linear_model_2d(hcat(X,Y),coefs))
    if residmse > 0
        seed = 89281
        rng = Random.MersenneTwister(seed)
        F .+= sqrt(residmse).*Random.randn(rng, Float64, (Nsamp,))
    end
    R2 = X.^2 .+ Y.^2
    W = exp.(-0.5*(R2./scale^2) ./ (1 .- R2./(support_radius^2)))
    ccdf = vec(sum(W .* (F .> levels'); dims=1))./sum(W)
    pdf = -diff(ccdf) ./ diff(levels)
    @assert all(isfinite.(ccdf))
    return ccdf,pdf
end
# ------------ Other statistics utilities -----------
#
function pdf2pmfnorm(pdf, edges)
    @assert minimum(pdf) >= 0
    pmf = pdf .* diff(edges)
    pmf ./= sum(pmf)
    return pmf
end

function check_ccdf_validity(ccdf::AbstractVector{Float64})
    return (minimum(ccdf) >= 0) & (maximum(diff(ccdf)) <= 0) & (ccdf[1] > 0)
end


function ccdf2pmf(ccdf; normalize::Bool=true)
    @infiltrate !check_ccdf_validity(ccdf)
    @assert check_ccdf_validity(ccdf)
    pmf = vcat(-diff(ccdf), [ccdf[end]])
    if normalize
        pmf ./= ccdf[1] 
    end
    return pmf
end


function fdiv_fun_ccdf(ccdf1, ccdf2, levels1, levels2, fdivname)
    # second argument should be interpreted as ground truth 
    pmf1 = ccdf2pmf(ccdf1)
    pmf2 = ccdf2pmf(ccdf2)
    if fdivname == "chi2"
        return chi2div_fun(pmf1,pmf2)
    elseif fdivname == "kl"
        return kldiv_fun(pmf2,pmf1)
    elseif fdivname == "tv"
        return tvdist_fun(pmf1,pmf2)
    elseif fdivname == "qrmse"
        return quantile_rmse_from_unnormalized(ccdf1, ccdf2, levels1, levels2)
    else
        error("Only supported F-divergences are chi2,kl,tv")
    end
end

function tvdist_fun(pmf1, pmf2)
    return sum(abs.(pmf1 .- pmf2)) / 2
end

function chi2div_fun(pmf1, pmf2)
    if any((pmf1 .!= 0) .& (pmf2 .== 0))
        @show size(pmf1),size(pmf2)
        println("pmf1,pmf2")
        display(hcat(pmf1,pmf2))
        print("Not absolutely continuous")
        error()
    end
    idx = findall(pmf2 .!= 0)
    return sum((pmf1[idx] .- pmf2[idx]).^2 ./ pmf2[idx])
end

function xlogx(x)
    if x < 0
        error()
    elseif x == 0
        return 0
    else
        return x * log(x)
    end
end


function kldiv_fun(pmf1, pmf2)
    # pmf2 is the reference measure 
    # should measure mean (under pmf2) surprise when trying to represent it with pmf1
    if any((pmf1 .== 0) .& (pmf2 .!= 0))
        @show pmf1,pmf2
        println("Warning, not abs cont in KL")
        @infiltrate
        error()
        #return Inf
    end
    idx = findall((pmf2 .>= 0) .& (pmf1 .> 0))
    kl = sum(xlogx.(pmf2[idx]) .- pmf2[idx] .* log.(pmf1[idx]))
    if kl < 0
        @show pmf1,sum(pmf1)
        @show pmf2,sum(pmf2)
        error()
    elseif !isfinite(kl)
        @show pmf1,pmf2
        error()
    end
    return kl
end

function entropy_fun_samples(xs::Vector{Float64}, weights::Vector{Float64}, lolim::Float64)
    # from Learned-Miller 2003 
    @assert minimum(weights) > 0
    order = sortperm(xs)
    N = length(xs)
    N1 = findfirst(xs[order] .> lolim)
    x0 = (N1 == 1 ? lolim : xs[order[N1-1]])
    gaps = vcat(xs[order[N1]]-x0, diff(xs[order[N1:N]]))
    weights_exc = weights[order[N1:end]]./sum(weights)
    pdf_in_gaps = weights_exc ./ gaps
    entropy_contributions = -xlogx.(pdf_in_gaps) .* gaps
    #entropy_contributions[1] *= (xs[order[N1]] - lolim) ./ (xs[order[N1]] - x0)
    condent = sum(entropy_contributions)
    #weightsum = sum(weights) 
    #dxs = vcat(xs[order[N1]]-x0, diff(xs[order[N1:N]]))
    #pdf_vals = weights[order[N1:N]] ./ (weightsum .* dxs)
    #entropy_contributions = -weights[order[N1:end]] ./ weightsum .* log.(pdf_vals) .* dxs
    #entropy_contributions[1] *= (xs[order[N1]] - lolim) ./ (xs[order[N1]] - x0)
    #condent = sum(entropy_contributions)
    @infiltrate
    return condent
end

function expected_improvement_samples(xs::Vector{Float64}, weights::Vector{Float64}, lolim::Float64)
    order = sortperm(xs)
    N = length(xs)
    N1 = findfirst(xs[order] .> lolim)
    if xs[order[N]] <= lolim
        return 0.0
    end
    x0 = (N1 == 1 ? lolim : xs[order[N1-1]])
    weightsum = sum(weights) 
    exp_imp_contributions = (vcat((lolim+xs[order[N1]])/2, (xs[order[N1:N-1]] .+ xs[order[N1+1:N]])./2) .- lolim) .* weights[order[N1:N]] ./ weightsum 
    exp_imp_contributions[1] *= (xs[order[N1]] - lolim) / (xs[order[N1]] - x0)
    exp_imp = sum(exp_imp_contributions)
    return exp_imp
end

function ccdf_gridded_from_samples!(ccdf::AbstractArray{Float64}, pdf::AbstractArray{Float64}, xs::Vector{Float64}, weights::Vector{Float64}, levels::Vector{Float64})
    Nlev = length(levels)
    Nx = length(xs)
    order = sortperm(xs)
    tailsum = reverse(cumsum(reverse(weights[order])))
    tailsum ./= tailsum[1]
    i_x = 1
    ccdf_prev = 1.0
    ccdf_curr = 1.0
    for i_lev = 1:Nlev
        while i_x <= Nx && xs[order[i_x]] <= levels[i_lev]
            ccdf_prev = ccdf_curr
            ccdf_curr = (i_x == Nx ? 0 : tailsum[i_x+1])
            i_x += 1
        end
        if i_x == 1
            ccdf[i_lev] = ccdf_prev
        elseif 1 < i_x <= Nx
            frac = (levels[i_lev] - xs[order[i_x-1]]) / (xs[order[i_x]] - xs[order[i_x-1]])
            @assert (0 <= frac <= 1)
            if i_x == Nx
                ccdf[i_lev] = ccdf_prev * (1-frac)
            else
                ccdf[i_lev] = clamp(exp(log(ccdf_prev)*(1-frac) + log(ccdf_curr)*frac), ccdf_curr, ccdf_prev)
            end
        end # otherwise all remaining ccdf values are zero

    end
    @infiltrate !check_ccdf_validity(ccdf)
    pdf .= -diff(ccdf) ./ diff(levels)
    return 
end





function threshold_exceedance_probability_samples(xs::Vector{Float64}, weights::Vector{Float64}, lolim::Float64)
    order = sortperm(xs)
    N = length(xs)
    if xs[order[N]] <= lolim
        return 0.0
    end
    N1 = findfirst(xs[order] .> lolim)
    @infiltrate isnothing(N1)
    x0 = (N1 == 1 ? lolim : xs[order[N1-1]])
    weightsum = sum(weights) 
    prob_exc = sum(weights[order[N1+1:N]])
    prob_exc += (xs[order[N1]] - lolim) / (xs[order[N1]] - x0) * weights[order[N1]]
    prob_exc /= weightsum
    return prob_exc 
end

function entropy_fun_ccdf(ccdf; normalize::Bool=true)
    pmf = ccdf2pmf(ccdf; normalize=normalize)
    return -sum(xlogx.(pmf))
end

function quantile_rmse_from_unnormalized(ccdf1, ccdf2, levels1, levels2)
    N = 50
    either_is_zero = false
    logccdfmin = Inf
    if ccdf1[1] == 0
        either_is_zero = true
    else
        N1 = findlast(ccdf1 .> 0)
        logccdf1 = log.(ccdf1[1:N1])
        logccdf1 .-= logccdf1[1]
        logccdfmin = min(logccdfmin, logccdf1[end])
    end
    if ccdf2[1] == 0
        either_is_zero = true
    else
        N2 = findlast(ccdf2 .> 0)
        logccdf2 = log.(ccdf2[1:N2])
        logccdf2 .-= logccdf2[1]
        logccdfmin = min(logccdfmin, logccdf2[end])
    end
    dlogccdf = -logccdfmin/(N-2)
    logccdf = collect(range(0, logccdfmin-dlogccdf; length=N))
    if ccdf1[1] == 0
        levels1_interp = levels1[1] .* ones(Float64, N)
    else
        levels1_interp = interpolate_logccdf_to_grid(levels1[1:N1], logccdf1, logccdf)
    end
    if ccdf2[1] == 0
        levels2_interp = levels2[1] .* ones(Float64, N)
    else
        levels2_interp = interpolate_logccdf_to_grid(levels2[1:N2], logccdf2, logccdf)
    end
    return sqrt(sum((levels1_interp .- levels2_interp).^2) / N)
end



function interpolate_logccdf_to_grid(levels_src, logccdf_src, logccdf_dst)
    Ndst = length(logccdf_dst)
    Nsrc = length(logccdf_src)
    @assert 0 == logccdf_src[1] == logccdf_dst[1]
    @assert (Nsrc == 1) || (minimum(diff(levels_src)) > 0)
    @assert maximum(diff(logccdf_dst)) < 0
    @assert logccdf_dst[end] < logccdf_src[end]
    levels_dst = zeros(Float64, Ndst)
    levels_dst[1] = levels_src[1]
    i_dst_prev = 1
    i_dst_next = 2
    i_src_prev = 1 # keep track of this in case we need to skp one 
    for i_src = 1:Nsrc
        if logccdf_src[i_src] == logccdf_src[i_src_prev]
            i_src_prev = i_src
            continue
        end
        i_dst_next = i_dst_prev + findfirst(logccdf_src[i_src] .> logccdf_dst[i_dst_prev+1:Ndst]) - 1
        # account for multiple values at same ccdf value
        
        dlev_dlogccdf = (levels_src[i_src] - levels_src[i_src_prev]) / (logccdf_src[i_src] - logccdf_src[i_src_prev])
        @assert dlev_dlogccdf <= 0
        levels_dst[i_dst_prev+1:i_dst_next] .= levels_src[i_src_prev] .+ (dlev_dlogccdf) .* (logccdf_dst[i_dst_prev+1:i_dst_next] .- logccdf_src[i_src_prev])
        @assert levels_dst[i_dst_next] <= levels_src[Nsrc]
        i_dst_prev = i_dst_next
        i_src_prev = i_src
    end
    levels_dst[i_dst_next+1:Ndst] .= levels_src[Nsrc]
    return levels_dst
end

function test_interpolate_logccdf_to_grid()
    levels_src = [9,9.5,10.25,10.75,11.1,11.2,11.8]
    logccdf_src = [0,-1.5,-2.25,-2.75,-3.5,-3.8,-5.5]
    logccdf_dst = collect(range(0,-7;step=-1))
    levels_dst = interpolate_logccdf_to_grid(levels_src,logccdf_src,logccdf_dst)
    return levels_dst
end


   









