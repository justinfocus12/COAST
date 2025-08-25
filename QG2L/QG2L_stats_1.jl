function compute_observable_local_moments(hist_filenames::Vector{String}, obs_fun::Function, sdm::SpaceDomain; num_output_layers=2)
    # TODO pre-allocate the obs_val array to not do excessive allocation
    num_moments = 4
    moments = zeros(Float64, (sdm.Nx, sdm.Ny, 2, num_moments+1))
    moments_xall = zeros(Float64, (1, sdm.Ny, 2, num_moments+1))
    println("Opening files...")
    for (i,hist_filename) in enumerate(hist_filenames)
        print("$i, ")
        obs_val = histfile2obs(hist_filename, obs_fun)
        for m = 0:num_moments
            moments[:,:,:,m+1] .+= sum(obs_val.^m, dims=4)
            moments_xall[:,:,:,m+1] .+= sum(obs_val.^m, dims=(1,4))
        end
    end
    println()
    moments[:,:,:,2:end] ./= moments[:,:,:,1]
    moments_xall[:,:,:,2:end] ./= moments_xall[:,:,:,1]
    # Standardize the moments
    standardize_moments(mom::Array{Float64,4}) = begin
        momstandard = zeros(Float64, (size(mom,1), size(mom,2), size(mom,3), num_moments))
        m1,m2,m3,m4 = (@view(mom[:,:,:,m]) for m=2:5)
        sigsq = replace(max.(0, m2 - m1.^2), 0=>NaN)
        momstandard[:,:,:,1] .= m1
        momstandard[:,:,:,2] .= sqrt.(sigsq)
        momstandard[:,:,:,3] .= (m3 - 3*m2.*m1 + 2*m1.^3) ./ sigsq.^(3/2)
        momstandard[:,:,:,4] .= (m4 - 4*m3.*m1 + 6*m2.*m1.^2 - 3*m1.^4) ./ sigsq.^2 # not excess
        return momstandard
    end
    mssk = standardize_moments(moments)
    mssk_xall = standardize_moments(moments_xall)
    return (moments,mssk,moments_xall,mssk_xall)
end

function compute_observable_local_extrema(hist_filenames::Vector{String}, obs_fun::Function, sdm::SpaceDomain; num_output_layers=2)
    min_vals = Inf * ones(Float64, (sdm.Nx, sdm.Ny, 2))
    max_vals = -Inf * ones(Float64, (sdm.Nx, sdm.Ny, 2))
    println("Opening files...")
    for (i,hist_filename) in enumerate(hist_filenames)
        print("$i, ")
        obs_val = histfile2obs(hist_filename, obs_fun)
        min_vals .= min.(min_vals, minimum(obs_val, dims=4))
        max_vals .= max.(max_vals, maximum(obs_val, dims=4))
    end
    println()
    return (min_vals,max_vals)
end

function compute_local_GEV_params(hist_filenames::Vector{String}, obs_fun::Function, block_size_phys::Float64, sdm::SpaceDomain)
    U = cat((histfile2obs(fname, obs_fun) for fname=hist_filenames)..., dims=4)
    Nx,Ny,Nz,Nt = size(U)
    #@show size(U)
    #@assert size(U)[1:3] == (sdm.Nx, sdm.Ny, 2)
    # TODO determine block size automatically from decorrelation 
    block_size = round(Int, block_size_phys/sdm.tu)
    num_blocks = floor(Int, Nt/block_size)
    @show num_blocks
    function gev_fit(data)
        block_ends = collect(range(1, num_blocks; step=1)) .* block_size
        block_starts = block_ends .- (block_size - 1)
        block_maxima = NaN .* ones(Float64, num_blocks)
        for i_block = 1:num_blocks
            block_maxima[i_block] = maximum(data[block_starts[i_block]:block_ends[i_block]])
        end
        if !all(isfinite.(block_maxima))
            println("block_maxima = ")
            display(block_maxima)
            error("Some block maxima are not finite")
        end
        model = Ext.gevfitpwm(block_maxima)
        location = model.θ̂[1]
        scale = exp(model.θ̂[2])
        shape = model.θ̂[3]
        return (location,scale,shape)
    end


    gev_par = mapslices(gev_fit, U, dims=(4,))
    location,scale,shape = [zeros(Float64, (Nx,Ny,Nz)) for _=1:3]
   
    for ix = 1:Nx
        for iy = 1:Ny
            for iz = 1:Nz
                location[ix,iy,iz],scale[ix,iy,iz],shape[ix,iy,iz] = gev_par[ix,iy,iz]
            end
        end
    end
    return location,scale,shape
end

function compute_mean_residual_life(Xoft::Vector{Float64}, threshes::Vector{Float64}; lead_time=0, follow_time=0)
    peak_vals_lolim,_,_,_ = peaks_over_threshold(Xoft, threshes[1]; lead_time=lead_time, follow_time=follow_time)
    Nthresh = length(threshes)
    dthresh = threshes[2] - threshes[1]
    mrl = zeros(Float64, Nthresh)
    for (i_thresh,thresh) in enumerate(threshes)
        mrl[i_thresh] = SB.mean(filter(x->x>thresh, peak_vals_lolim)) - thresh
    end
    # Find a range of maximum agreement with linearity 
    radius = 4
    meansquare_nonlinearity = zeros(Float64, Nthresh)
    meansquare_nonlinearity[1:radius] .= Inf
    meansquare_nonlinearity[Nthresh-radius+1:Nthresh] .= Inf
    local_slopes = NaN .* ones(Float64, Nthresh)
    for i_thresh = (radius+1):(Nthresh-radius)
        local_slopes[i_thresh] = (mrl[i_thresh+radius] - mrl[i_thresh-radius])/(threshes[i_thresh+radius]-threshes[i_thresh-radius])
        linear_model = mrl[i_thresh] .+ local_slopes[i_thresh] .* range(-radius, radius; step=1).*dthresh
        meansquare_nonlinearity[i_thresh] = SB.mean((mrl[(i_thresh-radius):(i_thresh+radius)] .- linear_model).^2)
    end
    meansquare_nonlinearity = replace(meansquare_nonlinearity, NaN=>Inf)
    @show meansquare_nonlinearity
    i_best_thresh = argmin(meansquare_nonlinearity)
    return local_slopes,mrl,threshes[i_best_thresh],meansquare_nonlinearity[i_best_thresh]
end

function flag_running_max(x::Vector{Float64}, buffer::Int64)
    # sequence of times that the peak is the maximum over the preceding (lead_time) window
    # Get running max over time horizons
    Nt = length(x)
    mop = -Inf .* ones(Float64, Nt) # x is max of past
    xismop = ones(Bool, Nt)
    xismop[1:buffer] .= 0
    mop[buffer+1:Nt] .= x[buffer+1:Nt]
    idx_mop = (buffer+1):Nt
    for dt = 1:buffer
        idx_x = idx_mop .- dt
        xismop[idx_mop] .&= (mop[idx_mop] .> x[idx_x])
        mop[idx_mop] .= max.(mop[idx_mop], x[idx_x])
    end
    return xismop
end

function peaks_over_threshold(x::Vector{Float64}, thresh::Float64, prebuffer::Int64, postbuffer::Int64, initbuffer::Int64)
    if prebuffer + postbuffer + initbuffer >= length(x)
        return nothing
    end
    xismop = flag_running_max(x, prebuffer) # x is maximum of past
    xismof = reverse(flag_running_max(reverse(x), postbuffer)) # x is maximum of future 
    xispeak = zeros(Bool,length(x))
    xispeak .= (xismop .& xismof .& (x .> thresh))
    xispeak[1:initbuffer] .= false
    peak_tidx = findall(xispeak)
    # Make sure there's an upcrossing before
    if length(peak_tidx) == 0
        println("NO PEAKS!!")
        return nothing
    end
    @show thresh
    @show peak_tidx[1],peak_tidx[end]
    @show x[peak_tidx[1]], x[peak_tidx[end]]
    @show prebuffer
    # ensure there's an upcrossing before and a downcrossing after 
    peak_vals = x[peak_tidx]
    Npeak = length(peak_tidx)

    upcross_tidx,downcross_tidx = (zeros(Int64, Npeak) for _=1:2)
    # make sure an upcross happens before and a downcross happens after 
    first_peak = 1
    last_peak = Npeak
    for (i_peak,it) in enumerate(peak_tidx)
        itbelow = findlast(x[1:it-1] .< thresh)
        if isnothing(itbelow)
            first_peak = i_peak + 1
        else
            upcross_tidx[i_peak] = itbelow + 1
        end
        itbelow = findfirst(x[(it+1):end] .< thresh)
        if isnothing(itbelow)
            last_peak = i_peak - 1
        else
            downcross_tidx[i_peak] = it + itbelow
        end
    end
    if first_peak > last_peak
        println("WARNING found peaks, but not surrounded by anything under threshold")
        return nothing
    end
    results = (peak_vals, peak_tidx, upcross_tidx, downcross_tidx)
    for v = results
        v = v[first_peak:last_peak]
    end
    #IFT.@infiltrate (length(peak_vals) == 1)
    println("^^^^^^^^^^^\n NUMBER OF PEAKS = $(length(results[1]))\n^^^^^^^^^^^^^^^^")
    return results
end

        
function peaks_over_threshold_old(Xoft, thresh_hi; thresh_lo=nothing, lead_time=0, follow_time=0)
    # TODO augment this with a high- and low-threshold pair, with a dip below the low threshold being required between two consecutive peaks above the high one (a la TPT with sets A and B)
    # NEw CONDITION: the peak needs to be the maximum of the preceeding (lead_time) interval 
    peak_vals = Vector{Float64}([])
    peak_tidx = Vector{Int64}([])
    upcross_tidx = Vector{Int64}([])
    downcross_tidx = Vector{Int64}([])
    if isnothing(thresh_lo)
        thresh_lo = thresh_hi
    end
    Nt = length(Xoft)
    it = findfirst(Xoft[(lead_time+2):end] .< thresh_hi)
    if isnothing(it)
        return peak_vals,peak_tidx,upcross_tidx,downcross_tidx
    end
    it += lead_time + 1
    it_upcross = it
    it_downcross = it
    more_peaks_ahead = true
    while more_peaks_ahead
        it_upcross = findfirst(Xoft[it+1:end] .> thresh_hi)
        if isnothing(it_upcross)
            more_peaks_ahead = false
            continue
        end
        it_upcross += it
        #@show Xoft[it_upcross-1:it_upcross]
        it_downcross = findfirst(Xoft[it_upcross+1:end] .< thresh_lo)
        if isnothing(it_downcross)
            more_peaks_ahead = false
            continue
        end
        it_downcross += it_upcross
        #@show Xoft[it_downcross-1:it_downcross]
        it_peak = it_upcross-1 + argmax(Xoft[it_upcross:it_downcross-1])
        if !((Xoft[it_peak] >= Xoft[it_peak-1]) && (Xoft[it_peak] >= Xoft[it_peak+1]))
            @show Xoft[it_peak-1:it_peak]
            error()
        end
        push!(peak_vals, Xoft[it_peak])
        push!(peak_tidx, it_peak)
        push!(upcross_tidx, it_upcross)
        push!(downcross_tidx, it_downcross)
        it = it_downcross + follow_time + lead_time
        it_next = findfirst(Xoft[it+1:end] .< thresh_lo)
        if isnothing(it_next)
            more_peaks_ahead = false
            continue
        end
        it += it_next
    end
    return (peak_vals, peak_tidx, upcross_tidx, downcross_tidx)
end

function compute_GPD_params_from_moments(thresh, mean, var)
    shape = 0.5*(1 - (mean - thresh)^2/var)
    scale = (mean - thresh)*(1 - shape)
    return [scale, shape]
end

function strrep_GPD(thresh, scale, shape)
    mustr = @sprintf("%.1f", thresh)
    scstr = @sprintf("%.2f", scale)
    shstr = @sprintf("%+3.2f", shape)
    gpdstr = "GPD($(mustr),$(scstr),$(shstr))"
    return gpdstr
end

function strrep_GPD(D)
    return strrep_GPD(D.μ,D.σ,D.ξ)
end

function compute_GPD_params_from_histogram(thresh, bin_edges, bin_weights)
    bin_centers = 0.5 .* (bin_edges[1:end-1] .+ bin_edges[2:end])
    sumweights = sum(bin_weights)
    mean = sum(bin_centers .* bin_weights)./sumweights
    var = sum(bin_centers.^2 .* bin_weights)./sumweights - mean^2
    return compute_GPD_params_from_moments(thresh, mean, var)
end

function compute_GPD_params_from_ccdf(thresh, levels, ccdf)
    Nlev = length(levels)
    bin_edges = vcat(levels, [2*levels[Nlev]-levels[Nlev-1]])
    bin_weights = -diff(vcat(ccdf, [0.0])) ./ diff(bin_edges)
    return compute_GPD_params_from_histogram(thresh, bin_edges, bin_weights)
end


function compute_GPD_params(peak_vals, thresh; method="MLE") #Xoft::Vector{Float64}, thresh_hi::Float64, thresh_lo::Float64; lead_time::Int64=0, follow_time::Int64=0)
    #peak_vals, peak_tidx, upcross_tidx, downcross_tidx = peaks_over_threshold(Xoft, thresh_hi; thresh_lo=thresh_lo, lead_time=lead_time, follow_time=follow_time)
    #@show size(peak_vals)
    #@show size(thresh)
    idx = findall(peak_vals .> thresh)
    if "PWM" == method
        gp_params = Ext.gpfitpwm(peak_vals[idx] .- thresh)
    elseif "MLE" == method
        gp_params = Ext.gpfit(peak_vals[idx] .- thresh)
    end
    scale = exp(gp_params.θ̂[1])
    shape = gp_params.θ̂[2]
    return [scale,shape]
end

function compute_local_objective_and_stats_zonsym(hist_filenames::Vector{String}, tfins::Vector{Int64}, tinitreq::Int64, tfinreq::Int64, obs_fun_xshifts, ccdf_levels::Vector{Float64})
    # TODO compute for multiple timespans 
    # req = requested or required
    @assert tfinreq <= tfins[end]
    @assert minimum(diff(tfins)) > 0
    memfirst = findfirst(tfins .> tinitreq)
    memlast = findfirst(tfins .>= tfinreq)
    @show memfirst,memlast,tinitreq,tfinreq
    Rfun_seplon(f::JLD2.JLDFile) = reduce(hcat, (obs_fun(f) for obs_fun=obs_fun_xshifts))
    Roft_seplon = reduce(vcat, compute_observable_ensemble(hist_filenames[memfirst:memlast], Rfun_seplon)) # (timesteps) x (longitudes) vector
    Nt,Nlon = size(Roft_seplon)
    tinit = tfins[memfirst] - (size(Roft_seplon,1) - (tfins[memlast]-tfins[memfirst]))
    @show tinit,tfins[memfirst],tfins[memlast],size(Roft_seplon)
    @assert tinitreq >= tinit
    tidx = (tinitreq - tinit + 1):1:(tfinreq - tinit)
    Rccdf_seplon = reduce(vcat, (quantile_sliced(Roft_seplon, 1-ccdf_level, 1) for ccdf_level=ccdf_levels))
    Rccdf_agglon = [SB.quantile(vec(Roft_seplon), 1-ccdf_level) for ccdf_level=ccdf_levels]
    tgridreq = collect((tinitreq+1):1:tfinreq)
    return (tgridreq,Roft_seplon,Rccdf_seplon,Rccdf_agglon)
end

function compute_local_pot_zonsym(Roft_seplon::Matrix{Float64}, levels_geq_thresh::Vector{Float64}, prebuffer::Int64, postbuffer::Int64, initbuffer::Int64, boost_cost_per_ancestor::Int64)
    # first entry of levels_geq_thresh is thresh itself 
    Nt,Nlon = size(Roft_seplon)
    Nlev = length(levels_geq_thresh)
    @show Nlev
    ccdf_pot_seplon = zeros(Float64, (Nlev,Nlon))
    ccdf_pot_agglon = zeros(Float64, Nlev)
    num_peaks_total = 0
    all_peaks = Vector{Float64}([])
    Npeaks_per_lon = zeros(Int64, Nlon)
    peak_tidx_eachlon = Vector{Vector{Int64}}([])
    peak_vals_eachlon = Vector{Vector{Float64}}([])
    for i_lon = 1:Nlon
        pot_results = peaks_over_threshold(Roft_seplon[:,i_lon], levels_geq_thresh[1], prebuffer, postbuffer, initbuffer)
        if isnothing(pot_results)
            continue
        end
        peak_vals,peak_tidx,upcross_tidx,downcross_tidx = pot_results
        # Record peak values and timings for the following loop with smaller sample sizes
        push!(peak_tidx_eachlon, peak_tidx)
        push!(peak_vals_eachlon, peak_vals)
        num_peaks_exceeding_level = sum(peak_vals .> levels_geq_thresh'; dims=1)[1,:]
        ccdf_pot_seplon[:,i_lon] .= num_peaks_exceeding_level ./ length(peak_vals)
        num_peaks_total += length(peak_vals)
        ccdf_pot_agglon .+= num_peaks_exceeding_level
        append!(all_peaks, peak_vals)
        # Take only subsets 
        # TODO detect when it' an unfair comparison, and restrict Nancsubs where necessary 
    end
    ccdf_pot_agglon ./= num_peaks_total
    mean_return_period = Nt*Nlon / num_peaks_total
    Nancsub_comparable_max = floor(Int, Nt/(boost_cost_per_ancestor + mean_return_period))
    Nancsubs = collect(unique(round.(Int,range(1, Nancsub_comparable_max; length=8))))
    N_Nancsub = length(Nancsubs)
    ccdf_pot_seplon_eqcost = zeros(Float64, (Nlev,Nlon,N_Nancsub))
    for i_lon = 1:Nlon
        for (i_Nancsub,Nancsub) in enumerate(Nancsubs)
            equal_cost_timespan = (boost_cost_per_ancestor + mean_return_period) * Nancsub
            num_peaks = sum(peak_tidx_eachlon[i_lon].< equal_cost_timespan)
            num_peaks_exceeding_level = sum(peak_vals_eachlon[i_lon][1:num_peaks] .> levels_geq_thresh'; dims=1)[1,:]
            ccdf_pot_seplon_eqcost[:,i_lon,i_Nancsub] .= num_peaks_exceeding_level ./ num_peaks
        end
    end
    # Also compute GPD parameters here
    gpdpar_agglon = compute_GPD_params(all_peaks, levels_geq_thresh[1])
    std_agglon = SB.mean(SB.std(Roft_seplon; dims=1); dims=2)[1,1]
    return (ccdf_pot_seplon, ccdf_pot_agglon, gpdpar_agglon, std_agglon, ccdf_pot_seplon_eqcost, mean_return_period,Nancsubs)
end

function compute_local_GPD_params_zonsym_multiple_fits(hist_filenames::Vector{String}, obs_fun_xshiftable::Function, prebuffer_time::Int64, follow_time::Int64, initbuffer::Int64, Nxshifts::Int64, xstride::Int64, figdir::String, obs_label)
    # should return a scalar 
    @show Nxshifts, xstride
    obs_fun_allshifts(f::JLD2.JLDFile) = reduce(hcat, (obs_fun_xshiftable(f, xshift) for xshift=range(0,Nxshifts-1,step=1).*xstride))
    obs_val_allshifts = reduce(vcat, compute_observable_ensemble(hist_filenames, obs_fun_allshifts))
    println("computed all shifts")
    @show size(obs_val_allshifts)
    # uncdertainty quantification by looking at different longitudes separatel
    Nthresh = 15
    Nthresh_actual = Nthresh # might reduce length in case no peaks are found 
    m1 = maximum(obs_val_allshifts)
    m2 = maximum(filter(x->x<m1, obs_val_allshifts))
    threshes = collect(range(SB.mean(obs_val_allshifts), m2; length=Nthresh))
    scales,shapes = (NaN.*ones(Float64, (Nthresh, Nxshifts)) for _=1:2)
    best_threshes,meansquare_nonlinearities = (zeros(Float64, Nxshifts) for _=1:2)
    for i_shift = 1:Nxshifts
        for i_thresh = 1:Nthresh
            pot = peaks_over_threshold(obs_val_allshifts[:,i_shift], threshes[i_thresh], prebuffer_time, follow_time, initbuffer)
            if isnothing(pot)
                Nthresh_actual = min(Nthresh_actual, i_thresh)
                continue
            end
            peak_vals = pot[1]
            @show i_thresh,threshes[i_thresh],length(peak_vals)
            if length(peak_vals) > 5
                @show extrema(peak_vals)
                scales[i_thresh,i_shift],shapes[i_thresh,i_shift] = compute_GPD_params(peak_vals, threshes[i_thresh]; method="MLE")
            end
        end
    end
    Nthresh = Nthresh_actual
    threshes = threshes[1:Nthresh]
    scales = scales[1:Nthresh,:]
    shapes = shapes[1:Nthresh,:]
    scales_xmean = vec(SB.mean(scales; dims=2))
    scales_xstd = vec(SB.std(scales; dims=2))
    shapes_xmean = vec(SB.mean(shapes; dims=2))
    shapes_xstd = vec(SB.std(shapes; dims=2))

    # Choose parameters with minimum variability on either side 
    # Option 1: variance of shapes in neighborhood
    penalty_shapevar = Inf .* ones(Float64, Nthresh)
    for i_thresh = 3:Nthresh-2
        penalty_shapevar[i_thresh] = SB.std(shapes[i_thresh-2:i_thresh+2,:])
    end
    # Option 2: difference in mean-shape on either side
    dshape = abs.(diff(shapes_xmean))
    penalty_dshape= vcat([Inf], replace(max.(dshape[1:end-1], dshape[2:end]), NaN=>Inf), [Inf])
    # Option 3: mismatch between d(sigma)/d(mu) and xi
    dscale_dthresh = diff(scales; dims=1) ./ diff(threshes)
    penalty_slope = Inf .* ones(Float64, (Nthresh,Nxshifts))
    @show Nthresh,Nxshifts,size(dscale_dthresh),size(threshes),size(shapes)
    for i_thresh = 2:Nthresh-1
        penalty_slope[i_thresh,:] .= ((scales[i_thresh+1,:] .- scales[i_thresh-1,:]) .- (threshes[i_thresh+1]-threshes[i_thresh-1]) .* shapes[i_thresh,:]).^2
    end
    
    
    penalty = replace(vec(SB.mean(penalty_slope; dims=2)), NaN=>Inf)
    @show shapes_xmean
    @show penalty 
    i_best_thresh = argmin(penalty)
    thresh = threshes[i_best_thresh]
    @show thresh,i_best_thresh
    ccdf_at_thresh = SB.mean(obs_val_allshifts .> thresh)
    ccdf_at_thresh_str = @sprintf("%.2f", ccdf_at_thresh)


    nanquantile(a, q) = any(isnan.(a)) ? NaN : SB.quantile(a, q)
    rowquantile(A, q) = vec(mapslices(a->nanquantile(a,q), A; dims=2))
    scale_xq25 = rowquantile(scales, 0.25)
    scale_xq75 = rowquantile(scales, 0.75)
    shape_xq25 = rowquantile(shapes, 0.25)
    shape_xq75 = rowquantile(shapes, 0.75)

    # Now for a global GPD fit 
    peak_vals_xall,peak_tidx_xall,_,_, = peaks_over_threshold(obs_val_allshifts[:], thresh, prebuffer_time, follow_time, initbuffer)
    scale_xall,shape_xall = compute_GPD_params(peak_vals_xall, thresh)
    mustr = @sprintf("%.2f", thresh)
    sigstr = @sprintf("%.2f", scale_xall)
    xistr = @sprintf("%.2f", shape_xall)
    GPDstr = L"GPD($%$(obs_label)^\theta=%$(mustr),\sigma=%$(sigstr),\xi=%$(xistr)$"

    GPD_xall = Dists.GeneralizedPareto(thresh, scale_xall, shape_xall)




    ccdf_GPD_xall = ccdf_at_thresh .* Dists.ccdf(GPD_xall, threshes)
    edges = vcat(threshes, [2*threshes[Nthresh]-threshes[Nthresh-1]])
    hg = SB.fit(SB.Histogram, peak_vals_xall, edges)
    #hg = SB.fit(SB.Histogram, obs_val_allshifts[:], edges)
    ccdf_emp_xall = reverse(cumsum(reverse(hg.weights)))./sum(hg.weights) .* ccdf_at_thresh


    ccdfs_emp = zeros(Float64, (Nthresh, Nxshifts))
    peak_vals = Vector{Vector{Float64}}([])
    peak_tidx,upcross_tidx,downcross_tidx = (Vector{Vector{Int64}}([]) for _=1:3)
    for i_shift = 1:Nxshifts
        pot = peaks_over_threshold(obs_val_allshifts[:,i_shift], thresh, prebuffer_time, follow_time, initbuffer)
        push!(peak_vals, pot[1])
        push!(peak_tidx, pot[2])
        push!(upcross_tidx, pot[3])
        push!(downcross_tidx, pot[4])
        hg = SB.fit(SB.Histogram, peak_vals[i_shift], edges)
        ccdfs_emp[:,i_shift] .= reverse(cumsum(reverse(hg.weights)))./sum(hg.weights) .* ccdf_at_thresh
    end


    fig = Figure(size=(400,600))
    lout = fig[1:4,1] = GridLayout()
    ax1 = Axis(lout[1,1]; xlabel=L"$\mu$", ylabel=L"$\sigma$", xgridvisible=false, ygridvisible=false, xlabelvisible=false, xticklabelsvisible=false)
    ax2 = Axis(lout[2,1]; xlabel=L"$\mu$", ylabel=L"$\xi$", xgridvisible=false, ygridvisible=false)
    ax3 = Axis(lout[3,1]; xlabel=L"$%$(obs_label)$", ylabel=L"CCDF$$", xgridvisible=false, ygridvisible=false, xlabelvisible=true, xticklabelsvisible=true, yscale=log10)
    for i_shift = 1:Nxshifts
        kwargs = Dict(:color=>:gray, :alpha=>0.5)
        lines!(ax1, threshes, scales[:,i_shift]; kwargs...,)
        lines!(ax2, threshes, shapes[:,i_shift]; kwargs...,)
        lines!(ax3, threshes[i_best_thresh:end], ccdfs_emp[i_best_thresh:end,i_shift]; kwargs..., label=L"one lon$$")
    end
    scatterlines!(ax1, threshes, scales_xmean; color=:black, linestyle=:solid, linewidth=2)
    hlines!(ax1, scale_xall; color=:red, linestyle=:dot, linewidth=3, label=L"$\sigma=%$(sigstr)$")
    scatterlines!(ax2, threshes, shapes_xmean; color=:black, linestyle=:solid, linewidth=2)
    hlines!(ax2, shape_xall; color=:red, linestyle=:dot, linewidth=3, label=L"$\xi=%$(xistr)$")
    dthresh = threshes[2] - threshes[1]
    lines!(ax1, [thresh-2*dthresh,thresh+2*dthresh], scale_xall .+ shape_xall*2*dthresh.*[-1,1]; color=:red, linewidth=3, linestyle=:solid, label=L"Slope $\xi$")
    lines!(ax3, threshes[i_best_thresh:end], ccdf_emp_xall[i_best_thresh:end]; color=:black, linestyle=:solid, linewidth=2, label=L"all lon$$")
    vlines!(ax3, thresh; color=:red, linestyle=:dot, linewidth=3, label=L"$\mu=%$(mustr)$ \n CCDF %$(ccdf_at_thresh_str)")
    hlines!(ax3, ccdf_at_thresh; color=:red, linestyle=:dot, linewidth=3)
    lines!(ax3, threshes[i_best_thresh:end], ccdf_GPD_xall[i_best_thresh:end]; color=:red, linestyle=:solid, linewidth=3, label=L"GPD$(\mu,\sigma,\xi)$")


    axislegend(ax1; merge=true, framevisible=false, position=:rt)
    axislegend(ax2; merge=true, framevisible=false, position=:rt)
    axislegend(ax3; merge=true, framevisible=false, position=:lb)

    linkxaxes!(ax1,ax2,ax3)
    rowgap!(lout, 1, 0.0)

    save(joinpath(figdir,"GPD_dns.png"), fig)
    # ---------------------------------------------------------------
    # Refine the levels
    return (
            ccdf_at_thresh,thresh,scale_xall,shape_xall,
            peak_vals,peak_tidx,upcross_tidx,downcross_tidx,
            threshes,ccdfs_emp,ccdf_GPD_xall
           )
end

