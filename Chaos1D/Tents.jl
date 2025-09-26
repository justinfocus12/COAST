# Verify the optimal Advance Split Time is what I think it is for the Bernoulli map
#
import Random
import StatsBase as SB
using Printf: @sprintf
using JLD2: jldopen
using CairoMakie

function van_der_corput(N)
    # Generate the first N points of the van der corput sequence 
    max_bit_length = floor(Int, 1+log2(N))
    xs = zeros(Float64, N)
    n = 1
    for bit_length = 1:max_bit_length
        for k = 1:(2^(bit_length-1))
            if n <= N
                xs[n] = (2*k-1)/(2^bit_length)
            end
            n += 1
        end
    end
    return xs
end
        


struct TentMapParams
    # For tent maps, the location of the peak; for bit shift, a little unclear
    tentpeak::Float64
end

function BoostParams()
    return (
            duration_valid = 2^16,
            duration_ancgen = 2^12, 
            duration_spinup = 2^4,
            threshold_neglog = 5, # 2^(-threshold_neglog) is the threshold
            perturbation_neglog = 12,  # how many bits to keep when doing the perturbation 
            min_cluster_gap = 2^6,
            bit_precision = 32,
            ast_min = 1,
            ast_max = 12,
            bst = 2,
            num_descendants = 15
           )
end

function strrep(bpar::NamedTuple)
    # For naming file 
    s = @sprintf("Tv%d_Ta%d_thr%d_prt%d_bp%d", round(Int, log2(bpar.duration_valid)), round(Int, log2(bpar.duration_ancgen)), bpar.threshold_neglog, bpar.perturbation_neglog, bpar.bit_precision)
    return s
end

function intensity(xs::Vector{Float64})
    return xs[1]
end

function intensity(xs::Matrix{Float64})
    return xs[1,:]
end

function get_themes()
    theme_ax = (xticklabelsize=8, yticklabelsize=8, xlabelsize=10, ylabelsize=10, xgridvisible=false, ygridvisible=false, titlefont=:bold, titlesize=10)
    theme_leg = (labelsize=8, framevisible=false)
    return theme_ax,theme_leg
end

function empirical_ccdf(x::Vector{<:Number})
    N = length(x)
    order = sortperm(x)
    ccdf = (collect(range(N, 1; step=-1)) .- 0.5)./N
    return x[order], ccdf
end

function simulate!(xs::Matrix{Float64}, bit_precision::Int64, x_init::Vector{Float64}, rng::Random.AbstractRNG)
    duration = size(xs, 2)
    x = x_init[1] # Within this function it is just the one
    for t = 1:duration
        x = mod(2*(x < 0.5 ? x : 1-x), 1)
        x = mod(
                (div(x, 1/(2^bit_precision)) + Random.rand(rng, [0,1]))
                / (2^bit_precision), 
                1
               )
        xs[1,t] = x
    end
    return
end

function simulate(x_init::Vector{Float64}, duration::Int64, bit_precision::Int64, rng::Random.AbstractRNG)
    xs = zeros(Float64, (1,duration))
    x = x_init[1]
    ts = collect(1:duration)
    for t = 1:duration
        x = mod(2*(x < 0.5 ? x : 1-x), 1)
        x = mod(
                (div(x, 1/(2^bit_precision)) + Random.rand(rng, [0,1]))
                / (2^bit_precision), 
                1
               )
        xs[1,t] = x
    end
    return xs, ts
end



function simulate(x0::Vector{Float64}, duration::Int64, bit_precision::Int64, rng::Random.AbstractRNG, datadir::String, outfile_suffix::String)
    xs, ts = simulate(x0, duration, bit_precision, rng)
    jldopen(joinpath(datadir, "dns_$(outfile_suffix).jld2"),"w") do f
        f["xs"] = xs
        f["ts"] = ts
    end
    return 
end

function plot_peaks_over_threshold(thresh::Float64, duration_spinup::Int64, duration_plot::Int64, datadir::String, figdir::String, file_suffix::String)

    ts, xs = jldopen(joinpath(datadir, "dns_$(file_suffix).jld2"), "r") do f
        return f["ts"], f["xs"]
    end
    Rs = intensity(xs) # pedantically, a scalar 
    ts_peak, Rs_peak, cluster_starts, cluster_stops = jldopen(joinpath(datadir, "dns_peaks_$(file_suffix).jld2"), "r") do f
        return (
                f["ts_peak"],
                f["Rs_peak"], 
                f["cluster_starts"], 
                f["cluster_stops"],
               )
    end
    
    ts2plot = duration_spinup .+ (1:duration_plot)
    peaks2plot = findall(ts2plot[1] .<= ts_peak .<= ts2plot[end])


    theme_ax,theme_leg = get_themes()
    fig = Figure(size=(400,400))
    lout = fig[1,1] = GridLayout()
    ax_Rs = Axis(lout[1,1]; theme_ax..., ylabel="𝑅(𝑋(𝑡))")
    ax_peaks = Axis(lout[2,1]; theme_ax..., ylabel="Peaks {𝑅(𝑋(𝑡ₙ*))}")
    ax_waits = Axis(lout[3,1]; theme_ax..., ylabel="𝑡*ₙ₊₁-𝑡*ₙ")
    ax_hist_Rs = Axis(lout[1,2]; theme_ax..., xlabel="ℙ{𝑅(𝑋)>𝑟}", yticklabelsvisible=false)
    ax_hist_peaks = Axis(lout[2,2]; theme_ax..., xlabel="ℙ{𝑅*>𝑟}")
    ax_hist_waits = Axis(lout[3,2]; theme_ax..., xlabel="ℙ{τ > 𝑡*ₙ₊₁-𝑡*ₙ}", xscale=log2)

    # Full timeseries
    lines!(ax_Rs, ts2plot, Rs[ts2plot]; color=:black)
    hlines!(ax_Rs, thresh; color=:gray, linewidth=1, alpha=0.5)
    scatter!(ax_Rs, ts_peak[peaks2plot], Rs_peak[peaks2plot]; color=:black, marker=:star5)
    bin_edges_Rs = collect(range(0, 1; length=65))
    bin_centers_Rs = (bin_edges_Rs[1:end-1] .+ bin_edges_Rs[2:end])./2
    hist_Rs = SB.normalize(SB.fit(SB.Histogram, Rs[cluster_starts[1]:cluster_stops[end]], bin_edges_Rs); mode=:pdf)
    scatterlines!(ax_hist_Rs, hist_Rs.weights, bin_centers_Rs; color=:black, markersize=4)
    ylims!(ax_hist_Rs, 0, 1)
    ylims!(ax_Rs, 0, 1)
    xlims!(ax_hist_Rs, 0, 1.25)
    linkyaxes!(ax_Rs, ax_hist_Rs)

    # Peak timeseries 
    scatter!(ax_peaks, ts_peak, Rs_peak; color=:black, marker=:circle)
    peaks_sorted,ccdf_peaks = empirical_ccdf(Rs_peak)
    scatterlines!(ax_hist_peaks, ccdf_peaks, peaks_sorted; color=:black, marker=:circle, markersize=2)
    ylims!(ax_peaks, thresh, 1.0)
    ylims!(ax_hist_peaks, thresh, 1.0)
    linkyaxes!(ax_peaks, ax_hist_peaks)
    xlims!(ax_hist_peaks, 0, 1)

    # Wait timeseries
    waits = diff(ts_peak)
    waits_sorted, ccdf_waits = empirical_ccdf(waits)
    scatter!(ax_waits, ts_peak[2:end], waits; color=:black, marker=:circle)
    scatterlines!(ax_hist_waits, ccdf_waits, waits_sorted; color=:black, marker=:circle, markersize=2)
    linkyaxes!(ax_waits, ax_hist_waits)

    for ax = (ax_hist_Rs, ax_hist_peaks, ax_hist_waits)
        ax.ylabelvisible = ax.yticklabelsvisible=false
    end

    rowgap!(lout, 1, 10)
    rowgap!(lout, 2, 10)
    colgap!(lout, 1, 0)
    colsize!(lout, 1, Relative(3/4))

    save(joinpath(figdir, "dns_peaks_over_threshold_$(file_suffix).png"), fig)
end

function fit_gpd_peaks_over_threshold(thresh::Float64, datadir::String, file_suffix::String)
    jldopen(joinpath(datadir, "dns_peaks_$(file_suffix).jld2"), "w") do f
        f["ts_peak"] = ts_peak
        f["Rs_peak"] = Rs[ts_peak]
        f["cluster_starts"] = cluster_starts
        f["cluster_stops"] = cluster_stops
    end
    # TODO fit a GPD, measure return periods, etc. The resulting parameters will go back into the plot of the POTs
end

    

function find_peaks_over_threshold(thresh::Float64, duration_spinup::Int64, duration_spinon::Int64, min_cluster_gap::Int64, datadir::String, file_suffix::String)

    # Collect all independent peaks over the threshold, fit a GPD
    ts,xs = jldopen(joinpath(datadir, "dns_$(file_suffix).jld2"), "r") do f
        return f["ts"], f["xs"]
    end
    # Count valleys rather than peaks, since valleys are simpler (just long-enough runs of values below threshold)
    tmax = duration_spinup + duration_spinon
    valley_starts,valley_stops = (Vector{Int64}([]) for _=1:2)
    valley_start = valley_stop = 0
    # Find first timestemp below threshold
    t = duration_spinup + 1
    while (t < tmax) && (xs[1,t] > thresh)
        t += 1
    end
    if t >= tmax
        return nothing
    end
    in_valley = true
    valley_length = 0
    while t < tmax
        if xs[1,t] <= thresh
            if valley_length == 0
                valley_start = t
            end
            valley_length += 1
        else
            if valley_length >= min_cluster_gap
                valley_stop = t-1
                push!(valley_starts, valley_start)
                push!(valley_stops, valley_stop)
            end
            valley_length = 0
        end
        t += 1
    end
    # If a valley has started but not finished, we should still count it as a valley
    if (xs[1,tmax] <= thresh) && (valley_length >= min_cluster_gap)
        push!(valley_starts, valley_start)
        push!(valley_stops, tmax)
    end
    # Time between valleys is mountains 
    Npeaks = length(valley_stops) - 1
    cluster_starts = valley_stops[1:Npeaks] .+ 1
    cluster_stops = valley_starts[2:Npeaks+1] .- 1
    @assert all(cluster_stops .>= cluster_starts)
    ts_peak = zeros(Int64, Npeaks)
    for i_peak = 1:Npeaks
        ts_peak[i_peak] = cluster_starts[i_peak] - 1 + argmax(cluster_starts[i_peak]:cluster_stops[i_peak])
    end

    jldopen(joinpath(datadir, "dns_peaks_$(file_suffix).jld2"), "w") do f
        f["ts_peak"] = ts_peak
        f["Rs_peak"] = intensity(xs[:,ts_peak])
        f["cluster_starts"] = cluster_starts
        f["cluster_stops"] = cluster_stops
    end
end

function plot_dns(duration_spinup, duration_spinon, datadir, figdir, outfile_suffix)
    xs,ts = jldopen(joinpath(datadir, "dns_$(outfile_suffix).jld2"), "r") do f
        return f["xs"],f["ts"]
    end
    t0 = duration_spinup
    t1 = t0 + duration_spinon
    h = SB.normalize(SB.fit(SB.Histogram, xs[1,t0:t1]; nbins=16); mode=:pdf)
    bincenters = (h.edges[1][1:end-1] .+ h.edges[1][2:end])./2

    Nt2plot = 128
    fig = Figure(size=(600,150))
    lout = fig[1,1] = GridLayout()
    theme_ax,theme_leg = get_themes()
    ax_ts = Axis(lout[1,1]; xlabel="𝑡", ylabel="𝑥", theme_ax...)
    ax_hist = Axis(lout[1,2]; xlabel="𝑝(𝑥)", ylabel="𝑥", ylabelvisible=false, yticklabelsvisible=false, theme_ax...)

    scatterlines!(ax_ts, ts[t0:t0+Nt2plot], xs[1,t0:t0+Nt2plot]; color=:black)
    xlims!(ax_hist, 0, 2)
    ylims!(ax_hist, 0, 1)
    ylims!(ax_ts, 0, 1)
    vlines!(ax_hist, 1.0; color=:black, linestyle=(:dash,:dense))
    scatterlines!(ax_hist, h.weights, bincenters; color=:steelblue2)

    colsize!(lout, 2, Relative(1/6))
    colgap!(lout, 1, 0)

    save(joinpath(figdir, "dns_timeseries_hist_$(outfile_suffix).png"), fig)


end

function boost_peaks(threshold::Float64, perturbation_width::Float64, asts::Vector{Int64}, bst::Int64, bit_precision::Int64, Ndsc_per_leadtime::Int64, seed::Int64, datadir::String, file_suffix::String; overwrite_boosts::Bool=false) 
    ts_anc, xs_anc = jldopen(joinpath(datadir, "dns_$(file_suffix).jld2"), "r") do f
        return f["ts"], f["xs"]
    end
    ts_peak, Rs_peak, cluster_starts, cluster_stops = jldopen(joinpath(datadir, "dns_peaks_$(file_suffix).jld2"), "r") do f
        return (
                f["ts_peak"],
                f["Rs_peak"], 
                f["cluster_starts"], 
                f["cluster_stops"],
               )
    end
    @show ts_peak[1:4]
    Npeaks = length(ts_peak)
    pert_seq = van_der_corput(Ndsc_per_leadtime) .* perturbation_width
    datafile = joinpath(datadir,"xs_dscs.jld2")
    boostfile = joinpath(datadir,"xs_dscs.jld2")
    if overwrite_boosts
        rm(boostfile)
    end

    iomode = (overwrite_boosts ? "w" : "a+")
    for i_peak = 1:Npeaks
        anckey = "ianc$(i_peak)"
        for (i_ast,ast) in enumerate(ast_min:ast_max)
            astkey = "iast$(i_ast)"
            t_split = ts_peak[i_peak] - ast
            x_init_anc = xs_anc[:,t_split-ts_anc[1]+1]
            xs_dsc = zeros(Float64, (1, ast+bst))
            # Depending on time and cost of simulation, maybe open the file inside the loop 
            jldopen(boostfile, "a+") do f
                Ndsc_already_simulated = anckey in keys(f) && astkey in keys(f[joinpath(anckey)]) ? length(f[joinpath(anckey,astkey)]) : 0
                for i_dsc = Ndsc_already_simulated+1:Ndsc_per_leadtime
                    rng = Random.MersenneTwister(seed)
                    x_init_dsc = [perturbation_width*div(x_init_anc[1], perturbation_width) + pert_seq[i_dsc]]
                    xs_dsc,ts_dsc = simulate(x_init_dsc, ast+bst, bit_precision, rng)
                    f[joinpath(anckey,astkey,"idsc$(i_dsc)","t_split")] = t_split
                    f[joinpath(anckey,astkey,"idsc$(i_dsc)","xs")] = xs_dsc
                    f[joinpath(anckey,astkey,"idsc$(i_dsc)","x_init")] = x_init_dsc
                end
            end
        end
    end
    return
end

function analyze_boosts(datadir::String, figdir::String, asts::Vector{Int64}, N_dsc::Int64, bst::Int64, bin_lower_edges::Vector{Float64}, i_bin_thresh::Int64)
    ts_anc, xs_anc = jldopen(joinpath(datadir, "dns_ancgen.jld2"), "r") do f
        return f["ts"], f["xs"]
    end
    ts_peak_valid, Rs_peak_valid, cluster_starts_valid, cluster_stops_valid = jldopen(joinpath(datadir, "dns_peaks_ancgen.jld2"), "r") do f
        return (
                f["ts_peak"],
                f["Rs_peak"], 
                f["cluster_starts"], 
                f["cluster_stops"],
               )
    end
    ts_peak, Rs_peak_anc, cluster_starts, cluster_stops = jldopen(joinpath(datadir, "dns_peaks_ancgen.jld2"), "r") do f
        return (
                f["ts_peak"],
                f["Rs_peak"], 
                f["cluster_starts"], 
                f["cluster_stops"],
               )
    end

    Rmax = maximum(Rs_peak_valid)
    N_bin = length(bin_lower_edges)

    ccdf_anc = compute_empirical_ccdf(Rs_peak_anc, bin_lower_edges)
    ccdf_valid = compute_empirical_ccdf(Rs_peak_valid , bin_lower_edges)
    # Store the following data:
    # - peak (timing,value) of ancestor
    # - peak (timing,value) of descendants at every (ancestor, AST)
    # - then calculate entropy curves for each one 

    # Put results into a big array: (i_dsc, i_ast, i_anc) ∈ [N_dsc] x [N_ast] x [N_anc]...or rather a vector of vector of vectors etc. to enable active sampling 
    N_ast = length(asts)
    N_anc = length(ts_peak)

    Rs_peak_dsc = zeros(Float64, (N_dsc, N_ast, N_anc))
    jldopen(joinpath(datadir,"xs_dscs.jld2"), "r") do f
        for i_anc = 1:N_anc
            for i_ast = 1:N_ast
                for i_dsc = 1:N_dsc
                    if !("ianc$(i_anc)" in keys(f))
                        @show i_anc,i_ast,i_dsc
                        @show keys(f)
                        error()
                    end
                    Rs_dsc = intensity(f[joinpath("ianc$(i_anc)", "iast$(i_ast)", "idsc$(i_dsc)","xs")])
                    Rs_peak_dsc[i_dsc,i_ast,i_anc] = maximum(Rs_dsc)
                end
            end
        end
    end

    # Calculate thresholded entropy 
    idx_astmaxthrent = zeros(Int64, N_anc)

    # Initialize conditional and mixed CCDFs, both full (including under threshold) and rectified (with accept reject)
    ccdfs_dsc,ccdfs_dsc_rect = (zeros(Float64, (N_b,N_ast,N_anc)) for N_b=(N_bin,N_bin-i_bin_thresh+1))
    ccdfs_moctail_astunif,ccdfs_moctail_astunif_rect = (zeros(Float64, (N_b,N_ast)) for N_b=(N_bin,N_bin-i_bin_thresh+1))
    ccdfs_moctail_astmaxthrent,ccdfs_moctail_astmaxthrent_rect = (zeros(Float64, (N_b,)) for N_b=(N_bin,N_bin-i_bin_thresh+1))
    thresholded_entropy = zeros(Float64, (N_ast, N_anc))
    for i_anc = 1:N_anc
        for i_ast = 1:N_ast
            thresholded_entropy[i_ast,i_anc] = compute_thresholded_entropy(Rs_peak_dsc[:,i_ast,i_anc], bin_lower_edges[i_bin_thresh:end])
            ccdfs_dsc[:,i_ast,i_anc] .= compute_empirical_ccdf(Rs_peak_dsc[:,i_ast,i_anc], bin_lower_edges)
            ccdfs_dsc_rect[:,i_ast,i_anc] .= ccdfs_dsc[i_bin_thresh:N_bin,i_ast,i_anc] .+ (1-ccdfs_dsc[i_bin_thresh]).*(Rs_peak_anc[i_anc] .> bin_lower_edges[i_bin_thresh:N_bin])
            ccdfs_moctail_astunif[:,i_ast] .+= ccdfs_dsc[:,i_ast,i_anc]./N_anc
            ccdfs_moctail_astunif_rect[:,i_ast] .+= ccdfs_dsc_rect[:,i_ast,i_anc]./N_anc
        end
        idx_astmaxthrent[i_anc] = argmax(thresholded_entropy[:,i_anc])
        # Oh wait but need to apply adjustment...
        ccdfs_moctail_astmaxthrent .+= ccdfs_dsc[:,idx_astmaxthrent[i_anc],i_anc]./N_anc
        ccdfs_moctail_astmaxthrent_rect .+= ccdfs_dsc_rect[:,idx_astmaxthrent[i_anc],i_anc]./N_anc
    end

    theme_ax,theme_leg = get_themes()
    # MoCTail estimator at (1) fixed lead times, (2) COASTs (which should all be the same...)
    # Row 1: uniform-AST mixed CCDFs
    # Row 2: Just one panel (maxthrent-AST mixed CCDF), positioned at average timing of max-AST
    fig = Figure(size=(100*N_ast, 400))
    lout = fig[1,1] = GridLayout()
    for i_ast = 1:N_ast
        ax = Axis(lout[1,N_ast-i_ast+1]; theme_ax..., title=@sprintf("−%d",asts[i_ast]), xscale=log2, yscale=nlg1m)
        scatterlines!(ax, ccdfs_moctail_astunif_rect[:,i_ast], bin_lower_edges[i_bin_thresh:N_bin]; color=:red)
    end
    ax = Axis(lout[2,1]; theme_ax..., xscale=log2, yscale=nlg1m, title="Max-TE AST")
    scatterlines!(ax, ccdfs_moctail_astmaxthrent_rect, bin_lower_edges[i_bin_thresh:N_bin]; color=:red)

    for i_col = 1:N_ast
        ax = content(lout[1,i_col])
        if i_col < N_ast; colgap!(lout, i_col, 0); end
        if i_col > 1; ax.ylabelvisible = ax.yticklabelsvisible = false; end
        xlims!(ax, minimum(filter(c->c>0, ccdf_valid))/8, 1.0)
        #ylims!(ax, 1.1*threshold-0.1*1, 1)
    end
    ax = content(lout[2,1])
    xlims!(ax, minimum(filter(c->c>0, ccdf_valid))/8, 1.0)
    #ylims!(ax, 1.1*threshold-0.1*1, 1)

    save(joinpath(figdir, "ccdfs_moctail.png"), fig)




    # Plot entropies as functions of AST 
    fig = Figure(size=(300,150))
    lout = fig[1,1] = GridLayout()
    ax = Axis(lout[1,1]; theme_ax..., xlabel="−AST", ylabel="Thresh. Ent.")
    for i_anc = 1:N_anc
        scatterlines!(ax, reverse(-asts), reverse(thresholded_entropy[:,i_anc]), color=:gray79, alpha=0.5, marker=:circle)
        i_ast_argmax = idx_astmaxthrent[i_anc]
        scatter!(ax, -asts[i_ast_argmax], thresholded_entropy[i_ast_argmax,i_anc]; color=:gray, marker=:star6)
    end
    scatterlines!(ax, reverse(-asts), reverse(SB.mean(thresholded_entropy; dims=2))[:,1]; color=:black, label="Mean", marker=:circle)
    vlines!(ax, -SB.mean(asts[idx_astmaxthrent]); color=:black, linestyle=(:dash,:dense))
    ax.xticks = reverse(-asts)
    #ax.xticklabels = string.(reverse(-asts))
    save(joinpath(figdir, "thrent_overlay.png"), fig)

    # Plot the descendant peaks as a function of -AST; one row for each ancestor
    theme_ax = (xticklabelsize=12, yticklabelsize=12, xlabelsize=16, ylabelsize=16, xgridvisible=false, ygridvisible=false, titlefont=:regular, titlesize=16)
    fig = Figure(size=(400,60*N_anc))
    lout = fig[1,1] = GridLayout()
    for i_anc = 1:N_anc
        # Left column: maxima due to each AST 
        ax = Axis(lout[i_anc,1]; theme_ax..., xticks=(-reverse(asts), string.(-reverse(asts))))
        for i_ast = 1:N_ast
            scatter!(ax, -asts[i_ast]*ones(N_dsc), Rs_peak_dsc[:,i_ast,i_anc]; color=:red, marker=:circle, markersize=3)
        end
        hlines!(ax, bin_lower_edges[i_bin_thresh]; color=:gray79)
        hlines!(ax, Rs_peak_anc[i_anc]; color=:black, linestyle=(:dash,:dense))
        ax = Axis(lout[i_anc,2]; theme_ax..., xticks=(-reverse(asts), string.(-reverse(asts))))
        scatterlines!(ax, -reverse(asts), reverse(thresholded_entropy[:,i_anc]); color=:red)


    end
    for i_anc = 1:N_anc
        ax1,ax2 = [content(lout[i_anc,j]) for j=1:2]
        ax1.xticklabelsvisible = ax1.xlabelvisible = (i_anc==N_anc)
        ax2.xticklabelsvisible = ax1.xlabelvisible = (i_anc==N_anc)
        ax1.ylabel = "Anc. $(i_anc)"
        ax1.yticklabelsvisible = false
        ax1.ylabelrotation = 0
        ax2.ylabel = ""
        if i_anc == 1; ax1.title = "𝑅*"; ax2.title = "Thresh. Ent."; end
        if i_anc < N_anc; rowgap!(lout, i_anc, 0); end
    end
    linkyaxes!((content(lout[i_anc,2]) for i_anc=1:N_anc)...)
    colgap!(lout, 1, 15)
    # TODO make a column for entropy
    content(lout[end,1]).xlabel = "−AST"
    save(joinpath(figdir, "peaks_dsc_stacked.png"), fig)
    return thresholded_entropy 
end

function nlg1m(x::Number) 
    return -log1p(-x)/log(2) #log_2(1/(1-x))
end
function nlg1m_inv(y::Number) 
    return -expm1(-y*log(2))
end

Makie.inverse_transform(nlg1m) = nlg1m_inv
Makie.defaultlimits(::typeof(nlg1m)) = (nlg1m_inv(2.0), nlg1m_inv(8.0))
Makie.defined_interval(::typeof(nlg1m)) = Makie.OpenInterval(0.0,1.0) 

function compute_empirical_ccdf(xs::Vector{Float64}, bin_lower_edges::Vector{Float64})
    @assert all(diff(bin_lower_edges) .> 0)
    @assert length(xs) > 0
    ccdf = sum(Float64, xs .> bin_lower_edges'; dims=1)[1,:] ./ length(xs)
    return ccdf
end


function compute_thresholded_entropy(xs::Vector{Float64}, bin_lower_edges::Vector{Float64})
    pmf = compute_empirical_ccdf(xs, bin_lower_edges) #sum(Float64, xs .> bin_lower_edges'; dims=1)[1,:]
    pmf[1:end-1] .-= pmf[2:end]
    if all(pmf .== 0)
        return 0.0
    end
    pmf ./= length(xs)
    entropy = 0.0
    for (i_bin,bin_lo) in enumerate(bin_lower_edges)
        if pmf[i_bin] > 0
            entropy -= pmf[i_bin]*log2(pmf[i_bin])
        end
    end
    return entropy
end



function plot_boosts(datadir::String, figdir::String, asts::Vector{Int64}, bst::Int64, N_dsc::Int64, bin_lower_edges::Vector{Float64}, i_bin_thresh::Int64, ) # could also have decreasing intervals, as in COAST paper.
    ts_anc, xs_anc = jldopen(joinpath(datadir, "dns_ancgen.jld2"), "r") do f
        return f["ts"], f["xs"]
    end
    ts_peak, Rs_peak, cluster_starts, cluster_stops = jldopen(joinpath(datadir, "dns_peaks_ancgen.jld2"), "r") do f
        return (
                f["ts_peak"],
                f["Rs_peak"], 
                f["cluster_starts"], 
                f["cluster_stops"],
               )
    end

    theme_ax = (xticklabelsize=12, yticklabelsize=12, xlabelsize=16, ylabelsize=16, xgridvisible=false, ygridvisible=false, titlefont=:regular, titlesize=16)

    N_anc = length(Rs_peak)
    N_ast = length(asts)
    threshold = bin_lower_edges[i_bin_thresh]

    jldopen(joinpath(datadir,"xs_dscs.jld2"), "r") do f
        for i_anc = 1:N_anc
            entropy_thresholded = zeros(Float64, N_ast)
            entropy_total = zeros(Float64, N_ast)

            fig = Figure(size=(200*4,75*N_ast))
            lout = fig[1,1] = GridLayout()
            for i_ast = 1:N_ast
                ax1 = Axis(lout[i_ast,1]; ylabel="AST=$(asts[i_ast])", ylabelrotation=0, yticklabelsvisible=false, xlabel="𝑡", title="𝑅(𝑥(𝑡))", theme_ax...)
                ax2 = Axis(lout[i_ast,2]; ylabel="AST=$(asts[i_ast])", ylabelrotation=0, yticklabelsvisible=false, xlabel="𝑡", title="Peak 𝑅*", theme_ax...)
                ax3 = Axis(lout[i_ast,3]; ylabel="AST=$(asts[i_ast])", ylabelrotation=0, yticklabelsvisible=false, xlabel="δ𝑥(𝑡*-𝐴)", title="𝑅*(δ𝑥)", theme_ax...)
                # Plot the ancestor
                tidx_anc = ts_peak[i_anc]-ts_anc[1]+1 .+ (-asts[end]:bst)
                lines!(ax1, ts_anc[tidx_anc], xs_anc[1,tidx_anc]; color=:black, linewidth=2, linestyle=(:dash,:dense))
                for ax = (ax1,ax2)
                    xlims!(ax, ts_anc[tidx_anc[1]], ts_anc[tidx_anc[end]])
                    ax.xlabelvisible = ax.xticklabelsvisible = (i_ast == N_ast)
                end
                peaks_dsc = zeros(Float64, N_dsc)
                for i_dsc = 1:N_dsc
                    dscfullkey = joinpath("ianc$(i_anc)","iast$(i_ast)","idsc$(i_dsc)")
                    x_init = f[joinpath(dscfullkey,"x_init")]
                    xs_dsc = f[joinpath(dscfullkey,"xs")] 
                    t_init = f[joinpath(dscfullkey,"t_split")]
                    Nt = size(xs_dsc,2)
                    ts_dsc = t_init .+ collect(1:Nt)
                    lines!(ax1, ts_dsc, xs_dsc[1,:]; color=:red)
                    for ax = (ax1,ax2)
                        vlines!(ax, t_init; color=:red)
                        scatter!(ax, t_init, x_init[1]; color=:red, marker=:star6)
                        scatter!(ax, ts_dsc, intensity(xs_dsc); color=:red)
                    end
                    scatter!(ax3, intensity(x_init)-intensity(xs_anc[:,tidx_anc[1]]), maximum(xs_dsc[1,:]); color=:red, marker=:star5)
                    peaks_dsc[i_dsc] = maximum(xs_dsc[1,:])
                end
                entropy_thresholded[i_ast] = compute_thresholded_entropy(peaks_dsc, bin_lower_edges[i_bin_thresh:end])
                entropy_total[i_ast] = compute_thresholded_entropy(peaks_dsc, bin_lower_edges)
                hlines!(ax1, threshold; color=:gray)
                hlines!(ax2, threshold; color=:gray)
                hlines!(ax3, threshold; color=:gray)
                ylims!(ax1, 0, 1)
                ylims!(ax2, 3*threshold-2, 1)
                ylims!(ax3, extrema(peaks_dsc)...)
                for ax = (ax1,ax2)
                    xlims!(ax, ts_anc[tidx_anc[1]], ts_anc[tidx_anc[end]])
                end
                for ax = (ax2,ax3)
                    ax.ylabelvisible = false
                end
                if i_ast < N_ast
                    for ax = (ax1,ax2,ax3)
                        ax.xlabelvisible = ax.xticklabelsvisible = (i_ast == N_ast)
                    end
                end
                if i_ast > 1
                    for ax = (ax1,ax2,ax3)
                        ax.titlevisible = false
                    end
                end
            end
            ax4 = Axis(lout[:,4]; title="Thresh. Ent.", theme_ax..., ylabelvisible=false, yticklabelsvisible=false)
            scatterlines!(ax4, entropy_thresholded, -asts, color=:red)
            ax5 = Axis(lout[:,5]; title="Total Ent.", theme_ax..., ylabelvisible=false, yticklabelsvisible=false)
            scatterlines!(ax5, entropy_total, -asts; color=:steelblue)
            ylims!(ax4, -1.5*asts[end]+0.5*asts[end-1], -1.5*asts[1]+0.5*asts[2])
            for i_ast = 1:N_ast-1
                rowgap!(lout, i_ast, 0)
            end
            colgap!(lout, 1, 10)
            colgap!(lout, 2, 10)
            colgap!(lout, 3, 0)
            colgap!(lout, 4, 0)

            colsize!(lout, 4, Relative(1/8))
            colsize!(lout, 5, Relative(1/8))

            save(joinpath(figdir, "boosts_anc$(i_anc).png"), fig)
        end
    end
end

function main()
    todo = Dict{String,Bool}(
                             "run_dns_valid" =>            0,
                             "plot_dns_valid" =>           0,
                             "run_dns_ancgen" =>           0,
                             "plot_dns_ancgen" =>          0,
                             "analyze_peaks_valid" =>      0,
                             "analyze_peaks_ancgen" =>     0,
                             "boost_peaks" =>              0,
                             "plot_boosts" =>              0,
                             "analyze_boosts" =>           1,
                             "evaluate_mixing_criteria" => 0,
                             "mix_conditional_tails" =>    0,
                            )

    overwrite_boosts = false

    bpar = BoostParams()

    # Set up folders and filenames 
    exptdir = joinpath("/Users/justinfinkel/Documents/postdoc_mit/computing/COAST_results/Chaos1D","2025-09-25",strrep(bpar))
    datadir = joinpath(exptdir, "data")
    figdir = joinpath(exptdir, "figures")
    mkpath(exptdir)
    mkpath(datadir)
    mkpath(figdir)

    threshold = nlg1m_inv(bpar.threshold_neglog)
    bin_lower_edges_neglog = collect(1:1:(bpar.threshold_neglog+7)) 
    i_bin_thresh = findfirst(bin_lower_edges_neglog .== bpar.threshold_neglog)
    bin_lower_edges = nlg1m_inv.(bin_lower_edges_neglog)
    threshold = bin_lower_edges[i_bin_thresh]
    asts = collect(range(bpar.ast_min, bpar.ast_max; step=1))
    duration_plot = 3*2^bpar.threshold_neglog # long enough to capture ~3 peaks 
    perturbation_width = 1/(2^bpar.perturbation_neglog)

    if todo["run_dns_valid"]
        seed_dns_valid = 9281
        rng_dns_valid = Random.MersenneTwister(seed_dns_valid)
        x0 = Random.rand(rng_dns_valid, Float64, (1,))
        simulate(x0, bpar.duration_spinup+bpar.duration_valid, bpar.bit_precision, rng_dns_valid, datadir, "valid")
    end
    if todo["plot_dns_valid"]
        plot_dns(bpar.duration_spinup, bpar.duration_valid, datadir, figdir, "valid")
    end
    if todo["run_dns_ancgen"]
        seed_dns_ancgen = 3827
        rng_dns_ancgen = Random.MersenneTwister(seed_dns_ancgen)
        x0 = Random.rand(rng_dns_ancgen, Float64, (1,))
        simulate(x0, bpar.duration_spinup+bpar.duration_ancgen, bpar.bit_precision, rng_dns_ancgen, datadir, "ancgen")
    end
    if todo["plot_dns_valid"]
        plot_dns(bpar.duration_spinup, bpar.duration_ancgen, datadir, figdir, "ancgen")
    end
    if todo["analyze_peaks_valid"]
        find_peaks_over_threshold(threshold, bpar.duration_spinup, bpar.duration_valid, bpar.min_cluster_gap, datadir, "valid")
        plot_peaks_over_threshold(threshold, bpar.duration_spinup, duration_plot, datadir, figdir, "valid")
    end
    if todo["analyze_peaks_ancgen"]
        find_peaks_over_threshold(threshold, bpar.duration_spinup, bpar.duration_ancgen, bpar.min_cluster_gap, datadir, "ancgen")
        plot_peaks_over_threshold(threshold, bpar.duration_spinup, duration_plot, datadir, figdir, "ancgen")
    end
    if todo["boost_peaks"]
        seed_boost = 8086
        boost_peaks(threshold, perturbation_width, asts, bpar.bst, bpar.bit_precision, bpar.num_descendants, seed_boost, datadir, "ancgen"; overwrite_boosts=overwrite_boosts)
    end
    if todo["plot_boosts"]
        plot_boosts(datadir, figdir, asts, bpar.bst, bpar.num_descendants, bin_lower_edges, i_bin_thresh)
    end
    if todo["analyze_boosts"]
        analyze_boosts(datadir, figdir, asts, bpar.num_descendants, bpar.bst, bin_lower_edges, i_bin_thresh)
    end
end


main()
