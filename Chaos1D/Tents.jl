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
    ts_peak, xs_peak, cluster_starts, cluster_stops = jldopen(joinpath(datadir, "dns_peaks_$(file_suffix).jld2"), "r") do f
        return (
                f["ts_peak"],
                f["xs_peak"], 
                f["cluster_starts"], 
                f["cluster_stops"],
               )
    end
    
    ts2plot = duration_spinup .+ (1:duration_plot)
    peaks2plot = findall(ts2plot[1] .<= ts_peak .<= ts2plot[end])


    theme_ax,theme_leg = get_themes()
    fig = Figure(size=(400,400))
    lout = fig[1,1] = GridLayout()
    ax_xs = Axis(lout[1,1]; theme_ax..., ylabel="𝑋(𝑡)")
    ax_peaks = Axis(lout[2,1]; theme_ax..., ylabel="Peaks {𝑋(𝑡ₙ*)}")
    ax_waits = Axis(lout[3,1]; theme_ax..., ylabel="𝑡*ₙ₊₁-𝑡*ₙ")
    ax_hist_xs = Axis(lout[1,2]; theme_ax..., xlabel="ℙ{𝑋>𝑥}", yticklabelsvisible=false)
    ax_hist_peaks = Axis(lout[2,2]; theme_ax..., xlabel="ℙ{X*>𝑥*}")
    ax_hist_waits = Axis(lout[3,2]; theme_ax..., xlabel="ℙ{τ > 𝑡*ₙ₊₁-𝑡*ₙ}", xscale=log2)

    # Full timeseries
    lines!(ax_xs, ts2plot, xs[ts2plot]; color=:black)
    hlines!(ax_xs, thresh; color=:gray, linewidth=1, alpha=0.5)
    scatter!(ax_xs, ts_peak[peaks2plot], xs_peak[peaks2plot]; color=:black, marker=:star5)
    bin_edges_xs = collect(range(0, 1; length=65))
    bin_centers_xs = (bin_edges_xs[1:end-1] .+ bin_edges_xs[2:end])./2
    hist_xs = SB.normalize(SB.fit(SB.Histogram, xs[1,cluster_starts[1]:cluster_stops[end]], bin_edges_xs); mode=:pdf)
    scatterlines!(ax_hist_xs, hist_xs.weights, bin_centers_xs; color=:black, markersize=4)
    ylims!(ax_hist_xs, 0, 1)
    ylims!(ax_xs, 0, 1)
    xlims!(ax_hist_xs, 0, 1.25)
    linkyaxes!(ax_xs, ax_hist_xs)

    # Peak timeseries 
    scatter!(ax_peaks, ts_peak, xs_peak; color=:black, marker=:circle)
    peaks_sorted,ccdf_peaks = empirical_ccdf(xs_peak)
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

    for ax = (ax_hist_xs, ax_hist_peaks, ax_hist_waits)
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
        f["xs_peak"] = xs[ts_peak]
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
    while (t < tmax) && (xs[t] > thresh)
        t += 1
    end
    if t >= tmax
        return nothing
    end
    in_valley = true
    valley_length = 0
    while t < tmax
        if xs[t] <= thresh
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
    if (xs[tmax] <= thresh) && (valley_length >= min_cluster_gap)
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
        f["xs_peak"] = xs[ts_peak]
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

function boost_peaks(threshold::Float64, perturbation_width::Float64, ast_min::Int64, ast_max::Int64, bst::Int64, bit_precision::Int64, Ndsc_per_leadtime::Int64, seed::Int64, datadir::String, file_suffix::String) 
    ts, xs = jldopen(joinpath(datadir, "dns_$(file_suffix).jld2"), "r") do f
        return f["ts"], f["xs"]
    end
    ts_peak, xs_peak, cluster_starts, cluster_stops = jldopen(joinpath(datadir, "dns_peaks_$(file_suffix).jld2"), "r") do f
        return (
                f["ts_peak"],
                f["xs_peak"], 
                f["cluster_starts"], 
                f["cluster_stops"],
               )
    end
    @show ts_peak[1:4]
    Npeaks = length(ts_peak)
    pert_seq = van_der_corput(Ndsc_per_leadtime) .* perturbation_width
    datafile = joinpath(datadir,"xs_dscs.jld2")
    for i_peak = 1:Npeaks
        anckey = "anc$(i_peak)"
        for ast = ast_min:ast_max
            astkey = "ast$(ast)"
            t_split = ts_peak[i_peak] - ast
            x_init_anc = xs[:,t_split-ts[1]+1]
            xs_dsc = zeros(Float64, (1, ast+bst))
            # Depending on time and cost of simulation, maybe open the file inside the loop 
            jldopen(joinpath(datadir,"xs_dscs.jld2"), "a+") do f
                Ndsc_already_simulated = anckey in keys(f) && astkey in keys(f[joinpath(anckey)]) ? length(f[joinpath(anckey,astkey)]) : 0
                @show keys(f)
                @show Ndsc_already_simulated
                for i_dsc = Ndsc_already_simulated+1:Ndsc_per_leadtime
                    rng = Random.MersenneTwister(seed)
                    x_init_dsc = [perturbation_width*div(x_init_anc[1], perturbation_width) + pert_seq[i_dsc]]
                    xs_dsc,ts_dsc = simulate(x_init_dsc, ast+bst, bit_precision, rng)
                    @show keys(f)
                    f[joinpath(anckey,astkey,"dsc$(i_dsc)","t_split")] = t_split
                    f[joinpath(anckey,astkey,"dsc$(i_dsc)","xs")] = xs_dsc
                    f[joinpath(anckey,astkey,"dsc$(i_dsc)","x_init")] = x_init_dsc
                end
            end
        end
    end
    return
end

function analyze_boosts(datadir::String, figdir::String, N_dsc::Int64, asts::Vector{Int64}, bst::Int64, threshold::Float64)
    ts_anc, xs_anc = jldopen(joinpath(datadir, "dns_ancgen.jld2"), "r") do f
        return f["ts"], f["xs"]
    end
    ts_peak, xs_peak, cluster_starts, cluster_stops = jldopen(joinpath(datadir, "dns_peaks_ancgen.jld2"), "r") do f
        return (
                f["ts_peak"],
                f["xs_peak"], 
                f["cluster_starts"], 
                f["cluster_stops"],
               )
    end
    # Store the following data:
    # - peak (timing,value) of ancestor
    # - peak (timing,value) of descendants at every (ancestor, AST)
    # - then calculate entropy curves for each one 

    # Put results into a big array: (i_dsc, i_ast, i_anc) ∈ [N_dsc] x [N_ast] x [N_anc]...or rather a vector of vector of vectors etc. to enable active sampling 
    N_ast = length(asts)
    N_anc = length(ts_peak)

    peaks = zeros(Float64, (N_dsc, N_ast, N_anc))
    jldopen(joinpath(datadir,"xs_dscs.jld2"), "r") do f
        @show keys(f)
        for i_anc = 1:N_anc
            for i_ast = 1:N_ast
                for i_dsc = 1:N_dsc
                    peaks[i_dsc,i_ast,i_anc] = maximum(f[joinpath("anc$(i_anc)", "ast$(asts[i_ast])", "dsc$(i_dsc)","xs")])
                end
            end
        end
    end

    # Calculate thresholded entropy 
    thresholded_entropy = zeros(Float64, (N_ast, N_anc))
    max_peak = maximum(peaks)
    bin_lower_edges = 1 .- 1 ./ (2 .^ (collect(range(round(Int,-lg1p(-threshold)), round(Int,-lg1p(-max_peak))+1; step=1))))
    idx_ast_argmax = zeros(Int64, N_anc)
    for i_anc = 1:N_anc
        for i_ast = 1:N_ast
            thresholded_entropy[i_ast,i_anc] = compute_thresholded_entropy(peaks[:,i_ast,i_anc], bin_lower_edges)
        end
        idx_ast_argmax[i_anc] = argmax(thresholded_entropy[:,i_anc])
    end

    # Plot it 
    theme_ax,theme_leg = get_themes()
    fig = Figure(size=(300,300))
    lout = fig[1,1] = GridLayout()
    ax = Axis(lout[2,1]; theme_ax..., xlabel="−AST", ylabel="Thresh. Ent.")
    for i_anc = 1:N_anc
        scatterlines!(ax, reverse(-asts), reverse(thresholded_entropy[:,i_anc]), color=:gray79, alpha=0.5, marker=:circle)
        i_ast_argmax = idx_ast_argmax[i_anc]
        scatter!(ax, -asts[i_ast_argmax], thresholded_entropy[i_ast_argmax,i_anc]; color=:gray, marker=:star6)
    end
    scatterlines!(ax, reverse(-asts), reverse(SB.mean(thresholded_entropy; dims=2))[:,1]; color=:black, label="Mean", marker=:circle)
    vlines!(ax, -SB.mean(asts[idx_ast_argmax]); color=:black, linestyle=(:dash,:dense))
    ax.xticks = reverse(-asts)
    #ax.xticklabels = string.(reverse(-asts))
    save(joinpath(figdir, "thrent_overlay.png"), fig)

    # Plot the max-scores as a function of -AST; one row for each ancestor
    fig = Figure(size=(200,60*N_anc))
    lout = fig[1,1] = GridLayout()
    for i_anc = 1:N_anc
        ax = Axis(lout[i_anc,1]; theme_ax..., xticks=(-reverse(asts), string.(-reverse(asts))))
        for i_ast = 1:N_ast
            scatter!(ax, -asts[i_ast]*ones(N_dsc), peaks[:,i_ast,i_anc]; color=:red, marker=:circle, markersize=3)
        end
        hlines!(ax, threshold; color=:gray79)
        hlines!(ax, xs_peak[i_anc]; color=:black, linestyle=(:dash,:dense))
    end
    for i_row = 1:N_anc-1
        content(lout[i_row,1]).xticklabelsvisible = content(lout[i_row,1]).xlabelvisible = false
        rowgap!(lout, i_row, 0)
    end
    content(lout[end,1]).xlabel = "−AST"
    save(joinpath(figdir, "peaks_dsc_stacked.png"), fig)
    




    return thresholded_entropy 
end

function lg1p(x) # log_2(1+x)
    return log1p(x)/log(2)
end

function compute_thresholded_entropy(xs::Vector{Float64}, bin_lower_edges::Vector{Float64})
    pmf = sum(Float64, xs .> bin_lower_edges'; dims=1)[1,:]
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



function plot_boosts(datadir::String, figdir::String, ast_min::Int64, ast_max::Int64, bst::Int64, threshold::Float64, entropy_bin_width_neglog::Float64) # could also have decreasing intervals, as in COAST paper.
    ts_anc, xs_anc = jldopen(joinpath(datadir, "dns_ancgen.jld2"), "r") do f
        return f["ts"], f["xs"]
    end
    ts_peak, xs_peak, cluster_starts, cluster_stops = jldopen(joinpath(datadir, "dns_peaks_ancgen.jld2"), "r") do f
        return (
                f["ts_peak"],
                f["xs_peak"], 
                f["cluster_starts"], 
                f["cluster_stops"],
               )
    end

    theme_ax,theme_leg = get_themes()

    jldopen(joinpath(datadir,"xs_dscs.jld2"), "r") do f
        anckeys = filter(k->startswith(k,"anc"), keys(f))
        for (i_anc,anckey) in enumerate(anckeys)
            #if i_anc > 1
            #    continue
            #end
            whorlkeys = keys(f[anckey])
            # Track the entropy for each AST 
            entropy_of_ast = zeros(Float64, length(whorlkeys))

            fig = Figure(size=(400*3,50*length(whorlkeys)))
            lout = fig[1,1] = GridLayout()
            for (i_whorl,whorlkey) in enumerate(whorlkeys)
                ax1 = Axis(lout[i_whorl,1]; xlabel="𝑡", ylabel="𝑥(𝑡)", theme_ax...)
                ax2 = Axis(lout[i_whorl,2]; xlabel="𝑡", ylabel="𝑥(𝑡)", theme_ax...)
                ax3 = Axis(lout[i_whorl,3]; xlabel="δ𝑥(𝑡*-𝐴)", ylabel="𝑥*", theme_ax...)
                # Plot the ancestor
                tidx_anc = ts_peak[i_anc]-ts_anc[1]+1 .+ (-ast_max:bst)
                lines!(ax1, ts_anc[tidx_anc], xs_anc[1,tidx_anc]; color=:black, linewidth=2, linestyle=(:dash,:dense))
                for ax = (ax1,ax2)
                    xlims!(ax, ts_anc[tidx_anc[1]], ts_anc[tidx_anc[end]])
                    ax.xlabelvisible = ax.xticklabelsvisible = (i_whorl == length(whorlkeys))
                end
                @show keys(f[joinpath(anckey,whorlkey)])
                dsckeys = filter(k->startswith(k,"dsc"), keys(f[joinpath(anckey,whorlkey)]))
                peaks_dsc = zeros(Float64, length(dsckeys))
                for (i_dsc,dsckey) in enumerate(dsckeys)
                    @show keys(f[joinpath(anckey,whorlkey,dsckey)])
                    x_init = f[joinpath(anckey,whorlkey,dsckey,"x_init")]
                    xs_dsc = f[joinpath(anckey,whorlkey,dsckey,"xs")]
                    t_init = f[joinpath(anckey,whorlkey,dsckey,"t_split")]
                    @show xs_dsc[1,:]
                    @show t_init
                    Nt = size(xs_dsc,2)
                    ts_dsc = t_init .+ collect(1:Nt)
                    lines!(ax1, ts_dsc, xs_dsc[1,:]; color=:red)
                    for ax = (ax1,ax2)
                        vlines!(ax, t_init; color=:red)
                        scatter!(ax, t_init, x_init[1]; color=:red, marker=:star6)
                        scatter!(ax, ts_dsc, xs_dsc[1,:]; color=:red)
                    end
                    scatter!(ax3, x_init[1]-xs_anc[1,tidx_anc[1]], maximum(xs_dsc[1,:]); color=:red, marker=:star5)
                    peaks_dsc[i_dsc] = maximum(xs_dsc[1,:])
                end
                threshold_neglog = round(Int64,-log2(1-threshold))
                max_peak_neglog = round(Int64, -log2(1-maximum(peaks_dsc)))
                for k = threshold_neglog:(max_peak_neglog+1)
                    level = 1-1/(2^k)
                    next_level = 1-1/(2^(k+1))
                    pk = sum(level .< peaks_dsc .<= next_level)/length(peaks_dsc)
                    if pk > 0
                        entropy_of_ast[i_whorl] -= pk*log2(pk)
                    end
                end
                @show entropy_of_ast
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
                if i_whorl < length(whorlkeys)
                    for ax = (ax1,ax2,ax3)
                        ax.xlabelvisible = ax.xticklabelsvisible = (i_whorl == length(whorlkeys))
                    end
                end
            end
            ax4 = Axis(lout[:,4]; xlabel="TE", )
            scatterlines!(ax4, entropy_of_ast, reverse(1:length(whorlkeys)); color=:red)
            ylims!(ax4, 1/2, length(whorlkeys)+1/2)
            for i_whorl = 1:length(whorlkeys)-1
                rowgap!(lout, i_whorl, 0)
            end
            colgap!(lout, 1, 10)
            colgap!(lout, 2, 10)
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


    bpar = BoostParams()

    # Set up folders and filenames 
    exptdir = joinpath("/Users/justinfinkel/Documents/postdoc_mit/computing/COAST_results/Chaos1D","2025-09-24",strrep(bpar))
    datadir = joinpath(exptdir, "data")
    figdir = joinpath(exptdir, "figures")
    mkpath(exptdir)
    mkpath(datadir)
    mkpath(figdir)

    threshold = 1-1/(2^bpar.threshold_neglog)
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
        boost_peaks(threshold, perturbation_width, bpar.ast_min, bpar.ast_max, bpar.bst, bpar.bit_precision, bpar.num_descendants, seed_boost, datadir, "ancgen")
    end
    if todo["plot_boosts"]
        plot_boosts(datadir, figdir, bpar.ast_min, bpar.ast_max, bpar.bst, threshold, perturbation_width)
    end
    if todo["analyze_boosts"]
        asts = collect(range(bpar.ast_min, bpar.ast_max; step=1))
        analyze_boosts(datadir, figdir, bpar.num_descendants, asts, bpar.bst, threshold)
    end
end


main()
