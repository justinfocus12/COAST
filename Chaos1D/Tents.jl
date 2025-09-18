# Verify the optimal Advance Split Time is what I think it is for the Bernoulli map
#
import Random
import StatsBase as SB
using Printf: @sprintf
using JLD2: jldopen
using CairoMakie

struct TentMapParams
    # For tent maps, the location of the peak; for bit shift, a little unclear
    tentpeak::Float64
end

function BoostParams()
    return (
            duration_valid = 2^16,
            duration_ancgen = 2^12, 
            duration_spinup = 2^4,
            threshold_neglog = 8, # 2^(-threshold_neglog) is the threshold
            perturbation_neglog = 12,  # how many bits to keep when doing the perturbation 
            min_cluster_gap = 2^6,
            bit_precision = 32
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

function simulate(x0::Vector{Float64}, duration::Int64, bit_precision::Int64, rng::Random.AbstractRNG, datadir::String, outfile_suffix::String)
    xs = zeros(Float64, (1,duration))
    x = Random.rand(rng, Float64)
    ts = collect(1:duration)
    for t = 1:duration
        x = mod(2*(x < 0.5 ? x : 1-x), 1)
        x = mod(mod(x, 1/(2^bit_precision)) + Random.rand(rng, [0,1])/(2^bit_precision), 1)
        xs[1,t] = x
    end
    jldopen(joinpath(datadir, "dns_$(outfile_suffix).jld2"),"w") do f
        f["xs"] = xs
        f["ts"] = ts
    end

    return 
end

function plot_peaks_over_threshold(thresh::Float64, datadir::String, figdir::String, file_suffix::String)

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
    
    Npeaks2plot = 3
    ts2plot = (cluster_starts[1]-1):(cluster_stops[Npeaks2plot]+1)


    theme_ax,theme_leg = get_themes()
    fig = Figure(size=(600,150))
    lout = fig[1,1] = GridLayout()
    ax = Axis(lout[1,1]; theme_ax...)
    lines!(ax, ts2plot, xs[ts2plot]; color=:black)
    hlines!(ax, thresh; color=:gray, linewidth=1, alpha=0.25)
    for i_peak = 1:Npeaks2plot
        vlines!(ax, cluster_starts[i_peak]; color=:black, linestyle=(:dash,:dense))
        vlines!(ax, cluster_stops[i_peak]; color=:red, linestyle=(:dash,:dense))
        scatter!(ax, ts_peak[i_peak], xs_peak[i_peak]; color=:black, marker=:star5)
    end
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


function main()
    todo = Dict{String,Bool}(
                             "run_dns_valid" =>            0,
                             "plot_dns_valid" =>           0,
                             "run_dns_ancgen" =>           0,
                             "plot_dns_ancgen" =>          0,
                             "analyze_peaks_valid" =>      1,
                             "analyze_peaks_ancgen" =>     0,
                             "boost_peaks" =>              0,
                             "evaluate_mixing_criteria" => 0,
                             "mix_conditional_tails" =>    0,
                            )


    bpar = BoostParams()

    # Set up folders and filenames 
    exptdir = joinpath("/Users/justinfinkel/Documents/postdoc_mit/computing/COAST_results/Chaos1D",strrep(bpar))
    datadir = joinpath(exptdir, "data")
    figdir = joinpath(exptdir, "figures")
    mkpath(exptdir)
    mkpath(datadir)
    mkpath(figdir)

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
        thresh = 1-1/(2^bpar.threshold_neglog)
        peaks_over_threshold(thresh, bpar.duration_spinup, bpar.duration_valid, bpar.min_cluster_gap, datadir, "valid")
        plot_peaks_over_threshold(thresh, datadir, figdir, "valid")
    end
    if todo["analyze_peaks_ancgen"]
    end
end

main()
