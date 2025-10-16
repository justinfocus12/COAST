import Random
import StatsBase as SB
using Printf: @sprintf
using JLD2: jldopen
using CairoMakie

include("./MapsOneDim.jl")

struct LogisticMapParams
    carrying_capacity::Float64
end

function BoostParams()
    return (
            duration_valid = 2^18,
            duration_ancgen = 2^12, 
            duration_spinup = 2^4,
            threshold_neglog = 5, # 2^(-threshold_neglog) is the threshold
            perturbation_neglog = 12,  # how many bits to keep when doing the perturbation 
            min_cluster_gap = 2^6,
            bit_precision = 32,
            ast_min = 1,
            ast_max = 12,
            bst = 2,
            num_descendants = 31,
            latentize = false,# Do we transform to Z space? 
            latentize_bins = false,
           )
end

function strrep(bpar::NamedTuple)
    # For naming folder with experiments 
    s = @sprintf("LogisticMap_Lat%d_Latbins%d_Tv%d_Ta%d_thr%d_prt%d_bp%d", bpar.latentize, bpar.latentize_bins, round(Int, log2(bpar.duration_valid)), round(Int, log2(bpar.duration_ancgen)), bpar.threshold_neglog, bpar.perturbation_neglog, bpar.bit_precision)
    return s
end

function get_themes()
    theme_ax = (xticklabelsize=8, yticklabelsize=8, xlabelsize=10, ylabelsize=10, xgridvisible=false, ygridvisible=false, titlefont=:bold, titlesize=10)
    theme_leg = (labelsize=8, framevisible=false)
    return theme_ax,theme_leg
end

conjugate_fwd(x::Float64) = (2/pi) * asin(sqrt(x))
conjugate_bwd(z::Float64) = sin(pi/2*z)^2
compute_cquant_peak_wholetruth(q::Float64) = conjugate_bwd(1-q)
compute_ccdf_peak_wholetruth(x::Float64) = 1-conjugate_fwd(x)

function compute_pdf_wholetruth(x::Float64)
    return 1/(pi*sqrt(x*(1-x)))
end

function simulate(x_init::Vector{Float64}, duration::Int64, bit_precision::Int64, rng::Random.AbstractRNG)
    xs = zeros(Float64, (1,duration))
    x = x_init[1]
    ts = collect(1:duration)
    for t = 1:duration
        x = mod(4*x*(1-x), 1)
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

function main()
    todo = Dict{String,Bool}(
                             "run_dns_valid" =>            1,
                             "plot_dns_valid" =>           1,
                             "run_dns_ancgen" =>           1,
                             "plot_dns_ancgen" =>          1,
                             "analyze_peaks_valid" =>      1,
                             "analyze_peaks_ancgen" =>     1,
                             "boost_peaks" =>              1,
                             "plot_boosts" =>              1,
                             "mix_conditional_tails" =>    1,
                             "plot_moctails" =>            1,
                            )

    overwrite_boosts = true

    bpar = BoostParams()

    # Set up folders and filenames 
    exptdir = joinpath("/Users/justinfinkel/Documents/postdoc_mit/computing/COAST_results/Chaos1D","2025-10-16",strrep(bpar))
    datadir = joinpath(exptdir, "data")
    figdir = joinpath(exptdir, "figures")
    mkpath(exptdir)
    mkpath(datadir)
    mkpath(figdir)

    N_bin_over = 24
    threshold = compute_cquant_peak_wholetruth(1/2^bpar.threshold_neglog)
    N_bin = N_bin_over * 2^bpar.threshold_neglog
    i_bin_thresh = N_bin - N_bin_over + 1
    if bpar.latentize_bins
        bin_lower_edges = vcat(range(0, threshold; length=i_bin_thresh)[1:end-1], compute_cquant_peak_wholetruth.(range(1/2^bpar.threshold_neglog, 0.0; length=N_bin_over+1)[1:end-1]))
    else
        bin_lower_edges = vcat(range(0, threshold; length=i_bin_thresh)[1:end-1], range(threshold, 1; length=N_bin_over+1)[1:end-1])
    end
    bin_edges = vcat(bin_lower_edges, 1.0)
    bin_centers = vcat((bin_lower_edges[1:N_bin-1] .+ bin_lower_edges[2:N_bin])./2, (bin_lower_edges[N_bin]+1.0)/2)
    ccdf_peak_wholetruth = compute_ccdf_peak_wholetruth.(bin_lower_edges[i_bin_thresh:N_bin]) ./ compute_ccdf_peak_wholetruth(threshold)
    pdf_wholetruth = compute_pdf_wholetruth.(bin_centers)
    threshold = bin_lower_edges[i_bin_thresh]
    #@assert i_bin_thresh == argmin(abs.(nlg1m.(bin_lower_edges) .- bpar.threshold_neglog))

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
        plot_dns(bpar.duration_spinup, bpar.duration_valid, datadir, figdir, "valid"; edges=vcat(bin_lower_edges,1.0), pdf_wholetruth=pdf_wholetruth)
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
        plot_peaks_over_threshold(threshold, bpar.duration_spinup, duration_plot, datadir, figdir, "valid"; bin_edges=bin_edges, i_bin_thresh=i_bin_thresh, ccdf_peak_wholetruth=ccdf_peak_wholetruth, pdf_wholetruth=pdf_wholetruth)
    end
    if todo["analyze_peaks_ancgen"]
        find_peaks_over_threshold(threshold, bpar.duration_spinup, bpar.duration_ancgen, bpar.min_cluster_gap, datadir, "ancgen")
        plot_peaks_over_threshold(threshold, bpar.duration_spinup, duration_plot, datadir, figdir, "ancgen"; bin_edges=bin_edges, i_bin_thresh=i_bin_thresh, ccdf_peak_wholetruth=ccdf_peak_wholetruth, pdf_wholetruth=pdf_wholetruth)
    end
    if todo["boost_peaks"]
        seed_boost = 8086
        boost_peaks(simulate, bpar.latentize, conjugate_fwd, conjugate_bwd, threshold, bpar.perturbation_neglog, asts, bpar.bst, bpar.bit_precision, bpar.num_descendants, seed_boost, datadir, "ancgen"; overwrite_boosts=overwrite_boosts)
    end
    if todo["plot_boosts"]
        plot_boosts(datadir, figdir, asts, bpar.bst, bpar.num_descendants, bin_lower_edges, i_bin_thresh, bpar.perturbation_neglog)
    end
    if todo["mix_conditional_tails"]
        mix_conditional_tails(datadir, asts, bpar.num_descendants, bpar.bst, bin_lower_edges, i_bin_thresh, ; ccdf_peak_wholetruth=ccdf_peak_wholetruth)
    end
    if todo["plot_moctails"]
        plot_moctails(datadir, figdir, asts, bpar.num_descendants, bpar.bst, bin_lower_edges, i_bin_thresh, bpar.perturbation_neglog, bpar.threshold_neglog; ccdf_peak_wholetruth=ccdf_peak_wholetruth)
    end
end

main()
