module Logistics

import Random
import StatsBase as SB
using Printf: @sprintf
using JLD2: jldopen
using CairoMakie

export pert_thresh_loop, illustrate_map, logisticmap

include("./MapsOneDim.jl")

struct LogisticMapParams
    carrying_capacity::Float64
end

function BoostParams()
    return (
            duration_valid = 2^18,
            duration_ancgen = 2^16,  
            duration_spinup = 2^4,
            threshold_neglog = 5, # 2^(-threshold_neglog) is the threshold
            perturbation_neglog = 9,  # how many bits to keep when doing the perturbation 
            min_cluster_gap = 2^4, # longer than maximum possible AST 
            ast_min = 1,
            ast_max = 12,
            bst = 2,
            num_descendants = 16,
            latentize_bins = true,
            bin_width_neglog = 13,
           )
end

function strrep(bpar::NamedTuple)
    # For naming folder with experiments 
    s = @sprintf("LogisticMap_Latbins%d_Tv%d_Ta%d_thr%d_prt%d_bw%d", bpar.latentize_bins, round(Int, log2(bpar.duration_valid)), round(Int, log2(bpar.duration_ancgen)), bpar.threshold_neglog, bpar.perturbation_neglog, bpar.bin_width_neglog)
    return s
end

conjugate_fwd(x::Float64) = (2/pi) * asin(sqrt(x))
conjugate_bwd(z::Float64) = sin(pi/2*z)^2
compute_cquant_peak_wholetruth(q::Float64) = conjugate_bwd(1-q)
compute_ccdf_peak_wholetruth(x::Float64) = 1-conjugate_fwd(x)

function compute_pdf_wholetruth(x::Float64)
    return 1/(pi*sqrt(x*(1-x)))
end

function logisticmap(x::Float64)
    return clamp(4*x*(1-x), 0, 1)
end

function logisticmap(Z::UInt32)
    W = UInt64(Z) 
    omW = UInt64(xor(W,typemax(UInt32)))
    V = W*omW
    return UInt32(V >> 30)
end


function perturb(X::Float32, perturbation_neglog::Integer, rng::Random.AbstractRNG)
    Z = perturb(float64_to_uint32(X), perturbation_neglog, rng)
    return uint32_to_float64(Z)
end

function perturb(Z::UInt32, perturbation_neglog::Integer, rng::Random.AbstractRNG)
    return xor(Z, rand(rng, UInt32)>>perturbation_neglog)
end

function logisticmap_derivative(x::Float64) 
    return 4*(1 - 2*x)
end

function illustrate_map(plotdir::String)
    Z0 = 0.23
    X0 = conjugate_bwd(Z0)
    rng = Random.MersenneTwister(238)
    F(X) = uint32_to_float64(logisticmap(float64_to_uint32(X)))
    illustrate_map(X0, F, simulate, rng, "𝐿", "𝑥", "Logistic map", plotdir, "logisticmap.png")
    return
end

function simulate(x_init::Vector{Float64}, duration::Int64, rng::Random.AbstractRNG, init_perturbation_neglog::Integer=33)
    Zs = zeros(UInt32, duration)
    X_init = x_init[1]
    if !(0<X_init<1)
        error()
    end
    ts = collect(1:duration)
    Z_init = float64_to_uint32(X_init) 
    Z_init = perturb(Z_init, init_perturbation_neglog, rng)
    x_init_pert = [uint32_to_float64(Z_init), ]
    Z = Z_init
    for t = 1:duration
        Z = logisticmap(Z) #
        Z = xor(Z, UInt32(rand(rng,Bool)))
        Zs[t] = Z
    end
    xs = zeros(Float64,(1,duration))
    xs[1,:] .= uint32_to_float64.(Zs)
    return x_init_pert, xs, ts
end


function simulate_save(x0::Vector{Float64}, duration::Int64, rng::Random.AbstractRNG, datadir::String, outfile_suffix::String)
    _, xs, ts = simulate(x0, duration, rng)
    jldopen(joinpath(datadir, "dns_$(outfile_suffix).jld2"),"w") do f
        f["xs"] = xs
        f["ts"] = ts
    end
    return 
end

function main(bpar_adj)
    todo = Dict{String,Bool}(
                             "illustrate_map" =>           0,
                             "run_dns_valid" =>            0,
                             "plot_dns_valid" =>           0,
                             "run_dns_ancgen" =>           0,
                             "plot_dns_ancgen" =>          0,
                             "analyze_peaks_valid" =>      0,
                             "analyze_peaks_ancgen" =>     0,
                             "boost_peaks" =>              0,
                             "mix_conditional_tails" =>    0,
                             "plot_moctails" =>            1,
                             "plot_boosts" =>              0,
                            )

    overwrite_boosts = true

    bpar_default = BoostParams()
    bpar = (; bpar_default..., bpar_adj...)
    
    # Set up folders and filenames 
    exptdir = joinpath("/Users/justinfinkel/Documents/postdoc_mit/computing/COAST_results/Chaos1D","2026-05-25/1",strrep(bpar))
    datadir = joinpath(exptdir, "data")
    figdir = joinpath(exptdir, "figures")
    mkpath(exptdir)
    mkpath(datadir)
    mkpath(figdir)

    N_bin_over = 2^(bpar.bin_width_neglog - bpar.threshold_neglog)
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
    duration_plot = 2^bpar.threshold_neglog # long enough to capture ~3 peaks 
    perturbation_width = 1/(2^bpar.perturbation_neglog)

    if todo["illustrate_map"]
        illustrate_map(figdir)
    end

    if todo["run_dns_valid"]
        seed_dns_valid = 9281
        rng_dns_valid = Random.MersenneTwister(seed_dns_valid)
        x0 = Random.rand(rng_dns_valid, Float64, (1,))
        simulate_save(x0, bpar.duration_spinup+bpar.duration_valid, rng_dns_valid, datadir, "valid")
    end

    if todo["plot_dns_valid"]
        plot_dns(bpar.duration_spinup, bpar.duration_valid, datadir, figdir, "valid"; edges=vcat(bin_lower_edges,1.0), pdf_wholetruth=pdf_wholetruth, statesymbol="𝑥")
    end

    if todo["run_dns_ancgen"]
        seed_dns_ancgen = 3827
        rng_dns_ancgen = Random.MersenneTwister(seed_dns_ancgen)
        x0 = Random.rand(rng_dns_ancgen, Float64, (1,))
        simulate_save(x0, bpar.duration_spinup+bpar.duration_ancgen, rng_dns_ancgen, datadir, "ancgen")
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
        boost_peaks(simulate, threshold, bpar.perturbation_neglog, asts, bpar.bst, bpar.num_descendants, seed_boost, datadir, "ancgen"; overwrite_boosts=overwrite_boosts)
    end

    if todo["mix_conditional_tails"]
        rngseed_boot = 3900
        mix_conditional_tails(datadir, asts, bpar.num_descendants, bpar.bst, bin_lower_edges, i_bin_thresh, rngseed_boot; ccdf_peak_wholetruth=ccdf_peak_wholetruth)
    end

    if todo["plot_boosts"]
        plot_boosts(datadir, figdir, asts, bpar.bst, bpar.num_descendants, bin_lower_edges, i_bin_thresh, bpar.perturbation_neglog, statesymbol="𝑥")
    end

    if todo["plot_moctails"]
        plot_moctails(datadir, figdir, asts, bpar.num_descendants, bpar.bst, bin_lower_edges, i_bin_thresh, bpar.perturbation_neglog, bpar.threshold_neglog, logisticmap_derivative; ccdf_peak_wholetruth=ccdf_peak_wholetruth)
    end

end

function thresh_pert_loop()
    for perturbation_neglog = [14, 17]
        for threshold_neglog = [8, 9, 10]
            bpar_adj = (; threshold_neglog, perturbation_neglog)
            main(bpar_adj)
        end
    end
end

end # module Logistics


