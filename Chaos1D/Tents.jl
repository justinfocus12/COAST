module Tents

import Random
import StatsBase as SB
using Printf: @sprintf
using JLD2: jldopen
using CairoMakie

export pert_thresh_loop, illustrate_map

include("./MapsOneDim.jl")


struct TentMapParams
    tentpeak::Float64
end

function BoostParams()
    return (;
            duration_valid = 2^18,
            duration_ancgen = 2^16, 
            duration_spinup = 2^4,
            threshold_neglog = 5, # 2^(-threshold_neglog) is the threshold
            perturbation_neglog = 9,  # how many bits to keep when doing the perturbation 
            min_cluster_gap = 2^4, # longer than bit precision
            ast_min = 1,
            ast_max = 12,
            bst = 2, # how long to run each descendant past the ancestor's peak 
            num_descendants = 128,
            bin_width_neglog = 13,
           )
end


function strrep(bpar::NamedTuple)
    # For naming file 
    s = @sprintf("TentMap_Tv%d_Ta%d_thr%d_prt%d", round(Int, log2(bpar.duration_valid)), round(Int, log2(bpar.duration_ancgen)), bpar.threshold_neglog, bpar.perturbation_neglog,)
    return s
end


conjugate_fwd(x::Float64) = x
conjugate_bwd(z::Float64) = z
compute_cquant_peak_wholetruth(q::Float64) = 1-q 
compute_ccdf_peak_wholetruth(x::Float64) = 1-x 

function compute_pdf_wholetruth(x::Float64)
    return 1.0
end

function tentmap(x::Float64) 
    return clamp(2*(x < 0.5 ? x : 1-x), 0, 1)
end

function tentmap(Z::UInt32)
    msb = isodd(Z >> 31)
    return xor(Z<<1, msb*typemax(UInt32))
end

function perturb(X::Float32, perturbation_neglog::Integer, rng::Random.AbstractRNG)
    Z = perturb(float64_to_uint32(X), perturbation_neglog, rng)
    return uint32_to_float64(Z)
end

function perturb(Z::UInt32, perturbation_neglog::Integer, rng::Random.AbstractRNG)
    return xor(Z, rand(rng, UInt32)>>perturbation_neglog)
end

tentmap_derivative(x::Float64) = 2*sign(x - 0.5)

function illustrate_map(plotdir::String)
    X0 = 0.23
    rng = Random.MersenneTwister(238)
    F(X) = uint32_to_float64(tentmap(float64_to_uint32(X)))
    illustrate_map(X0, F, simulate, rng, "𝑇", "𝑧", "Tent map", plotdir, "tentmap.png")
    return
end

function simulate(x_init::Vector{Float64}, duration::Int64, rng::Random.AbstractRNG, init_perturbation_neglog::Integer=33)
    Zs = zeros(UInt32, duration)
    X_init = x_init[1]
    if !(0<X_init<1)
        @infiltrate
    end
    ts = collect(1:duration)
    Z_init = float64_to_uint32(X_init) 
    Z_init = perturb(Z_init, init_perturbation_neglog, rng)
    x_init_pert = [uint32_to_float64(Z_init), ]
    Z = Z_init
    for t = 1:duration
        Z = tentmap(Z) #
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
                             "illustrate_map" =>           1,
                             "run_dns_valid" =>            1,
                             "plot_dns_valid" =>           1,
                             "run_dns_ancgen" =>           1,
                             "plot_dns_ancgen" =>          1,
                             "analyze_peaks_valid" =>      1,
                             "analyze_peaks_ancgen" =>     1,
                             "boost_peaks" =>              1,
                             "mix_conditional_tails" =>    1,
                             "plot_moctails" =>            1,
                             "plot_boosts" =>              1,
                            )

    overwrite_boosts = true

    bpar_default = BoostParams()
    bpar = (; bpar_default..., bpar_adj...)

    # Set up folders and filenames 
    exptdir = joinpath("/Users/justinfinkel/Documents/postdoc_mit/computing/COAST_results/Chaos1D","2026-05-24/1",strrep(bpar))
    datadir = joinpath(exptdir, "data")
    figdir = joinpath(exptdir, "figures")
    mkpath(exptdir)
    mkpath(datadir)
    mkpath(figdir)

    N_bin_over = 2^(bpar.bin_width_neglog - bpar.threshold_neglog)
    threshold = compute_cquant_peak_wholetruth(exp2(-bpar.threshold_neglog))
    N_bin = N_bin_over * 2^bpar.threshold_neglog
    i_bin_thresh = N_bin - N_bin_over + 1
    bin_lower_edges = vcat(range(0, threshold; length=i_bin_thresh)[1:end-1], range(threshold, 1; length=N_bin_over+1)[1:end-1])
    bin_edges = vcat(bin_lower_edges, 1.0)
    bin_centers = vcat((bin_lower_edges[1:N_bin-1] .+ bin_lower_edges[2:N_bin])./2, (bin_lower_edges[N_bin]+1)/2)
    ccdf_peak_wholetruth = compute_ccdf_peak_wholetruth.(bin_lower_edges[i_bin_thresh:N_bin]) ./ compute_ccdf_peak_wholetruth(threshold)
    pdf_wholetruth = compute_pdf_wholetruth.(bin_centers)
    threshold = bin_lower_edges[i_bin_thresh]
    @assert i_bin_thresh == argmin(abs.(nlg1m.(bin_lower_edges) .- bpar.threshold_neglog))

    asts = collect(range(bpar.ast_min, bpar.ast_max; step=1))
    duration_plot = 2^bpar.threshold_neglog # long enough to capture ~1 peaks 
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
        plot_dns(bpar.duration_spinup, bpar.duration_valid, datadir, figdir, "valid"; edges=bin_edges, pdf_wholetruth=pdf_wholetruth, statesymbol="𝑧")
    end
    if todo["run_dns_ancgen"]
        seed_dns_ancgen = 6028
        rng_dns_ancgen = Random.MersenneTwister(seed_dns_ancgen)
        x0 = Random.rand(rng_dns_ancgen, Float64, (1,))
        simulate_save(x0, bpar.duration_spinup+bpar.duration_ancgen, rng_dns_ancgen, datadir, "ancgen")
    end
    if todo["plot_dns_ancgen"]
        plot_dns(bpar.duration_spinup, bpar.duration_ancgen, datadir, figdir, "ancgen"; edges=bin_edges, pdf_wholetruth=pdf_wholetruth, statesymbol="𝑧")
    end
    if todo["analyze_peaks_valid"]
        find_peaks_over_threshold(threshold, bpar.duration_spinup, bpar.duration_valid, bpar.min_cluster_gap, datadir, "valid")
        plot_peaks_over_threshold(threshold, bpar.duration_spinup, duration_plot, datadir, figdir, "valid"; bin_edges=bin_edges, i_bin_thresh=i_bin_thresh, ccdf_peak_wholetruth=ccdf_peak_wholetruth, pdf_wholetruth=pdf_wholetruth, statesymbol="𝑧")
    end
    if todo["analyze_peaks_ancgen"]
        find_peaks_over_threshold(threshold, bpar.duration_spinup, bpar.duration_ancgen, bpar.min_cluster_gap, datadir, "ancgen")
        plot_peaks_over_threshold(threshold, bpar.duration_spinup, duration_plot, datadir, figdir, "ancgen"; bin_edges=bin_edges, i_bin_thresh=i_bin_thresh, ccdf_peak_wholetruth=ccdf_peak_wholetruth, pdf_wholetruth=pdf_wholetruth, statesymbol="𝑧")
    end
    if todo["boost_peaks"]
        seed_boost = 8086
        boost_peaks(simulate, conjugate_fwd, conjugate_bwd, threshold, bpar.perturbation_neglog, asts, bpar.bst, bpar.num_descendants, seed_boost, datadir, "ancgen"; overwrite_boosts=overwrite_boosts)
    end
    if todo["mix_conditional_tails"]
        rngseed_boot = 3900
        mix_conditional_tails(datadir, asts, bpar.num_descendants, bpar.bst, bin_lower_edges, i_bin_thresh, rngseed_boot; ccdf_peak_wholetruth=ccdf_peak_wholetruth)
    end
    if todo["plot_boosts"]
        plot_boosts(datadir, figdir, asts, bpar.bst, bpar.num_descendants, bin_lower_edges, i_bin_thresh, bpar.perturbation_neglog; statesymbol="𝑧")
    end
    if todo["plot_moctails"]
        plot_moctails(datadir, figdir, asts, bpar.num_descendants, bpar.bst, bin_lower_edges, i_bin_thresh, bpar.perturbation_neglog, bpar.threshold_neglog, tentmap_derivative; ccdf_peak_wholetruth=ccdf_peak_wholetruth,statesymbol="𝑧")
    end
end

function thresh_pert_loop()
    for perturbation_neglog = [14, 16, 18]
        for threshold_neglog = [8, 10, 12]
            bpar_adj = (; threshold_neglog, perturbation_neglog)
            main(bpar_adj)
        end
    end
end

end # module Tents
