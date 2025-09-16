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

function CoastParams()
    return (
            duration_valid = 2^16,
            duration_ancgen = 2^12, 
            duration_burnin = 2^4,
            threshold_neglog = 8, # 2^(-threshold_neglog) is the threshold
            perturbation_neglog = 12,  # how many bits to keep when doing the perturbation 
            bit_precision = 32
           )
end

function strrep(cpar::NamedTuple)
    # For naming file 
    s = @sprintf("Tv%d_Ta%d_thr%d_prt%d_bp%d", round(Int, log2(cpar.duration_valid)), round(Int, log2(cpar.duration_ancgen)), cpar.threshold_neglog, cpar.perturbation_neglog, cpar.bit_precision)
    return s
end

function get_themes()
    theme_ax = (xticklabelsize=8, yticklabelsize=8, xlabelsize=10, ylabelsize=10, xgridvisible=false, ygridvisible=false, titlefont=:bold, titlesize=10)
    theme_leg = (labelsize=8, framevisible=false)
    return theme_ax,theme_leg
end

function run_dns_valid(datadir, duration_valid, duration_burnin, bit_precision, seed)
    xs = zeros(Float64, (1,duration_valid+duration_burnin))
    rng = Random.MersenneTwister(seed)
    x = Random.rand(rng, Float64)
    ts = collect(1:duration_valid+duration_burnin)
    for t = 1:duration_burnin+duration_valid
        #x = (2*min(x, 1-x) + Random.rand(rng, [0,1])/(2^bit_precision)) % 1
        x = 4*x*(1-x)
        xs[1,t] = x
    end
    jldopen(joinpath(datadir,"dns_valid.jld2"),"w") do f
        f["xs"] = xs
        f["ts"] = ts
    end

    return 
end

function plot_dns_valid(datadir, figdir, duration_valid, duration_burnin)
    xs,ts = jldopen(joinpath(datadir,"dns_valid.jld2"),"r") do f
        return f["xs"],f["ts"]
    end
    t0 = duration_burnin
    t1 = duration_burnin + duration_valid
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

    save(joinpath(figdir, "tent_timeseries_hist.png"), fig)


end


function main()
    todo = Dict{String,Bool}(
                             "run_dns_valid" =>            1,
                             "plot_dns_valid" =>           1,
                             "run_dns_ancgen" =>           1,
                             "find_peaks" =>               1,
                             "boost_peaks" =>              1,
                             "evaluate_mixing_criteria" => 1,
                             "mix_conditional_tails" =>    1,
                            )


    cpar = CoastParams()
    exptdir = strrep(cpar)
    mkpath(exptdir)
    datadir = joinpath(exptdir, "data")
    mkpath(datadir)
    figdir = joinpath(exptdir, "figures")
    mkpath(figdir)

    seed_dns_valid = 9281


    if todo["run_dns_valid"]
        run_dns_valid(datadir, cpar.duration_valid, cpar.duration_burnin, cpar.bit_precision, seed_dns_valid)
    end
    if todo["plot_dns_valid"]
        plot_dns_valid(datadir, figdir, cpar.duration_valid, cpar.duration_burnin)
    end
end

main()
