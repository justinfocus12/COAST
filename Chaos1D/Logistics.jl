# Verify the optimal Advance Split Time is what I think it is for the Bernoulli map
#
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
            ast_max = 15,
            bst = 2,
            num_descendants = 31,
            # Do we transform to Z space? 
            latentize = false,
           )
end

function strrep(bpar::NamedTuple)
    # For naming folder with experiments 
    s = @sprintf("LogisticMap_Lat%d_Tv%d_Ta%d_thr%d_prt%d_bp%d", bpar.latentize, round(Int, log2(bpar.duration_valid)), round(Int, log2(bpar.duration_ancgen)), bpar.threshold_neglog, bpar.perturbation_neglog, bpar.bit_precision)
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

function boost_peaks(threshold::Float64, perturbation_neglog::Int64, asts::Vector{Int64}, bst::Int64, bit_precision::Int64, Ndsc_per_leadtime::Int64, seed::Int64, datadir::String, file_suffix::String; latentize::Bool=false, overwrite_boosts::Bool=false) 
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
    pert_seq = van_der_corput(Ndsc_per_leadtime) #.* perturbation_width
    boostfile = joinpath(datadir,"xs_dscs.jld2")
    if isfile(boostfile) && overwrite_boosts
        rm(boostfile)
    end


    iomode = (overwrite_boosts ? "w" : "a+")
    for i_peak = 1:Npeaks
        anckey = "ianc$(i_peak)"
        for (i_ast,ast) in enumerate(asts)
            astkey = "iast$(i_ast)"
            t_split = ts_peak[i_peak] - ast
            x_init_anc = xs_anc[:,t_split-ts_anc[1]+1]
            xs_dsc = zeros(Float64, (1, ast+bst))
            # Depending on time and cost of simulation, maybe open the file inside the loop 
            jldopen(boostfile, "a+") do f
                Ndsc_already_simulated = anckey in keys(f) && astkey in keys(f[joinpath(anckey)]) ? length(f[joinpath(anckey,astkey)]) : 0
                for i_dsc = Ndsc_already_simulated+1:Ndsc_per_leadtime
                    rng = Random.MersenneTwister(seed)
                    z_init_anc = (latentize ? conjugate_fwd : identity)(x_init_anc[1])
                    z_init_dsc = mod(
                                     (
                                      floor(Int, z_init_anc*2^perturbation_neglog)
                                      + pert_seq[i_dsc]
                                     ) / (2^perturbation_neglog), 
                                     1
                                    )
                    x_init_dsc = [(latentize ? conjugate_bwd : identity)(z_init_dsc)]
                    #x_init_dsc = [perturbation_width*div(x_init_anc[1], perturbation_width) + pert_seq[i_dsc]]
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

function plot_moctails(datadir::String, figdir::String, asts::Vector{Int64}, N_dsc::Int64, bst::Int64, bin_lower_edges::Vector{Float64}, i_bin_thresh::Int64, perturbation_neglog::Int64, threshold_neglog::Int64; ccdf_peak_wholetruth::ornot(Vector{Float64})=nothing)

    # ----------------------------------------------------
    # Plotting 
    #
    (
     ccdf_peak_anc,
     ccdf_peak_valid,
     Rs_peak_dsc,
     idx_astmaxthrent,
     ccdfs_dsc,
     ccdfs_dsc_rect,
     ccdfs_moctail_astunif,
     ccdfs_moctail_astunif_rect,
     ccdf_moctail_astmaxthrent,
     ccdf_moctail_astmaxthrent_rect,
     losses_astunif_hell,
     losses_astunif_chi2,
     losses_astunif_wass,
     loss_astmaxthrent_hell,
     loss_astmaxthrent_chi2,
     loss_astmaxthrent_wass,
     thresholded_entropy,
    ) = (
         jldopen(joinpath(datadir,"boost_stats.jld2"), "r") do f
             return (
                     f["ccdf_peak_anc"], 
                     f["ccdf_peak_valid"], 
                     f["Rs_peak_dsc"], 
                     f["idx_astmaxthrent"], 
                     f["ccdfs_dsc"], 
                     f["ccdfs_dsc_rect"], 
                     f["ccdfs_moctail_astunif"], 
                     f["ccdfs_moctail_astunif_rect"], 
                     f["ccdf_moctail_astmaxthrent"], 
                     f["ccdf_moctail_astmaxthrent_rect"], 
                     f["losses_astunif_hell"], 
                     f["losses_astunif_chi2"], 
                     f["losses_astunif_wass"], 
                     f["loss_astmaxthrent_hell"], 
                     f["loss_astmaxthrent_chi2"], 
                     f["loss_astmaxthrent_wass"],
                     f["thresholded_entropy"],
                    )
        end
       )
    ts_anc, xs_anc = jldopen(joinpath(datadir, "dns_ancgen.jld2"), "r") do f
        return f["ts"], f["xs"]
    end
    ts_peak_valid, Rs_peak_valid, cluster_starts_valid, cluster_stops_valid = jldopen(joinpath(datadir, "dns_peaks_valid.jld2"), "r") do f
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

    N_ast = length(asts)
    N_bin = length(bin_lower_edges)
    N_anc = length(Rs_peak_anc)
    bin_centers = vcat((bin_lower_edges[1:N_bin-1] .+ bin_lower_edges[2:N_bin])./2, (bin_lower_edges[N_bin]+1)/2)
    bin_edges = vcat(bin_lower_edges, 1.0)


    fig = Figure(size=(80*N_ast, 850))
    theme_ax = (xticklabelsize=12, yticklabelsize=12, xlabelsize=16, ylabelsize=16, xgridvisible=false, ygridvisible=false, titlefont=:regular, titlesize=16)
    lout = fig[1,1] = GridLayout()


    # ---------- Row 1: CCDFs at each AST separately ----------
    for i_ast = 1:N_ast
        ax = Axis(lout[1,N_ast-i_ast+1]; theme_ax..., xscale=log10, yscale=identity, ylabel="Tail PDFs,\nUniform AST", ylabelrotation=0)
        lines!(ax, ccdf2pdf(ccdf_peak_anc, bin_edges[i_bin_thresh:end]), bin_centers[i_bin_thresh:N_bin]; color=:gray79, linestyle=:solid, linewidth=3, label="Ancestors only")
        lines!(ax, ccdf2pdf(ccdf_peak_valid, bin_edges[i_bin_thresh:end]), bin_centers[i_bin_thresh:N_bin]; color=:black, linestyle=:solid, label="Long DNS", linewidth=2)
        if !isnothing(ccdf_peak_wholetruth)
            lines!(ax, ccdf2pdf(ccdf_peak_wholetruth, bin_edges[i_bin_thresh:end]), bin_centers[i_bin_thresh:N_bin]; color=:black, linestyle=(:dash,:dense), label="Whole truth", linewidth=2.5)
        end
        lines!(ax, ccdf2pdf(ccdfs_moctail_astunif_rect[:,i_ast], bin_edges[i_bin_thresh:end]), bin_centers[i_bin_thresh:N_bin]; color=:red, linewidth=1.5)
    end

    # ----------- Rows 2-3: thrent and COAST frequency ------------
    ax = Axis(lout[2,1:N_ast]; xlabel="−AST", ylabel="Thresholded\nEntropy.", ylabelrotation=0, xgridvisible=false, ygridvisible=false, xticks=(-asts, string.(-asts)), xlabelvisible=false, xticklabelsvisible=false)
    xlims!(ax, -(1.5*asts[end]-0.5*asts[end-1]), -(1.5*asts[1]-0.5*asts[2]))
    for i_anc = 1:N_anc
        scatterlines!(ax, -asts, thresholded_entropy[:,i_anc]; color=:gray79)
    end
    scatterlines!(ax, -asts, SB.mean(thresholded_entropy; dims=2)[:,1]; color=:red)
    ax = Axis(lout[3,1:N_ast]; xlabel="−AST", ylabel="COAST\nfrequency", ylabelrotation=0, xgridvisible=false, ygridvisible=false, xticks=(-asts, string.(-asts)), xlabelvisible=true, xticklabelsvisible=true)
    xlims!(ax, -(1.5*asts[end]-0.5*asts[end-1]), -(1.5*asts[1]-0.5*asts[2]))
    for i_ast = 1:N_ast
        scatterlines!(ax, -asts[i_ast].*ones(2), [0, SB.mean(idx_astmaxthrent.==i_ast)]; color=:black, linewidth=8)
    end
    scatter!(ax, -threshold_neglog, 0.5; marker=:star6, color=:cyan, markersize=18)
    scatter!(ax, -perturbation_neglog, 0.5; marker=:star6, color=:orange, markersize=18)
    scatter!(ax, -(perturbation_neglog-threshold_neglog), 0.5; marker=:star6, color=:red, markersize=18)
    ylims!(ax, 0, 1)

    # --------- Row 4: the Thrent-based mixture --------------
    i_astmaxthrent_mean = round(Int, SB.mode(idx_astmaxthrent)) # Put it horizontally at the mean COAST position 
    @show i_astmaxthrent_mean
    @show idx_astmaxthrent
    ax = Axis(lout[4,N_ast-i_astmaxthrent_mean+1]; theme_ax..., xscale=log10, yscale=identity, ylabel="AST = argmax(thresh. ent.)", ylabelrotation=0)
    lines!(ax, ccdf2pdf(ccdf_peak_anc, bin_edges[i_bin_thresh:end]), bin_centers[i_bin_thresh:N_bin]; color=:gray79, linestyle=:solid, linewidth=3, label="Ancestors only")
    lines!(ax, ccdf2pdf(ccdf_peak_valid, bin_edges[i_bin_thresh:end]), bin_centers[i_bin_thresh:N_bin]; color=:black, linestyle=:solid, label="Long DNS", linewidth=2)
    if !isnothing(ccdf_peak_wholetruth)
        lines!(ax, ccdf2pdf(ccdf_peak_wholetruth, bin_edges[i_bin_thresh:end]), bin_centers[i_bin_thresh:N_bin]; color=:black, linestyle=(:dash,:dense), label="Whole truth", linewidth=2.5)
    end
    lines!(ax, ccdf2pdf(ccdf_moctail_astmaxthrent_rect, bin_edges[i_bin_thresh:end]), bin_centers[i_bin_thresh:N_bin]; color=:red, linewidth=1.5)
    if i_astmaxthrent_mean < N_ast; ax.ylabelvisible = ax.yticklabelsvisible = false; end

    for i_col = 1:N_ast
        for i_row = [1,4]
            if length(contents(lout[i_row,i_col])) == 0; continue; end
            ax = content(lout[i_row,i_col])
            if i_col < N_ast; colgap!(lout, i_col, 0); end
            if i_col > 1; ax.ylabelvisible = ax.yticklabelsvisible = false; end
            ax.xlabelvisible = ax.xticklabelsvisible = false
            #xlims!(ax, minimum(filter(c->c>0, ccdf_peak_valid))/16, 1.0)
            #ylims!(ax, 1.1*threshold-0.1*1, 1)
        end
    end

    # ------------ Rows 5-7:  various divergences ------
    for (i_row,losses_astunif,loss_astmaxthrent,divname) = zip(
                                                             5:7,
                                                             (losses_astunif_hell,losses_astunif_chi2,losses_astunif_wass),
                                                             (loss_astmaxthrent_hell,loss_astmaxthrent_chi2,loss_astmaxthrent_wass),
                                                             ("Hellinger\nDistance","χ² Divergence","𝐿¹ Distance")
                                                            )
        ax = Axis(lout[i_row,1:N_ast]; xlabel="−AST", ylabel=divname, ylabelrotation=0, yscale=log10, xgridvisible=false, ygridvisible=false, xticks=(-asts, string.(-asts)))
        xlims!(ax, -(1.5*asts[end]-0.5*asts[end-1]), -(1.5*asts[1]-0.5*asts[2]))
        scatterlines!(ax, -asts, losses_astunif; color=:black)
        hlines!(ax, loss_astmaxthrent; color=:black, linestyle=(:dash,:dense))
    end

    for row = [1,4] # make the PMF rows bigger
        rowsize!(lout, row, Relative(3/11))
    end
    for row = [2,3,5,6] # remove gaps between lineplot rows
        ax = content(lout[row,:])
        ax.xlabelvisible = ax.xticklabelsvisible = false
        rowgap!(lout, row, 0)
    end
    
    save(joinpath(figdir, "ccdfs_moctail.png"), fig)




    # Plot entropies as functions of AST 
    fig = Figure(size=(300,150))
    lout = fig[1,1] = GridLayout()
    ax = Axis(lout[1,1]; theme_ax..., xlabel="−AST", ylabel="Thresh. Ent.")
    for i_anc = 1:N_anc
        scatterlines!(ax, reverse(-asts), reverse(thresholded_entropy[:,i_anc]), color=:gray79, marker=:circle)
        i_ast_argmax = idx_astmaxthrent[i_anc]
        scatter!(ax, -asts[i_ast_argmax], thresholded_entropy[i_ast_argmax,i_anc]; color=:gray, marker=:star6)
    end
    scatterlines!(ax, reverse(-asts), reverse(SB.mean(thresholded_entropy; dims=2))[:,1]; color=:black, label="Mean", marker=:circle)
    vlines!(ax, -SB.mean(asts[idx_astmaxthrent]); color=:black, linestyle=(:dash,:dense))
    ax.xticks = reverse(-asts)
    #ax.xticklabels = string.(reverse(-asts))
    save(joinpath(figdir, "thrent_overlay.png"), fig)

    # Plot the descendant peaks as a function of -AST; one row for each ancestor
    theme_ax = (xticklabelsize=16, yticklabelsize=16, xlabelsize=20, ylabelsize=20, xgridvisible=false, ygridvisible=false, titlefont=:regular, titlesize=20)
    fig = Figure(size=(500,60*N_anc))
    lout = fig[1,1] = GridLayout()
    xtickvals = reverse(-round.(Int, range(asts[1], asts[end]; length=3)))
    xticks = (xtickvals, string.(xtickvals))
    for i_anc = 1:N_anc
        # Left column: maxima due to each AST 
        ax = Axis(lout[i_anc,1]; theme_ax..., xticks=xticks, xticklabelrotation=-pi/2)
        for i_ast = 1:N_ast
            scatter!(ax, -asts[i_ast]*ones(N_dsc), Rs_peak_dsc[:,i_ast,i_anc]; color=:red, marker=:circle)
        end
        hlines!(ax, bin_lower_edges[i_bin_thresh]; color=:gray79)
        hlines!(ax, Rs_peak_anc[i_anc]; color=:black, linestyle=(:dash,:dense))
        ax = Axis(lout[i_anc,2]; theme_ax..., xticks=xticks, xticklabelrotation=-pi/2, yticklabelsvisible=false)
        scatterlines!(ax, -reverse(asts), reverse(thresholded_entropy[:,i_anc]); color=:red)


    end
    for i_anc = 1:N_anc
        ax1,ax2 = [content(lout[i_anc,j]) for j=1:2]

        ax1.xticklabelsvisible = ax1.xlabelvisible = (i_anc==N_anc)
        ax2.xticklabelsvisible = ax1.xlabelvisible = (i_anc==N_anc)

        ax1.ylabel = "$(i_anc==1 ? "Ancestor " : "") $(i_anc)"
        ax1.yticklabelsvisible = false
        ax1.ylabelrotation = 0

        ax2.ylabel = ""

        if i_anc == 1; ax1.title = "𝑅*"; ax2.title = "Thresh. Ent."; end
        if i_anc < N_anc; rowgap!(lout, i_anc, 0); end

        ylims!(ax1, 2*bin_lower_edges[i_bin_thresh]-1, 1.0)
    end
    linkyaxes!((content(lout[i_anc,2]) for i_anc=1:N_anc)...)
    colgap!(lout, 1, 15)
    # TODO make a column for entropy
    content(lout[end,1]).xlabel = "−AST"
    save(joinpath(figdir, "peaks_dsc_stacked.png"), fig)
    return 
end

function analyze_boosts(datadir::String, figdir::String, asts::Vector{Int64}, N_dsc::Int64, bst::Int64, bin_lower_edges::Vector{Float64}, i_bin_thresh::Int64, perturbation_neglog::Int64, threshold_neglog::Int64)
    # ----------------------------------------------------
    # Plotting 
    #

    fig = Figure(size=(80*N_ast, 850))
    theme_ax = (xticklabelsize=12, yticklabelsize=12, xlabelsize=16, ylabelsize=16, xgridvisible=false, ygridvisible=false, titlefont=:regular, titlesize=16)
    lout = fig[1,1] = GridLayout()


    # ---------- Row 1: CCDFs at each AST separately ----------
    for i_ast = 1:N_ast
        xlimits = collect(extrema(ccdf2pmf(ccdf_peak_anc)))
        ax = Axis(lout[1,N_ast-i_ast+1]; theme_ax..., xscale=identity, yscale=identity, ylabel="Tail PDFs,\nUniform AST", ylabelrotation=0)
        lines!(ax, ccdf2pmf(ccdf_peak_anc), bin_centers[i_bin_thresh:N_bin]; color=:gray79, linestyle=:solid, linewidth=3, label="Ancestors only")
        lines!(ax, ccdf2pmf(ccdf_peak_valid), bin_centers[i_bin_thresh:N_bin]; color=:black, linestyle=:solid, label="Long DNS", linewidth=2)
        lines!(ax, ccdf2pmf(ccdf_peak_wholetruth), bin_centers[i_bin_thresh:N_bin]; color=:black, linestyle=(:dash,:dense), label="Whole truth", linewidth=2)
        lines!(ax, ccdf2pmf(ccdfs_moctail_astunif_rect[:,i_ast]), bin_centers[i_bin_thresh:N_bin]; color=:red, linewidth=1)
        xlims!(ax, xlimits...)
    end

    # ----------- Rows 2-3: thrent and COAST frequency ------------
    ax = Axis(lout[2,1:N_ast]; xlabel="−AST", ylabel="Thresholded\nEntropy.", ylabelrotation=0, xgridvisible=false, ygridvisible=false, xticks=(-asts, string.(-asts)), xlabelvisible=false, xticklabelsvisible=false)
    xlims!(ax, -(1.5*asts[end]-0.5*asts[end-1]), -(1.5*asts[1]-0.5*asts[2]))
    for i_anc = 1:N_anc
        scatterlines!(ax, -asts, thresholded_entropy[:,i_anc]; color=:gray79)
    end
    scatterlines!(ax, -asts, SB.mean(thresholded_entropy; dims=2)[:,1]; color=:red)
    ax = Axis(lout[3,1:N_ast]; xlabel="−AST", ylabel="COAST\nfrequency", ylabelrotation=0, xgridvisible=false, ygridvisible=false, xticks=(-asts, string.(-asts)), xlabelvisible=true, xticklabelsvisible=true)
    xlims!(ax, -(1.5*asts[end]-0.5*asts[end-1]), -(1.5*asts[1]-0.5*asts[2]))
    for i_ast = 1:N_ast
        scatterlines!(ax, -asts[i_ast].*ones(2), [0, SB.mean(idx_astmaxthrent.==i_ast)]; color=:black, linewidth=8)
    end
    scatter!(ax, -threshold_neglog, 0.5; marker=:star6, color=:cyan, markersize=18)
    scatter!(ax, -perturbation_neglog, 0.5; marker=:star6, color=:orange, markersize=18)
    scatter!(ax, -(perturbation_neglog-threshold_neglog), 0.5; marker=:star6, color=:red, markersize=18)
    ylims!(ax, 0, 1)

    # --------- Row 4: the Thrent-based mixture --------------
    i_astmaxthrent_mean = round(Int, SB.mean(idx_astmaxthrent)) # Put it horizontally at the mean COAST position 
    @show i_astmaxthrent_mean
    ax = Axis(lout[4,N_ast-i_astmaxthrent_mean+1]; theme_ax..., xscale=identity, yscale=identity, ylabel="AST = argmax(thresh. ent.)", ylabelrotation=0)
    lines!(ax, ccdf2pmf(ccdf_peak_anc), bin_centers[i_bin_thresh:N_bin]; color=:gray79, linestyle=:solid, linewidth=3, label="Ancestors only")
    lines!(ax, ccdf2pmf(ccdf_peak_valid), bin_centers[i_bin_thresh:N_bin]; color=:black, linestyle=:solid, label="Long DNS", linewidth=2)
    lines!(ax, ccdf2pmf(ccdf_peak_wholetruth), bin_centers[i_bin_thresh:N_bin]; color=:black, linestyle=(:dash,:dense), label="Whole truth", linewidth=2)
    lines!(ax, ccdf2pmf(ccdf_moctail_astmaxthrent_rect), bin_centers[i_bin_thresh:N_bin]; color=:red, linewidth=1)
    if i_astmaxthrent_mean < N_ast; ax.ylabelvisible = ax.yticklabelsvisible = false; end

    for i_col = 1:N_ast
        for i_row = [1,4]
            if length(contents(lout[i_row,i_col])) == 0; continue; end
            ax = content(lout[i_row,i_col])
            if i_col < N_ast; colgap!(lout, i_col, 0); end
            if i_col > 1; ax.ylabelvisible = ax.yticklabelsvisible = false; end
            ax.xlabelvisible = ax.xticklabelsvisible = false
            #xlims!(ax, minimum(filter(c->c>0, ccdf_peak_valid))/16, 1.0)
            #ylims!(ax, 1.1*threshold-0.1*1, 1)
        end
    end

    # ------------ Rows 5-7:  various divergences ------
    for (i_row,losses_astunif,loss_astmaxthrent,divname) = zip(
                                                             5:7,
                                                             (losses_astunif_hell,losses_astunif_chi2,losses_astunif_wass),
                                                             (loss_astmaxthrent_hell,loss_astmaxthrent_chi2,loss_astmaxthrent_wass),
                                                             ("Hellinger\nDistance","χ² Divergence","𝐿¹ Distance")
                                                            )
        ax = Axis(lout[i_row,1:N_ast]; xlabel="−AST", ylabel=divname, ylabelrotation=0, yscale=log10, xgridvisible=false, ygridvisible=false, xticks=(-asts, string.(-asts)))
        xlims!(ax, -(1.5*asts[end]-0.5*asts[end-1]), -(1.5*asts[1]-0.5*asts[2]))
        scatterlines!(ax, -asts, losses_astunif; color=:black)
        hlines!(ax, loss_astmaxthrent; color=:black, linestyle=(:dash,:dense))
    end

    for row = [1,4] # make the PMF rows bigger
        rowsize!(lout, row, Relative(3/11))
    end
    for row = [2,3,5,6] # remove gaps between lineplot rows
        ax = content(lout[row,:])
        ax.xlabelvisible = ax.xticklabelsvisible = false
        rowgap!(lout, row, 0)
    end
    
    save(joinpath(figdir, "ccdfs_moctail.png"), fig)




    # Plot entropies as functions of AST 
    fig = Figure(size=(300,150))
    lout = fig[1,1] = GridLayout()
    ax = Axis(lout[1,1]; theme_ax..., xlabel="−AST", ylabel="Thresh. Ent.")
    for i_anc = 1:N_anc
        scatterlines!(ax, reverse(-asts), reverse(thresholded_entropy[:,i_anc]), color=:gray79, marker=:circle)
        i_ast_argmax = idx_astmaxthrent[i_anc]
        scatter!(ax, -asts[i_ast_argmax], thresholded_entropy[i_ast_argmax,i_anc]; color=:gray, marker=:star6)
    end
    scatterlines!(ax, reverse(-asts), reverse(SB.mean(thresholded_entropy; dims=2))[:,1]; color=:black, label="Mean", marker=:circle)
    vlines!(ax, -SB.mean(asts[idx_astmaxthrent]); color=:black, linestyle=(:dash,:dense))
    ax.xticks = reverse(-asts)
    #ax.xticklabels = string.(reverse(-asts))
    save(joinpath(figdir, "thrent_overlay.png"), fig)

    # Plot the descendant peaks as a function of -AST; one row for each ancestor
    theme_ax = (xticklabelsize=16, yticklabelsize=16, xlabelsize=20, ylabelsize=20, xgridvisible=false, ygridvisible=false, titlefont=:regular, titlesize=20)
    fig = Figure(size=(500,60*N_anc))
    lout = fig[1,1] = GridLayout()
    xtickvals = reverse(-round.(Int, range(asts[1], asts[end]; length=3)))
    xticks = (xtickvals, string.(xtickvals))
    for i_anc = 1:N_anc
        # Left column: maxima due to each AST 
        ax = Axis(lout[i_anc,1]; theme_ax..., xticks=xticks, xticklabelrotation=-pi/2)
        for i_ast = 1:N_ast
            scatter!(ax, -asts[i_ast]*ones(N_dsc), Rs_peak_dsc[:,i_ast,i_anc]; color=:red, marker=:circle)
        end
        hlines!(ax, bin_lower_edges[i_bin_thresh]; color=:gray79)
        hlines!(ax, Rs_peak_anc[i_anc]; color=:black, linestyle=(:dash,:dense))
        ax = Axis(lout[i_anc,2]; theme_ax..., xticks=xticks, xticklabelrotation=-pi/2, yticklabelsvisible=false)
        scatterlines!(ax, -reverse(asts), reverse(thresholded_entropy[:,i_anc]); color=:red)


    end
    for i_anc = 1:N_anc
        ax1,ax2 = [content(lout[i_anc,j]) for j=1:2]

        ax1.xticklabelsvisible = ax1.xlabelvisible = (i_anc==N_anc)
        ax2.xticklabelsvisible = ax1.xlabelvisible = (i_anc==N_anc)

        ax1.ylabel = "$(i_anc==1 ? "Ancestor " : "") $(i_anc)"
        ax1.yticklabelsvisible = false
        ax1.ylabelrotation = 0

        ax2.ylabel = ""

        if i_anc == 1; ax1.title = "𝑅*"; ax2.title = "Thresh. Ent."; end
        if i_anc < N_anc; rowgap!(lout, i_anc, 0); end

        ylims!(ax1, 2*bin_lower_edges[i_bin_thresh]-1, 1.0)
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
function hatickvals(ylo,yhi)
    nlg_first = ceil(Int, nlg1m(ylo))
    nlg_last = floor(Int, nlg1m(yhi))
    nlgs = unique(round.(Int, range(nlg_first, nlg_last; length=3)))
    tickvals = nlg1m_inv.(nlgs)
    ticklabs = ["1−2^(−$(tv))" for tv=tickvals] 
    return (tickvals,ticklabs)
end

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



function plot_boosts(datadir::String, figdir::String, asts::Vector{Int64}, bst::Int64, N_dsc::Int64, bin_lower_edges::Vector{Float64}, i_bin_thresh::Int64, perturbation_neglog::Int64, latentize::Bool) # could also have decreasing intervals, as in COAST paper.
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

    theme_ax = (xticklabelsize=16, yticklabelsize=16, xlabelsize=20, ylabelsize=20, xgridvisible=false, ygridvisible=false, titlefont=:regular, titlesize=20)

    N_anc = length(Rs_peak)
    N_ast = length(asts)
    threshold = bin_lower_edges[i_bin_thresh]


    jldopen(joinpath(datadir,"xs_dscs.jld2"), "r") do f
        for i_anc = 1:min(2,N_anc)
            @show i_anc
            entropy_thresholded = zeros(Float64, N_ast)
            entropy_total = zeros(Float64, N_ast)

            fig = Figure(size=(200*4,75*N_ast))
            lout = fig[1,1] = GridLayout()
            for i_ast = 1:N_ast
                @show i_ast
                ax1 = Axis(lout[i_ast,1]; ylabel="$(i_ast==1 ? "AST = " : "")$(asts[i_ast])", ylabelrotation=0, yticklabelsvisible=false, xlabel="𝑡", title="𝑅(𝑥(𝑡))", theme_ax...)
                ax2 = Axis(lout[i_ast,2]; ylabel="AST=$(asts[i_ast])", ylabelrotation=0, yticklabelsvisible=false, xlabel="𝑡", title="Peak 𝑅*", theme_ax...)
                ax3 = Axis(lout[i_ast,3]; ylabel="AST=$(asts[i_ast])", ylabelrotation=0, yticklabelsvisible=false, xlabel="δ𝑥(𝑡*−𝐴)", title="𝑅*(δ𝑥)", theme_ax..., xticklabelrotation=-pi/2) # xticks=([-1,-1/2,0,1/2,1]./(2^perturbation_neglog), ["−1","−½","0","+½","+1"]))
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
                    scatter!(ax3, x_init[1]-xs_anc[1,ts_peak[i_anc]-asts[i_ast]-ts_anc[1]+1], maximum(xs_dsc[1,:]); color=:red, marker=:star5)
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
                xlims!(ax3, ([-1,1]./(2^perturbation_neglog))...)
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
            ax4 = Axis(lout[:,4]; title="Thresh.\nEntropy", theme_ax..., ylabelvisible=false, yticklabelsvisible=false, xticklabelrotation=-pi/2)
            scatterlines!(ax4, entropy_thresholded, -asts, color=:red)
            ax5 = Axis(lout[:,5]; title="Total\nEntropy", theme_ax..., ylabelvisible=false, yticklabelsvisible=false, xticklabelrotation=-pi/2)
            scatterlines!(ax5, entropy_total, -asts; color=:steelblue)
            ylims!(ax4, -1.5*asts[end]+0.5*asts[end-1], -1.5*asts[1]+0.5*asts[2])
            for i_ast = 1:N_ast-1
                rowgap!(lout, i_ast, 0)
            end
            linkxaxes!((content(lout[i_ast,3]) for i_ast=1:N_ast)...)
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
    exptdir = joinpath("/Users/justinfinkel/Documents/postdoc_mit/computing/COAST_results/Chaos1D","2025-09-30",strrep(bpar))
    datadir = joinpath(exptdir, "data")
    figdir = joinpath(exptdir, "figures")
    mkpath(exptdir)
    mkpath(datadir)
    mkpath(figdir)

    N_bin_over = 8
    threshold = compute_cquant_peak_wholetruth(1/2^bpar.threshold_neglog)
    N_bin = N_bin_over * 2^bpar.threshold_neglog
    i_bin_thresh = N_bin - N_bin_over + 1
    # - do NOT adjust bins dependig on latentizatio -
    if false && bpar.latentize
        threshold_z = conjugate_fwd(threshold)
        bin_lower_edges_z = vcat(range(0, threshold_z; length=i_bin_thresh)[1:end-1], range(threshold_z, 1; length=N_bin_over+1)[1:end-1])
        bin_lower_edges = conjugate_bwd.(bin_lower_edges_z)
    else
        bin_lower_edges = vcat(range(0, threshold; length=i_bin_thresh)[1:end-1], range(threshold, 1; length=N_bin_over+1)[1:end-1])
    end
    bin_edges = vcat(bin_lower_edges, 1.0)
    bin_centers = vcat((bin_lower_edges[1:N_bin-1] .+ bin_lower_edges[2:N_bin])./2, (bin_lower_edges[N_bin]+1.0)/2)
    ccdf_peak_wholetruth = compute_ccdf_peak_wholetruth.(bin_lower_edges[i_bin_thresh:N_bin]) ./ compute_ccdf_peak_wholetruth(threshold)
    pdf_wholetruth = compute_pdf_wholetruth.(bin_centers)

    asts = collect(range(bpar.ast_min, bpar.ast_max; step=1))
    duration_plot = 3*2^bpar.threshold_neglog # long enough to capture ~3 peaks 
    perturbation_width = 1/(2^bpar.perturbation_neglog)
    @show threshold,bin_lower_edges[i_bin_thresh-1:i_bin_thresh+1]

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
        boost_peaks(threshold, bpar.perturbation_neglog, asts, bpar.bst, bpar.bit_precision, bpar.num_descendants, seed_boost, datadir, "ancgen"; latentize=bpar.latentize, overwrite_boosts=overwrite_boosts)
    end
    if todo["plot_boosts"]
        plot_boosts(datadir, figdir, asts, bpar.bst, bpar.num_descendants, bin_lower_edges, i_bin_thresh, bpar.perturbation_neglog, bpar.latentize)
    end
    if todo["mix_conditional_tails"]
        mix_conditional_tails(datadir, asts, bpar.num_descendants, bpar.bst, bin_lower_edges, i_bin_thresh, ; ccdf_peak_wholetruth=ccdf_peak_wholetruth)
    end
    if todo["plot_moctails"]
        plot_moctails(datadir, figdir, asts, bpar.num_descendants, bpar.bst, bin_lower_edges, i_bin_thresh, bpar.perturbation_neglog, bpar.threshold_neglog; ccdf_peak_wholetruth=ccdf_peak_wholetruth)
    end
end


main()
