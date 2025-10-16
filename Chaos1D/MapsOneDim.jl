
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

ornot(dt::DataType) = Union{Nothing,dt}

function intensity(xs::Vector{Float64})
    return xs[1]
end

function intensity(xs::Matrix{Float64})
    return xs[1,:]
end


function empirical_ccdf(x::Vector{<:Number})
    N = length(x)
    order = sortperm(x)
    ccdf = (collect(range(N, 1; step=-1)) .- 0.5)./N
    return x[order], ccdf
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


function ccdf2pmf(ccdf::Vector{Float64})
    pmf = vcat(-diff(ccdf), ccdf[end])
    return pmf
end

function ccdf2pdf(ccdf::Vector{Float64}, bin_edges::Vector{Float64})
    return ccdf2pmf(ccdf) ./ diff(bin_edges)
end

function chi2div(ccdf_truth::Vector{Float64}, ccdf_approx::Vector{Float64})
    pmf_truth = ccdf2pmf(ccdf_truth)
    pmf_approx = ccdf2pmf(ccdf_approx)
    return sum((pmf_truth .- pmf_approx).^2 ./ pmf_truth)
end

function hellingerdist(ccdf_truth::Vector{Float64}, ccdf_approx::Vector{Float64})
    pmf_truth = ccdf2pmf(ccdf_truth)
    pmf_approx = ccdf2pmf(ccdf_approx)
    return sum((sqrt.(pmf_truth) .- sqrt.(pmf_approx)).^2)
end

function wassersteindist(ccdf_truth::Vector{Float64}, ccdf_approx::Vector{Float64})
    return sum(abs.(ccdf2pmf(ccdf_truth) .- ccdf2pmf(ccdf_approx)))
end

function powerofhalfstring(k::Int64)
    symbols = ["(½)","(½)²","(½)³","(½)⁴","(½)⁵","(½)⁶","(½)⁷","(½)⁸","(½)⁹","(½)¹⁰","(½)¹¹","(½)¹²","(½)¹³","(½)¹⁴","(½)¹⁵"]
    if 1 <= k <= length(symbols)
        return symbols[k]
    end
    return "(½)^$(k)"
end

function boost_peaks(simulate_fun::Function, latentize::Bool, conjugate_fwd_fun::Function, conjugate_bwd_fun::Function, threshold::Float64, perturbation_neglog::Int64, asts::Vector{Int64}, bst::Int64, bit_precision::Int64, Ndsc_per_leadtime::Int64, seed::Int64, datadir::String, file_suffix::String; overwrite_boosts::Bool=false) 
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
    datafile = joinpath(datadir,"xs_dscs.jld2")
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
                    z_init_anc = (latentize ? conjugate_fwd_fun : identity)(x_init_anc[1])
                    z_init_dsc = mod(
                                     (
                                      floor(Int, z_init_anc*2^perturbation_neglog)
                                      + pert_seq[i_dsc]
                                     ) / (2^perturbation_neglog), 
                                     1
                                    )
                    x_init_dsc = [(latentize ? conjugate_bwd_fun : identity)(z_init_dsc)]
                    xs_dsc,ts_dsc = simulate_fun(x_init_dsc, ast+bst, bit_precision, rng)
                    f[joinpath(anckey,astkey,"idsc$(i_dsc)","t_split")] = t_split
                    f[joinpath(anckey,astkey,"idsc$(i_dsc)","xs")] = xs_dsc
                    f[joinpath(anckey,astkey,"idsc$(i_dsc)","x_init")] = x_init_dsc
                end
            end
        end
    end
    return
end

function plot_peaks_over_threshold(thresh::Float64, duration_spinup::Int64, duration_plot::Int64, datadir::String, figdir::String, file_suffix::String; bin_edges::ornot(Vector{Float64})=nothing, i_bin_thresh::ornot(Int64)=nothing, ccdf_peak_wholetruth::ornot(Vector{Float64})=nothing, pdf_wholetruth::ornot(Vector{Float64})=nothing, return_time_wholetruth::ornot(Float64)=nothing)

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
    ax_hist_Rs = Axis(lout[1,2]; theme_ax..., xlabel="𝑝(𝑟)", yticklabelsvisible=false,)
    ax_hist_peaks = Axis(lout[2,2]; theme_ax..., xlabel="ℙ{𝑅(𝑋)*>𝑟}")
    ax_hist_waits = Axis(lout[3,2]; theme_ax..., xlabel="ℙ{τ > 𝑡*ₙ₊₁-𝑡*ₙ}", xscale=log2)

    # Full timeseries
    lines!(ax_Rs, ts2plot, Rs[ts2plot]; color=:black)
    hlines!(ax_Rs, thresh; color=:gray, linewidth=1, alpha=0.5)
    scatter!(ax_Rs, ts_peak[peaks2plot], Rs_peak[peaks2plot]; color=:black, marker=:star5)
    bin_edges_Rs = (isnothing(bin_edges) ? collect(range(0, 1; length=65)) : bin_edges)
    bin_centers_Rs = (bin_edges_Rs[1:end-1] .+ bin_edges_Rs[2:end])./2
    bins2plot = round.(Int, range(1, length(bin_centers_Rs); length=33))
    hist_Rs = SB.normalize(SB.fit(SB.Histogram, Rs[cluster_starts[1]:cluster_stops[end]], bin_edges_Rs); mode=:pdf)
    if !isnothing(pdf_wholetruth)
        lines!(ax_hist_Rs, pdf_wholetruth[bins2plot], bin_centers_Rs[bins2plot]; color=:gray79, linewidth=4)
    end
    lines!(ax_hist_Rs, hist_Rs.weights[bins2plot], bin_centers_Rs[bins2plot]; color=:black)
    ylims!(ax_hist_Rs, 0, 1)
    ylims!(ax_Rs, 0, 1)
    xlims!(ax_hist_Rs, 0, 1.25)
    ax_hist_Rs.xticks = ([0, 1, 1.25], ["0","1","1.25"])
    linkyaxes!(ax_Rs, ax_hist_Rs)

    # Peak timeseries 
    scatter!(ax_peaks, ts_peak, Rs_peak; color=:black, marker=:circle)
    peaks_sorted,ccdf_peaks = empirical_ccdf(Rs_peak)
    if !isnothing(ccdf_peak_wholetruth)
        lines!(ax_hist_peaks, ccdf_peak_wholetruth, bin_edges[i_bin_thresh:end-1]; color=:gray79, linewidth=3)
    end
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


function nlg1m(x::Number) 
    return -log1p(-x)/log(2) # log_2(1/(1-x))
end
function nlg1m_inv(y::Number) # 1 - 1/(2^y)
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




function plot_boosts(datadir::String, figdir::String, asts::Vector{Int64}, bst::Int64, N_dsc::Int64, bin_lower_edges::Vector{Float64}, i_bin_thresh::Int64, perturbation_neglog::Int64) # could also have decreasing intervals, as in COAST paper.
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
            entropy_thresholded = zeros(Float64, N_ast)
            entropy_total = zeros(Float64, N_ast)

            fig = Figure(size=(200*4,75*N_ast))
            lout = fig[1,1] = GridLayout()
            for i_ast = 1:N_ast
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
        ax = Axis(lout[1,N_ast-i_ast+1]; theme_ax..., xscale=identity, yscale=identity, ylabel="Tail PDFs,\nUniform AST", ylabelrotation=0)
        lines!(ax, ccdf2pdf(ccdf_peak_anc, bin_edges[i_bin_thresh:end]), bin_centers[i_bin_thresh:N_bin]; color=:gray79, linestyle=:solid, linewidth=3, label="Ancestors only")
        lines!(ax, ccdf2pdf(ccdf_peak_valid, bin_edges[i_bin_thresh:end]), bin_centers[i_bin_thresh:N_bin]; color=:black, linestyle=:solid, label="Long DNS", linewidth=2)
        if !isnothing(ccdf_peak_wholetruth)
            lines!(ax, ccdf2pdf(ccdf_peak_wholetruth, bin_edges[i_bin_thresh:end]), bin_centers[i_bin_thresh:N_bin]; color=:black, linestyle=(:dash,:dense), label="Whole truth", linewidth=2)
        end
        lines!(ax, ccdf2pdf(ccdfs_moctail_astunif_rect[:,i_ast], bin_edges[i_bin_thresh:end]), bin_centers[i_bin_thresh:N_bin]; color=:red, linewidth=1)
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
    #for i_ast = 1:N_ast
    #    scatterlines!(ax, -asts[i_ast].*ones(2), [0, SB.mean(idx_astmaxthrent.==i_ast)]; color=:black, linewidth=8)
    #end
    coast_freq = zeros(Int64, N_ast)
    @show coast_freq
    for i_ast = 1:N_ast
        coast_freq[i_ast] += sum(idx_astmaxthrent.==i_ast)
    end
    @show coast_freq
    stairs!(ax, -asts, coast_freq/N_anc, color=:black, linewidth=3, step=:center)
    scatter!(ax, -threshold_neglog, 0.5; marker=:star6, color=:cyan, markersize=18)
    scatter!(ax, -perturbation_neglog, 0.5; marker=:star6, color=:orange, markersize=18)
    scatter!(ax, -(perturbation_neglog-threshold_neglog), 0.5; marker=:star6, color=:red, markersize=18)
    ylims!(ax, -0.01, 1.01)

    # --------- Row 4: the Thrent-based mixture --------------
    i_astmaxthrent_mean = round(Int, SB.mean(idx_astmaxthrent)) # Put it horizontally at the mean COAST position 
    ax = Axis(lout[4,N_ast-i_astmaxthrent_mean+1]; theme_ax..., xscale=identity, yscale=identity, ylabel="AST = argmax(thresh. ent.)", ylabelrotation=0)
    lines!(ax, ccdf2pmf(ccdf_peak_anc), bin_centers[i_bin_thresh:N_bin]; color=:gray79, linestyle=:solid, linewidth=3, label="Ancestors only")
    lines!(ax, ccdf2pmf(ccdf_peak_valid), bin_centers[i_bin_thresh:N_bin]; color=:black, linestyle=:solid, label="Long DNS", linewidth=2)
    if !isnothing(ccdf_peak_wholetruth)
        lines!(ax, ccdf2pmf(ccdf_peak_wholetruth), bin_centers[i_bin_thresh:N_bin]; color=:black, linestyle=(:dash,:dense), label="Whole truth", linewidth=2)
    end
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
    return 
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
    @show Npeaks
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


function plot_dns(duration_spinup::Int64, duration_spinon::Int64, datadir::String, figdir::String, outfile_suffix::String; edges::ornot(Vector{Float64})=nothing, pdf_wholetruth::ornot(Vector{Float64})=nothing)
    xs,ts = jldopen(joinpath(datadir, "dns_$(outfile_suffix).jld2"), "r") do f
        return f["xs"],f["ts"]
    end
    t0 = duration_spinup
    t1 = t0 + duration_spinon
    if isnothing(edges)
        edges = collect(range(0,1;length=33))
    end
    h = SB.normalize(SB.fit(SB.Histogram, xs[1,t0:t1], edges); mode=:pdf)
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
    if !isnothing(pdf_wholetruth)
        lines!(ax_hist, pdf_wholetruth, bincenters; color=:black, linestyle=(:dash,:dense), linewidth=3)
    end
    scatterlines!(ax_hist, h.weights, bincenters; color=:steelblue2, markersize=2)

    colsize!(lout, 2, Relative(1/6))
    colgap!(lout, 1, 0)

    save(joinpath(figdir, "dns_timeseries_hist_$(outfile_suffix).png"), fig)
end

function mix_conditional_tails(datadir::String, asts::Vector{Int64}, N_dsc::Int64, bst::Int64, bin_lower_edges::Vector{Float64}, i_bin_thresh::Int64; ccdf_peak_wholetruth::Union{Nothing,Vector{Float64}}=nothing)
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
    N_bin = length(bin_lower_edges)

    ccdf_peak_anc = compute_empirical_ccdf(Rs_peak_anc, bin_lower_edges[i_bin_thresh:N_bin])
    ccdf_peak_valid = compute_empirical_ccdf(Rs_peak_valid, bin_lower_edges[i_bin_thresh:N_bin])
    #ccdf_peak_wholetruth = (1 .- bin_lower_edges[i_bin_thresh:N_bin])./(1 .- bin_lower_edges[i_bin_thresh]) # even better than ground truth 
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
    ccdf_moctail_astmaxthrent,ccdf_moctail_astmaxthrent_rect = (zeros(Float64, (N_b,)) for N_b=(N_bin,N_bin-i_bin_thresh+1))
    thresholded_entropy = zeros(Float64, (N_ast, N_anc))
    for i_anc = 1:N_anc
        for i_ast = 1:N_ast
            thresholded_entropy[i_ast,i_anc] = compute_thresholded_entropy(Rs_peak_dsc[:,i_ast,i_anc], bin_lower_edges[i_bin_thresh:end])
            ccdfs_dsc[:,i_ast,i_anc] .= compute_empirical_ccdf(Rs_peak_dsc[:,i_ast,i_anc], bin_lower_edges)
            ccdfs_dsc_rect[:,i_ast,i_anc] .= ccdfs_dsc[i_bin_thresh:N_bin,i_ast,i_anc] .+ (1-ccdfs_dsc[i_bin_thresh,i_ast,i_anc]).*(Rs_peak_anc[i_anc] .> bin_lower_edges[i_bin_thresh:N_bin])
            ccdfs_moctail_astunif[:,i_ast] .+= ccdfs_dsc[:,i_ast,i_anc]./N_anc
            ccdfs_moctail_astunif_rect[:,i_ast] .+= ccdfs_dsc_rect[:,i_ast,i_anc]./N_anc
        end
        # Max-thresholded-entropy: take LAST instance of maximum
        idx_astmaxthrent[i_anc] = N_ast - argmax(reverse(thresholded_entropy[:,i_anc])) + 1
        @show thresholded_entropy[i_anc]
        # Oh wait but need to apply adjustment...
        ccdf_moctail_astmaxthrent .+= ccdfs_dsc[:,idx_astmaxthrent[i_anc],i_anc]./N_anc
        ccdf_moctail_astmaxthrent_rect .+= ccdfs_dsc_rect[:,idx_astmaxthrent[i_anc],i_anc]./N_anc
    end
    @show idx_astmaxthrent

    # compute losses: with respect to the whole truth if it is available, but otherwise the ground truth 
    ccdf_peak_truth = (isnothing(ccdf_peak_wholetruth) ? ccdf_peak_valid : ccdf_peak_wholetruth)
    losses_astunif_hell,losses_astunif_chi2,losses_astunif_wass = (zeros(Float64, N_ast) for _=1:3)
    loss_astmaxthrent_hell = hellingerdist(ccdf_peak_truth, ccdf_moctail_astmaxthrent_rect)
    loss_astmaxthrent_chi2 = chi2div(ccdf_peak_truth, ccdf_moctail_astmaxthrent_rect)
    loss_astmaxthrent_wass = wassersteindist(ccdf_peak_truth, ccdf_moctail_astmaxthrent_rect)
    for i_ast = 1:N_ast
        losses_astunif_hell[i_ast] = hellingerdist(ccdf_peak_truth, ccdfs_moctail_astunif_rect[:,i_ast])
        losses_astunif_chi2[i_ast] = chi2div(ccdf_peak_truth, ccdfs_moctail_astunif_rect[:,i_ast])
        losses_astunif_wass[i_ast] = wassersteindist(ccdf_peak_truth, ccdfs_moctail_astunif_rect[:,i_ast])
    end
    println("asts, losses_astunif_hell")
    display(hcat(asts, losses_astunif_hell))
    println("asts, losses_astunif_chi2")
    display(hcat(asts, losses_astunif_chi2))
    println("asts, losses_astunif_wass")
    display(hcat(asts, losses_astunif_wass))

    # Save results to file 

    jldopen(joinpath(datadir,"boost_stats.jld2"), "w") do f
        f["ccdf_peak_anc"] = ccdf_peak_anc
        f["ccdf_peak_valid"] = ccdf_peak_valid
        f["Rs_peak_dsc"] = Rs_peak_dsc
        f["idx_astmaxthrent"] = idx_astmaxthrent
        f["ccdfs_dsc"] = ccdfs_dsc
        f["ccdfs_dsc_rect"] = ccdfs_dsc_rect
        f["ccdfs_moctail_astunif"] = ccdfs_moctail_astunif
        f["ccdfs_moctail_astunif_rect"] = ccdfs_moctail_astunif_rect
        f["ccdf_moctail_astmaxthrent"] = ccdf_moctail_astmaxthrent
        f["ccdf_moctail_astmaxthrent_rect"] = ccdf_moctail_astmaxthrent_rect
        f["losses_astunif_hell"] = losses_astunif_hell
        f["losses_astunif_chi2"] = losses_astunif_chi2
        f["losses_astunif_wass"] = losses_astunif_wass
        f["loss_astmaxthrent_hell"] = loss_astmaxthrent_hell
        f["loss_astmaxthrent_chi2"] = loss_astmaxthrent_chi2
        f["loss_astmaxthrent_wass"] = loss_astmaxthrent_wass
        f["thresholded_entropy"] = thresholded_entropy
    end
    return
end


