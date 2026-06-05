using Random: MersenneTwister
import StatsBase as SB
using Statistics: mean, quantile
using Printf: @sprintf
using JLD2: jldopen, jldsave
using Infiltrator: @infiltrate
using LogExpFunctions: xlogx, xlogy
using CairoMakie

include("statfuns.jl")
include("displayfuns.jl")

ornot(dt::DataType) = Union{Nothing,dt}

function illustrate_map(x_init::Float64, F::Function, simulate_fun::Function, rng::Random.AbstractRNG, mapsymbol::String, statesymbol::String, mapname::String, plotdir::String, outfilename::String)
    T = 11
    _, xs, ts = simulate([x_init,], T, rng, 32)

    xgrid = collect(range(0, 1; length=65)[1:end-1])
    pofx = compute_pdf_wholetruth.(xgrid[2:end-1])
    pdflo = minimum(pofx)
    pdfhi = maximum(pofx)

    theme_ax,theme_leg = get_themes()
    theme_ax = (theme_ax..., xlabelsize=14, ylabelsize=14, titlesize=14, xticklabelsize=12, yticklabelsize=12)

    fig = Figure(size=(400,300))
    lout = fig[1,1] = GridLayout()
    ax = Axis(lout[1,1]; xlabel=statesymbol, ylabel="$(mapsymbol)($(statesymbol))", title=mapname, limits=((0,1),(0,1)), titlefont="Menlo",  xticklabelfont="Menlo", yticklabelfont="Menlo", xgridvisible=false, ygridvisible=false)
    lines!(ax, xgrid, F.(xgrid); color=:black)
    lines!(ax, xgrid, xgrid; color=:grey79, linewidth=3)
    scatter!(ax,x_init,x_init,color=:goldenrod,marker=:star6,markersize=25)
    scatter!(ax,xs[T],xs[T],color=:firebrick,marker=:star6,markersize=25)
    x0 = x_init
    for t = 1:T
        x1 = xs[1,t]
        arrows2d!(ax, [x0], [x0], [0.0], [x1-x0]; lengthscale=1.0, color=:goldenrod, align=:tail, shaftwidth=2, tipwidth=5)
        arrows2d!(ax, [x0], [x1], [x1-x0], [0.0]; lengthscale=1.0, color=:firebrick, align=:tail, shaftwidth=2, tipwidth=5)
        x0 = x1
    end
    xlo = -pdflo/10
    xmid = 1.0
    xhi = 2.0 #minimum(pofx)*2
    xtickvalues = unique([0, 1, 2])
    xticklabels = (x->@sprintf("%d",x)).(xtickvalues)
    ax = Axis(lout[1,2]; title="PDF", xlabel="𝑝($(statesymbol))", xgridvisible=false, ygridvisible=false, ylabel=statesymbol, titlefont="Menlo", xlabelfont="Menlo", xticklabelfont="Menlo", ylabelfont="Menlo", yticklabelfont="Menlo",  yticklabelsvisible=false, limits=((xlo,xhi),(0,1)), xticks=(xtickvalues,xticklabels))
    vlines!(ax, 0; color=:black, linestyle=(:dash,:dense))
    lines!(ax, pofx, xgrid[2:end-1]; color=:black)
    scatter!(ax, zeros(T), xs[1,:]; color=:black)
    colsize!(lout, 2, Relative(1/4))
    colgap!(lout,1,10)
    save(joinpath(plotdir, outfilename), fig)
    return
end

function intensity(xs::Vector{Float64})
    return xs[1]
end

function intensity(xs::Matrix{Float64})
    return xs[1,:]
end


function boost_peaks(
        simulate_fun::Function, 
        threshold::Float64, 
        perturbation_neglog::Int64, 
        asts::Vector{Int64}, 
        bst::Int64, 
        Ndsc_per_leadtime::Int64, 
        seed::Int64, 
        datadir::String, 
        file_suffix::String; 
        overwrite_boosts::Bool=false
    ) 
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
    Npeaks = length(ts_peak)
    pert_seq = van_der_corput(Ndsc_per_leadtime) #.* perturbation_width
    pert_seq_uint32 = van_der_corput_uint32(Ndsc_per_leadtime) #.* perturbation_width
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
                Ndsc_already_simulated = (anckey in keys(f) && astkey in keys(f[joinpath(anckey)])) ? length(f[joinpath(anckey,astkey)]) : 0
                for i_dsc = Ndsc_already_simulated+1:Ndsc_per_leadtime
                    rng = MersenneTwister(seed)
                    #x_init_dsc = x_init_anc .+ pert_seq[i_dsc]/(2^perturbation_neglog).*ones(1)
                    #x_init_dsc = mod(
                    #                 (
                    #                  floor(Int, x_init_anc[1]*2^perturbation_neglog)
                    #                  + pert_seq[i_dsc]
                    #                 ) / (2^perturbation_neglog), 
                    #                 1
                    #                ) .* ones(1)
                    # Kill all zeros past the 32nd 
                    X_init_dsc = float64_to_uint32(x_init_anc[1]) & (typemax(UInt32) << (32-perturbation_neglog))
                    X_init_dsc = xor((X_init_dsc>>(32-perturbation_neglog))<<(32-perturbation_neglog), pert_seq_uint32[i_dsc] >> perturbation_neglog) | true # put a zero in the last position if all zeros 
                    x_init_dsc = uint32_to_float64(X_init_dsc) .* ones(1)
                    #@infiltrate
                    _, xs_dsc, ts_dsc = simulate_fun(x_init_dsc, ast+bst, rng) #, perturbation_neglog)
                    f[joinpath(anckey,astkey,"idsc$(i_dsc)","t_split")] = t_split
                    f[joinpath(anckey,astkey,"idsc$(i_dsc)","xs")] = xs_dsc
                    f[joinpath(anckey,astkey,"idsc$(i_dsc)","x_init")] = x_init_dsc
                end
            end
        end
    end
    return
end

function plot_peaks_over_threshold(thresh::Float64, duration_spinup::Int64, duration_plot::Int64, datadir::String, figdir::String, file_suffix::String; bin_edges::ornot(Vector{Float64})=nothing, i_bin_thresh::ornot(Int64)=nothing, ccdf_peak_wholetruth::ornot(Vector{Float64})=nothing, pdf_wholetruth::ornot(Vector{Float64})=nothing, return_time_wholetruth::ornot(Float64)=nothing, nlg2_thresh::ornot(Float64)=nothing, statesymbol::String="𝑥")

    ts, xs = jldopen(joinpath(datadir, "dns_$(file_suffix).jld2"), "r") do f
        return f["ts"], f["xs"]
    end
    Rs = intensity(xs) # pedantically, a scalar 
    bin_edges_Rs = (isnothing(bin_edges) ? collect(range(0, 1; length=65)) : bin_edges)
    bin_centers_Rs = (bin_edges_Rs[1:end-1] .+ bin_edges_Rs[2:end])./2
    bins2plot = round.(Int, range(1, length(bin_centers_Rs); length=33))
    hist_Rs = SB.normalize(SB.fit(SB.Histogram, Rs[duration_spinup:end]#=cluster_starts[1]:cluster_stops[end]]=#, bin_edges_Rs); mode=:pdf)
    ts_peak, Rs_peak, cluster_starts, cluster_stops = jldopen(joinpath(datadir, "dns_peaks_$(file_suffix).jld2"), "r") do f
        return (
                f["ts_peak"],
                f["Rs_peak"], 
                f["cluster_starts"], 
                f["cluster_stops"],
               )
    end
    waits = diff(ts_peak)
    waits_sorted, ccdf_waits = empirical_ccdf(waits)


    ts2plot = duration_spinup .+ (1:duration_plot)
    tlimits = (ts2plot[1],ts2plot[end])
    peaks2plot = 1:length(ts_peak) #findall(ts2plot[1] .<= ts_peak .<= ts2plot[end])

    theme_ax,theme_leg = get_themes()
    theme_ax = (theme_ax..., xlabelsize=14, ylabelsize=14, titlesize=14, xticklabelsize=12, yticklabelsize=12)
    theme_leg = (theme_leg..., labelsize=14, titlesize=14, framevisible=false)
    fig = Figure(size=(620,300))
    lout = fig[1,1] = GridLayout()
    ax_Rs = Axis(lout[1,1]; theme_ax..., title="$(statesymbol)(𝑡)", xlabel="𝑡", limits=(tlimits,(0,1)), xlabelvisible=false)
    ttickvalues = round.(Int64,ts[1].+[1/6,3/6,5/6].*(ts[end]-ts[1]))
    tticklabels = scinot2.(ttickvalues)
    ytickvalues = [thresh, (thresh+1)/2, 1]
    yticklabels = vcat((y->@sprintf("1−2%s",supscr(round(Int64,-nlg1m(y))))).(ytickvalues[1:2]), "1")
    ax_peaks = Axis(lout[2,1]; theme_ax..., title="Peaks {$(statesymbol)ₙ*=$(statesymbol)(𝑡ₙ*)}", xlabel="𝑡ₙ*", limits=((ts[1],ts[end]),(thresh,1)), xticks=(ttickvalues,tticklabels), xlabelvisible=false, yticks=(ytickvalues,yticklabels))
    ax_hist_Rs = Axis(lout[1,2]; theme_ax..., title="PDF", yticklabelsvisible=false, xticklabelrotation=0, xlabelvisible=false, limits=((0,2),(0,1)), xticks=([0, 1, 2], ["0","1","2"]), yticks=(ytickvalues,yticklabels))
    linkyaxes!(ax_Rs, ax_hist_Rs)
    ax_hist_peaks = Axis(lout[2,2]; theme_ax..., title="Peak CCDF", ylabel="$(statesymbol)*", xlabelvisible=false, limits=((0,1),(thresh,1)))
    linkyaxes!(ax_peaks, ax_hist_peaks)


    # Full timeseries
    lines!(ax_Rs, ts2plot, Rs[ts2plot]; color=:black)
    hlines!(ax_Rs, thresh; color=:black, linewidth=1, linestyle=(:dash,:dense))
    scatter!(ax_Rs, ts_peak[peaks2plot], Rs_peak[peaks2plot]; color=:black, marker=:star5, markersize=4)
    if !isnothing(pdf_wholetruth)
        lines!(ax_hist_Rs, pdf_wholetruth[bins2plot], bin_centers_Rs[bins2plot]; color=:gray79, linewidth=4, label="Whole\nTruth")
    end
    simlength = length(ts)-duration_spinup
    log2simlength = round(Int, log2(simlength))
    lines!(ax_hist_Rs, hist_Rs.weights[bins2plot], bin_centers_Rs[bins2plot]; color=:black, label=@sprintf("DNS\n%.1f×%s", simlength/2^log2simlength, poweroftwostring(log2simlength)))
    for ax = (ax_Rs,ax_hist_Rs)
        hlines!(ax, thresh; color=:black, linewidth=1, linestyle=(:dash,:dense), label=@sprintf("Thresh\n1-2%s", supscr(-round(Int64,nlg1m(thresh)))))
    end
    leg = Legend(lout[1,3], ax_hist_Rs; theme_leg...,)

    # Peak timeseries 
    scatter!(ax_peaks, ts_peak, Rs_peak; color=:black, marker=:circle, markersize=3)
    peaks_sorted,ccdf_peaks = empirical_ccdf(Rs_peak)
    if !isnothing(ccdf_peak_wholetruth)
        lines!(ax_hist_peaks, ccdf_peak_wholetruth, bin_edges[i_bin_thresh:end-1]; color=:gray79, linewidth=4)
    end
    lines!(ax_hist_peaks, ccdf_peaks, peaks_sorted; color=:black, )

    # Wait timeseries
    #ax_hist_waits.xscale = log2

    for ax = (ax_hist_Rs, ax_hist_peaks, )
        ax.ylabelvisible = ax.yticklabelsvisible = false
    end

    rowgap!(lout, 1, 0)
    colgap!(lout, 1, 10)
    colsize!(lout, 1, Relative(3/4))
    colsize!(lout, 2, Relative(1/8))

    save(joinpath(figdir, "dns_peaks_over_threshold_$(file_suffix).png"), fig)
end






function plot_boosts(datadir::String, figdir::String, asts::Vector{Int64}, bst::Int64, N_dsc::Int64, bin_lower_edges::Vector{Float64}, i_bin_thresh::Int64, perturbation_neglog::Int64; statesymbol::String="𝑥") 
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

    (
     ccdfs_dsc,
     total_entropy,
     thresholded_entropy,
     extreme_conditional_entropy,
     Rs_peak_dsc,
    ) = jldopen(joinpath(datadir,"boost_stats.jld2"), "r") do f
        return (
                f["ccdfs_dsc"],
                f["total_entropy"],
                f["thresholded_entropy"],
                f["extreme_conditional_entropy"],
                f["Rs_peak_dsc"],
               )
    end

    theme_ax,theme_leg = get_themes()
    theme_ax = (; theme_ax..., xticklabelsize=16, yticklabelsize=16, xlabelsize=20, ylabelsize=20, xgridvisible=false, ygridvisible=false, titlesize=20)
    astcols = astcolors()

    N_anc = length(Rs_peak)
    N_anc2plot = min(2,N_anc)
    N_ast = length(asts)
    threshold = bin_lower_edges[i_bin_thresh]


    xs_init = zeros(Float64, (1, N_dsc, N_ast, N_anc))
    xs_dsc = [zeros(Float64, (1, ast+bst, N_dsc, N_anc)) for ast=asts] # will have some filler values
    ts_init = zeros(Int64, (N_dsc, N_ast, N_anc))
    jldopen(joinpath(datadir,"xs_dscs.jld2"), "r") do f
        for i_anc = 1:N_anc
            for i_ast = 1:N_ast
                for i_dsc = 1:N_dsc
                    dscfullkey = joinpath("ianc$(i_anc)","iast$(i_ast)","idsc$(i_dsc)")
                    xs_init[:,i_dsc,i_ast,i_anc] .= f[joinpath(dscfullkey,"x_init")]
                    xs_dsc[i_ast][:,:,i_dsc,i_anc] .= f[joinpath(dscfullkey,"xs")]
                    ts_init[i_dsc, i_ast, i_anc] = f[joinpath(dscfullkey,"t_split")]
                end
            end
        end
    end
    for i_anc = 1:N_anc2plot
        ytickvalues_ax2 = [2*threshold-1, threshold, 1] 
        yticklabels_ax2 = scinot2near1.(ytickvalues_ax2)
        xtickvalues_ax2 = [-1,0,1]./2^perturbation_neglog
        xticklabels_ax2 = scinot2.(xtickvalues_ax2)
        asttickvalues = [-asts[end], -asts[div(N_ast,2)], 0]
        astticklabels = (A->@sprintf("%d",A)).(asttickvalues)
        xtickvalues_ax4 = round.(Int64, vcat(0,[1/2,1].*log2(N_dsc+1)))
        xticklabels_ax4 = (ee->@sprintf("%d", ee)).(xtickvalues_ax4)
        log2xtickvalues_ax3 = round.(Int64,range(-log2(N_dsc), 0, length=3))
        xtickvalues_ax3 = 2.0 .^ log2xtickvalues_ax3
        xticklabels_ax3 = scinot2.(xtickvalues_ax3)
        title3 = @sprintf("ℙ{%s* >\n%s}", statesymbol, scinot2near1(threshold))
        title4 = (
                  rich("TotEnt", color=astcols["TotEnt"], font="Menlo")
                  * "\n" * 
                  rich("ThrEnt", color=astcols["ThrEnt"], font="Menlo")
                  * "\n" * 
                  rich("XclEnt", color=astcols["XclEnt"], font="Menlo")
                 )

        xs_dsc_mean,xs_dsc_min,xs_dsc_max = ((x_dsc->func(x_dsc[:,:,:,i_anc]; dims=3)[:,:,1]).(xs_dsc)
                                             for func = (mean,minimum,maximum))
        xs_dsc_init_mean,xs_dsc_init_min,xs_dsc_init_max = (func(xs_init[:,:,:,i_anc]; dims=2)[:,1,:]
                                                            for func = (mean,minimum,maximum))
        fig = Figure(size=(200*4,75*N_ast))
        lout = fig[1,1] = GridLayout()
        for i_ast = 1:N_ast
            ast = asts[i_ast]
            ax1 = Axis(lout[i_ast,1]; xlabel=@sprintf("𝑡−𝑡*"), xticks=(asttickvalues,astticklabels), xticklabelrotation=0, xlabelvisible=(i_ast==N_ast), xticklabelsvisible=(i_ast==N_ast), title="$statesymbol(𝑡)", titlevisible=(i_ast==1), theme_ax..., limits=((-asts[end]-1/4, bst+1),(-0.1,1.1)), yticks=([0,1/2,1],["0","½","1"]), yticklabelsvisible=(i_ast==1))
            ax2 = Axis(lout[i_ast,2]; yticklabelsvisible=(i_ast==1), xlabelvisible=(i_ast==N_ast), xticklabelsvisible=(i_ast==N_ast), xlabel="Pert. δ$(statesymbol) at 𝑡*−𝐴", title="Peak $(statesymbol)* = $(statesymbol)(𝑡*)", titlevisible=(i_ast==1), theme_ax..., xticks=(xtickvalues_ax2,xticklabels_ax2), xticklabelrotation=0, limits=(1.25/2^perturbation_neglog.*(-1,1),(2*threshold-1,1)), yticks=(ytickvalues_ax2,yticklabels_ax2))
            hlines!(ax1, [0,1]; color=:grey79, linestyle=:solid)
            vlines!(ax2, [-1,1]./(2^perturbation_neglog); color=:grey79, linestyle=:solid)
            vlines!(ax1, -asts[i_ast]; color=:red)
            vlines!(ax1, 0; color=:black, linestyle=:solid)
            text!(ax1, -asts[end], 0; text=@sprintf("𝐴=%d", asts[i_ast]), font="Menlo", fontsize=14, align=(:left,:bottom))
            # Plot the ancestor
            tidx_anc = ts_peak[i_anc]-ts_anc[1]+1 .+ (-asts[end]:bst)
            # Plot descendant ensemble summary stats
            band!(ax1, -ast:bst, vcat(xs_dsc_init_min[1,i_ast],xs_dsc_min[i_ast][1,:]), vcat(xs_dsc_init_max[1,i_ast],xs_dsc_max[i_ast][1,:]); color=:red, alpha=0.5)
            lines!(ax1, (-ast):bst, vcat(xs_dsc_init_mean[1,i_ast],xs_dsc_mean[i_ast][1,:]); color=:red)
            #lines!(ax1, (-ast):bst, vcat(xs_dsc_init_min[1,i_ast],xs_dsc_min[i_ast][1,:]); color=:red)
            #lines!(ax1, (-ast):bst, vcat(xs_dsc_init_max[1,i_ast],xs_dsc_max[i_ast][1,:]); color=:red)
            scatter!(ax1, -ast, xs_dsc_init_mean[i_ast];  color=:red, marker=:cross)

            scatter!(ax2, xs_init[1,:,i_ast,i_anc].-xs_anc[1,tidx_anc[1+asts[N_ast]-ast]], Rs_peak_dsc[:,i_ast,i_anc]; color=:red, marker=:star5)
            scatter!(ax2, 0, Rs_peak[i_anc]; color=:black, marker=:star6)

            lines!(ax1, ts_anc[tidx_anc].-ts_peak[i_anc], xs_anc[1,tidx_anc]; color=:black, linewidth=1, linestyle=(:dash,:dense))
            hlines!(ax1, threshold; color=:purple, linestyle=(:dash,:dense))
            hlines!(ax2, threshold; color=:purple, linestyle=(:dash,:dense))
        end

        ax3 = Axis(lout[:,3]; xscale=log2, title=title3, theme_ax..., ylabelvisible=false, yticklabelsvisible=false, xticklabelrotation=0, limits=((xtickvalues_ax3[1]/2,xtickvalues_ax3[end]+1/4),(-asts[end]-1/2,-1/2)), xticks=(xtickvalues_ax3,xticklabels_ax3))
        ax4 = Axis(lout[:,4]; title=title4, theme_ax..., ylabelvisible=false, yticklabelsvisible=false, xticklabelrotation=0, xlabel="[bits]", limits=((-1,maximum(total_entropy)+1),(-asts[end]-1/2,-1/2)), xticks=(xtickvalues_ax4,xticklabels_ax4))
        vlines!(ax3, xtickvalues_ax3[[1,end]]; color=:grey79, linestyle=:solid)
        vlines!(ax4, xtickvalues_ax4[[1,end]]; color=:grey79, linestyle=:solid)
        scatterlines!(ax3, ccdfs_dsc[i_bin_thresh,:,i_anc], -asts; color=:red)
        scatterlines!(ax4, total_entropy[:,i_anc], -asts; color=astcols["TotEnt"], linewidth=4, label="TotEnt")
        scatterlines!(ax4, thresholded_entropy[:,i_anc], -asts, color=astcols["ThrEnt"], label="ThrEnt")
        scatterlines!(ax4, extreme_conditional_entropy[:,i_anc], -asts, color=astcols["XclEnt"], label="XclEnt")
        for i_ast = 1:N_ast-1
            rowgap!(lout, i_ast, 0)
        end
        colgap!(lout, 1, 10)
        colgap!(lout, 2, 12)
        colgap!(lout, 3, 10)

        colsize!(lout, 1, Relative(3/8))
        colsize!(lout, 2, Relative(3/8))
        colsize!(lout, 3, Relative(1/8))
        colsize!(lout, 4, Relative(1/8))

        save(joinpath(figdir, "boosts_anc$(i_anc).png"), fig)
        
    end
end


function plot_moctails(
        datadir::String, figdir::String, asts::Vector{Int64}, N_dsc::Int64, bst::Int64, bin_lower_edges::Vector{Float64}, i_bin_thresh::Int64, perturbation_neglog::Int64, threshold_neglog::Int64, mapfun_derivative::Function; 
        ccdf_peak_wholetruth::ornot(Vector{Float64})=nothing, statesymbol::String="𝑥")

    todo = Dict{String,Bool}(
                             "edtast" =>       1,
                             "klconv" =>       1,
                             "moctail" =>      1,
                            )

    # ----------------------------------------------------
    # Plotting 
    #
    (
     ccdf_peak_anc,
     ccdf_peak_valid,
     Rs_peak_dsc,
     idx_coast,
     ccdfs_dsc,
     ccdfs_moctail_astunif,
     ccdf_moctail_coast,
     losses_moctail_astunif_hell,
     losses_moctail_astunif_chi2,
     losses_moctail_astunif_wass,
     losses_moctail_astunif_kldiv,
     loss_moctail_coast_hell,
     loss_moctail_coast_chi2,
     loss_moctail_coast_wass,
     loss_moctail_coast_kldiv,
     thresholded_entropy,
     extreme_conditional_entropy,
     total_entropy,
     Ns_anc_boot,
     Ns_anc_boot_valid,
     ccdfs_moctail_astunif_boot,
     ccdfs_moctail_coast_boot,
     ccdfs_anconly_boot,
     ccdfs_valid_boot,
     losses_moctail_astunif_kldiv_boot,
     losses_moctail_coast_kldiv_boot,
     losses_anconly_kldiv_boot,
     losses_valid_kldiv_boot,
    ) = (
         jldopen(joinpath(datadir,"boost_stats.jld2"), "r") do f
             return (
                     f["ccdf_peak_anc"], 
                     f["ccdf_peak_valid"], 
                     f["Rs_peak_dsc"], 
                     f["idx_coast"], 
                     f["ccdfs_dsc"], 
                     f["ccdfs_moctail_astunif"], 
                     f["ccdf_moctail_coast"], 
                     f["losses_moctail_astunif_hell"], 
                     f["losses_moctail_astunif_chi2"], 
                     f["losses_moctail_astunif_wass"], 
                     f["losses_moctail_astunif_kldiv"], 
                     f["loss_moctail_coast_hell"], 
                     f["loss_moctail_coast_chi2"], 
                     f["loss_moctail_coast_wass"],
                     f["loss_moctail_coast_kldiv"],
                     f["thresholded_entropy"],
                     f["extreme_conditional_entropy"],
                     f["total_entropy"],
                     f["Ns_anc_boot"],
                     f["Ns_anc_boot_valid"],
                     f["ccdfs_moctail_astunif_boot"],
                     f["ccdfs_moctail_coast_boot"],
                     f["ccdfs_anconly_boot"],
                     f["ccdfs_valid_boot"],
                     f["losses_moctail_astunif_kldiv_boot"],
                     f["losses_moctail_coast_kldiv_boot"],
                     f["losses_anconly_kldiv_boot"],
                     f["losses_valid_kldiv_boot"],
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
    N_ast = length(asts)
    N_anc = length(Rs_peak_anc)
    N_bin = length(bin_lower_edges)
    N_bin_over = N_bin - i_bin_thresh + 1
    confint_width = 0.9

    theme_ax,theme_leg = get_themes()
    theme_ax = (; theme_ax..., xlabelsize=10, ylabelsize=10, xticklabelsize=8, yticklabelsize=8, titlesize=10)
    theme_leg = (; theme_leg..., framevisible=true)
    astcols = astcolors()

    # ----------- State-dependence of COAST and FTLE ----------
    !(todo["edtast"] || todo["klconv"] || todo["moctail"]) && return
    #          ----------------------------           
    #          |             |            |             
    #    (AXC) |             |            |
    #          |             |            |             
    #           --------------------------        
    #          |             |            |          
    #Tex2(AXC) |             |            | 
    #Tex2(AUC) |             |            |         
    #           --------------------------             
    #            x(t*-A)        Tex2(AUC)
    #
    #
    #      |   
    #
    #
    xs_prepeak_coast = zeros(Float64, N_anc)
    xs_prepeak_uncoast = zeros(Float64, N_anc)
    coasts = zeros(Int64, N_anc)
    ftles_prepeak_uncoast = zeros(Float64, N_anc)
    ftles_prepeak_coast = zeros(Float64, N_anc)
    uncoast = perturbation_neglog-threshold_neglog
    for i_anc = 1:N_anc
        coasts[i_anc] = asts[idx_coast[i_anc]]
        xs_prepeak_coast[i_anc] = xs_anc[1,ts_peak[i_anc]-ts_anc[1]+1-coasts[i_anc]]
        xs_prepeak_uncoast[i_anc] = xs_anc[1,ts_peak[i_anc]-ts_anc[1]+1-uncoast]
        tidx_coast2peak = ts_peak[i_anc]-ts_anc[1]+1 .+ collect(range(-coasts[i_anc],-1;step=1))
        tidx_uncoast2peak = ts_peak[i_anc]-ts_anc[1]+1 .+ collect(range(-(uncoast),-1;step=1))
        ftles_prepeak_coast[i_anc] = mean(log2.(abs.(mapfun_derivative.(xs_anc[1,tidx_coast2peak]))))
        ftles_prepeak_uncoast[i_anc] = mean(log2.(abs.(mapfun_derivative.(xs_anc[1,tidx_uncoast2peak]))))
    end
    edts_prepeak_coast = 1 ./ ftles_prepeak_coast
    edts_prepeak_uncoast = 1 ./ ftles_prepeak_uncoast
    N_ftle_unif_sample = 5000
    xs_unif_sample = xs_anc[1,1:(N_ftle_unif_sample*uncoast)]
    derivatives_unif_sample = log2.(abs.(mapfun_derivative.(xs_unif_sample)))
    ftles_unif_sample = collect(map(t->mean(derivatives_unif_sample[t:t+uncoast-1]), 1:(N_ftle_unif_sample*uncoast-uncoast)))
                                #mean(reshape(derivatives_unif_sample, (uncoast,N_ftle_unif_sample)); dims=1)[1,:]
    edts_unif_sample = 1 ./ ftles_unif_sample
    xs_unif_sample_spaced = xs_unif_sample[1:N_ftle_unif_sample*uncoast-uncoast] #[1:uncoast:end]


    xlimits = (-0.05,1.05)
    xticks = ([0,1/2,1],["0","1/2","1"])
    Rlimits = (bin_lower_edges[i_bin_thresh],1)
    Rtickvalues = Rlimits[1] .+ [1/6,1/2,2/3].*(Rlimits[2]-Rlimits[1])
    Rticklabels = scinot2near1.(Rtickvalues)
    asttickvalues = round.(Int64, range(1,asts[end];length=3))
    astticklabels = (A->@sprintf("%d",A)).(asttickvalues)
    astticks = (asttickvalues,astticklabels)
    edttickvalues = asttickvalues
    edtticklabels = astticklabels 
    edtticks = (edttickvalues,edtticklabels)
    ftlelimits = extrema(vcat(ftles_prepeak_coast,ftles_prepeak_uncoast,ftles_unif_sample))
    ftletickvalues = sort(unique([ftlelimits[1],0,ftlelimits[2]]))
    ftleticklabels = (t->@sprintf("%d",t)).(ftletickvalues)
    ftleticks = (ftletickvalues,ftleticklabels)
    astlimits = (0,asts[end]+1/2)
    edtlimits = nothing #(0,asts[end]+1/2) # error-doubling time limits 
    scatargs = (; markersize=2)

    fig = Figure(size=(200,180).*1.5)
    lout = fig[1,1] = GridLayout()
    ax11 = Axis(lout[1,1]; theme_ax..., limits=(xlimits,astlimits), xlabel=@sprintf("%s(𝑡*-𝐴ˣᶜ)", statesymbol), xticks=xticks, yticks=astticks, ylabel="AST 𝐴ˣᶜ", title=@sprintf("𝐾=%d, 𝑀=%d", perturbation_neglog, threshold_neglog), titlealign=:left, )
    ax21 = Axis(lout[2,1]; theme_ax..., limits=(xlimits,ftlelimits), xlabel=statesymbol, xticks=xticks, yticks=ftleticks, ylabel="FTLE")
    ax12 = Axis(lout[1,2]; theme_ax..., limits=(ftlelimits,astlimits), xticks=ftleticks, yticks=astticks, xlabel="FTLE")
    ax22 = Axis(lout[2,2]; theme_ax..., limits=(ftlelimits,ftlelimits), xticks=ftleticks, yticks=astticks, xlabel="FTLE", ylabel="FTLE")

    rowsize!(lout,1,Relative(1/2))
    colsize!(lout,2,Relative(100/300))

    scatter!(ax11, xs_prepeak_coast, coasts; color=astcols["XclEnt"], marker=:circle, markersize=4, label="𝐴=XclEnt-COAST")
    scatter!(ax12, ftles_prepeak_coast, coasts; color=astcols["XclEnt"], marker=:circle, markersize=4,label="𝐴=XclEnt-COAST")

    scatter!(ax21, xs_prepeak_coast, ftles_prepeak_coast; color=astcols["XclEnt"], marker=:circle, markersize=4, label="τ=XclEnt-COAST")
    scatter!(ax21, xs_prepeak_uncoast, ftles_prepeak_uncoast; color=astcols["astunif"], marker=:circle, markersize=4, label="τ=𝐾−𝑀")
    scatter!(ax21, xs_unif_sample_spaced, ftles_unif_sample; color=:grey79, marker=:circle, markersize=2, label="τ=𝐾−𝑀\n𝑥∼𝑝")


    idx_long_coast = findall(coasts .> uncoast)
    idx_short_coast = findall(coasts .< uncoast)
    idx_mid_coast = findall(coasts .== uncoast)
    for (idx,color,comparator) = ((idx_short_coast,astcols["astunif"],"<"),(idx_mid_coast,"black","="),(idx_long_coast,astcols["XclEnt"],">"))
        length(idx)>0 && scatter!(ax22, ftles_prepeak_coast[idx], ftles_prepeak_uncoast[idx]; color=color, marker=:circle, markersize=4, label=" $(comparator) 𝐾 − 𝑀")
    end

    for ax=(ax11,ax12)
        ax.xlabelvisible = ax.xticklabelsvisible = true
        hlines!(ax,uncoast; color=astcols["astunif"], linewidth=0.5, label="𝐴=𝐾−𝑀")
    end
    for ax=(ax12,ax22)
        ax.ylabelvisible = ax.yticklabelsvisible = false
        vlines!(ax,[-1,1]; color=:black, linewidth=0.5)
        vlines!(ax,0; color=:black,linestyle=(:dash,:dense), linewidth=0.5)
    end
    for ax=(ax21,ax22)
        hlines!(ax,[-1,1]; color=:black, linewidth=0.5)
        hlines!(ax,0; color=:black,linestyle=(:dash,:dense), linewidth=0.5)
    end
    #leg21 = Legend(lout[3,1], ax21; theme_leg..., markersize=50, rowgap=0)
    #leg22 = Legend(lout[3,2], ax22, "XclEnt-\nCOAST:"; theme_leg..., markersize=50, rowgap=0)


    colgap!(lout,1,0)
    rowgap!(lout,1,15)
    linkyaxes!(ax11,ax12)
    #linkyaxes!(ax21,ax22)
    linkxaxes!(ax11,ax21)
    linkxaxes!(ax12,ax22)
    save(joinpath(figdir,"coast_state_dependence.png"), fig)



    !(todo["klconv"] || todo["moctail"]) && return


    # ---------- convergence with N -----------
    theme_leg = (; theme_leg..., framevisible=false)
    mean_return_period = mean(diff(ts_peak_valid))
    uncoast = perturbation_neglog - threshold_neglog # unconditionally optimal 
    alllosses = vcat(losses_moctail_astunif_kldiv_boot[uncoast,:,:][:], losses_moctail_coast_kldiv_boot[:], losses_valid_kldiv_boot[:])
    kllo,klhi = (func(filter(loss->(isfinite(loss)&(loss>1e-10)),alllosses)) for func=(minimum,maximum))
    klposlo = -xlogx(1/N_dsc)
    ylo = -klposlo/2
    yhi = round(log2(N_bin))

    quantfun(arr,qua) = finitequantile(arr,qua) #(arrpos = filter(ispos,arr); length(arrpos) == 0 ? NaN : quantile(arrpos,qua))
    losses_astunif_lomidhi = [
                              [mapslices(arr->quantfun(arr,qua), losses_moctail_astunif_kldiv_boot[i_ast,:,:]; dims=1)[1,:] for qua=1/2 .+ confint_width.*[-1/2, 0, 1/2]]
                              for i_ast=1:N_ast
                             ]
    losses_coast_lomidhi = [mapslices(arr->quantfun(arr,qua), losses_moctail_coast_kldiv_boot[:,:]; dims=1)[1,:] for qua=1/2 .+ confint_width.*[-1/2,0,1/2]]
    losses_anconly_lomidhi = [mapslices(arr->quantfun(arr,qua), losses_anconly_kldiv_boot[:,:]; dims=1)[1,:] for qua=1/2 .+ confint_width.*[-1/2,0,1/2]]
    losses_valid_lomidhi = [mapslices(arr->quantfun(arr,qua), losses_valid_kldiv_boot[:,:]; dims=1)[1,:] for qua=1/2 .+ confint_width.*[-1/2,0,1/2]]
    function draw_bands!(ax_Nanc, ax_cost, Ns_anc, cost_per_anc, losses_lomidhi, color, label; alpha=0.25, linestyle=:solid)
        losses_lo,losses_mid,losses_hi = losses_lomidhi
        band!(ax_Nanc, Ns_anc, losses_lo, losses_hi; color=color, alpha=alpha)
        scatterlines!(ax_Nanc, Ns_anc, losses_mid; color=color, label=label, marker=:circle, markersize=1, linestyle=linestyle)
        band!(ax_cost, Ns_anc.*cost_per_anc, losses_lo, losses_hi; color=color, alpha=alpha)
        scatterlines!(ax_cost, Ns_anc.*cost_per_anc, losses_mid; color=color, marker=:circle, markersize=1, linestyle=linestyle)
    end
    i_uncoast = findfirst(==(uncoast), asts)

    fig = Figure(size=(450,400))
    # Top panel: KL vs N 
    lout = fig[1,1] = GridLayout()

    xtickvalues_N = unique(2 .^ floor.(Int, range(0, log2(Ns_anc_boot_valid[end]); step=1)))
    xtickvalues_C = xtickvalues_N.*mean_return_period
    xticklabels_N = scinot2.(xtickvalues_N)
    xticklabels_C = scinot2.(xtickvalues_C)
    ytickvalues = [0,klposlo,1,yhi] #[0,sqrt(klposlo*yhi),yhi]
    yticklabels = (y->@sprintf("%.1f",y)).(ytickvalues)
    #ytickvalues = unique(vcat(0, exp10.(round.(Int64, range(log10(kllo), log10(yhi); length=6))), N_bin_over))
    #yticklabels = scinot.(ytickvalues)
    klyscale = Makie.Symlog10(klposlo)

    ax_N = Axis(lout[1,1]; theme_ax..., title=@sprintf("𝐾=%d, 𝑀=%d", perturbation_neglog, threshold_neglog), xlabel="𝑁", ylabel="KL divergence", xgridvisible=false, ygridvisible=false, xscale=log2, yscale=klyscale, limits=((1,Ns_anc_boot_valid[end]), (ylo,yhi)), xticks=(xtickvalues_N,xticklabels_N), yticks=(ytickvalues,yticklabels), titlealign=:left)
    ax_C = Axis(lout[2,1]; theme_ax..., title=@sprintf("𝐾=%d, 𝑀=%d", perturbation_neglog, threshold_neglog), xlabel="Cost", ylabel="KL divergence", xscale=log2, yscale=klyscale, limits=((mean_return_period,max(Ns_anc_boot_valid[end]*mean_return_period, Ns_anc_boot[end]*(mean_return_period+N_dsc*asts[end]))), (ylo,yhi)), yticks=(ytickvalues,yticklabels), xticks=(xtickvalues_C,xticklabels_C), titlealign=:left, )
    # all non-optimal uniform ASTs 
    for i_ast = 1:N_ast
        draw_bands!(ax_N, ax_C, 
                    Ns_anc_boot, mean_return_period+N_dsc*asts[i_ast], 
                    losses_astunif_lomidhi[i_ast],     
                    i_ast==i_uncoast ? astcols["astunif"] : i_ast<i_uncoast ? :lightsalmon : :salmon4,
                    i_ast in [1,N_ast,i_uncoast] ? @sprintf("𝐴%s𝐾−𝑀", i_ast==i_uncoast ? "=" : i_ast==1 ? "<" : ">") : nothing,
                    alpha=(i_ast==i_uncoast ? 0.5 : 0.25),
                   )
    end
    # Optimal unconditional advance split time (un-COAST)
    #draw_bands!(ax_N, ax_C, 
    #            Ns_anc_boot, mean_return_period+N_dsc*uncoast,
    #            losses_astunif_lomidhi[i_uncoast],     
    #            astcols["astunif"], @sprintf("𝐴=𝐾−𝑀"))
    # Validation 
    draw_bands!(ax_N, ax_C, 
                Ns_anc_boot_valid, mean_return_period, 
                losses_valid_lomidhi, astcols["valid"],  "SiMC"; alpha=0.5)
    # Ancestors only 
    draw_bands!(ax_N, ax_C,
               Ns_anc_boot, mean_return_period,
               losses_anconly_lomidhi, astcols["anconly"],  "No\nboosting")
    # XclEnt
    draw_bands!(ax_N, ax_C,
                Ns_anc_boot, mean_return_period + N_dsc*uncoast, 
                losses_coast_lomidhi, astcols["XclEnt"], "𝐴=XclEnt-\nCOAST"; alpha=0.5, linestyle=(:dash,:dense))
    Legend(lout[2,2], ax_N; theme_leg...)

    colsize!(lout, 1, Relative(4/5))
    colgap!(lout,1,0)
    rowgap!(lout,1,15)
    save(joinpath(figdir, "KL_vs_N.png"), fig)
    
    !(todo["moctail"]) && return

    # -----------------------------------------

    Rmax = maximum(Rs_peak_valid)

    bin_centers = vcat((bin_lower_edges[1:N_bin-1] .+ bin_lower_edges[2:N_bin])./2, (bin_lower_edges[N_bin]+1)/2)
    bin_edges = vcat(bin_lower_edges, 1.0)
    xlimits = [minimum(filter(ispos, isnothing(ccdf_peak_wholetruth) ? ccdf_peak_valid : ccdf_peak_wholetruth))/2, 1]
    i_coast_mean = round(Int64,mean(idx_coast))
    # which bootstrap to use
    i_boot_size = div(length(Ns_anc_boot),2)
    (ccdfs_anconly_boot_lomidhi,
     ccdfs_moctail_coast_boot_lomidhi,
    ) = map(
            ccdfs->map(
                       qq->mapslices(
                                     cc->finitequantile(
                                                        filter(isfinite,cc),qq
                                                       ),
                                     ccdfs[:,:,i_boot_size]; dims=2
                                    )[:,1],
                       0.5 .+ confint_width.*[-1/2,0,1/2]
                      ), 
            (ccdfs_anconly_boot,ccdfs_moctail_coast_boot)
           )
    (ccdfs_moctail_astunif_boot_lomidhi
    ) = map(
            qq->mapslices(
                          cc->finitequantile(
                                             filter(isfinite,cc),qq
                                            ),
                          ccdfs_moctail_astunif_boot[:,:,:,i_boot_size]; dims=3
                         )[:,:,1],
            0.5 .+ confint_width.*[-1/2,0,1/2]
           )

    ylo,ymid,yhi = let
        ylohi = bin_edges[[i_bin_thresh,N_bin]]
        ylohi .= nlg1m.(ylohi)
        ymid = mean(ylohi)
        padding = 0.1
        ylohi .= (1+padding) .* ylohi .- padding*ymid
        ylo = floor(Int64,ylohi[1])
        yhi = ceil(Int64,ylohi[2])
        ymid = div(ylo+yhi,2)
        nlg1m_inv.((ylo,ymid,yhi))
    end
    ylimits = (ylo,yhi)
    ytickvalues = unique([ylo,ymid,yhi])
    yticklabels = scinot2near1.(ytickvalues) 

    theme_ax = (; 
                xgridvisible=false, ygridvisible=false, 
                xticklabelsize=14, yticklabelsize=14, 
                xlabelsize=16, ylabelsize=16, titlesize=16,
                xlabelfont="Menlo", ylabelfont="Menlo", titlefont="Menlo",
                xticklabelfont="Menlo", yticklabelfont="Menlo", 
                titlealign=:left
               )
    theme_leg = (; labelfont="Menlo",  titlefont="Menlo", framevisible=false)
    astcols = astcolors()


    fig = Figure(size=(80*(N_ast+1), 800))

    # ---------- Row 1: CCDFs at each AST separately ----------

    ccdf_todivideby = ccdf_peak_wholetruth 

    lout = fig[1,1] = GridLayout()
    for i_ast = 1:N_ast
        i_col = N_ast-i_ast+1
        ax = Axis(lout[1,i_col]; theme_ax..., xscale=log10, yscale=nlg1m, limits=(tuple(xlimits...), tuple(ylimits...)), title="Tail CCDFs", titlevisible=false, yticklabelsvisible=(i_col==1), ylabelvisible=(i_col==1), yticks=(ytickvalues,yticklabels), yticklabelrotation=0, ylabel="Peak $(statesymbol)*")
        lines!(ax, ccdf_peak_anc, bin_edges[i_bin_thresh:N_bin]; color=astcols["anconly"], linewidth=2, label="No boosting")
        finite_idx = findall(isfinite.(ccdfs_anconly_boot_lomidhi[1]) .& isfinite.(ccdfs_anconly_boot_lomidhi[3]))
        band!(ax, 
              Point2f.(ccdfs_anconly_boot_lomidhi[1][finite_idx], bin_edges[i_bin_thresh:N_bin][finite_idx]), 
              Point2f.(ccdfs_anconly_boot_lomidhi[3][finite_idx], bin_edges[i_bin_thresh:N_bin][finite_idx]), 
              color=astcols["anconly"], alpha=0.25)
        lines!(ax, ccdfs_moctail_astunif[:,i_ast], bin_edges[i_bin_thresh:N_bin]; color=astcols["astunif"], linewidth=1, label="Uniform-AST")
        finite_idx = findall(isfinite.(ccdfs_moctail_astunif_boot_lomidhi[1][:,i_ast]) .& isfinite.(ccdfs_moctail_astunif_boot_lomidhi[3][:,i_ast]))
        band!(ax,
              Point2f.(ccdfs_moctail_astunif_boot_lomidhi[1][finite_idx,i_ast], bin_edges[i_bin_thresh:N_bin][finite_idx]),
              Point2f.(ccdfs_moctail_astunif_boot_lomidhi[3][finite_idx,i_ast], bin_edges[i_bin_thresh:N_bin][finite_idx]),
              color=astcols["astunif"], alpha=0.25)
        if i_ast == i_coast_mean
            lines!(ax, ccdf_moctail_coast, bin_edges[i_bin_thresh:N_bin]; color=astcols["XclEnt"], linewidth=2, label="XclEnt-COAST")
            finite_idx = findall(isfinite.(ccdfs_moctail_coast_boot_lomidhi[1]) .& isfinite.(ccdfs_moctail_coast_boot_lomidhi[3]))
            band!(ax, 
                  Point2f.(ccdfs_moctail_coast_boot_lomidhi[1][finite_idx], bin_edges[i_bin_thresh:N_bin][finite_idx]), 
                  Point2f.(ccdfs_moctail_coast_boot_lomidhi[3][finite_idx], bin_edges[i_bin_thresh:N_bin][finite_idx]), 
                  color=astcols["XclEnt"], alpha=0.25
                 )
        end
        scatterlines!(ax, (isnothing(ccdf_peak_wholetruth) ? ccdf_peak_valid : ccdf_peak_wholetruth), bin_edges[i_bin_thresh:N_bin]; color=:black, linewidth=1.5, linestyle=(:dash,:dense), label=(isnothing(ccdf_peak_wholetruth) ? "Ground truth" : "Whole truth"), marker=:cross, markersize=6)
        if i_ast == i_coast_mean
            legtitle = @sprintf("𝑁 = %d ancestors\nmedians & %d%% CIs\n(%d-ancestor bootstraps)", N_anc, round(Int,confint_width*100), Ns_anc_boot[i_boot_size], )
            Legend(lout[1,N_ast+1], ax, legtitle; theme_leg..., )
        end
        if i_col == 1
            log2xlims = [ceil(Int64,log2(xlimits[1])),floor(Int64,log2(xlimits[2]))]
            log2xtickvals = [log2xlims[1],div(log2xlims[1]+log2xlims[2],2),log2xlims[2]]
            ax.xticks = (2.0 .^ log2xtickvals, (lol->@sprintf("2%s",supscr(lol))).(log2xtickvals))
            ax.xticklabelsvisible = ax.xlabelvisible = ax.titlevisible = true
        else
            ax.xticklabelsvisible = ax.xlabelvisible = ax.titlevisible = false
        end
        i_ast>1 && colgap!(lout, i_col, 0)
    end

    # ----------- Rows 2-3: thrent and COAST frequency ------------
    ax = Axis(lout[2,1:N_ast]; theme_ax..., xlabel="−AST", title="Extreme-conditional entropy (XclEnt)", ylabel="[bits]", ylabelrotation=pi/2, xgridvisible=false, ygridvisible=false, xticks=(-asts, string.(-asts)), yticks=round.(Int64, [0,1/2,1].*log2(N_dsc)), limits=((-(1.5*asts[end]-0.5*asts[end-1]), -(1.5*asts[1]-0.5*asts[2])),(-1,round(Int64,log2(N_dsc)+1))), xlabelvisible=false, xticklabelsvisible=false)
    hlines!(ax, [0,round(Int64,log2(N_dsc))]; color=:grey79)
    for i_anc = 1:N_anc
        scatterlines!(ax, -asts, extreme_conditional_entropy[:,i_anc]; color=astcols["XclEntOne"], label=(i_anc==1 ? "Single-family" : nothing))
    end
    scatterlines!(ax, -asts, mean(extreme_conditional_entropy; dims=2)[:,1]; color=astcols["XclEnt"], linewidth=3, label="Multi-family\nmean")
    Legend(lout[2,N_ast+1], ax; theme_leg...,)

    coast_freq = zeros(Int64, N_ast)
    for i_ast = 1:N_ast
        coast_freq[i_ast] += sum(idx_coast.==i_ast)
    end
    ax = Axis(lout[3,1:N_ast]; theme_ax..., xlabel="−AST", ylabel="Number of\nfamilies", title="XclEnt-COAST frequencies", ylabelrotation=pi/2, xgridvisible=false, ygridvisible=false, xticks=(-asts, string.(-asts)), xlabelvisible=false, xticklabelsvisible=false, limits=((-(1.5*asts[end]-0.5*asts[end-1]), -(1.5*asts[1]-0.5*asts[2])),(-0.01*N_anc,1.35*maximum(coast_freq))))
    stairs!(ax, -asts, coast_freq, color=astcols["XclEnt"], linewidth=3, step=:center)
    scatter!(ax, -(perturbation_neglog-threshold_neglog), 1.1*maximum(coast_freq); marker=:star6, color=astcols["astunif"], markersize=18)
    text!(ax, -(perturbation_neglog-threshold_neglog), 1.15*maximum(coast_freq); text=@sprintf("𝐴 = 𝐾 − 𝑀 = %d − %d = %d",perturbation_neglog,threshold_neglog,perturbation_neglog-threshold_neglog), align=(:center,:bottom), font="Menlo")
    #Legend(lout[3,N_ast+1], ax; framevisible=false)

    # ------------ Row 4:  KL divergence ------
    # compute y-axis limits 
    loss_min_astunif,loss_max_astunif = (func(filter(isfinite, losses_moctail_astunif_kldiv_boot[:,:,i_boot_size])) for func=(minimum,maximum))
    #yhi = 2.0 * maximum(filter(isnonneg, [loss_max_astunif, loss_moctail_coast_kldiv]))
    #klposlo = minimum(filter(ispos, vcat(losses_moctail_astunif_kldiv_boot[:,:,i_boot_size][:], losses_moctail_coast_kldiv_boot[:,i_boot_size])))

    klposlo = -xlogx(1/N_dsc)
    ylo = -klposlo/2
    yhi = round(log2(N_bin))
    ytickvalues = [0,klposlo,1,yhi] #[0,sqrt(klposlo*yhi),yhi]
    yticklabels = (y->@sprintf("%.1f",y)).(ytickvalues)
    ax = Axis(lout[4,1:N_ast]; theme_ax..., 
              xlabel="−AST", title="KL divergence", yscale=Makie.Symlog10(klposlo),
              ylabel="[bits]", ylabelrotation=pi/2, xgridvisible=false, ygridvisible=false, 
              xticks=(-asts, string.(-asts)), 
              limits=((-(1.5*asts[end]-0.5*asts[end-1]), -(1.5*asts[1]-0.5*asts[2])), (ylo,yhi)),
              yticks=(ytickvalues,yticklabels),
             )
    klunif_lomidhi = map(
                         qq->mapslices(
                                       klarr->finitequantile(filter(isfinite,klarr), qq),
                                       losses_moctail_astunif_kldiv_boot[:,:,i_boot_size];
                                       dims=2
                                      )[:,1],
                         1/2 .+ confint_width.*[-1/2,0,1/2]
                        )
    klcoast_lomidhi = map(
                          qq->finitequantile(filter(isfinite,losses_moctail_coast_kldiv_boot[:,i_boot_size]), qq), 
                          1/2 .+ confint_width.*[-1/2,0,1/2]
                         )
    band!(ax, -asts, klunif_lomidhi[[1,3]]...; color=astcols["astunif"], alpha=0.25)
    klclip(kl) = (kl < -1/(2^32) ? kl : max(0, kl))
    scatterlines!(ax, -asts, klunif_lomidhi[2]; color=astcols["astunif"], label=@sprintf("Uniform AST\nKL ≥ %.1E",klclip(minimum(klunif_lomidhi[2]))))
    band!(ax, -asts, (klcoast_lomidhi[i_qq].*ones(N_ast) for i_qq=[1,3])...; color=astcols["XclEnt"], alpha=0.25)
    lines!(ax, -asts, klcoast_lomidhi[2].*ones(N_ast), color=astcols["XclEnt"], linestyle=:solid, linewidth=2, label=@sprintf("XclEnt-COAST\nKL = %.1E", klclip(klcoast_lomidhi[2]))) #loss_moctail_coast_kldiv))
    Legend(lout[4,N_ast+1], ax; theme_leg...,)

    rowgap!(lout, 1, 0)
    rowgap!(lout, 2, 0)
    rowgap!(lout, 3, 0)
    colsize!(lout, N_ast+1, Relative(1/5))
    
    save(joinpath(figdir, "ccdfs_moctail_N$(Ns_anc_boot[i_boot_size]).png"), fig)


    return 
end

function find_peaks_over_threshold(
        thresh::Float64, 
        duration_spinup::Int64, 
        duration_spinon::Int64, 
        min_cluster_gap::Int64, 
        datadir::String, 
        file_suffix::String
    )

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


function plot_dns(duration_spinup::Int64, duration_spinon::Int64, datadir::String, figdir::String, outfile_suffix::String; edges::ornot(Vector{Float64})=nothing, pdf_wholetruth::ornot(Vector{Float64})=nothing, statesymbol::String="𝑥")
    xs,ts = jldopen(joinpath(datadir, "dns_$(outfile_suffix).jld2"), "r") do f
        return f["xs"],f["ts"]
    end
    t0 = duration_spinup
    t1 = t0 + duration_spinon
    if isnothing(edges)
        edges = collect(range(0,1;length=33))
    end
    cdf = mean(xs[1,t0:t1] .<= edges'; dims=1)[1,:]
    pdf = diff(cdf) ./ diff(edges)
    bincenters = (edges[1:end-1] .+ edges[2:end])./2

    Nt2plot = 128
    fig = Figure(size=(600,150))
    lout = fig[1,1] = GridLayout()
    theme_ax,theme_leg = get_themes()
    ax_ts = Axis(lout[1,1]; xlabel="𝑡", ylabel=statesymbol) #, theme_ax...)
    #scatterlines!(ax_ts, 0:0.1:2pi, sin.(0:0.1:2pi); color="black")
    ylims!(ax_ts, -1, 1)
    ax_hist = Axis(lout[1,2]; xlabel="𝑝($statesymbol)", ylabel=statesymbol, ylabelvisible=false, yticklabelsvisible=false, theme_ax...)

    scatterlines!(ax_ts, ts[t0:t0+Nt2plot], xs[1,t0:t0+Nt2plot]; color=:black)
    xlims!(ax_ts, ts[t0], ts[t0+Nt2plot])
    ylims!(ax_ts, 0, 1)
    xlims!(ax_hist, 0, 2)
    ylims!(ax_hist, 0, 1)
    scatterlines!(ax_hist, pdf, bincenters; color=:steelblue2, markersize=2)
    if !isnothing(pdf_wholetruth)
        lines!(ax_hist, pdf_wholetruth, bincenters; color=:black, linestyle=(:dash,:dense), linewidth=3, label="WholeTruth")
    end
    

    colsize!(lout, 2, Relative(1/6))
    colgap!(lout, 1, 0)


    save(joinpath(figdir, "dns_timeseries_hist_$(outfile_suffix).png"), fig)
end

function mix_conditional_tails(
        datadir::String, asts::Vector{Int64}, N_dsc::Int64, bst::Int64, bin_lower_edges::Vector{Float64}, i_bin_thresh::Int64, rngseed_boot::Int64;
        ccdf_peak_wholetruth::Union{Nothing,Vector{Float64}}=nothing, accrej::Bool=false
    )
    # TODO add a dimension for number of ancestors to mix, and bootstratpps for UQ
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

    # TODO extend all these result structures to have bootstraps with varying amounts of data held out. Maybe halving successively until none.  

    Rmax = maximum(Rs_peak_valid)
    N_bin = length(bin_lower_edges)
    N_bin_over = N_bin - i_bin_thresh + 1 # number of bins in the tail


    # Put results into a big array: (i_dsc, i_ast, i_anc) ∈ [N_dsc] x [N_ast] x [N_anc]
    N_ast = length(asts)
    N_anc = length(ts_peak)
    N_anc_valid = length(ts_peak_valid)

    ccdf_peak_anc = compute_empirical_ccdf(Rs_peak_anc, bin_lower_edges[i_bin_thresh:N_bin])
    ccdf_peak_valid = compute_empirical_ccdf(Rs_peak_valid, bin_lower_edges[i_bin_thresh:N_bin])

    Rs_peak_ancanddsc = zeros(Float64, (N_dsc+1, N_ast, N_anc))
    println("About to open xs_dsc for reading...")
    jldopen(joinpath(datadir,"xs_dscs.jld2"), "r") do f
        for i_anc = 1:N_anc
            Rs_peak_ancanddsc[1,:,:] .= Rs_peak_anc[i_anc]
            for i_ast = 1:N_ast
                for i_dsc = 1:N_dsc
                    if !("ianc$(i_anc)" in keys(f))
                        @show i_anc,i_ast,i_dsc
                        @show keys(f)
                        error()
                    end
                    Rs_dsc = intensity(f[joinpath("ianc$(i_anc)", "iast$(i_ast)", "idsc$(i_dsc)","xs")])
                    # full maximum
                    #Rs_peak_dsc[i_dsc,i_ast,i_anc] = maximum(Rs_dsc)
                    # at same point 
                    Rs_peak_ancanddsc[1+i_dsc,i_ast,i_anc] = Rs_dsc[asts[i_ast]]
                end
            end
        end
    end
    println("...finished reading xs_dscs")


    # Keep track of which AST is best for each ancestor, according to XclEnt 
    idx_coast = zeros(Int64, N_anc)

    # Initialize conditional and mixed CCDFs, both full (including under threshold) and rectified (with accept reject)
    ccdfs_dsc = zeros(Float64, (N_bin,N_ast,N_anc)) # no rectifying
    ccdfs_ctail = zeros(Float64, (N_bin_over,N_ast,N_anc)) # either with accrej or not 


    thresholded_entropy = zeros(Float64, (N_ast, N_anc))
    extreme_conditional_entropy = fill(NaN, (N_ast, N_anc))
    total_entropy = zeros(Float64, (N_ast, N_anc))

    anc_has_tail = zeros(Bool, (N_ast, N_anc))
    anc_counts_moctail_coast = 0
    ccdfs_dsc_tail_noaccrej = zeros(Float64,N_bin_over) # scratch space
    for i_anc = 1:N_anc
        @show i_anc,N_anc
        for i_ast = 1:N_ast
            # Stuff not pertainingto tails
            total_entropy[i_ast,i_anc] = compute_thresholded_entropy(Rs_peak_ancanddsc[2:end,i_ast,i_anc], bin_lower_edges)
            ccdfs_dsc[:,i_ast,i_anc] .= compute_empirical_ccdf(Rs_peak_ancanddsc[2:end,i_ast,i_anc], bin_lower_edges)
            # Stuff pertaining to tails
            anc_has_tail[i_ast,i_anc] = (maximum(Rs_peak_ancanddsc[2:end,i_ast,i_anc]) > bin_lower_edges[i_bin_thresh])

            ccdfs_dsc_tail_accrej = ccdfs_dsc[i_bin_thresh:N_bin,i_ast,i_anc] .+ (1-ccdfs_dsc[i_bin_thresh,i_ast,i_anc]).*(Rs_peak_anc[i_anc] .> bin_lower_edges[i_bin_thresh:N_bin])
            thresholded_entropy[i_ast,i_anc] = compute_thresholded_entropy(Rs_peak_ancanddsc[2:end,i_ast,i_anc], bin_lower_edges[i_bin_thresh:end])
            extreme_conditional_entropy[i_ast, i_anc] = compute_extreme_conditional_entropy(Rs_peak_ancanddsc[2:end,i_ast,i_anc], bin_lower_edges[i_bin_thresh:end]) 
            if anc_has_tail[i_ast,i_anc]
                ccdfs_dsc_tail_noaccrej .= ccdfs_dsc[i_bin_thresh:N_bin,i_ast,i_anc] ./ ccdfs_dsc[i_bin_thresh,i_ast,i_anc]
                extreme_conditional_entropy[i_ast, i_anc] /= ccdfs_dsc[i_bin_thresh,i_ast,i_anc]
            else
                ccdfs_dsc_tail_noaccrej .= 0 
            end
            ccdfs_ctail[:,i_ast,i_anc] .=      accrej ? ccdfs_dsc_tail_accrej : ccdfs_dsc_tail_noaccrej
        end
        # 
        # ------------ Maximize whatever the acquisition function is. These only work for moctail, which is computed separately per ancestor. ---------------
        #
        first_decrease = findfirst(diff(thresholded_entropy[:,i_anc]) .< 0) 
        argmax_thrent = argmax(thresholded_entropy[:,i_anc]) 
        argmax_xclent = argmax(extreme_conditional_entropy[:,i_anc])
        argmax_xclent_last = N_ast + 1 - argmax(reverse(extreme_conditional_entropy[:,i_anc]))
        argmin_xclent = argmin(extreme_conditional_entropy[:,i_anc])
        argmax_xclent_later = argmax(extreme_conditional_entropy[1:argmin_xclent,i_anc])
        idx_coast[i_anc] = argmax_xclent_last
        # 
        # ---------------------------------------------------------
    end
    # Now mix stuff together 

    # Uniform AST
    ccdfs_poptail_astunif = sum(ccdfs_dsc[i_bin_thresh:end,:,:].*insertdims(anc_has_tail; dims=1); dims=3)
    ccdfs_poptail_astunif ./= ccdfs_poptail_astunif[1:1,:,:]
    ccdfs_moctail_astunif = sum(
                                ccdfs_ctail 
                                .* insertdims(
                                              anc_has_tail./sum(anc_has_tail; dims=2);
                                              dims=1
                                             );
                                dims=3
                               ) 
    #ccdfs_moctail_astunif = sum(ccdfs_ctail .* insertdims(anc_has_tail; dims=1); dims=3) ./ insertdims(sum(anc_has_tail; dims=2); dims=1)
    # COAST 
    ccdf_moctail_coast,ccdf_poptail_coast = ntuple(_->zeros(Float64, (N_bin_over)), 2)
    for i_anc = 1:N_anc
        if anc_has_tail[idx_coast[i_anc],i_anc]
            ccdf_moctail_coast .+= ccdfs_ctail[:,idx_coast[i_anc],i_anc] 
            ccdf_poptail_coast .+= ccdfs_dsc[i_bin_thresh:end,idx_coast[i_anc],i_anc] #
            anc_counts_moctail_coast += 1
        end
    end
    ccdf_poptail_coast ./= ccdf_poptail_coast[1]
    ccdf_moctail_coast ./= anc_counts_moctail_coast

    # ------------------- Bootstrap resampling  -----------------
    println("About to start bootstrapping")
    Ns_anc_boot = unique(vcat(floor.(Int64, 2 .^ range(0, log2(N_anc); step=0.25)), N_anc))
    Ns_anc_boot_valid = unique(vcat(Ns_anc_boot, floor.(Int64, 2 .^ range(log2(N_anc)+0.25, log2(N_anc_valid); step=0.25))))
    N_boot_sizes = length(Ns_anc_boot)
    N_boot_sizes_valid = length(Ns_anc_boot_valid)
    N_boot_resamps = 500
    ccdfs_moctail_astunif_boot,ccdfs_poptail_astunif_boot = ntuple(_->zeros(Float64, (N_bin_over,N_ast,N_boot_resamps,N_boot_sizes)), 2)
    ccdfs_moctail_coast_boot,ccdfs_poptail_coast_boot = ntuple(_->zeros(Float64, (N_bin_over,N_boot_resamps,N_boot_sizes)), 2)
    ccdfs_anconly_boot = zeros(Float64, (N_bin_over,N_boot_resamps,N_boot_sizes))
    ccdfs_valid_boot = zeros(Float64, (N_bin_over,N_boot_resamps,N_boot_sizes_valid))
    rng = MersenneTwister(rngseed_boot)
    for (i_boot_size,N_anc_boot) in enumerate(Ns_anc_boot)
        @show i_boot_size,N_anc_boot
        for i_boot_resamp = 1:N_boot_resamps
            ancs_boot = rand(rng, 1:N_anc, N_anc_boot)
            anc_mults = sum(1:N_anc .== ancs_boot'; dims=2)[:,1]
            # uniform AST 
            anc_weights_astunif = 1.0 .* anc_has_tail .* insertdims(anc_mults; dims=1)
            anc_weights_astunif ./= sum(anc_weights_astunif; dims=2)
            ccdfs_moctail_astunif_boot[:,:,i_boot_resamp,i_boot_size] .= sum(ccdfs_ctail .* insertdims(anc_weights_astunif; dims=1); dims=3)
            ccdfs_poptail_astunif_boot[:,:,i_boot_resamp,i_boot_size] .= sum(ccdfs_dsc[i_bin_thresh:end,:,:] .* insertdims(anc_weights_astunif; dims=1); dims=3)
            ccdfs_poptail_astunif_boot[:,:,i_boot_resamp,i_boot_size] ./= ccdfs_poptail_astunif_boot[1:1,:,i_boot_resamp,i_boot_size]
            # COAST
            coast_total_weight = 0
            for i_anc = 1:N_anc
                anc_mult = anc_mults[i_anc] * anc_has_tail[idx_coast[i_anc],i_anc]
                coast_total_weight += anc_mult
                ccdfs_moctail_coast_boot[:,i_boot_resamp,i_boot_size] .+= ccdfs_ctail[:,idx_coast[i_anc],i_anc] * anc_mult
                ccdfs_poptail_coast_boot[:,i_boot_resamp,i_boot_size] .+= ccdfs_dsc[i_bin_thresh:end,idx_coast[i_anc],i_anc] * anc_mult
            end
            ccdfs_moctail_coast_boot[:,i_boot_resamp,i_boot_size] ./= coast_total_weight
            ccdfs_poptail_coast_boot[:,i_boot_resamp,i_boot_size] ./= ccdfs_poptail_coast_boot[1,i_boot_resamp,i_boot_size]
            # Ancestors only 
            ccdfs_anconly_boot[:,i_boot_resamp,i_boot_size] .= compute_empirical_ccdf(Rs_peak_anc[ancs_boot], bin_lower_edges[i_bin_thresh:end])
        end
    end
    # Now bootstrapping on the validation, which can extend to longer N_ancs
    for (i_boot_size,N_anc_boot) in enumerate(Ns_anc_boot_valid)
        for i_boot_resamp = 1:N_boot_resamps
            ancs_boot_valid = rand(rng, 1:N_anc_valid, N_anc_boot)
            ccdfs_valid_boot[:,i_boot_resamp,i_boot_size] .= compute_empirical_ccdf(Rs_peak_valid[ancs_boot_valid], bin_lower_edges[i_bin_thresh:end]) 
        end
    end

    # compute losses: with respect to the whole truth if it is available, but otherwise the ground truth 
    ccdf_peak_truth = (isnothing(ccdf_peak_wholetruth) ? ccdf_peak_valid : ccdf_peak_wholetruth)
    losses_moctail_astunif_hell,losses_moctail_astunif_chi2,losses_moctail_astunif_wass,losses_moctail_astunif_kldiv = (zeros(Float64, N_ast) for _=1:4)
    loss_moctail_coast_hell = hellingerdist(ccdf_peak_truth, ccdf_moctail_coast)
    loss_moctail_coast_chi2 = chi2div(ccdf_peak_truth, ccdf_moctail_coast)
    loss_moctail_coast_wass = wassersteindist(ccdf_peak_truth, ccdf_moctail_coast)
    loss_moctail_coast_kldiv = kldiv(ccdf_peak_truth, ccdf_moctail_coast)
    for i_ast = 1:N_ast
        losses_moctail_astunif_hell[i_ast] = hellingerdist(ccdf_peak_truth, ccdfs_moctail_astunif[:,i_ast])
        losses_moctail_astunif_chi2[i_ast] = chi2div(ccdf_peak_truth, ccdfs_moctail_astunif[:,i_ast])
        losses_moctail_astunif_wass[i_ast] = wassersteindist(ccdf_peak_truth, ccdfs_moctail_astunif[:,i_ast])
        losses_moctail_astunif_kldiv[i_ast] = kldiv(ccdf_peak_truth, ccdfs_moctail_astunif[:,i_ast])
    end

    losses_moctail_coast_kldiv_boot,losses_poptail_coast_kldiv_boot = ntuple(_->zeros(Float64, (N_boot_resamps, N_boot_sizes)), 2)
    losses_moctail_astunif_kldiv_boot,losses_poptail_astunif_kldiv_boot = ntuple(_->zeros(Float64, (N_ast,N_boot_resamps,N_boot_sizes)), 2)
    losses_anconly_kldiv_boot = zeros(Float64,(N_boot_resamps,N_boot_sizes))
    losses_valid_kldiv_boot = zeros(Float64,(N_boot_resamps,N_boot_sizes_valid))
    for (i_boot_size,N_anc_boot) in enumerate(Ns_anc_boot)
        for i_boot_resamp = 1:N_boot_resamps
            for i_ast = 1:N_ast
                losses_moctail_astunif_kldiv_boot[i_ast,i_boot_resamp,i_boot_size] = kldiv(ccdf_peak_truth, ccdfs_moctail_astunif_boot[:,i_ast,i_boot_resamp,i_boot_size])
                losses_poptail_astunif_kldiv_boot[i_ast,i_boot_resamp,i_boot_size] = kldiv(ccdf_peak_truth, ccdfs_poptail_astunif_boot[:,i_ast,i_boot_resamp,i_boot_size])
            end
            losses_moctail_coast_kldiv_boot[i_boot_resamp,i_boot_size] = kldiv(ccdf_peak_truth, ccdfs_moctail_coast_boot[:,i_boot_resamp,i_boot_size])
            losses_poptail_coast_kldiv_boot[i_boot_resamp,i_boot_size] = kldiv(ccdf_peak_truth, ccdfs_poptail_coast_boot[:,i_boot_resamp,i_boot_size])
            losses_anconly_kldiv_boot[i_boot_resamp,i_boot_size] = kldiv(ccdf_peak_truth, ccdfs_anconly_boot[:,i_boot_resamp,i_boot_size])
        end
    end
    for (i_boot_size,N_anc_boot) in enumerate(Ns_anc_boot_valid)
        for i_boot_resamp = 1:N_boot_resamps
            losses_valid_kldiv_boot[i_boot_resamp,i_boot_size] = kldiv(ccdf_peak_truth, ccdfs_valid_boot[:,i_boot_resamp,i_boot_size])
        end
    end

    # TODO compute costs, and make an equal-cost DNS estimator 

    # Save results to file 

    println("About to save boost_stats")
    Rs_peak_dsc = Rs_peak_ancanddsc[2:N_dsc+1,:,:]
    jldsave(joinpath(datadir,"boost_stats.jld2");
            (;
             ccdf_peak_anc,
             ccdf_peak_valid,
             Rs_peak_dsc,
             idx_coast,
             ccdfs_dsc,
             ccdfs_moctail_astunif,
             ccdf_moctail_coast,
             losses_moctail_astunif_hell,
             losses_moctail_astunif_chi2,
             losses_moctail_astunif_wass,
             losses_moctail_astunif_kldiv,
             loss_moctail_coast_hell,
             loss_moctail_coast_chi2,
             loss_moctail_coast_wass,
             loss_moctail_coast_kldiv,
             thresholded_entropy,
             extreme_conditional_entropy,
             total_entropy,
             # bootstraps
             Ns_anc_boot,
             Ns_anc_boot_valid,
             ccdfs_moctail_astunif_boot,
             ccdfs_moctail_coast_boot,
             ccdfs_anconly_boot,
             ccdfs_valid_boot,
             losses_moctail_astunif_kldiv_boot,
             losses_moctail_coast_kldiv_boot,
             losses_anconly_kldiv_boot,
             losses_valid_kldiv_boot,
            )...
           )
    return
end


