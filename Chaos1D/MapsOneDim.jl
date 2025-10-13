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
    @show pdf_wholetruth
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


