function plot_quantiles_latdep(tgrid::Vector{Int64}, f::Array{Float64,3}, sdm::SpaceDomain, outfile::String; flabel="")
    ps = [1e-4, 1e-3,1e-2,1e-1,0.25]
    ps = vcat(ps, [0.5], reverse(1 .- ps))
    dy = sdm.Ly / sdm.Ny
    qs = zeros(Float64, (sdm.Ny, length(ps)))
    qs_gaussian = zeros(Float64, (sdm.Ny, length(ps)))
    for (i_y,y) in enumerate(sdm.ygrid)
        data = vec(f[:,i_y,:])
        G = Dists.Normal(SB.mean(data), SB.std(data))
        for (i_p,p) in enumerate(ps)
            qs[i_y,i_p] = SB.quantile(vec(f[:,i_y,:]), p)
            qs_gaussian[i_y,i_p] = Dists.quantile(G, p)
        end
    end
    fig = Figure(size=(500,500))
    lout = fig[1:1,1:2] = GridLayout()
    ax1 = Axis(lout[1,1], xlabel="Quantile", ylabel="Latitude")
    ax2 = Axis(lout[1,2], xlabel="Quantile in Gaussian frame", ylabel="Latitude")
    qs_gaussian_latmean = SB.mean(qs_gaussian; dims=1)
    for (i_p,p) in enumerate(ps)
        lines!(ax1, qs[:,i_p], sdm.ygrid .+ dy/2, color=:red)
        lines!(ax1, qs_gaussian[:,i_p], sdm.ygrid .+ dy/2, color=:black, linestyle=:dash)
        lines!(ax2, qs[:,i_p] .- qs_gaussian[:,i_p] .+ qs_gaussian_latmean[1,i_p], sdm.ygrid .+ dy/2, color=:red)
        lines!(ax2, qs_gaussian_latmean[1,i_p]*ones(sdm.Ny), sdm.ygrid .+ dy/2, color=:black, linestyle=:dash)
    end
    save(outfile, fig)
    return
end



function plot_hovmoller(tgrid::Vector{Int64}, f::Array{Float64,4}, sdm::SpaceDomain, outfile::String; flabel="")
    fig = Figure()
    lout = fig[1:2,1:1] = GridLayout()
    fzm = SB.mean(f, dims=1)
    for iz = 1:2
        ax = Axis(lout[iz,1], xlabel="time", ylabel="y", title="$(flabel) zonal mean")
        image!(ax,(tgrid[1]*sdm.tu,(tgrid[end]+1)*sdm.tu),(0,1.0), fzm[1,:,iz,:]', colormap=:vik)
    end
    save(outfile,fig)
    return
end



function plot_fluctuations(tgrid::Vector{Int64}, f::Array{Float64,3}, sdm::SpaceDomain, outfile::String; flabel="")
    numlats = 4
    dx = sdm.Lx/sdm.Nx
    dy = sdm.Ly/sdm.Ny
    idx_y = round.(Int, (range(0, sdm.Ly, length=numlats+1)[1:end-1] .+ sdm.Ly/(2*numlats))/dy)
    numlons = 1
    idx_x = round.(Int, (range(0, sdm.Lx, length=numlons+1)[1:end-1] .+ sdm.Lx/(2*numlats))/dx)
    @show idx_y,idx_x
    fig = Figure(size=(1000,2000))
    lout = fig[1:numlats,1:2] = GridLayout()
    tidx_timeseries = findfirst(tgrid*sdm.tu .> tgrid[end]*sdm.tu-400):1:length(tgrid)
    axs_timeseries = []
    axs_hists = []
    for (i_y,i_lat) in enumerate(idx_y)
        y_str = Printf.@sprintf("%.1f", sdm.ygrid[i_lat] + dy/2)
        ax = Axis(lout[numlats-i_y+1,1], xlabel="time", ylabel=flabel, title=L"$c(y=%$(y_str),t)$")
        push!(axs_timeseries, ax)
        for (i_x,i_lon) in enumerate(idx_x)
            x_str = Printf.@sprintf("%.1f", sdm.xgrid[i_lon] + dx/2)
            lines!(ax, tgrid[tidx_timeseries]*sdm.tu, f[i_x,i_y,tidx_timeseries])
        end
        ax = Axis(lout[numlats-i_y+1,2], xlabel="time", title=L"PDF of $c(y=%$(y_str))$", yscale=log10)
        push!(axs_hists, ax)
        data = vec(f[:,i_y,:])
        edges = collect(range(minimum(data), maximum(data)+1e-10, length=20))
        centers = (edges[1:end-1]+edges[2:end])/2
        h = SB.normalize(SB.fit(SB.Histogram, data, edges; closed=:left); mode=:pdf)
        @show h.weights
        lines!(ax, centers, replace(h.weights, 0=>NaN), color=:black)
        # Plot a Gaussian with same mean and variance
        gaussian_pdf = Dists.pdf.(Dists.Normal(SB.mean(data),SB.std(data)), centers)
        @show gaussian_pdf
        lines!(ax, centers, gaussian_pdf, color=:black, linestyle=:dash)
    end
    for i_lat = 1:numlats
        linkyaxes!(axs_timeseries[i_lat], axs_timeseries[1])
        linkyaxes!(axs_hists[i_lat], axs_hists[1])
    end
    save(outfile, fig)
    return
end

function animate_fields_contour_divergence(
        tgrid::Vector{Int64}, 
        fconts::Vector{Array{Float64,3}},
        # TODO put a heatmap on the same image 
        outfile::String,
        sdm::SpaceDomain,
        # where to draw a box 
        xcenter::Float64,
        xradius::Float64,
        ycenter::Float64,
        yradius::Float64,
        ;
        title = nothing,
        colors_cont = nothing,
        colors_objoft = nothing,
    )
    fig = Figure(size=(300+30,300+100))
    lout = fig[1:2,1:1] = GridLayout()
    if isnothing(title)
        title = ""
    end
    Nmem_cont = length(fconts)
    if isnothing(colors_cont)
        colors_cont = vcat([:gray for _=1:Nmem_cont-2], [:red, :black])
    end
    ax_cont = Axis(lout[1,1], xlabel="x", ylabel="y", title=title)
    xlims!(ax_cont, [0,sdm.Lx])
    ylims!(ax_cont, [0,sdm.Ly])

    # Plot timeseries of the objective function underneath 
    ax_locavg = Axis(lout[2,1], xlabel="Time") 
    for i_mem = 1:Nmem_cont
        loc_avg = vec(horz_avg(fconts[i_mem], sdm, xcenter, xradius, ycenter, yradius))
        lines!(ax_locavg, tgrid.*sdm.tu, loc_avg, color=colors_cont[i_mem])
    end

    # Mark the region where the score is calculated
    locavg_rect = poly!(ax_cont, [(xcenter + sgnx*xradius)*sdm.Lx for sgnx=[-1,1,1,-1]], [(ycenter + sgny*yradius)*sdm.Ly for sgny=[-1,-1,1,1]], color=:cyan, alpha=0.5)

    rowsize!(lout, 1, Relative(5/8))
    rowsize!(lout, 2, Relative(3/8))

    valrange_cont = maximum([maximum(abs.(fcont)) for fcont=fconts]) .* [-1,1]

    record(fig, outfile, framerate=12) do io
        for (i_t,t) in enumerate(tgrid .* sdm.tu)
            objs = []
            ax = ax_cont
            #img = image!(ax, (0,sdm.Lx), (0,sdm.Ly), fconts[1][:,:,i_t], colormap=:bam, colorrange=valrange_cont)
            #push!(objs, (ax,img))
            levneg = collect(range(valrange_cont[1], 0, length=7)[1:end-1])
            levpos = collect(range(0, valrange_cont[2], length=7)[2:end])
            for i_mem = 1:Nmem_cont
                contneg = contour!(ax, sdm.xgrid, sdm.ygrid, fconts[i_mem][:,:,i_t],levels=levneg,color=colors_cont[i_mem],linestyle=:dash)
                push!(objs, (ax,contneg))
                contpos = contour!(ax, sdm.xgrid, sdm.ygrid, fconts[i_mem][:,:,i_t],levels=levpos,color=colors_cont[i_mem])
                push!(objs, (ax,contpos))
            end
            #locmarker = scatter!(ax, [sdm.Lx/2], [sdm.Ly/2], marker=:cross, color=:black)
            #push!(objs, (ax,locmarker))

            v = vlines!(ax_locavg, [t], color=:black)
            push!(objs, (ax_locavg, v))
            recordframe!(io)
            for obj in objs
                delete!(obj...)
            end
        end
    end
end

function animate_fields_pair_divergence(
        tgrid::Vector{Int64}, 
        fcont1::Array{Float64,3}, 
        fcont2::Array{Float64,3}, 
        fheat1::Array{Float64,3},
        fheat2::Array{Float64,3},
        obj1::Vector{Float64},
        obj2::Vector{Float64},
        outfile::String,
        sdm::SpaceDomain,
        ;
        titles = nothing,
        global_colorscale=true
    )
    fig = Figure(size=(500*3+50,500+100+300))
    lout = fig[1:3,1:3] = GridLayout()
    if isnothing(titles)
        titles = ["" for _=1:3]
    end
    axs_img = [Axis(lout[1,i_col], xlabel="x", ylabel="y", title=titles[i_col]) for i_col=1:3]
    #ax_cont_overlay = Axis(lout[1,4], xlabel="x", ylabel="y", title="both")
    axs_cbar = [Axis(lout[2,i_col]) for i_col=1:3]

    # Plot timeseries of the objective function underneath 
    ax_objoft = Axis(lout[3,1:3], xlabel="Time") 
    lines!(ax_objoft, tgrid.*sdm.tu, obj1, color=:black)
    lines!(ax_objoft, tgrid.*sdm.tu, obj2, color=:red)

    rowsize!(lout, 1, Relative(5/9))
    rowsize!(lout, 2, Relative(1/9))
    rowsize!(lout, 3, Relative(3/9))

    valrange_cont = zeros((2,3))
    valrange_heat = zeros((2,3))

    fconts = [fcont1, fcont2, fcont2 .- fcont1]
    fheats = [fheat1, fheat2, fheat2 .- fheat1]
    for i_col = 1:3
        valrange_cont[:,i_col] .= maximum(abs.(fconts[i_col])) .* [-1,1]
        valrange_heat[:,i_col] .= maximum(abs.(fheats[i_col])) .* [-1,1]
    end


    record(fig, outfile, framerate=12) do io
        for (i_t,t) in enumerate(tgrid .* sdm.tu)
            @show i_t,length(tgrid)
            objs = []
            for i_ax = 1:3
                ax = axs_img[i_ax]
                colorrange_args = i_ax == 3 ? Dict() : Dict(:colorrange=>valrange_heat[:,i_ax])
                img = image!(ax, (0,sdm.Lx), (0,sdm.Ly), fheats[i_ax][:,:,i_t]; colormap=:vik, colorrange_args...)
                push!(objs, (ax,img))
                cbar = Colorbar(lout[2,i_ax], img, vertical=false)
                push!(objs, (cbar,))
                levneg = collect(range(valrange_cont[1,i_ax], 0, length=7)[1:end-1])
                levpos = collect(range(0, valrange_cont[2,i_ax], length=7)[2:end])
                contneg = contour!(ax, sdm.xgrid, sdm.ygrid, fconts[i_ax][:,:,i_t],levels=levneg,color=:black,linestyle=:dash)
                push!(objs, (ax,contneg))
                contpos = contour!(ax, sdm.xgrid, sdm.ygrid, fconts[i_ax][:,:,i_t],levels=levpos,color=:black)
                push!(objs, (ax,contpos))
            end

            v = vlines!(ax_objoft, [t], color=:black)
            push!(objs, (ax_objoft, v))
            recordframe!(io)
            #println("Recorded frame")
            for obj in objs
                delete!(obj...)
            end
        end
    end
end

function animate_fields(tgrid::Vector{Int64}, fcont::Array{Float64,4}, fheat::Array{Float64,4}, sdm::SpaceDomain, outfile::String; fcont_label="", fheat_label="", titles=nothing, global_colorscale=true)
    # Animate
    tidx_anim_step = max(1, round(Int, length(tgrid)/400))
    tidx = collect(1:tidx_anim_step:length(tgrid))
    @show size(tidx)


    figsize = (500 + 50, 1000)
    fig = Figure(size=figsize)
    lout = fig[1:2,1:2] = GridLayout()
    if isnothing(titles)
        titles = collect(L"Layer %$(iz) %$(fcont_label), %$(fheat_label)" for iz=1:2)
    end
    axes = collect(Axis(lout[iz,1], xlabel="x", ylabel="y", title=titles[iz]) for iz=1:2)
    axes_cb = collect(Axis(lout[iz,2]) for iz=1:2)
    colsize!(lout, 1, Aspect(1, 1.0))
    colsize!(lout, 2, Relative(50/figsize[1]))
    for ax in axes_cb
        hidedecorations!(ax)
        hidespines!(ax)
    end
    for iz = 1:2
        xlims!(axes[iz], (0,sdm.Lx))
        ylims!(axes[iz], (0,sdm.Ly))
    end
    fcont_max_glob = vec(maximum(abs.(fcont[:,:,:,tidx]), dims=[1,2,4]))
    fheat_max_glob = vec(maximum(abs.(fheat[:,:,:,tidx]), dims=[1,2,4]))

    # Decide the colormap and color ranges
    fheat_posdef = (minimum(fheat) >= 0)
    fheat_negdef = (maximum(fheat) <= 0)
    if fheat_posdef
        colormap = :lipari
    elseif fheat_negdef
        colormap = Reverse(:lipari)
    else
        colormap = :vik
    end
    record(fig, outfile, framerate=12) do io
        for (i_snap,i_t) in enumerate(tidx)
            @show i_snap, length(tidx)
            objs = []
            fcont_max_loc = vec(maximum(abs.(fcont[:,:,:,i_t]), dims=[1,2]))
            fheat_max_loc = vec(maximum(abs.(fheat[:,:,:,i_t]), dims=[1,2]))
            fcont_max = global_colorscale ? fcont_max_glob : fcont_max_loc
            fheat_max = global_colorscale ? fheat_max_glob : fheat_max_loc
            for iz = 1:2
                ax = axes[iz]
                tstr = @sprintf("%.2f", tgrid[i_t]*sdm.tu)
                ax.title = L"%$(titles[iz]), t=%$(tstr)"
                img = image!(ax, (0,sdm.Lx), (0,sdm.Ly), fheat[:,:,iz,i_t], colormap=colormap, colorrange=(-fheat_max[iz]*(!fheat_posdef),fheat_max[iz]*(!fheat_negdef)),)
                push!(objs, (ax,img))
                ## colorbar 
                cbar = Colorbar(lout[iz,2], img, vertical=true)
                push!(objs, (cbar,))
                ## contours
                levneg = collect(range(-fcont_max[iz], 0, length=8)[1:end-1])
                levpos = collect(range(0, fcont_max[iz], length=8)[2:end])
                contneg = contour!(ax, sdm.xgrid, sdm.ygrid, fcont[:,:,iz,i_t],levels=levneg,color=:black,linestyle=:dash)
                push!(objs, (ax,contneg))
                contpos = contour!(ax, sdm.xgrid, sdm.ygrid, fcont[:,:,iz,i_t],levels=levpos,color=:black)
                push!(objs, (ax,contpos))
            end
            recordframe!(io)
            for obj in objs
                delete!(obj...)
            end
        end
    end
    return
end

function plot_snapshots(tgrid::Vector{Int64}, fheat::Array{Float64,4},fcont::Array{Float64,4},sdm::SpaceDomain, outfile_prefix::String; fcont_label="", fheat_label="", titles=nothing, scale_choice="global")
    figsize = (500 + 50, 1000)
    fig = Figure(size=figsize)
    lout = fig[1:2,1:2] = GridLayout()
    if isnothing(titles)
        titles = collect("Layer $(iz) $(fcont_label), $(fheat_label)" for iz=1:2)
    end
    axes = collect(Axis(lout[iz,1], xlabel="𝑥/𝐿", ylabel="𝑦/𝐿", title=titles[iz], titlesize=20, xlabelsize=20, ylabelsize=20, titlefont=:regular) for iz=1:2)
    axes_cb = collect(Axis(lout[iz,2]) for iz=1:2)
    colsize!(lout, 1, Aspect(1, 1.0))
    colsize!(lout, 2, Relative(50/figsize[1]))
    for ax in axes_cb
        hidedecorations!(ax)
        hidespines!(ax)
    end
    for iz = 1:2
        xlims!(axes[iz], (0,1)) #sdm.Lx))
        ylims!(axes[iz], (0,1)) #sdm.Ly))
    end
    fcont_max_glob = vec(maximum(abs.(fcont[:,:,:,:]), dims=[1,2,4]))
    fheat_max_glob = vec(maximum(abs.(fheat[:,:,:,:]), dims=[1,2,4]))

    # Decide the colormap and color ranges
    fheat_posdef = (minimum(fheat) >= 0)
    fheat_negdef = (maximum(fheat) <= 0)
    if fheat_posdef
        colormap = :lipari
    elseif fheat_negdef
        colormap = Reverse(:lipari)
    else
        colormap = :vik
    end
    global_colorscale = (scale_choice == "global")
    for (i_snap,t_snap) in enumerate(tgrid)
        objs = []
        fcont_max_loc = vec(maximum(abs.(fcont[:,:,:,i_snap]), dims=[1,2]))
        fheat_max_loc = vec(maximum(abs.(fheat[:,:,:,i_snap]), dims=[1,2]))
        fcont_max = global_colorscale ? fcont_max_glob : fcont_max_loc
        fheat_max = global_colorscale ? fheat_max_glob : fheat_max_loc
        for iz = 1:2
            ax = axes[iz]
            tstr = @sprintf("%.2f", t_snap*sdm.tu)
            ax.title = "$(titles[iz]), 𝑡 = $(tstr)"
            ax.titlefont = :regular
            img = image!(ax, (0,1), (0,1), fheat[:,:,iz,i_snap], colormap=colormap, colorrange=(-fheat_max[iz]*(!fheat_posdef),fheat_max[iz]*(!fheat_negdef)),)
            push!(objs, (ax,img))
            ## colorbar 
            cbar = Colorbar(lout[iz,2], img, vertical=true)
            push!(objs, (cbar,))
            ## contours
            levneg = collect(range(-fcont_max[iz], 0, length=8)[1:end-1])
            levpos = collect(range(0, fcont_max[iz], length=8)[2:end])
            contneg = contour!(ax, sdm.xgrid./sdm.Lx, sdm.ygrid./sdm.Ly, fcont[:,:,iz,i_snap],levels=levneg,color=:black,linestyle=:dash)
            push!(objs, (ax,contneg))
            contpos = contour!(ax, sdm.xgrid./sdm.Lx, sdm.ygrid./sdm.Ly, fcont[:,:,iz,i_snap],levels=levpos,color=:black)
            push!(objs, (ax,contpos))
        end
        save("$(outfile_prefix)_$(i_snap).png", fig)
        for obj in objs
            delete!(obj...)
        end
    end
    return 
end


function animate_fields_and_zonal_stats(tgrid::Vector{Int64}, fcont::Array{Float64,4}, fheat::Array{Float64,4}, sdm::SpaceDomain, outfile::String; fcont_label="", fheat_label="", titles=nothing, julia_old=false, scale_choice="global", plot_moments=true)
    # Animate
    duration = (tgrid[end] - tgrid[1])*sdm.tu
    if duration > 600
        tidxfirst = findfirst(tgrid*sdm.tu .> 300)
    else
        tidxfirst = findfirst(tgrid .> tgrid[end]-200/sdm.tu)
    end
    tidxlast = length(tgrid)
    tidx_stats = collect(tidxfirst:1:tidxlast)

    tidxfirst_anim = findfirst(tgrid .> tgrid[end]-100)
    tidx_anim_step = 1 #max(1, round(Int, (tidxlast - tidxfirst_anim)/400))
    tidx = collect(tidxfirst_anim:tidx_anim_step:tidxlast)
    @show tidx

    # Zonal moments
    Nmom = 4
    fheat_mom_x = zeros(Float64, (Nmom, sdm.Ny, 2, length(tidx)))
    fheat_mom_xt = zeros(Float64, (Nmom, sdm.Ny, 2, 1))

    # mean
    fheat_mom_x[1:1,:,:,:] .= sum(fheat[:,:,:,tidx], dims=1)/sdm.Nx
    fheat_mom_xt[1:1,:,:,1:1] .= sum(fheat[:,:,:,tidx_stats], dims=[1,4])/(sdm.Nx*length(tidx_stats))
    # standard deviation
    fheat_mom_x[2:2,:,:,:] .= sqrt.(sum((fheat[:,:,:,tidx] .- fheat_mom_xt[1:1,:,:,1:1]).^2, dims=1)/sdm.Nx)
    fheat_mom_xt[2:2,:,:,1:1] .= sqrt.(sum((fheat[:,:,:,tidx_stats] .- fheat_mom_xt[1:1,:,:,1:1]).^2, dims=[1,4])/(sdm.Nx*length(tidx_stats)))
    # For higher moments, standardize
    fheat_standardized = (fheat .- fheat_mom_xt[1:1,:,:,1:1])./fheat_mom_xt[2:2,:,:,1:1]
    @show size(fheat_standardized)
    @show size(fheat_mom_x)
    for i_mom = 3:Nmom
        fheat_mom_x[i_mom:i_mom,:,:,:] .= sum(fheat_standardized[:,:,:,tidx].^i_mom, dims=1) / sdm.Nx 
        fheat_mom_xt[i_mom:i_mom,:,:,1:1] .= sum(fheat_standardized[:,:,:,tidx_stats].^i_mom, dims=[1,4]) / (sdm.Nx * length(tidx_stats))
    end
    fheat_mom_xt[4,:,:,:] .-= 3
    mom_bounds = zeros(Float64, (2,Nmom,2))
    for i_mom = 1:Nmom
        for iz = 1:2
            mom_bounds[1,i_mom,iz] = minimum(fheat_mom_xt[i_mom,:,iz,:])
            mom_bounds[2,i_mom,iz] = maximum(fheat_mom_xt[i_mom,:,iz,:])
        end
    end
    println("mom_bounds = ")
    display(mom_bounds)


    #for i_mom = 1:Nmom
    #    heat_mom_x[i_mom:i_mom,:,:,:] .= sum(fheat[:,:,:,tidx].^i_mom, dims=1)/sdm.Nx
    #    fheat_mom_xt[i_mom:i_mom,:,:,1] .= sum(fheat[:,:,:,tidx_stats].^i_mom, dims=[1,4])/(sdm.Nx*length(tidx_stats))
    #end


    figsize = (500 + 50 + 150*Nmom, 1000)
    #ax = Axis(lout[1,3], xlabel=L"\overline{h}(y)", xticklabelrotation=pi/2)
    topo_zonal_mean = vec(SB.mean(cop.topography[:,:,2], dims=1))
    thlo,thhi = extrema(theta_zonal_mean)
    topolo,topohi = extrema(topo_zonal_mean)
    topo_rescaled = (thlo+thhi)/2 .+ (thhi-thlo)/(topohi-topolo) .* topo_zonal_mean
    lines!(ax, topo_rescaled, sdm.ygrid, color=:black)

    rowsize!(lout, 1, Relative(5/6))
    colsize!(lout, 1, Relative(2/3))
    colsize!(lout, 2, Relative(1/3))
    return
end

function plot_composite_extreme(obsoft::Vector{Float64}, thresh::Float64, lead_time_phys::Float64, follow_time_phys::Float64)
    # TODO
    return
end

function plot_hovmoller_ydep!(lout::GridLayout, fheat_nom::Array{Float64,2}, cop::ConstantOperators, sdm::SpaceDomain; title="", flabel="", tinit::Int64=0, anomaly::Bool=false)
    fheat_mean_xt = SB.mean(fheat_nom; dims=2)
    fheat = fheat_nom .- anomaly .* fheat_mean_xt
    @assert size(fheat,1) == sdm.Ny
    Nt = size(fheat,2)
    ax = Axis(lout[1,1], xlabel="𝑡", ylabel="𝑦/𝐿", titlefont=:regular)
    fheat_posdef = (minimum(fheat) >= 0)
    fheat_negdef = (maximum(fheat) <= 0)
    if fheat_posdef
        colormap = :lipari
    elseif fheat_negdef
        colormap = Reverse(:lipari)
    else
        colormap = :vik
    end
    img = image!(ax, (tinit*sdm.tu,(tinit+Nt-1)*sdm.tu), (0,sdm.Ly), fheat', colormap=colormap)
    cbar = Colorbar(lout[2,1], img, vertical=false)
    # zonal mean 
    # topography
    topo_zonal_mean = vec(SB.mean(cop.topography[:,:,2], dims=1))
    flo,fhi = extrema(fheat_mean_xt)
    topolo,topohi = extrema(topo_zonal_mean)
    topo_rescaled = (flo+fhi)/2 .+ (fhi-flo)/(topohi-topolo) .* topo_zonal_mean
    ax = Axis(lout[1,2], xlabel="Topo.")
    lines!(ax, topo_zonal_mean, sdm.ygrid, color=:black)
    lines!(ax, vec(fheat_mean_xt), sdm.ygrid, color=:red)

    colsize!(lout, 1, Relative(9/10))
    rowsize!(lout, 1, Relative(5/6))
    return
end

function plot_hovmoller_ydep!(lout::GridLayout, fheat_nom::Array{Float64,2}, fcont_nom::Array{Float64,2}, fheat_mssk::Array{Float64,2}, fcont_mssk::Array{Float64,2}, cop::ConstantOperators, sdm::SpaceDomain; title="", tinit::Int64=0, anomaly_heat::Bool=false, anomaly_cont::Bool=false)
    #fheat_mean_xt = SB.mean(fheat_nom; dims=2)
    fheat = fheat_nom .- anomaly_heat .* fheat_mssk[:,1]
    fcont = fcont_nom .- anomaly_cont .* fcont_mssk[:,1]
    @assert size(fcont,1) == sdm.Ny
    Nt = size(fheat,2)
    lblargs = Dict(:xticklabelsize=>30,:xlabelsize=>36,:yticklabelsize=>30,:ylabelsize=>36,:titlesize=>40,:xticklabelrotation=>0.0,:titlefont=>:regular)
    if anomaly_heat
        title = "$(title) anomaly"
        cbartickvals = maximum(abs.(fheat)) .* [-1,0,1]
        vmin,vmax = cbartickvals[[1,3]]
    else
        cbartickvals = collect(range(extrema(fheat)...; length=3))
        vmin,vmax = extrema(fheat)
    end
    ax = Axis(lout[1,2], xlabel="𝑡", ylabel="𝑦/𝐿"; lblargs..., title=title)
    fheat_posdef = (minimum(fheat) >= 0)
    fheat_negdef = (maximum(fheat) <= 0)
    if fheat_posdef
        colormap = :lipari
        color_cont = :cyan
    elseif fheat_negdef
        colormap = Reverse(:lipari)
        color_cont = :cyan
    else
        colormap = :BrBg
        color_cont = :black
    end
    img = image!(ax, (tinit*sdm.tu,(tinit+Nt-1)*sdm.tu), (0.0,1.0), fheat', colormap=colormap, colorrange=(vmin,vmax))

    fcont_posdef = (minimum(fcont) >= 0)
    fcont_negdef = (maximum(fcont) <= 0)
    if fcont_posdef
        levels_neg = [0]
        levels_pos = range(0, maximum(abs(fcont)), length=9)[2:end]
    elseif fheat_negdef
        levels_neg = range(-maximum(abs.(fcont)), 0, length=9)
        levels_pos = [0]
    else
        levels_neg = range(-maximum(abs.(fcont)), 0, length=5)[1:end-1]
        levels_pos = -reverse(levels_neg)
    end
    contneg = contour!(ax, range(tinit,tinit+Nt-1,step=1).*sdm.tu, sdm.ygrid./sdm.Ly, fcont', levels=levels_neg, linestyle=:dash, color=color_cont, linewidth=2)
    contpos = contour!(ax, range(tinit,tinit+Nt-1,step=1).*sdm.tu, sdm.ygrid./sdm.Ly, fcont', levels=levels_pos, linestyle=:solid, color=color_cont)
    cbarticklabs = (F->@sprintf("%+3.2f",F)).(cbartickvals)
    cbar = Colorbar(lout[1,1], img, vertical=true, labelsize=30, ticklabelsize=24, ticks=(cbartickvals,cbarticklabs))
    # zonal mean 
    # topography
    topo_zonal_mean = vec(SB.mean(cop.topography[:,:,2], dims=1))
    lblargs[:xticklabelrotation] = pi/2
    lblargs[:ylabelvisible] = lblargs[:yticklabelsvisible] = false
    xtickfun(f) = (maximum(f)-minimum(f) == 0 ? [-1.0,0.0,1.0] : range(minimum(f),maximum(f), 3))
    ax = Axis(lout[1,3]; title="Topo.", lblargs..., xticks=xtickfun(topo_zonal_mean))
    lines!(ax, topo_zonal_mean, sdm.ygrid, color=:black)
    for (i_mom,mom_name) = zip(1:1:4, ("Mean","Std. Dev.","Skew.","Kurt."))
        ax = Axis(lout[1,3+i_mom]; title=mom_name, lblargs..., xticks=xtickfun(fheat_mssk[:,i_mom]), xtickformat="{:.2f}")
        lines!(ax, fheat_mssk[:,i_mom], sdm.ygrid./sdm.Ly; color=:black)
        if i_mom == 4
            vlines!(ax, 3.0; color=:black, linestyle=:dash)
        end
    end

    colsize!(lout, 1, Relative(1/30))
    colsize!(lout, 2, Relative(1/2))
    colgap!(lout, 1, 0.0)
    colgap!(lout, 2, 100.0)
    for i_col = 3:6
        colgap!(lout, i_col, 10.0)
    end
    return
end



function plot_summary_statistic!(lout::GridLayout, theta::Matrix{Float64}, cop::ConstantOperators, sdm::SpaceDomain; theta_pivot=nothing, title="", theta_label=L"\theta")
    if isnothing(theta_pivot)
        colorrange = extrema(filter(isfinite,theta))
        colormap = :acton
    elseif theta_pivot >= maximum(theta)
        colorrange = [minimum(filter(isfinite, theta)), theta_pivot]
        colormap = :acton
    elseif theta_pivot <= minimum(theta)
        colorrange = [theta_pivot, maximum(filter(isfinite, theta))]
        colormap = :acton
    else
        colorrange = theta_pivot .+ maximum(filter(isfinite, abs.(theta .- theta_pivot))) .* [-1,1]
        colormap = :vik
    end
    @show colorrange
    ax = Axis(lout[1,1], xlabel="x", ylabel="y", title=title)
    img = image!(ax, (0,sdm.Lx), (0,sdm.Ly), theta; colormap=colormap, colorrange=colorrange)
    cbar = Colorbar(lout[2,1], img, vertical=false)
    # Zonal-mean statistic
    ax = Axis(lout[1,2], xlabel=L"\overline{%$(theta_label)},\overline{h}", xticklabelrotation=pi/2)
    theta_zonal_mean = vec(SB.mean(theta, dims=1))
    theta_zonal_std = vec(SB.std(theta, dims=1))
    lines!(ax, theta_zonal_mean, sdm.ygrid, color=:red)
    for sgn = [-1,1]
        lines!(ax, theta_zonal_mean .+ sgn.*theta_zonal_std, sdm.ygrid, color=:red, linestyle=:dot)
    end
    #ax = Axis(lout[1,3], xlabel=L"\overline{h}(y)", xticklabelrotation=pi/2)
    topo_zonal_mean = vec(SB.mean(cop.topography[:,:,2], dims=1))
    thlo,thhi = extrema(theta_zonal_mean)
    topolo,topohi = extrema(topo_zonal_mean)
    topo_rescaled = (thlo+thhi)/2 .+ (thhi-thlo)/(topohi-topolo) .* topo_zonal_mean
    lines!(ax, topo_rescaled, sdm.ygrid, color=:black)

    rowsize!(lout, 1, Relative(5/6))
    colsize!(lout, 1, Relative(2/3))
    colsize!(lout, 2, Relative(1/3))
    return
end
