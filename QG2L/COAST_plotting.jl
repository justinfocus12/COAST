function plot_objective_spaghetti(cfg, sdm, cop, pertop, ens, coast, i_anc, thresh, figdir)
    println("About to plot mom's spaghetti from ancestor $(i_anc)")
    t0 = coast.anc_tRmax[i_anc]
    t0ph = t0*sdm.tu
    t0str = @sprintf("%.0f", t0ph)
    ytgtstr = @sprintf("%.2f", cfg.target_yPerL*sdm.Ly)
    rxystr = @sprintf("%.3f", cfg.target_ryPerL*sdm.Ly)
    Rmin = minimum([minimum(coast.anc_Roft[i_anc]) for i_anc=1:length(coast.ancestors)]) 
    obj_label,short_obj_label = label_objective(cfg)

    # ------- Plot 0: ancestor only --------
    traj = ens.trajs[i_anc]
    t_anc = floor(Int, traj.tphinit/sdm.tu)+1:1:traj.tfin
    tph_anc = t_anc .* sdm.tu
    fig = Figure(size=(400,200))
    ax = Axis(fig[1,1],xlabel="𝑡-$(t0str)", ylabel="Intensity", xgridvisible=false, ygridvisible=false)
    hlines!(ax, thresh; color=:gray, alpha=0.25)
    lines!(ax, tph_anc.-t0ph, coast.anc_Roft[i_anc]; color=:black, linestyle=:dash, linewidth=1.5)
    Rmax_lims = [Rmin, maximum(coast.anc_Rmax)]
    ylims!(ax, Rmax_lims...)
    save(joinpath(figdir, "objective_anc$(i_anc)_nodesc.png"), fig)
    # ---------------------------------------
    #
    # -------- Plot 1: two panels: trajectories on top, score spreads on bottom -----
    anc = coast.ancestors[i_anc]
    lblargs = Dict(:xticklabelsize=>6,:xlabelsize=>9,:yticklabelsize=>6,:ylabelsize=>9,)
    descendants = Graphs.outneighbors(ens.famtree, anc)
    #pert_dim = length(pertop.camplitudes)
    fig = Figure(size=(400,300))
    lout = fig[1:2,1] = GridLayout()
    lblargs = Dict(:xlabelsize=>12, :ylabelsize=>12, :xticklabelsize=>10, :yticklabelsize=>10, :titlesize=>14)
    ax1 = Axis(lout[1,1]; xlabel="𝑡 − $(t0str)", ylabel="Intensity 𝑅(𝐱(𝑡))", title=label_target(cfg,sdm), xgridvisible=false, ygridvisible=false, xticklabelsvisible=false, xlabelvisible=false, titlefont=:regular, lblargs...)
    ax2 = Axis(lout[2,1]; xlabel="𝑡 − $(t0str)", ylabel="Severity 𝑅*", xgridvisible=false, ygridvisible=false, yticks=[minimum(coast.desc_Rmax[i_anc]), coast.anc_Rmax[i_anc]], ytickformat="{:.2f}", lblargs...)
    for ax = (ax1,ax2)
        hlines!(ax, thresh; color=:gray, alpha=0.25)
        hlines!(ax, coast.anc_Rmax[i_anc]; color=:black, linestyle=(:dash,:dense), linewidth=1.0)
        vlines!(ax, 0.0; color=:black, linestyle=(:dash,:dense), linewidth=1.0, alpha=1.0)
    end
    # First panel: just the timeseries
    #
    kwargs = Dict(:colormap=>:RdYlBu_4, :colorrange=>(cfg.lead_time_min,cfg.lead_time_max), :color=>1)
    leadtimes = collect(range(cfg.lead_time_min, cfg.lead_time_max; step=cfg.lead_time_inc))
    idx_leadtimes2plot = reverse(unique(clamp.(round.(Int, length(leadtimes).*[1/20, 2/5, 4/5]), 1, length(leadtimes))))
    spaghetti_leadtimes = leadtimes[reverse(idx_leadtimes2plot)]
    for (i_desc,desc) in enumerate(descendants)
        desc = descendants[i_desc]
        traj = ens.trajs[desc]
        kwargs[:color] = t0 - round(Int, coast.desc_tphpert[i_anc][i_desc]/sdm.tu)
        Nt = length(coast.desc_Roft[i_anc][i_desc])
        t_desc = (traj.tfin - Nt) .+ collect(1:1:Nt)
        tph_desc = t_desc .* sdm.tu
        leadtime = t0 - round(Int, coast.desc_tphpert[i_anc][i_desc]/sdm.tu) 
        itpert = argmin(abs.(t_desc.*sdm.tu .- coast.desc_tphpert[i_anc][i_desc]))
        if leadtime in spaghetti_leadtimes
            lines!(ax1, tph_desc[itpert:end] .- t0ph, coast.desc_Roft[i_anc][i_desc][itpert:end]; kwargs...)
            vlines!(ax1, coast.desc_tphpert[i_anc][i_desc]-t0ph; kwargs...)
        end
        scatter!(ax1, coast.desc_tphpert[i_anc][i_desc]-t0ph, coast.desc_Rmax[i_anc][i_desc]; kwargs..., markersize=6) 
        scatter!(ax2, coast.desc_tphpert[i_anc][i_desc]-t0ph, coast.anc_Rmax[i_anc]; kwargs..., markersize=8, label="Splits") 
        scatter!(ax2, coast.desc_tRmax[i_anc][i_desc]*sdm.tu-t0ph, coast.desc_Rmax[i_anc][i_desc]; kwargs..., markersize=8, marker=:star6, alpha=0.8, label="Peaks")
    end
    traj = ens.trajs[anc]
    t_anc = traj.tfin .+ collect(range(-length(coast.anc_Roft[i_anc])+1, 0; step=1)) 
    lines!(ax1, t_anc.*sdm.tu .- t0ph, coast.anc_Roft[i_anc]; color=:black, linestyle=(:dash,:dense), linewidth=1.5)
    axislegend(ax2; position=:lb, orientation=:horizontal, labelsize=10, merge=true, unique=true, framevisible=true, markercolor=:black, height=30)
    linkxaxes!(ax1,ax2)
    rowsize!(lout, 1, Relative(2/3))
    rowgap!(lout, 0.0)

    save(joinpath(figdir,"objectives_anc$(i_anc).png"), fig)
   
end

function plot_objective_response_linquad(
        cfg, sdm, cop, pertop, ens, coast, i_anc, 
        coefs_zernike, residmse_zernike, rsquared_zernike,
        coefs_linear, residmse_linear, rsquared_linear,
        coefs_quadratic, residmse_quadratic, rsquared_quadratic,
        hessian_eigvals, hessian_eigvecs,
        figdir
    )
    (
     leadtimes,r2threshes,dsts,rsps,mixobjs,
     mixcrit_labels,mixobj_labels,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,i_thresh_cquantile
    ) = expt_config_COAST_analysis(cfg,pertop)
    thresh_cquantile = ccdf_levels[i_thresh_cquantile]
    r2thresh = r2threshes[1]

    color_lin = :cyan
    color_quad = :sienna1
    color_zern = :purple
    Nleadtime = length(leadtimes)
    Nanc = length(coast.ancestors)
    println("Gonna show phases and responses now")
    anc = coast.ancestors[i_anc]
    descendants = Graphs.outneighbors(ens.famtree, anc)

    t0 = coast.anc_tRmax[i_anc]
    t0ph = t0*sdm.tu
    t0str = @sprintf("%.0f", t0ph)
    ytgtstr = @sprintf("%.2f", cfg.target_yPerL*sdm.Ly)
    rxystr = @sprintf("%.3f", cfg.target_ryPerL*sdm.Ly)
    Rmin = minimum([minimum(coast.anc_Roft[i_anc]) for i_anc=1:Nanc])
    obj_label,short_obj_label = label_objective(cfg)
    lblargs = Dict(
                   :xlabelsize=>10,
                   :xlabelvisible=>false,
                   :xlabelpadding=>1.5, 
                   :xticksize=>2.0, 
                   :xticklabelsize=>10,
                   :xticklabelsvisible=>false, 
                   :xticklabelrotation=>0, 
                   :xgridvisible=>false, 

                   :ylabelsize=>10,
                   :ylabelvisible=>false,
                   :ylabelpadding=>1.5,
                   :yticksize=>2.0, 
                   :yticklabelsize=>10,
                   :titlesize=>12,
                   :yticklabelsvisible=>false, 
                   :yticklabelrotation=>0,
                   :ygridvisible=>false, 

                   :titlefont=>:regular, 
                   :titlevisible=>false,
                  )

    #idx_leadtimes2plot = reverse(unique(clamp.(round.(Int, length(leadtimes).*[1/20, 1/5, 2/5, 3/5]), 1, length(leadtimes))))
    idx_leadtimes2plot = reverse(collect(round.(Int, range(1, length(leadtimes); length=6))))
    Nleadtimes2plot = length(idx_leadtimes2plot)
    # split vertical space to liking
    vert_shares = Dict(
                       "pert" => 1,
                       "resp" => 1,
                       "r2" => 0.75,
                       "slope" => 0.75,
                       "eig" => 0.75,
                       "toplabel" => 0.2,
                       "bottomlabel" => 0.2,
                      )
    horz_shares = Dict(
                       "leftlabel" => 1.0,
                       "panels" => Nleadtimes2plot,
                      )
    horz_shares_total = sum(values(horz_shares))
    vert_shares_total = sum(values(vert_shares))
    figwidth,figheight = (100*horz_shares_total, 100*vert_shares_total)
    fig = Figure(size=(figwidth,figheight))
    i_mode_sf = 1
    Amin,Amax = pertop.sf_pert_amplitudes_min[i_mode_sf], pertop.sf_pert_amplitudes_max[i_mode_sf]
    @show leadtimes
    lout = fig[1,1] = GridLayout()

    # -------------- Labels -----------------
    toplabel = Label(lout[1,2:Nleadtimes2plot+1], label_target(cfg, sdm), padding=(5.0,5.0,0.0,5.0), valign=:bottom, halign=:center, fontsize=15, font=:regular)
    label_pert_text = """
    Perturbation
        Im    
        ↑     
        ω → Re
    """
    label_resp_text = """
    Response 
     True
      ↑             
      𝑅* → Fit
    """
    label_slope_text = """
    Linear coefficient
    magnitudes
    ‖(θ₁,θ₂)‖
    """
    label_eig_text = """
    Quadratic fit
    eigenvalues 
      λ₁,λ₂
    """
    label_r2_text = """
    Coefficient of 
    determination
        𝑅²
    """
    leftlab2d,leftlab1d,leftlabslope,leftlabeig,leftlabr2 = [Axis(lout[i_row,1], limits=(0,1,0,1)) for i_row=2:6] 
    Makie.text!(leftlab2d, 0, 0.5; text=label_pert_text, fontsize=12, align=(:left,:center))
    Makie.text!(leftlab1d, 0, 0.5; text=label_resp_text, fontsize=12, align=(:left,:center))
    Makie.text!(leftlabslope, 0, 0.5; text=label_slope_text, fontsize=12, align=(:left,:center))
    Makie.text!(leftlabeig, 0, 0.5; text=label_eig_text, fontsize=12, align=(:left,:center))
    Makie.text!(leftlabr2, 0, 0.5; text=label_r2_text, fontsize=12, align=(:left,:center))
    for lab = [leftlab2d,leftlab1d,leftlabr2,leftlabslope,leftlabeig]
        hidedecorations!(lab)
        hidespines!(lab)
    end
    bottomlabel = Label(lout[7,2:Nleadtimes2plot+1], @sprintf("−AST (𝑡*=%d)", coast.anc_tRmax[i_anc]/sdm.tu), halign=:center)

    # --------- Main axes ------------

    axs2d = [Axis(lout[2,i_col]; lblargs...) for i_col=2:Nleadtimes2plot+1]
    axs1d = [Axis(lout[3,i_col]; lblargs...) for i_col=2:Nleadtimes2plot+1]

    lblargs[:yticklabelsvisible] = true
    ax_slope = Axis(lout[4,2:Nleadtimes2plot+1]; lblargs...)
    ax_eig = Axis(lout[5,2:Nleadtimes2plot+1]; xlabelsize=12, lblargs...)
    lblargs[:xticklabelsvisible] = true
    ax_r2 = Axis(lout[6,2:Nleadtimes2plot+1]; yticks=[0.0,0.5,1.0], lblargs...)

    scores = vcat(coast.desc_Rmax[i_anc], [coast.anc_Rmax[i_anc]])
    scorerange = maximum(abs.(scores .- coast.anc_Rmax[i_anc])).*[-1,1].+coast.anc_Rmax[i_anc]
    
    Rbounds = [minimum(scores),maximum(scores)]
    Us = vcat(zeros(Float64, (1,2)), collect(transpose(coast.pert_seq_qmc)))
    amplitudes = sqrt.(
                       pertop.sf_pert_amplitudes_min[i_mode_sf]^2 * (1 .- Us[:,1]) .+
                       pertop.sf_pert_amplitudes_max[i_mode_sf]^2 * Us[:,1]
                    )
    phases = 2pi .* Us[:,2]
    Xpert = hcat(amplitudes .* cos.(phases), amplitudes .* sin.(phases))
    Ngrid = 30
    p1grid = collect(range(-Amax,Amax; length=Ngrid))
    p2grid = collect(range(-Amax,Amax; length=Ngrid))
    p12grid = hcat(vec(p1grid .* ones((1,Ngrid))), vec(ones(Ngrid) .* p2grid'))
    p12grid_mask = (p1grid.^2 .+ (p2grid.^2)' .<= Amax^2)
    plot_zernike_flag = false
    limits_2d = (1.1*Amax) .* [-1,1]
    limits_1d = [Rbounds[1],Rbounds[2]]

    # Keep track of maximizing points
    rsps = ["e","1","2","z"]
    rsp_labels = get_rsp_labels()
    best_pert = Dict(rsp=>zeros(Float64, (2, Nleadtime)) for rsp=rsps)
    best_resp = Dict(rsp=>zeros(Float64, (2, Nleadtime)) for rsp=rsps)

    i_col = 0
    for i_leadtime = reverse(1:Nleadtime) #idx_leadtimes2plot
        leadtime = leadtimes[i_leadtime]
        tpert = coast.anc_tRmax[i_anc] - leadtime
        tpstr = @sprintf("%.2f", leadtime*sdm.tu)
        if i_leadtime in idx_leadtimes2plot
            i_col += 1
            ax2d = axs2d[i_col]
            ax1d = axs1d[i_col]
            if i_col == 1
                ax2d.title = @sprintf("%s−%.2f", (i_col == 1 ? "−AST" : ""), leadtime*sdm.tu)
                ax2d.title = @sprintf("−AST=−%d", leadtime*sdm.tu)
                #ax2d.ylabel = label_pert_text
                ax2d.yticklabelsvisible = true
                #ax1d.ylabel = label_resp_text
                ax1d.yticklabelsvisible = true
            else
                ax2d.title = @sprintf("−%d", leadtime*sdm.tu)
            end
            if i_col < Nleadtimes2plot
                colgap!(lout, i_col, 0)
            end
        end
        # extra formatting
        idx_desc = desc_by_leadtime(coast, i_anc, leadtime, sdm) 
        Ndesc = length(idx_desc)
        (Rmax_pred_zernike,Rmax_pred_linear,Rmax_pred_quadratic) = (zeros(Float64, length(idx_desc)) for _=1:3)
        Rmax_pred_zernike_anc = coefs_zernike[1,i_leadtime,i_anc]
        Rmax_pred_linear_anc = coefs_linear[1,i_leadtime,i_anc]
        Rmax_pred_quadratic_anc = coefs_quadratic[1,i_leadtime,i_anc]
        limits_1d[1] = min(limits_1d[1], minimum(vcat(Rmax_pred_linear_anc,Rmax_pred_quadratic_anc)))
        limits_1d[2] = max(limits_1d[2], maximum(vcat(Rmax_pred_linear_anc,Rmax_pred_quadratic_anc)))

        #  ------------- Contours ---------------------
        response_surface_zernike = reshape(QG2L.zernike_model_2d(p12grid, Amax, coefs_zernike[:,i_leadtime,i_anc]), (Ngrid,Ngrid)) #r = c[1] .+ c[2].*p1grid .+ c[3].*transpose(p2grid)
        response_surface_linear = reshape(QG2L.linear_model_2d(p12grid, coefs_linear[:,i_leadtime,i_anc]), (Ngrid,Ngrid)) #r = c[1] .+ c[2].*p1grid .+ c[3].*transpose(p2grid)
        response_surface_quadratic = reshape(QG2L.quadratic_model_2d(p12grid, coefs_quadratic[:,i_leadtime,i_anc]), (Ngrid,Ngrid)) #r = c[1] .+ c[2].*p1grid .+ c[3].*transpose(p2grid)
        #for rs = (response_surface_zernike,response_surface_linear,response_surface_quadratic)
        #    rs[p12grid_mask .== 0] .= -Inf
        #end
        max_dR = max((maximum(filter(isfinite, (r.-coast.anc_Rmax[i_anc]))) for r=(response_surface_zernike, response_surface_linear, response_surface_quadratic)[2-plot_zernike_flag:3])...)
        min_dR = min((minimum(filter(isfinite, (r.-coast.anc_Rmax[i_anc]))) for r=(response_surface_zernike, response_surface_linear, response_surface_quadratic)[2-plot_zernike_flag:3])...)
        vmin = coast.anc_Rmax[i_anc] + min_dR
        vmax = coast.anc_Rmax[i_anc] + max_dR

        best_pert["e"][:,i_leadtime] .= Xpert[argmax(vcat(coast.anc_Rmax[i_anc], coast.desc_Rmax[i_anc][idx_desc])),:]
        best_pert["1"][:,i_leadtime] .= p12grid[argmax(vec(response_surface_linear)),:]
        best_pert["2"][:,i_leadtime] .= p12grid[argmax(vec(response_surface_quadratic)),:]
        best_pert["z"][:,i_leadtime] .= p12grid[argmax(vec(response_surface_zernike)),:]


        #vmin = min((minimum(r) for r=(response_surface_linear, response_surface_quadratic))...)
        @show vmin,vmax,coast.anc_Rmax[i_anc]
        if i_leadtime in idx_leadtimes2plot
            levels = sort(vcat(coast.anc_Rmax[i_anc], collect(range(vmin, vmax; length=10))))
            levpos = levels[levels .> coast.anc_Rmax[i_anc]]
            levneg = levels[levels .< coast.anc_Rmax[i_anc]]
            lev0 = [coast.anc_Rmax[i_anc]]
            # Contours: Zernike model
            if plot_zernike_flag
                contour!(ax2d, p1grid, p2grid, response_surface_zernike; levels=levpos, linestyle=:solid, color=color_zern, linewidth=1)
                contour!(ax2d, p1grid, p2grid, response_surface_zernike; levels=levneg, linestyle=(:dot,:dense), color=color_zern, linewidth=1)
                contour!(ax2d, p1grid, p2grid, response_surface_zernike; levels=lev0, linestyle=(:dash,:dense), color=color_zern, linewidth=1)
            end
            # Contours: linear model
            contour!(ax2d, p1grid, p2grid, response_surface_linear; levels=levpos, linestyle=:solid, color=color_lin, linewidth=1)
            contour!(ax2d, p1grid, p2grid, response_surface_linear; levels=levneg, linestyle=(:dot,:dense), color=color_lin, linewidth=1)
            contour!(ax2d, p1grid, p2grid, response_surface_linear; levels=lev0, linestyle=(:dash,:dense), color=color_lin, linewidth=1)
            # Contours: quadratic model
            contour!(ax2d, p1grid, p2grid, response_surface_quadratic; levels=levpos, linestyle=:solid, color=color_quad, linewidth=2)
            contour!(ax2d, p1grid, p2grid, response_surface_quadratic; levels=levneg, linestyle=(:dot,:dense), color=color_quad, linewidth=2)
            contour!(ax2d, p1grid, p2grid, response_surface_quadratic; levels=lev0, linestyle=(:dash,:dense), color=color_quad, linewidth=2)
            # ---------------------- Points ---------------------------
            scorekwargs = Dict(:color=>:black, :marker=>:star5, :markersize=>6, :alpha=>1.0)
            scatter!(ax2d, 0, 0; scorekwargs...)
            Ndesc = length(idx_desc)
            for (i_desc,desc) in enumerate(descendants[idx_desc])
                #@show amplitude,phase*360/(2pi),scores[i_desc]
                scorekwargs[:color] = :black
                gain = coast.desc_Rmax[i_anc][idx_desc[i_desc]] - coast.anc_Rmax[i_anc]
                scorekwargs[:markersize] = 6 + 16 * 1*abs(gain)/(scorerange[2]-scorerange[1])
                scorekwargs[:marker] = (gain > 0 ? :cross : :circle)
                scatter!(ax2d, Xpert[i_desc+1,:]...; scorekwargs...)
                # Plot predicted and actual
                Rmax_pred_zernike[i_desc] = let
                    c = coefs_zernike[:,i_leadtime,i_anc]
                    QG2L.zernike_model_2d(Xpert[i_desc+1:i_desc+2,:], Amax, c)[1]
                end
                Rmax_pred_linear[i_desc] = let
                    c = coefs_linear[:,i_leadtime,i_anc]
                    QG2L.linear_model_2d(Xpert[i_desc+1:i_desc+2,:], c)[1]
                end
                Rmax_pred_quadratic[i_desc] = let
                    c = coefs_quadratic[:,i_leadtime,i_anc]
                    QG2L.quadratic_model_2d(Xpert[i_desc+1:i_desc+2,:], c)[1]
                end
                if plot_zernike_flag
                    scorekwargs[:color] = color_zern
                    scatter!(ax1d, Rmax_pred_zernike[i_desc], coast.desc_Rmax[i_anc][idx_desc[i_desc]]; scorekwargs...)
                end
                scorekwargs[:color] = color_lin
                scatter!(ax1d, Rmax_pred_linear[i_desc], coast.desc_Rmax[i_anc][idx_desc[i_desc]]; scorekwargs...)
                scorekwargs[:color] = color_quad
                scatter!(ax1d, Rmax_pred_quadratic[i_desc], coast.desc_Rmax[i_anc][idx_desc[i_desc]]; scorekwargs...)
            end
            if plot_zernike_flag
                scatter!(ax1d, Rmax_pred_zernike_anc, coast.anc_Rmax[i_anc]; marker=:star5, color=color_zern)
            end
            scatter!(ax1d, Rmax_pred_linear_anc, coast.anc_Rmax[i_anc]; marker=:star5, color=color_lin)
            scatter!(ax1d, Rmax_pred_quadratic_anc, coast.anc_Rmax[i_anc]; marker=:star5, color=color_quad)
            hlines!(ax1d, coast.anc_Rmax[i_anc]; color=:black, linestyle=(:dash, :dense))
            scores_lit = vcat([coast.anc_Rmax[i_anc]], coast.desc_Rmax[i_anc][idx_desc], Rmax_pred_zernike_anc, Rmax_pred_linear_anc, Rmax_pred_quadratic_anc)
            scoremin_lit,scoremax_lit = extrema(scores_lit)
            inflation = 0.1
            scorebounds_lit = [(1+inflation)*scoremin_lit-inflation*scoremax_lit, (1+inflation)*scoremax_lit-inflation*scoremin_lit]
            #xlims!(ax1d, scorebounds_lit...)
            #ylims!(ax1d, scorebounds_lit...)
            arc!(ax2d, Point2f(0,0), Amin, 0, 2pi; color=:gray, alpha=0.5)
            arc!(ax2d, Point2f(0,0), Amax, 0, 2pi; color=:gray, alpha=0.5)
        end
    end
    for ax1d = axs1d
        lines!(ax1d, limits_1d, limits_1d; color=:black, linestyle=(:dash,:dense))
    end

    if plot_zernike_flag
        scatterlines!(ax_r2, -leadtimes.*sdm.tu, rsquared_zernike[:,i_anc]; color=color_zern, label="Zern")
    end
    scatterlines!(ax_r2, -leadtimes.*sdm.tu, rsquared_linear[:,i_anc]; color=color_lin, label="Lin")
    scatterlines!(ax_r2, -leadtimes.*sdm.tu, rsquared_quadratic[:,i_anc]; color=color_quad, label="Quad")
    for ax = (ax_slope,ax_eig,ax_r2)
        vlines!(ax, -leadtimes[idx_leadtimes2plot].*sdm.tu; color=:gray, alpha=0.5)
    end
    for r2lev = [0,0.5,1]
        hlines!(ax_r2, r2lev; color=:gray, alpha=0.5)
    end
    ylims!(ax_r2, -0.1, 1.1)

    # Slope 
    ydata = zeros(Float64, length(leadtimes))
    maxydata = 0.0
    for (coefs,color) in ((coefs_linear,color_lin),(coefs_quadratic,color_quad))
        ydata .= sqrt.(sum(coefs[2:3,:,i_anc].^2; dims=1)[1,:])
        maxydata = max(maxydata, maximum(abs.(ydata)))
        scatterlines!(ax_slope, -leadtimes.*sdm.tu, ydata, color=color)
    end
    coefcosfun(c1,c2) = sum(c1[2:3,:,i_anc].*c2[2:3,:,i_anc]; dims=1)[1,:]
    lin_quad_dotprod = coefcosfun(coefs_linear,coefs_quadratic) #./ sqrt.(coefcosfun(coefs_linear,coefs_linear) .* coefcosfun(coefs_quadratic,coefs_quadratic))
    #scatterlines!(ax_slope, -leadtimes.*sdm.tu, lin_quad_dotprod, color=:black)
    hlines!(ax_slope, 0.0; color=:gray, alpha=0.5)
    ylims!(ax_slope, 0, maxydata)
    # Hessian
    maxydata = 0.0
    for i_eig = 1:2
        ydata = hessian_eigvals[i_eig,:,i_anc]
        maxydata = max(maxydata, maximum(abs.(ydata)))
        scatterlines!(ax_eig, -leadtimes.*sdm.tu, ydata, color=color_quad)
    end
    ylims!(ax_eig, -maxydata, maxydata)
    hlines!(ax_eig, 0.0; color=:gray, alpha=0.5)

    colsize!(lout, 1, Relative(horz_shares["leftlabel"]/horz_shares_total))
    for i_col = 1:Nleadtimes2plot
        if i_col < Nleadtimes2plot
            colgap!(lout, i_col+1, 0.0)
            colsize!(lout, i_col+1, Relative(horz_shares["panels"]/horz_shares_total/Nleadtimes2plot))
        end
        xlims!(axs2d[i_col], limits_2d...)
        ylims!(axs2d[i_col], limits_2d...)
        xlims!(axs1d[i_col], limits_1d...)
        ylims!(axs1d[i_col], limits_1d...)
    end
    rowgap!(lout, 1, 0.0)
    rowgap!(lout, 2, 0.0)
    rowgap!(lout, 3, 10.0)
    rowgap!(lout, 4, 0.0)
    rowgap!(lout, 5, 0.0)

    rowsize!(lout, 1, Relative(vert_shares["toplabel"]/vert_shares_total))
    rowsize!(lout, 2, Relative(vert_shares["pert"]/vert_shares_total))
    rowsize!(lout, 3, Relative(vert_shares["resp"]/vert_shares_total))
    rowsize!(lout, 4, Relative(vert_shares["slope"]/vert_shares_total))
    rowsize!(lout, 5, Relative(vert_shares["eig"]/vert_shares_total))
    rowsize!(lout, 6, Relative(vert_shares["r2"]/vert_shares_total))
    rowsize!(lout, 7, Relative(vert_shares["bottomlabel"]/vert_shares_total))

    #colsize!(lout, 1, Relative(1/(1+length(idx_leadtimes2plot))))

    resize_to_layout!(fig)
    save(joinpath(figdir,"objective_response_anc$(i_anc)_linquad.png"), fig)
    println("Saved linquad for i_anc $(i_anc)")

    # --------------- Plot of maximizing point on disc due to both linear and quadratic models --------------
    fig = Figure(size=(1200,300))
    lout = fig[1,1] = GridLayout()
    axs = Dict(
               rsp=>Axis(
                         lout[1,i_col]; 
                         xlabel="Re{ω}",ylabel="Im{ω}",
                         title=rsp_labels[rsp], titlefont=:regular
                        ) 
               for (i_col,rsp) in enumerate(rsps)
              )
    for (i_rsp,rsp) in enumerate(rsps)
        scatterlines!(axs[rsp], best_pert[rsp][1,:], best_pert[rsp][2,:]; color=leadtimes, colorrange=(leadtimes[1],leadtimes[end]), colormap=:RdYlBu_4)
        if i_rsp > 1
            axs[rsp].ylabelvisible = axs[rsp].yticklabelsvisible = false
        end
        if i_rsp < length(rsps)
            colgap!(lout, i_rsp, 0)
        end
    end
    save(joinpath(figdir,"objective_response_anc$(i_anc)_linquad_maximizers.png"), fig)


end

function plot_fit_coefs() # HIBERNATING
    fig = Figure(size=(400,400))
    lout = fig[1,1] = GridLayout()
    lblargs = Dict(
                   :xlabel=>"Pert. time",
                   :xticklabelsize=>6,:xlabelsize=>9,:yticklabelsize=>6,:ylabelsize=>9,
                   :titlevisible=>false, :xticklabelsvisible=>false, :xlabelvisible=>false
                  )
    tperts = coast.anc_tRmax[i_anc] .- leadtimes
    tperts_ph = tperts .* sdm.tu
    # first row: constant offsets
    ax0 = Axis(lout[1,1]; title="Constant offsets", lblargs...)
    b100 = lines!(ax0, tperts_ph, vec(coefs_linear[1,:,i_anc]); color=:dodgerblue, label=L"$\beta^{(1)}_{0,0}$")
    b200 = lines!(ax0, tperts_ph, vec(coefs_quadratic[1,:,i_anc]); color=:red, label=L"$\beta^{(2)}_{0,0}$")
    hlines!(ax0, coast.anc_Rmax[i_anc], color=:black, linestyle=:dash)
    vlines!(ax0, coast.anc_tRmax[i_anc]*sdm.tu, color=:black, linestyle=:dash)
    Legend(lout[1,2], ax0)
    # second row: linear coefficients 
    ax1 = Axis(lout[2,1]; title="Linear coefficients", lblargs...)
    b101 = lines!(ax1, tperts, vec(coefs_linear[2,:,i_anc]); color=:dodgerblue, label=L"$\beta^{(1)}_{0,1}$")
    b110 = lines!(ax1, tperts, vec(coefs_linear[3,:,i_anc]); color=:dodgerblue, label=L"$\beta^{(1)}_{1,0}$")
    b201 = lines!(ax1, tperts, vec(coefs_quadratic[2,:,i_anc]); color=:red, label=L"$\beta^{(2)}_{0,1}$")
    b210 = lines!(ax1, tperts, vec(coefs_quadratic[3,:,i_anc]); color=:red, label=L"$\beta^{(2)}_{1,0}$")
    hlines!(ax1, 0, color=:black, linestyle=:dash)
    vlines!(ax1, coast.anc_tRmax[i_anc]*sdm.tu, color=:black, linestyle=:dash)
    Legend(lout[2,2], ax1)
    # third row: quadratic coefficients 
    ax2 = Axis(lout[3,1]; title="Quadratic coefficients", lblargs...)
    b202 = lines!(ax2, tperts, vec(coefs_quadratic[4,:,i_anc]); color=:red, label=L"$\beta^{(2)}_{0,2}$")
    b211 = lines!(ax2, tperts, vec(coefs_quadratic[5,:,i_anc]); color=:blue, label=L"$\beta^{(2)}_{1,1}$")
    b220 = lines!(ax2, tperts, vec(coefs_quadratic[6,:,i_anc]); color=:purple, label=L"$\beta^{(2)}_{2,0}$")
    for i_evec = 1:2
        b2 = lines!(ax2, tperts, (hessian_eigvals[i_evec,:,i_anc]); color=:cyan, label=L"$\lambda_{%$(i_evec)}(\beta^{(2)})$")
    end
    hlines!(ax2, 0; color=:black, linestyle=:dash)
    vlines!(ax2, coast.anc_tRmax[i_anc]*sdm.tu, color=:black, linestyle=:dash)
    Legend(lout[3,2], ax2)
    #axleg = Legend(lout[4,1], ax2) #[b00,b10,b01,b20,b11,b02])
    ax2.xticklabelsvisible = true
    ax2.xlabelvisible = true
    save(joinpath(figdir,"objective_responses_anc$(i_anc)_polycoefs.png"), fig)
    
end

function plot_contours_1family(
        coast::COASTState, 
        ens::EM.Ensemble, 
        i_anc::Int64, 
        dsc_weights::Vector{Float64}, 
        i_leadtime::Int64,
        cfg::ConfigCOAST, 
        thresh::Float64,
        sdm::QG2L.SpaceDomain,
        cop::QG2L.ConstantOperators, 
        pertop::QG2L.PerturbationOperator,
        contour_dispersion_filename::String, 
        figfile::String
    )
    (
     leadtimes,r2threshes,dsts,rsps,mixobjs,
     mixcrit_labels,mixobj_labels,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,
     i_thresh_cquantile,adjust_ccdf_per_ancestor
    ) = expt_config_COAST_analysis(cfg,pertop)

    Nanc = length(coast.ancestors)
    Nmem = EM.get_Nmem(ens)
    Ndsc = Nmem - Nanc
    Nleadtime = length(leadtimes)
    Ndsc_per_leadtime = div(Ndsc, Nleadtime*Nanc)
    # Two 2-panel plots
    # 1a. Average slightly-evolved perturbation (would be near 0 if not evolved)
    # 1b. Std. Dev. slightly-evolved perturbation
    # 2a. Average perturbed peak (not necessarily at same time)
    # 2b. Std. dev. perturbed peak
    leadtime = leadtimes[i_leadtime]
    anc = coast.ancestors[i_anc]
    idx_dsc = desc_by_leadtime(coast, i_anc, leadtime, sdm)[1:Ndsc_per_leadtime]
    dscs = Graphs.outneighbors(ens.famtree, anc)[idx_dsc]
    mems = vcat([anc], dscs)
    idx_dsc2plot = [argmin(coast.desc_Rmax[i_anc][idx_dsc]), argmax(coast.desc_Rmax[i_anc][idx_dsc])]
    Nt = cfg.follow_time + cfg.lead_time_max
    conc1fun!(conc1_onemem::Array{Float64,4},i_mem::Int64,mem::Int64) = begin
        JLD2.jldopen(ens.trajs[mem].history, "r") do f
            conc1_onemem[1:sdm.Nx,1:sdm.Ny,1:Nt,i_mem] .= f["conc_hist"][:,:,1,:]
        end
    end

    conc1 = zeros(Float64, (sdm.Nx, sdm.Ny, Nt, Ndsc_per_leadtime+1))

    for (i_mem,mem) in enumerate(mems)
        conc1fun!(conc1, i_mem, mem)
    end

    contcorrs,globcorrs = JLD2.jldopen(contour_dispersion_filename, "r") do f
        return [f[corrname][1:Nt,i_leadtime,1:Ndsc_per_leadtime,i_anc] for corrname=["contcorr","globcorr"]]
    end

    mean_contcorr = (contcorrs[1:Nt,1:Ndsc_per_leadtime] * dsc_weights[1:Ndsc_per_leadtime]) / sum(dsc_weights[1:Ndsc_per_leadtime])
    mean_globcorr = (globcorrs[1:Nt,1:Ndsc_per_leadtime] * dsc_weights[1:Ndsc_per_leadtime]) / sum(dsc_weights[1:Ndsc_per_leadtime])

    # Start with only the field at the timing of the original peak 
    # sub-select the y's 
    ybounds = [0.0, 1.0] #cfg.target_yPerL .+ 2*cfg.target_ryPerL .* [-1,1]
    iymin,iymax = findfirst(sdm.ygrid./sdm.Ly .>= ybounds[1]),findlast(sdm.ygrid./sdm.Ly .<= ybounds[2])
    iytgt = round(Int, sdm.Ny*cfg.target_yPerL)
    fig = Figure(size=(1200,800))
    lout = fig[1,1] = GridLayout()
    # First row: contour messes
    axs_contour = [Axis(lout[1,it], xlabel="𝑥/𝐿", ylabel="𝑦/𝐿", titlesize=24, xlabelsize=24, ylabelsize=24, xticklabelsize=18, yticklabelsize=18, ylabelvisible=(it==1), yticklabelsvisible=(it==1), titlefont=:regular, xgridvisible=false, ygridvisible=false, xlabelvisible=false, xticklabelsvisible=false) for it=1:3]
    # Second row: concentrations along a fixed latitude 
    axs_concaty = [Axis(lout[2,it], xlabel="𝑥/𝐿", ylabel="𝑐(⋅,𝑦₀)", titlesize=24, xlabelsize=24, ylabelsize=24, xticklabelsize=18, yticklabelsize=18, ylabelvisible=(it==1), yticklabelsvisible=(it==1), titlefont=:regular, xgridvisible=false, ygridvisible=false) for it=1:3]
    tinit = floor(Int, ens.trajs[i_anc].tphinit/sdm.tu)
    contour_levels = [thresh]  #collect(range(0,1;length=8))
    for ax = axs_contour
        locavg_rect = poly!(ax, [(cfg.target_xPerL + sgnx*cfg.target_rxPerL) for sgnx=[-1,1,1,-1]], [(cfg.target_yPerL + sgny*cfg.target_ryPerL) for sgny=[-1,-1,1,1]], color=:grey69,)
    end
    for ax = axs_concaty
        vlines!(ax, [(cfg.target_xPerL .+ sgnx*cfg.target_rxPerL for sgnx=[-1,1])...]; color=:grey60)
    end
    Rmaxs = vcat([coast.anc_Rmax[i_anc]], coast.desc_Rmax[i_anc][idx_dsc])
    Rmaxbounds = extrema(Rmaxs)
    Rmaxcolors = [:cyan, :firebrick] #cgrad(:lipari, Ndsc+1; categorical=true).colors
    order = sortperm(Rmaxs)
    Rmaxrank = sortperm(order)
    it_anc_tRmax = coast.anc_tRmax[i_anc] - tinit
    it_after_split = it_anc_tRmax - leadtime + 1
    for (i_dsc2plot,i_dsc) in enumerate(idx_dsc2plot)
        cargs = Dict(:color=>Rmaxcolors[i_dsc2plot]) #Rmaxrank[i_dsc+1]])
        # First column: right after the split
        it_dsc_tRmax = coast.desc_tRmax[i_anc][idx_dsc[i_dsc]] - tinit
        for (i_col,it) in enumerate((it_after_split,it_anc_tRmax,it_dsc_tRmax))
            contour!(axs_contour[i_col], sdm.xgrid./sdm.Lx, sdm.ygrid[iymin:iymax]./sdm.Ly, conc1[:,iymin:iymax,it,i_dsc+1]; cargs..., levels=contour_levels, linewidth=2.0)
            lines!(axs_concaty[i_col], sdm.xgrid./sdm.Lx, conc1[:,iytgt,it,i_dsc+1]; cargs..., linewidth=2.0)
        end
    end
    axs_contour[1].title = "Post-split:\n 𝑡 = 𝑡* − AST + 1 = 𝑡* − $(round(Int,sdm.tu*(leadtime-1)))"
    axs_contour[2].title = "Ancestor peak:\n 𝑡 = 𝑡* = $(round(Int,sdm.tu*coast.anc_tRmax[i_anc]))"
    axs_contour[3].title = "Descendant peaks:\n 𝑡 = variable"
    cargs = Dict(:color=>:black) #Rmaxcolors[Rmaxrank[1]])
    for (i_col,it) in enumerate((it_after_split,it_anc_tRmax,it_anc_tRmax))
        contour!(axs_contour[i_col], sdm.xgrid./sdm.Lx, sdm.ygrid[iymin:iymax]./sdm.Ly, conc1[:,iymin:iymax,it,1]; cargs..., levels=contour_levels, linestyle=(:dash,:dense), linewidth=2)
        lines!(axs_concaty[i_col], sdm.xgrid./sdm.Lx, conc1[:,iytgt,it,1]; cargs..., linestyle=(:dash,:dense), linewidth=2)
    end
    t0str = @sprintf("%.0f",coast.anc_tRmax[i_anc]/sdm.tu)
    # Third row: scores
    ax = Axis(lout[3,1:3], xlabel="𝑡−𝑡*",ylabel="𝑅(𝐱(𝑡))",ylabelsize=24, yticklabelsize=18, xlabelsize=24, xticklabelsize=18, xgridvisible=false, ygridvisible=false, xlabelvisible=false, xticklabelsvisible=false)
    for (i_dsc2plot,i_dsc) in enumerate(idx_dsc2plot)
        lines!(sdm.tu.*(collect(1:1:Nt) .+ tinit .- coast.anc_tRmax[i_anc]), coast.desc_Roft[i_anc][idx_dsc[i_dsc]]; color=Rmaxcolors[i_dsc2plot], linewidth=2) #Rmaxrank[i_dsc+1]])
    end
    lines!(sdm.tu.*(collect(1:1:Nt) .+ tinit .- coast.anc_tRmax[i_anc]), coast.anc_Roft[i_anc]; color=:black, linewidth=2, linestyle=(:dash,:dense))
    vlines!(ax, 0; color=:gray, linestyle=:solid, alpha=0.75)
    vlines!(ax, sdm.tu*(tinit+it_after_split-coast.anc_tRmax[i_anc]); color=:gray, linestyle=:solid, alpha=0.75)
    vlines!(ax, sdm.tu*(tinit+it_after_split-1-coast.anc_tRmax[i_anc]); color=:gray, linestyle=(:dash,:dense), alpha=0.75)
    scatter!(ax, sdm.tu*(coast.desc_tRmax[i_anc][idx_dsc[idx_dsc2plot]] .- coast.anc_tRmax[i_anc]), coast.desc_Rmax[i_anc][idx_dsc[idx_dsc2plot]]; color=Rmaxcolors#=[Rmaxrank[2:end]]=#, marker=:star6, markersize=20)
    xlims!(ax, (sdm.tu.*(tinit - coast.anc_tRmax[i_anc] .+ [0,Nt]))...)
    # ---------------- fourth row: correlations  ---------------
    ax = Axis(lout[4,1:3], xlabel="𝑡 − 𝑡*", ylabelsize=24, xlabelsize=24, xticklabelsize=18, yticklabelsize=18, xgridvisible=false, ygridvisible=false)
    contcorrlabel,globcorrlabel = ["σ⁻¹($(mixcrit_labels[corrname]))" for corrname=["contcorr","globcorr"]]
    # 1-epsilon^2 level
    epsilon = 3/8
    hlines!(ax, transcorr.([1.0, 1-epsilon^2, 0.0]); color=:grey79, linewidth=2.0)
    for (i_dsc2plot,i_dsc) in enumerate(idx_dsc2plot)
        lines!(ax, sdm.tu.*(collect(1:1:Nt) .+ tinit .- coast.anc_tRmax[i_anc]), transcorr.(contcorrs[:,i_dsc]); color=Rmaxcolors[i_dsc2plot], linewidth=2, label=contcorrlabel)
        lines!(ax, sdm.tu.*(collect(1:1:Nt) .+ tinit .- coast.anc_tRmax[i_anc]), transcorr.(globcorrs[:,i_dsc]); color=Rmaxcolors[i_dsc2plot], linewidth=2, linestyle=(:dash,:dense), label=globcorrlabel)
    end
    # Ensemble mean
    lines!(ax, sdm.tu.*(collect(1:1:Nt) .+ tinit .- coast.anc_tRmax[i_anc]), transcorr.(mean_contcorr); color=:black, linestyle=:solid, linewidth=2, label=contcorrlabel)
    lines!(ax, sdm.tu.*(collect(1:1:Nt) .+ tinit .- coast.anc_tRmax[i_anc]), transcorr.(mean_globcorr); color=:black, linestyle=(:dash,:dense), linewidth=2, label=globcorrlabel)
    axislegend(ax; linecolor=:black, framevisible=true, labelsize=18, position=:lb, merge=true)
    vlines!(ax, 0; color=:grey60, linestyle=:solid)
    vlines!(ax, sdm.tu*(tinit+it_after_split-coast.anc_tRmax[i_anc]); color=:grey60, linestyle=:solid)
    vlines!(ax, sdm.tu*(tinit+it_after_split-1-coast.anc_tRmax[i_anc]); color=:gray, linestyle=(:dash,:dense), alpha=0.5)
    xlims!(ax, (sdm.tu.*(tinit - coast.anc_tRmax[i_anc] .+ [0,Nt]))...)

    for ax = (axs_contour..., axs_concaty...)
        xlims!(ax, (0,1))
    end
    linkyaxes!(axs_concaty...)

    

    rowsize!(lout, 1, Relative(1/2))
    rowsize!(lout, 2, Relative(1/6))
    rowsize!(lout, 3, Relative(1/6))
    rowgap!(lout, 1, 15)
    rowgap!(lout, 3, 15)
    colgap!(lout, 1, 20)
    colgap!(lout, 2, 20)
        
    save(figfile, fig)



end
