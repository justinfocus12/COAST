function plot_objective_spaghetti(cfg, sdm, cop, pertop, ens, coast, i_anc, thresh, figdir)
    t0 = coast.anc_tRmax[i_anc]
    t0ph = t0*sdm.tu
    t0str = @sprintf("%.0f", t0ph)
    ytgtstr = @sprintf("%.2f", cfg.target_yPerL*sdm.Ly)
    rxystr = @sprintf("%.3f", cfg.target_ryPerL*sdm.Ly)
    Rmin = minimum([minimum(coast.anc_Roft[i_anc]) for i_anc=1:cfg.num_init_conds_max])
    obj_label,short_obj_label = label_objective(cfg)

    # ------- Plot 0: ancestor only --------
    traj = ens.trajs[i_anc]
    t_anc = floor(Int, traj.tphinit/sdm.tu)+1:1:traj.tfin
    tph_anc = t_anc .* sdm.tu
    fig = Figure(size=(400,200))
    ax = Axis(fig[1,1],xlabel=L"$t-%$(t0str)$", ylabel=L"$$Intensity", xgridvisible=false, ygridvisible=false)
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
   ax1 = Axis(lout[1,1], xlabel="t-$(t0str)", ylabel="Conc.", title=label_target(cfg,sdm), xgridvisible=false, ygridvisible=false, xticklabelsvisible=false, xlabelvisible=false, )
   ax3 = Axis(lout[2,1], xlabel="t-$(t0str)", ylabel="Conc. (peak)", xgridvisible=false, ygridvisible=false, yticks=[minimum(coast.desc_Rmax[i_anc]), coast.anc_Rmax[i_anc]], ytickformat="{:.2f}")
   # First panel: just the timeseries
   kwargs = Dict(:colormap=>:managua10, :colorrange=>(cfg.lead_time_min,cfg.lead_time_max), :color=>1)
   for (i_desc,desc) in enumerate(descendants)
       desc = descendants[i_desc]
       traj = ens.trajs[desc]
       kwargs[:color] = t0 - round(Int, coast.desc_tphpert[i_anc][i_desc]/sdm.tu)
       Nt = length(coast.desc_Roft[i_anc][i_desc])
       t_desc = (traj.tfin - Nt) .+ collect(1:1:Nt)
       tph_desc = t_desc .* sdm.tu
       lines!(ax1, tph_desc .- t0ph, coast.desc_Roft[i_anc][i_desc]; kwargs...)
       itpert = argmin(abs.(t_desc.*sdm.tu .- coast.desc_tphpert[i_anc][i_desc]))
       scatter!(ax1, coast.desc_tphpert[i_anc][i_desc]-t0ph, coast.desc_Roft[i_anc][i_desc][itpert]; kwargs..., markersize=5)
       scatter!(ax1, coast.desc_tphpert[i_anc][i_desc]-t0ph, coast.desc_Rmax[i_anc][i_desc]; kwargs..., markersize=5) 
       scatter!(ax3, coast.desc_tRmax[i_anc][i_desc]*sdm.tu-t0ph, coast.desc_Rmax[i_anc][i_desc]; kwargs..., markersize=8, alpha=0.55)
   end
   for ax = (ax1,ax3)
       hlines!(ax, thresh; color=:gray, alpha=0.25)
       hlines!(ax, coast.anc_Rmax[i_anc]; color=:black, linestyle=(:dash,:dense), linewidth=1.0)
       vlines!(ax, 0.0; color=:black, linestyle=(:dash,:dense), linewidth=1.0, alpha=1.0)
   end
   traj = ens.trajs[anc]
   t_anc = traj.tfin .+ collect(range(-length(coast.anc_Roft[i_anc])+1, 0; step=1)) 
   lines!(ax1, t_anc.*sdm.tu .- t0ph, coast.anc_Roft[i_anc]; color=:black, linestyle=(:dash,:dense), linewidth=1.5)
   linkxaxes!(ax1,ax3)
   rowsize!(lout, 1, Relative(2/3))
   rowgap!(lout, 20.0)

   save(joinpath(figdir,"objectives_anc$(i_anc).png"), fig)
   
end

function plot_objective_response_linquad(
        cfg, sdm, cop, pertop, ens, coast, i_anc, 
        coefs_linear, residmse_linear, rsquared_linear,
        coefs_quadratic, residmse_quadratic, rsquared_quadratic,
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
    Nleadtime = length(leadtimes)
    Nleadtimes2plot = 4
    println("Gonna show phases and responses now")
    anc = coast.ancestors[i_anc]
    descendants = Graphs.outneighbors(ens.famtree, anc)

    t0 = coast.anc_tRmax[i_anc]
    t0ph = t0*sdm.tu
    t0str = @sprintf("%.0f", t0ph)
    ytgtstr = @sprintf("%.2f", cfg.target_yPerL*sdm.Ly)
    rxystr = @sprintf("%.3f", cfg.target_ryPerL*sdm.Ly)
    Rmin = minimum([minimum(coast.anc_Roft[i_anc]) for i_anc=1:cfg.num_init_conds_max])
    obj_label,short_obj_label = label_objective(cfg)

    fig = Figure(size=(100*Nleadtimes2plot,125*(2+0.5)))
    i_mode_sf = 1
    @show leadtimes
    lout = fig[1:3,1] = GridLayout()
    lout_2d = lout[1,1] = GridLayout()
    lout_1d = lout[2,1] = GridLayout()
    lout_r2 = lout[3,1] = GridLayout()
    scores = vcat(coast.desc_Rmax[i_anc], [coast.anc_Rmax[i_anc]])
    scorerange = maximum(abs.(scores .- coast.anc_Rmax[i_anc])).*[-1,1].+coast.anc_Rmax[i_anc]
    Amin,Amax = pertop.sf_pert_amplitudes_min[i_mode_sf], pertop.sf_pert_amplitudes_max[i_mode_sf]
    
    i_col = 0
    if minimum(rsquared_quadratic[:,i_anc]) < r2thresh
        i_last_leadtime = max(Nleadtimes2plot, findfirst(rsquared_quadratic[:,i_anc] .< r2thresh))
    else
        i_last_leadtime = Nleadtime
    end
    all_descs_2plot = vcat([desc_by_leadtime(coast, i_anc, leadtimes[ilt], sdm) for ilt=1:i_last_leadtime]...) #findall(coast.desc_tphpert[i_anc] .<= leadtimes[i_last_leadtime]*sdm.tu)
    Rbounds = [extrema(vcat([coast.anc_Rmax[i_anc]], coast.desc_Rmax[i_anc][all_descs_2plot]))...]
    for i_leadtime = round.(Int, range(i_last_leadtime, 1; length=Nleadtimes2plot))
        i_col += 1
        leadtime = leadtimes[i_leadtime]
        tpert = coast.anc_tRmax[i_anc] - leadtime
        tpstr = @sprintf("%.2f", leadtime*sdm.tu)
        r2_lin_str = @sprintf("%.2f", rsquared_linear[i_leadtime,i_anc])
        r2_quad_str = @sprintf("%.2f", rsquared_quadratic[i_leadtime,i_anc])
        lblargs = Dict(:xticklabelsize=>9,:xlabelsize=>12,:yticklabelsize=>9,:ylabelsize=>12,:titlesize=>12,:xlabelvisible=>false,:ylabelvisible=>(i_col==1),:xticklabelsvisible=>false,:yticklabelsvisible=>(i_col==1), :xgridvisible=>false, :ygridvisible=>false, :xticklabelrotation=>pi/2)
        title_2d = "-$(tpstr)"
        title_1d = "($(r2_lin_str),$(r2_quad_str))"
        if i_col == 1
            title_2d = "-AST=$(title_2d)"
            title_1d = "R2 (1,2) = $(title_1d)"
        end
        ax2d = Axis(lout_2d[1,i_col]; xlabel="Re", ylabel="Impulse", title=title_2d,lblargs...,titlevisible=true, titlealign=:right)
        ax1d = Axis(lout_1d[1,i_col]; xlabel="Fitted peak conc.", ylabel="Conc. (peak)", title=title_1d, lblargs..., titlevisible=true, titlealign=:right)
        scorekwargs = Dict(:color=>:black, :marker=>:star5, :markersize=>8, :alpha=>1.0)
        scatter!(ax2d, 0, 0; scorekwargs...)
        idx_desc = findall(round.(Int, coast.desc_tphpert[i_anc]./sdm.tu) .== tpert)
        (Rmax_pred_linear,Rmax_pred_quadratic) = (zeros(Float64, length(idx_desc)) for _=1:2)
        Rmax_pred_linear_anc = coefs_linear[1,i_leadtime,i_anc]
        Rmax_pred_quadratic_anc = coefs_quadratic[1,i_leadtime,i_anc]
        for (i_desc,desc) in enumerate(descendants[idx_desc])
            pert = QG2L.read_perturbation_sequence(ens.trajs[desc].forcing).perts[1]
            i_pert_dim = 2*i_mode_sf-1
            amplitude = sqrt(
                             Amin^2 * (1-pert[i_pert_dim]) 
                             + Amax^2 * pert[i_pert_dim]
                            )
            phase = 2pi*pert[i_pert_dim+1]
            #@show amplitude,phase*360/(2pi),scores[i_desc]
            scorekwargs[:color] = :black
            gain = coast.desc_Rmax[i_anc][idx_desc[i_desc]] - coast.anc_Rmax[i_anc]
            scorekwargs[:markersize] = 8 + 8 * 2*abs(gain)/(scorerange[2]-scorerange[1])
            scorekwargs[:marker] = (gain > 0 ? :cross : :circle)
            scatter!(ax2d, amplitude*cos(phase), amplitude*sin(phase); scorekwargs...)
            # Plot predicted and actual
            Rmax_pred_linear[i_desc] = let
                c = coefs_linear[:,i_leadtime,i_anc]
                p1 = amplitude*cos(phase)
                p2 = amplitude*sin(phase)
                QG2L.linear_model_2d([p1 p2], c)[1]
            end
            Rmax_pred_quadratic[i_desc] = let
                c = coefs_quadratic[:,i_leadtime,i_anc]
                p1 = amplitude*cos(phase)
                p2 = amplitude*sin(phase)
                QG2L.quadratic_model_2d([p1 p2], c)[1]
            end
            scorekwargs[:color] = color_lin
            scatter!(ax1d, Rmax_pred_linear[i_desc], coast.desc_Rmax[i_anc][idx_desc[i_desc]]; scorekwargs...)
            scorekwargs[:color] = color_quad
            scatter!(ax1d, Rmax_pred_quadratic[i_desc], coast.desc_Rmax[i_anc][idx_desc[i_desc]]; scorekwargs...)
        end
        scatter!(ax1d, Rmax_pred_linear_anc, coast.anc_Rmax[i_anc]; marker=:star5, color=color_lin)
        scatter!(ax1d, Rmax_pred_quadratic_anc, coast.anc_Rmax[i_anc]; marker=:star5, color=color_quad)
        lines!(ax1d, Rbounds, Rbounds; color=:black, linestyle=(:dash,:dense))
        hlines!(ax1d, coast.anc_Rmax[i_anc]; color=:black, linestyle=(:dash, :dense))
        arc!(ax2d, Point2f(0,0), Amin, 0, 2pi; color=:gray, alpha=0.5)
        arc!(ax2d, Point2f(0,0), Amax, 0, 2pi; color=:gray, alpha=0.5)
        # Plot contours of the quadratic response function 
        Ngrid = 30
        p1grid = collect(range(-Amax,Amax; length=Ngrid))
        p2grid = collect(range(-Amax,Amax; length=Ngrid))
        p12grid = hcat(vec(p1grid .* ones((1,Ngrid))), vec(ones(Ngrid) .* p2grid'))
        response_surface_linear = reshape(QG2L.linear_model_2d(p12grid, coefs_linear[:,i_leadtime,i_anc]), (Ngrid,Ngrid)) #r = c[1] .+ c[2].*p1grid .+ c[3].*transpose(p2grid)
        response_surface_quadratic = reshape(QG2L.quadratic_model_2d(p12grid, coefs_quadratic[:,i_leadtime,i_anc]), (Ngrid,Ngrid)) #r = c[1] .+ c[2].*p1grid .+ c[3].*transpose(p2grid)
        max_dR = max((maximum(abs.(r.-coast.anc_Rmax[i_anc])) for r=(response_surface_linear, response_surface_quadratic))...)
        vmin = coast.anc_Rmax[i_anc] - max_dR
        vmax = coast.anc_Rmax[i_anc] + max_dR
        #vmin = min((minimum(r) for r=(response_surface_linear, response_surface_quadratic))...)
        @show vmin,vmax,coast.anc_Rmax[i_anc]
        levpos = collect(range(coast.anc_Rmax[i_anc], vmax; length=5)[2:end])
        lev0 = [coast.anc_Rmax[i_anc]]
        levneg = collect(range(vmin, coast.anc_Rmax[i_anc]; length=5)[1:end-1])
        contour!(ax2d, p1grid, p2grid, response_surface_linear; levels=levpos, linestyle=:solid, color=color_lin, linewidth=2)
        contour!(ax2d, p1grid, p2grid, response_surface_linear; levels=levneg, linestyle=(:dot,:dense), color=color_lin, linewidth=2)
        contour!(ax2d, p1grid, p2grid, response_surface_linear; levels=lev0, linestyle=(:dash,:dense), color=color_lin, linewidth=2)
        contour!(ax2d, p1grid, p2grid, response_surface_quadratic; levels=levpos, linestyle=:solid, color=color_quad, linewidth=2)
        contour!(ax2d, p1grid, p2grid, response_surface_quadratic; levels=levneg, linestyle=(:dot,:dense), color=color_quad, linewidth=2)
        contour!(ax2d, p1grid, p2grid, response_surface_quadratic; levels=lev0, linestyle=(:dash,:dense), color=color_quad, linewidth=2)
    end
    for i_col = 1:ncols(lout_2d)-1
        colgap!(lout_2d, i_col, 0.0)
        colgap!(lout_1d, i_col, 0.0)
    end
    ax_r2 = Axis(lout_r2[1,1], xlabel="-AST (t*=$(t0str))", ylabel="R2", ylabelsize=12, xlabelsize=12, yticklabelsize=9, xticklabelsize=9, xgridvisible=false, ygridvisible=false, yticks=[0.0,0.5,1.0])
    scatterlines!(ax_r2, -leadtimes.*sdm.tu, rsquared_linear[:,i_anc]; color=color_lin, label="Lin")
    scatterlines!(ax_r2, -leadtimes.*sdm.tu, rsquared_quadratic[:,i_anc]; color=color_quad, label="Quad")
    hlines!(ax_r2, r2thresh; color=:gray)
    ylims!(ax_r2, 0.0, 1.0)

    Label(lout_2d[1,:,Top()], label_target(cfg, sdm), padding=(5.0,5.0,15.0,5.0), valign=:bottom, fontsize=12)
    rowgap!(lout, 1, 5.0)
    rowgap!(lout, 2, 0.0)
    rowsize!(lout, 3, Relative(1/5))

    save(joinpath(figdir,"objective_response_anc$(i_anc)_linquad.png"), fig)
    println("Saved linquad for i_anc $(i_anc)")
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
