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
    idx_leadtimes2plot = reverse(unique(clamp.(round.(Int, length(leadtimes).*[1/20, 2/5]), 1, length(leadtimes))))
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
        scatter!(ax1, coast.desc_tphpert[i_anc][i_desc]-t0ph, coast.desc_Roft[i_anc][i_desc][itpert]; kwargs..., markersize=5)
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

    idx_leadtimes2plot = reverse(unique(clamp.(round.(Int, length(leadtimes).*[1/20, 1/5, 2/5, 3/5]), 1, length(leadtimes))))
    Nleadtimes2plot = length(idx_leadtimes2plot)
    fig = Figure(size=(150*Nleadtimes2plot,150*(2+0.5)))
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
    Rbounds = [minimum(scores),maximum(scores)]
    for i_leadtime = idx_leadtimes2plot
        i_col += 1
        leadtime = leadtimes[i_leadtime]
        tpert = coast.anc_tRmax[i_anc] - leadtime
        tpstr = @sprintf("%.2f", leadtime*sdm.tu)
        r2_lin_str = @sprintf("%.2f", rsquared_linear[i_leadtime,i_anc])
        r2_quad_str = @sprintf("%.2f", rsquared_quadratic[i_leadtime,i_anc])
        lblargs = Dict(:xticklabelsize=>7,:xlabelsize=>8,:yticklabelsize=>7,:ylabelsize=>8,:titlesize=>10,:xlabelvisible=>false,:ylabelvisible=>(i_col==1),:xticklabelsvisible=>true, :xticklabelrotation=>pi/2, :yticklabelsvisible=>true, :xgridvisible=>false, :ygridvisible=>false, :titlefont=>:regular, :xticksize=>2.0, :yticksize=>2.0, :xlabelpadding=>1.5, :ylabelpadding=>1.5)
        title_2d = "−$(tpstr)"
        title_1d = "($(r2_lin_str),$(r2_quad_str))"
        if i_col == 1
            title_2d = "−AST=$(title_2d)"
            title_1d = "𝑅² (1,2) = $(title_1d)"
        end
        ax2d = Axis(lout_2d[1,i_col]; xlabel="Re{ω}", ylabel="Im{ω}", title=title_2d,lblargs...,titlevisible=false, titlealign=:right)
        ax1d = Axis(lout_1d[1,i_col]; xlabel="Fitted severity", ylabel="Actual severity 𝑅*(ω)", title=title_1d, lblargs..., titlevisible=false, titlealign=:right)
        idx_desc = findall(round.(Int, coast.desc_tphpert[i_anc]./sdm.tu) .== tpert)
        #@infiltrate
        (Rmax_pred_linear,Rmax_pred_quadratic) = (zeros(Float64, length(idx_desc)) for _=1:2)
        Rmax_pred_linear_anc = coefs_linear[1,i_leadtime,i_anc]
        Rmax_pred_quadratic_anc = coefs_quadratic[1,i_leadtime,i_anc]
        #  ------------- Contours ---------------------
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
        # ---------------------- Points ---------------------------
        scorekwargs = Dict(:color=>:black, :marker=>:star5, :markersize=>6, :alpha=>1.0)
        scatter!(ax2d, 0, 0; scorekwargs...)
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
            scorekwargs[:markersize] = 6 + 16 * 2*abs(gain)/(scorerange[2]-scorerange[1])
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
        scores_lit = vcat([coast.anc_Rmax[i_anc]], coast.desc_Rmax[i_anc][idx_desc], Rmax_pred_linear_anc, Rmax_pred_quadratic_anc)
        scoremin_lit,scoremax_lit = extrema(scores_lit)
        inflation = 0.1
        scorebounds_lit = [(1+inflation)*scoremin_lit-inflation*scoremax_lit, (1+inflation)*scoremax_lit-inflation*scoremin_lit]
        xlims!(ax1d, scorebounds_lit...)
        ylims!(ax1d, scorebounds_lit...)
        arc!(ax2d, Point2f(0,0), Amin, 0, 2pi; color=:gray, alpha=0.5)
        arc!(ax2d, Point2f(0,0), Amax, 0, 2pi; color=:gray, alpha=0.5)
    end
    for i_col = 1:ncols(lout_2d)-1
        colgap!(lout_2d, i_col, 10.0)
        colgap!(lout_1d, i_col, 10.0)
    end
    ax_r2 = Axis(lout_r2[1,1], xlabel="−AST (𝑡* = $(t0str))", ylabel="𝑅²", ylabelsize=8, xlabelsize=8, yticklabelsize=7, xticklabelsize=7, xgridvisible=false, ygridvisible=false, yticks=[0.0,0.5,1.0], xticksize=2, yticksize=2, titlefont=:regular)
    scatterlines!(ax_r2, -leadtimes.*sdm.tu, rsquared_linear[:,i_anc]; color=color_lin, label="Lin")
    scatterlines!(ax_r2, -leadtimes.*sdm.tu, rsquared_quadratic[:,i_anc]; color=color_quad, label="Quad")
    hlines!(ax_r2, r2thresh; color=:gray, alpha=0.5)
    vlines!(ax_r2, -leadtimes[idx_leadtimes2plot].*sdm.tu; color=:gray, alpha=0.5)
    ylims!(ax_r2, -0.1, 1.1)

    Label(lout_2d[1,:,Top()], label_target(cfg, sdm), padding=(5.0,5.0,5.0,5.0), valign=:bottom, fontsize=9)
    Label(lout_2d[1,:,Bottom()], "Re{ω}", padding=(0,5,0,18), valign=:bottom, fontsize=8)
    Label(lout_1d[1,:,Bottom()], "(Linear, quadratic)-fitted severity 𝑅̂*(ω)", padding=(0,5,0,20), valign=:bottom, fontsize=8)
    #rowsize!(lout_1d, 1, Relative(14/15))
    #rowsize!(lout_2d, 1, Relative(14/15))
    rowgap!(lout, 1, 4.0)
    rowgap!(lout, 2, 4.0)
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
    # Two 2-panel plots
    # 1a. Average slightly-evolved perturbation (would be near 0 if not evolved)
    # 1b. Std. Dev. slightly-evolved perturbation
    # 2a. Average perturbed peak (not necessarily at same time)
    # 2b. Std. dev. perturbed peak
    leadtime = leadtimes[i_leadtime]
    anc = coast.ancestors[i_anc]
    idx_dsc = desc_by_leadtime(coast, i_anc, leadtime, sdm)
    dscs = Graphs.outneighbors(ens.famtree, anc)[idx_dsc]
    mems = vcat([anc], dscs)
    Ndsc = length(dscs)
    idx_dsc2plot = [argmin(coast.desc_Rmax[i_anc][idx_dsc]), argmax(coast.desc_Rmax[i_anc][idx_dsc])]
    Nt = cfg.follow_time + cfg.lead_time_max
    conc1fun!(conc1_onemem::Array{Float64,4},i_mem::Int64,mem::Int64) = begin
        JLD2.jldopen(ens.trajs[mem].history, "r") do f
            conc1_onemem[1:sdm.Nx,1:sdm.Ny,1:Nt,i_mem] .= f["conc_hist"][:,:,1,:]
        end
    end

    conc1 = zeros(Float64, (sdm.Nx, sdm.Ny, Nt, Ndsc+1))

    for (i_mem,mem) in enumerate(mems)
        conc1fun!(conc1, i_mem, mem)
    end

    contcorrs,globcorrs = JLD2.jldopen(contour_dispersion_filename, "r") do f
        return [f[corrname][1:Nt,i_leadtime,1:Ndsc,i_anc] for corrname=["contcorr","globcorr"]]
    end

    mean_contcorr = (contcorrs[1:Nt,1:Ndsc] * dsc_weights[1:Ndsc]) / sum(dsc_weights[1:Ndsc])
    mean_globcorr = (globcorrs[1:Nt,1:Ndsc] * dsc_weights[1:Ndsc]) / sum(dsc_weights[1:Ndsc])

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
