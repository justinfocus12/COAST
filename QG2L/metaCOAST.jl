


function metaCOAST_latdep_procedure(expt_supdir::String, resultdir_dns::String; i_expt=nothing)
    todo = Dict(
                "remove_pngs" =>                    0,
                "plot_GPD_params_ydep" =>           0,
                "plot_ccdfs_latdep" =>              1,
                "print_simtimes" =>                 0,
                "plot_mixcrits_ydep" =>             0,
                "compare_tvs" =>                    0,
                "plot_toast" =>                     0,
               )
    php,sdm = QG2L.expt_config()
    cop_pertop_file = joinpath(expt_supdir,"cop_pertop.jld2")
    if isfile(cop_pertop_file)
        println("Loading...")
        cop,pertop = JLD2.jldopen(cop_pertop_file, "r") do f
            return f["cop"],f["pertop"]
        end
    else
        println("Computing...")
        cop,pertop = QG2L.expt_setup(php, sdm)
        JLD2.jldopen(cop_pertop_file, "w") do f
            f["cop"] = cop
            f["pertop"] = pertop
        end
    end
    @show pertop.pert_dim
    phpstr = QG2L.strrep_PhysicalParams(php)
    sdmstr = QG2L.strrep_SpaceDomain(sdm)
    pertopstr = QG2L.strrep_PerturbationOperator(pertop, sdm)
    target_r, = expt_config_metaCOAST_analysis(i_expt=i_expt)
    rxystr = @sprintf("(%d/%d)L",round(Int,target_r*sdm.Ny),sdm.Ny)
    ytgts,_ = paramsets()
    Nytgt = length(ytgts)
    cfgs = Vector{ConfigCOAST}([])
    exptdirs_COAST = Vector{String}([])
    for y = ytgts
        cfg = (
              ConfigCOAST(
                           sdm.tu
                           ;
                           target_rxPerL=target_r, 
                           target_yPerL=y, 
                           target_ryPerL=target_r
                          ))
        push!(cfgs, cfg)
    end
    (
     leadtimes,r2threshes,dsts,rsps,mixobjs,
     mixcrit_labels,mixobj_labels,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,i_thresh_cquantile,adjust_ccdf_per_ancestor
    ) = expt_config_COAST_analysis(cfgs[1],pertop)
    thresh_cquantile = ccdf_levels[i_thresh_cquantile]
    threshstr = @sprintf("thrt%d", round(Int, 1/thresh_cquantile))
    for (i_y,y) in enumerate(ytgts)
        cfgstr = strrep_ConfigCOAST(cfgs[i_y])
        exptdir_COAST = joinpath(expt_supdir,"COAST_$(cfgstr)_$(pertopstr)_$(threshstr)")
        push!(exptdirs_COAST, exptdir_COAST)
    end
    println("Collected the cfgs")
    obj_label,short_obj_label = label_objective(cfgs[1])
    Nanc = cfgs[1].num_init_conds_max
    resultdir = joinpath(expt_supdir,"$(strrep_ConfigCOAST_varying_yPerL(cfgs[1]))_$(threshstr)")
    mkpath(resultdir)
    if todo["remove_pngs"] == 1
        for filename = readdir(resultdir, join=true)
            if endswith(filename,"png")
                rm(filename)
            end
        end
    end


    if 1 == todo["plot_ccdfs_latdep"]
        Rccdfs_ancgen = zeros(Float64, (length(ccdf_levels),Nytgt))
        Rccdfs_valid = zeros(Float64, (length(ccdf_levels),Nytgt))
        Rmean_ancgen = zeros(Float64, Nytgt)
        Rmean_valid = zeros(Float64, Nytgt)
        (gpd_scale_valid,gpd_shape_valid,std_valid) = (zeros(Float64,Nytgt) for _=1:3)

        for (i_ytgt,ytgt) in enumerate(ytgts)
            cfgstr = strrep_ConfigCOAST(cfgs[i_ytgt])
            resultdir_COAST = joinpath(exptdirs_COAST[i_ytgt], "results")
            (
             Rccdfs_ancgen[:,i_ytgt],Rccdfs_valid[:,i_ytgt],
             Rmean_ancgen[i_ytgt],Rmean_valid[i_ytgt],
             gpd_scale_valid[i_ytgt],gpd_shape_valid[i_ytgt],std_valid[i_ytgt],
            ) = (
                 JLD2.jldopen(joinpath(resultdir_COAST, "objective_dns_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).jld2"), "r") do f
                     return (
                             f["Rccdf_ancgen_seplon"][:,1],f["Rccdf_valid_agglon"],
                             SB.mean(f["Roft_ancgen_seplon"][:,1]),SB.mean(f["Roft_valid_seplon"]),
                             f["gpdpar_valid_agglon"][1],f["gpdpar_valid_agglon"][2],f["std_valid_agglon"]
                            )
                 end
                )
        end
        # ----------------- Latitude dependent quantiles -------------------
        colargs = Dict(:colormap=>Reverse(:roma), :colorrange=>(1,length(ccdf_levels)))
        fig = Figure()
        lout = fig[1,1] = GridLayout()
        Rccdf_rough_intercept = 0
        Rccdf_rough_slope = 1.0
        Rccdf_rough = Rccdf_rough_intercept .+ Rccdf_rough_slope .* ytgts #0.5 .+ 0.5 .* ytgts
        ax = Axis(lout[1,1],xlabel="𝑐 − 𝑦/𝐿",ylabel="(Target 𝑦)/𝐿", xgridvisible=false, ygridvisible=false)
        toplabel = label_target(target_r,sdm)
        Label(lout[1,1:1,Top()], toplabel, padding=(5.0,5.0,15.0,5.0), valign=:bottom, halign=:center, fontsize=15, font=:regular)
        #scatterlines!(ax, Rmean_valid .- Rccdf_rough, ytgts; color=:black, linestyle=(:dash,:dense))
        #scatterlines!(ax, Rmean_ancgen .- Rccdf_rough, ytgts; color=:black, linestyle=:solid, label="Mean")
        lines!(ax, 1 .- Rccdf_rough, ytgts; color=:gray, alpha=0.5, linewidth=2)
        lines!(ax, 0 .- Rccdf_rough, ytgts; color=:gray, alpha=0.5, linewidth=2)
        for i_cl = reverse(1:length(ccdf_levels))
            cl = ccdf_levels[i_cl]
            lines!(ax, Rccdfs_valid[i_cl,:].-Rccdf_rough, ytgts; linewidth=2, linestyle=(:dash,:dense), color=i_cl, colargs..., label="Short DNS")
            lines!(ax, Rccdfs_ancgen[i_cl,:].-Rccdf_rough, ytgts; linewidth=1, linestyle=:solid, color=i_cl, colargs..., label="Long DNS")
        end
        xmin = min(minimum(Rmean_valid .- Rccdf_rough), minimum(Rccdfs_valid.-Rccdf_rough')) - 0.02
        xmax = max(maximum(Rmean_valid .- Rccdf_rough), maximum(Rccdfs_valid.-Rccdf_rough')) + 0.02
        xlims!(ax, xmin, xmax)
        ylims!(ax, 0.0, 1.0)
        lout[1,2] = Legend(fig, ax, "Exceedance probabilities\n(½)ᵏ, k ∈ {1,...,15}"; framevisible=false, titlefont=:regular, titlehalign=:left, merge=true, linecolor=:black)
        save(joinpath(resultdir,"ccdfs_latdep_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).png"),fig)

        # ---------- Plot GPD parameters along with moments ---------
        mssk = JLD2.jldopen(joinpath(resultdir_dns,"moments_mssk_$(cfgs[1].target_field[1:end-1]).jld2"),"r") do f
            return f["mssk_xall"][1,:,parse(Int,cfgs[1].target_field[end]),:]
        end
        fig = Figure()
        lout = fig[1,1] = GridLayout()
        axargs = Dict(:ylabel=>"(Target 𝑦)/𝐿", :xgridvisible=>false, :ygridvisible=>false, :xticklabelrotation=>pi/2)
        axstd = Axis(lout[1,1]; xlabel="Std. Dev.", axargs...)
        axargs[:ylabelvisible] = axargs[:yticklabelsvisible] = false
        axscale = Axis(lout[1,2]; xlabel="GPD scale σ", axargs...)
        axshape = Axis(lout[1,3]; xlabel="GPD shape ξ", axargs...)
        threshcquantstr = @sprintf("%.2E",thresh_cquantile)
        toplabel = "Threshold exceedance probability $(powerofhalfstring(i_thresh_cquantile))=$(threshcquantstr)"
        Label(lout[1,2:3,Top()], toplabel, padding=(5.0,5.0,15.0,5.0), valign=:bottom, halign=:left, fontsize=15, font=:regular)
        # Std. Dev.
        lines!(axstd, mssk[:,2], sdm.ygrid./sdm.Ly; color=:black, linestyle=(:dash,:dense), label="(1/$(sdm.Ny))𝐿")
        scatterlines!(axstd, std_valid, ytgts; color=:black, label="($(round(Int, target_r*sdm.Ny))/$(sdm.Ny))/𝐿")
        axislegend(axstd, "Box radius"; position=:lt, fontsize=8, titlefont=:regular, framevisible=false)
        vlines!(axstd, 0.0; color=:gray, alpha=0.5, linewidth=3)
        # scale
        scatterlines!(axscale, gpd_scale_valid, ytgts; color=:black)
        vlines!(axscale, 0.0; color=:gray, alpha=0.5, linewidth=3)
        # shape
        vlines!(axshape, 0.0; color=:gray, alpha=0.5, linewidth=3)
        scatterlines!(axshape, gpd_shape_valid, ytgts; color=:black)
        xlims!(axshape, -1, 0.25)

        colgap!(lout, 1, 0.0)
        colgap!(lout, 2, 0.0)
        save(joinpath(resultdir,"gpdpars_latdep_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).png"),fig)
    end



    if 1 == todo["print_simtimes"]
        simtimes = zeros(Float64, Nytgt)
        Nancs = zeros(Int64, Nytgt)
        Nmems = zeros(Int64, Nytgt)
        for (i_ytgt,ytgt) in enumerate(ytgts)
            coastfile_COAST = joinpath(exptdirs_COAST[i_ytgt], "ensemble_data", "coast.jld2")
            ensfile_COAST = joinpath(exptdirs_COAST[i_ytgt], "ensemble_data", "ens.jld2")
            coast = load_COASTState(coastfile_COAST)
            ens = EM.load_Ensemble(ensfile_COAST)
            Nancs[i_ytgt] = length(coast.ancestors)
            Nmems[i_ytgt] = EM.get_Nmem(ens)
            simtimes[i_ytgt] = coast.anc_tRmax[Nancs[i_ytgt]]/sdm.tu
        end
        println("simtimes = ")
        display(simtimes)
        println("max(simtimes) = $(maximum(simtimes))")
        println("Nancs, Nmems = ")
        display(hcat(Nancs,Nmems))
    end

    @show keys(mixobjs)
    Nleadtime = length(leadtimes)
    
    if todo["plot_GPD_params_ydep"] == 1
        ccdfs_at_thresh,threshes,scales,shapes = (zeros(Float64,Nytgt) for _=1:4)
        levelss,ccdfss_emp = (Vector{Vector{Float64}}([]) for _=1:2)
        for (i_cfg,cfg) in enumerate(cfgs)
            cfgstr = strrep_ConfigCOAST(cfg)
            resultdir_COAST = joinpath(expt_supdir,"COAST_$(cfgstr)_$(pertopstr)/results")
            (
             ccdfs_at_thresh[i_cfg],threshes[i_cfg],scales[i_cfg],shapes[i_cfg],
             levels,ccdfs_emp
            ) = (
                 JLD2.jldopen(joinpath(resultdir_COAST,"GPD_dns.jld2"),"r") do f
                     return (
                             f["ccdf_at_thresh"],f["thresh"],f["scale"],f["shape"],
                             f["levels"],f["ccdfs_emp"]
                            )
                 end
                )

            push!(levelss, levels)
            push!(ccdfss_emp, vec(SB.mean(ccdfs_emp; dims=2)))
        end
        @show ccdfs_at_thresh
        @show threshes
        @show scales
        @show shapes

        mssk = JLD2.jldopen(joinpath(resultdir_dns,"moments_mssk_$(cfgs[1].target_field[1:end-1]).jld2"),"r") do f
            return f["mssk_xall"][:,:,parse(Int,cfgs[1].target_field[end]),:]
        end

        fig = Figure(size=(600,300))
        lout = fig[1,1] = GridLayout()
        kwargs = Dict(:ylabel=>L"$y$",:xgridvisible=>false, :ygridvisible=>false, :xticklabelrotation=>pi/2)
        ax1 = Axis(lout[1,1]; xlabel=L"$\mu$", kwargs...)
        kwargs[:ylabelvisible] = false
        kwargs[:yticklabelsvisible] = false
        ax2 = Axis(lout[1,2]; xlabel=L"$\sigma$", kwargs...)
        ax3 = Axis(lout[1,3]; xlabel=L"$\xi$", kwargs...)
        logccdf_grid = vcat(log.([0.5, 0.4, 0.3, 0.2]), collect(range(-1, -4; length=4)).*log(10))
        @show logccdf_grid
        invlogccdfs_GPD = NaN .* ones(Float64, (length(logccdf_grid),Nytgt))
        invlogccdfs_emp = zeros(Float64, (length(logccdf_grid),Nytgt))
        for (i_y,y) in enumerate(ytgts)
            GPD = Dists.GeneralizedPareto(threshes[i_y],scales[i_y],shapes[i_y])
            first_lev = findfirst(logccdf_grid .< log(ccdfs_at_thresh[i_y]))
            invlogccdfs_GPD[first_lev:end,i_y] .= Dists.invlogccdf.(GPD, logccdf_grid[first_lev:end] .- log(ccdfs_at_thresh[i_y]))
            for (i_logccdf,logccdf) in enumerate(logccdf_grid)
                Nlev = length(levelss[i_y])
                if ccdfss_emp[i_y][Nlev] > exp(logccdf)
                    invlogccdfs_emp[i_logccdf,i_y] = NaN
                else
                    invlogccdfs_emp[i_logccdf,i_y] = levelss[i_y][findfirst(ccdfss_emp[i_y] .<= exp(logccdf))]
                end
            end
        end
        for (i_logccdf,logccdf) in enumerate(logccdf_grid)
            lblargs = Dict(:color=>i_logccdf, :colorrange=>(1,length(logccdf_grid)), :colormap=>Reverse(:hawaii), :label=>@sprintf("%.1e",exp(logccdf)))
            lines!(ax1, vec(invlogccdfs_emp[i_logccdf,:]), ytgts.*sdm.Ly; linestyle=:solid, lblargs...)
            lines!(ax1, vec(invlogccdfs_GPD[i_logccdf,:]), ytgts.*sdm.Ly; linestyle=:dash, lblargs...)
        end
        scatterlines!(ax1, threshes, ytgts.*sdm.Ly; color=:black)
        lines!(ax1, mssk[1,:,1], sdm.ygrid; color=:dodgerblue)
        #ax0 = Axis(lout[1,0]; title=L"CCDF levels$$")
        Legend(lout[1,0], ax1; merge=true, labelsize=8, framevisible=false, title=L"CCDF$$", titlevisible=true, nbanks=1)
        scatterlines!(ax2, scales, ytgts.*sdm.Ly; color=:black)
        lines!(ax2, mssk[1,:,2], sdm.ygrid; color=:dodgerblue, label=L"stdev$$")
        scatterlines!(ax3, shapes, ytgts.*sdm.Ly; color=:black)
        Label(lout[1,1:3,Top()], L"Mean %$(short_obj_label) over box size %$(rxystr)$$",valign=:bottom)
        Label(lout[1,0,Top()], L"CCDF$$")

        for ax = (ax1,ax2,ax3,ax3)
            ylims!(ax, 0, sdm.Ly)
        end
        if target_field[1:4] == "conc"
            xlims!(ax1, -0.05, 1.05)
            vlines!(ax1, 0; color=:black, linestyle=:dash)
            vlines!(ax1, 1; color=:black, linestyle=:dash)
            xlims!(ax3, -0.5, 0.1)
        end
        vlines!(ax2,0; color=:black,linestyle=:dash)
        vlines!(ax3,0; color=:black,linestyle=:dash)


        save(joinpath(resultdir,"GPD_dns_latdep.png"),fig)

    end

    if todo["plot_toast"] == 1
        iltmixs = Dict()
        for dst = dsts
            iltmixs[dst] = Dict()
            for rsp = rsps
                iltmixs[dst][rsp] = Dict()
                for mc = keys(mixobjs)
                    iltmixs[dst][rsp][mc] = zeros(Int64, (Nytgt,length(mixobjs[mc]),Nanc,length(distn_scales[dst])))
                end
            end
        end
        for (i_y,y) in enumerate(ytgts)
            resultdir_y = joinpath(exptdirs_COAST[i_y],"results")
            iltmixs_y = JLD2.jldopen(joinpath(resultdir_y,"ccdfs_regressed.jld2"),"r") do f
                return f["iltmixs"]
            end
            for dst = dsts
                for rsp = rsps
                    if ("g" == dst) && ("2" == rsp)
                        continue
                    end
                    for mc = keys(mixobjs)
                        iltmixs[dst][rsp][mc][i_y,:,:,:] .= iltmixs_y[dst][rsp][mc]
                    end
                end
            end
        end
        for dst = ["b"]
            for rsp = ["2"]
                if ("g" == dst) && (rsp in ["1+u","2","2+u"])
                    continue
                end
                for mc = ["ent"] #keys(mixobjs)
                    i_mixobj = 1
                    for i_scl = 1:length(distn_scales[dst])
                        ost_counts = zeros(Int64, (Nleadtime,Nytgt))
                        ilts = iltmixs[dst][rsp][mc][:,i_mixobj,:,i_scl]
                        fig = Figure(size=(300,300))
                        lout = fig[1,1] = GridLayout()
                        ax = Axis(lout[1,1]; xlabel=L"$t_{\mathrm{pert}}$", ylabel=L"$$Target lat.", title=L"$t_{\mathrm{pert}}$ PMF, Box size %$(rxystr), Scale$=$%$(distn_scales[dst][i_scl])", ylabelsize=12, xlabelsize=12, xticklabelsize=9, yticklabelsize=9, xticklabelrotation=pi/2, titlesize=9)
                        for (i_y,ytgt) in enumerate(ytgts)
                            for i_anc = 1:Nanc
                                ost_counts[ilts[i_y,i_anc],i_y] += 1
                            end
                        end
                        hm = heatmap!(ax, -sdm.tu.*leadtimes, ytgts, ost_counts./sum(ost_counts; dims=1); colormap=:Reds)
                        #cbar = Colorbar(lout[1,2], hm; vertical=true, labelsiz=6)
                        (ost_mode,ost_mean,ost_mean_lo,ost_mean_hi,ost_q1,ost_q3) = (zeros(Float64,(Nytgt,)) for _=1:6)
                        for i_y = 1:Nytgt
                            ost_mean[i_y] = sum(leadtimes .* ost_counts[:,i_y]) / sum(ost_counts[:,i_y])
                            ost_mean_lo[i_y] = sum(leadtimes .* (leadtimes .< ost_mean[i_y]) .* ost_counts[:,i_y]) / sum((leadtimes .< ost_mean[i_y]) .* ost_counts[:,i_y])
                            ost_mean_hi[i_y] = sum(leadtimes .* (leadtimes .> ost_mean[i_y]) .* ost_counts[:,i_y]) / sum((leadtimes .> ost_mean[i_y]) .* ost_counts[:,i_y])
                            ost_mode[i_y] = leadtimes[argmax(ost_counts[:,i_y])]
                        end
                        lines!(ax, -sdm.tu.*(ost_mean), ytgts; color=:black, linestyle=:solid, label=L"$$Mean")
                        lines!(ax, -sdm.tu.*(ost_mean_lo), ytgts; color=:black, linestyle=(:dot,:dense),label=L"$$Trunc. mean")
                        lines!(ax, -sdm.tu.*(ost_mean_hi), ytgts; color=:black, linestyle=(:dot,:dense))
                        lines!(ax, -sdm.tu.*(ost_mode), ytgts; color=:cyan, linestyle=:solid, label=L"$$Mode")
                        axislegend(ax; labelsize=7, position=:lt)
                        save(joinpath(resultdir,"toast_$(dst)_$(rsp)_$(mc)_$(i_scl).png"), fig)
                    end
                end
            end
        end
    end

    if todo["compare_tvs"] == 1
        # Plot the TV achieved by (a) maximizing entropy, and (b) choosing a specific lead time, as a function of latitude. Do this with two vertically stacked plots
        dst = "b"
        rsp = "2"
        i_boot = 1

        @show Nleadtime

        tvs_lt = zeros(Float64, (Nytgt,Nleadtime,length(distn_scales[dst])))
        tvs_ent = zeros(Float64, (Nytgt,length(distn_scales[dst])))
        for (i_ytgt,ytgt) in enumerate(ytgts)
            resultdir_y = joinpath(exptdirs_COAST[i_ytgt],"results")
            tvs_lt[i_ytgt,:,:],tvs_ent[i_ytgt,:] = JLD2.jldopen(joinpath(resultdir_y,"ccdfs_regressed.jld2"),"r") do f
                return (
                        f["fdivs"][dst][rsp]["lt"]["tv"][i_boot,:,:],
                        f["fdivs"][dst][rsp]["ent"]["tv"][i_boot,1,:],
                       )
            end
        end

        xlims = [0, maximum(tvs_lt[1:end-1,:,:])]
        for i_scl = 1:length(distn_scales[dst])
            fig = Figure(size=(400,400))
            lout = fig[1,1] = GridLayout()
            rxystr = @sprintf("%.3f",cfgs[1].target_ryPerL*sdm.Ly)
            ax = Axis(lout[1,1]; xlabel="TV", ylabel="Target lat.", title=L"$$Box size %$(rxystr), Scale %$(distn_scales[dst][i_scl])")
            for (i_leadtime,leadtime) in enumerate(leadtimes)
                scatterlines!(ax, tvs_lt[:,i_leadtime,i_scl], ytgts; color=i_leadtime, colormap=:managua, colorrange=(0,Nleadtime), label="-$(sdm.tu*leadtimes[i_leadtime])")
            end
            scatterlines!(ax, tvs_ent[:,i_scl], ytgts; color=:red, linestyle=(:dot,:dense), linewidth=3, label="Max. Ent.")
            xlims!(ax, xlims...)
            lout[1,2] = Legend(fig, ax; framevisible=false, labelsize=15)
            save(joinpath(resultdir,"comparison_tv_$(i_scl).png"), fig)
        end
    end



    if todo["plot_mixcrits_ydep"] == 1
        i_scl = 1
        # for any criterion like ei or r2, we can stratify by that criterion andmeasure the other criteria (or measures of PDF agreement) conditionally 
        mixcrits_ancmean = Dict()
        fdivs = Dict()
        fdivs_ancs = Dict(fdivname=>zeros(Float64, (Nboot+1,Nytgt,)) for fdivname=fdivnames)
        for fdivname = fdivnames
            fdivs_ancs[fdivname] = zeros(Float64, (Nboot+1,Nytgt,))
        end
        gpdpars = Dict()
        gpdpars_ancs = Dict(key=>zeros(Float64,Nboot+1, Nytgt,) for key=("scale","shape"))
        gpdpars_dns = Dict(key=>zeros(Float64, Nytgt) for key=("scale","shape"))
        gpd_threshes = zeros(Float64, (Nboot+1,Nytgt))

        for dst = dsts
            mixcrits_ancmean[dst] = Dict()
            fdivs[dst] = Dict()
            gpdpars[dst] = Dict()
            for rsp = rsps
                mixcrits_ancmean[dst][rsp] = Dict()
                fdivs[dst][rsp] = Dict()
                gpdpars[dst][rsp] = Dict()
                for mc = keys(mixobjs)
                    mixcrits_ancmean[dst][rsp][mc] = zeros(Float64, (length(mixobjs[mc]), Nytgt))
                    fdivs[dst][rsp][mc] = Dict()
                    gpdpars[dst][rsp][mc] = Dict(key=>zeros(Float64, (Nboot+1,length(mixobjs[mc]),Nytgt)) for key=("scale","shape"))
                    for fdivname = fdivnames
                        fdivs[dst][rsp][mc][fdivname] = zeros(Float64, (Nboot+1,length(mixobjs[mc]), Nytgt))
                    end
                end
            end
        end
        for (i_y,y) in enumerate(ytgts)
            resultdir_y = joinpath(exptdirs_COAST[i_y],"results")
            mixcrits_y, mixobjs_y, iltmixs_y, fdivs_y, fdivs_ancs_y, gpdpar_y, gpdpar_ancs_y = JLD2.jldopen(joinpath(resultdir_y,"ccdfs_regressed.jld2"),"r") do f
                return (f["mixcrits"],f["mixobjs"],f["iltmixs"],f["fdivs"],f["fdivs_ancs"],f["gpdpar"],f["gpdpar_ancs"])
            end
            @show keys(gpdpar_ancs_y)
            gpdpars_ancs["scale"][:,i_y] .= gpdpar_ancs_y["scale_gpd"]
            gpdpars_ancs["shape"][:,i_y] .= gpdpar_ancs_y["shape_gpd"]
            (
             gpd_threshes[i_y],gpdpars_dns["scale"][i_y],gpdpars_dns["shape"][i_y]
            ) = (
                 JLD2.jldopen(joinpath(resultdir_y,"GPD_dns.jld2"),"r") do f
                     return (f["thresh"],f["scale"],f["shape"])
                 end
                )
            for fdivname = fdivnames
                fdivs_ancs[fdivname][:,i_y] .= fdivs_ancs_y[fdivname]
            end
            i_scl = 1
            for dst = dsts
                for rsp = rsps
                    if (dst == "g") && (rsp == "2")
                        continue
                    end
                    for mc = keys(mixobjs)
                        for (i_mcobj,mcobj) in enumerate(mixobjs[mc])
                            if any(iltmixs_y[dst][rsp][mc] .== 0)
                                error()
                            end
                            mixcrits_ancmean[dst][rsp][mc][i_mcobj,i_y] = SB.mean(mixcrits_y[dst][rsp][mc][iltmixs_y[dst][rsp][mc][i_mcobj,:,i_scl]])
                            for fdivname = fdivnames
                                lhs = fdivs[dst][rsp][mc][fdivname][:,i_mcobj,i_y]
                                rhs = fdivs_y[dst][rsp][mc][fdivname][:,i_mcobj,i_scl]
                                if !(size(lhs) == size(rhs))
                                    @show size(lhs)
                                    @show size(rhs)
                                    @show i_y,dst,rsp,mc,i_mcobj,fdivname
                                    error()
                                end
                                fdivs[dst][rsp][mc][fdivname][:,i_mcobj,i_y] .= fdivs_y[dst][rsp][mc][fdivname][:,i_mcobj,i_scl]
                            end
                            for gpdparname = ("scale","shape")
                                gpdpars[dst][rsp][mc][gpdparname][:,i_mcobj,i_y] .= gpdpar_y[dst][rsp][mc][gpdparname][:,i_mcobj,i_scl]
                            end
                        end
                    end
                end
            end
        end
        # Plot the GPD parameters
        for dst = dsts
            for rsp = rsps
                for mc = keys(mixobjs)
                    fig = Figure(size=(600,300))
                    lout = fig[1,1] = GridLayout()
                    ax1 = Axis(lout[1,1], xlabel=L"$\sigma$", ylabel=L"$y$")
                    lines!(ax1, gpdpars_dns["scale"], ytgts; color=:black, linestyle=(:dot,:dense))
                    ax2 = Axis(lout[1,2], xlabel=L"$\xi$", ylabel=L"$y$")
                    lines!(ax2, gpdpars_dns["shape"], ytgts; color=:black, linestyle=(:dot,:dense))
                    for i_mcobj = 1:length(mixobjs[mc])
                        colargs = Dict(:color=>i_mcobj,:colormap=>(length(mixobjs[mc]) > 1 ? :managua : :coolwarm),:colorrange=>(0,length(mixobjs[mc])))
                        scale_pt = gpdpars[dst][rsp][mc]["scale"][1,i_mcobj,:]
                        scale_lo = vec(mapslices(m->SB.quantile(m,0.05), gpdpars[dst][rsp][mc]["scale"][2:end,i_mcobj,:]; dims=1))
                        scale_hi = vec(mapslices(m->SB.quantile(m,0.95), gpdpars[dst][rsp][mc]["scale"][2:end,i_mcobj,:]; dims=1))
                        lines!(ax1,scale_pt,ytgts; colargs..., linewidth=1.5)
                        lines!(ax1,scale_lo,ytgts; colargs..., linewidth=0.5)
                        lines!(ax1,scale_hi,ytgts; colargs..., linewidth=0.5)
                        shape_pt = gpdpars[dst][rsp][mc]["shape"][1,i_mcobj,:]
                        shape_lo = vec(mapslices(m->SB.quantile(m,0.05), gpdpars[dst][rsp][mc]["shape"][2:end,i_mcobj,:]; dims=1))
                        shape_hi = vec(mapslices(m->SB.quantile(m,0.95), gpdpars[dst][rsp][mc]["shape"][2:end,i_mcobj,:]; dims=1))
                        lines!(ax2,shape_pt,ytgts; colargs..., linewidth=1.5)
                        lines!(ax2,shape_lo,ytgts; colargs..., linewidth=0.5)
                        lines!(ax2,shape_hi,ytgts; colargs..., linewidth=0.5)
                    end
                    lines!(ax1,gpdpars_ancs["scale"][1,:],ytgts,color=:dodgerblue,linewidth=1.5)
                    scale_pt = gpdpars_ancs["scale"][1,:]
                    scale_lo = vec(mapslices(m->SB.quantile(m,0.05),gpdpars_ancs["scale"][2:end,:]; dims=1))
                    scale_hi = vec(mapslices(m->SB.quantile(m,0.95),gpdpars_ancs["scale"][2:end,:]; dims=1))
                    lines!(ax1,scale_pt,ytgts,color=:dodgerblue,linewidth=1.5)
                    lines!(ax1,scale_lo,ytgts,color=:dodgerblue,linewidth=0.5)
                    lines!(ax1,scale_hi,ytgts,color=:dodgerblue,linewidth=0.5)
                    shape_pt = gpdpars_ancs["shape"][1,:]
                    shape_lo = vec(mapslices(m->SB.quantile(m,0.05),gpdpars_ancs["shape"][2:end,:]; dims=1))
                    shape_hi = vec(mapslices(m->SB.quantile(m,0.95),gpdpars_ancs["shape"][2:end,:]; dims=1))
                    lines!(ax2,shape_pt,ytgts,color=:dodgerblue,linewidth=1.5)
                    lines!(ax2,shape_lo,ytgts,color=:dodgerblue,linewidth=0.5)
                    lines!(ax2,shape_hi,ytgts,color=:dodgerblue,linewidth=0.5)
                    save(joinpath(resultdir,"gpd_$(dst)_$(rsp)_$(mc).png"),fig)
                end
            end
        end


        # Plot the F-divergences according to various criteria 
        fdivlabels = Dict("chi2"=>L"$\chi^2$", "kl"=>L"KL$$", "tv"=>L"TV$$")
        for dst = dsts
            for rsp = rsps
                for mc = keys(mixobjs)
                    mc_multiplier = (mc == "lt" ? -sdm.tu : 1)
                    for fdivname = fdivnames
                        if length(mixobjs[mc]) > 1
                            fig = Figure(size=(300,400))
                            lout = fig[1,1:2] = GridLayout()
                            ax = Axis(lout[1,1]; xlabel=mixcrit_labels[mc], ylabel="Target latitude", title=L"Mean %$(fdivlabels[fdivname]) grouped by %$(mixcrit_labels[mc])$$")
                            hm = heatmap!(ax, mc_multiplier.*mixobjs[mc], ytgts, fdivs[dst][rsp][mc][fdivname][1,:,:]; colormap=:managua)
                            cbar = Colorbar(lout[1,2], hm, vertical=true)
                            ax = Axis(lout[1,2]; xlabel=fdivlabels[fdivname], ylabelvisible=false, yticklabelsvisible=false)
                            save(joinpath(resultdir, "mixfidelity_$(mc)_$(dst)_$(rsp)_$(fdivname)_heatmap.png"), fig)
                        end
                        fig = Figure(size=(400,300))
                        lout = fig[1,1] = GridLayout()
                        ax = Axis(lout[1,1]; xlabel=fdivlabels[fdivname], ylabel="Target latitude", xscale=log10, xticklabelrotation=pi/2)
                        for (i_mcobj,mcobj) in enumerate(mixobjs[mc])
                            label = mixobj_labels[mc][i_mcobj]
                            lines!(ax, fdivs[dst][rsp][mc][fdivname][1,i_mcobj,:], ytgts; color=i_mcobj, colorrange=(0,length(mixobjs[mc])), colormap=(length(mixobjs[mc])>1 ? :managua : :coolwarm), label=label, alpha=0.5)
                        end
                        lines!(ax, fdivs_ancs[fdivname][1,:], ytgts; color=:black, linewidth=1.5, linestyle=(:dash,:dense), label="KDE")
                        xlims!(ax, extrema(filter(F->((F>0)&(isfinite(F))), vcat(vec(fdivs[dst][rsp][mc][fdivname][1,:,:]),fdivs_ancs[fdivname][1,:])))...)
                        lout[1,2] = Legend(fig, ax; framevisible=false, labelsize=6, title=mixobj_labels[mc])
                        colsize!(lout, 1, Relative(8/10))
                        save(joinpath(resultdir, "mixfidelity_$(mc)_$(dst)_$(rsp)_$(fdivname)_lines.png"), fig)
                    end
                end
            end
        end
    end
end

