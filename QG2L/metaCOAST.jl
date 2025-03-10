
function metaCOAST_latdep_boxsizedep_procedure(expt_supdir::String, resultdir_dns::String, i_expt=nothing)
    todo = Dict{String,Bool}(
                             "plot_ccdfs_latdep" =>              1,
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
    ytgts,rtgts = paramsets()
    Nytgts,Nrtgts = length(ytgts),length(rtgts)
    cfgs = Vector{ConfigCOAST}([])
    exptdirs_COAST = Vector{String}([])
    for (i_rtgt,rtgt) in enumerate(rtgts)
        for (i_ytgt,ytgt) in enumerate(ytgts)
            ConfigCOAST(
                           sdm.tu
                           ;
                           target_rxPerL=rtgt, 
                           target_yPerL=ytgt, 
                           target_ryPerL=rtgt,
                          )
            push!(cfgs, cfg)
        end
    end
    cfgs = reshape(cfgs, (Nytgt,Nrtgt))
    (
     leadtimes,r2threshes,dsts,rsps,mixobjs,
     mixcrit_labels,mixobj_labels,mixcrit_colors,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,
     i_thresh_cquantile,adjust_ccdf_per_ancestor
    ) = expt_config_COAST_analysis(cfgs[1],pertop)
    thresh_cquantile = ccdf_levels[i_thresh_cquantile]
    scales2plot = [1,4,8,12]
    threshstr = @sprintf("thrt%d", round(Int, 1/thresh_cquantile))
    for (i_y,y) in enumerate(ytgts)
        cfgstr = strrep_ConfigCOAST(cfgs[i_y])
        exptdir_COAST = joinpath(expt_supdir,"COAST_$(cfgstr)_$(pertopstr)_$(threshstr)")
        push!(exptdirs_COAST, exptdir_COAST)
    end
    println("Collected the cfgs")
end

            



function metaCOAST_latdep_procedure(expt_supdir::String, resultdir_dns::String; i_expt=nothing)
    todo = Dict{String,Bool}(
                             "plot_mixcrits_ydep" =>             1,
                             "compile_fdivs" =>                  1,
                             "plot_fdivs" =>                     1,
                             "plot_ccdfs_latdep" =>              1,
                             # danger zone
                             "remove_pngs" =>                    0,
                             # defunct/hibernating
                             "print_simtimes" =>                 0,
                             "plot_pot_ccdfs_latdep" =>          0,
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
    target_r, = expt_config_metaCOAST_latdep_analysis(i_expt=i_expt)
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
     mixcrit_labels,mixobj_labels,mixcrit_colors,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,
     i_thresh_cquantile,adjust_ccdf_per_ancestor
    ) = expt_config_COAST_analysis(cfgs[1],pertop)
    thresh_cquantile = ccdf_levels[i_thresh_cquantile]
    scales2plot = [1,4,8,12]
    threshstr = @sprintf("thrt%d", round(Int, 1/thresh_cquantile))
    for (i_y,y) in enumerate(ytgts)
        cfgstr = strrep_ConfigCOAST(cfgs[i_y])
        exptdir_COAST = joinpath(expt_supdir,"COAST_$(cfgstr)_$(pertopstr)_$(threshstr)")
        push!(exptdirs_COAST, exptdir_COAST)
    end
    println("Collected the cfgs")
    obj_label,short_obj_label = label_objective(cfgs[1])
    # Collect sizes of everything
    Nleadtime = length(leadtimes)
    Nlev = length(ccdf_levels)
    Nlev_exc = Nlev - i_thresh_cquantile + 1
    Nmcs = Dict(mc=>length(mixobjs[mc]) for mc=keys(mixobjs))
    Nscales = Dict(dst=>length(distn_scales[dst]) for dst=dsts)

    resultdir = joinpath(expt_supdir,"$(strrep_ConfigCOAST_varying_yPerL(cfgs[1]))_$(threshstr)")
    mkpath(resultdir)
    if todo["remove_pngs"]
        for filename = readdir(resultdir, join=true)
            if endswith(filename,"png") #&& occursin("ccdfs_pot_coast", filename)
                rm(filename)
            end
        end
    end




    if todo["plot_ccdfs_latdep"]
        Rccdfs_ancgen = zeros(Float64, (length(ccdf_levels),Nytgt))
        Rccdfs_valid = zeros(Float64, (length(ccdf_levels),Nytgt))
        Rmean_ancgen = zeros(Float64, Nytgt)
        Rmean_valid = zeros(Float64, Nytgt)

        (gpd_scale_valid,gpd_shape_valid,std_valid) = (zeros(Float64,Nytgt) for _=1:3)

        for (i_ytgt,ytgt) in enumerate(ytgts)
            cfgstr = strrep_ConfigCOAST(cfgs[i_ytgt])
            resultdir_COAST = joinpath(exptdirs_COAST[i_ytgt], "results")
            JLD2.jldopen(joinpath(resultdir_COAST, "objective_dns_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).jld2"), "r") do f
                Rccdfs_ancgen[:,i_ytgt] .= f["Rccdf_ancgen_seplon"][:,1]
                Rccdfs_valid[:,i_ytgt] .= f["Rccdf_valid_agglon"]
                Rmean_valid[i_ytgt] = SB.mean(f["Roft_valid_seplon"])
                Rmean_ancgen[i_ytgt] = SB.mean(f["Roft_ancgen_seplon"])
                gpd_scale_valid[i_ytgt] = f["gpdpar_valid_agglon"][1]
                gpd_shape_valid[i_ytgt] = f["gpdpar_valid_agglon"][2]
                std_valid[i_ytgt] = f["std_valid_agglon"]
            end

        end
        # ----------------- Latitude dependent quantiles -------------------
        colargs = Dict(:colormap=>Reverse(:roma), :colorrange=>(1,length(ccdf_levels)))
        fig = Figure(size=(600,400))
        lout = fig[1,1] = GridLayout()
        Rccdf_rough_intercept = 0
        Rccdf_rough_slope = 1.0
        Rccdf_rough = Rccdf_rough_intercept .+ Rccdf_rough_slope .* ytgts #0.5 .+ 0.5 .* ytgts
        ylims = (1.5*ytgts[1]-0.5*ytgts[2], 1.5*ytgts[Nytgt]-0.5*ytgts[Nytgt-1])
        topo_zonal_mean = vec(SB.mean(cop.topography[:,:,2], dims=1))
        axtopo = Axis(lout[1,1],xlabel="ℎ(𝑦₀)",ylabel="𝑦₀/𝐿", xgridvisible=false, ygridvisible=false, xticklabelrotation=pi/2, title="Topography", titlefont=:regular, limits=(extrema(topo_zonal_mean)..., ylims...))
        axmean = Axis(lout[1,2],xlabel="⟨𝑅⟩ − 𝑦₀/𝐿",ylabel="𝑦₀/𝐿", xgridvisible=false, ygridvisible=false, ylabelvisible=false, yticklabelsvisible=false, xticklabelrotation=pi/2, title="Mean", titlefont=:regular, limits=(-0.035,0.035,ylims...))
        axquants = Axis(lout[1,3],xlabel="μ[(½)ᵏ] − ⟨𝑅⟩",ylabel="𝑦₀/𝐿", xgridvisible=false, ygridvisible=false, ylabelvisible=false, yticklabelsvisible=false, xticklabelrotation=pi/2, title="(½)ᵏ-complementary quantiles, 𝑘∈{1,...,15}", titlefont=:regular, limits=(-0.02,0.3,ylims...))
        #axquantyders = Axis(lout[1,4], xlabel="Δ(𝑅-⟨𝑅⟩)/Δ⟨𝑅⟩", ylabel="𝑦₀/𝐿", xgridvisible=false, ygridvisible=false, ylabelvisible=false, yticklabelsvisible=false, xticklabelrotation=pi/2, title="Relative 𝑦-gradients", titlefont=:regular, limits=(0,2,ylims...))
        toplabel = "Intensities 𝑅, $(label_target(target_r,sdm))"
        Label(lout[1,2:3,Top()], toplabel, padding=(5.0,5.0,25.0,5.0), valign=:center, halign=:left, fontsize=15, font=:regular)
        lines!(axtopo, topo_zonal_mean, sdm.ygrid./sdm.Ly; color=:black)
        lines!(axmean, Rmean_valid .- Rccdf_rough, ytgts; color=:black, linestyle=(:dash,:dense), label="Long DNS")
        lines!(axmean, Rmean_ancgen .- Rccdf_rough, ytgts; color=:black, linestyle=:solid, label="Short DNS")
        ytgts_mid = (ytgts[2:end] .+ ytgts[1:end-1])./2
        for i_cl = reverse(1:length(ccdf_levels))
            cl = ccdf_levels[i_cl]
            lines!(axquants, Rccdfs_valid[i_cl,:].-Rmean_valid, ytgts; linewidth=2, linestyle=(:dash,:dense), color=i_cl, colargs..., label="Long DNS")
            lines!(axquants, Rccdfs_ancgen[i_cl,:].-Rmean_ancgen, ytgts; linewidth=1, linestyle=:solid, color=i_cl, colargs..., label="Short DNS")
            #lines!(axquantyders, diff(Rccdfs_valid[i_cl,:])./diff(Rmean_valid), ytgts_mid; linewidth=2, linestyle=(:dash,:dense), color=i_cl, colargs..., label="Long DNS")
            #lines!(axquantyders, diff(Rccdfs_ancgen[i_cl,:] .- Rmean_valid)./diff(Rmean_valid), ytgts_mid; linewidth=1, linestyle=:solid, color=i_cl, colargs..., label="Short DNS")
        end
        lines!(axquants, 1 .- Rmean_valid, ytgts; color=:grey60, linewidth=2, linestyle=(:dash,:dense))
        lines!(axquants, 0 .- Rmean_valid, ytgts; color=:grey60, linewidth=2, linestyle=(:dash,:dense))
        #vlines!(axquantyders, -1; color=:grey79, linewidth=3)
        #lout[2,:] = Legend(fig, axquants, "Exceedance probabilities (½)ᵏ, k ∈ {1,...,15}"; framevisible=true, titlefont=:regular, titlehalign=:left, merge=true, linecolor=:black, nbanks=2, labelsize=10, titlesize=10)
        axislegend(axmean, ; merge=true, linecolor=:black, position=:rt, framevisible=false, titlefont=:regular, labelsize=8)
        colsize!(lout, 1, Relative(1/12))
        colsize!(lout, 2, Relative(3/12))
        #colsize!(lout, 3, Relative(8/12))
        colgap!(lout, 1, 10)
        colgap!(lout, 2, 10)
        #colgap!(lout, 3, 10)
        save(joinpath(resultdir,"ccdfs_latdep_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph))_accpa$(Int(adjust_ccdf_per_ancestor)).png"),fig)

        # ---------- Plot GPD parameters along with moments ---------
        mssk = JLD2.jldopen(joinpath(resultdir_dns,"moments_mssk_$(cfgs[1].target_field[1:end-1]).jld2"),"r") do f
            return f["mssk_xall"][1,:,parse(Int,cfgs[1].target_field[end]),:]
        end
        fig = Figure(size=(600,400))
        lout = fig[1,1] = GridLayout()
        axargs = Dict(:ylabel=>"𝑦₀/𝐿", :xgridvisible=>false, :ygridvisible=>false, :xticklabelrotation=>pi/2, :titlefont=>:regular)
        axtopo = Axis(lout[1,1],xlabel="ℎ(𝑦₀)"; axargs..., title="Topography", limits=(extrema(topo_zonal_mean)..., ylims...))
        axargs[:ylabelvisible] = axargs[:yticklabelsvisible] = false
        axstd = Axis(lout[1,2]; xlabel="Std. Dev.", axargs..., limits=(0, maximum(mssk[:,2])*1.01, ylims...))
        axscale = Axis(lout[1,3]; xlabel="GPD scale σ", axargs..., limits=(0,0.035,ylims...))
        axshape = Axis(lout[1,4]; xlabel="GPD shape ξ", axargs..., limits=(-0.5,0.05,ylims...))
        threshcquantstr = @sprintf("%.2E",thresh_cquantile)
        toplabel = @sprintf("Threshold μ[%s], exceedance probability %s = %.2E", powerofhalfstring(i_thresh_cquantile), powerofhalfstring(i_thresh_cquantile), thresh_cquantile)
        Label(lout[1,2:4,Top()], toplabel, padding=(5.0,5.0,5.0,5.0), valign=:bottom, halign=:right, fontsize=15, font=:regular)
        # Topography
        lines!(axtopo, topo_zonal_mean, sdm.ygrid./sdm.Ly; color=:black)
        # Std. Dev.
        lines!(axstd, mssk[:,2], sdm.ygrid./sdm.Ly; color=:black, linestyle=(:dash,:dense), label="(1/$(2*sdm.Ny))𝐿")
        scatterlines!(axstd, std_valid, ytgts; color=:black, label="($(round(Int, target_r*sdm.Ny))/$(sdm.Ny))/𝐿")
        axislegend(axstd, "Box radius"; position=:lc, labelsize=10, titlesize=12, titlefont=:regular, framevisible=false)
        # scale
        scatterlines!(axscale, gpd_scale_valid, ytgts; color=:black)
        # shape
        vlines!(axshape, 0.0; color=:grey60, linewidth=2, linestyle=(:dash,:dense))
        scatterlines!(axshape, gpd_shape_valid, ytgts; color=:black)

        colsize!(lout, 2, Relative(6/16))
        colsize!(lout, 3, Relative(4/16))
        colsize!(lout, 4, Relative(4/16))
        colgap!(lout, 1, 10.0)
        colgap!(lout, 2, 10.0)
        colgap!(lout, 3, 10.0)
        save(joinpath(resultdir,"gpdpars_latdep_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).png"),fig)
    end



    if todo["print_simtimes"]
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
    


    if todo["compile_fdivs"] 
        # Plot the TV achieved by (a) maximizing entropy, and (b) choosing a specific lead time, as a function of latitude. Do this with two vertically stacked plots
        dsts = ("b",)
        rsps = ("2","e")
        i_boot = 1

        @show Nleadtime

        fdivs = Dict()
        fdivpools = Dict()
        for dst = dsts
            fdivs[dst] = Dict()
            fdivpools[dst] = Dict()
            for rsp = rsps
                fdivs[dst][rsp] = Dict()
                fdivpools[dst][rsp] = Dict()
                for mc = ["contcorr","globcorr","ei","pim","pth","ent","lt","r2"]
                    fdivs[dst][rsp][mc] = Dict()
                    fdivpools[dst][rsp][mc] = Dict()
                    for fdivname = fdivnames
                        fdivs[dst][rsp][mc][fdivname] = zeros(Float64, (Nytgt,Nboot+1,Nmcs[mc],Nscales[dst]))
                        fdivpools[dst][rsp][mc][fdivname] = zeros(Float64, (Nytgt,Nboot+1,Nmcs[mc],Nscales[dst]))
                    end
                end
            end
        end
        fdivs_ancgen_valid = Dict()
        for fdivname = fdivnames
            fdivs_ancgen_valid[fdivname] = zeros(Float64, (Nytgt,div(sdm.Nx,xstride_valid_dns)))
        end
        for (i_ytgt,ytgt) in enumerate(ytgts)
            resultdir_y = joinpath(exptdirs_COAST[i_ytgt],"results")
            JLD2.jldopen(joinpath(resultdir_y,"ccdfs_regressed_accpa$(Int(adjust_ccdf_per_ancestor)).jld2"),"r") do f
                for dst = dsts
                    for rsp = rsps
                        for mc = ["contcorr","globcorr","ei","pim","pth","ent","lt","r2"]
                            for fdivname = fdivnames
                                fdivs[dst][rsp][mc][fdivname][i_ytgt,:,:,:] .= f["fdivs"][dst][rsp][mc][fdivname][:,:,:]
                                fdivpools[dst][rsp][mc][fdivname][i_ytgt,:,:,:] .= f["fdivpools"][dst][rsp][mc][fdivname][:,:,:]
                            end
                        end
                    end
                end
                for fdivname = fdivnames
                    fdivs_ancgen_valid[fdivname][i_ytgt,:] .= f["fdivs_ancgen_valid"][fdivname]
                end
            end
        end

        for filename = readdir(resultdir,join=true)
            if endswith(filename,"fdivs.jld2")
                rm(filename)
            end
        end
        JLD2.jldopen(joinpath(resultdir,"fdivs_accpa$(Int(adjust_ccdf_per_ancestor)).jld2"),"w") do f
            f["fdivs"] = fdivs
            f["fdivpools"] = fdivpools
            f["fdivs_ancgen_valid"] = fdivs_ancgen_valid
        end
    else
        fdivs,fdivpools,fdivs_ancgen_valid = JLD2.jldopen(joinpath(resultdir,"fdivs_accpa$(Int(adjust_ccdf_per_ancestor)).jld2"),"r") do f
            return f["fdivs"],f["fdivpools"],f["fdivs_ancgen_valid"]
        end
    end

    #IFT.@infiltrate
    #

    if todo["plot_fdivs"]

        dsts = ("b",)
        rsps = ("e",)
        i_boot = 1

        fdivs2plot = ["qrmse","kl","chi2","tv"]

        fdivlabels = Dict("qrmse"=>"𝐿²","tv"=>"TV","chi2"=>"χ²","kl"=>"KL")
        for fdivname = fdivs2plot
            fdivs_ancgen_valid_pt,fdivs_ancgen_valid_lo,fdivs_ancgen_valid_hi = let
                fdav = fdivs_ancgen_valid[fdivname]
                (SB.mean(fdav; dims=2)[:,1], (QG2L.quantile_sliced(fdav, q, 2)[:,1] for q=(0.05,0.95))...)
            end
            for dst = dsts
                for rsp = rsps
                    for i_scl = scales2plot
                        scalestr = @sprintf("Scale %.3f", distn_scales[dst][i_scl])
                        syncmcs = ["lt","contcorr","globcorr"]
                        fig = Figure(size=(600,400))
                        lout = fig[1,1] = GridLayout()
                        ax = Axis(lout[1,2], xlabel=fdivlabels[fdivname], ylabel="𝑦₀/𝐿", title="$(label_target(target_r,sdm)), $(scalestr)\nthreshold exc. prob. $(powerofhalfstring(i_thresh_cquantile))", titlevisible=true, titlefont=:regular, xscale=log10, xgridvisible=false, ygridvisible=false)
                        # Short simulation
                        band!(ax, Point2f.(fdivs_ancgen_valid_lo,ytgts), Point2f.(fdivs_ancgen_valid_hi,ytgts); color=:gray, alpha=0.25)
                        lines!(ax, fdivs_ancgen_valid_pt, ytgts; color=:black, linewidth=4, label="Short DNS\n(90% CI)")
                        # All desired mixing criteria
                        for (i_syncmc,syncmc) in enumerate(syncmcs)
                            idx_mcobj_best = mapslices(argmin, fdivs[dst][rsp][syncmc][fdivname][:,i_boot,:,i_scl]; dims=2)[:,1]
                            scatterlines!(ax, [fdivs[dst][rsp][syncmc][fdivname][i_ytgt,i_boot,idx_mcobj_best[i_ytgt],i_scl] for i_ytgt=1:Nytgt], ytgts; color=mixcrit_colors[syncmc], linestyle=:solid, label="Optimal $(mixcrit_labels[syncmc])")
                            idx_mcobj_best_pool = mapslices(argmin, fdivpools[dst][rsp][syncmc][fdivname][:,i_boot,:,i_scl]; dims=2)[:,1]
                            scatterlines!(ax, [fdivpools[dst][rsp][syncmc][fdivname][i_ytgt,i_boot,idx_mcobj_best_pool[i_ytgt],i_scl] for i_ytgt=1:Nytgt], ytgts; color=mixcrit_colors[syncmc], linestyle=(:dot,:dense), #=label="Optimal $(mixcrit_labels[syncmc])"=#)
                            # Max-entropy
                            #lines!(ax, fdivs[dst][rsp]["ent"][fdivname][:,i_boot,1,i_scl], ytgts; color=:red, linewidth=3, label=mixobj_labels["ent"][1], alpha=0.5)
                            # Max-EI 
                        end
                        scatterlines!(ax, fdivs[dst][rsp]["ei"][fdivname][:,i_boot,1,i_scl], ytgts; color=mixcrit_colors["ei"], linewidth=1, linestyle=:solid, label=mixobj_labels["ei"][1], alpha=1.0)
                        scatterlines!(ax, fdivpools[dst][rsp]["ei"][fdivname][:,i_boot,1,i_scl], ytgts; color=mixcrit_colors["ei"], linewidth=1, linestyle=(:dot,:dense), #=label=mixobj_labels["ei"][1],=# alpha=1.0)
                        scatterlines!(ax, fdivs[dst][rsp]["ent"][fdivname][:,i_boot,1,i_scl], ytgts; color=mixcrit_colors["ent"], linewidth=1, linestyle=:solid, label=mixobj_labels["ent"][1], alpha=1.0)
                        scatterlines!(ax, fdivpools[dst][rsp]["ent"][fdivname][:,i_boot,1,i_scl], ytgts; color=mixcrit_colors["ent"], linewidth=1, linestyle=(:dot,:dense), #=label=mixobj_labels["ent"][1],=# alpha=1.0)
                        lout[1,1] = Legend(fig, ax; labelsize=10, framevisible=false)
                        colsize!(lout, 2, Relative(4/5))

                        save(joinpath(resultdir,"fdivofy_$(fdivname)_$(dst)_$(rsp)_$(i_scl)_accpa$(Int(adjust_ccdf_per_ancestor)).png"), fig)
                        # TODO Do a parallel figure bit with leadtime and committor as the independent variables, fdiv as the dependent variable 9just a slice of the phdgms from the one-latitude cases)
                    end
                end
            end
        end
    end





    if todo["plot_mixcrits_ydep"]
        dst = "b"
        rsp = "e"
        i_boot = 1
        fdivname = "qrmse"
        fdivlabel = "𝐿²"
        mc = "ent" # this is the privileged mixing criterion on which to optimize 
        softmean(x,Eofx,beta) = SB.mean(x, SB.weights(exp.(-Eofx.*beta)))
        for i_scl = scales2plot
            (pim_of_ast,contcorr_of_ast,fdiv_of_ast,mc_of_ast) = (zeros(Float64, (Nleadtime,Nytgt)) for _=1:4)
            (ast_of_contcorr,fdiv_of_contcorr,mc_of_contcorr) = (zeros(Float64, (Nmcs["contcorr"],Nytgt)) for _=1:3)
            (ast_of_pim,fdiv_of_pim,mc_of_pim) = (zeros(Float64, (Nmcs["pim"],Nytgt)) for _=1:3)
            (idx_ast_best,idx_contcorr_best) = (mapslices(argmin, fdivs[dst][rsp][syncmc][fdivname][:,i_boot,:,i_scl]; dims=2)[:,1] for syncmc=("lt","contcorr"))
            for (i_ytgt,ytgt) in enumerate(ytgts)
                JLD2.jldopen(joinpath(exptdirs_COAST[i_ytgt],"results","ccdfs_regressed_accpa$(Int(adjust_ccdf_per_ancestor)).jld2"),"r") do f
                    Nancy = size(f["mixcrits"][dst][rsp]["lt"],2)
                    # AST as independent variable
                    contcorr_of_ast[:,i_ytgt] .= SB.mean(f["mixcrits"][dst][rsp]["contcorr"][1:Nleadtime,1:Nancy,i_scl]; dims=2)[:,1]
                    pim_of_ast[:,i_ytgt] .= SB.mean(f["mixcrits"][dst][rsp]["pim"][1:Nleadtime,1:Nancy,i_scl]; dims=2)[:,1]
                    mc_of_ast[:,i_ytgt] .= SB.mean(f["mixcrits"][dst][rsp][mc][1:Nleadtime,1:Nancy,i_scl]; dims=2)[:,1]
                    fdiv_of_ast[:,i_ytgt] .= (f["fdivs"][dst][rsp]["lt"][fdivname][i_boot,1:Nleadtime,i_scl])
                    # pim as independent variable
                    ilts = f["iltmixs"][dst][rsp]["pim"][1:Nmcs["pim"],1:Nancy,i_scl]
                    ast_of_pim[:,i_ytgt] .= sdm.tu.*SB.mean(leadtimes[ilts]; dims=2)[:,1]
                    fdiv_of_pim[:,i_ytgt] .= (f["fdivs"][dst][rsp]["pim"][fdivname][i_boot,1:Nmcs["pim"],i_scl])
                    mc_of_pim[:,i_ytgt] .= [
                                            SB.mean([f["mixcrits"][dst][rsp][mc][ilts[i_pim,i_anc],i_anc,i_scl] for i_anc=1:Nancy]) 
                                            for i_pim=1:Nmcs["pim"]
                                           ]
                    # contcorr as independent variable
                    ilts = f["iltmixs"][dst][rsp]["contcorr"][1:Nmcs["contcorr"],1:Nancy,i_scl]
                    ast_of_contcorr[:,i_ytgt] .= sdm.tu.*SB.mean(leadtimes[ilts]; dims=2)[:,1]
                    fdiv_of_contcorr[:,i_ytgt] .= (f["fdivs"][dst][rsp]["contcorr"][fdivname][i_boot,1:Nmcs["contcorr"],i_scl])
                    mc_of_contcorr[:,i_ytgt] .= [
                                            SB.mean([f["mixcrits"][dst][rsp][mc][ilts[i_contcorr,i_anc],i_anc,i_scl] for i_anc=1:Nancy]) 
                                            for i_contcorr=1:Nmcs["contcorr"]
                                           ]
                end
            end
            # --------------- Show fdiv as a function of (AST,contcorr) -----------
            #
            #
            colormap = :deep
            normalize_by_latitude = false
            logscale_flag = (mc != "ent")
            if logscale_flag
                for arr = (
                           fdiv_of_ast,fdiv_of_contcorr,fdiv_of_pim,
                           mc_of_ast,mc_of_contcorr,mc_of_pim
                          )
                    arr .= log10.(arr)
                end
                if normalize_by_latitude
                    errlabel = "log($(fdivlabel)/max $(fdivlabel)|y)"
                    mclabel = "log($(mixcrit_labels[mc])/max $(mixcrit_labels[mc])|y)"
                else
                    errlabel = "log₁₀($(fdivlabel))"
                    mclabel = "log₁₀($(mixcrit_labels[mc]))"
                end
            else
                errlabel = fdivlabel
                mclabel = mixcrit_labels[mc]
            end
            colorscale = identity
            fdivrange = [minimum(minimum.([fdiv_of_ast,fdiv_of_contcorr,fdiv_of_pim])),maximum(maximum.([fdiv_of_ast,fdiv_of_contcorr,fdiv_of_pim]))]
            mcrange = (minimum(minimum.([mc_of_ast,mc_of_contcorr,mc_of_pim])),maximum(maximum.([mc_of_ast,mc_of_contcorr,mc_of_pim])))
            if normalize_by_latitude
                for arr = (
                           fdiv_of_ast,fdiv_of_contcorr,fdiv_of_pim,
                           mc_of_ast,mc_of_contcorr,mc_of_pim
                          )
                    arr .= (arr .- minimum(arr; dims=1)) ./ (maximum(arr; dims=1) .- minimum(arr; dims=1))
                end
                colorrange_fdiv = (0,1)
                colorrange_mc = (0,1)
            else
                colorrange_fdiv = fdivrange
                colorrange_mc = mcrange
            end
            beta = 12.0
            ast_softbest = mapslices(F->softmean(leadtimes, F, beta), fdiv_of_ast; dims=1)[1,:]
            transcontcorr_softbest = mapslices(F->softmean(transcorr.(mixobjs["contcorr"]), F, beta), fdiv_of_contcorr; dims=1)[1,:]


            fig = Figure(size=(400,350))
            lout = fig[1,1] = GridLayout()
            axargs = Dict(:titlefont=>:regular, :xlabelsize=>8, :xticklabelsize=>6, :ylabelsize=>8, :yticklabelsize=>6)
            cbarargs = Dict(:labelfont=>:regular, :labelsize=>8, :ticklabelsize=>6, :valign=>:bottom, :size=>6)
            threshcquantstr = @sprintf("%.2E",thresh_cquantile)
            toplabel = "$(label_target(target_r, sdm)), scale $(distn_scales[dst][i_scl]), threshold exc. prob. $(powerofhalfstring(i_thresh_cquantile))=$(threshcquantstr)"
            lab = Label(lout[1,1:2], toplabel, padding=(0.0,0.0,0.0,0.0), valign=:center, halign=:center, fontsize=8, font=:regular)
            ax1 = Axis(lout[3,1]; xlabel="−AST", ylabel="𝑦₀/𝐿", title="", axargs...)
            axargs[:ylabelvisible] = axargs[:yticklabelsvisible] = false
            ax2 = Axis(lout[3,2]; xlabel=@sprintf("σ⁻¹(%s)", mixcrit_labels["contcorr"]), ylabel="𝑦₀/𝐿", title="", axargs...)
            leadtime_bounds = tuple((-sdm.tu .* [1.5*leadtimes[end]-0.5*leadtimes[end-1], 1.5*leadtimes[1]-0.5*leadtimes[2]])...)
            # First heatmap: AST as independent variable
            hm1 = heatmap!(ax1, -sdm.tu.*reverse(leadtimes), ytgts, reverse(fdiv_of_ast; dims=1); colormap=colormap, colorscale=colorscale, colorrange=colorrange_fdiv)
            co1 = contour!(ax1, -sdm.tu.*reverse(leadtimes), ytgts, reverse(mc_of_ast; dims=1); levels=range(mcrange...; length=7), colormap=:heat, labels=false)
            scatter!(ax1, -sdm.tu.*leadtimes[idx_ast_best], ytgts; color=:black)
            #lines!(ax1, -sdm.tu.*ast_softbest, ytgts; color=:black, linewidth=2, linestyle=(:dash,:dense))
            #co1pim = contour!(ax1, -leadtimes.*sdm.tu, ytgts, reverse(pim_of_ast; dims=1); color=:black, linestyle=(:dot,:dense), labels=false)
            cbar1 = Colorbar(lout[2,1], hm1; vertical=false, label="$(errlabel) (iso-$(mixcrit_labels["lt"]))", cbarargs...)
            # Second heatmap: contcorr as independent variable
            hm2 = heatmap!(ax2, transcorr.(mixobjs["contcorr"]), ytgts, fdiv_of_contcorr; colormap=colormap, colorscale=colorscale, colorrange=colorrange_fdiv) 
            co2 = contour!(ax2, transcorr.(mixobjs["contcorr"]), ytgts, mc_of_contcorr; levels=range(mcrange...; length=7), colormap=:heat, labels=false) 
            scatter!(ax2, transcorr.(mixobjs["contcorr"][idx_contcorr_best]), ytgts; color=:black)
            cbar2 = Colorbar(lout[2,2], hm2; vertical=false, label="$(errlabel) (iso-$(mixcrit_labels["contcorr"])", cbarargs...)
            rowgap!(lout, 1, 0)
            rowgap!(lout, 2, 5)
            colgap!(lout, 1, 0)

            rowsize!(lout, 1, Relative(1/9))
            #rowsize!(lout, 2, Relative(2/9))
            rowsize!(lout, 3, Relative(2/3))

            resize_to_layout!(fig)
            save(joinpath(resultdir,"phdgm_ast_contcorr_$(dst)_$(rsp)_$(i_scl)_$(fdivname)_accpa$(Int(adjust_ccdf_per_ancestor)).png"), fig)

            # --------------- Show EI as a function of (AST,contcorr) -----------
            beta = -12.0
            fig = Figure(size=(400,350), )
            lout = fig[1,1] = GridLayout()
            idx_maxei_ast,idx_maxei_contcorr = (mapslices(argmax, mcindep; dims=1)[1,:] for mcindep=(mc_of_ast,mc_of_contcorr))
            ast_softmaxei = mapslices(F->softmean(leadtimes, F, beta), mc_of_ast, dims=1)[1,:]
            transcontcorr_softmaxei = mapslices(F->softmean(transcorr.(mixobjs["contcorr"]), F, beta), mc_of_contcorr; dims=1)[1,:]
            axargs = Dict(:titlefont=>:regular, :xlabelsize=>8, :xticklabelsize=>6, :ylabelsize=>8, :yticklabelsize=>6)
            cbarargs = Dict(:labelfont=>:regular, :labelsize=>8, :ticklabelsize=>6, :valign=>:bottom)
            toplabel = "$(label_target(target_r, sdm)), scale $(distn_scales[dst][i_scl]), threshold exc. prob. $(powerofhalfstring(i_thresh_cquantile))=$(threshcquantstr)"
            Label(lout[1,1:2], toplabel, padding=(0.0,0.0,0.0,0.0), valign=:center, halign=:center, fontsize=8, font=:regular)
            ax1 = Axis(lout[3,1]; xlabel="−AST", ylabel="𝑦₀/𝐿", title="", axargs...)
            axargs[:ylabelvisible] = axargs[:yticklabelsvisible] = false
            ax2 = Axis(lout[3,2]; xlabel=@sprintf("σ⁻¹(%s)", mixcrit_labels["contcorr"]), ylabel="𝑦₀/𝐿", title="", axargs...)
            threshcquantstr = @sprintf("%.2E",thresh_cquantile)
            leadtime_bounds = tuple((-sdm.tu .* [1.5*leadtimes[end]-0.5*leadtimes[end-1], 1.5*leadtimes[1]-0.5*leadtimes[2]])...)
            # First heatmap: AST as independent variable
            hm1 = heatmap!(ax1, -sdm.tu.*reverse(leadtimes), ytgts, reverse(mc_of_ast; dims=1); colormap=Reverse(colormap), colorscale=colorscale, colorrange=colorrange_mc)
            scatterlines!(ax1, -sdm.tu.*leadtimes[idx_maxei_ast], ytgts; color=:black, linewidth=2)
            #lines!(ax1, -sdm.tu.*ast_softmaxei, ytgts; color=:black, linewidth=2, linestyle=(:dash,:dense))
            cbar1 = Colorbar(lout[2,1], hm1; vertical=false, label="$(mclabel) (iso-$(mixcrit_labels["lt"]))", cbarargs...)
            # Second heatmap: contcorr as independent variable
            hm2 = heatmap!(ax2, transcorr.(mixobjs["contcorr"]), ytgts, mc_of_contcorr; colormap=Reverse(colormap), colorscale=colorscale, colorrange=colorrange_mc) 
            scatterlines!(ax2, transcorr.(mixobjs["contcorr"][idx_maxei_contcorr]), ytgts; color=:black, linewidth=2)
            #lines!(ax2, transcontcorr_softmaxei, ytgts; color=:black, linewidth=2, linestyle=(:dash,:dense))
            cbar2 = Colorbar(lout[2,2], hm2; vertical=false, label="$(mclabel) (iso-$(mixcrit_labels["contcorr"]))", cbarargs...)
            #rowsize!(lout, 0, Relative(1/8))
            rowgap!(lout, 1, 0)
            rowgap!(lout, 2, 5)
            colgap!(lout, 1, 0)

            rowsize!(lout, 1, Relative(1/9))
            # rowsize!(lout, 2, Relative(2/9)) # TODO understand why uncommenting this line messes up all the proportions
            rowsize!(lout, 3, Relative(2/3))
            resize_to_layout!(fig)
            @infiltrate !(all(isfinite.(mc_of_ast)) && all(isfinite.(mc_of_contcorr)))
            save(joinpath(resultdir,"phdgm_ast_contcorr_$(dst)_$(rsp)_$(i_scl)_$(mc)_accpa$(Int(adjust_ccdf_per_ancestor)).png"), fig)
        end
    end
    if todo["plot_pot_ccdfs_latdep"]
        # Two panels: (1) compare valid to ancgen, (2) compare valid to coast 
        ccdfs_pot_valid,ccdfs_pot_ancgen = (zeros(Float64, (Nytgt,Nlev_exc)) for _=1:3)
        for (i_ytgt,ytgt) in enumerate(ytgts)
            cfgstr = strrep_ConfigCOAST(cfgs[i_ytgt])
            resultdir_COAST = joinpath(exptdirs_COAST[i_ytgt], "results")
            JLD2.jldopen(joinpath(resultdir_COAST, "objective_dns_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).jld2"), "r") do f
                ccdfs_pot_ancgen[i_ytgt,:] .= f["ccdf_pot_ancgen_seplon"][:,1]
                ccdfs_pot_valid[i_ytgt,:] .= f["ccdf_pot_valid_agglon"]
            end
        end
        potccdfmixs = Dict()
        for dst = dsts
            potccdfmixs[dst] = Dict()
            for rsp = rsps
                potccdfmixs[dst][rsp] = Dict()
                for mc = ["pim","pth","ent",]
                    potccdfmixs[dst][rsp][mc] = zeros(Float64, (Nytgt,Nlev_exc,Nmcs[mc],Nscales[dst]))
                end
            end
        end
        i_boot = 1
        for (i_ytgt,ytgt) in enumerate(ytgts)
            resultdir_COAST = joinpath(exptdirs_COAST[i_ytgt], "results")
            JLD2.jldopen(joinpath(resultdir_COAST, "ccdfs_regressed_accpa$(Int(adjust_ccdf_per_ancestor)).jld2"),"r") do f
                for dst = dsts
                    for rsp = rsps
                        for mc = ["pim","ent","pth"]
                            potccdfmixs[dst][rsp][mc][i_ytgt,1:Nlev_exc,1:Nmcs[mc],1:Nscales[dst]] .= f["ccdfmixs"][dst][rsp][mc][i_thresh_cquantile:end, i_boot, :, :]
                        end
                    end
                end
            end
        end

        for dst = ["b",]
            for rsp = ["e"]
                for mc = ["ent","pth","pim"]
                    for fdivname = ["qrmse"]
                        for i_scl = [1,4,8,12]
                            idx_mcobj = mapslices(argmin, fdivs[dst][rsp][mc][fdivname][:,i_boot,:,i_scl]; dims=2)
                            scalestr = @sprintf("Scale %.3f", distn_scales[dst][i_scl])
                            fig = Figure(size=(500,1000))
                            lout = fig[1,1] = GridLayout()
                            title_suffixes = ["Short DNS","best $(mc_labels[mc])"]
                            axag,axcoast = [
                                            Axis(lout[i,1], 
                                                 xlabel="Peak-adjusted log₂(exceedance probability)", ylabel="𝑦₀/𝐿", title="$(label_target(target_r,sdm)), $(scalestr), $(title_suffixes[i])", 
                                                 titlefont=:regular, xgridvisible=false, ygridvisible=false,
                                                 xscale=log2, xticks=(ccdf_levels[i_thresh_cquantile:end], string.(-i_thresh_cquantile:-1:-Nlev))
                                                )
                                            for i=1:2
                                           ]
                            colargs = Dict(:colormap=>Reverse(:roma), :colorrange=>(i_thresh_cquantile,length(ccdf_levels)))

                            for ax = (axag,axcoast)
                                for i_lev = 1:Nlev_exc
                                    #lines!(ax, thresh_cquantile.*ccdfs_pot_valid[:,i_lev], ytgts; color=i_lev+i_thresh_cquantile-1, colargs..., linestyle=(:dash,:dense))
                                    vlines!(ax, ccdf_levels[i_thresh_cquantile+i_lev-1]; color=i_lev+i_thresh_cquantile-1, colargs..., linestyle=(:dash,:dense))
                                end
                            end
                            epsilon = ccdf_levels[end]/2
                            for i_lev = 1:Nlev_exc
                                lines!(axag, 
                                       max.(epsilon, 
                                             ccdf_levels[i_thresh_cquantile+i_lev-1]
                                             .*ccdfs_pot_ancgen[:,i_lev]
                                             ./ccdfs_pot_valid[:,i_lev], 
                                            ),
                                        ytgts; color=i_lev+i_thresh_cquantile-1, colargs..., linestyle=:solid
                                      )
                                lines!(axcoast, 
                                       max.(epsilon, 
                                            ccdf_levels[i_thresh_cquantile+i_lev-1]
                                            .*[potccdfmixs[dst][rsp][mc][i_ytgt,i_lev,idx_mcobj[i_ytgt],i_scl] for i_ytgt=1:Nytgt]
                                            ./(thresh_cquantile.*ccdfs_pot_valid[:,i_lev])
                                           ),
                                       ytgts; color=i_lev+i_thresh_cquantile-1, colargs..., linestyle=:solid)
                            end
                            for ax = (axag,axcoast)
                                xlims!(ax, epsilon/2, thresh_cquantile*1.01)
                            end
                            linkxaxes!(axag, axcoast)
                            save(joinpath(resultdir,"ccdfs_pot_coast_$(dst)_$(rsp)_$(mc)_$(i_scl)_min$(fdivname)_accpa$(Int(adjust_ccdf_per_ancestor)).png"), fig)
                        end
                    end
                end
            end
        end
    end
end

