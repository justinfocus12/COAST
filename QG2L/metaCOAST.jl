


function metaCOAST_latdep_procedure(expt_supdir::String, resultdir_dns::String; i_expt=nothing)
    todo = Dict{String,Bool}(
                             "plot_mixcrits_ydep" =>             1,
                             "compile_fdivs" =>                  1,
                             "plot_fdivs" =>                     1,
                             "plot_ccdfs_latdep" =>              1,
                             # danger zone
                             "remove_pngs" =>                    1,
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
                gpd_scale_valid[i_ytgt] = f["gpdpar_valid_agglon"][1]
                gpd_shape_valid[i_ytgt] = f["gpdpar_valid_agglon"][2]
                std_valid[i_ytgt] = f["std_valid_agglon"]
            end

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
        xmin = minimum(Rccdfs_valid.-Rccdf_rough') - 0.02
        xmax = maximum(Rccdfs_valid.-Rccdf_rough') + 0.02
        xlims!(ax, xmin, xmax)
        ylims!(ax, 0.0, 1.0)
        lout[1,2] = Legend(fig, ax, "Exceedance probabilities\n(½)ᵏ, k ∈ {1,...,15}"; framevisible=false, titlefont=:regular, titlehalign=:left, merge=true, linecolor=:black)
        save(joinpath(resultdir,"ccdfs_latdep_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph))_accpa$(Int(adjust_ccdf_per_ancestor)).png"),fig)

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
        xlims!(axscale, -0.005, 0.035)
        # shape
        vlines!(axshape, 0.0; color=:gray, alpha=0.5, linewidth=3)
        scatterlines!(axshape, gpd_shape_valid, ytgts; color=:black)
        xlims!(axshape, -1, 0.25)

        colgap!(lout, 1, 0.0)
        colgap!(lout, 2, 0.0)
        for ax = (axstd,axscale,axshape)
            ylims!(ax, 0.0, 1.0)
        end
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
        for dst = dsts
            fdivs[dst] = Dict()
            for rsp = rsps
                fdivs[dst][rsp] = Dict()
                for mc = ["ei","pim","pth","ent","lt","r2"]
                    fdivs[dst][rsp][mc] = Dict()
                    for fdivname = fdivnames
                        fdivs[dst][rsp][mc][fdivname] = zeros(Float64, (Nytgt,Nboot+1,Nmcs[mc],Nscales[dst]))
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
                        for mc = ["ei","pim","pth","ent","lt","r2"]
                            for fdivname = fdivnames
                                fdivs[dst][rsp][mc][fdivname][i_ytgt,:,:,:] .= f["fdivs"][dst][rsp][mc][fdivname][:,:,:]
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
            f["fdivs_ancgen_valid"] = fdivs_ancgen_valid
        end
    else
        fdivs,fdivs_ancgen_valid = JLD2.jldopen(joinpath(resultdir,"fdivs_accpa$(Int(adjust_ccdf_per_ancestor)).jld2"),"r") do f
            return f["fdivs"],f["fdivs_ancgen_valid"]
        end
    end

    #IFT.@infiltrate
    #

    if todo["plot_fdivs"]

        dsts = ("b",)
        rsps = ("e",)
        i_boot = 1

        fdivs2plot = ["qrmse",]

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
                        threshcqstr = 
                        fig = Figure(size=(500,700))
                        lout = fig[1,1] = GridLayout()
                        axs = [Axis(lout[i_syncmc,1], xlabel=fdivlabels[fdivname], ylabel="(Target 𝑦)/L", title="$(label_target(target_r,sdm)), $(scalestr)\nthreshold exc. prob. $(powerofhalfstring(i_thresh_cquantile))", xlabelvisible=false, xticklabelsvisible=false, titlevisible=false, titlefont=:regular, xscale=log10) for i_syncmc=1:2]
                        legtitles = ["$(mixcrit_labels["lt"]) [synchron]", "$(mixcrit_labels["pim"]) [synimp]"]
                        for (i_syncmc,syncmc) in enumerate(["lt","pim"])
                            ax = axs[i_syncmc]
                            for (i_mcobj,mcobj) in enumerate(mixobjs[syncmc])
                                lines!(ax, fdivs[dst][rsp][syncmc][fdivname][:,i_boot,i_mcobj,i_scl], ytgts; color=i_mcobj, colorrange=(0,Nmcs[syncmc]), colormap=:RdYlBu_4, label=@sprintf("%.2f", mcobj))
                                # pick out the best mcobjs
                            end
                            idx_mcobj_best = mapslices(argmin, fdivs[dst][rsp][syncmc][fdivname][:,i_boot,:,i_scl]; dims=2)[:,1]
                            scatter!(ax, [fdivs[dst][rsp][syncmc][fdivname][i_ytgt,i_boot,idx_mcobj_best[i_ytgt],i_scl] for i_ytgt=1:Nytgt], ytgts; color=idx_mcobj_best, colorrange=(0,Nmcs[syncmc]), colormap=:RdYlBu_4)
                            # Max-entropy
                            #lines!(ax, fdivs[dst][rsp]["ent"][fdivname][:,i_boot,1,i_scl], ytgts; color=:red, linewidth=3, label=mixobj_labels["ent"][1], alpha=0.5)
                            # Max-EI 
                            lines!(ax, fdivs[dst][rsp]["ei"][fdivname][:,i_boot,1,i_scl], ytgts; color=:cyan, linewidth=4, linestyle=(:dash,:dense), label=mixobj_labels["ei"][1], alpha=1.0)

                            # Short simulation
                            band!(ax, Point2f.(fdivs_ancgen_valid_lo,ytgts), Point2f.(fdivs_ancgen_valid_hi,ytgts); color=:darkorange4, alpha=0.25)
                            lines!(ax, fdivs_ancgen_valid_pt, ytgts; color=:darkorange4, linewidth=4, label="Short DNS\n(90% CI)")
                            lout[i_syncmc,2] = Legend(fig, ax, legtitles[i_syncmc]; titlefont=:regular, titlesize=12, labelsize=8, nbanks=2, rowgap=1, framevisible=false)

                        end
                        linkxaxes!(axs...)
                        axs[1].titlevisible = true
                        axs[end].xlabelvisible = axs[end].xticklabelsvisible = true
                        for i = 1:length(axs)-1
                            rowgap!(lout, i, 10)
                        end
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
        mc = "ei"
        # synchron and synthrex and synimp
        for i_scl = scales2plot
            (pim_of_ast,pth_of_ast,fdiv_of_ast,mc_of_ast) = (zeros(Float64, (Nleadtime,Nytgt)) for _=1:4)
            (ast_of_pth,fdiv_of_pth,mc_of_pth) = (zeros(Float64, (Nmcs["pth"],Nytgt)) for _=1:3)
            (ast_of_pim,fdiv_of_pim,mc_of_pim) = (zeros(Float64, (Nmcs["pim"],Nytgt)) for _=1:3)
            for (i_ytgt,ytgt) in enumerate(ytgts)
                JLD2.jldopen(joinpath(exptdirs_COAST[i_ytgt],"results","ccdfs_regressed_accpa$(Int(adjust_ccdf_per_ancestor)).jld2"),"r") do f
                    Nancy = size(f["mixcrits"][dst][rsp]["pth"],2)
                    # AST as independent variable
                    pth_of_ast[:,i_ytgt] .= SB.mean(f["mixcrits"][dst][rsp]["pth"][1:Nleadtime,1:Nancy,i_scl]; dims=2)[:,1]
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
                    # pthas independent variable 
                    ilts = f["iltmixs"][dst][rsp]["pth"][1:Nmcs["pth"],1:Nancy,i_scl]
                    ast_of_pth[:,i_ytgt] .= sdm.tu.*SB.mean(leadtimes[ilts]; dims=2)[:,1]
                    fdiv_of_pth[:,i_ytgt] .= (f["fdivs"][dst][rsp]["pth"][fdivname][i_boot,1:Nmcs["pth"],i_scl])
                    mc_of_pth[:,i_ytgt] .= [
                                            SB.mean([f["mixcrits"][dst][rsp][mc][ilts[i_pth,i_anc],i_anc,i_scl] for i_anc=1:Nancy]) 
                                            for i_pth=1:Nmcs["pth"]
                                           ]
                end
            end
            # --------------- Show fdiv as a function of (AST,pth,pim) -----------
            colormap = :deep
            normalize_by_latitude = false
            logscale_flag = false
            if logscale_flag
                for arr = (
                           fdiv_of_ast,fdiv_of_pth,fdiv_of_pim,
                           mc_of_ast,mc_of_pth,mc_of_pim
                          )
                    arr .= log.(arr)
                end
                if normalize_by_latitude
                    errlabel = "log($(fdivlabel)/max $(fdivlabel)|y)"
                    mclabel = "log($(mixcrit_labels[mc])/max $(mixcrit_labels[mc])|y)"
                else
                    errlabel = "log($(fdivlabel))"
                    mclabel = "log($(mixcrit_labels[mc]))"
                end
            else
                errlabel = fdivlabel
                mclabel = mixcrit_labels[mc]
            end
            colorscale = identity
            fdivrange = [minimum(minimum.([fdiv_of_ast,fdiv_of_pth,fdiv_of_pim])),maximum(maximum.([fdiv_of_ast,fdiv_of_pth,fdiv_of_pim]))]
            mcrange = (minimum(minimum.([mc_of_ast,mc_of_pth,mc_of_pim])),maximum(maximum.([mc_of_ast,mc_of_pth,mc_of_pim])))
            if normalize_by_latitude
                for arr = (
                           fdiv_of_ast,fdiv_of_pth,fdiv_of_pim,
                           mc_of_ast,mc_of_pth,mc_of_pim
                          )
                    arr .= (arr .- minimum(arr; dims=1)) ./ (maximum(arr; dims=1) .- minimum(arr; dims=1))
                end
                colorrange_fdiv = (0,1)
                colorrange_mc = (0,1)
            else
                colorrange_fdiv = fdivrange
                colorrange_mc = mcrange
            end
            fig = Figure(size=(400,400))
            lout = fig[1,1] = GridLayout()
            axargs = Dict(:titlefont=>:regular, :xlabelsize=>8, :xticklabelsize=>6, :ylabelsize=>8, :yticklabelsize=>6)
            cbarargs = Dict(:labelfont=>:regular, :labelsize=>8, :ticklabelsize=>6, :valign=>:bottom, :size=>6)
            toplabel = "$(label_target(target_r, sdm)), scale $(distn_scales[dst][i_scl]), threshold exc. prob. $(powerofhalfstring(i_thresh_cquantile))=$(threshcquantstr)"
            lab = Label(lout[1,1:3], toplabel, padding=(0.0,0.0,0.0,0.0), valign=:center, halign=:center, fontsize=8, font=:regular)
            ax1 = Axis(lout[3,1]; xlabel="−AST", ylabel="(Target 𝑦)/𝐿", title="", axargs...)
            axargs[:ylabelvisible] = axargs[:yticklabelsvisible] = false
            ax2 = Axis(lout[3,2]; xlabel="𝑞(μ)", ylabel="(Target 𝑦)/𝐿", title="", axargs...)
            ax3 = Axis(lout[3,3]; xlabel="𝑞(𝑅*)", ylabel="(Target 𝑦)/𝐿", title="", axargs...)
            threshcquantstr = @sprintf("%.2E",thresh_cquantile)
            leadtime_bounds = tuple((-sdm.tu .* [1.5*leadtimes[end]-0.5*leadtimes[end-1], 1.5*leadtimes[1]-0.5*leadtimes[2]])...)
            # First heatmap: AST as independent variable
            hm1 = heatmap!(ax1, leadtime_bounds, (1.5*ytgts[1]-0.5*ytgts[2], 1.5*ytgts[Nytgt]-0.5*ytgts[Nytgt-1]), reverse(fdiv_of_ast; dims=1); colormap=colormap, colorscale=colorscale, colorrange=colorrange_fdiv)
            #co1pim = contour!(ax1, -leadtimes.*sdm.tu, ytgts, reverse(pim_of_ast; dims=1); color=:black, linestyle=(:dot,:dense), labels=false)
            cbar1 = Colorbar(lout[2,1], hm1; vertical=false, label="$(errlabel) [synchron]", cbarargs...)
            # Second heatmap: PTH as independent variable
            hm2 = heatmap!(ax2, (0,1), (1.5*ytgts[1]-0.5*ytgts[2], 1.5*ytgts[Nytgt]-0.5*ytgts[Nytgt-1]), reverse(fdiv_of_pth; dims=1); colormap=colormap, colorscale=colorscale, colorrange=colorrange_fdiv) 
            cbar2 = Colorbar(lout[2,2], hm2; vertical=false, label="$(errlabel) [synthrex]", cbarargs...)
            # Third heatmap: PIM as independent variable
            hm3 = heatmap!(ax3, (0,1), (1.5*ytgts[1]-0.5*ytgts[2], 1.5*ytgts[Nytgt]-0.5*ytgts[Nytgt-1]), reverse(fdiv_of_pim; dims=1); colormap=colormap, colorscale=colorscale, colorrange=colorrange_fdiv) 
            cbar3 = Colorbar(lout[2,3], hm3; vertical=false, label="$(errlabel) [synimp]", cbarargs...)
            rowgap!(lout, 1, 0)
            rowgap!(lout, 2, 5)
            colgap!(lout, 1, 0)
            colgap!(lout, 2, 0)

            rowsize!(lout, 1, Relative(1/9))
            #rowsize!(lout, 2, Relative(2/9))
            rowsize!(lout, 3, Relative(2/3))

            save(joinpath(resultdir,"phdgm_ast_pth_pim_$(dst)_$(rsp)_$(i_scl)_$(fdivname)_accpa$(Int(adjust_ccdf_per_ancestor)).png"), fig)

            # --------------- Show EI as a function of (AST,pth,pim) -----------
            fig = Figure(size=(400,400))
            lout = fig[1,1] = GridLayout()
            axargs = Dict(:titlefont=>:regular, :xlabelsize=>8, :xticklabelsize=>6, :ylabelsize=>8, :yticklabelsize=>6)
            cbarargs = Dict(:labelfont=>:regular, :labelsize=>8, :ticklabelsize=>6, :valign=>:bottom)
            toplabel = "$(label_target(target_r, sdm)), scale $(distn_scales[dst][i_scl]), threshold exc. prob. $(powerofhalfstring(i_thresh_cquantile))=$(threshcquantstr)"
            Label(lout[1,1:3], toplabel, padding=(0.0,0.0,0.0,0.0), valign=:center, halign=:center, fontsize=8, font=:regular)
            ax1 = Axis(lout[3,1]; xlabel="−AST", ylabel="(Target 𝑦)/𝐿", title="", axargs...)
            axargs[:ylabelvisible] = axargs[:yticklabelsvisible] = false
            ax2 = Axis(lout[3,2]; xlabel="𝑞(μ)", ylabel="(Target 𝑦)/𝐿", title="", axargs...)
            ax3 = Axis(lout[3,3]; xlabel="𝑞(𝑅*)", ylabel="(Target 𝑦)/𝐿", title="", axargs...)
            threshcquantstr = @sprintf("%.2E",thresh_cquantile)
            leadtime_bounds = tuple((-sdm.tu .* [1.5*leadtimes[end]-0.5*leadtimes[end-1], 1.5*leadtimes[1]-0.5*leadtimes[2]])...)
            # First heatmap: AST as independent variable
            hm1 = heatmap!(ax1, leadtime_bounds, (1.5*ytgts[1]-0.5*ytgts[2], 1.5*ytgts[Nytgt]-0.5*ytgts[Nytgt-1]), reverse(mc_of_ast; dims=1); colormap=Reverse(colormap), colorscale=colorscale, colorrange=colorrange_mc)
            cbar1 = Colorbar(lout[2,1], hm1; vertical=false, label="$(mclabel) [synchron]", cbarargs...)
            # Second heatmap: PTH as independent variable
            hm2 = heatmap!(ax2, (0,1), (1.5*ytgts[1]-0.5*ytgts[2], 1.5*ytgts[Nytgt]-0.5*ytgts[Nytgt-1]), reverse(mc_of_pth; dims=1); colormap=Reverse(colormap), colorscale=colorscale, colorrange=colorrange_mc) 
            cbar2 = Colorbar(lout[2,2], hm2; vertical=false, label="$(mclabel) [synthrex]", cbarargs...)
            # Third heatmap: PIM as independent variable
            hm3 = heatmap!(ax3, (0,1), (1.5*ytgts[1]-0.5*ytgts[2], 1.5*ytgts[Nytgt]-0.5*ytgts[Nytgt-1]), reverse(mc_of_pim; dims=1); colormap=Reverse(colormap), colorscale=colorscale, colorrange=colorrange_mc) 
            cbar3 = Colorbar(lout[2,3], hm3; vertical=false, label="$(mclabel) [synimp]", cbarargs...)
            #rowsize!(lout, 0, Relative(1/8))
            rowgap!(lout, 1, 0)
            rowgap!(lout, 2, 5)
            colgap!(lout, 1, 0)
            colgap!(lout, 2, 0)

            rowsize!(lout, 1, Relative(1/9))
            # rowsize!(lout, 2, Relative(2/9)) # TODO understand why uncommenting this line messes up all the proportions
            rowsize!(lout, 3, Relative(2/3))
            save(joinpath(resultdir,"phdgm_ast_pth_pim_$(dst)_$(rsp)_$(i_scl)_$(mc)_accpa$(Int(adjust_ccdf_per_ancestor)).png"), fig)
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
                                                 xlabel="Peak-adjusted log₂(exceedance probability)", ylabel="(Target 𝑦)/𝐿", title="$(label_target(target_r,sdm)), $(scalestr), $(title_suffixes[i])", 
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

