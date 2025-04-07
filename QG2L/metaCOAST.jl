
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
                             "compile_fdivs" =>                  0,
                             "plot_fdivs" =>                     1,
                             "plot_ccdfs_latdep" =>              0,
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
    rxystr = @sprintf("(%d/%d)𝐿",round(Int,target_r*sdm.Ny),sdm.Ny)
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
    scales2plot = collect(1:1:15) #[1,4,8,12]
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
        fig = Figure(size=(400,400))
        lout = fig[1,1] = GridLayout()
        Rccdf_rough_intercept = 0
        Rccdf_rough_slope = 1.0
        Rccdf_rough = Rccdf_rough_intercept .+ Rccdf_rough_slope .* ytgts #0.5 .+ 0.5 .* ytgts
        ylims = (1.5*ytgts[1]-0.5*ytgts[2], 1.5*ytgts[Nytgt]-0.5*ytgts[Nytgt-1])
        topo_zonal_mean = vec(SB.mean(cop.topography[:,:,2], dims=1))
        ytickvalues = ytgts[3:6:Nytgt]
        yticklabels = [@sprintf("%d/%d", round(Int, sdm.Ny*ytgt), sdm.Ny) for ytgt=ytickvalues]
        axargs = Dict(:titlefont=>:regular, :xgridvisible=>false, :ygridvisible=>false, :xticklabelrotation=>pi/2, :xlabelsize=>12, :ylabelsize=>12, :xticklabelsize=>10, :yticklabelsize=>10, :titlesize=>15, :ylabel=>"𝑦₀/𝐿", :yticks=>(ytickvalues, yticklabels))
        topolims = (minimum(topo_zonal_mean) == maximum(topo_zonal_mean) ? [-1,1] : extrema(topo_zonal_mean))
        axtopo = Axis(lout[1,1]; xlabel="ℎ(𝑦₀)", title="Topo.", titlefont=:regular, titlevisible=false, limits=(topolims..., ylims...), axargs...)
        axmean = Axis(lout[1,2]; xlabel="⟨𝑅⟩ − 𝑦₀/𝐿",ylabel="𝑦₀/𝐿", ylabelvisible=false, yticklabelsvisible=false, title="Mean", titlevisible=false, limits=(extrema(vcat(Rmean_ancgen.-Rccdf_rough,Rmean_valid.-Rccdf_rough).*1.25)..., ylims...), axargs...)
        axquants = Axis(lout[1,3]; xlabel="μ[(½)ᵏ] − 𝑦₀/𝐿\n𝑘∈{1,...,15}", ylabelvisible=false, yticklabelsvisible=false, title="(½)ᵏ-complementary\nquantiles, 𝑘∈{1,...,15}", titlevisible=false, limits=(extrema(Rccdfs_valid.-Rccdf_rough')...,ylims...), axargs...)
        #axquantyders = Axis(lout[1,4], xlabel="Δ(𝑅-⟨𝑅⟩)/Δ⟨𝑅⟩", ylabel="𝑦₀/𝐿", xgridvisible=false, ygridvisible=false, ylabelvisible=false, yticklabelsvisible=false, xticklabelrotation=pi/2, title="Relative 𝑦-gradients", titlefont=:regular, limits=(0,2,ylims...))
        toplabel = "Intensities 𝑅, $(label_target(target_r,sdm))"
        Label(lout[1,2:3,Top()], toplabel, padding=(5.0,5.0,5.0,5.0), valign=:center, halign=:left, fontsize=15, font=:regular)
        lines!(axtopo, topo_zonal_mean, sdm.ygrid./sdm.Ly; color=:black)
        for (ax,color) = ((axmean,:grey60),(axquants,:grey60))
            lines!(ax, Rmean_valid .- Rccdf_rough, ytgts; color=color, linestyle=(:dash,:dense), label="Long DNS")
            lines!(ax, Rmean_ancgen .- Rccdf_rough, ytgts; color=color, linestyle=:solid, label="Short DNS")
        end
        ytgts_mid = (ytgts[2:end] .+ ytgts[1:end-1])./2
        # For quantiles, subtract off either the mean or the rough trend 
        for i_cl = reverse(1:length(ccdf_levels))
            cl = ccdf_levels[i_cl]
            lines!(axquants, Rccdfs_valid[i_cl,:].-Rccdf_rough, ytgts; linewidth=2, linestyle=(:dash,:dense), color=i_cl, colargs..., label="Long DNS")
            lines!(axquants, Rccdfs_ancgen[i_cl,:].-Rccdf_rough, ytgts; linewidth=1, linestyle=:solid, color=i_cl, colargs..., label="Short DNS")
            #lines!(axquantyders, diff(Rccdfs_valid[i_cl,:])./diff(Rmean_valid), ytgts_mid; linewidth=2, linestyle=(:dash,:dense), color=i_cl, colargs..., label="Long DNS")
            #lines!(axquantyders, diff(Rccdfs_ancgen[i_cl,:] .- Rmean_valid)./diff(Rmean_valid), ytgts_mid; linewidth=1, linestyle=:solid, color=i_cl, colargs..., label="Short DNS")
        end
        lines!(axquants, 1 .- Rccdf_rough, ytgts; color=:grey60, linewidth=2, linestyle=(:dash,:dense))
        lines!(axquants, 0 .- Rccdf_rough, ytgts; color=:grey60, linewidth=2, linestyle=(:dash,:dense))
        #vlines!(axquantyders, -1; color=:grey79, linewidth=3)
        #lout[2,:] = Legend(fig, axquants, "Exceedance probabilities (½)ᵏ, k ∈ {1,...,15}"; framevisible=true, titlefont=:regular, titlehalign=:left, merge=true, linecolor=:black, nbanks=2, labelsize=10, titlesize=10)
        axislegend(axmean, ; merge=true, linecolor=:black, framevisible=false, titlefont=:regular, labelsize=8)
        colsize!(lout, 1, Relative(1/12))
        colsize!(lout, 2, Relative(5/12))
        #colsize!(lout, 3, Relative(8/12))
        colgap!(lout, 1, 10)
        colgap!(lout, 2, 10)
        #colgap!(lout, 3, 10)
        save(joinpath(resultdir,"ccdfs_latdep_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).png"),fig)

        # ---------- Plot GPD parameters along with moments ---------
        mssk = JLD2.jldopen(joinpath(resultdir_dns,"moments_mssk_$(cfgs[1].target_field[1:end-1]).jld2"),"r") do f
            return f["mssk_xall"][1,:,parse(Int,cfgs[1].target_field[end]),:]
        end
        fig = Figure(size=(600,400))
        lout = fig[1,1] = GridLayout()
        ytickvalues = ytgts[3:6:Nytgt]
        yticklabels = [@sprintf("%d/%d", round(Int, sdm.Ny*ytgt), sdm.Ny) for ytgt=ytickvalues]
        axargs = Dict(:titlefont=>:regular, :xgridvisible=>false, :ygridvisible=>false, :xticklabelrotation=>pi/2, :xlabelsize=>12, :ylabelsize=>12, :xticklabelsize=>10, :yticklabelsize=>10, :titlesize=>15, :ylabel=>"𝑦₀/𝐿", :yticks=>(ytickvalues, yticklabels))
        topolims = (minimum(topo_zonal_mean) == maximum(topo_zonal_mean) ? [-1,1] : extrema(topo_zonal_mean))
        axtopo = Axis(lout[1,1],xlabel="ℎ(𝑦₀)"; axargs..., title="Topo.", limits=(topolims..., ylims...))
        axargs[:ylabelvisible] = axargs[:yticklabelsvisible] = false
        axstd = Axis(lout[1,2]; xlabel="Std. Dev.", axargs..., limits=(0, maximum(mssk[:,2])*1.01, ylims...))
        axscale = Axis(lout[1,3]; xlabel="GPD scale σ", axargs..., limits=(0,maximum(gpd_scale_valid)*1.01,ylims...))
        axshape = Axis(lout[1,4]; xlabel="GPD shape ξ", axargs..., limits=(-0.5,0.05,ylims...))
        threshcquantstr = @sprintf("%.2E",thresh_cquantile)
        toplabel = @sprintf("Threshold μ[%s], exceedance probability %s = %.2E", powerofhalfstring(i_thresh_cquantile), powerofhalfstring(i_thresh_cquantile), thresh_cquantile)
        Label(lout[1,2:4,Top()], toplabel, padding=(5.0,5.0,5.0,5.0), valign=:bottom, halign=:right, fontsize=15, font=:regular)
        # Topography
        lines!(axtopo, topo_zonal_mean, sdm.ygrid./sdm.Ly; color=:black)
        # Std. Dev.
        lines!(axstd, mssk[:,2], sdm.ygrid./sdm.Ly; color=:black, linestyle=(:dash,:dense), label="(1/$(2*sdm.Ny))𝐿")
        scatterlines!(axstd, std_valid, ytgts; color=:black, label="($(round(Int, target_r*sdm.Ny))/$(sdm.Ny))𝐿")
        axislegend(axstd, "Box radius ℓ"; position=:lc, labelsize=10, titlesize=12, titlefont=:regular, framevisible=false)
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
        rsps = ("z","2","e",)
        i_boot = 1

        @show Nleadtime

        mcs2compile = ["contcorr","globcorr","ei","pim","pth","ent","lt","r2lin","r2quad"]

        fdivs = Dict{String,Dict}()
        for dst = dsts
            fdivs[dst] = Dict{String,Dict}()
            for rsp = rsps
                fdivs[dst][rsp] = Dict{String,Dict}()
                for mc = mcs2compile
                    fdivs[dst][rsp][mc] = Dict{String,Dict}()
                    for est = ["mix","pool"]
                        fdivs[dst][rsp][mc][est] = Dict{String,Array{Float64}}()
                        for fdivname = fdivnames
                            fdivs[dst][rsp][mc][est][fdivname] = zeros(Float64, (Nytgt,Nboot+1,Nmcs[mc],Nscales[dst]))
                        end
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
            JLD2.jldopen(joinpath(resultdir_y,"ccdfs_combined.jld2"),"r") do f
                for dst = dsts
                    for rsp = rsps
                        for mc = mcs2compile
                            for est = ["mix","pool"]
                                for fdivname = fdivnames
                                    fdivs[dst][rsp][mc][est][fdivname][i_ytgt,:,:,:] .= f["fdivs"][dst][rsp][mc][est][fdivname][:,:,:]
                                end
                            end
                        end
                    end
                end
                for fdivname = fdivnames
                    fdivs_ancgen_valid[fdivname][i_ytgt,:] .= f["fdivs_ancgen_valid"][fdivname]
                end
            end
        end

        JLD2.jldopen(joinpath(resultdir,"fdivs.jld2"),"w") do f
            f["fdivs"] = fdivs
            f["fdivs_ancgen_valid"] = fdivs_ancgen_valid
        end
    end

    #

    if todo["plot_fdivs"]

        fdivs,fdivs_ancgen_valid = JLD2.jldopen(joinpath(resultdir,"fdivs.jld2"),"r") do f
            return f["fdivs"],f["fdivs_ancgen_valid"]
        end

        dsts = ("b",)
        rsps = ("z","2","e",)
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
                        scalestr = @sprintf("𝑠 = %.2f", distn_scales[dst][i_scl])
                        syncmcs = ["lt","contcorr","globcorr"]
                        fig = Figure(size=(450,400))
                        lout = fig[1,1] = GridLayout()
                        ax = Axis(lout[1,2], xlabel=fdivlabels[fdivname], ylabel="𝑦₀/𝐿", title="$(label_target(target_r,sdm)), $(scalestr), μ[$(powerofhalfstring(i_thresh_cquantile))]", titlevisible=true, titlefont=:regular, xscale=log10, xgridvisible=false, ygridvisible=false)
                        # Short simulation
                        band!(ax, Point2f.(fdivs_ancgen_valid_lo,ytgts), Point2f.(fdivs_ancgen_valid_hi,ytgts); color=:gray, alpha=0.25)
                        lines!(ax, fdivs_ancgen_valid_pt, ytgts; color=:black, linewidth=4, label="Short DNS\n(90% CI)")
                        # All desired mixing criteria
                        for (i_syncmc,syncmc) in enumerate(syncmcs)
                            for (est,marker) = (("mix",:xcross),("pool",'O'))
                                idx_mcobj_best = mapslices(argmin, fdivs[dst][rsp][syncmc][est][fdivname][:,i_boot,:,i_scl]; dims=2)[:,1]
                                scatter!(ax, [fdivs[dst][rsp][syncmc][est][fdivname][i_ytgt,i_boot,idx_mcobj_best[i_ytgt],i_scl] for i_ytgt=1:Nytgt], ytgts; color=mixcrit_colors[syncmc], marker=marker, label="Optimal $(mixcrit_labels[syncmc])")
                                # Max-entropy
                                #lines!(ax, fdivs[dst][rsp]["ent"][fdivname][:,i_boot,1,i_scl], ytgts; color=:red, linewidth=3, label=mixobj_labels["ent"][1], alpha=0.5)
                                # Max-EI 
                            end
                        end
                        xlims!(ax, minimum(fdivs_ancgen_valid_lo)/10^1.5, maximum(fdivs_ancgen_valid_hi)*1.1)
                        for (est,marker) = (("mix",:xcross),("pool",'O'))
                            scatter!(ax, fdivs[dst][rsp]["ei"][est][fdivname][:,i_boot,1,i_scl], ytgts; color=mixcrit_colors["ei"], label=mixobj_labels["ei"][1], alpha=1.0, marker=marker)
                            scatter!(ax, fdivs[dst][rsp]["ent"][est][fdivname][:,i_boot,1,i_scl], ytgts; color=mixcrit_colors["ent"], label=mixobj_labels["ent"][1], alpha=1.0, marker=marker)
                            # TODO put error bars 
                        end
                        lout[1,1] = Legend(fig, ax; labelsize=10, framevisible=false, merge=true)
                        colsize!(lout, 2, Relative(3/5))
                        colgap!(lout, 1, 0)

                        save(joinpath(resultdir,"fdivofy_$(fdivname)_$(dst)_$(rsp)_$(i_scl).png"), fig)
                    end
                end
            end
        end
    end





    if todo["plot_mixcrits_ydep"]
        dst = "b"
        rsps = ("z","2","e")
        i_boot = 1
        mc = "ent" # this is the privileged mixing criterion on which to optimize 
        i_mcval = 1 # because we're optimizing it
        est = "mix"
        ϵ = 1-(3/8)^2
        for rsp = rsps
            for (fdivname,fdivlabel) = (("chi2","χ²"),("kl","KL"),("qrmse","𝐿²"),)
                (
                 fdiv_of_ast,
                 contcorr_of_ast,globcorr_of_ast,mc_of_ast,
                 iltfrac_mc_of_ast,
                ) = (
                     zeros(Float64, (Nleadtime,Nytgt,Nscales[dst])) for _=1:5
                    )
                (
                 fdiv_of_contcorr,
                 ast_of_contcorr,mc_of_contcorr,
                 iccfrac_mc_of_contcorr,
                ) = (
                     zeros(Float64, (Nmcs["contcorr"],Nytgt,Nscales[dst])) for _=1:4
                    )
                ilt_best_of_ast = zeros(Int64, (Nytgt,Nscales[dst]))
                icc_best_of_contcorr = zeros(Int64, (Nytgt,Nscales[dst]))
                for i_scl = scales2plot
                    for (i_ytgt,ytgt) in enumerate(ytgts)
                        JLD2.jldopen(joinpath(exptdirs_COAST[i_ytgt],"results","ccdfs_combined.jld2"),"r") do f
                            Nancy = size(f["mixcrits"][dst][rsp]["lt"],2)
                            # AST as independent variable
                            contcorr_of_ast[:,i_ytgt,i_scl] .= SB.mean(f["mixcrits"][dst][rsp]["contcorr"][1:Nleadtime,1:Nancy,i_scl]; dims=2)[:,1]
                            globcorr_of_ast[:,i_ytgt,i_scl] .= SB.mean(f["mixcrits"][dst][rsp]["globcorr"][1:Nleadtime,1:Nancy,i_scl]; dims=2)[:,1]
                            mc_of_ast[:,i_ytgt,i_scl] .= SB.mean(f["mixcrits"][dst][rsp][mc][1:Nleadtime,1:Nancy,i_scl]; dims=2)[:,1]
                            fdiv_of_ast[:,i_ytgt,i_scl] .= (f["fdivs"][dst][rsp]["lt"][est][fdivname][i_boot,1:Nleadtime,i_scl])
                            ilt_best_of_ast[i_ytgt,i_scl] = argmin(fdiv_of_ast[:,i_ytgt,i_scl])

                            iltfrac_mc_of_ast[:,i_ytgt,i_scl] .= f["iltcounts"][dst][rsp][mc][i_mcval,:,i_scl]./Nancy
                                                   
                            # contcorr as independent variable
                            ilts_contcorr = f["iltmixs"][dst][rsp]["contcorr"][1:Nmcs["contcorr"],1:Nancy,i_scl] # the lead time index at which each ancestor achieves the chosen contour correlation value 
                            ilts_globcorr = f["iltmixs"][dst][rsp]["globcorr"][1:Nmcs["globcorr"],1:Nancy,i_scl] # the lead time index at which each ancestor achieves the chosen globour correlation value 
                            fdiv_of_contcorr[:,i_ytgt,i_scl] .= (f["fdivs"][dst][rsp]["contcorr"][est][fdivname][i_boot,1:Nmcs["contcorr"],i_scl])
                            ast_of_contcorr[:,i_ytgt,i_scl] .= sdm.tu.*SB.mean(leadtimes[ilts_contcorr]; dims=2)[:,1]
                            for i_contcorr = 1:Nmcs["contcorr"]
                                mc_of_contcorr[i_contcorr,i_ytgt,i_scl] = SB.mean([f["mixcrits"][dst][rsp][mc][ilts_contcorr[i_contcorr,i_anc],i_anc,i_scl] for i_anc=1:Nancy])
                            end
                            # TODO figure out the frequency distribution of contour levels coinciding with maximum-entropy 
                            icc_best_of_contcorr[i_ytgt,i_scl] = argmin(fdiv_of_contcorr[:,i_ytgt,i_scl])
                            for i_anc = 1:Nancy
                                i_leadtime_mcmax = f["iltmixs"][dst][rsp][mc][i_mcval,i_anc,i_scl]
                                contcorr_mcmax = f["mixcrits"][dst][rsp]["contcorr"][i_leadtime_mcmax,i_anc,i_scl]
                                i_contcorr_mcmax = (f["mixobjs"]["contcorr"][end] >= contcorr_mcmax ? findfirst(f["mixobjs"]["contcorr"] .>= contcorr_mcmax) : Nmcs["contcorr"])
                                iccfrac_mc_of_contcorr[i_contcorr_mcmax,i_ytgt,i_scl] += 1/Nancy
                            end
                        end
                    end
                end
                colormap = Reverse(:coolwarm) #:bwr
                normalize_by_latitude = true
                logscale_flag = false #(mc != "ent")
                if logscale_flag
                    for arr = (
                               fdiv_of_ast,fdiv_of_contcorr,
                               mc_of_ast,mc_of_contcorr,
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
                fdivrange = [minimum(minimum.([fdiv_of_ast,fdiv_of_contcorr,])),maximum(maximum.([fdiv_of_ast,fdiv_of_contcorr,]))]
                mcrange = (minimum(minimum.([mc_of_ast,mc_of_contcorr,])),maximum(maximum.([mc_of_ast,mc_of_contcorr,])))
                ccrange = (minimum(contcorr_of_ast), maximum(contcorr_of_ast))
                gcrange = (minimum(globcorr_of_ast), maximum(globcorr_of_ast))
                if normalize_by_latitude
                    #for arr = (
                    #           fdiv_of_ast,fdiv_of_contcorr,
                    #           mc_of_ast,mc_of_contcorr,
                    #          )
                    #    arr .= (arr .- minimum(arr; dims=1)) ./ (maximum(arr; dims=1) .- minimum(arr; dims=1))
                    #end
                    colorrange_fdiv = (0,1)
                    colorrange_mc = (0,1)
                else
                    colorrange_fdiv = fdivrange
                    colorrange_mc = mcrange
                end
                for i_scl = scales2plot
                    ytickvalues = ytgts[3:6:Nytgt]
                    yticklabels = [@sprintf("%d/%d", round(Int, sdm.Ny*ytgt), sdm.Ny) for ytgt=ytickvalues]
                    #heatmap_theme = Theme(titlefont=:regular, xlabelsize=12, ylabelsize=12, xticklabelsize=9, yticklabelsize=9, titlesize=15, ylabel="𝑦₀/𝐿", yticks=(ytickvalues, yticklabels))
                    axargs = Dict(:title=>"", :titlefont=>:regular, :xlabelsize=>10, :ylabelsize=>10, :xticklabelsize=>8, :yticklabelsize=>8, :titlesize=>12, :ylabel=>"𝑦₀/𝐿", :yticks=>(ytickvalues, yticklabels))
                    cbarargs = Dict(:labelfont=>:regular, :labelsize=>10, :ticklabelsize=>8, :size=>6, :vertical=>false, )
                    # --------------- contcorr as function of AST -------------------
                    fig = Figure(size=(200,350))
                    lout = fig[1,1] = GridLayout()
                    ax = Axis(lout[2,1]; xlabel="-AST", axargs...)
                    hm = heatmap!(ax, -sdm.tu.*reverse(leadtimes), ytgts, reverse(transcorr.(contcorr_of_ast[1:Nleadtime,1:Nytgt,i_scl]); dims=1); colormap=Reverse(:deep), colorrange=transcorr.(ccrange))
                    co = contour!(ax, -sdm.tu.*leadtimes, ytgts, transcorr.(contcorr_of_ast[1:Nleadtime,1:Nytgt,i_scl]); levels=[transcorr(1-(3/8)^2)], color=:black, linestyle=(:dash,:dense), linewidth=2)
                    cbar = Colorbar(lout[1,1], hm; label=@sprintf("σ⁻¹(ρ[𝑐(𝑦₀,⋅)]); %s, 𝑠=%.2f", label_target(target_r, sdm), distn_scales[dst][i_scl]), cbarargs...)
                    rowgap!(lout, 1, 5)
                    save(joinpath(resultdir,"heatmap_contcorr_ofASTandY_$(dst)_$(rsp)_$(i_scl).png"), fig)

                    #
                    # --------------- F-divergence as a function of AST  -----------
                    fig = Figure(size=(400,350))
                    lout = fig[1,1] = GridLayout()
                    axfdiv,axmaxfdiv,axmc,axmaxmc = [Axis(lout[2,i]; axargs...) for i=1:4]
                    threshcquantstr = @sprintf("%.2E",thresh_cquantile)
                    suptitle = Label(lout[1,:], @sprintf("%s, 𝑠=%.2f", label_target(target_r, sdm), distn_scales[dst][i_scl]); fontsize=12)
                    axfdiv.title = fdivlabel
                    axmaxfdiv.title = "bounds"
                    axmc.title = mclabel
                    axmaxmc.title = "bounds"
                    # First heatmap: overlay optimal-entropy leadtime distribution on fdiv
                    maxfdiv = maximum(fdiv_of_ast[:,:,i_scl]; dims=1)
                    minfdiv = minimum(fdiv_of_ast[:,:,i_scl]; dims=1)
                    minmc = minimum(mc_of_ast[:,:,i_scl]; dims=1)
                    maxmc = maximum(mc_of_ast[:,:,i_scl]; dims=1)
                    scatterlines!(axmaxfdiv, vec(maxfdiv), ytgts; color=:black)
                    scatterlines!(axmaxfdiv, vec(minfdiv), ytgts; color=:black)
                    axmaxfdiv.xscale = log10
                    scatterlines!(axmaxmc, vec(maxmc), ytgts; color=:black)
                    scatterlines!(axmaxmc, vec(minmc), ytgts; color=:black)
                    hmfdiv = heatmap!(axfdiv, -sdm.tu.*reverse(leadtimes), ytgts, (reverse(fdiv_of_ast[:,:,i_scl]; dims=1) .- minfdiv)./(maxfdiv .- minfdiv); colormap=Reverse(:grays), colorscale=colorscale, colorrange=colorrange_fdiv)
                    scatter!(axfdiv, -sdm.tu.*leadtimes[ilt_best_of_ast[:,i_scl]], ytgts; color=:red, marker=:cross)
                    hmmc = heatmap!(axmc, -sdm.tu.*reverse(leadtimes), ytgts, (reverse(mc_of_ast[:,:,i_scl]; dims=1) .- minmc)./(maxmc .- minmc); colormap=:grays)
                    for (i_ytgt,ytgt) in enumerate(ytgts)
                        scatter!(axmc, -sdm.tu.*leadtimes, ytgt.*ones(Float64,Nleadtime); marker='O', markersize=60 .* iltfrac_mc_of_ast[:,i_ytgt,i_scl], color=:red)
                    end
                    rowgap!(lout, 1, 5)
                    rowsize!(lout, 1, Relative(1/9))

                    co_contcorr = contour!(axfdiv, -sdm.tu.*reverse(leadtimes), ytgts, reverse(contcorr_of_ast[:,:,i_scl]; dims=1); levels=[ϵ], color=:deepskyblue, linewidth=1.5, linestyle=(:dot,:dense))
                    co_globcorr = contour!(axfdiv, -sdm.tu.*reverse(leadtimes), ytgts, reverse(globcorr_of_ast[:,:,i_scl]; dims=1); levels=[ϵ], color=:deepskyblue, linewidth=1.5, linestyle=(:dash,:dense))
                    resize_to_layout!(fig)

                    colsize!(lout, 1, Relative(1/3))
                    colsize!(lout, 2, Relative(1/6))
                    colsize!(lout, 3, Relative(1/3))
                    for ax = (axmaxfdiv,axmc,axmaxmc)
                        ax.ylabelvisible = ax.yticklabelsvisible=false
                    end
                    for i_col = 1:3
                        colgap!(lout, i_col, 0)
                    end


                    save(joinpath(resultdir,"heatmap_$(fdivname)_ofASTandY_$(dst)_$(rsp)_$(i_scl)_nocoast.png"), fig)
                    for ax = (axfdiv,axmc)
                        scatter!(ax, -sdm.tu.*leadtimes[ilt_best_of_ast[:,i_scl]], ytgts; color=:green, marker=:cross)
                    end
                    for ax = (axmaxfdiv,axmaxmc)
                        ax.xticklabelrotation = pi/2
                    end
                    for ax = (axfdiv, axmc)
                        ax.xticklabel = "−AST"
                    end
                    save(joinpath(resultdir,"heatmap_$(fdivname)_ofASTandY_$(dst)_$(rsp)_$(i_scl).png"), fig)

                    # --------------- F-divergence as a function of contcorr  -----------
                    fig = Figure(size=(200,350))
                    lout = fig[1,1] = GridLayout()
                    threshcquantstr = @sprintf("%.2E",thresh_cquantile)
                    title = @sprintf("%s; %s, 𝑠=%.2f", errlabel, label_target(target_r, sdm), distn_scales[dst][i_scl])
                    ax1 = Axis(lout[2,1]; xlabel="σ⁻¹(ρ[𝑐(𝑦₀,⋅)])", ylabel="𝑦₀/𝐿", title="", axargs...)
                    axargs[:ylabelvisible] = axargs[:yticklabelsvisible] = false
                    # First heatmap: overlay optimal-entropy leadtime distribution on fdiv
                    hm1 = heatmap!(ax1, transcorr.(mixobjs["contcorr"]), ytgts, fdiv_of_contcorr[:,:,i_scl]; colormap=colormap, colorscale=colorscale, colorrange=colorrange_fdiv)
                    co1 = contour!(ax1, transcorr.(mixobjs["contcorr"]), ytgts, mc_of_contcorr[:,:,i_scl]; levels=range(mcrange...; length=7), color=:cyan, labels=false)
                    for (i_ytgt,ytgt) in enumerate(ytgts)
                        scatter!(ax1, transcorr.(mixobjs["contcorr"]), ytgt.*ones(Float64,Nmcs["contcorr"]); marker='O', markersize=60 .* iccfrac_mc_of_contcorr[:,i_ytgt,i_scl], color=:black)
                    end
                    cbar1 = Colorbar(lout[1,1], hm1; vertical=false, label=title, cbarargs...)
                    rowgap!(lout, 1, 5)
                    rowsize!(lout, 1, Relative(1/9))

                    resize_to_layout!(fig)
                    save(joinpath(resultdir,"heatmap_$(fdivname)_ofCCandY_$(dst)_$(rsp)_$(i_scl)_nocoast.png"), fig)
                    scatter!(ax1, transcorr.(mixobjs["contcorr"][icc_best_of_contcorr[:,i_scl]]), ytgts; color=:black, marker=:cross)
                    vlines!(ax1, transcorr(ϵ); color=:black, linestyle=(:dot,:dense))
                    save(joinpath(resultdir,"heatmap_$(fdivname)_ofCCandY_$(dst)_$(rsp)_$(i_scl).png"), fig)
                    # --------------- the chosen mixing criterion as a function of AST -----------
                    fig = Figure(size=(200,350), )
                    lout = fig[1,1] = GridLayout()
                    axargs = Dict(:titlefont=>:regular, :xlabelsize=>8, :xticklabelsize=>6, :ylabelsize=>8, :yticklabelsize=>6)
                    cbarargs = Dict(:labelfont=>:regular, :labelsize=>8, :ticklabelsize=>6, :valign=>:bottom)
                    ax1 = Axis(lout[2,1]; xlabel="−AST", ylabel="𝑦₀/𝐿", title="", axargs...)
                    axargs[:ylabelvisible] = axargs[:yticklabelsvisible] = false
                    threshcquantstr = @sprintf("%.2E",thresh_cquantile)
                    hm1 = heatmap!(ax1, -sdm.tu.*reverse(leadtimes), ytgts, reverse(mc_of_ast[:,:,i_scl]; dims=1); colormap=Reverse(colormap), colorscale=colorscale, colorrange=colorrange_mc)
                    cbar1 = Colorbar(lout[1,1], hm1; vertical=false, label="$(mclabel) (iso-$(mixcrit_labels["lt"]))", cbarargs...)
                    rowgap!(lout, 1, 5)
                    rowsize!(lout, 1, Relative(1/9))
                    resize_to_layout!(fig)
                    @assert (all(isfinite.(mc_of_ast)) && all(isfinite.(mc_of_contcorr)))
                    save(joinpath(resultdir,"heatmap_$(mc)ofASTandY_$(dst)_$(rsp).png"), fig)
                    #
                    # --------------- the chosen mixing criterion as a function of contcorr -----------
                    fig = Figure(size=(200,350), )
                    lout = fig[1,1] = GridLayout()
                    axargs = Dict(:titlefont=>:regular, :xlabelsize=>8, :xticklabelsize=>6, :ylabelsize=>8, :yticklabelsize=>6)
                    cbarargs = Dict(:labelfont=>:regular, :labelsize=>8, :ticklabelsize=>6, :valign=>:bottom)
                    ax1 = Axis(lout[2,1]; xlabel="σ⁻¹(ρ[𝑐(𝑦₀,⋅)])", axargs...)
                    hm1 = heatmap!(ax1, transcorr.(mixobjs["contcorr"]), ytgts, mc_of_contcorr[:,:,i_scl]; colormap=Reverse(colormap), colorscale=colorscale, colorrange=colorrange_mc)
                    cbar1 = Colorbar(lout[1,1], hm1; vertical=false, label="$(mclabel) (iso-$(mixcrit_labels["contcorr"]))", cbarargs...)
                    rowgap!(lout, 1, 5)
                    rowsize!(lout, 1, Relative(1/9))
                    # rowsize!(lout, 2, Relative(2/9)) # TODO understand why uncommenting this line messes up all the proportions
                    resize_to_layout!(fig)
                    @assert (all(isfinite.(mc_of_ast)) && all(isfinite.(mc_of_contcorr)))
                    save(joinpath(resultdir,"heatmap_$(mc)ofCCandY_$(dst)_$(rsp).png"), fig)
                end
            end
        end
    end
end

