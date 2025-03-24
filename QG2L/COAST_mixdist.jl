function evaluate_mixing_criteria(cfg, cop, pertop, coast, ens, resultdir, )
    # No need to go through thissong and dance of intermediate highly discretized CCDFs. Just estimate entropy and expected improvement empirically 
    (
     leadtimes,r2threshes,dsts,rsps,mixobjs,
     mixcrit_labels,mixobj_labels,mixcrit_colors,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,i_thresh_cquantile,adjust_ccdf_per_ancestor
    ) = expt_config_COAST_analysis(cfg,pertop)
    thresh_cquantile = ccdf_levels[i_thresh_cquantile]
    Nanc = length(coast.ancestors)
    Nleadtime = length(leadtimes)
    Nr2th = length(r2threshes)
    Nscales = Dict(dst=>length(distn_scales[dst]) for dst=dsts)
    Nlev = length(ccdf_levels)
    i_mode_sf = 1
    support_radius = pertop.sf_pert_amplitudes_max[i_mode_sf]
    (
     tgrid_ancgen,
     Roft_ancgen_seplon,
     Rccdf_ancgen_seplon,
     Rccdf_ancgen_agglon,
     ccdf_pot_ancgen_seplon,
     ccdf_pot_ancgen_agglon,
     tgrid_valid,
     Roft_valid_seplon,
     Rccdf_valid_seplon,
     Rccdf_valid_agglon,
     ccdf_pot_valid_seplon,
     ccdf_pot_valid_agglon,
    ) = (
         JLD2.jldopen(joinpath(resultdir,"objective_dns_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).jld2"), "r") do f
             return (
                     f["tgrid_ancgen"], # tgrid_ancgen
                     f["Roft_ancgen_seplon"], # Roft_ancgen_seplon
                     f["Rccdf_ancgen_seplon"], # Rccdf_ancgen_seplon
                     f["Rccdf_ancgen_agglon"], # Rccdf_ancgen_seplon
                     f["ccdf_pot_ancgen_seplon"],
                     f["ccdf_pot_ancgen_agglon"],
                     f["tgrid_valid"], # tgrid_valid
                     f["Roft_valid_seplon"], # Roft_valid_seplon
                     f["Rccdf_valid_seplon"], # Rccdf_valid_seplon
                     f["Rccdf_valid_agglon"], # Rccdf_valid_agglon
                     f["ccdf_pot_valid_seplon"],
                     f["ccdf_pot_valid_agglon"],
                    )
         end
        )
        levels = Rccdf_valid_agglon
    coefslin,coefsquad,r2lin,r2quad = JLD2.jldopen(joinpath(resultdir,"regression_coefs.jld2"), "r") do f
        return f["coefs_linear"],f["coefs_quadratic"],f["rsquared_linear"],f["rsquared_quadratic"]
    end
    globcorr,contcorr = JLD2.jldopen(joinpath(resultdir,"contour_dispersion.jld2"), "r") do f
        return f["globcorr"], f["contcorr"]
    end
    thresh = Rccdf_valid_agglon[i_thresh_cquantile] 
    # Pre-allocate some arrays for samples and weights 
    Npert = div(cfg.num_perts_max, Nleadtime)
    Rs,Ws = (zeros(Npert+1) for _=1:2)
    U = vcat(zeros(Float64, (1,2)), collect(transpose(coast.pert_seq_qmc[:,1:Npert])))
    mixcrits,ccdfs,pdfs,ilts,iltcounts = (Dict{String,Dict}() for _=1:5)
    mcdiff = zeros(Float64,Nleadtime-1)
    mc_locmax_flag = zeros(Bool, Nleadtime)
    for dst = ["b"]
        mixcrits[dst] = Dict{String,Dict}()
        ilts[dst] = Dict{String,Dict}()
        iltcounts[dst] = Dict{String,Dict}()
        ccdfs[dst],pdfs[dst] = (Dict{String,Array{Float64}}() for _=1:2)
        for rsp = ["e"]
            mixcrits[dst][rsp] = Dict{String,Array{Float64}}()
            ilts[dst][rsp] = Dict{String,Array{Int64}}()
            iltcounts[dst][rsp] = Dict{String,Array{Int64}}()
            ccdfs[dst][rsp] = zeros(Float64, (Nlev,Nleadtime,Nanc,Nscales[dst]))
            pdfs[dst][rsp] = zeros(Float64, (Nlev-1,Nleadtime,Nanc,Nscales[dst]))
            for mc = keys(mixobjs)
                mixcrits[dst][rsp][mc] = zeros(Float64, (Nleadtime,Nanc,Nscales[dst]))
                ilts[dst][rsp][mc] = zeros(Int64, (length(mixobjs[mc]),Nanc,Nscales[dst]))
                iltcounts[dst][rsp][mc] = zeros(Int64, (length(mixobjs[mc]),Nleadtime,Nscales[dst]))
            end
        end
    end
    for i_anc = 1:Nanc
        Rs[1] = coast.anc_Rmax[i_anc]
        for (i_leadtime,leadtime) in enumerate(leadtimes)
            idx_dsc = desc_by_leadtime(coast, i_anc, leadtime, sdm)
            @infiltrate
            Rs[2:Npert+1] .= coast.desc_Rmax[i_anc][idx_dsc]
            for dst = ["b"]
                for rsp = ["e"]
                    for i_scl = 1:Nscales[dst]
                        Ws .= QG2L.bump_density(U, distn_scales[dst][i_scl], support_radius)
                        mixcrits[dst][rsp]["lt"][i_leadtime,i_anc,i_scl] = leadtimes[i_leadtime]
                        mixcrits[dst][rsp]["pth"][i_leadtime,i_anc,i_scl] = QG2L.threshold_exceedance_probability_samples(Rs, Ws, thresh)
                        mixcrits[dst][rsp]["pim"][i_leadtime,i_anc,i_scl] = QG2L.threshold_exceedance_probability_samples(Rs, Ws, Rs[1])
                        #mixcrits[dst][rsp]["ent"][i_leadtime,i_anc,i_scl] = QG2L.entropy_fun_samples(Rs, Ws, thresh)
                        mixcrits[dst][rsp]["ei"][i_leadtime,i_anc,i_scl] = QG2L.expected_improvement_samples(Rs, Ws, Rs[1])
                        # For correlations, the averaging weight depends on the scale, so these two aren't totally independent
                        mixcrits[dst][rsp]["globcorr"][i_leadtime,i_anc,i_scl] = SB.mean(globcorr[cfg.lead_time_max, i_leadtime, :, i_anc], SB.weights(Ws[2:Npert+1]))
                        mixcrits[dst][rsp]["contcorr"][i_leadtime,i_anc,i_scl] = SB.mean(contcorr[cfg.lead_time_max, i_leadtime, :, i_anc], SB.weights(Ws[2:Npert+1]))
                        mixcrits[dst][rsp]["r2lin"][i_leadtime,i_anc,i_scl] = r2lin[i_leadtime,i_anc]
                        mixcrits[dst][rsp]["r2quad"][i_leadtime,i_anc,i_scl] = r2quad[i_leadtime,i_anc]
                        QG2L.ccdf_gridded_from_samples!(
                                                        @view(ccdfs[dst][rsp][:,i_leadtime,i_anc,i_scl]),@view(pdfs[dst][rsp][:,i_leadtime,i_anc,i_scl]),
                                                        Rs, Ws, levels
                                                       ) 
                        mixcrits[dst][rsp]["ent"][i_leadtime,i_anc,i_scl] = QG2L.entropy_fun_ccdf(ccdfs[dst][rsp][i_thresh_cquantile:Nlev,i_leadtime,i_anc,i_scl]; normalize=false)
                    end
                end
            end
        end
        for dst = ["b"]
            for rsp = ["e"]
                for i_scl = 1:Nscales[dst]
                    for mc = keys(mixobjs)
                        for (i_mcval,mcval) in enumerate(mixobjs[mc])
                            if "lt" == mc
                                ilts[dst][rsp][mc][i_mcval,i_anc,i_scl] = i_mcval
                            elseif mc in ["r2lin","r2quad","pth","pim","globcorr","contcorr"]
                                first_inceedance = findfirst(mixcrits[dst][rsp][mc][:,i_anc,i_scl] .<= mcval)
                                last_exceedance = findlast(mixcrits[dst][rsp][mc][:,i_anc,i_scl] .> mcval)
                                #ilts[dst][rsp][mc][i_mcval,i_anc,i_scl] = (isnothing(first_inceedance) ? Nleadtime : max(1,first_inceedance-1))
                                ilts[dst][rsp][mc][i_mcval,i_anc,i_scl] = (isnothing(last_exceedance) ? 1 : last_exceedance)
                            elseif mc in ["ei","ent"]
                                # Find first local maximum
                                mcdiff .= diff(mixcrits[dst][rsp][mc][1:Nleadtime,i_anc,i_scl])
                                mc_locmax_flag[2:end-1] .= (mcdiff[1:end-1] .> 0) .& (mcdiff[2:end] .< 0)
                                ilt_upper_bound = Nleadtime #findlast(sdm.tu .* leadtimes .< (3/4)/thresh_cquantile)
                                mc_locmax_flag[1] = false #(mcdiff[1] < 0)
                                mc_locmax_flag[ilt_upper_bound] = false #(mcdiff[end] > 0)
                                #@infiltrate #any(mc_locmax_flag)
                                # Could combine many different kinds of conditions for optimality and local maxima 
                                if any(mc_locmax_flag[1:ilt_upper_bound])

                                    idx_locmax = findall(mc_locmax_flag[1:ilt_upper_bound])
                                    ilts[dst][rsp][mc][i_mcval,i_anc,i_scl] = idx_locmax[argmax(mixcrits[dst][rsp][mc][idx_locmax,i_anc,i_scl])]
                                else
                                

                                    ilts[dst][rsp][mc][i_mcval,i_anc,i_scl] = argmax(mixcrits[dst][rsp][mc][1:ilt_upper_bound,i_anc,i_scl])
                                end
                            else
                                error()
                            end
                            iltcounts[dst][rsp][mc][
                                                    i_mcval,
                                                    ilts[dst][rsp][mc][i_mcval,i_anc,i_scl],
                                                    i_scl
                                                   ] += 1
                        end
                    end
                end
            end
        end
    end
    # Add in the statistics 
    JLD2.jldopen(joinpath(resultdir,"mixcrits_ccdfs_pdfs.jld2"),"w") do f
        f["mixcrits"] = mixcrits
        f["ccdfs"] = ccdfs
        f["pdfs"] = pdfs
        f["ilts"] = ilts
        f["iltcounts"] = iltcounts
    end
    return mixcrits
end


                        


function mix_COAST_distributions(cfg, cop, pertop, coast, ens, resultdir,)
    (
     leadtimes,r2threshes,dsts,rsps,mixobjs,
     mixcrit_labels,mixobj_labels,mixcrit_colors,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,i_thresh_cquantile,adjust_ccdf_per_ancestor
    ) = expt_config_COAST_analysis(cfg,pertop)
    thresh_cquantile = ccdf_levels[i_thresh_cquantile]
    Nleadtime = length(leadtimes)
    Nr2th = length(r2threshes)
    Nscales = Dict(dst=>length(distn_scales[dst]) for dst=dsts)
    (
     coefs_linear,residmse_linear,rsquared_linear,resid_range_linear,
     coefs_quadratic,residmse_quadratic,rsquared_quadratic,resid_range_quadratic,
     hessian_eigvals,hessian_eigvecs
    ) = JLD2.jldopen(joinpath(resultdir,"regression_coefs.jld2"), "r") do f
        return (
                f["coefs_linear"],
                f["residmse_linear"],
                f["rsquared_linear"],
                f["resid_range_linear"],
                f["coefs_quadratic"],
                f["residmse_quadratic"],
                f["rsquared_quadratic"],
                f["resid_range_quadratic"],
                f["hessian_eigvals"],
                f["hessian_eigvecs"],
               )
    end
    (
     tgrid_ancgen,
     Roft_ancgen_seplon,
     Rccdf_ancgen_seplon,
     Rccdf_ancgen_agglon,
     ccdf_pot_ancgen_seplon,
     ccdf_pot_ancgen_agglon,
     tgrid_valid,
     Roft_valid_seplon,
     Rccdf_valid_seplon,
     Rccdf_valid_agglon,
     ccdf_pot_valid_seplon,
     ccdf_pot_valid_agglon,
    ) = (
         JLD2.jldopen(joinpath(resultdir,"objective_dns_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).jld2"), "r") do f
             return (
                     f["tgrid_ancgen"], # tgrid_ancgen
                     f["Roft_ancgen_seplon"], # Roft_ancgen_seplon
                     f["Rccdf_ancgen_seplon"], # Rccdf_ancgen_seplon
                     f["Rccdf_ancgen_agglon"], # Rccdf_ancgen_seplon
                     f["ccdf_pot_ancgen_seplon"],
                     f["ccdf_pot_ancgen_agglon"],
                     f["tgrid_valid"], # tgrid_valid
                     f["Roft_valid_seplon"], # Roft_valid_seplon
                     f["Rccdf_valid_seplon"], # Rccdf_valid_seplon
                     f["Rccdf_valid_agglon"], # Rccdf_valid_agglon
                     f["ccdf_pot_valid_seplon"],
                     f["ccdf_pot_valid_agglon"],
                    )
         end
        )
    thresh = Rccdf_valid_agglon[i_thresh_cquantile] 
    levels = Rccdf_valid_agglon
    levels_mid = 0.5 .* (levels[1:end-1] .+ levels[2:end])
    levels_exc = levels[i_thresh_cquantile:end]
    levels_exc_mid = 0.5 .* (levels_exc[1:end-1] .+ levels_exc[2:end])
    dlev = diff(levels)
    Nlev = length(levels)
    i_level_highest_shortdns = Nlev #findlast(levels_exc .< maximum(Rccdf_ancgen_seplon))
    Nanc = length(coast.ancestors)
    # Load the various notions of field correlation
    globcorr,contcorr = JLD2.jldopen(joinpath(resultdir,"contour_dispersion.jld2"), "r") do f
        return f["globcorr"], f["contcorr"]
    end


    color_lin = :cyan
    color_quad = :sienna1
    i_mode_sf = 1
    max_score = maximum(vcat(coast.anc_Rmax, (coast.desc_Rmax[i_anc] for i_anc=1:Nanc)...))
    # Set up bootstrap resamplings
    rng_boot = Random.MersenneTwister(871940)
    ancs_boot = Random.rand(rng_boot, 1:Nanc, (Nanc, Nboot+1))
    anc_boot_mults = zeros(Int64, (Nanc,Nboot+1)) # multiplicities
    anc_boot_mults[:,1] .= 1
    for i_boot = 2:Nboot+1
        for i_anc = 1:Nanc
            anc_boot_mults[ancs_boot[i_anc,i_boot],i_boot] += 1
        end
    end

    mixcrits,ccdfs,pdfs,iltmixs,iltcounts = JLD2.jldopen(joinpath(resultdir,"mixcrits_ccdfs_pdfs.jld2"),"r") do f
        return f["mixcrits"],f["ccdfs"],f["pdfs"],f["ilts"],f["iltcounts"]
    end

    ccdfmixs,pdfmixs,fdivs = (Dict{String,Dict}() for _=1:3)
    for dst = ["b"]
        ccdfmixs[dst],pdfmixs[dst],fdivs[dst] = (Dict{String,Dict}() for _=1:3)
        for rsp = ["e"]
            ccdfmixs[dst][rsp],pdfmixs[dst][rsp],fdivs[dst][rsp] = (Dict{String,Dict}() for _=1:3)
            for mc = keys(mixobjs)
                ccdfmixs[dst][rsp][mc],pdfmixs[dst][rsp][mc] = (Dict{String,Array{Float64}}() for _=1:2)
                fdivs[dst][rsp][mc] = Dict{String,Dict}()
                for est = ["mix","pool"]
                    ccdfmixs[dst][rsp][mc][est],pdfmixs[dst][rsp][mc][est] = (zeros(Float64, (Nlev, Nboot+1, length(mixobjs[mc]), length(distn_scales[dst]))) for _=1:2)
                    fdivs[dst][rsp][mc][est] = Dict{String,Array{Float64}}()
                    for fdivname = fdivnames
                        fdivs[dst][rsp][mc][est][fdivname] = zeros(Float64, (Nboot+1, length(mixobjs[mc]), length(distn_scales[dst])))
                    end
                end
            end
        end
    end
    println("Initialized the fdivs and (cc,p)dfmixs")


    Nmem = EM.get_Nmem(ens)
    Ndsc = Nmem - Nanc
    idx_lev = collect(range(i_thresh_cquantile,i_level_highest_shortdns; step=1)) 
    Ndsc_per_leadtime = div(Ndsc, Nleadtime*Nanc)
    i_boot = 1
    for dst = ["b"]
        for rsp = ["e"]
            for i_scl = 1:length(distn_scales[dst])
                println("Starting scale $(i_scl)")
                # Iterate through each mixing objective of each mixing criterion
                for mc = keys(mixobjs)
                    Nmcv = length(mixobjs[mc])
                    for i_mcval = 1:Nmcv
                        for i_anc = 1:Nanc
                            @infiltrate !all(isfinite.(ccdfmixs[dst][rsp][mc]["pool"][i_thresh_cquantile:Nlev,i_boot,i_mcval,i_scl]))
                            ilt = iltmixs[dst][rsp][mc][i_mcval,i_anc,i_scl]
                            ccdfmixs[dst][rsp][mc]["pool"][:,i_boot,i_mcval,i_scl] .+= (anc_boot_mults[i_anc,i_boot]/Nanc) .* ccdfs[dst][rsp][:,ilt,i_anc,i_scl]
                            pth = ccdfs[dst][rsp][i_thresh_cquantile,ilt,i_anc,i_scl]
                            @infiltrate pth <= 0
                            pthmix = ccdfmixs[dst][rsp][mc]["pool"][i_thresh_cquantile,i_boot,i_mcval,i_scl]
                            @infiltrate !(pthmix > 0)
                            ccdfmixs[dst][rsp][mc]["mix"][i_thresh_cquantile:Nlev,i_boot,i_mcval,i_scl] .+= (anc_boot_mults[i_anc,i_boot]/Nanc) .* (ccdfs[dst][rsp][i_thresh_cquantile:Nlev,ilt,i_anc,i_scl] .+ (1-pth).*(coast.anc_Rmax[i_anc] .> levels[i_thresh_cquantile:Nlev]))
                            @infiltrate !QG2L.check_ccdf_validity(ccdfmixs[dst][rsp][mc]["mix"][i_thresh_cquantile:Nlev,i_boot,i_mcval,i_scl])
                            @infiltrate !QG2L.check_ccdf_validity(ccdfmixs[dst][rsp][mc]["pool"][i_thresh_cquantile:Nlev,i_boot,i_mcval,i_scl])
                        end
                    end
                    # normalize 
                    ccdfmixs[dst][rsp][mc]["pool"][i_thresh_cquantile:Nlev, 1:Nboot+1, 1:Nmcv, i_scl] ./= ccdfmixs[dst][rsp][mc]["pool"][i_thresh_cquantile:i_thresh_cquantile, 1:Nboot+1, 1:Nmcv, i_scl]
                    ccdfmixs[dst][rsp][mc]["pool"][1:i_thresh_cquantile-1, :, :, i_scl] .= NaN
                    ccdfmixs[dst][rsp][mc]["mix"][1:i_thresh_cquantile-1, :, :, i_scl] .= NaN
                    for est = ["mix","pool"]
                        # TODO manual broadcast 
                        pdfmixs[dst][rsp][mc][est][1:Nlev-1, 1:Nboot+1, 1:Nmcv, 1:Nscales[dst]] .= -diff(ccdfmixs[dst][rsp][mc][est][1:Nlev,1:Nboot+1,1:Nmcv,1:Nscales[dst]]; dims=1) ./ diff(levels)
                        pdfmixs[dst][rsp][mc][est][Nlev, 1:Nboot+1, 1:Nmcv, 1:Nscales[dst]] .= -ccdfmixs[dst][rsp][mc][est][Nlev,1:Nboot+1,1:Nmcv,1:Nscales[dst]] ./ (levels[Nlev]-levels[Nlev-1])
                    end
                    # Penalize 
                    for fdivname = fdivnames
                        for est = ["mix","pool"]
                            for i_mcval = 1:length(mixobjs[mc])
                                @assert maximum(abs.(1 .- [ccdfmixs[dst][rsp][mc][est][i_thresh_cquantile,i_boot,i_mcval,i_scl], ccdf_pot_valid_agglon[1]])) < 1e-6
                                # Only integrte up to the maximum achieve by short DNS 
                                pmf1,pmf2 = ccdfmixs[dst][rsp][mc][est][idx_lev,i_boot,i_mcval,i_scl], ccdf_pot_valid_agglon[1:length(idx_lev)]
                                fdivs[dst][rsp][mc][est][fdivname][i_boot,i_mcval,i_scl] = QG2L.fdiv_fun_ccdf(ccdfmixs[dst][rsp][mc][est][idx_lev,i_boot,i_mcval,i_scl], ccdf_pot_valid_agglon[1:length(idx_lev)], levels_exc[1:length(idx_lev)], levels_exc[1:length(idx_lev)], fdivname)
                            end
                        end
                    end
                end
            end
        end
    end
    fdivs_ancgen_valid = Dict()
    for fdivname = fdivnames

        fdivs_ancgen_valid[fdivname] = mapslices(ccdf_pot->QG2L.fdiv_fun_ccdf(
                                                                              ccdf_pot[1:length(idx_lev)], 
                                                                              ccdf_pot_valid_agglon[1:length(idx_lev)], 
                                                                              levels[idx_lev], 
                                                                              levels[idx_lev], 
                                                                              fdivname
                                                                         ), 
                                                 ccdf_pot_ancgen_seplon; dims=1)[1,:]
        #@infiltrate any(isnan.(fdivs_ancgen_valid[fdivname]))
    end

    JLD2.jldopen(joinpath(resultdir,"ccdfs_combined.jld2"),"w") do f
        # coordinates for parameters of distributions 
        f["levels"] = levels
        f["levels_mid"] = levels_mid
        f["dstns"] = dsts
        f["rsps"] = rsps
        f["mixobjs"] = mixobjs
        f["distn_scales"] = distn_scales
        # output distributions 
        f["ccdfs"] = ccdfs
        f["pdfs"] = pdfs
        f["fdivs"] = fdivs
        f["fdivs_ancgen_valid"] = fdivs_ancgen_valid
        f["mixcrits"] = mixcrits
        f["iltmixs"] = iltmixs
        f["iltcounts"] = iltcounts
        f["ccdfmixs"] = ccdfmixs
        f["pdfmixs"] = pdfmixs 
    end
end
