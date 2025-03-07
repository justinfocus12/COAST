function mix_COAST_distributions_polynomial(cfg, cop, pertop, coast, ens, resultdir,)
    (
     leadtimes,r2threshes,dsts,rsps,mixobjs,
     mixcrit_labels,mixobj_labels,mixcrit_colors,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,i_thresh_cquantile,adjust_ccdf_per_ancestor
    ) = expt_config_COAST_analysis(cfg,pertop)
    thresh_cquantile = ccdf_levels[i_thresh_cquantile]
    Nleadtime = length(leadtimes)
    Nr2th = length(r2threshes)
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


    # Doubly nested dictionary: by input distribution, and by map 
    ccdfs,pdfs,mixcrits,ccdfmixs,pdfmixs,iltmixs,fdivs = (Dict() for _=1:7)
    # Add in ETH ensemble boosting estimator (even though we could derive it)
    ccweights = Dict() # complementary cumulative weights
    ccdfpools = Dict()
    fdivpools = Dict()

    # Set the global radii for input distributions 
    i_mode_sf = 1
    support_radius = pertop.sf_pert_amplitudes_max[i_mode_sf]
    
    for dst = dsts
        ccweights[dst],ccdfpools[dst],ccdfs[dst],pdfs[dst],mixcrits[dst],ccdfmixs[dst],pdfmixs[dst],iltmixs[dst],fdivs[dst],fdivpools[dst] = (Dict() for _=1:10)
        for rsp = rsps
            if ("g" == dst) && (rsp in ["1+u","2","2+u"])
                continue
            end
            ccweights[dst][rsp] = zeros(Float64, (Nlev, Nleadtime, Nanc, length(distn_scales[dst])))
            ccdfs[dst][rsp] = zeros(Float64, (Nlev, Nleadtime, Nanc, length(distn_scales[dst])))
            pdfs[dst][rsp] = zeros(Float64, (Nlev-1, Nleadtime, Nanc, length(distn_scales[dst])))
            ccdfpools[dst][rsp],mixcrits[dst][rsp],ccdfmixs[dst][rsp],pdfmixs[dst][rsp],iltmixs[dst][rsp],fdivs[dst][rsp],fdivpools[dst][rsp] = (Dict() for _=1:7)
            for mc = keys(mixobjs)
                mixcrits[dst][rsp][mc] = zeros(Float64, (Nleadtime, Nanc, length(distn_scales[dst])))
                iltmixs[dst][rsp][mc] = zeros(Int64, (length(mixobjs[mc]), Nanc, length(distn_scales[dst])))
                # TODO bootstrap for confidence intervals on mixture 
                ccdfmixs[dst][rsp][mc] = zeros(Float64, (Nlev, Nboot+1, length(mixobjs[mc]), length(distn_scales[dst])))
                pdfmixs[dst][rsp][mc] = zeros(Float64, (Nlev-1, Nboot+1, length(mixobjs[mc]), length(distn_scales[dst])))
                ccdfpools[dst][rsp][mc] = zeros(Float64, (Nlev, Nboot+1, length(mixobjs[mc]), length(distn_scales[dst])))
                fdivs[dst][rsp][mc] = Dict(fdivname=>zeros(Float64, (Nboot+1, length(mixobjs[mc]), length(distn_scales[dst]))) for fdivname = fdivnames)
                fdivpools[dst][rsp][mc] = Dict(fdivname=>zeros(Float64, (Nboot+1, length(mixobjs[mc]), length(distn_scales[dst]))) for fdivname = fdivnames)
            end
        end
    end
    println("Initialized the fdivs and mixs")



    # Combine based on Rsq and other indicators 
    #
    # pre-allocate the QMC sequence
    Nsamp_reg2dist = 128^2
    U_reg2dist = collect(transpose(QMC.sample(Nsamp_reg2dist, zeros(Float64, 2), ones(Float64, 2), QMC.LatticeRuleSample())))
    function r2dfun(d,r,s)
        # Below, resid might refer to either a single MSE value or a range. 
        if "b" == d # bump function
            if "1" == r
                return ((coefs,resid)->QG2L.regression2distn_linear_bump(coefs, s, support_radius, levels, U_reg2dist))
            elseif "2" == r
                return ((coefs,resid)->QG2L.regression2distn_quadratic_bump(coefs, s, support_radius, levels, U_reg2dist))
            elseif "e" == r
                return ((scores,resid)->QG2L.regression2distn_empirical_bump(scores, s, support_radius, levels, U_reg2dist[1:length(scores),:]))
            end
        elseif "u" == d
            if "1" == r
                return ((coefs,resid)->QG2L.regression2distn_linear_uniform(coefs, s, levels))
            elseif "2" == r
                return ((coefs,resid)->QG2L.regression2distn_quadratic_uniform(coefs, s, levels, U_reg2dist))
            elseif "1+u" == r
                return ((coefs,resid)->QG2L.regression2distn_linear_uniform(coefs, resid, s, levels))
            elseif "1+g" == r
                return ((coefs,resid)->QG2L.regression2distn_linear_uniform(coefs, resid, s, levels))
            elseif "2+u" == r
                return ((coefs,resid)->QG2L.regression2distn_quadratic_uniform(coefs, resid, s, levels))
            end
        elseif "g" == d
            if "1" == r
                return ((coefs,resid)->QG2L.regression2distn_linear_gaussian(coefs, s, levels))
            elseif "1+g" == r
                return ((coefs,resid)->QG2L.regression2distn_linear_gaussian(coefs, resid, s, levels))
            else
                error("r2dfun is not defined for the combination of d = $d, r = $r")
            end
        end
    end
    #Ndsc_per_leadtime = div(cfg.num_perts_max, Nleadtime)
    Nmem = EM.get_Nmem(ens)
    Ndsc = Nmem - Nanc
    Ndsc_per_leadtime = div(Ndsc, Nleadtime*Nanc)
    anc_dsc_weights = ones(Float64, 1+Ndsc_per_leadtime) # For computing averages for regression skills or correlations 
    for dst = dsts
        for rsp = rsps
            coefs_at_anc_and_leadtime(i_leadtime,i_anc) = begin
                if rsp in ["1","1+u","1+g"] # == rsp
                    return coefs_linear[:,i_leadtime,i_anc]
                elseif rsp in ["2","2+u","1+u"] #"2" == rsp
                    return coefs_quadratic[:,i_leadtime,i_anc]
                elseif "e" == rsp
                    idx_desc = desc_by_leadtime(coast, i_anc, leadtimes[i_leadtime], sdm)
                    return vcat([coast.anc_Rmax[i_anc]], coast.desc_Rmax[i_anc][idx_desc])
                end
            end
            if ("g" == dst) && (rsp in ["2","2+u","1+u"])
                continue
            end
            @show dst,rsp
            residmse = 0.0
            resid_range = zeros(Float64,2)
            resid_arg = NaN
            #Threads.@threads for i_scl = 1:length(distn_scales[dst])
            anc_dsc_scores = zeros(Float64, 1+Ndsc_per_leadtime)
            for i_scl = 1:length(distn_scales[dst])
                println("Starting scale $(i_scl)")
                scl = distn_scales[dst][i_scl]
                anc_dsc_weights[2:Ndsc_per_leadtime+1] .= QG2L.bump_density(U_reg2dist[1:Ndsc_per_leadtime,:], scl, support_radius)
                anc_dsc_weights[1:2] .= QG2L.bump_density(zeros(Float64, (1,2)), scl, support_radius)
                #Threads.@threads for i_anc = 1:Nanc
                for i_anc = 1:Nanc
                    Rmaxanc = coast.anc_Rmax[i_anc]
                    for i_leadtime = 1:Nleadtime
                        #if "g" == dst
                        #    resid_range .= resid_range_linear[:,i_leadtime,i_anc]
                        #    residmse = residmse_linear[i_leadtime,i_anc]
                        #    resid_arg = residmse
                        if rsp in ["1","2","e"]
                            resid_range .= [0.0, 0.0] 
                            resid_arg = resid_range
                        elseif rsp in ["1+u"]
                            resid_range .= resid_range_linear[:,i_leadtime,i_anc]
                            resid_arg = resid_range
                        elseif rsp in ["2+u"]
                            resid_range .= resid_range_quadratic[:,i_leadtime,i_anc]
                            resid_arg = resid_range
                        elseif "1+g" == rsp
                            residmse = residmse_linear[i_leadtime,i_anc]
                            resid_arg = residmse
                        else
                            error()
                        end
                        ccdf,pdf = r2dfun(dst,rsp,scl)(coefs_at_anc_and_leadtime(i_leadtime,i_anc), resid_arg)
                        # --------------- pooled version ---------------
                        anc_dsc_scores .= vcat([coast.anc_Rmax[i_anc]], coast.desc_Rmax[i_anc][desc_by_leadtime(coast, i_anc, leadtimes[i_leadtime], sdm)[1:Ndsc_per_leadtime]])
                        ccweights[dst][rsp][:,i_leadtime,i_anc,i_scl] .+= sum(anc_dsc_weights .* (anc_dsc_scores .> levels'); dims=1)[1,:] 
                        # ----------------------------------------------
                        pth = ccdf[i_thresh_cquantile]
                        ccdfs[dst][rsp][:,i_leadtime,i_anc,i_scl] .= ccdf #.* adjustment
                        pdfs[dst][rsp][:,i_leadtime,i_anc,i_scl] .= pdf #.* adjustment
                        #ccdfs[dst][rsp][1:i_thresh_cquantile-1,i_leadtime,i_anc,i_scl] .= thresh_cquantile
                        if !(all(isfinite.(pdfs[dst][rsp][:,i_leadtime,i_anc,i_scl])) && all(isfinite.(ccdfs[dst][rsp][:,i_leadtime,i_anc,i_scl])))
                            println("non-finite pdf or ccdf")
                            display(pdfs[dst][rsp][:,i_leadtime,i_anc,i_scl])
                            display(ccdfs[dst][rsp][:,i_leadtime,i_anc,i_scl])
                            @show i_anc, ccdf[1] 
                            error()
                        end
                        # Evaluate these distributions by mixing criteria 
                        mixcrits[dst][rsp]["pth"][i_leadtime,i_anc,i_scl] = pth
                        mixcrits[dst][rsp]["lt"][i_leadtime,i_anc,i_scl] = leadtimes[i_leadtime]
                        mixcrits[dst][rsp]["r2"][i_leadtime,i_anc,i_scl] = (rsp in ["1","1+u","1+g"] ? rsquared_linear : rsquared_quadratic)[i_leadtime,i_anc]
                        mixcrits[dst][rsp]["ei"][i_leadtime,i_anc,i_scl] = sum(max.(0, levels_mid .- Rmaxanc) .* (levels[1:Nlev-1] .> Rmaxanc) .* pdf .* dlev)
                        mixcrits[dst][rsp]["eot"][i_leadtime,i_anc,i_scl] = sum(0.5 .* (ccdf[i_thresh_cquantile:Nlev-1] .+ ccdf[i_thresh_cquantile+1:Nlev]) .* dlev[i_thresh_cquantile:Nlev-1]) 
                        #mixcrits[dst][rsp]["eot"][i_leadtime,i_anc,i_scl] += ccdf[Nlev]/(2*dlev[Nlev-1])

                        # weight the entropy by the probability of exceeding the threshold 
                        mixcrits[dst][rsp]["ent"][i_leadtime,i_anc,i_scl] = QG2L.entropy_fun_ccdf(ccdf[i_thresh_cquantile:end]; normalize=false) #pth*QG2L.entropy_fun_ccdf(ccdf[i_thresh_cquantile:end])
                        mixcrits[dst][rsp]["went"][i_leadtime,i_anc,i_scl] = ccdf[i_thresh_cquantile] * QG2L.entropy_fun_ccdf(ccdf[i_thresh_cquantile:end])
                        # For correlations, the averaging weight depends on the scale, so these two aren't totally independent
                        mixcrits[dst][rsp]["globcorr"][i_leadtime,i_anc,i_scl] = SB.mean(globcorr[cfg.lead_time_max, i_leadtime, :, i_anc], SB.weights(anc_dsc_weights[2:Ndsc_per_leadtime+1]))
                        mixcrits[dst][rsp]["contcorr"][i_leadtime,i_anc,i_scl] = SB.mean(contcorr[cfg.lead_time_max, i_leadtime, :, i_anc], SB.weights(anc_dsc_weights[2:Ndsc_per_leadtime+1]))
                        if levels[end] >= Rmaxanc
                            i_lev_lo = findlast(levels .< Rmaxanc)
                            i_lev_hi = findfirst(levels .>= Rmaxanc)
                            slope = (ccdf[i_lev_hi] - ccdf[i_lev_lo])/(levels[i_lev_hi] - levels[i_lev_lo])
                            mixcrits[dst][rsp]["pim"][i_leadtime,i_anc,i_scl] = ccdf[i_lev_lo] + slope*(Rmaxanc - levels[i_lev_lo])
                            @assert ccdf[i_lev_lo] >= mixcrits[dst][rsp]["pim"][i_leadtime,i_anc,i_scl] >= ccdf[i_lev_hi]
                            mixcrits[dst][rsp]["ei"][i_leadtime,i_anc,i_scl] += (mixcrits[dst][rsp]["pim"][i_leadtime,i_anc,i_scl] - ccdf[i_lev_hi]) * (levels[i_lev_hi] + Rmaxanc)/2 * (levels[i_lev_hi] - Rmaxanc)

                        end
                    end
                    # optimize each mixing objective across lead times 
                    # R-squared
                    for (i_r2thresh,r2thresh) in enumerate(r2threshes)
                        first_inceedance = findfirst(mixcrits[dst][rsp]["r2"][:,i_anc,i_scl] .< r2thresh)
                        #IFT.@infiltrate ("b" == dst) && ("2" == rsp) && (i_scl >= 3) 
                        # TODO investigate whether last exceedance is better
                        iltmixs[dst][rsp]["r2"][i_r2thresh,i_anc,i_scl] = (isnothing(first_inceedance) ? Nleadtime : max(1, first_inceedance-1))
                    end
                    ilt_r2 = ("e" == rsp ? Nleadtime : iltmixs[dst][rsp]["r2"][1,i_anc,i_scl])
                    for i_leadtime = 1:Nleadtime
                        iltmixs[dst][rsp]["lt"][i_leadtime,i_anc,i_scl] = min(i_leadtime, ilt_r2)
                    end
                    for (i_pth,pth) in enumerate(mixobjs["pth"])
                        first_inceedance = findfirst(mixcrits[dst][rsp]["pth"][1:ilt_r2,i_anc,i_scl] .< pth)
                        last_exceedance = findlast(mixcrits[dst][rsp]["pth"][1:ilt_r2,i_anc,i_scl] .>= pth)
                        #IFT.@infiltrate ("b" == dst) && ("2" == rsp) && (i_scl >= 3) 
                        # TODO investigate whether last exceedance is better
                        #iltmixs[dst][rsp]["pth"][i_pth,i_anc,i_scl] = (isnothing(first_inceedance) ? ilt_r2 : max(1, first_inceedance-1))
                        iltmixs[dst][rsp]["pth"][i_pth,i_anc,i_scl] = (isnothing(last_exceedance) ? 1 : last_exceedance)
                    end
                    for (i_pim,pim) in enumerate(mixobjs["pim"])
                        first_inceedance = findfirst(mixcrits[dst][rsp]["pim"][1:ilt_r2,i_anc,i_scl] .< pim)
                        last_exceedance = findlast(mixcrits[dst][rsp]["pim"][1:ilt_r2,i_anc,i_scl] .>= pim)
                        #IFT.@infiltrate ("b" == dst) && ("2" == rsp) && (i_scl >= 3) 
                        # TODO investigate whether last exceedance is better
                        #iltmixs[dst][rsp]["pim"][i_pim,i_anc,i_scl] = (isnothing(first_inceedance) ? ilt_r2 : max(1, first_inceedance-1))
                        iltmixs[dst][rsp]["pim"][i_pim,i_anc,i_scl] = (isnothing(last_exceedance) ? 1 : last_exceedance)
                    end

                    for corrkey = ["globcorr","contcorr"]
                        for (i_corr,corr) in enumerate(mixobjs[corrkey])
                            first_inceedance = findfirst(mixcrits[dst][rsp][corrkey][1:ilt_r2,i_anc,i_scl] .< corr)
                            last_exceedance = findlast(mixcrits[dst][rsp][corrkey][1:ilt_r2,i_anc,i_scl] .>= corr)
                            iltmixs[dst][rsp][corrkey][i_corr,i_anc,i_scl] = (isnothing(last_exceedance) ? 1 : last_exceedance)
                        end
                    end

                    # Other objectives to condition on R^2 
                    for mc = ("ent","ei","eot")
                        iltmixs[dst][rsp][mc][1,i_anc,i_scl] = argmax(mixcrits[dst][rsp][mc][1:ilt_r2,i_anc,i_scl])
                    end
                end
                # Now average together the PDFs and CCDFs based on the criteria from above 
                println("Starting to sum together pdfs and ccdfs")
                #IFT.@infiltrate ((dst=="b")&(rsp=="2"))
                anc_weights = zeros(Float64, Nanc)
                for mc = ("globcorr","contcorr","pim","pth","ent","ei","eot","lt","r2",) #keys(mixobjs)
                    for (i_mcobj,mcobj) in enumerate(mixobjs[mc])
                        for i_boot = 1:Nboot+1
                            anc_weights .= 0
                            if adjust_ccdf_per_ancestor
                                for i_anc = 1:Nanc
                                    ilt = iltmixs[dst][rsp][mc][i_mcobj,i_anc,i_scl] 
                                    pth = ccdfs[dst][rsp][i_thresh_cquantile,ilt,i_anc,i_scl]
                                    @assert pth > 0
                                    anc_weights[i_anc] = (0 == pth ? 0.0 : anc_boot_mults[i_anc]/pth)
                                    ccdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl] .+= anc_weights[i_anc].*ccdfs[dst][rsp][:,ilt,i_anc,i_scl]
                                    pdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl] .+= anc_weights[i_anc].*pdfs[dst][rsp][:,ilt,i_anc,i_scl] 
                                    #@infiltrate
                                    if !all(isfinite.(pdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl]))
                                        println("non-finite PDF for i_boot=$(i_boot)")
                                        display(pdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl])
                                        error()
                                    end
                                end
                                adjustment = thresh_cquantile / ccdfmixs[dst][rsp][mc][i_thresh_cquantile,i_boot,i_mcobj,i_scl]
                                ccdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl] .*= adjustment
                                pdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl] .*= adjustment
                            else
                                for i_anc = 1:Nanc
                                    ilt = iltmixs[dst][rsp][mc][i_mcobj,i_anc,i_scl] 
                                    pth = ccdfs[dst][rsp][i_thresh_cquantile,ilt,i_anc,i_scl]
                                    @assert pth > 0
                                    anc_weights[i_anc] = anc_boot_mults[i_anc]
                                    i_lev_anc = findfirst(coast.anc_Rmax[i_anc] .> levels)
                                    ccdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl] .+= anc_weights[i_anc].*(ccdfs[dst][rsp][:,ilt,i_anc,i_scl] .+ (1-pth).*(i_lev_anc .> (1:Nlev)))

                                    pdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl] .+= anc_weights[i_anc].*pdfs[dst][rsp][:,ilt,i_anc,i_scl] .* (1 .+ (1-pth).*(1:(Nlev-1) .== i_lev_anc))
                                    ccdfpools[dst][rsp][mc][:,i_boot,i_mcobj,i_scl] .+= anc_weights[i_anc].*ccweights[dst][rsp][:,ilt,i_anc,i_scl]
                                    #@infiltrate
                                    if !all(isfinite.(pdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl]))
                                        println("non-finite PDF for i_boot=$(i_boot)")
                                        display(pdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl])
                                        error()
                                    end
                                end
                                adjustment = thresh_cquantile / ccdfmixs[dst][rsp][mc][i_thresh_cquantile,i_boot,i_mcobj,i_scl]
                                ccdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl] .*= adjustment
                                pdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl] .*= adjustment
                                adjustment_pooled = thresh_cquantile / ccdfpools[dst][rsp][mc][i_thresh_cquantile,i_boot,i_mcobj,i_scl]
                                ccdfpools[dst][rsp][mc][:,i_boot,i_mcobj,i_scl] .*= adjustment_pooled
                                #@infiltrate
                            end
                        end
                        #IFT.@infiltrate ((dst=="b")&(rsp=="2")&(i_scl==2))
                        #println("Starting to compute fdivs")
                        #IFT.@infiltrate ("lt"==mc)
                        for i_boot = 1:Nboot+1
                            for fdivname = fdivnames
                                # TODO get the right baseline for subasymptotic POt
                                fdivs[dst][rsp][mc][fdivname][i_boot,i_mcobj,i_scl] = QG2L.fdiv_fun_ccdf(ccdfmixs[dst][rsp][mc][i_thresh_cquantile:end,i_boot,i_mcobj,i_scl], ccdf_pot_valid_agglon, levels_exc, levels_exc, fdivname)
                                fdivpools[dst][rsp][mc][fdivname][i_boot,i_mcobj,i_scl] = QG2L.fdiv_fun_ccdf(ccdfpools[dst][rsp][mc][i_thresh_cquantile:end,i_boot,i_mcobj,i_scl], ccdf_pot_valid_agglon, levels_exc, levels_exc, fdivname)
                            end
                        end
                    end
                end
                #IFT.@infiltrate true
            end
        end
    end
    fdivs_ancgen_valid = Dict()
    for fdivname = fdivnames
        fdivs_ancgen_valid[fdivname] = mapslices(ccdf->QG2L.fdiv_fun_ccdf(ccdf, ccdf_pot_valid_agglon, levels[i_thresh_cquantile:end], levels[i_thresh_cquantile:end], fdivname), ccdf_pot_ancgen_seplon; dims=1)[1,:]
        #@infiltrate any(isnan.(fdivs_ancgen_valid[fdivname]))
    end

    for filename = readdir(resultdir,join=true)
        if endswith(filename, "ccdfs_regressed.jld2")
            rm(filename)
        end
    end
    JLD2.jldopen(joinpath(resultdir,"ccdfs_regressed_accpa$(Int(adjust_ccdf_per_ancestor)).jld2"),"w") do f
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
        f["ccdfmixs"] = ccdfmixs
        f["pdfmixs"] = pdfmixs 
        f["ccweights"] = ccweights
        f["ccdfpools"] = ccdfpools
        f["fdivpools"] = fdivpools
    end
end
