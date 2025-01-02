function mix_COAST_distributions_polynomial(cfg, cop, pertop, coast, resultdir,)
    (
     leadtimes,r2threshes,dsts,rsps,mixobjs,
     mixcrit_labels,mixobj_labels,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,thresh_cquantile
    ) = expt_config_COAST_analysis(cfg,pertop)
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
     tgrid_valid,
     Roft_valid_seplon,
     Rccdf_valid_seplon,
     Rccdf_valid_agglon,
    ) = (
         JLD2.jldopen(joinpath(resultdir,"objective_dns_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).jld2"), "r") do f
             return (
                     f["tgrid_ancgen"], # tgrid_ancgen
                     f["Roft_ancgen_seplon"], # Roft_ancgen_seplon
                     f["Rccdf_ancgen_seplon"], # Rccdf_ancgen_seplon
                     f["Rccdf_ancgen_agglon"], # Rccdf_ancgen_seplon
                     f["tgrid_valid"], # tgrid_valid
                     f["Roft_valid_seplon"], # Roft_valid_seplon
                     f["Rccdf_valid_seplon"], # Rccdf_valid_seplon
                     f["Rccdf_valid_agglon"], # Rccdf_valid_agglon
                    )
         end
        )
    thresh = Rccdf_valid_agglon[argmin(abs.(ccdf_levels .- thresh_cquantile))] 
    pdf_valid_agglon = -diff(Rccdf_valid_agglon) ./ diff(ccdf_levels)
    levels = Rccdf_valid_agglon
    levels_mid = 0.5 .* (levels[1:end-1] .+ levels[2:end])
    dlev = diff(levels)
    Nlev = length(levels)
    Nanc = length(coast.ancestors)


    color_lin = :cyan
    color_quad = :sienna1
    i_mode_sf = 1
    max_score = maximum(vcat(coast.anc_Rmax, (coast.desc_Rmax[i_anc] for i_anc=1:Nanc)...))
    # Set up bootstrap resamplings
    anc_weights = zeros(Float64, (Nanc,Nboot+1))
    anc_weights[:,1] .= 1 ./ Nanc
    rng_boot = Random.MersenneTwister(871940)
    ancs_boot = Random.rand(rng_boot, 1:Nanc, (Nanc, Nboot+1))
    for i_boot = 2:Nboot+1
        for i_anc = 1:Nanc
            anc_weights[ancs_boot[i_anc,i_boot],i_boot] += 1/Nanc
        end
    end


    # Doubly nested dictionary: by input distribution, and by map 
    ccdfs,pdfs,mixcrits,ccdfmixs,pdfmixs,iltmixs,fdivs = (Dict() for _=1:7)

    # Set the global radii for input distributions 
    i_mode_sf = 1
    support_radius = pertop.sf_pert_amplitudes_max[i_mode_sf]
    
    for dst = dsts
        ccdfs[dst],pdfs[dst],mixcrits[dst],ccdfmixs[dst],pdfmixs[dst],iltmixs[dst],fdivs[dst] = (Dict() for _=1:7)
        for rsp = rsps
            if ("g" == dst) && (rsp in ["1+u","2","2+u"])
                continue
            end
            ccdfs[dst][rsp] = zeros(Float64, (Nlev, Nleadtime, Nanc, length(distn_scales[dst])))
            pdfs[dst][rsp] = zeros(Float64, (Nlev-1, Nleadtime, Nanc, length(distn_scales[dst])))
            mixcrits[dst][rsp],ccdfmixs[dst][rsp],pdfmixs[dst][rsp],iltmixs[dst][rsp],fdivs[dst][rsp] = (Dict() for _=1:5)
            for mc = keys(mixobjs)
                mixcrits[dst][rsp][mc] = zeros(Float64, (Nleadtime, Nanc, length(distn_scales[dst])))
                iltmixs[dst][rsp][mc] = zeros(Int64, (length(mixobjs[mc]), Nanc, length(distn_scales[dst])))
                # TODO bootstrap for confidence intervals on mixture 
                ccdfmixs[dst][rsp][mc] = zeros(Float64, (Nlev, Nboot+1, length(mixobjs[mc]), length(distn_scales[dst])))
                pdfmixs[dst][rsp][mc] = zeros(Float64, (Nlev-1, Nboot+1, length(mixobjs[mc]), length(distn_scales[dst])))
                fdivs[dst][rsp][mc] = Dict(fdivname=>zeros(Float64, (Nboot+1, length(mixobjs[mc]), length(distn_scales[dst]))) for fdivname = fdivnames)
            end
        end
    end
    println("Initialized the fdivs and mixs")



    # Combine based on Rsq and other indicators 
    #
    function r2dfun(d,r,s)
        # Below, resid might refer to either a single MSE value or a range. 
        if "b" == d # bump function
            if "1" == r
                return ((coefs,resid)->QG2L.regression2distn_linear_bump(coefs, s, support_radius, levels))
            elseif "2" == r
                return ((coefs,resid)->QG2L.regression2distn_quadratic_bump(coefs, s, support_radius, levels))
            end
        elseif "u" == d
            if "1" == r
                return ((coefs,resid)->QG2L.regression2distn_linear_uniform(coefs, s, levels))
            elseif "2" == r
                return ((coefs,resid)->QG2L.regression2distn_quadratic_uniform(coefs, s, levels))
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
    for dst = dsts
        for rsp = rsps
            if rsp in ["1","1+u","1+g"] # == rsp
                coefs = coefs_linear
            elseif rsp in ["2","2+u","1+u"] #"2" == rsp
                coefs = coefs_quadratic
            end
            if ("g" == dst) && (rsp in ["2","2+u","1+u"])
                continue
            end
            @show dst,rsp
            residmse = 0.0
            resid_range = zeros(Float64,2)
            resid_arg = NaN
            for i_scl = 1:length(distn_scales[dst])
            #for (i_scl,scl) in enumerate(distn_scales[dst])
                scl = distn_scales[dst][i_scl]
                Threads.@threads for i_anc = 1:Nanc
                #for i_anc = 1:Nanc
                    Rmaxanc = coast.anc_Rmax[i_anc]
                    for i_leadtime = 1:Nleadtime
                        #if "g" == dst
                        #    resid_range .= resid_range_linear[:,i_leadtime,i_anc]
                        #    residmse = residmse_linear[i_leadtime,i_anc]
                        #    resid_arg = residmse
                        if rsp in ["1","2"]
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
                        ccdf,pdf = r2dfun(dst,rsp,scl)(coefs[:,i_leadtime,i_anc], resid_arg)
                        adjustment = (ccdf[1] > 1e-6 ? thresh_cquantile/ccdf[1] : 0)
                        ccdfs[dst][rsp][:,i_leadtime,i_anc,i_scl] .= ccdf .* adjustment
                        pdfs[dst][rsp][:,i_leadtime,i_anc,i_scl] .= pdf .* adjustment
                        if !(all(isfinite.(pdfs[dst][rsp][:,i_leadtime,i_anc,i_scl])) && all(isfinite.(ccdfs[dst][rsp][:,i_leadtime,i_anc,i_scl])))
                            println("non-finite pdf or ccdf")
                            display(pdfs[dst][rsp][:,i_leadtime,i_anc,i_scl])
                            display(ccdfs[dst][rsp][:,i_leadtime,i_anc,i_scl] .= ccdf .* adjustment)
                            @show i_anc, adjustment, ccdf[1] 
                            error()
                        end
                        # Evaluate these distributions by mixing criteria 
                        mixcrits[dst][rsp]["lt"][i_leadtime,i_anc,i_scl] = leadtimes[i_leadtime]
                        mixcrits[dst][rsp]["r2"][i_leadtime,i_anc,i_scl] = (rsp in ["1","1+u","1+g"] ? rsquared_linear : rsquared_quadratic)[i_leadtime,i_anc]
                        mixcrits[dst][rsp]["ei"][i_leadtime,i_anc,i_scl] = sum(max.(0, levels_mid .- thresh) .* pdf .* dlev)
                        pdfrect = pdf .* (levels_mid .> thresh)
                        # weight the entropy by the probability of exceeding the threshold 
                        mixcrits[dst][rsp]["ent"][i_leadtime,i_anc,i_scl] = QG2L.entropy_fun(pdf .* (levels[1:end-1] .> thresh))
                        mixcrits[dst][rsp]["went"][i_leadtime,i_anc,i_scl] = sum(pdf .* dlev .* (levels[1:end-1] .> thresh)) * QG2L.entropy_fun(pdf .* (levels_mid .> thresh))
                        if levels[end] > Rmaxanc
                            mixcrits[dst][rsp]["pi"][i_leadtime,i_anc,i_scl] = ccdf[findfirst(levels .> Rmaxanc)]
                        end
                    end
                    # optimize each mixing objective across lead times 
                    # Lead time itself (this loop is just pedantic) 
                    for i_leadtime = 1:Nleadtime
                        iltmixs[dst][rsp]["lt"][i_leadtime,i_anc,i_scl] = i_leadtime
                    end
                    # R-squared
                    for (i_r2thresh,r2thresh) in enumerate(r2threshes)
                        first_inceedance = findfirst(mixcrits[dst][rsp]["r2"][:,i_anc,i_scl] .< r2thresh)
                        iltmixs[dst][rsp]["r2"][i_r2thresh,i_anc,i_scl] = (isnothing(first_inceedance) ? Nleadtime : max(1, first_inceedance-1))
                    end
                    for mc = ("ent",)
                        # SUBJECT TO R^2 > some threshold
                        first_inceedance = let
                            r2 = mixcrits[dst][rsp]["r2"][:,i_anc,i_scl]
                            (minimum(r2) < r2threshes[1]) ? (findfirst(r2 .< r2threshes[1])) : Nleadtime
                        end
                        iltmixs[dst][rsp][mc][1,i_anc,i_scl] = argmax(mixcrits[dst][rsp][mc][1:max(1,first_inceedance-1),i_anc,i_scl])
                    end
                end
                # Now average together the PDFs and CCDFs based on the criteria from above 
                for mc = ("ent",) #keys(mixobjs)
                    for i_anc = 1:Nanc
                        for (i_mcobj,mcobj) in enumerate(mixobjs[mc])
                            ilt = iltmixs[dst][rsp][mc][i_mcobj,i_anc,i_scl] 
                            for i_boot = 1:Nboot+1
                                ccdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl] .+= ccdfs[dst][rsp][:,ilt,i_anc,i_scl] .* anc_weights[i_anc,i_boot]
                                pdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl] .+= pdfs[dst][rsp][:,ilt,i_anc,i_scl] .* anc_weights[i_anc,i_boot]
                                if !all(isfinite.(pdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl]))
                                    println("non-finite PDF for i_boot=$(i_boot)")
                                    display(pdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl])
                                    error()
                                end
                            end
                        end
                    end
                end
                for mc = ("ent",) #keys(mixobjs)
                    for (i_mcobj,mcobj) in enumerate(mixobjs[mc])
                        for i_boot = 1:Nboot+1
                            for fdivname = fdivnames
                                # TODO get the right baseline for subasymptotic POt
                                fdivs[dst][rsp][mc][fdivname][i_boot,i_mcobj,i_scl] = QG2L.fdiv_fun(pdfmixs[dst][rsp][mc][:,i_boot,i_mcobj,i_scl], pdf_valid_agglon, levels, fdivname)
                            end
                        end
                    end
                end
            end
        end
    end

    JLD2.jldopen(joinpath(resultdir,"ccdfs_regressed.jld2"),"w") do f
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
        f["mixcrits"] = mixcrits
        f["iltmixs"] = iltmixs
        f["ccdfmixs"] = ccdfmixs
        f["pdfmixs"] = pdfmixs 
    end

end
