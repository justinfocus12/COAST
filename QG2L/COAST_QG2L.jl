include("./QG2L.jl")
include("./EnsembleMod.jl")

import .QG2L as QG2L 
import .EnsembleMod as EM

import Printf
using Printf: @sprintf
import Random
import Serialization
import StatsBase as SB
import LinearAlgebra as LA
import Graphs
import Distributions as Dists
import KernelDensity
import JLD2
import QuasiMonteCarlo as QMC
import Extremes as Ext
import Infiltrator as IFT

using CairoMakie

# Other possible acronyms: OATH (optimize a(t?) time horizon); FLOAT (Fixed-Lead Optimization of Antecedent Trajectories); COAST ((Fixed-Lead time)/(Finding Leverage/Likely)/(Forcing Lightly)/(Flicking Likely Extreme Event Trajectory Sampling), EPIC (Early Perturbation of Initial Conditions) or of course iTEAMS; OPAQUE (Optimal Perturbations Ahead of Quasigeostrophic Extremes), PASTE (Perturbing Ahead (of Sudden)/(to Sample) Transient Extremes) 


include("./COAST_basics.jl")
include("./COAST_plotting.jl")
include("./COAST_mixdist.jl")
include("./metaCOAST.jl")




function COAST_procedure(ensdir_dns::String, expt_supdir::String; i_expt=nothing, overwrite_expt_setup=false, overwrite_ensemble=false, old_path_part::String, new_path_part::String)
    todo = Dict(
                "upgrade_ensemble" =>                               0,
                "update_paths" =>                                   0,
                "compute_dns_objective" =>                          1,
                "plot_dns_objective_stats" =>                       1,
                "fit_dns_pot" =>                                    0, 
                "anchor" =>                                         1,
                "sail" =>                                           1, 
                "remove_pngs" =>                                    0,
                "regress_lead_dependent_risk_polynomial" =>         1, 
                "plot_objective" =>                                 1, 
                "mix_COAST_distributions_polynomial" =>             1,
                "plot_COAST_mixture" =>                             1,
                "mixture_COAST_phase_diagram" =>                    0,
                # vestigial or hibernating
                "plot_contour_divergence" =>                        0,
                "plot_dispersion_metrics" =>                        0,
                "quantify_dispersion" =>                            0,
                "plot_risk_regression_polynomial" =>                0,
               )

    php,sdm = QG2L.expt_config()
    cop,pertop = QG2L.expt_setup(php, sdm)
    @show pertop.pert_dim
    phpstr = QG2L.strrep_PhysicalParams(php)
    sdmstr = QG2L.strrep_SpaceDomain(sdm)
    pertopstr = QG2L.strrep_PerturbationOperator(pertop, sdm)
    target_yPerL,target_r = expt_config_COAST(i_expt=i_expt)

    cfg = ConfigCOAST(
                       sdm.tu
                       ; 
                       target_rxPerL=target_r, 
                       target_yPerL=target_yPerL, 
                       target_ryPerL=target_r
                      )
    obj_label,short_obj_label = label_objective(cfg)
    time_spinup_dns_ph = 500.0
    cfgstr = strrep_ConfigCOAST(cfg)
    (
     leadtimes,r2threshes,dsts,rsps,mixobjs,
     mixcrit_labels,mixobj_labels,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,i_thresh_cquantile
    ) = expt_config_COAST_analysis(cfg,pertop)
    thresh_cquantile = ccdf_levels[i_thresh_cquantile]
    threshstr = @sprintf("thrt%d", round(Int, 1/thresh_cquantile))
    exptdir_COAST = joinpath(expt_supdir,"COAST_$(cfgstr)_$(pertopstr)_$(threshstr)")
    @show xstride_valid_dns
    r2thresh = r2threshes[1]
    # Parameters related to the target
    obj_fun_COAST_from_file,obj_fun_COAST_from_histories = obj_fun_COAST_registrar(cfg.target_field, sdm, cop, cfg.target_xPerL, cfg.target_rxPerL, cfg.target_yPerL, cfg.target_ryPerL)
    ensdir_COAST = joinpath(exptdir_COAST,"ensemble_data")
    ensfile_COAST = joinpath(ensdir_COAST,"ens.jld2")
    coastfile_COAST = joinpath(ensdir_COAST,"coast.jld2") 
    rngfile_COAST = joinpath(ensdir_COAST,"rng.bin")
    resultdir = joinpath(exptdir_COAST,"results")
    figdir = joinpath(exptdir_COAST,"figures")
    
    mkpath(exptdir_COAST)
    mkpath(ensdir_COAST)
    # TODO wrap the below in a loop 
    init_cond_dir = joinpath(ensdir_COAST,"init_cond")
    mkpath(init_cond_dir)
    mkpath(resultdir)
    mkpath(figdir)

    if todo["remove_pngs"] == 1
        for filename = readdir(figdir, join=true)
            if endswith(filename,"png") && (!startswith(filename,"GPD"))
                rm(filename)
            end
        end
    end

    QG2L.plot_PerturbationOperator(pertop, sdm, figdir)

    if (!isfile(ensfile_COAST)) || overwrite_ensemble
        println("Starting afresh")
        ens = EM.Ensemble()
        rng = Random.MersenneTwister(3718)
        coast = COASTState(pertop.pert_dim)
        EM.save_Ensemble(ens, ensfile_COAST)
        save_COASTState(coast, coastfile_COAST)
        open(rngfile_COAST,"w") do f
            Serialization.serialize(f, rng)
        end
    end

    ensfile_dns = joinpath(ensdir_dns, "ens.jld2")

    if todo["upgrade_ensemble"] == 1
        #upgrade_Ensemble!(ensfile_dns, ensfile_dns)
        upgrade_Ensemble!(ensfile_COAST, ensfile_COAST)
    end

    ens_dns = EM.load_Ensemble(ensfile_dns)
    @show EM.get_Nmem(ens_dns)
    @show ens_dns.trajs[end].tfin
    ens = EM.load_Ensemble(ensfile_COAST)
    coast = load_COASTState(coastfile_COAST)
    if (1==todo["update_paths"]) && (!isnothing(old_path_part)) && (!isnothing(new_path_part))
        #adjust_paths!(ens_dns, old_path_part, new_path_part)
        #EM.save_Ensemble(ens_dns, ensfile_dns)
        adjust_paths!(ens, old_path_part, new_path_part)
        EM.save_Ensemble(ens, ensfile_COAST)
        adjust_paths!(coast, old_path_part, new_path_part)
        save_COASTState(coast, coastfile_COAST)
    end
    adjust_scores!(coast, ens, cfg, sdm)
    @show coast.pert_seq_qmc[:,1:6]
    rng = Serialization.deserialize(rngfile_COAST)


    if 1 == todo["compute_dns_objective"]
        Nmem = EM.get_Nmem(ens_dns)
        # Collect a whole family of parameterized functions 
        Nxshifts = div(sdm.Nx, xstride_valid_dns)
        xshifts = collect(range(0,Nxshifts-1,step=1).*xstride_valid_dns)
        obj_fun_COAST_xshifts = [obj_fun_COAST_registrar(cfg.target_field, sdm, cop, cfg.target_xPerL+xshift/sdm.Nx, cfg.target_rxPerL, cfg.target_yPerL, cfg.target_ryPerL)[1] for xshift=xshifts]

        hist_filenames = [ens_dns.trajs[mem].history for mem=1:Nmem]
        tfins = [ens_dns.trajs[mem].tfin for mem=1:Nmem]
        @show tfins
        tinitreq = round(Int64, time_spinup_dns_ph/sdm.tu)
        tfinreq = tinitreq + round(Int64, time_ancgen_dns_ph/sdm.tu)
        (
         tgrid_ancgen,Roft_ancgen_seplon,
         Rccdf_ancgen_seplon,Rccdf_ancgen_agglon
        ) = QG2L.compute_local_objective_and_stats_zonsym(hist_filenames, tfins, tinitreq, tfinreq, obj_fun_COAST_xshifts, ccdf_levels)
        tinitreq = round(Int64, (time_spinup_dns_ph+time_ancgen_dns_ph_max)/sdm.tu) 
        tfinreq = tinitreq + round(Int64, time_valid_dns_ph/sdm.tu)
        (
         tgrid_valid,Roft_valid_seplon,
         Rccdf_valid_seplon,Rccdf_valid_agglon
        ) = QG2L.compute_local_objective_and_stats_zonsym(hist_filenames, tfins, tinitreq, tfinreq, obj_fun_COAST_xshifts, ccdf_levels)
        # after choosing the threshold, 
        

        JLD2.jldopen(joinpath(resultdir,"objective_dns_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).jld2"), "w") do f
            f["tgrid_ancgen"] = tgrid_ancgen
            f["Roft_ancgen_seplon"] = Roft_ancgen_seplon
            f["Rccdf_ancgen_seplon"] = Rccdf_ancgen_seplon
            f["Rccdf_ancgen_agglon"] = Rccdf_ancgen_agglon
            f["tgrid_valid"] = tgrid_valid
            f["Roft_valid_seplon"] = Roft_valid_seplon
            f["Rccdf_valid_seplon"] = Rccdf_valid_seplon
            f["Rccdf_valid_agglon"] = Rccdf_valid_agglon
        end
    else
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
    end
    if 1 == todo["plot_dns_objective_stats"]
        # TODO do some plotting and quantify the difference between ancgen and valid (train and test?) distributions 
        fig = Figure()
        lout = fig[1,1] = GridLayout()
        axhist = Axis(lout[1,1], xlabel="Intensity", ylabel="PDF", yscale=log10)
        axquant = Axis(lout[2,1], xlabel="Intensity", ylabel="CCDF", yscale=log10)
        bin_edges = collect(range(0,1,200))
        bin_centers = 0.5 .* (bin_edges[2:end] .+ bin_edges[1:end-1])
        Nxshifts = div(sdm.Nx, xstride_valid_dns)
        (pdfs_ancgen,pdfs_valid) = (zeros(Float64, (length(bin_centers),Nxshifts)) for _=1:2)
        for i_lon = 1:Nxshifts
            hg_ancgen = SB.normalize(SB.fit(SB.Histogram, Roft_ancgen_seplon[:,i_lon], bin_edges); mode=:pdf)
            pdfs_ancgen[:,i_lon] .= hg_ancgen.weights
            hg_valid = SB.normalize(SB.fit(SB.Histogram, Roft_valid_seplon[:,i_lon], bin_edges); mode=:pdf)
            pdfs_valid[:,i_lon] .= hg_valid.weights
        end

        zero2nan(p) = replace(p, 0=>NaN)

        lines!(axhist, bin_centers, zero2nan(SB.mean(pdfs_ancgen; dims=2)[:,1]); color=:cyan, linewidth=2)
        lines!(axhist, bin_centers, zero2nan(SB.mean(pdfs_valid; dims=2)[:,1]); color=:black, linewidth=2)
        lines!(axquant, zero2nan(SB.mean(Rccdf_ancgen_seplon; dims=2)[:,1]), ccdf_levels; color=:cyan, linewidth=2)
        lines!(axquant, zero2nan(SB.mean(Rccdf_valid_seplon; dims=2)[:,1]), ccdf_levels; color=:black, linewidth=2)
        lines!(axquant, zero2nan(Rccdf_valid_agglon), ccdf_levels; color=:gray, alpha=0.5, linewidth=3)
        for q = [0.05,0.95]
            qancgen = zero2nan(QG2L.quantile_sliced(pdfs_ancgen, q, 2)[:,1])
            qvalid = zero2nan(QG2L.quantile_sliced(pdfs_valid, q, 2)[:,1])
            lines!(axhist, bin_centers, zero2nan(QG2L.quantile_sliced(pdfs_valid, q, 2)[:,1]); color=:black, linestyle=(:dot,:dense))
            lines!(axhist, bin_centers, qancgen; color=:cyan, linestyle=(:dot,:dense))
            qancgen = zero2nan(QG2L.quantile_sliced(Rccdf_ancgen_seplon, q, 2)[:,1])
            qvalid = zero2nan(QG2L.quantile_sliced(Rccdf_valid_seplon, q, 2)[:,1])
            lines!(axquant, qvalid, ccdf_levels; color=:black, linestyle=(:dot,:dense))
            lines!(axquant, qancgen, ccdf_levels; color=:cyan, linestyle=(:dot,:dense))
            println("For q = $(q), qancgen = ")
            display(qancgen')
        end

        linkxaxes!(axhist, axquant)
        save(joinpath(figdir,"histograms_ancgen_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).png"),fig)
        
        
        @show SB.mean(Roft_ancgen_seplon),SB.std(Roft_ancgen_seplon),length(Roft_ancgen_seplon)
        @show SB.mean(Roft_valid_seplon),SB.std(Roft_valid_seplon),length(Roft_valid_seplon)


        @show figdir
    end


    #
    # ------------ Establish single threshold to be used by all downstream tasks ----------------
    if todo["fit_dns_pot"] == 1
        # Measure the objective function in DNS, augmenting with zonal symmetry, to be used downstream. This might replicate some effort from DNS. 
        # ------------------- DNS: fit a GP to threshold exceedances -------------
        duration_dns_pot = 40000.0
        duration_dns = round(Int, duration_dns_pot/sdm.tu)
        ens_dns = EM.load_Ensemble(ensfile_dns)
        Nmem_dns = EM.get_Nmem(ens_dns)
        tphinits_dns = [ens_dns.trajs[mem].tphinit for mem=1:Nmem_dns]
        @show Nmem_dns, tphinits_dns
        tfins = [ens_dns.trajs[mem].tfin for mem=1:Nmem_dns]
        memfirst = findfirst(tphinits_dns .> time_spinup_dns_ph)
        tfin_requested = round(Int, (time_spinup_dns_ph + duration_dns)/sdm.tu)
        memlast = Nmem_dns
        if tfins[memlast] > tfin_requested
            memlast = findfirst(tfins .>= tfin_requested)
        end
        @show memfirst,memlast
        mems_dns = collect(range(memfirst, memlast, step=1))
        tgrid_dns = collect((ens_dns.trajs[mems_dns[1]-1].tfin+1):1:ens_dns.trajs[mems_dns[end]].tfin)
        hist_filenames = [ens_dns.trajs[mem].history for mem=mems_dns]
        xstride = div(sdm.Nx, 8)
        Nxshifts = div(sdm.Nx, xstride)
        GPD_dns_results = (
             QG2L.compute_local_GPD_params_zonsym_multiple_fits(hist_filenames, obj_fun_COAST_xshiftable, cfg.peak_prebuffer_time, cfg.follow_time, cfg.lead_time_max, Nxshifts, xstride, figdir, obj_label)
            )
        JLD2.jldopen(joinpath(resultdir,"GPD_dns.jld2"), "w") do f
            keys = split(
                         "ccdf_at_thresh,thresh,scale,shape,"*
                         "peak_vals,peak_tidx,upcross_tidx,downcross_tidx,"*
                         "levels,ccdfs_emp,ccdf_GPD_xall"
                         ,","
                         )
            for (i_key,key) in enumerate(keys)
                f[key] = GPD_dns_results[i_key]
            end
        end
    end


    if false
        thresh,scale,shape,levels_dns_emp,ccdfs_dns_emp = JLD2.jldopen(joinpath(resultdir,"GPD_dns.jld2"),"r") do f
            return f["thresh"],f["scale"],f["shape"],f["levels"],f["ccdfs_emp"]
        end
        pdfs_dns_emp = -diff(ccdfs_dns_emp; dims=1) ./ diff(levels_dns_emp)
        levels_mid_dns_emp = 0.5*(levels_dns_emp[1:end-1] .+ levels_dns_emp[2:end])
    end

    # --------------- THE KING OF THRESHES -----------
    thresh = Rccdf_valid_agglon[i_thresh_cquantile] 
    # ------------------------------------------------


    # -------------------------------------------------------------------------------------------

    if todo["anchor"] == 1
        new_peak_frontier_time = round(Int, time_spinup_dns_ph/sdm.tu)
        if length(coast.ancestor_init_conds) > 0
            new_peak_frontier_time = coast.peak_times_upper_bounds[end]
        end
        while length(coast.ancestor_init_conds) < cfg.num_init_conds_max
            i_anc = length(coast.ancestor_init_conds) + 1
            println("Preparing initial condition for i_anc $(i_anc)")
            init_cond_file = joinpath(init_cond_dir,"init_cond_anc$(i_anc).jld2")
            prehistory_file = joinpath(init_cond_dir,"init_cond_prehistory_anc$(i_anc).jld2")
            init_prep = prepare_init_cond_from_dns(ens_dns, obj_fun_COAST_from_file, cfg, sdm, cop, php, pertop, init_cond_file, prehistory_file, new_peak_frontier_time, thresh; num_peaks_to_skip=0)
            if isnothing(init_prep)
                println("WARNING ran out of peaks to boost")
                cfg.num_init_conds_max = i_anc-1
                continue
            end
            new_dns_peak,new_dns_peak_timing,t_upcross,t_downcross,dns_Roft = init_prep
            push!(coast.dns_peaks, new_dns_peak)
            push!(coast.dns_peak_times, new_dns_peak_timing)
            push!(coast.peak_times_lower_bounds, t_upcross)
            push!(coast.peak_times_upper_bounds, t_downcross)
            push!(coast.dns_anc_Roft, dns_Roft)
            push!(coast.ancestor_init_conds, init_cond_file)
            push!(coast.ancestor_init_cond_prehistories, prehistory_file)
            save_COASTState(coast, coastfile_COAST)
            new_peak_frontier_time = t_downcross
        end
        @show coast.dns_peaks
        @show coast.dns_peak_times

        # Reset termination, in case the capacity has been expanded
        if length(coast.ancestors) < cfg.num_init_conds_max
            coast.terminate = false 
        else
            coast.terminate = true
            for i_anc = 1:cfg.num_init_conds_max
                anc = coast.ancestors[i_anc]
                num_desc = length(Graphs.outneighbors(ens.famtree, anc))
                @show anc, num_desc
                if num_desc < cfg.num_perts_max
                    coast.terminate = false
                end
            end
        end
    end


    if todo["sail"] == 1
        # pre-allocate  
        # TODO if we extend this to multi-threading, must allocate separately per thread 
        tgrid = zeros(Int64, cfg.lead_time_max + cfg.follow_time)
        sf_the = QG2L.FlowField(sdm.Nx, sdm.Ny)
        sf_hist = QG2L.FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
        sf_hist_the = QG2L.FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
        conc_hist = zeros(Float64, (sdm.Nx, sdm.Ny, 2, length(tgrid)))
        while !coast.terminate
            num_members_max = cfg.num_init_conds_max * (1 + cfg.num_perts_max)
            println("About to extend Ensemble to $(EM.get_Nmem(ens)+1) members, out of $(num_members_max)")
            set_sail!(ens, coast, rng, obj_fun_COAST_from_file, obj_fun_COAST_from_histories, ensdir_COAST, cfg, cop, pertop, sdm, php, )
            Nmem = EM.get_Nmem(ens)
            ancs = Graphs.inneighbors(ens.famtree, Nmem)
            if length(ancs) > 0
                anc = ancs[1]
                i_anc = findfirst(coast.ancestors .== anc)
            end
            # Save the state for next round
            EM.save_Ensemble(ens, ensfile_COAST)
            open(rngfile_COAST,"w") do f
                Serialization.serialize(f, rng)
            end
            save_COASTState(coast, coastfile_COAST)
        end
    end
    adjust_scores!(coast, ens, cfg, sdm)
    Nmem = EM.get_Nmem(ens)
    Nanc = length(coast.ancestors)
    leadtimes = collect(range(cfg.lead_time_min,cfg.lead_time_max; step=cfg.lead_time_inc))
    Nleadtime = length(leadtimes)
    min_score,max_score = extrema(vcat(coast.anc_Rmax, (coast.desc_Rmax[i_anc] for i_anc=1:Nanc)...))
    # --------------------- Decide on a subset of ancestors to display ---------------
    #
    ancorder = sortperm(coast.anc_Rmax; rev=true)
    idx_anc_strat = sort(unique(ancorder[range(1,min(12,Nanc); step=1)]))
    idx_anc_strat = intersect(1:Nanc, idx_anc_strat)
    # ---------------------------------------------------------------------------------
    #
    if 1 == todo["regress_lead_dependent_risk_polynomial"]
        (
         coefs_linear,residmse_linear,rsquared_linear,resid_range_linear,
         coefs_quadratic,residmse_quadratic,rsquared_quadratic,resid_range_quadratic,
         hessian_eigvals,hessian_eigvecs
        ) = regress_lead_dependent_risk_linear_quadratic(coast, ens, cfg, sdm, pertop)
        JLD2.jldopen(joinpath(resultdir,"regression_coefs.jld2"), "w") do f
            f["coefs_linear"] = coefs_linear
            f["residmse_linear"] = residmse_linear
            f["rsquared_linear"] = rsquared_linear
            f["resid_range_linear"] = resid_range_linear
            f["coefs_quadratic"] = coefs_quadratic
            f["residmse_quadratic"] = residmse_quadratic
            f["rsquared_quadratic"] = rsquared_quadratic
            f["resid_range_quadratic"] = resid_range_linear
            f["hessian_eigvals"] = hessian_eigvals
            f["hessian_eigvecs"] = hessian_eigvecs
            return
        end
    end
        

    if 1 == todo["plot_objective"]
        rxystr = @sprintf("%.3f",cfg.target_ryPerL*sdm.Ly)
        ytgtstr = @sprintf("%.2f",cfg.target_yPerL*sdm.Ly)
        todosub = Dict(
                       "plot_spaghetti" =>              1,
                       "plot_response" =>               1,
                      )
        @show idx_anc_strat

        regcoefs_filename = joinpath(resultdir,"regression_coefs.jld2")
        coefs_linear,residmse_linear,rsquared_linear,coefs_quadratic,residmse_quadratic,rsquared_quadratic,hessian_eigvals,hessian_eigvecs = JLD2.jldopen(regcoefs_filename, "r") do f
            return (
                    f["coefs_linear"],
                    f["residmse_linear"],
                    f["rsquared_linear"],
                    f["coefs_quadratic"],
                    f["residmse_quadratic"],
                    f["rsquared_quadratic"],
                    f["hessian_eigvals"],
                    f["hessian_eigvecs"],
                   )
        end

        Threads.@threads for i_anc = idx_anc_strat
            if 1 == todosub["plot_spaghetti"]
                plot_objective_spaghetti(cfg, sdm, cop, pertop, ens, coast, i_anc, thresh, figdir)
            end


            # ------------- Plot 3: perturbations in the complex plane, along with level sets of the linear and quadratic models. Also show R^2 ------------
            if 1 == todosub["plot_response"]
                plot_objective_response_linquad(
                    cfg, sdm, cop, pertop, ens, coast, i_anc, 
                    coefs_linear, residmse_linear, rsquared_linear,
                    coefs_quadratic, residmse_quadratic, rsquared_quadratic,
                    figdir
                   )
            end
        end
    end




    if todo["mix_COAST_distributions_polynomial"] == 1
        println("About to mix COAST distributions")
        mix_COAST_distributions_polynomial(cfg, cop, pertop, coast, resultdir)
        println("Finished mixing")
    end
    if 1 == todo["plot_COAST_mixture"]
        println("Aabout to plot COAST mixtures")
        todosub = Dict(
                       "gains_topt" =>              0,
                       "rainbow_pdfs" =>            0,
                       "mixed_pdfs" =>              1,
                      )

        ytgtstr = @sprintf("%.2f", cfg.target_yPerL*sdm.Ly)
        rxystr = @sprintf("%.3f", cfg.target_ryPerL*sdm.Ly)
        # TODO incorporate bootstraps into this 
        (
         levels,levels_mid,
         dsts,rsps,mixobjs,distn_scales,
         ccdfs,pdfs,
         fdivs,
         mixcrits,iltmixs,
         ccdfmixs,pdfmixs,
        ) = (JLD2.jldopen(joinpath(resultdir,"ccdfs_regressed.jld2"),"r") do f
                 # coordinates for parameters of distributions 
                 return (
                    f["levels"],# levels
                    f["levels_mid"],# levels_mid
                    f["dstns"],# dsts
                    f["rsps"],# rsps
                    f["mixobjs"],# mixobjs
                    f["distn_scales"],# distn_scales
                    # output distributions 
                    f["ccdfs"],# ccdfs
                    f["pdfs"],# pdfs
                    f["fdivs"],# fdivs
                    f["mixcrits"],# mixcrits
                    f["iltmixs"],# iltmixs
                    f["ccdfmixs"],# ccdfmixs
                    f["pdfmixs"],# pdfmixs 
                   )
             end
            )
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
        i_thresh_level = argmin(abs.(ccdf_levels .- thresh_cquantile))
        thresh = Rccdf_valid_agglon[i_thresh_level] 
        pdf_valid_agglon = - diff(ccdf_levels) ./ diff(Rccdf_valid_agglon)
        pdf_valid_seplon = - diff(ccdf_levels) ./ diff(Rccdf_valid_seplon; dims=1)
        #IFT.@infiltrate true

        # ------------- Plots -----------------------
        # For each distribution type, plot the PDFs along the top and a row for each mixing criterion
        epsilon = 1e-6
        ylims = (thresh, max(max_score, levels[end]))
        clipccdf(x) = (x <= epsilon ? NaN : x)
        clippdf(x) = (x <= epsilon/(levels[2]-levels[1]) ? NaN : x)


        for dst = ["b"]
            Nscales = length(distn_scales[dst])
            for rsp = ["2"]
                if ("g" == dst) && (rsp in ["1+u","2","2+u"])
                    continue
                end

                # --------------- Dependence between (lead time chosen by criteria) and (score) ---------
                if 1 == todosub["gains_topt"]
                    #Threads.@threads for i_scl = 1:Nscales
                    for i_scl = 1:Nscales
                        scalestr = @sprintf("%.3f", distn_scales[dst][i_scl])
                        i_mcobj = 1
                        for (i_mc,mc) in enumerate(["ent"])
                            fig = Figure(size=(600,400))
                            lout = fig[1:2,1] = GridLayout()
                            ax1 = Axis(lout[1,1], xlabel=L"$-$AST", ylabel=L"$$%$(mixcrit_labels[mc])", xlabelvisible=false, xticklabelsvisible=false, title=L"$$Target lat. %$(ytgtstr), Box size %$(rxystr), Scale %$(scalestr)")
                            ax2 = Axis(lout[2,1], xlabel=L"$-\mathrm{AST}$", ylabel=L"$$Severity", title="", titlevisible=false)
                            #@show iltmixs[dst][rsp][mc][i_mcobj,:,i_scl]
                            for i_anc = 1:Nanc
                                i_leadtime = iltmixs[dst][rsp][mc][i_mcobj,i_anc,i_scl]
                                if i_leadtime == 0
                                    @show dst,rsp,mc,i_mcobj,i_anc,i_scl
                                    error()
                                end
                                leadtime = leadtimes[i_leadtime]
                                colargs = Dict(:colormap=>:managua, :color=>coast.anc_Rmax[i_anc], :colorrange=>extrema(coast.anc_Rmax))
                                lines!(ax1, -leadtimes.*sdm.tu, mixcrits[dst][rsp][mc][:,i_anc,i_scl]; colargs..., alpha=0.25, linewidth=2.0)
                                scatter!(ax1, -leadtime*sdm.tu, mixcrits[dst][rsp][mc][i_leadtime,i_anc,i_scl]; colargs...)
                                scatter!(ax2, -leadtime*sdm.tu, coast.anc_Rmax[i_anc]; color=:dodgerblue, markersize=10)
                                idx_desc = desc_by_leadtime(coast, i_anc, leadtime, sdm)
                                scatter!(ax2, -leadtime*sdm.tu.*ones(Float64, length(idx_desc)), coast.desc_Rmax[i_anc][idx_desc]; color=:firebrick, markersize=5)
                            end
                            let
                                mcdrm = mixcrits[dst][rsp][mc][:,:,i_scl]
                                m = SB.mean(mcdrm; dims=2)
                                m0 = sum(mcdrm .* (mcdrm .< m); dims=2) ./ replace(sum(mcdrm .< m; dims=2), 0=>1)
                                m1 = sum(mcdrm .* (mcdrm .> m); dims=2) ./ replace(sum(mcdrm .> m; dims=2), 0=>1)
                                lines!(ax1, -leadtimes.*sdm.tu, vec(m); color=:black)
                                lines!(ax1, -leadtimes.*sdm.tu, vec(m0); color=:black, linestyle=(:dash,:dense))
                                lines!(ax1, -leadtimes.*sdm.tu, vec(m1); color=:black, linestyle=(:dash,:dense))
                            end
                            rowgap!(lout, 1, 0.0)
                            for ax = (ax1,ax2)
                                xlims!(ax, -(leadtimes[end]+0.5*cfg.lead_time_inc)*sdm.tu, -(leadtimes[1]-0.5*cfg.lead_time_inc)*sdm.tu)
                            end
                            save(joinpath(figdir,"gains_topt_$(dst)_$(rsp)_$(mc)_$(i_scl).png"), fig)
                        end
                    end
                end



                # ---------------- Single-ancestor plots ---------------
                if 1 == todosub["rainbow_pdfs"]
                    mixcrits2plot = ["r2","ent"]
                    mixcrits_ylabels = [L"$R_2^2$",L"$$CondEnt"]
                    Nleadtimes2plot = Nleadtime
                    #Threads.@threads for i_anc = idx_anc_strat
                    for i_anc = idx_anc_strat
                        t0str = @sprintf("%d", coast.anc_tRmax[i_anc])
                        fig = Figure(size=(100*Nleadtime,100*(3+length(mixcrits2plot))))
                        lout = fig[1,1] = GridLayout()
                        lout_pdfs = lout[1,1] = GridLayout()
                        lout_mixcrits = lout[2,1] = GridLayout()
                        i_col = 0
                        for i_leadtime = round.(Int, range(Nleadtime, 1; length=Nleadtimes2plot))
                            leadtime = leadtimes[i_leadtime]
                            i_col += 1
                            ltstr = @sprintf("%d",-leadtime*sdm.tu)
                            if i_col == 1
                                ltstr = L"$-\mathrm{AST} = $%$(ltstr)"
                            end
                            lblargs = Dict(:ylabelvisible=>(i_col==1), :yticklabelsvisible=>(i_col==1), :xlabelvisible=>false, :xticklabelsvisible=>true, :ylabelsize=>20, :xlabelsize=>20, :title=>L"%$(ltstr)", :xgridvisible=>false, :ygridvisible=>false, :titlealign=>:right)
                            ax = Axis(lout_pdfs[1,i_col]; xlabel=L"$$PDF", ylabel=L"$$Intensity", lblargs...)
                            for (i_scl,scl) in enumerate(distn_scales[dst])
                                lines!(ax, pdfs[dst][rsp][:,i_leadtime, i_anc, i_scl], levels_mid; color=i_scl,colorrange=(0,length(distn_scales[dst])), colormap=:managua)
                            end
                            lines!(ax, pdf_valid_agglon[i_thresh_level:end], levels_mid[i_thresh_level:end]; color=:black, linestyle=(:dash,:dense), linewidth=1.5)
                            hlines!(ax, coast.anc_Rmax[i_anc]; color=:black, linewidth=1.0)
                            idx_desc = desc_by_leadtime(coast, i_anc, leadtime, sdm)
                            scatter!(ax, maximum(pdfs[dst][rsp][:,i_leadtime, i_anc, :]).*ones(length(idx_desc)), coast.desc_Rmax[i_anc][idx_desc]; color=:firebrick, markersize=10)
                            #IFT.@infiltrate true
                            ylims!(ax, ylims...)
                        end
                        for (i_mc,mc) in enumerate(mixcrits2plot)
                            ax = Axis(lout_mixcrits[i_mc,1]; ylabel=mixcrits_ylabels[i_mc], xlabel=L"$-\mathrm{AST}$ $(t^*=%$(t0str))$", xlabelvisible=(i_mc==length(mixcrits2plot)), xticklabelsvisible=(i_mc==length(mixcrits2plot)), xlabelsize=20, ylabelsize=20, xgridvisible=false, ygridvisible=false)
                            for (i_scl,scl) in enumerate(distn_scales[dst])
                                colargs = (mc == "r2" ? Dict(:color=>:black) : Dict(:color=>i_scl,:colormap=>:managua,:colorrange=>(0,length(distn_scales[dst]))))
                                scatterlines!(ax, -leadtimes.*sdm.tu, mixcrits[dst][rsp][mc][:,i_anc,i_scl]; colargs...)
                                if mc == "ent"
                                    vlines!(ax, -leadtimes[iltmixs[dst][rsp][mc][1,i_anc,i_scl]]*sdm.tu; colargs...)
                                elseif mc == "r2"
                                    hlines!(ax, r2thresh, color=:gray)
                                end
                            end
                            xlims!(ax, -(cfg.lead_time_max+0.5*cfg.lead_time_inc)*sdm.tu, -(cfg.lead_time_min-0.5*cfg.lead_time_inc)*sdm.tu)
                            if mc == "r2"
                                ylims!(ax, 0, 1)
                            end
                        end
                        for i_row = 1:nrows(lout_mixcrits)-1
                            rowgap!(lout_mixcrits, i_row, 20.0)
                        end
                        for i_col = 1:ncols(lout_pdfs)-1
                            colgap!(lout_pdfs, i_col, 0.0)
                        end
                        Label(lout_pdfs[1,:,Top()], L"$$Target lat. %$(ytgtstr), Box size %$(rxystr)", padding=(5.0,5.0,20.0,5.0), valign=:bottom, fontsize=20)
                        rowsize!(lout, 1, Relative(2/(2+length(mixcrits2plot))))
                        save(joinpath(figdir,"reg2dist_$(dst)_$(rsp)_anc$(i_anc).png"), fig)
                    end
                end
                # ----------------- Mixed-ancestor plots -----------------
                if 1 == todosub["mixed_pdfs"]



                    for i_scl = 1:length(distn_scales[dst])
                        scalestr = @sprintf("%.3f", distn_scales[dst][i_scl])
                        pdf_ancgen_seplon = -diff(ccdf_levels) ./ diff(Rccdf_ancgen_seplon, dims=1)
                        pdf_ancgen_pt = pdf_ancgen_seplon[:,1]
                        pdf_ancgen_lo,pdf_ancgen_hi = (QG2L.quantile_sliced(pdf_ancgen_seplon, q, 2) for q=(0.05,0.95))

                        i_boot = 1
                        ilt_tv_sync = argmin(fdivs[dst][rsp]["lt"]["tv"][i_boot,:,i_scl])
                        tv_sync = fdivs[dst][rsp]["lt"]["tv"][i_boot,ilt_tv_sync,i_scl]
                        tv_cond = fdivs[dst][rsp]["ent"]["tv"][i_boot,1,i_scl]
                        tpstr = @sprintf("%.1f",leadtimes[ilt_tv_sync]*sdm.tu)
                        tvsyncstr = @sprintf("%.3f",tv_sync)
                        tvcondstr = @sprintf("%.3f",tv_cond)
                        boot_midlohi_pdf(p) = begin
                            pmid = clippdf.(p[:,1])
                            plo = clippdf.(mapslices(pp->SB.quantile(pp,0.05), p; dims=2)[:,1])
                            phi = clippdf.(mapslices(pp->SB.quantile(pp,0.95), p; dims=2)[:,1])
                            return (pmid,plo,phi)
                        end
                        boot_midlohi_ccdf(p) = begin
                            pmid = clipccdf.(p[:,1])
                            plo = clipccdf.(mapslices(pp->SB.quantile(pp,0.05), p; dims=2)[:,1])
                            phi = clipccdf.(mapslices(pp->SB.quantile(pp,0.95), p; dims=2)[:,1])
                            return (pmid,plo,phi)
                        end
                            
                        pdf_cond_mid,pdf_cond_lo,pdf_cond_hi = boot_midlohi_pdf(pdfmixs[dst][rsp]["ent"][:,:,1,i_scl])
                        pdf_sync_mid,pdf_sync_lo,pdf_sync_hi = boot_midlohi_pdf(pdfmixs[dst][rsp]["lt"][:,:,ilt_tv_sync,i_scl])
                        ccdf_cond_mid,ccdf_cond_lo,ccdf_cond_hi = boot_midlohi_ccdf(ccdfmixs[dst][rsp]["ent"][:,:,1,i_scl])
                        ccdf_sync_mid,ccdf_sync_lo,ccdf_sync_hi = boot_midlohi_ccdf(ccdfmixs[dst][rsp]["lt"][:,:,ilt_tv_sync,i_scl])
                        # Plot CCDFS 
                        fig = Figure(size=(700,400))
                        lout = fig[1,1] = GridLayout()
                        ax = Axis(lout[1,1]; xscale=log10, xlabel="CCDF", ylabel=L"$$Severity", title=L"$$Target lat. %$(ytgtstr), Box size %$(rxystr), Scale %$(scalestr)")
                        lines!(ax, clipccdf.(ccdf_levels), levels; color=:black, linestyle=(:dash,:dense), linewidth=2, label=@sprintf("Long DNS"))
                        lines!(ax, clipccdf.(ccdf_levels), Rccdf_ancgen_seplon[:,1]; color=:chocolate3, linestyle=(:dash,:dense), linewidth=2, label=@sprintf("Short DNS"))
                        lines!(ax, ccdf_sync_mid, levels; color=:cyan, linestyle=:solid, label="AST: $(tpstr)\nTV from DNS=$(tvsyncstr)", linewidth=2)
                        #lines!(ax, ccdf_sync_lo, levels; color=:cyan, linestyle=:solid, linewidth=1)
                        #lines!(ax, ccdf_sync_hi, levels; color=:cyan, linestyle=:solid, linewidth=1)
                        lines!(ax, ccdf_cond_mid, levels; color=:firebrick, linestyle=:solid, linewidth=2, label="AST: Max. CondEnt\nTV from DNS =$(tvcondstr)")
                        #lines!(ax, ccdf_coast_lo, levels; color=:firebrick, linestyle=:solid, linewidth=1)
                        #lines!(ax, ccdf_coast_hi, levels; color=:firebrick, linestyle=:solid, linewidth=1)
                        lout[1,2] = Legend(fig, ax; framevisible=true)
                        colsize!(lout, 1, Relative(2/3))
                        save(joinpath(figdir,"ccdfmixs_$(dst)_$(rsp)_$(i_scl).png"), fig)
                        # Plot PDFs 
                        fig = Figure(size=(700,400))
                        lout = fig[1,1] = GridLayout()
                        ax = Axis(lout[1,1]; xscale=log10, xlabel="PDF", ylabel=L"$$Severity", title=L"$$Target lat. %$(ytgtstr), Box size %$(rxystr), Scale %$(scalestr)")
                        lines!(ax, clippdf.(pdf_valid_agglon), levels_mid; color=:black, linestyle=:solid)
                        lines!(ax, pdf_sync_mid, levels_mid; color=:cyan, linestyle=:solid, label="AST: $(tpstr)\nTV from DNS=$(tvsyncstr)", linewidth=2)
                        lines!(ax, clippdf.(pdf_valid_seplon[:,1]), levels_mid; color=:chocolate3, linestyle=(:dash,:dense), linewidth=2, label=@sprintf("Short DNS"))
                        #lines!(ax, pdf_sync_lo, levels_mid; color=:cyan, linestyle=:solid, linewidth=1)
                        #lines!(ax, pdf_sync_hi, levels_mid; color=:cyan, linestyle=:solid, linewidth=1)
                        lines!(ax, pdf_cond_mid, levels_mid; color=:firebrick, linestyle=:solid, linewidth=2, label="AST: Max. CondEnt\nTV from DNS =$(tvcondstr)")
                        lines!(ax, pdf_cond_lo, levels_mid; color=:firebrick, linestyle=:solid, linewidth=1)
                        lines!(ax, pdf_cond_hi, levels_mid; color=:firebrick, linestyle=:solid, linewidth=1)
                        lout[1,2] = Legend(fig, ax; framevisible=true)
                        colsize!(lout, 1, Relative(2/3))
                        save(joinpath(figdir,"pdfmixs_$(dst)_$(rsp)_$(i_scl).png"), fig)

                    end
                end
            end
        end
    end


    if todo["plot_contour_divergence"] == 1
        todo_anim = Dict(
                         "eke1" =>         ("eke1" == cfg.target_field),
                         "sf2" =>          ("sf2" == cfg.target_field),
                         "pv_1" =>         0,
                         "conc1" =>        ("conc1" == cfg.target_field),
                         "conc2" =>        ("conc2" == cfg.target_field),
                         "temp" =>         0,
                         "heatflux" =>     0,
                         "v_bt" =>         0,
                        )
        anim_labels = Dict(
                           "eke1" =>       L"$\frac{1}{2}|\nabla\Psi|^2$",
                           "conc2" =>      L"$c_2$",
                           "conc1" =>      L"$c_1$",
                          )
        todo_snap = Dict((key,value) for (key,value) in todo_anim)

        # ------------- Videos -----------
        contour_funs = Dict(
                            "eke1" => (f -> QG2L.obs_fun_eke_hist(f, sdm, cop)[:,:,1,:]),
                            "sf2" => (f -> QG2L.obs_fun_sf_hist(f, sdm, cop)[:,:,2,:]),
                            "pv_1" => (f -> QG2L.obs_fun_pv_hist(f, sdm, cop)[:,:,1,:]),
                            "conc2" => (f -> QG2L.obs_fun_conc_hist(f, sdm, cop)[:,:,2,:]),
                            "conc1" => (f -> QG2L.obs_fun_conc_hist(f, sdm, cop)[:,:,1,:]),
                            "temp" => (f -> QG2L.obs_fun_temperature_hist(f, sdm, cop)[:,:,1,:]),
                            "heatflux" => (f -> QG2L.obs_fun_heatflux_hist(f, sdm, cop)[:,:,1,:]),
                            "v_bt" => (f -> sum(QG2L.obs_fun_meridional_velocity_hist(f, sdm, cop), dims=3)[:,:,1,:]),
                           )
        Threads.@threads for i_anc = idx_anc_strat
            anc = coast.ancestors[i_anc]
            tgrid = collect(range(round(Int, ens.trajs[anc].tphinit/sdm.tu), ens.trajs[ancs].tfin; step=1))
            descendants = Graphs.outneighbors(ens.famtree, anc)
            Ndesc = length(descendants)
            if 0 == Ndesc
                continue
            end
            Nmem_cont = min(2, Ndesc+1)
            Nmem_objoft = min(12, Ndesc+1)

            idx_descs_cont = zeros(Int64, Nleadtime)
            for (i_leadtime,leadtime) in enumerate(leadtimes)
                idx_desc_ilt = desc_by_leadtime(coast, i_anc, leadtime, sdm)
                idx_desc_cont[i_leadtime] = idx_desc_ilt[argmax(coast.desc_Rmax[idx_desc_ilt])]
            end
                # TODO
                
            tRmaxs = zeros(Int64, length(mems_ordered))
            for (i_desc,desc) in enumerate(descendants)
                tRmaxs[i_desc] = coast.desc_tRmax[i_anc] # TODO
            end


            mems_ordered = vcat(descendants[sortperm(coast.desc_Rmax[i_anc])], [anc])
            @show mems_ordered
            i_desc = argmax(coast.desc_Rmax[i_anc])
            mems_cont = [descendants[i_desc], anc] 
            mems_objoft = mems_ordered[end-Nmem_objoft+1:end]

            todo_anim = Dict(
                             "eke1" =>         ("eke1" == cfg.target_field),
                             "sf2" =>          ("sf2" == cfg.target_field),
                             "pv_1" =>         0,
                             "conc1" =>        ("conc1" == cfg.target_field),
                             "conc2" =>        ("conc2" == cfg.target_field),
                             "temp" =>         0,
                             "heatflux" =>     0,
                             "v_bt" =>         0,
                            )
            anim_labels = Dict(
                               "eke1" =>       L"$\frac{1}{2}|\nabla\Psi|^2$",
                               "conc2" =>      L"$c_2$",
                               "conc1" =>      L"$c_1$",
                              )

            # ------------- Videos -----------
            contour_funs = Dict(
                                "eke1" => (f -> QG2L.obs_fun_eke_hist(f, sdm, cop)[:,:,1,:]),
                                "sf2" => (f -> QG2L.obs_fun_sf_hist(f, sdm, cop)[:,:,2,:]),
                                "pv_1" => (f -> QG2L.obs_fun_pv_hist(f, sdm, cop)[:,:,1,:]),
                                "conc2" => (f -> QG2L.obs_fun_conc_hist(f, sdm, cop)[:,:,2,:]),
                                "conc1" => (f -> QG2L.obs_fun_conc_hist(f, sdm, cop)[:,:,1,:]),
                                "temp" => (f -> QG2L.obs_fun_temperature_hist(f, sdm, cop)[:,:,1,:]),
                                "heatflux" => (f -> QG2L.obs_fun_heatflux_hist(f, sdm, cop)[:,:,1,:]),
                                "v_bt" => (f -> sum(QG2L.obs_fun_meridional_velocity_hist(f, sdm, cop), dims=3)[:,:,1,:]),
                               )
            tgrid = JLD2.jldopen(ens.trajs[mems_cont[1]].history, "r") do f; return f["tgrid"]; end
            filename2obs(fname,obsfun) = JLD2.jldopen(fname, "r") do f; return obsfun(f); end
            filename2obs_snap(fname,obsfun,t_snap) = JLD2.jldopen(fname, "r") do f
                return obsfun(f)[:,:,t_snap-tgrid[1]+1]
            end

            # For video
            sfs = [filename2obs(ens.trajs[mem].history, contour_funs["sf2"]) for mem=mems_cont]
            objs = [coast.anc_Roft[i_anc], coast.desc_Roft[i_anc][i_desc]]

            # for snapshots
            objs_snap = [filename2obs_snap(ens.trajs[descendants[i_desc]].history, contour_funs["cond1"], coast.desc_tRmax[i_anc][i_desc]) for i_desc=descs2plot]
            @assert all([size(sfs[i], 3) == length(objs[i]) for i=1:length(sfs)])
            for key in keys(contour_funs)
                if todo_anim[key] == 1
                    println("About to animate $(key)")
                    fields_cont = [filename2obs(ens.trajs[mem].history,contour_funs[key]) for mem=mems_cont]
                    # heatmap 
                    outfile = joinpath(resultdir,"anim_pair_diff_$(key)_anc$(i_anc).mp4")
                    QG2L.animate_fields_pair_divergence(tgrid, fields_cont[2], fields_cont[1], sfs[2], sfs[1], objs[1], objs[2], outfile, sdm)
                    # pairs of contours 
                    outfile = joinpath(resultdir,"anim_pair_overlay_$(key)_anc$(i_anc).mp4")
                    QG2L.animate_fields_contour_divergence(tgrid, fields_cont, outfile, sdm, cfg.target_xPerL, cfg.target_rxPerL, cfg.target_yPerL, cfg.target_ryPerL) 
                end
            end
        end
    end


    if todo["mixture_COAST_phase_diagram"] == 1
        # TODO upgrade
        ytgtstr = @sprintf("%.2f", cfg.target_yPerL*sdm.Ly)
        rxystr = @sprintf("%.3f", cfg.target_ryPerL*sdm.Ly)
        # ------------- Lead-time parameterized ---------------
        # heat map of TV as a function of lead time and scale
        (
         leadtimes,r2threshes,dsts,rsps,mixobjs,
         mixcrit_labels,mixobj_labels,distn_scales,
         fdivnames,Nboot
        ) = expt_config_COAST_analysis(cfg,pertop)
        thresh_dns,scale_dns,shape_dns,peaks_dns,_,ccdf_at_thresh_dns,ccdf_gpd_dns_coarse = JLD2.jldopen(joinpath(resultdir,"GPD_dns.jld2"), "r") do f
            return (
                f["thresh"],
                f["scale"],
                f["shape"],
                f["peak_vals"],
                f["levels"],
                f["ccdf_at_thresh"],
                f["ccdf_GPD_xall"]
           )
        end
        fdivs,iltmixs,mixcrits,gpdpar,gpdpar_ancs = JLD2.jldopen(joinpath(resultdir,"ccdfs_regressed.jld2"),"r") do f
            return f["fdivs"],f["iltmixs"],f["mixcrits"],f["gpdpar"],f["gpdpar_ancs"]
        end
        fdivs2plot = ["tv"]
        fdivlabels = ["TV"]
        i_boot = 1
        i_r2thresh = 1
        for dst = dsts
            for rsp = rsps
                if ("g" == dst) && (rsp in ("2","1+u","2+u"))
                    continue
                end
                # Plot the entropy as a 2D phase plot: both its mean and its variance (not just the proportion of time it's optimal)
                lt_r2thresh_mean,lt_r2thresh_mean_lo,lt_r2thresh_mean_hi = let
                    ost = leadtimes[iltmixs[dst][rsp]["r2"][i_r2thresh,:,1]]
                    mid = sum(ost)/length(ost)
                    hi = sum(ost .* (ost .> mid))/sum(ost .> mid)
                    lo = sum(ost .* (ost .< mid))/sum(ost .< mid)
                    (mid,lo,hi)
                end
                for mc = ["ent"]
                    fig = Figure(size=(500,400))
                    loutmean = fig[1,1] = GridLayout()
                    #loutstd = fig[1,2] = GridLayout()
                    axmean = Axis(loutmean[1,1], xlabel=L"$-\mathrm{AST}$", ylabel=L"$$Scale", title=L"$$Mean %$(mixcrit_labels[mc]), Target lat. %$(ytgtstr), Box size %$(rxystr)", xlabelsize=16, ylabelsize=16, titlesize=16)
                    #axstd = Axis(loutstd[1,1], xlabel=L"$$Lead time", ylabel=L"$$Scale", title=L"$$Max. Ent. std")
                    entmean = SB.mean(mixcrits[dst][rsp][mc][:,:,:]; dims=2)[:,1,:]
                    entstd = SB.std(mixcrits[dst][rsp][mc][:,:,:]; dims=2)[:,1,:]
                    println("entmean = ")
                    display(entmean)
                    println("entstd = ")
                    display(entstd)
                    hmmean = heatmap!(axmean, -sdm.tu.*leadtimes, distn_scales[dst], entmean; colormap=Reverse(:managua))
                    #hmstd = heatmap!(axstd, -sdm.tu.*leadtimes, distn_scales[dst], entstd; colormap=:viridis, colorrange=(0,maximum(entstd)))
                    for ax = (axmean,) #axstd)
                        vlines!(ax, -sdm.tu*lt_r2thresh_mean; color=:black, linestyle=:solid, linewidth=2)
                        vlines!(ax, -sdm.tu*lt_r2thresh_mean_lo; color=:black, linestyle=:dash, linewidth=2)
                        vlines!(ax, -sdm.tu*lt_r2thresh_mean_hi; color=:black, linestyle=:dash, linewidth=2)
                    end
                    cbarmean = Colorbar(loutmean[1,2], hmmean, vertical=true)
                    #cbarstd = Colorbar(loutstd[1,2], hmstd, vertical=true)
                    save(joinpath(figdir,"phdgm_$(dst)_$(rsp)_$(mc).png"), fig)
                end

                for (i_fdivname,fdivname) in enumerate(fdivs2plot)
                    fig = Figure(size=(500,400))
                    lout = fig[1,1] = GridLayout()
                    ax = Axis(lout[1,1], xlabel=L"$-\mathrm{AST}$", ylabel=L"$$Scale", title=L"$$%$(fdivlabels[i_fdivname]), Target lat. %$(ytgtstr), Box size %$(rxystr)", xlabelsize=16, ylabelsize=16, titlesize=16) 
                    hm = heatmap!(ax, -sdm.tu.*leadtimes, distn_scales[dst], fdivs[dst][rsp]["lt"][fdivname][i_boot,:,:]; colormap=:managua)
                    cbar = Colorbar(lout[1,2], hm, vertical=true)
                    # Plot the optimal split time (according to TV) as a function of scale 
                    (ost_mean,ost_mean_lo,ost_mean_hi) = (zeros(Float64, length(distn_scales[dst])) for _=1:3)
                    for i_scl = 1:length(distn_scales[dst])
                        ilts = iltmixs[dst][rsp]["ent"][1,:,i_scl]
                        ost_mean[i_scl] = SB.mean(leadtimes[ilts])
                        ost_mean_lo[i_scl] = sum(leadtimes[ilts] .* (leadtimes[ilts] .< ost_mean[i_scl]))/sum(leadtimes[ilts] .< ost_mean[i_scl])
                        ost_mean_hi[i_scl] = sum(leadtimes[ilts] .* (leadtimes[ilts] .> ost_mean[i_scl]))/sum(leadtimes[ilts] .> ost_mean[i_scl])
                    end
                    scatterlines!(ax, -sdm.tu*ost_mean, distn_scales[dst]; color=:black, linewidth=2) 
                    scatterlines!(ax, -sdm.tu*ost_mean_lo, distn_scales[dst]; color=:black, linestyle=(:dot,:dense), linewidth=2) 
                    scatterlines!(ax, -sdm.tu*ost_mean_hi, distn_scales[dst]; color=:black, linestyle=(:dot,:dense), linewidth=2) 
                    vlines!(ax, -sdm.tu.*lt_r2thresh_mean; color=:black, linestyle=:solid, linewidth=2)
                    vlines!(ax, -sdm.tu.*lt_r2thresh_mean_lo; color=:black, linestyle=:dash, linewidth=2)
                    vlines!(ax, -sdm.tu.*lt_r2thresh_mean_hi; color=:black, linestyle=:dash, linewidth=2)
                    save(joinpath(figdir,"phdgm_$(dst)_$(rsp)_$(fdivname).png"), fig)
                end
                # Now plot the generalized Pareto parameters
                @show shape_dns,scale_dns
                fig = Figure(size=(1000,400))
                loutscale = fig[1,1] = GridLayout()
                loutshape = fig[1,2] = GridLayout()
                axscale = Axis(loutscale[1,1], xlabel=L"$-\mathrm{AST}$", ylabel=L"$$Scale", title=L"$|$GPD scale ($-$DNS)$|$") 
                axshape = Axis(loutshape[1,1], xlabel=L"$-\mathrm{AST}$", ylabel=L"$$Shape", title=L"$|$GPD shape ($-$DNS)$|$") 
                dscale = abs.(gpdpar[dst][rsp]["lt"]["scale"][i_boot,:,:].-scale_dns)
                dshape = abs.(gpdpar[dst][rsp]["lt"]["shape"][i_boot,:,:].-shape_dns)
                println("scale diff = ")
                display(dscale)
                println("shape diff = ")
                display(dshape)
                hmscale = heatmap!(axscale, -sdm.tu.*leadtimes, distn_scales[dst], dscale; colormap=:managua, colorrange=maximum(filter(isfinite,abs.(dscale))).*[0,1])
                cbarscale = Colorbar(loutscale[1,2], hmscale, vertical=true)
                hmshape = heatmap!(axshape, -sdm.tu.*leadtimes, distn_scales[dst], dshape; colormap=:managua, colorrange=maximum(filter(isfinite,abs.(dshape))).*[0,1])
                cbarshape = Colorbar(loutshape[1,2], hmshape, vertical=true)
                for ax = (axscale,axshape)
                    vlines!(ax, -sdm.tu*lt_r2thresh_mean; color=:black, linestyle=:solid)
                    vlines!(ax, -sdm.tu*lt_r2thresh_mean_lo; color=:black, linestyle=:solid)
                    vlines!(ax, -sdm.tu*lt_r2thresh_mean_hi; color=:black, linestyle=:solid)
                end
                save(joinpath(figdir,"phdgm_$(dst)_$(rsp)_gpdpar.png"), fig)
                # Compare entropy-optimizing advance split time with the one-size-fits-all advance split time 
                # 3 curves to compare: (1) constant lead time across all scales, (2) maximum-average-entropy leadtime across all scales, (3) maximum-initial-condition-dependent-entropy leadtime
                for (i_fdivname,fdivname) in enumerate(["tv"])
                    fig = Figure(size=(400,400))
                    lout = fig[1,1] = GridLayout()
                    ax = Axis(lout[1,1], xlabel=fdivlabels[i_fdivname], ylabel=L"$$Scale", title=L"$$Target lat. %$(ytgtstr), Box size %$(rxystr)")
                    fdiv_max_ent = fdivs[dst][rsp]["ent"][fdivname][i_boot,1,:]
                    scatterlines!(ax, fdiv_max_ent, distn_scales[dst]; label="Max. Ent.", color=:red, linestyle=(:dash,:dense))
                    for ilt = 1:Nleadtime
                        scatterlines!(ax, fdivs[dst][rsp]["lt"][fdivname][i_boot,ilt,:], distn_scales[dst]; label="-$(leadtimes[ilt]*sdm.tu)", color=ilt, colorrange=(1,Nleadtime), colormap=:managua)
                    end
                    lout[1,2] = Legend(fig, ax; framevisible=false, labelsize=15)
                    save(joinpath(figdir,"phdgm_slices_$(dst)_$(rsp)_$(fdivname).png"), fig)
                end

                # Same but for GPD parameters
                fig = Figure(size=(800,400))
                loutscale = fig[1,1] = GridLayout()
                loutshape = fig[1,2] = GridLayout()
                axscale = Axis(loutscale[1,1], xlabel=L"$$GPD scale", ylabel=L"$$Pert. scale")
                axshape = Axis(loutshape[1,1], xlabel=L"$$GPD shape", ylabel=L"$$Pert. scale")
                for ilt = 1:Nleadtime
                    scatterlines!(axscale, gpdpar[dst][rsp]["lt"]["scale"][i_boot,ilt,:], distn_scales[dst]; label="-$(leadtimes[ilt]*sdm.tu)", color=ilt, colorrange=(1,Nleadtime), colormap=:managua)
                    scatterlines!(axshape, gpdpar[dst][rsp]["lt"]["shape"][i_boot,ilt,:], distn_scales[dst]; label="-$(leadtimes[ilt]*sdm.tu)", color=ilt, colorrange=(1,Nleadtime), colormap=:managua)
                end
                scatterlines!(axscale, gpdpar[dst][rsp]["ent"]["scale"][i_boot,1,:], distn_scales[dst]; label="Max. ent.", color=:red, linestyle=(:dash,:dense))
                scatterlines!(axshape, gpdpar[dst][rsp]["ent"]["shape"][i_boot,1,:], distn_scales[dst]; label="Max. ent.", color=:red, linestyle=(:dash,:dense))
                vlines!(axscale, scale_dns; color=:black, linestyle=(:dash,:dense), label="DNS")
                vlines!(axshape, shape_dns; color=:black, linestyle=(:dash,:dense), label="DNS")
                vlines!(axscale, gpdpar_ancs["scale_gpd"]; color=:dodgerblue, label="Init", linestyle=(:dash,:dense))
                vlines!(axshape, gpdpar_ancs["shape_gpd"]; color=:dodgerblue, label="Init", linestyle=(:dash,:dense))
                loutscale[1,2] = Legend(fig, axscale; framevisible=false, fontsize=15)
                loutshape[1,2] = Legend(fig, axshape; framevisible=false, fontsize=15)
                save(joinpath(figdir,"phdgm_slices_$(dst)_$(rsp)_gpdpar.png"), fig)
            end
        end
    end

    @show resultdir
    return
end




all_procedures = ["COAST","metaCOAST"]
i_proc = 1

# TODO augment META with composites, lead times displays etc

idx_expt = Vector{Int64}([])
if length(ARGS) > 0
    for i_arg = 1:length(ARGS)
        push!(idx_expt, parse(Int, ARGS[i_arg]))
    end
else
    if "metaCOAST" == all_procedures[i_proc]
        idx_expt = [1,2,3]
    elseif "COAST" == all_procedures[i_proc]
        idx_expt = Vector{Int64}([0])
    elseif "metaCOAST" == all_procedures[i_proc]
        idx_expt = [0] # large or small boxes 
    end
end

php,sdm = QG2L.expt_config()
phpstr = QG2L.strrep_PhysicalParams(php)
sdmstr = QG2L.strrep_SpaceDomain(sdm)
computer = "engaging"
if computer == "engaging"
    expt_supdir_dns = "/orcd/archive/pog/001/ju26596/COAST/QG2L/2024-12-29/0/$(phpstr)_$(sdmstr)"
    expt_supdir_COAST = "/orcd/archive/pog/001/ju26596/COAST/QG2L/2024-12-29/2/$(phpstr)_$(sdmstr)"
else
    expt_supdir_dns = "/Users/justinfinkel/Documents/postdoc_mit/computing/tracer_extremes_resuls/2024-10-22/0/$(phpstr)_$(sdmstr)"
    expt_supdir_COAST = "/Users/justinfinkel/Documents/postdoc_mit/computing/tracer_extremes_resuls/2024-10-22/0/$(phpstr)_$(sdmstr)"
end
dir_dns = joinpath(expt_supdir_dns, "DNS")
ensdir_dns = joinpath(dir_dns,"ensemble_data")
resultdir_dns = joinpath(dir_dns,"analysis")

for i_expt = idx_expt
    println("\n------------------- Starting experiment $i_expt ----------------")
    if "metaCOAST" == all_procedures[i_proc] 
        metaCOAST_latdep_procedure(expt_supdir_COAST, resultdir_dns; i_expt=i_expt)
    elseif "COAST" == all_procedures[i_proc]
        COAST_procedure(ensdir_dns, expt_supdir_COAST; i_expt=i_expt, overwrite_expt_setup=false, overwrite_ensemble=false, old_path_part="net/bstor002.ib",new_path_part="orcd/archive")
    end
    println()
end
