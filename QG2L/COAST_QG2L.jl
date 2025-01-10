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
                "plot_pertop" =>                                    0,
                "compute_dns_objective" =>                          0,
                "plot_dns_objective_stats" =>                       0,
                "anchor" =>                                         0,
                "sail" =>                                           0, 
                "regress_lead_dependent_risk_polynomial" =>         0, 
                "plot_objective" =>                                 0, 
                "mix_COAST_distributions_polynomial" =>             0,
                "plot_COAST_mixture" =>                             0,
                "mixture_COAST_phase_diagram" =>                    1,
                # Danger zone 
                "remove_pngs" =>                                    0,
                # vestigial or hibernating
                "fit_dns_pot" =>                                    0, 
                "plot_contour_divergence" =>                        0,
                "plot_dispersion_metrics" =>                        0,
                "quantify_dispersion" =>                            0,
                "plot_risk_regression_polynomial" =>                0,
               )

    println("expt_config: ")
    php,sdm = QG2L.expt_config()
    println("done")
    println("cop_pertop:")
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
    println("Done")
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
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,
     i_thresh_cquantile,adjust_ccdf_per_ancestor
    ) = expt_config_COAST_analysis(cfg,pertop)
    println("i_thresh_cquantile = $(i_thresh_cquantile)")
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

    if 1 == todo["remove_pngs"]
        for filename = readdir(figdir, join=true)
            if endswith(filename,"png") && (!startswith(filename,"GPD"))
                rm(filename)
            end
        end
    end

    if 1 == todo["plot_pertop"]
        QG2L.plot_PerturbationOperator(pertop, sdm, figdir)
    end

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
        # after choosing the threshold, collect the peaks thereover 
        levels = Rccdf_valid_agglon
        levels_mid = 0.5 .* (levels[1:end-1] .+ levels[2:end])
        buffers = (cfg.peak_prebuffer_time, cfg.follow_time, cfg.lead_time_max)
        ccdf_pot_valid_seplon,ccdf_pot_valid_agglon,gpdpar_valid_agglon,std_valid_agglon = QG2L.compute_local_pot_zonsym(Roft_valid_seplon, levels[i_thresh_cquantile:end], buffers...)
        ccdf_pot_ancgen_seplon,ccdf_pot_ancgen_agglon,gpdpar_ancgen_agglon,std_ancgen_agglon = QG2L.compute_local_pot_zonsym(Roft_ancgen_seplon, levels[i_thresh_cquantile:end], buffers...)
        # Generalized Pareto. TODO compare across a range of thresholds
        #gpdpar_valid_agglon[i_thresh,:] .= QG2L.compute_GPD_params_from_ccdf(thresh, levels[i_thresh:end], ccdf_pot_valid_agglon[i_thresh:end])
        #gpdpar_valid_agglon = QG2L.compute_GPD_params(thresh, levels[i_thresh:end], ccdf_pot_valid_agglon[i_thresh:end])

        

        JLD2.jldopen(joinpath(resultdir,"objective_dns_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).jld2"), "w") do f
            f["tgrid_ancgen"] = tgrid_ancgen
            f["Roft_ancgen_seplon"] = Roft_ancgen_seplon
            f["Rccdf_ancgen_seplon"] = Rccdf_ancgen_seplon
            f["Rccdf_ancgen_agglon"] = Rccdf_ancgen_agglon
            f["ccdf_pot_ancgen_seplon"] = ccdf_pot_ancgen_seplon
            f["ccdf_pot_ancgen_agglon"] = ccdf_pot_ancgen_agglon
            f["tgrid_valid"] = tgrid_valid
            f["Roft_valid_seplon"] = Roft_valid_seplon
            f["Rccdf_valid_seplon"] = Rccdf_valid_seplon
            f["Rccdf_valid_agglon"] = Rccdf_valid_agglon
            f["ccdf_pot_valid_seplon"] = ccdf_pot_valid_seplon
            f["ccdf_pot_valid_agglon"] = ccdf_pot_valid_agglon
            f["gpdpar_valid_agglon"] = gpdpar_valid_agglon
            f["std_valid_agglon"] = std_valid_agglon
        end
    else
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
         gpdpar_valid_agglon,
         std_valid_agglon,
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
                         f["gpdpar_valid_agglon"],
                         f["std_valid_agglon"],
                        )
             end
            )
    end
    if 1 == todo["plot_dns_objective_stats"]
        # TODO do some plotting and quantify the difference between ancgen and valid (train and test?) distributions 
        fig = Figure()
        lout = fig[1,1] = GridLayout()
        axpdf = Axis(lout[1,1], ylabel="Box mean 𝑐", xlabel="PDF", xscale=log10, xgridvisible=false, ygridvisible=false)
        axccdf = Axis(lout[1,2], ylabel="Box mean 𝑐", xlabel="CCDF", xscale=log10, xgridvisible=false, ygridvisible=false)
        Label(lout[1,:,Top()], label_target(cfg, sdm), padding=(5.0,5.0,15.0,5.0), valign=:bottom, halign=:left, fontsize=12)
        bin_edges = collect(range(0,1,200))
        bin_centers = (bin_edges[2:end] .+ bin_edges[1:end-1]) ./ 2
        levels = Rccdf_valid_agglon
        levels_mid = (levels[1:end-1] .+ levels[2:end]) ./ 2
        levels_exc = levels[i_thresh_cquantile:end]
        levels_exc_mid = (levels_exc[1:end-1] .+ levels_exc[2:end]) ./ 2
        thresh_cquantile = ccdf_levels[i_thresh_cquantile]
        thresh = levels[i_thresh_cquantile]
        Nxshifts = div(sdm.Nx, xstride_valid_dns)
        (pdfs_ancgen,pdfs_valid) = (zeros(Float64, (length(bin_centers),Nxshifts)) for _=1:2)
        for i_lon = 1:Nxshifts
            hg_ancgen = SB.normalize(SB.fit(SB.Histogram, Roft_ancgen_seplon[:,i_lon], bin_edges); mode=:pdf)
            pdfs_ancgen[:,i_lon] .= hg_ancgen.weights
            hg_valid = SB.normalize(SB.fit(SB.Histogram, Roft_valid_seplon[:,i_lon], bin_edges); mode=:pdf)
            pdfs_valid[:,i_lon] .= hg_valid.weights
        end
        pdf_valid_agglon = zero2nan(SB.mean(pdfs_valid; dims=2))[:,1]
        GPD = Dists.GeneralizedPareto(thresh, gpdpar_valid_agglon...)

        # PDF
        confint = 0.9
        #lines!(axpdf, thresh_cquantile.*clippdf.(Dists.pdf.(GPD, levels_exc_mid)), levels_exc_mid, color=:gray, linewidth=3, alpha=0.5)
        lines!(axpdf, zero2nan(SB.mean(pdfs_ancgen; dims=2)[:,1]), bin_centers; color=:orange, linewidth=2, linestyle=(:dash,:dense), label="Short DNS ($(round(Int,confint*100))% CI)")
        lines!(axpdf, pdf_valid_agglon, bin_centers; color=:black, linewidth=2, linestyle=(:dash,:dense), label="Long DNS ($(round(Int,confint*100))% CI)")
        for level = (levels[1],levels[end])
            hlines!(axpdf, level; color=:gray, linewidth=1, alpha=0.5)
        end
        # CCDF
        lines!(axccdf, thresh_cquantile.*clipccdf.(Dists.ccdf.(GPD, levels_exc)), levels_exc, color=:gray, linewidth=3, alpha=0.5, label=@sprintf("GPD(%.2f,%.2f,%.2f)\nThresh. exc. prob. %s", thresh, gpdpar_valid_agglon..., powerofhalfstring(i_thresh_cquantile)))
        lines!(axccdf, ccdf_levels, zero2nan(SB.mean(Rccdf_valid_seplon; dims=2)[:,1]); color=:black, linewidth=2, linestyle=(:dash,:dense))
        scatterlines!(axccdf, thresh_cquantile.*zero2nan(SB.mean(ccdf_pot_valid_seplon, ; dims=2)[:,1]), levels[i_thresh_cquantile:end]; color=:black, marker=:star6, label="Long DNS,\npeaks over threshold")
        lines!(axccdf, ccdf_levels, zero2nan(Rccdf_ancgen_seplon[:,1]); color=:orange, linewidth=2, linestyle=(:dash,:dense))
        scatterlines!(axccdf, thresh_cquantile.*zero2nan(ccdf_pot_ancgen_seplon[:,1]), levels[i_thresh_cquantile:end]; color=:orange, marker=:star6, label="Short DNS,\npeaks over threshold")
        vlines!(axccdf, thresh_cquantile; color=:gray, alpha=0.5)
        hlines!(axccdf, thresh; color=:gray, alpha=0.5)
        #lines!(axccdf, zero2nan(Rccdf_valid_agglon), ccdf_levels; color=:gray, alpha=0.5, linewidth=3)
        for q = 1/2 .+ (confint/2).*[-1,1]
            qancgen = zero2nan(QG2L.quantile_sliced(pdfs_ancgen, q, 2)[:,1])
            qvalid = zero2nan(QG2L.quantile_sliced(pdfs_valid, q, 2)[:,1])
            lines!(axpdf, zero2nan(QG2L.quantile_sliced(pdfs_valid, q, 2)[:,1]), bin_centers; color=:black, linestyle=(:dot,:dense))
            lines!(axpdf, qancgen, bin_centers; color=:orange, linestyle=(:dot,:dense))
            qancgen = zero2nan(QG2L.quantile_sliced(Rccdf_ancgen_seplon, q, 2)[:,1])
            qvalid = zero2nan(QG2L.quantile_sliced(Rccdf_valid_seplon, q, 2)[:,1])
            lines!(axccdf, ccdf_levels, qvalid; color=:black, linestyle=(:dot,:dense))
            lines!(axccdf, ccdf_levels, qancgen; color=:orange, linestyle=(:dot,:dense))
            println("For q = $(q), qancgen = ")
            display(qancgen')
            ccancgen = thresh_cquantile.*zero2nan(QG2L.quantile_sliced(ccdf_pot_ancgen_seplon, q, 2)[:,1])
            ccvalid = thresh_cquantile.*zero2nan(QG2L.quantile_sliced(ccdf_pot_valid_seplon, q, 2)[:,1])
            #scatter!(axccdf, ccancgen, levels[i_thresh_cquantile:end]; color=:black, marker=:xcross)
            #scatter!(axccdf, ccvalid, levels[i_thresh_cquantile:end]; color=:orange, marker=:xcross)
        end
        axislegend(axpdf; position=:lb, framevisible=true, labelsize=10)
        axislegend(axccdf; position=:lb, framevisible=true, labelsize=10)

        #linkyaxes!(axpdf, axccdf)
        ylims!(axpdf, 0.0, 1.0)
        ylims!(axccdf, levels[1], levels[end])

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
            if mod(Nmem,100) == 0
                GC.gc()
            end
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
                       "rainbow_pdfs" =>            1,
                       "mixed_ccdfs" =>              1,
                      )

        ytgtstr = @sprintf("%.2f", cfg.target_yPerL*sdm.Ly)
        rxystr = @sprintf("%.3f", cfg.target_ryPerL*sdm.Ly)
        # TODO incorporate bootstraps into this 
        (
         levels,levels_mid,
         dsts,rsps,mixobjs,distn_scales,
         ccdfs,pdfs,
         fdivs,fdivs_ancgen_valid,
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
                    f["fdivs_ancgen_valid"],
                    f["mixcrits"],# mixcrits
                    f["iltmixs"],# iltmixs
                    f["ccdfmixs"],# ccdfmixs
                    f["pdfmixs"],# pdfmixs 
                   )
             end
            )
        println("loaded ccdfs_regressed")
        levels_exc = levels[i_thresh_cquantile:end]
        levels_exc_mid = (levels_exc[1:end-1] .+ levels_exc[2:end]) ./ 2
        pdf_valid_agglon = - diff(ccdf_levels) ./ diff(Rccdf_valid_agglon)
        pdf_valid_seplon = - diff(ccdf_levels) ./ diff(Rccdf_valid_seplon; dims=1)
        pdf_pot_valid_agglon = -diff(ccdf_pot_valid_agglon) ./ diff(levels_exc)
        pdf_pot_valid_seplon = -diff(ccdf_pot_valid_seplon; dims=1) ./ diff(levels_exc)
        #IFT.@infiltrate true

        # ------------- Plots -----------------------
        # For each distribution type, plot the PDFs along the top and a row for each mixing criterion
        ylims = (thresh, max(max_score, levels[end]))


        for dst = ["b"]
            Nscales = length(distn_scales[dst])
            for rsp = ["e","2"]
                if ("g" == dst) && (rsp in ["1+u","2","2+u"])
                    continue
                end

                # --------------- Dependence between (lead time chosen by criteria) and (score) ---------
                if 1 == todosub["gains_topt"]
                    #Threads.@threads for i_scl = 1:Nscales
                    for i_scl = 1:Nscales
                        scalestr = @sprintf("%.3f", distn_scales[dst][i_scl])
                        i_mcobj = 1
                        for (i_mc,mc) in enumerate(["ent","r2"])
                            fig = Figure(size=(600,400))
                            lout = fig[1:2,1] = GridLayout()
                            ax1 = Axis(lout[1,1], xlabel="-AST", ylabel="$(mixcrit_labels[mc])", xlabelvisible=false, xticklabelsvisible=false, title=label_target(cfg, sdm, distn_scales[dst][i_scl])) #L"$$Target lat. %$(ytgtstr), Box size %$(rxystr), Scale %$(scalestr)")
                            ax2 = Axis(lout[2,1], xlabel="-AST", ylabel="Conc.", title="", titlevisible=false)
                            #@show iltmixs[dst][rsp][mc][i_mcobj,:,i_scl]
                            for i_anc = 1:Nanc
                                i_leadtime = iltmixs[dst][rsp][mc][i_mcobj,i_anc,i_scl]
                                if i_leadtime == 0
                                    @show dst,rsp,mc,i_mcobj,i_anc,i_scl
                                    error()
                                end
                                leadtime = leadtimes[i_leadtime]
                                colargs = Dict(:colormap=>:managua, :color=>coast.anc_Rmax[i_anc], :colorrange=>extrema(coast.anc_Rmax))
                                lines!(ax1, -leadtimes.*sdm.tu, mixcrits[dst][rsp][mc][:,i_anc,i_scl]; colargs..., alpha=0.5, linewidth=2.0)
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
                    @assert all(isfinite.(pdfs[dst][rsp]))
                    mixcrits2plot = ["pth","r2","ent"]
                    rspstr = ("1" == rsp ? "linear" : "quadratic")
                    mixcrits_ylabels = ["𝑞(μ)","𝑅² ($(rspstr))","Entropy"]
                    Nleadtimes2plot = Nleadtime
                    #Threads.@threads for i_anc = idx_anc_strat
                    for i_anc = idx_anc_strat
                        pdf_adjustments = 1 ./ replace(ccdfs[dst][rsp][i_thresh_cquantile, :, i_anc, :], 0 => NaN)
                        xlims = maximum(
                                        maximum(pdfs[dst][rsp][i_thresh_cquantile:end,:,i_anc,:]; dims=1)[1,:,:] 
                                        .* pdf_adjustments
                                       ) .* [-0.05, 1.05]

                        t0str = @sprintf("%d", coast.anc_tRmax[i_anc])
                        fig = Figure(size=(100*Nleadtime,100*(3+length(mixcrits2plot))))
                        lout = fig[1,1] = GridLayout()
                        lout_pdfs = lout[1,1] = GridLayout()
                        lout_mixcrits = lout[2,1] = GridLayout()
                        i_col = 0
                        for i_leadtime = round.(Int, range(Nleadtime, 1; length=Nleadtimes2plot))
                            leadtime = leadtimes[i_leadtime]
                            i_col += 1
                            ltstr = @sprintf("−%d",leadtime*sdm.tu)
                            if i_col == 1
                                ltstr = "−AST = $(ltstr)"
                            end
                            lblargs = Dict(:ylabelvisible=>(i_col==1), :yticklabelsvisible=>(i_col==1), :xlabelvisible=>false, :xticklabelsvisible=>true, :ylabelsize=>15, :xlabelsize=>20, :title=>ltstr, :xgridvisible=>false, :ygridvisible=>false, :titlealign=>:right, :titlefont=>:regular)
                            ax = Axis(lout_pdfs[1,i_col]; xlabel="PDF", ylabel="Severity 𝑅*", lblargs...)
                            for (i_scl,scl) in enumerate(distn_scales[dst])
                                scatterlines!(ax, pdf_adjustments[i_scl].*pdfs[dst][rsp][:,i_leadtime, i_anc, i_scl], levels_mid; color=i_scl,colorrange=(0,length(distn_scales[dst])), colormap=:managua, marker=:star6)
                            end
                            #lines!(ax, thresh_cquantile.*pdf_valid_agglon[i_thresh_cquantile:end], levels_exc_mid; color=:black, linestyle=(:dash,:dense), linewidth=1.5)
                            scatterlines!(ax, pdf_pot_valid_agglon, levels_exc_mid; color=:black, linestyle=:solid, marker=:star6,)
                            #scatterlines!(ax, ccdf_levels[i_thresh_cquantile:end], levels_exc; color=:black, linestyle=:solid, linewidth=1.5)
                            #scatterlines!(ax, ccdf_levels[i_thresh_cquantile:end], levels_exc; color=:black, linestyle=:solid, marker=:star6,)
                            hlines!(ax, coast.anc_Rmax[i_anc]; color=:black, linewidth=1.0)
                            idx_desc = desc_by_leadtime(coast, i_anc, leadtime, sdm)
                            scatterlines!(ax, sum([.2,.8].*xlims).*ones(Float64, length(idx_desc)), coast.desc_Rmax[i_anc][idx_desc]; color=:firebrick, markersize=10)
                            #IFT.@infiltrate true
                            ylims!(ax, ylims...)
                            xlims!(ax, xlims...)
                        end
                        Label(lout_pdfs[1,:,Bottom()], "PDF", padding=(0,10,0,25), valign=:bottom, fontsize=16)
                        for (i_mc,mc) in enumerate(mixcrits2plot)
                            ax = Axis(lout_mixcrits[i_mc,1]; ylabel=mixcrits_ylabels[i_mc], xlabel="−AST (𝑡*=$(t0str))", xlabelvisible=(i_mc==length(mixcrits2plot)), xticklabelsvisible=(i_mc==length(mixcrits2plot)), xlabelsize=20, ylabelsize=15, xgridvisible=false, ygridvisible=false)
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
                                ylims!(ax, -0.05, 1.05)
                            end
                        end
                        for i_row = 1:nrows(lout_mixcrits)-1
                            rowgap!(lout_mixcrits, i_row, 10.0)
                        end
                        for i_col = 1:ncols(lout_pdfs)-1
                            colgap!(lout_pdfs, i_col, 0.0)
                        end
                        Label(lout_pdfs[1,:,Top()], label_target(cfg, sdm), padding=(5.0,5.0,20.0,5.0), valign=:bottom, fontsize=20)
                        rowsize!(lout, 1, Relative(2/(2+length(mixcrits2plot))))
                        save(joinpath(figdir,"reg2dist_$(dst)_$(rsp)_anc$(i_anc).png"), fig)
                    end
                end
                # ----------------- Mixed-ancestor plots -----------------
                if 1 == todosub["mixed_ccdfs"]
                    boot_midlohi_pdf(p,avg=false) = begin
                        pmid = clippdf.((avg ? SB.mean(p; dims=2) : p)[:,1])
                        plo,phi = (clippdf.(QG2L.quantile_sliced(p, q, 2)[:,1]) for q=(0.05,0.95))
                        return (pmid,plo,phi)
                    end
                    boot_midlohi_ccdf(cc,avg=false) = begin
                        ccmid = clipccdf.((avg ? SB.mean(cc; dims=2) : cc)[:,1])
                        cclo,cchi = (clipccdf.(QG2L.quantile_sliced(cc, q, 2)[:,1]) for q=(0.05,0.95))
                        return (ccmid,cclo,cchi)
                    end
                    levels_exc = levels[i_thresh_cquantile:end]
                    levels_exc_mid = 0.5 .* (levels_exc[1:end-1] .+ levels_exc[2:end])
                    # Straight tails (not peaks)
                    # ancestors
                    Rccdf_ancgen_pt,Rccdf_ancgen_lo,Rccdf_ancgen_hi = boot_midlohi_ccdf(Rccdf_ancgen_seplon)
                    pdf_ancgen_seplon = -diff(ccdf_levels) ./ diff(Rccdf_ancgen_seplon, dims=1)
                    pdf_ancgen_pt,pdf_ancgen_lo,pdf_ancgen_hi = boot_midlohi_pdf(pdf_ancgen_seplon)
                    # validation
                    Rccdf_valid_pt,Rccdf_valid_lo,Rccdf_valid_hi = boot_midlohi_ccdf(Rccdf_valid_seplon, true)
                    pdf_valid_seplon = -diff(ccdf_levels) ./ diff(Rccdf_valid_seplon, dims=1)
                    pdf_valid_pt,pdf_valid_lo,pdf_valid_hi = boot_midlohi_pdf(pdf_valid_seplon, true)
                    # peaks over thresholds 
                    # ancestors
                    ccdf_pot_ancgen_pt,ccdf_pot_ancgen_lo,ccdf_pot_ancgen_hi = boot_midlohi_ccdf(ccdf_pot_ancgen_seplon, false)
                    pdf_pot_ancgen_seplon = -diff(ccdf_pot_ancgen_seplon; dims=1) ./ diff(levels_exc)
                    pdf_pot_ancgen_pt,pdf_pot_ancgen_lo,pdf_pot_ancgen_hi = boot_midlohi_pdf(pdf_pot_ancgen_seplon, false)
                    # validation
                    ccdf_pot_valid_pt,ccdf_pot_valid_lo,ccdf_pot_valid_hi = boot_midlohi_ccdf(ccdf_pot_valid_seplon, true)
                    pdf_pot_valid_seplon = -diff(ccdf_pot_valid_seplon; dims=1) ./ diff(levels_exc)
                    pdf_pot_valid_pt,pdf_pot_valid_lo,pdf_pot_valid_hi = boot_midlohi_pdf(pdf_pot_valid_seplon, true)


                    GPD = Dists.GeneralizedPareto(levels[i_thresh_cquantile], gpdpar_valid_agglon...)
                    pdf_gpd = thresh_cquantile.*clippdf.(Dists.pdf.(GPD, levels_exc_mid))
                    ccdf_gpd = thresh_cquantile.*clipccdf.(Dists.ccdf.(GPD, levels_exc))

                    for i_scl = [1,round(Int,length(distn_scales[dst])/2)]
                        for (fdivname,fdivlabel) = (("kl","KL"),("chi2","χ²"),("tv","TV"))
                            scalestr = @sprintf("%.3f", distn_scales[dst][i_scl])

                            i_boot = 1
                            fdiv_synclt,ilt_fdiv_synclt = findmin(fdivs[dst][rsp]["lt"][fdivname][i_boot,:,i_scl]) # index for best synchronized advance split time 
                            fdiv_syncpth,ilt_fdiv_syncpth = findmin(fdivs[dst][rsp]["pth"][fdivname][i_boot,:,i_scl]) # index for best synchronized threshold exceedance probability
                            fdiv_ancgen_valid_pt,fdiv_ancgen_valid_lo,fdiv_ancgen_valid_hi = let
                                fdav = fdivs_ancgen_valid[fdivname] 
                                (SB.mean(fdav), SB.quantile(fdav, 0.05), SB.quantile(fdav, 0.95))
                            end
                            for mc = ["ent"] #,"r2","ent"]
                                for i_mcobj = 1:length(mixobjs[mc])
                                    fdiv_cond = fdivs[dst][rsp][mc][fdivname][i_boot,i_mcobj,i_scl]
                                    ltstr = @sprintf("%.1f",leadtimes[ilt_fdiv_synclt]*sdm.tu)
                                    pthstr = @sprintf("%.1f",mixobjs["pth"][ilt_fdiv_syncpth])
                                    fdivstr_synclt = @sprintf("%.3f",fdiv_synclt)
                                    fdivstr_syncpth = @sprintf("%.3f",fdiv_syncpth)
                                    fdivstr_cond = @sprintf("%.3f",fdiv_cond)
                                    fdivstr_ancgen = @sprintf("%.3f (%.3f, %.3f)",fdiv_ancgen_valid_pt, fdiv_ancgen_valid_lo, fdiv_ancgen_valid_hi)
                                        
                                    pdf_cond_mid,pdf_cond_lo,pdf_cond_hi = boot_midlohi_pdf(pdfmixs[dst][rsp][mc][:,:,i_mcobj,i_scl])
                                    pdf_synclt_mid,pdf_synclt_lo,pdf_synclt_hi = boot_midlohi_pdf(pdfmixs[dst][rsp]["lt"][:,:,ilt_fdiv_synclt,i_scl])
                                    pdf_syncpth_mid,pdf_syncpth_lo,pdf_syncpth_hi = boot_midlohi_pdf(pdfmixs[dst][rsp]["pth"][:,:,ilt_fdiv_syncpth,i_scl])
                                    ccdf_cond_mid,ccdf_cond_lo,ccdf_cond_hi = boot_midlohi_ccdf(ccdfmixs[dst][rsp][mc][:,:,i_mcobj,i_scl])
                                    ccdf_synclt_mid,ccdf_synclt_lo,ccdf_synclt_hi = boot_midlohi_ccdf(ccdfmixs[dst][rsp]["lt"][:,:,ilt_fdiv_synclt,i_scl])
                                    ccdf_syncpth_mid,ccdf_syncpth_lo,ccdf_syncpth_hi = boot_midlohi_ccdf(ccdfmixs[dst][rsp]["pth"][:,:,ilt_fdiv_syncpth,i_scl])
                                    #
                                    # Plot CCDFS 
                                    #
                                    #
                                    dnspot = thresh_cquantile.*ccdf_pot_valid_pt
                                    fig = Figure(size=(1000,400))
                                    lout = fig[1,1] = GridLayout()
                                    ax1 = Axis(lout[1,2]; xscale=log10, xlabel="CCDF", ylabel="Box mean 𝑐", title=label_target(cfg,sdm,distn_scales[dst][i_scl],rsp), titlefont=:regular, xgridvisible=false, ygridvisible=false)
                                    ax2 = Axis(lout[1,3]; xscale=log10, xlabel="CCDF/CCDF(DNS)", ylabel="Box mean 𝑐", titlefont=:regular, ylabelvisible=false, xgridvisible=false, ygridvisible=false)
                                    # GPD
                                    lines!(ax1, ccdf_gpd, levels_exc; color=:gray, alpha=0.5, linewidth=3, label=@sprintf("GPD(%.2f,%.2f,%s%.2f)", levels[i_thresh_cquantile], gpdpar_valid_agglon[1], (gpdpar_valid_agglon[2] > 0 ? "+" : "−"), abs(gpdpar_valid_agglon[2])))
                                    lines!(ax2, clipccdfratio.(ccdf_gpd./dnspot), levels_exc; color=:gray, alpha=0.5, linewidth=3)
                                    # DNS (non-peak, peak)
                                    lines!(ax1, clipccdf.(ccdf_levels[i_thresh_cquantile:end]), levels_exc; color=:black, linestyle=(:dash,:dense), linewidth=2, label=@sprintf("Long DNS"))
                                    scatterlines!(ax1, dnspot, levels_exc; linewidth=4, color=:black, marker=:star6, label="Long DNS, peaks")
                                    scatterlines!(ax2, clipccdfratio.(dnspot./dnspot), levels_exc; linewidth=4, color=:black, marker=:star6)
                                    # Ancestor run (non-peak, peak)
                                    lines!(ax1, clipccdf.(ccdf_levels[i_thresh_cquantile:end]), Rccdf_ancgen_pt[i_thresh_cquantile:end]; linewidth=2, color=:orange, linestyle=(:dash,:dense), label=@sprintf("Short DNS"))
                                    scatterlines!(ax1, thresh_cquantile.*ccdf_pot_ancgen_pt, levels[i_thresh_cquantile:end]; linewidth=4, color=:orange, marker=:star6, label="Short DNS, peaks\n$(fdivlabel) = $(fdivstr_ancgen)")
                                    scatterlines!(ax2, clipccdfratio.(thresh_cquantile.*ccdf_pot_ancgen_pt./dnspot), levels[i_thresh_cquantile:end]; linewidth=4, color=:orange, marker=:star6)
                                    # Synced AST 
                                    scatterlines!(ax1, ccdf_synclt_mid[i_thresh_cquantile:end], levels_exc; color=:cyan, linestyle=:solid, marker=:star6, label="AST = $(ltstr) [synchron] \n$(fdivlabel) = $(fdivstr_synclt)", linewidth=3)
                                    scatterlines!(ax2, clipccdfratio.(ccdf_synclt_mid[i_thresh_cquantile:end]./dnspot), levels_exc; color=:cyan, linestyle=:solid, marker=:star6, linewidth=3)
                                    # Probability-thresholded AST 
                                    scatterlines!(ax1, ccdf_syncpth_mid[i_thresh_cquantile:end], levels_exc; color=:purple, linestyle=:solid, marker=:star6, label="𝑞ₙ(μ) = $(pthstr) [synprob] \n$(fdivlabel) = $(fdivstr_syncpth)", linewidth=2)
                                    scatterlines!(ax2, clipccdfratio.(ccdf_syncpth_mid[i_thresh_cquantile:end]./dnspot), levels_exc; color=:purple, linestyle=:solid, marker=:star6, label="𝑞ₙ(μ) = $(pthstr) [synprob] \n$(fdivlabel) = $(fdivstr_syncpth)", linewidth=2)
                                    # Conditionally optimal AST 
                                    scatterlines!(ax1, ccdf_cond_mid[i_thresh_cquantile:end], levels_exc; linewidth=1, color=:red, linestyle=:solid, label="$(mixobj_labels[mc][i_mcobj])\n$(fdivlabel) = $(fdivstr_cond)", marker=:star6)
                                    scatterlines!(ax2, clipccdfratio.(ccdf_cond_mid[i_thresh_cquantile:end]./dnspot), levels_exc; linewidth=1, color=:red, linestyle=:solid, marker=:star6)

                                    lout[1,1] = Legend(fig, ax1; framevisible=true)
                                    colsize!(lout, 2, Relative(1/2))
                                    colsize!(lout, 3, Relative(1/4))
                                    for ax = (ax1,ax2)
                                        ylims!(ax, 1.1*levels[i_thresh_cquantile]-0.1*levels[i_thresh_cquantile+1], 1.1*levels[end]-0.1*levels[end-1])
                                    end
                                    xlims!(ax1, 1/(time_valid_dns_ph*sdm.tu*10), thresh_cquantile*1.1)
                                    xlims!(ax2, 1/10, 10)
                                    save(joinpath(figdir,"ccdfmixs_$(dst)_$(rsp)_$(mc)_$(i_mcobj)_$(fdivname)_$(i_scl).png"), fig)
                                end
                            end
                        end
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
        todosub = Dict(
                       "mcmean_heatmap" =>             1,
                       "fdiv_heatmap" =>               1,
                       "phdgm_slices" =>               1,
                      )
        (
         leadtimes,r2threshes,dsts,rsps,mixobjs,
         mixcrit_labels,mixobj_labels,distn_scales,
         fdivnames,Nboot,ccdf_levels,
         time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,
         i_thresh_cquantile,adjust_ccdf_per_ancestor
        ) = expt_config_COAST_analysis(cfg,pertop)
        fdivs,fdivs_ancgen_valid,iltmixs,mixcrits = JLD2.jldopen(joinpath(resultdir,"ccdfs_regressed.jld2"),"r") do f
            return f["fdivs"],f["fdivs_ancgen_valid"],f["iltmixs"],f["mixcrits"]
        end
        fdivs2plot = ["tv","chi2","kl"]
        fdivlabels = ["TV","χ²","KL"]
        i_boot = 1
        i_r2thresh = 1
        for dst = ["b"]
            for rsp = ["e","2"]
                if ("g" == dst) && (rsp in ("2","1+u","2+u"))
                    continue
                end
                # Plot the entropy as a 2D phase plot: both its mean and its variance (not just the proportion of time it's optimal)
                if 1 == todosub["mcmean_heatmap"]
                    lt_r2thresh_mean,lt_r2thresh_mean_lo,lt_r2thresh_mean_hi = let
                        ost = leadtimes[iltmixs[dst][rsp]["r2"][i_r2thresh,:,1]]
                        mid = sum(ost)/length(ost)
                        hi = sum(ost .* (ost .> mid))/sum(ost .> mid)
                        lo = sum(ost .* (ost .< mid))/sum(ost .< mid)
                        (mid,lo,hi)
                    end
                    # Heatmap of various mixing criteria as a function of AST; below, reverse it
                    for mc = ["pth","ent","r2"]
                        fig = Figure(size=(500,400))
                        loutmean = fig[1,1] = GridLayout()
                        axmean = Axis(loutmean[1,1], xlabel="−AST", ylabel="Scale", title="Mean $(mixcrit_labels[mc]), $(label_target(cfg, sdm, rsp))", xlabelsize=16, ylabelsize=16, titlesize=16, titlefont=:regular)
                        mcmean = SB.mean(mixcrits[dst][rsp][mc][:,:,:]; dims=2)[:,1,:]
                        mcstd = SB.std(mixcrits[dst][rsp][mc][:,:,:]; dims=2)[:,1,:]
                        println("mcmean = ")
                        display(mcmean)
                        println("mcstd = ")
                        display(mcstd)
                        hmmean = heatmap!(axmean, -sdm.tu.*leadtimes, distn_scales[dst], mcmean; colormap=Reverse(:managua))
                        for ax = (axmean,) #axstd)
                            vlines!(ax, -sdm.tu*lt_r2thresh_mean; color=:black, linestyle=:solid, linewidth=2)
                            vlines!(ax, -sdm.tu*lt_r2thresh_mean_lo; color=:black, linestyle=:dash, linewidth=2)
                            vlines!(ax, -sdm.tu*lt_r2thresh_mean_hi; color=:black, linestyle=:dash, linewidth=2)
                        end
                        cbarmean = Colorbar(loutmean[1,2], hmmean, vertical=true)
                        save(joinpath(figdir,"phdgm_$(dst)_$(rsp)_$(mc).png"), fig)
                    end
                    # TODO heat map of leadtime as a function of probability
                end

                if 1 == todosub["fdiv_heatmap"]
                    lt_r2thresh_mean = SB.mean(leadtimes[iltmixs[dst][rsp]["r2"][i_r2thresh,:,1]])
                    for (i_fdivname,fdivname) in enumerate(fdivs2plot)
                        # pth as the independent variable 
                        fig = Figure(size=(500,400))
                        lout = fig[1,1] = GridLayout()
                        ax = Axis(lout[1,1], xlabel="𝑞(μ)", ylabel="Scale", title="$(fdivlabels[i_fdivname]), $(label_target(cfg,sdm,rsp))", xlabelsize=16, ylabelsize=16, titlesize=16, titlefont=:regular) 
                        Npth = length(mixobjs["pth"])
                        @show Npth
                        Nscale = length(distn_scales[dst])
                        ltmean = zeros(Float64, (Npth, Nscale))
                        for i_pth = 1:Npth
                            for i_scl = 1:Nscale
                                ltmean[i_pth,i_scl] = SB.mean(leadtimes[iltmixs[dst][rsp]["pth"][i_pth,:,i_scl]])
                            end
                        end
                        hm = heatmap!(ax, mixobjs["pth"], distn_scales[dst], fdivs[dst][rsp]["pth"][fdivname][i_boot,:,:]; colormap=:managua)
                        cbar = Colorbar(lout[1,2], hm, vertical=true)
                        co = contour!(ax, mixobjs["pth"], distn_scales[dst], -sdm.tu.*ltmean; levels=-sdm.tu.*leadtimes, color=:black, labels=true)
                        # Plot the optimal split time (according to fdiv) as a function of scale 
                        save(joinpath(figdir,"phdgm_$(dst)_$(rsp)_$(fdivname)_indeppth.png"), fig)
                        # --------------- AST as the independent variable ------------
                        fig = Figure(size=(500,400))
                        lout = fig[1,1] = GridLayout()
                        ax = Axis(lout[1,1], xlabel="−AST", ylabel="Scale", title="$(fdivlabels[i_fdivname]), $(label_target(cfg,sdm))", xlabelsize=16, ylabelsize=16, titlesize=16, titlefont=:regular) 
                        hm = heatmap!(ax, -sdm.tu.*leadtimes, distn_scales[dst], fdivs[dst][rsp]["lt"][fdivname][i_boot,:,:]; colormap=:managua)
                        cbar = Colorbar(lout[1,2], hm, vertical=true)
                        co = contour!(ax, -sdm.tu.*leadtimes, distn_scales[dst], SB.mean(mixcrits[dst][rsp]["pth"][:,:,:]; dims=2)[:,1,:]; levels=mixobjs["pth"], color=:black, labels=true)
                        # Plot the optimal split time (according to fdiv) as a function of scale 
                        if dst in ["1","2"]
                            vlines!(ax, -sdm.tu.*lt_r2thresh_mean; color=:black, linestyle=(:dash,:dense), linewidth=2)
                        end
                        save(joinpath(figdir,"phdgm_$(dst)_$(rsp)_$(fdivname).png"), fig)
                    end
                end
                # Compare entropy-optimizing advance split time with the one-size-fits-all advance split time 
                # 3 curves to compare: (1) constant lead time across all scales, (2) maximum-average-entropy leadtime across all scales, (3) maximum-initial-condition-dependent-entropy leadtime
                if 1 == todosub["phdgm_slices"]
                    for (i_fdivname,fdivname) in enumerate(fdivs2plot)
                        fdiv_max_ent = fdivs[dst][rsp]["ent"][fdivname][i_boot,1,:]
                        fdiv_last_r2 = fdivs[dst][rsp]["r2"][fdivname][i_boot,1,:]
                        fig = Figure(size=(600,600))
                        lout = fig[1,1] = GridLayout()
                        (axlt,axpth) = (Axis(lout[i,1], xlabel=fdivlabels[i_fdivname], ylabel="Scale", title=label_target(cfg, sdm), titlefont=:regular) for i=1:2)
                        for ax = (axlt,axpth)
                            scatterlines!(ax, fdiv_max_ent, distn_scales[dst]; label="Max. Ent.", color=:red, linestyle=(:dash,:dense), linewidth=3)
                            vlines!(ax, SB.mean(fdivs_ancgen_valid[fdivname]); color=:orange, alpha=1.0, linewidth=4, label="Short DNS peaks\n(90% CI)")
                            for sgn = [-1,1]
                                vlines!(ax, SB.quantile(fdivs_ancgen_valid[fdivname], 0.5+0.5*sgn*0.9); color=:orange, linewidth=4, linestyle=(:dot,:dense))
                            end
                            if ax == axlt
                                scatterlines!(ax, fdiv_last_r2, distn_scales[dst]; label="𝑅²≈$(r2thresh)", color=:black, linestyle=(:dash,:dense), linewidth=3)
                            end
                        end
                        # Synchron
                        ax = axlt
                        for ilt = 1:Nleadtime
                            lines!(ax, fdivs[dst][rsp]["lt"][fdivname][i_boot,ilt,:], distn_scales[dst]; label="-$(leadtimes[ilt]*sdm.tu)", color=ilt, colorrange=(1,Nleadtime), colormap=:managua)
                        end
                        # Synprob
                        ax = axpth
                        Npth = length(mixobjs["pth"])
                        for i_pth = 1:Npth
                            lines!(ax, fdivs[dst][rsp]["pth"][fdivname][i_boot,i_pth,:], distn_scales[dst]; label=@sprintf("%.2f",mixobjs["pth"][i_pth]), color=i_pth, colorrange=(1,Npth), colormap=:managua)
                        end
                        lout[1,2] = Legend(fig, axlt; framevisible=true, labelsize=8, nbanks=2, title="AST [synchron]")
                        lout[2,2] = Legend(fig, axpth; framevisible=true, labelsize=8, nbanks=2, title="𝑞(μ) [synprob]")

                        linkxaxes!(axlt,axpth)
                        linkyaxes!(axlt,axpth)
                        axlt.xlabelvisible = axlt.xticklabelsvisible = false
                        axpth.titlevisible = false
                        colsize!(lout, 1, Relative(1/2))
                        rowgap!(lout, 1, 0)
                        save(joinpath(figdir,"phdgm_slices_$(dst)_$(rsp)_$(fdivname).png"), fig)
                    end
                end
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
        idx_expt = vec([6,9][2:2] .+ [0,1,2][2:2]'.*15) #Vector{Int64}([6,9])
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
