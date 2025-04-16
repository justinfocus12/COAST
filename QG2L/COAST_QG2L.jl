include("./QG2L.jl")
include("./EnsembleMod.jl")

import .QG2L as QG2L 
import .EnsembleMod as EM

import Printf
using Printf: @sprintf
import ZernikePolynomials as ZP
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
using Infiltrator: @infiltrate

using CairoMakie

# Other possible acronyms: OATH (optimize a(t?) time horizon); FLOAT (Fixed-Lead Optimization of Antecedent Trajectories); COAST ((Fixed-Lead time)/(Finding Leverage/Likely)/(Forcing Lightly)/(Flicking Likely Extreme Event Trajectory Sampling), EPIC (Early Perturbation of Initial Conditions) or of course iTEAMS; OPAQUE (Optimal Perturbations Ahead of Quasigeostrophic Extremes), PASTE (Perturbing Ahead (of Sudden)/(to Sample) Transient Extremes) 


include("./COAST_basics.jl")
include("./COAST_plotting.jl")
include("./COAST_mixdist.jl")
include("./metaCOAST.jl")




function COAST_procedure(ensdir_dns::String, resultdir_dns::String, expt_supdir::String; i_expt=nothing, overwrite_expt_setup=false, overwrite_ensemble=false, old_path_part::String, new_path_part::String)
    todo = Dict{String,Bool}(
                             "upgrade_ensemble" =>                               0,
                             "update_paths" =>                                   0,
                             "plot_transcorr" =>                                 0,
                             "plot_pertop" =>                                    0,
                             "plot_bumps" =>                                     0,
                             "compute_dns_objective" =>                          0,
                             "plot_dns_objective_stats" =>                       0,
                             "use_backups" =>                                    0,
                             "anchor" =>                                         0,
                             "sail" =>                                           0, 
                             "compute_contour_dispersion" =>                     0,
                             "plot_contour_dispersion_distribution" =>           0,
                             "regress_lead_dependent_risk_polynomial" =>         0, 
                             "evaluate_mixing_criteria" =>                       0,
                             "plot_objective" =>                                 0, 
                             "plot_conditional_pdfs" =>                          0,
                             "plot_mixcrits_overlay" =>                          0,
                             "mix_COAST_distributions" =>                        0,
                             "plot_COAST_mixture" =>                             1,
                             "mixture_COAST_phase_diagram" =>                    0,
                             "plot_composite_contours" =>                        0,
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
    mkpath(expt_supdir)
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
     mixcrit_labels,mixobj_labels,mixcrit_colors,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,
     i_thresh_cquantile,adjust_ccdf_per_ancestor
    ) = expt_config_COAST_analysis(cfg,pertop)
    Nleadtime = length(leadtimes)
    num_perts_max_per_leadtime = div(cfg.num_perts_max, Nleadtime)
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
    ensfile_COAST_backup = joinpath(ensdir_COAST,"ens_backup.jld2")
    coastfile_COAST = joinpath(ensdir_COAST,"coast.jld2") 
    coastfile_COAST_backup = joinpath(ensdir_COAST,"coast_backup.jld2") 
    rngfile_COAST = joinpath(ensdir_COAST,"rng.bin")
    rngfile_COAST_backup = joinpath(ensdir_COAST,"rng_backup.bin")
    resultdir = joinpath(exptdir_COAST,"results")
    figdir = joinpath(exptdir_COAST,"figures")
    
    mkpath(exptdir_COAST)
    mkpath(ensdir_COAST)
    # TODO wrap the below in a loop 
    init_cond_dir = joinpath(ensdir_COAST,"init_cond")
    mkpath(init_cond_dir)
    mkpath(resultdir)
    mkpath(figdir)

    if todo["remove_pngs"]
        for filename = readdir(figdir, join=true)
            revised_figure_substrings = ["conditionalpdf","objectives_anc","nodesc","contours_anc","corrs_anc","linquad"]
            if endswith(filename,"png") && any(occursin.(revised_figure_substrings,filename))
                rm(filename)
            end
        end
    end


    if (!isfile(ensfile_COAST)) || overwrite_ensemble
        println("Starting afresh")
        ens = EM.Ensemble()
        rng = Random.MersenneTwister(3718)
        coast = COASTState(pertop.pert_dim)
        EM.save_Ensemble(ens, ensfile_COAST_backup)
        cp(ensfile_COAST_backup, ensfile_COAST, force=true)
        save_COASTState(coast, coastfile_COAST_backup)
        cp(coastfile_COAST_backup, coastfile_COAST, force=true)
        open(rngfile_COAST_backup,"w") do f
            Serialization.serialize(f, rng)
        end
        cp(rngfile_COAST_backup, rngfile_COAST, force=true)
    end

    if todo["use_backups"]
        cp(coastfile_COAST_backup, coastfile_COAST, force=true)
        cp(rngfile_COAST_backup, rngfile_COAST, force=true)
        cp(ensfile_COAST_backup, ensfile_COAST, force=true)
    end


    ensfile_dns = joinpath(ensdir_dns, "ens.jld2")

    if todo["upgrade_ensemble"]
        #upgrade_Ensemble!(ensfile_dns, ensfile_dns)
        upgrade_Ensemble!(ensfile_COAST, ensfile_COAST)
    end

    ens_dns = EM.load_Ensemble(ensfile_dns)
    @show EM.get_Nmem(ens_dns)
    @show ens_dns.trajs[end].tfin
    ens = EM.load_Ensemble(ensfile_COAST)
    coast = load_COASTState(coastfile_COAST)

    #@infiltrate


    if todo["plot_transcorr"]
        plot_transcorr(figdir)
    end
    if todo["plot_pertop"]
        QG2L.plot_PerturbationOperator(pertop, sdm, coast.pert_seq_qmc[:,1:min(3,num_perts_max_per_leadtime)], figdir)
    end
    if todo["plot_bumps"]
        i_mode_sf = 1
        support_radius = pertop.sf_pert_amplitudes_max[i_mode_sf]
        QG2L.plot_bump_densities_2d(distn_scales["b"], support_radius, coast.pert_seq_qmc[:,1:num_perts_max_per_leadtime], figdir)
    end
    #@infiltrate
    #error()
    if (1==todo["update_paths"]) && (!isnothing(old_path_part)) && (!isnothing(new_path_part))
        #adjust_paths!(ens_dns, old_path_part, new_path_part)
        #EM.save_Ensemble(ens_dns, ensfile_dns)
        adjust_paths!(ens, old_path_part, new_path_part)
        EM.save_Ensemble(ens, ensfile_COAST)
        adjust_paths!(coast, old_path_part, new_path_part)
        save_COASTState(coast, coastfile_COAST)
    end
    # ------------ KLUG for failed synchronization previous time ----------------
    Nmem_ens = EM.get_Nmem(ens)
    Nmem_coast = length(coast.ancestors) + sum(length.(coast.desc_Rmax))
    if Nmem_ens != Nmem_coast
        if Nmem_ens != Nmem_coast + 1
            error()
        else
            EM.remove_final_trajectory!(ens)
        end
    end
    adjust_scores!(coast, ens, cfg, sdm)
    @show coast.pert_seq_qmc[:,1:6]
    rng = Serialization.deserialize(rngfile_COAST)

    dns_objective_filename = joinpath(resultdir,"objective_dns_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).jld2")

    if todo["compute_dns_objective"]
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

        

        JLD2.jldopen(dns_objective_filename, "w") do f
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
             JLD2.jldopen(dns_objective_filename, "r") do f
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
    if todo["plot_dns_objective_stats"]
        bin_edges = collect(range(0,1,200))
        bin_centers = (bin_edges[2:end] .+ bin_edges[1:end-1]) ./ 2
        levels = Rccdf_valid_agglon
        Nlev = length(levels)
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
        #
        #
        #
        # Plots 
        #
        shortcolor = :firebrick
        # confidence interval 
        pdf_ci_ancgen,pdf_ci_valid = (zeros(length(bin_centers), 2) for _=1:2)
        Rccdf_ci_ancgen,Rccdf_ci_valid = (zeros(length(ccdf_levels), 2) for _=1:2)
        ccdf_pot_ci_ancgen,ccdf_pot_ci_valid = (zeros(length(levels_exc), 2) for _=1:2)
        confint = 0.9
        for (i_q,q) in enumerate(1/2 .+ (confint/2).*[-1,1])
            pdf_ci_ancgen[:,i_q] .= zero2nan(QG2L.quantile_sliced(pdfs_ancgen, q, 2)[:,1])
            pdf_ci_valid[:,i_q] .= zero2nan(QG2L.quantile_sliced(pdfs_valid, q, 2)[:,1])
            Rccdf_ci_ancgen[:,i_q] .= zero2nan(QG2L.quantile_sliced(Rccdf_ancgen_seplon, q, 2)[:,1])
            Rccdf_ci_valid[:,i_q] .= zero2nan(QG2L.quantile_sliced(Rccdf_valid_seplon, q, 2)[:,1])
            ccdf_pot_ci_ancgen[:,i_q] .= zero2nan(QG2L.quantile_sliced(ccdf_pot_ancgen_seplon, q, 2)[:,1])
            ccdf_pot_ci_valid[:,i_q] .= zero2nan(QG2L.quantile_sliced(ccdf_pot_valid_seplon, q, 2)[:,1])
        end

        # Determine the indices to plot. d = pdf; c = cdf; p = pot; a = ancgen; v = valid
        ida = findall(all(isfinite.(pdf_ci_ancgen); dims=2)[:,1])
        ica = findall(all(isfinite.(Rccdf_ci_ancgen); dims=2)[:,1])
        ipa = findall(all(isfinite.(ccdf_pot_ci_ancgen); dims=2)[:,1])
        idv = findall(all(isfinite.(pdf_ci_valid); dims=2)[:,1])
        icv = findall(all(isfinite.(Rccdf_ci_valid); dims=2)[:,1])
        ipv = findall(all(isfinite.(ccdf_pot_ci_valid); dims=2)[:,1])

        shortlabel,longlabel = [
                                @sprintf(
                                         "%s (%d-day) DNS & %d%% CI", 
                                         squal, squant, confint*100
                                         ) 
                                for (squal,squant)=[("Short",time_ancgen_dns_ph),("Long",time_valid_dns_ph)]
                               ]
        #
        fig = Figure()
        lout = fig[1,1] = GridLayout()
        axargs = Dict(:xscale=>log10, :xgridvisible=>false, :ygridvisible=>false, :xlabelsize=>12, :ylabelsize=>12, :yticklabelsize=>10, :titlesize=>12, :titlefont=>:regular)
        axpdf = Axis(lout[2,1]; title="Intensity 𝑅", xlabel="PDF", axargs...)
        axccdf = Axis(lout[2,2]; title="Intensity 𝑅", xlabel="CCDF", axargs...)
        axpot = Axis(lout[2,3]; title="Severity 𝑅*", xlabel="CCDF", axargs...)
        Label(lout[1,:], @sprintf("%s, threshold μ[%s]=%.2f", label_target(cfg, sdm), powerofhalfstring(i_thresh_cquantile), thresh); padding=(5.0,5.0,0.0,5.0), valign=:top, halign=:center, fontsize=12)

        # PDF
        #lines!(axpdf, thresh_cquantile.*clippdf.(Dists.pdf.(GPD, levels_exc_mid)), levels_exc_mid, color=:gray, linewidth=3, alpha=0.5)
        band!(axpdf, (Point2f.(pdf_ci_ancgen[ida,i_q],bin_centers[ida]) for i_q=1:2)...; color=shortcolor, alpha=0.5)
        band!(axpdf, (Point2f.(pdf_ci_valid[idv,i_q],bin_centers[idv]) for i_q=1:2)...; color=:gray60, alpha=0.5)
        lines!(axpdf, zero2nan(SB.mean(pdfs_ancgen; dims=2)[:,1]), bin_centers; color=shortcolor, linewidth=2, linestyle=(:dash,:dense), label=shortlabel)
        lines!(axpdf, pdf_valid_agglon, bin_centers; color=:black, linewidth=2, linestyle=(:dash,:dense), label=longlabel)
        for level = (levels[1],levels[end])
            hlines!(axpdf, level; color=:gray, linewidth=1, alpha=0.5)
        end
        for ax = (axccdf,axpot)
            lines!(ax, thresh_cquantile.*clipccdf.(Dists.ccdf.(GPD, levels_exc)), levels_exc, color=:gray, linewidth=3, alpha=1.0, label=@sprintf("GPD(%.2f,%.2f,%s%.2f)", thresh, gpdpar_valid_agglon[1], (gpdpar_valid_agglon[2] >= 0 ? "+" : "−"), abs(gpdpar_valid_agglon[2])))
            vlines!(ax, thresh_cquantile; color=:gray, alpha=0.5)
            hlines!(ax, thresh; color=:gray, alpha=0.5)
        end
        # CCDF
        band!(axccdf, (Point2f.(ccdf_levels[ica],Rccdf_ci_ancgen[ica,i_q]) for i_q=1:2)...; color=shortcolor, alpha=0.5)
        band!(axccdf, (Point2f.(ccdf_levels[icv],Rccdf_ci_valid[icv,i_q]) for i_q=1:2)...; color=:gray60, alpha=0.5)
        lines!(axccdf, ccdf_levels, zero2nan(SB.mean(Rccdf_valid_seplon; dims=2)[:,1]); color=:black, linewidth=2, linestyle=(:dash,:dense))
        lines!(axccdf, ccdf_levels, zero2nan(Rccdf_ancgen_seplon[:,1]); color=shortcolor, linewidth=2, linestyle=(:dash,:dense))
        # POT CCDF
        band!(axpot, (Point2f.(thresh_cquantile.*ccdf_pot_ci_ancgen[ipa,i_q],levels_exc[ipa]) for i_q=1:2)...; color=shortcolor, alpha=0.5)
        band!(axpot, (Point2f.(thresh_cquantile.*ccdf_pot_ci_valid[ipv,i_q],levels_exc[ipv]) for i_q=1:2)...; color=:gray60, alpha=0.5)
        lines!(axpot, thresh_cquantile.*zero2nan(SB.mean(ccdf_pot_valid_seplon, ; dims=2)[:,1]), levels[i_thresh_cquantile:end]; color=:black, linestyle=(:dash,:dense), label=longlabel)
        lines!(axpot, thresh_cquantile.*zero2nan(ccdf_pot_ancgen_seplon[:,1]), levels[i_thresh_cquantile:end]; color=shortcolor, linestyle=(:dash,:dense), label=shortlabel)

        # ------------ Legend code from julia forum via Perplexity ------------

        plots_in_fig = AbstractPlot[]
        labels_in_fig = AbstractString[]
        for ax = (axpdf,axccdf,axpot)
            pl, lb = Makie.get_labeled_plots(ax, merge=false, unique=false)
            append!(plots_in_fig, pl)
            append!(labels_in_fig, lb)
        end
        ulabels = Base.unique(labels_in_fig)
        mergedplots = [
                       [lp for (i, lp) in enumerate(plots_in_fig) if labels_in_fig[i] == ul]
                       for ul in ulabels
                      ]
        
        Legend(lout[3,:], mergedplots, ulabels, nbanks=3, merged=true, labelsize=10)
        # ------------------------------------------------------------------



        ylims!(axpdf, 0.0, 1.0)
        ylims!(axccdf, levels[1], levels[end])
        ylims!(axpot, levels[i_thresh_cquantile], levels[end])

        rowsize!(lout, 1, Relative(1/10))
        rowsize!(lout, 3, Relative(1/8))

        colgap!(lout, 1, 10)
        colgap!(lout, 2, 10)
        rowgap!(lout, 1, 0)
        rowgap!(lout, 2, 0)

        save(joinpath(figdir,"histograms_ancgen_tancgen$(round(Int,time_ancgen_dns_ph))_tvalid$(round(Int,time_valid_dns_ph)).png"),fig)
        
        
        @show SB.mean(Roft_ancgen_seplon),SB.std(Roft_ancgen_seplon),length(Roft_ancgen_seplon)
        @show SB.mean(Roft_valid_seplon),SB.std(Roft_valid_seplon),length(Roft_valid_seplon)


        @show figdir
    end



    # --------------- THE KING OF THRESHES -----------
    thresh = Rccdf_valid_agglon[i_thresh_cquantile] 
    # ------------------------------------------------


    # -------------------------------------------------------------------------------------------

    if todo["anchor"]
        new_peak_frontier_time = round(Int, time_spinup_dns_ph/sdm.tu)
        new_peak_backier_time = new_peak_frontier_time + round(Int, time_ancgen_dns_ph/sdm.tu)
        if length(coast.ancestor_init_conds) > 0
            new_peak_frontier_time = coast.peak_times_upper_bounds[end]
        end
        while (new_peak_frontier_time < new_peak_backier_time) && (length(coast.ancestor_init_conds) < cfg.num_init_conds_max)
            i_anc = length(coast.ancestor_init_conds) + 1
            println("Preparing initial condition for i_anc $(i_anc)")
            init_cond_file = joinpath(init_cond_dir,"init_cond_anc$(i_anc).jld2")
            prehistory_file = joinpath(init_cond_dir,"init_cond_prehistory_anc$(i_anc).jld2")
            init_prep = prepare_init_cond_from_dns(ens_dns, obj_fun_COAST_from_file, cfg, sdm, cop, php, pertop, init_cond_file, prehistory_file, new_peak_frontier_time, new_peak_backier_time, thresh; num_peaks_to_skip=0)
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
            cp(ensfile_COAST, ensfile_COAST_backup, force=true)
            cp(coastfile_COAST, coastfile_COAST_backup, force=true)
            new_peak_frontier_time = t_downcross
        end
        @show coast.dns_peaks
        @show coast.dns_peak_times



        # Reset termination, in case the capacity has been expanded
        if coast.peak_times_upper_bounds[end] >= new_peak_backier_time
            coast.terminate = true
        elseif length(coast.ancestors) < cfg.num_init_conds_max
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



    if todo["sail"]
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
            # ensemble 
            EM.save_Ensemble(ens, ensfile_COAST)
            # rng
            open(rngfile_COAST,"w") do f
                Serialization.serialize(f, rng)
            end
            # coast 
            save_COASTState(coast, coastfile_COAST)

            # Overwrite backup files 
            cp(ensfile_COAST, ensfile_COAST_backup, force=true)
            cp(rngfile_COAST, rngfile_COAST_backup, force=true)
            cp(coastfile_COAST, coastfile_COAST_backup, force=true)

            # collect garbage 

            if mod(Nmem,100) == 0
                GC.gc()
            end
        end
    end
    adjust_scores!(coast, ens, cfg, sdm)
    Nmem = EM.get_Nmem(ens)
    Nanc = length(coast.ancestors)
    min_score,max_score = extrema(vcat(coast.anc_Rmax, (coast.desc_Rmax[i_anc] for i_anc=1:Nanc)...))
    #
    # --------------------- Decide on a subset of ancestors to display ---------------
    #
    ancorder = sortperm(coast.anc_Rmax; rev=true)
    idx_anc_strat_Rmax = ancorder[round.(Int, range(1, Nanc; length=4))]
    # Choose ancestors based on other criteria
    #idx_anc_strat = sort(unique(ancorder[range(1,min(12,Nanc); step=1)]))
    #idx_anc_strat = intersect(1:Nanc, idx_anc_strat)[1:min(4,length(idx_anc_strat))]
    # ---------------------------------------------------------------------------------
    #

    contour_dispersion_filename = joinpath(resultdir,"contour_dispersion.jld2")
    if todo["compute_contour_dispersion"]
        dns_stats_filename = joinpath(resultdir_dns, "moments_mssk_conc.jld2")
        compute_contour_dispersion(coast, ens, cfg, sdm, cop, pertop, dns_stats_filename, contour_dispersion_filename,thresh)
    end





    if todo["regress_lead_dependent_risk_polynomial"]
        (
         coefs_linear,residmse_linear,rsquared_linear,resid_range_linear,
         coefs_quadratic,residmse_quadratic,rsquared_quadratic,resid_range_quadratic,
         hessian_eigvals,hessian_eigvecs,
         coefs_zernike,residmse_zernike,rsquared_zernike,resid_range_zernike,
        ) = regress_lead_dependent_risk(coast, ens, cfg, sdm, pertop)
        JLD2.jldopen(joinpath(resultdir,"regression_coefs.jld2"), "w") do f
            f["coefs_zernike"] = coefs_zernike
            f["residmse_zernike"] = residmse_zernike
            f["rsquared_zernike"] = rsquared_zernike
            f["resid_range_zernike"] = resid_range_zernike

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
        
    if todo["evaluate_mixing_criteria"]
        println("About to evaluate mixing criteria")
        evaluate_mixing_criteria(cfg, cop, pertop, coast, ens, resultdir)
        println("finished evaluating mixing criteria")
    end

    # ------- Select ancestors to stratify on dispersion speed ---------
    iltmixs = JLD2.jldopen(joinpath(resultdir,"mixcrits_ccdfs_pdfs.jld2"),"r") do f
        return f["ilts"]
    end
    ancorder_ltmix_ent = sortperm(iltmixs["b"]["e"]["ent"][1,:,div(length(distn_scales["b"]),2)])
    idx_anc_strat_ent = ancorder_ltmix_ent[round.(Int, range(1, Nanc; length=4))]
    # ---------------------------------------------------

    if todo["plot_contour_dispersion_distribution"]
        plot_contour_dispersion_distribution(coast, ens, cfg, sdm, cop, pertop, contour_dispersion_filename, idx_anc_strat_ent, figdir)
    end

    if todo["plot_objective"]
        rxystr = @sprintf("%.3f",cfg.target_ryPerL*sdm.Ly)
        ytgtstr = @sprintf("%.2f",cfg.target_yPerL*sdm.Ly)
        todosub = Dict{String,Bool}(
                                    "plot_spaghetti" =>                 0,
                                    "plot_response" =>                  1,
                                    "plot_response_fixpert_varytime" => 0,
                                   )

        regcoefs_filename = joinpath(resultdir,"regression_coefs.jld2")
        (
         coefs_zernike,residmse_zernike,rsquared_zernike,
         coefs_linear,residmse_linear,rsquared_linear,
         coefs_quadratic,residmse_quadratic,rsquared_quadratic,
         hessian_eigvals,hessian_eigvecs
        )= JLD2.jldopen(regcoefs_filename, "r") do f
            return (
                    f["coefs_zernike"],
                    f["residmse_zernike"],
                    f["rsquared_zernike"],
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

        for i_anc = idx_anc_strat_ent
            if todosub["plot_spaghetti"]
                plot_objective_spaghetti(cfg, sdm, cop, pertop, ens, coast, i_anc, thresh, figdir)
            end


            # ------------- Plot 3: perturbations in the complex plane, along with level sets of the linear and quadratic models. Also show R^2 ------------
            if todosub["plot_response"]
                plot_objective_response_linquad(
                    cfg, sdm, cop, pertop, ens, coast, i_anc, 
                    coefs_zernike, residmse_zernike, rsquared_zernike,
                    coefs_linear, residmse_linear, rsquared_linear,
                    coefs_quadratic, residmse_quadratic, rsquared_quadratic,
                    hessian_eigvals, hessian_eigvecs,
                    figdir
                   )
            end
            if todosub["plot_response_fixpert_varytime"]
                Nperts2plot = 6
                fig = Figure(size=(400,100*Nperts2plot))
                lout = fig[1,1] = GridLayout()
                axs = [Axis(lout[i_pert,1], xlabel="-AST", ylabel="𝑅") for i_pert=1:Nperts2plot]
                Rmaxoflt = zeros(Float64, Nleadtime)
                traj_anc = ens.trajs[i_anc]
                t_anc = floor(Int, traj_anc.tphinit/sdm.tu)+1:1:traj_anc.tfin
                itRmax = coast.anc_tRmax[i_anc]-t_anc[1]-1
                for i_pert = 1:Nperts2plot
                    ax = axs[i_pert]
                    for (i_leadtime,leadtime) in enumerate(leadtimes) # = 1:Nleadtime
                        idx_dsc = desc_by_leadtime(coast,i_anc,leadtime,sdm)
                        #Rmaxoflt[i_leadtime] = coast.desc_Rmax[i_anc][idx_dsc[i_pert]] #[i_leadtime]
                        Rmaxoflt[i_leadtime] = coast.desc_Roft[i_anc][idx_dsc[i_pert]][itRmax] #[i_leadtime]
                    end
                    lines!(ax, -reverse(leadtimes.*sdm.tu), Rmaxoflt; color=:black)
                    hlines!(ax, coast.anc_Rmax[i_anc]; color=:black, linestyle=(:dash,:dense))
                    for i_pert = 1:Nperts2plot-1
                        rowgap!(lout, i_pert, 0)
                        ax.xlabelvisible = ax.xticklabelsvisible = false
                    end
                end
                save(joinpath(figdir,"response_fixpert_varytime_anc$(i_anc).png"), fig)

            end

        end
    end

    if todo["plot_conditional_pdfs"]
        mixcrits,ccdfs,pdfs,iltmixs = JLD2.jldopen(joinpath(resultdir,"mixcrits_ccdfs_pdfs.jld2"),"r") do f
            return f["mixcrits"],f["ccdfs"],f["pdfs"],f["ilts"]
        end
        mixcrits2plot = ["ei","ent","globcorr","contcorr"] # TODO group them 
        #idx_leadtimes2plot = reverse(unique(clamp.(round.(Int, length(leadtimes).*[1/20, 1/5, 2/5, 3/5]), 1, length(leadtimes))))
        idx_leadtimes2plot = reverse(collect(round.(Int, range(1, length(leadtimes); length=6))))
        levels = Rccdf_valid_agglon
        Nlev = length(levels)
        levels_exc = levels[i_thresh_cquantile:end]
        levels_exc_mid = (levels_exc[1:end-1] .+ levels_exc[2:end]) ./ 2
        pdf_valid_agglon = - diff(ccdf_levels) ./ diff(Rccdf_valid_agglon)
        pdf_valid_seplon = - diff(ccdf_levels) ./ diff(Rccdf_valid_seplon; dims=1)
        pdf_pot_valid_agglon = -diff(ccdf_pot_valid_agglon) ./ diff(levels_exc)
        pdf_pot_valid_seplon = -diff(ccdf_pot_valid_seplon; dims=1) ./ diff(levels_exc)
        ylims = (thresh, max(max_score, levels[end]))
        dst = "b"
        for rsp = ["z","2","e"]
            for i_anc = idx_anc_strat_ent
                xlims = maximum(
                                maximum(pdfs[dst][rsp][i_thresh_cquantile:Nlev-1,:,i_anc,:]./ccdfs[dst][rsp][i_thresh_cquantile:i_thresh_cquantile,:,i_anc,:]; dims=1)[1,:,:] 
                               ) .* [-0.05, 1.05] .* thresh_cquantile

                t0str = @sprintf("%d", coast.anc_tRmax[i_anc])
                fig = Figure(size=(600,100*(3+length(mixcrits2plot))))
                lout = fig[1,1] = GridLayout()
                lout_pdfs = lout[1,1] = GridLayout()
                lout_mixcrits = lout[2,1] = GridLayout()
                i_col = 0
                for i_leadtime = idx_leadtimes2plot
                    leadtime = leadtimes[i_leadtime]
                    i_col += 1
                    ltstr = @sprintf("−%d",leadtime*sdm.tu)
                    if i_col == 1
                        ltstr = "−AST = $(ltstr)"
                    end
                    lblargs = Dict(:ylabelvisible=>(i_col==1), :yticklabelsvisible=>(i_col==1), :xlabelvisible=>false, :xticklabelsvisible=>true, :ylabelsize=>15, :xlabelsize=>20, :title=>ltstr, :xgridvisible=>false, :ygridvisible=>false, :titlealign=>:right, :titlefont=>:regular, :xticklabelrotation=>-pi/2)
                    ax = Axis(lout_pdfs[1,i_col]; xlabel="PDF", ylabel="Severity 𝑅*", lblargs...)
                    for (i_scl,scl) in enumerate(distn_scales[dst])
                        # TODO turn into a bar plot 
                        pth = ccdfs[dst][rsp][i_thresh_cquantile,i_leadtime,i_anc,i_scl]
                        scatterlines!(ax, 
                                pdfs[dst][rsp][i_thresh_cquantile:Nlev-1,i_leadtime, i_anc, i_scl] .* (thresh_cquantile/pth), 
                                levels_exc_mid, 
                                ;
                                color=i_scl,colorrange=(0,length(distn_scales[dst])), 
                                colormap=:RdYlBu_4, marker=:star6
                               )
                    end
                    scatterlines!(ax, 
                            thresh_cquantile.*pdf_pot_valid_agglon, 
                            levels_exc_mid,
                            ; 
                            color=:black, linestyle=:solid, marker=:star6,)
                    hlines!(ax, coast.anc_Rmax[i_anc]; color=:black, linewidth=1.0)
                    idx_desc = desc_by_leadtime(coast, i_anc, leadtime, sdm)
                    scatterlines!(ax, sum([.2,.8].*xlims).*ones(Float64, length(idx_desc)), coast.desc_Rmax[i_anc][idx_desc]; color=:firebrick, markersize=10)
                    ylims!(ax, ylims...)
                    xlims!(ax, xlims...)
                end
                Label(lout_pdfs[1,:,Bottom()], "PDF", padding=(0,10,0,25), valign=:bottom, fontsize=16)
                for (i_mc,mc) in enumerate(mixcrits2plot)
                    ax = Axis(lout_mixcrits[i_mc,1]; ylabel=mixcrit_labels[mc], ylabelvisible=true, ylabelrotation=0, titlefont=:regular, xlabel="−AST (𝑡*=$(t0str))", xlabelvisible=(i_mc==length(mixcrits2plot)), xticklabelsvisible=(i_mc==length(mixcrits2plot)), xlabelsize=20, ylabelsize=15, xgridvisible=false, ygridvisible=false)
                    for (i_scl,scl) in enumerate(distn_scales[dst])
                        if mc in ["globcorr","contcorr"]
                            colargs = Dict(:color=>i_scl,:colormap=>:RdYlBu_4,:colorrange=>(0,length(distn_scales[dst])))
                            scatterlines!(ax, -leadtimes.*sdm.tu, transcorr.(mixcrits[dst][rsp][mc][:,i_anc,i_scl]); colargs...)
                            hlines!(ax, transcorr(1-(3/8)^2); color=:gray, alpha=0.25, linewidth=2, linestyle=(:dash,:dense))
                            ax.ylabel = @sprintf("σ⁻¹(%s)", mixcrit_labels[mc])

                        else
                            colargs = Dict(:color=>i_scl,:colormap=>:RdYlBu_4,:colorrange=>(0,length(distn_scales[dst])))
                            scatterlines!(ax, -leadtimes.*sdm.tu, mixcrits[dst][rsp][mc][:,i_anc,i_scl]; colargs...)
                        end
                        if mc in ["ent","ei"]
                            vlines!(ax, -leadtimes[iltmixs[dst][rsp][mc][1,i_anc,i_scl]]*sdm.tu; colargs...,linewidth=5*(1 - (i_scl-1)/length(distn_scales[dst])))
                        end
                    end
                    xlims!(ax, -(cfg.lead_time_max+0.5*cfg.lead_time_inc)*sdm.tu, -(cfg.lead_time_min-0.5*cfg.lead_time_inc)*sdm.tu)
                    if mc in ["pth","pim","r2"]
                        ylims!(ax, -0.05, 1.05)
                    end
                end
                rowsize!(lout, 1, 3/(3+length(mixcrits2plot)))
                for i_row = 1:nrows(lout_mixcrits)-1
                    rowgap!(lout_mixcrits, i_row, 2.0)
                end
                for i_col = 1:ncols(lout_pdfs)-1
                    colgap!(lout_pdfs, i_col, 0.0)
                end
                Label(lout_pdfs[1,:,Top()], label_target(cfg, sdm), padding=(5.0,5.0,20.0,5.0), valign=:bottom, fontsize=20)
                rowsize!(lout, 1, Relative(1.5/(1.5+length(mixcrits2plot))))
                save(joinpath(figdir,"conditionalpdfs_$(dst)_$(rsp)_anc$(i_anc).png"), fig)
            end
        end
    end

    if todo["plot_mixcrits_overlay"]
        mixcrits,ccdfs,pdfs,iltmixs = JLD2.jldopen(joinpath(resultdir,"mixcrits_ccdfs_pdfs.jld2"),"r") do f
            return f["mixcrits"],f["ccdfs"],f["pdfs"],f["ilts"]
        end
        mixcrits2plot = ["ei","ent","globcorr","contcorr"] # TODO group them 
        Nmc = length(mixcrits2plot)
        dst = "b"
        for rsp = ["z","2","e"]
            Nscales = length(distn_scales[dst])
            fig = Figure(size=(450,150*(Nmc+1)))
            lout = fig[1,1] = GridLayout()
            axs = [
                   Axis(lout[i_mc,1]; xlabel="−AST", ylabel=mixcrit_labels[mixcrits2plot[i_mc]], titlefont=:regular, xlabelvisible=(i_mc==Nmc), xticklabelsvisible=(i_mc==Nmc), xlabelsize=15, ylabelsize=15, xticklabelsize=12, yticklabelsize=12, xgridvisible=false, ygridvisible=false) 
                   for i_mc=1:Nmc
                  ]
            mcmean,mclo,mchi = (zeros(Float64, (Nleadtime,1,Nscales)) for _=1:3)
            scales2plot = [1,12]
            for (i_mc,mc) in enumerate(mixcrits2plot)
                let mcofast = mixcrits[dst][rsp][mc][1:Nleadtime,1:Nanc,1:Nscales] 
                    mcmean .= SB.mean(mcofast; dims=2)
                    mclo .= SB.sum(mcofast .* (mcofast .<= mcmean); dims=2) ./ SB.sum(mcofast .<= mcmean; dims=2)
                    mchi .= SB.sum(mcofast .* (mcofast .>= mcmean); dims=2) ./ SB.sum(mcofast .>= mcmean; dims=2)
                end
                ax = axs[i_mc]
                if mc in ["globcorr","contcorr"]
                    for mcstat = (mcmean,mclo,mchi)
                        mcstat .= transcorr.(mcstat)
                    end
                    ax.ylabel = "σ⁻¹($(mixcrit_labels[mc]))"
                end
                if mc in ["globcorr","contcorr"]
                    ylims!(ax, transcorr(-0.05), transcorr(1.0))
                    hlines!(ax, transcorr(0.0); color=:gray, alpha=0.5)
                end
                if mc in ["pth","pim","r2"]
                    ylims!(ax, 0.0, 1.0)
                end
                if mc in ["ei","eot","ent"]
                    ylims!(0.0, maximum(mixcrits[dst][rsp][mc]))
                end
                for i_scl = scales2plot
                    band!(ax, -leadtimes.*sdm.tu, mclo[:,1,i_scl], mchi[:,1,i_scl]; color=:gray, alpha=0.25)
                end
                for i_scl = scales2plot
                    colargs = Dict(:color=>i_scl, :colorrange=>(0,Nscales), :colormap=>:RdYlBu_4)
                    scatterlines!(ax, -leadtimes.*sdm.tu, mcmean[:,1,i_scl]; colargs..., linewidth=2, label="scale $(@sprintf("%.1f",distn_scales[dst][i_scl]))")
                    lines!(ax, -leadtimes.*sdm.tu, mclo[:,1,i_scl]; colargs..., linewidth=2, linestyle=(:dash,:dense))
                    lines!(ax, -leadtimes.*sdm.tu, mchi[:,1,i_scl]; colargs..., linewidth=2, linestyle=(:dash,:dense))
                end
                if i_mc < Nmc
                    rowgap!(lout, i_mc, 0)
                end
            end
            axs[1].title = "$(label_target(cfg, sdm)), μ[$(powerofhalfstring(i_thresh_cquantile))]"
            #lout[Nmc+1,1] = Legend(fig, axs[Nmc]; framevisible=false, labelsize=15)
            #rowsize!(lout, Nmc+1, Relative(1/(3*Nmc)))

            save(joinpath(figdir, "mixcrits_overlay_$(dst)_$(rsp).png"), fig)
        end
    end



    if todo["mix_COAST_distributions"]
        println("About to mix COAST distributions")
        mix_COAST_distributions(cfg, cop, pertop, coast, ens, resultdir)
        println("Finished mixing")
    end
    if todo["plot_COAST_mixture"]
        println("About to plot COAST mixtures")

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
        ) = (JLD2.jldopen(joinpath(resultdir,"ccdfs_combined.jld2"),"r") do f
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
        println("loaded ccdfs_combined")
        #@infiltrate
        levels_exc = levels[i_thresh_cquantile:end]
        Nlev = length(levels)
        levels_exc_mid = (levels_exc[1:end-1] .+ levels_exc[2:end]) ./ 2
        pdf_valid_agglon = - diff(ccdf_levels) ./ diff(Rccdf_valid_agglon)
        pdf_valid_seplon = - diff(ccdf_levels) ./ diff(Rccdf_valid_seplon; dims=1)
        pdf_pot_valid_agglon = -diff(ccdf_pot_valid_agglon) ./ diff(levels_exc)
        pdf_pot_valid_seplon = -diff(ccdf_pot_valid_seplon; dims=1) ./ diff(levels_exc)

        # ------------- Plots -----------------------
        # For each distribution type, plot the PDFs along the top and a row for each mixing criterion
        ylims = (thresh, max(max_score, levels[end]))


        for dst = ["b"]
            Nscales = length(distn_scales[dst])
            for rsp = ["z","2","e"] #,"2"]
                if ("g" == dst) && (rsp in ["1+u","2","2+u"])
                    continue
                end
                scales2plot = [1,4,8,12]
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

                mcs2mix = ["lt","contcorr","globcorr","ei","ent"]

                for i_scl = scales2plot
                    for (fdivname,fdivlabel) = (("qrmse","𝐿²"),("kl","KL"),("chi2","χ²")) #("kl","KL"),("chi2","χ²"),("tv","TV"))
                        scalestr = @sprintf("%.3f", distn_scales[dst][i_scl])

                        ccdfs_opt,fdivs_opt,imcs_opt = (Dict{String,Dict}() for _=1:3)
                        mcstrs_opt,fdivstrs_opt = (Dict{String,Dict}() for _=1:2)
                        labels_opt = Dict{String,String}()
                        #
                        # Ancestor run 
                        colargs = Dict(:color=>:black,)

                        for mc = mcs2mix
                            ccdfs_opt[mc] = Dict{String,Array{Float64}}()
                            fdivs_opt[mc] = Dict{String,Array{Float64}}()
                            imcs_opt[mc] = Dict{String,Array{Int64}}()
                            mcstrs_opt[mc],fdivstrs_opt[mc] = (Dict{String,String}() for _=1:2)
                            for est = ["mix","pool"]
                                fdivs_opt[mc][est] = zeros(Float64, Nboot+1)
                                imcs_opt[mc][est] = zeros(Int64, Nboot+1)
                                ccdfs_opt[mc][est] = zeros(Float64, (Nlev,Nboot+1))
                                for i_boot = 1:Nboot+1
                                    fdivs_opt[mc][est][i_boot],imcs_opt[mc][est][i_boot] = findmin(fdivs[dst][rsp][mc][est][fdivname][i_boot,:,i_scl])
                                    ccdfs_opt[mc][est][:,i_boot] .= ccdfmixs[dst][rsp][mc][est][:,i_boot,imcs_opt[mc][est][i_boot],i_scl]
                                end
                                if mc in ["globcorr","contcorr"]
                                    mcstrs_opt[mc][est] = @sprintf("σ(%.1f)",transcorr(mixobjs[mc][imcs_opt[mc][est][1]]))
                                elseif mc in ["lt"]
                                    mcstrs_opt[mc][est] = @sprintf("%d",mixobjs[mc][imcs_opt[mc][est][1]])
                                else
                                    mcstrs_opt[mc][est] = "max" #@sprintf("%.1f",mixobjs[mc][imcs_opt[mc][est]])
                                end
                                fdivstrs_opt[mc][est] = @sprintf("%.1E",fdivs_opt[mc][est][1])
                            end
                            if mc in ["globcorr","contcorr","lt"]
                                labels_opt[mc] = "$(mixcrit_labels[mc])\n=$(mcstrs_opt[mc]["mix"]) / $(mcstrs_opt[mc]["pool"])" #\n$(fdivlabel) = $(fdivstrs_opt[mc]["mix"]) / $(fdivstrs_opt[mc]["pool"])"
                            else
                                labels_opt[mc] = "max $(mixcrit_labels[mc])" # \n$(fdivlabel) = $(fdivstrs_opt[mc]["mix"]) / $(fdivstrs_opt[mc]["pool"])"
                            end
                            
                        end
                        fdiv_ancgen_valid_pt,fdiv_ancgen_valid_lo,fdiv_ancgen_valid_hi = let
                            fdav = filter(isfinite, fdivs_ancgen_valid[fdivname]) 
                            (SB.mean(fdav), SB.quantile(fdav, 0.05), SB.quantile(fdav, 0.95))
                        end
                        fdivstr_ancgen = @sprintf("%.1E\n(%.2E, %.2E)",fdiv_ancgen_valid_pt, fdiv_ancgen_valid_lo, fdiv_ancgen_valid_hi)
                        dnspot = thresh_cquantile.*ccdf_pot_valid_pt
                        # -------------- Plot --------------
                        axargs = Dict(:xscale=>log10, :xlabel=>"CCDF", :ylabel=>"Severity 𝑅*", :titlefont=>:regular, :xgridvisible=>false, :ygridvisible=>false, :ylabelvisible=>false, :yticklabelsvisible=>false, :titlesize=>10, :xlabelsize=>8, :xticklabelsize=>6)
                        Nmcs2mix = length(mcs2mix)
                        # ---------------- Figure layout -------------------
                        fig = Figure(size=(100*(Nmcs2mix+2),400))
                        lout = fig[1,1] = GridLayout()
                        axs_mcseps = [
                                      Axis(lout[1,1+i_mc]; axargs..., title=labels_opt[mcs2mix[i_mc]], )
                                      for i_mc=1:Nmcs2mix
                                     ]
                        #ax1 = Axis(lout[1,1]; xscale=log10, xlabel="CCDF", ylabel="Severity 𝑅*", title="$(label_target(cfg,sdm,distn_scales[dst][i_scl])), μ[$(powerofhalfstring(i_thresh_cquantile))]", titlefont=:regular, xgridvisible=false, ygridvisible=false)
                        axargs[:title] = "Short DNS"
                        ax1 = Axis(lout[1,1]; axargs...)
                        axargs[:xlabel] = "CCDF/CCDF(DNS)"
                        ax2 = Axis(lout[1,1+Nmcs2mix+1]; axargs...)
                        axargs[:xscale] = identity
                        axargs[:yscale] = log10
                        delete!(axargs, :xlabel)
                        delete!(axargs, :title)
                        axargs[:xticklabelsvisible] = false
                        ax3 = Axis(lout[2,1:Nmcs2mix+1]; axargs...)

                        # DNS 
                        for ax = (ax1,axs_mcseps...)
                            lines!(ax, dnspot, levels_exc; linewidth=3, color=:black, linestyle=(:dash,:dense), label="Long DNS")
                        end
                        lines!(ax2, clipccdfratio.(dnspot./dnspot), levels_exc; linewidth=3, color=:black, linestyle=(:dash,:dense))
                        # GPD
                        lines!(ax1, ccdf_gpd, levels_exc; color=:gray, alpha=0.5, linewidth=3)
                        lines!(ax2, clipccdfratio.(ccdf_gpd./dnspot), levels_exc; color=:gray, alpha=0.5, linewidth=3)
                        # Short DNS 
                        # TODO make into a band
                        #nnidx = findlast((isfinite.(ccdf) for ccdf=[ccdf_pot_ancgen_lo,ccdf_pot_ancgen_hi,ccdf_pot_ancgen_pt])...)
                        #band!(ax1, Point2f.(thresh_cquantile.*ccdf_pot_ancgen_lo[1:nnidx], levels_exc[1:nnidx]), Point2f.(thresh_cquantile.*ccdf_pot_ancgen_hi[1:nnidx]), levels_exc[1:nnidx]; color=:gray, alpha=0.5)
                        for ccdf = (ccdf_pot_ancgen_lo,ccdf_pot_ancgen_pt,ccdf_pot_ancgen_hi)
                            scatterlines!(ax1, thresh_cquantile.*ccdf, levels[i_thresh_cquantile:end]; linewidth=3, colargs..., marker=:circle)
                        end
                        scatter!(ax3, 1, fdiv_ancgen_valid_pt; color=:black, marker=:circle, markersize=12)
                        lines!(ax3, [1,1], [fdiv_ancgen_valid_lo, fdiv_ancgen_valid_hi]; color=:black, linewidth=2)
                        scatterlines!(ax2, clipccdfratio.(thresh_cquantile.*ccdf_pot_ancgen_pt./dnspot), levels[i_thresh_cquantile:end]; marker=:circle, linewidth=3, colargs...)
                        for (i_mc,mc) in enumerate(mcs2mix)
                            for (est,linestyle,marker,yoffset) in (("mix",:solid,:xcross,0.1),("pool",(:dot,:dense),'O',-0.1))
                                fdivlo,fdivhi = [SB.quantile(fdivs_opt[mc][est][2:Nboot+1], q) for q=[0.05,0.95]]
                                ccdflo,ccdfhi = [thresh_cquantile.*QG2L.quantile_sliced(ccdfs_opt[mc][est][i_thresh_cquantile:end,2:Nboot+1], q, 2)[:,1] for q=[0.05,0.95]]
                                if est == "mix"; band!(axs_mcseps[i_mc], Point2f.(ccdflo, levels_exc), Point2f.(ccdfhi, levels_exc); color=mixcrit_colors[mc], alpha=0.5); end
                                #band!(ax2, Point2f.(ccdflo./dnspot, levels_exc), Point2f.(ccdfhi./dnspot, levels_exc); color=mixcrit_colors[mc], alpha=0.5) # TODO 
                                scatter!(ax3, (1+i_mc+yoffset), fdivs_opt[mc][est][1]; color=mixcrit_colors[mc], marker=marker, markersize=12)
                                lines!(ax3, (1+i_mc+yoffset).*ones(Float64, 2), [fdivlo,fdivhi]; color=mixcrit_colors[mc], linestyle=linestyle)
                                for i_boot = 1:1
                                    scatterlines!(axs_mcseps[i_mc], thresh_cquantile.*ccdfs_opt[mc][est][i_thresh_cquantile:end,i_boot], levels_exc; color=mixcrit_colors[mc], linestyle=linestyle, marker=marker, label=labels_opt[mc], linewidth=1.5, )
                                    scatterlines!(ax2, clipccdfratio.(thresh_cquantile.*ccdfs_opt[mc][est][i_thresh_cquantile:end,i_boot]./dnspot), levels_exc; color=mixcrit_colors[mc], linestyle=linestyle, marker=marker, linewidth=1.5)
                                end
                            end
                        end
                        #lougt[1,1] = Legend(fig, ax1; framevisible=true, rowgap=8, merge=true)
                        #
                        #colsize!(lout, 3, Relative(200/1200))
                        for ax = (ax1,axs_mcseps...,ax2)
                            ylims!(ax, 1.1*levels[i_thresh_cquantile]-0.1*levels[i_thresh_cquantile+1], 1.1*levels[end]-0.1*levels[end-1])
                        end
                        for ax = axs_mcseps
                            xlims!(ax, 1/(time_valid_dns_ph*sdm.tu*10), thresh_cquantile*1.1)
                        end
                        xlims!(ax2, 1/10, 10)
                        xlims!(ax3, 0.5, Nmcs2mix+1.5)

                        ax1.ylabelvisible = ax1.yticklabelsvisible = true
                        for i_col = 1:Nmcs2mix
                            colgap!(lout, i_col, 0)
                        end

                        rowsize!(lout, 1, Relative(4/5))
                        rowgap!(lout, 1, 0)

                        save(joinpath(figdir,"ccdfmixs_$(dst)_$(rsp)_$(fdivname)_$(i_scl)_accpa$(Int(adjust_ccdf_per_ancestor)).png"), fig)
                    end
                end
                
            end
        end
    end

    if todo["plot_composite_contours"]
        dst = "b"
        rsp = "e"
        mc = "ent"
        i_mcval = 1
        i_scl = 12
        (
         levels,levels_mid,
         dsts,rsps,mixobjs,distn_scales,
         ccdfs,pdfs,
         fdivs,fdivs_ancgen_valid,
         mixcrits,iltmixs,
         ccdfmixs,pdfmixs,
        ) = (JLD2.jldopen(joinpath(resultdir,"ccdfs_combined.jld2"),"r") do f
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
        # compute weights on descendants
        i_mode_sf = 1
        support_radius = pertop.sf_pert_amplitudes_max[i_mode_sf]
        desc_weights = QG2L.bump_density(coast.pert_seq_qmc[:,1:cfg.num_perts_max]', distn_scales[dst][i_scl], support_radius)
        for i_anc = idx_anc_strat_ent
            i_leadtime = round(Int, Nleadtime*2/5) 
            figfile = joinpath(figdir,"contours_anc$(i_anc).png")
            plot_contours_1family(coast, ens, i_anc, desc_weights, i_leadtime, cfg, thresh, sdm, cop, pertop, contour_dispersion_filename, figfile)
        end
    end


    if todo["plot_contour_divergence"]
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
        for i_anc = idx_anc_strat_ent
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


    
    if todo["mixture_COAST_phase_diagram"]
        # TODO upgrade
        ytgtstr = @sprintf("%.2f", cfg.target_yPerL*sdm.Ly)
        rxystr = @sprintf("%.3f", cfg.target_ryPerL*sdm.Ly)
        # ------------- Lead-time parameterized ---------------
        # heat map of TV as a function of lead time and scale
        todosub = Dict{String,Bool}(
                                    "mcmean_heatmap" =>             1,
                                    "fdiv_heatmap" =>               1,
                                   )
        (
         leadtimes,r2threshes,dsts,rsps,mixobjs,
         mixcrit_labels,mixobj_labels,mixcrit_colors,distn_scales,
         fdivnames,Nboot,ccdf_levels,
         time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,
         i_thresh_cquantile,adjust_ccdf_per_ancestor
        ) = expt_config_COAST_analysis(cfg,pertop)
        fdivs,fdivs_ancgen_valid,iltmixs,iltcounts,mixcrits = JLD2.jldopen(joinpath(resultdir,"ccdfs_combined.jld2"),"r") do f
            return f["fdivs"],f["fdivs_ancgen_valid"],f["iltmixs"],f["iltcounts"],f["mixcrits"]
        end
        fdivs2plot = ["chi2","kl","qrmse",] #"tv","chi2","kl"]
        fdivlabels = ["χ²","KL","𝐿²",] #"TV","χ²","KL"]
        Nmcs = Dict(mc=>length(mixobjs[mc]) for mc=keys(mixobjs))
        i_boot = 1
        est = "mix"
        mc = "ent"
        for dst = ["b"]
            Nscales = length(distn_scales[dst])
            for rsp = ["z","2","e"]
                # Plot the entropy as a 2D phase plot: both its mean and its variance (not just the proportion of time it's optimal)
                if todosub["mcmean_heatmap"]
                    # Heatmap of various mixing criteria as a function of AST; below, invert these functions it
                    for mc = ["contcorr","globcorr","ei","ent","pim","pth"]
                        fig = Figure(size=(500,400))
                        loutmean = fig[1,1] = GridLayout()
                        axmean = Axis(loutmean[1,1], xlabel="−AST", ylabel="Scale", title="Mean $(mixcrit_labels[mc]), $(label_target(cfg, sdm))", xlabelsize=16, ylabelsize=16, titlesize=16, titlefont=:regular)
                        # account for NaNs 
                        nanmean(arr) = SB.mean(filter(!isnan, arr)) 
                        nanstd(arr) = SB.std(filter(!isnan, arr))
                        mcmean = mapslices(nanmean, mixcrits[dst][rsp][mc]; dims=2)[:,1,:]
                        mcstd = mapslices(nanstd, mixcrits[dst][rsp][mc]; dims=2)[:,1,:]
                        if 1 == Nanc
                            mcstd .= 0
                        end
                        hmmean = heatmap!(axmean, -sdm.tu.*leadtimes, distn_scales[dst], mcmean; colormap=Reverse(:deep), colorscale=identity)
                        cbarmean = Colorbar(loutmean[1,2], hmmean, vertical=true)
                        # Plot all the arg-maxima
                        if mc in ["ent","ei"]
                            for (i_scl,scl) in enumerate(distn_scales[dst])
                                scatter!(axmean, -sdm.tu.*leadtimes, scl.*ones(Nleadtime); color=:black, marker='O', markersize=60 .* iltcounts[dst][rsp][mc][1,:,i_scl] ./ Nanc)
                            end
                        end
                        save(joinpath(figdir,"phdgm_$(dst)_$(rsp)_$(mc)_accpa$(Int(adjust_ccdf_per_ancestor)).png"), fig)
                    end
                end

                if todosub["fdiv_heatmap"]
                    for (i_fdivname,fdivname) in enumerate(fdivs2plot)
                        # TODO put a circle on each grid cell, with the size proportional to the fraction of ancestors at which that lead time was the optimizer of entropy 

                        # globcorr and contcorr as the independent variable
                        for corrkey = ["globcorr","contcorr"]
                            fig = Figure(size=(500,400))
                            lout = fig[1,1] = GridLayout()
                            ax = Axis(lout[1,1], xlabel=@sprintf("σ⁻¹(%s)",mixcrit_labels[corrkey]), ylabel="Scale", title="$(fdivlabels[i_fdivname]), $(label_target(cfg,sdm))", xlabelsize=16, ylabelsize=16, titlesize=16, titlefont=:regular) 
                            Ncorr = length(mixobjs[corrkey])
                            Nscale = length(distn_scales[dst])
                            ltmean = zeros(Float64, (Ncorr, Nscale))
                            for i_corr = 1:Ncorr
                                for i_scl = 1:Nscale
                                    ltmean[i_corr,i_scl] = SB.mean(leadtimes[iltmixs[dst][rsp][corrkey][i_corr,:,i_scl]])
                                end
                            end
                            hm = heatmap!(ax, transcorr.(mixobjs[corrkey]), distn_scales[dst], fdivs[dst][rsp][corrkey][est][fdivname][i_boot,:,:]; colormap=:deep, colorscale=log10)
                            cbar = Colorbar(lout[1,2], hm, vertical=true)
                            co = contour!(ax, transcorr.(mixobjs[corrkey]), distn_scales[dst], sdm.tu.*ltmean; levels=sdm.tu.*round.(Int, leadtimes[1:round(Int,Nleadtime/7):Nleadtime]), color=:black, labels=true)
                            save(joinpath(figdir,"phdgm_$(dst)_$(rsp)_$(fdivname)_syn$(corrkey)_$(est).png"), fig)
                        end
                        # --------------- AST as the independent variable ------------
                        fig = Figure(size=(500,400))
                        lout = fig[1,1] = GridLayout()
                        ax = Axis(lout[1,1], xlabel="−AST", ylabel="Scale", title="$(fdivlabels[i_fdivname]), $(label_target(cfg,sdm))", xlabelsize=16, ylabelsize=16, titlesize=16, titlefont=:regular) 
                        hm = heatmap!(ax, reverse(-sdm.tu.*leadtimes; dims=1), distn_scales[dst], reverse(fdivs[dst][rsp]["lt"][est][fdivname][i_boot,:,:]; dims=1); colormap=:deep, colorscale=log10)
                        levels_contcorr = transcorr.(mixobjs["contcorr"][1:round(Int,Nmcs["contcorr"]/7):Nmcs["contcorr"]])
                        co = contour!(ax, -sdm.tu.*reverse(leadtimes), distn_scales[dst], reverse(SB.mean(transcorr.(mixcrits[dst][rsp]["contcorr"]); dims=2)[1:Nleadtime,1,1:Nscales]; dims=1); levels=levels_contcorr, color=:black, labels=true)
                        cbar = Colorbar(lout[1,2], hm, vertical=true)
                        levels_pth = collect(range(mixobjs["pth"][[1,end]]..., length=12))
                        levels_pim = collect(range(mixobjs["pim"][[1,end]]..., length=12))
                        #co = contour!(ax, reverse(-sdm.tu.*leadtimes; dims=1), distn_scales[dst], reverse(SB.mean(mixcrits[dst][rsp]["pth"]; dims=2)[:,1,:]; dims=1); levels=levels_pth, color=:gray, linewidth=2, alpha=0.25, labels=true)
                        #co = contour!(ax, reverse(-sdm.tu.*leadtimes; dims=1), distn_scales[dst], reverse(SB.mean(mixcrits[dst][rsp]["pim"]; dims=2)[:,1,:]; dims=1); levels=levels_pim, color=:black, linestyle=(:dash,:dense), labels=true)
                        # Plot the optimal split time (according to fdiv) as a function of scale 
                        save(joinpath(figdir,"phdgm_$(dst)_$(rsp)_$(fdivname)_synchron_$(est).png"), fig)
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
        idx_expt = [2,]
    elseif "COAST" == all_procedures[i_proc]
        idx_expt = (vec([9,15] .+ [0,1]'.*23))[1:1]
        #idx_expt = [31]
    end
end

php,sdm = QG2L.expt_config()
phpstr = QG2L.strrep_PhysicalParams(php)
sdmstr = QG2L.strrep_SpaceDomain(sdm)
computer = "engaging"
if computer == "engaging"
    expt_supdir_dns = "/orcd/archive/pog/001/ju26596/COAST/QG2L/2025-02-06/0/$(phpstr)_$(sdmstr)"
    expt_supdir_COAST = "/orcd/archive/pog/001/ju26596/COAST/QG2L/2025-02-06/0/$(phpstr)_$(sdmstr)"
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
        COAST_procedure(ensdir_dns, resultdir_dns, expt_supdir_COAST; i_expt=i_expt, overwrite_expt_setup=false, overwrite_ensemble=false, old_path_part="net/bstor002.ib",new_path_part="orcd/archive")
    end
    println()
end
