include("./QG2L.jl")
include("./EnsembleMod.jl")

import .QG2L as QG2L
import .EnsembleMod as EM

import Printf
using Printf: @sprintf
using Infiltrator: @infiltrate
import Random
import StatsBase as SB
import Distributions as Dists
import JLD2

using CairoMakie

struct ConfigDNS
    duration_spinup::Int64
    duration_spinon::Int64 # The normal chunk size 
    num_chunks_max::Int64
end

function ConfigDNS(;
        duration_spinup_ph::Float64 = 500, 
        duration_spinon_ph::Float64 = 1000, 
        num_chunks_max::Int64 = 2, 
        tu::Float64 = 1.0,
    )
    duration_spinup = round(Int, duration_spinup_ph/tu)
    duration_spinon = round(Int, duration_spinon_ph/tu)
    return ConfigDNS(duration_spinup, duration_spinon, num_chunks_max)
end

    
function extend_dns!(ens::EM.Ensemble, rng::Random.AbstractRNG, ensdir::String, cfg::ConfigDNS, cop::QG2L.ConstantOperators, pertop::QG2L.PerturbationOperator, sdm::QG2L.SpaceDomain, php::QG2L.PhysicalParams)
    Nmem = EM.get_Nmem(ens)
    child = Nmem+1
    memdir = joinpath(ensdir, "mem$(child)")
    mkpath(memdir)
    if Nmem == 0
        sf_init = QG2L.initialize_FlowField_random_unstable(sdm, cop, rng)
        conc_init = zeros(Float64, (sdm.Nx, sdm.Ny, 2))
        tphinit = 0.0
        tinit = round(Int, tphinit/sdm.tu)
        tfin = tinit + cfg.duration_spinup
        flow_init = QG2L.FlowState(tphinit, sf_init, conc_init)
        init_cond_file = joinpath(memdir, "init_cond_coldstart.jld2")
        QG2L.write_state(flow_init, init_cond_file)
    else
        parent = Nmem
        init_cond_file = ens.trajs[parent].term_cond
        tinit = ens.trajs[parent].tfin
        tfin = tinit + cfg.duration_spinon
    end

    term_cond_file = joinpath(memdir, "term_cond.jld2")
    pert_seq_filename = joinpath(memdir, "pert_seq.jld2")
    pert_seq = QG2L.NullPerturbationSequence()
    QG2L.write_perturbation_sequence(pert_seq, pert_seq_filename)
    history_file = joinpath(memdir, "history.jld2")
    flow_init = QG2L.read_state(init_cond_file)
    # ---- Integrate ----- 
    flow_fin,sf_hist,sf_hist_the,conc_hist = QG2L.integrate(flow_init,tfin,pert_seq,pertop,cop,sdm,php; nonlinear = true, verbose = true)
    # ----------
    QG2L.write_state(flow_fin, term_cond_file)
    QG2L.write_history(sf_hist, conc_hist, history_file)
    traj = EM.Trajectory(flow_init.tph, tfin, init_cond_file, pert_seq_filename, term_cond_file, history_file)
    EM.add_trajectory!(ens, traj)
    return
end

function direct_numerical_simulation_procedure(; i_expt=nothing, overwrite_expt_setup=false, overwrite_ensemble=false)
    todo = Dict(
                "integrate" =>                      0,
                "compute_moments" =>                0,
                "plot_moment_map" =>                0,
                "compute_global_histograms" =>      0,
                "compute_extrema" =>                0,
                "plot_energy" =>                    0,
                "plot_hovmoller" =>                 1, 
                "animate" =>                        0,
                # ---------- defunct ---------
                "compute_rough_quantiles" =>        0,
                "compute_local_GPD_params" =>       0,
                "plot_GPD_param_map" =>             0,
               )
    php,sdm = QG2L.expt_config(i_expt=i_expt)
    cfg = ConfigDNS(; duration_spinup_ph=500.0, duration_spinon_ph=1000.0, num_chunks_max=28)

    # ------------------ Set up save-out place --------------
    phpstr = QG2L.strrep_PhysicalParams(php)
    sdmstr = QG2L.strrep_SpaceDomain(sdm)
    computer = "engaging"
    if computer == "engaging"
        #savedir = "/orcd/archive/pog/001/ju26596/COAST/QG2L/2024-12-29/0/$(phpstr)_$(sdmstr)/DNS"
        savedir = "/orcd/archive/pog/001/ju26596/COAST/QG2L/2025-02-06/0/$(phpstr)_$(sdmstr)/DNS"
    else
        savedir = "/Users/justinfinkel/Documents/postdoc_mit/computing/tracer_extremes_resuls/2024-10-22/0/$(phpstr)_$(sdmstr)/DNS"
    end
    mkpath(savedir)
    @show savedir
    
    paramfile = joinpath(savedir,"expt_setup.jld2")
    if isfile(paramfile) && (!overwrite_expt_setup)
        cop,pertop = JLD2.jldopen(paramfile, "r") do f; return f["cop"], f["pertop"]; end
    else
        cop,pertop = QG2L.expt_setup(php, sdm)
        JLD2.jldopen(paramfile,"w") do f
            f["sdm"] = sdm
            f["php"] = php
            f["cop"] = cop
            f["pertop"] = pertop
        end
    end

    ensdir = joinpath(savedir,"ensemble_data")
    mkpath(ensdir)
    ensfile = joinpath(ensdir, "ens.jld2")
    resultdir = joinpath(savedir,"analysis")
    mkpath(resultdir)
    figdir = joinpath(savedir,"figures")
    mkpath(figdir)

    

    if todo["integrate"] == 1
        if isfile(ensfile) && (!overwrite_ensemble)
            ens = EM.load_Ensemble(ensfile)
        else
            ens = EM.Ensemble()
            EM.save_Ensemble(ens, ensfile)
        end
        rng = Random.MersenneTwister(3718)
        while EM.get_Nmem(ens) < cfg.num_chunks_max
            println("About to extend Ensemble to $(EM.get_Nmem(ens)+1) members")
            extend_dns!(ens, rng, ensdir, cfg, cop, pertop, sdm, php)
            EM.save_Ensemble(ens, ensfile)
        end
    else
        ens = EM.load_Ensemble(ensfile)
    end

    obs_funs = Dict(
                    "sf" => (args...) -> QG2L.obs_fun_sf_hist(args..., sdm, cop),
                    "temp" => (args...) -> QG2L.obs_fun_temperature_hist(args..., sdm, cop),
                    "conc" => (args...) -> QG2L.obs_fun_conc_hist(args..., sdm, cop),
                    "heatflux" => (args...) -> QG2L.obs_fun_heatflux_hist(args..., sdm, cop),
                    "eke" => (args...) -> QG2L.obs_fun_eke_hist(args..., sdm, cop),
                    "pv" => (args...) -> QG2L.obs_fun_pv_hist(args..., sdm, cop) .+ php.beta.*sdm.ygrid',
                    "u" => (args...) -> QG2L.obs_fun_zonal_velocity_hist(args..., sdm, cop),
                   )
    obs_labels = Dict(
                      "sf" => "ψ",
                      "temp" => "𝑇",
                      "conc" => "𝑐",
                      "heatflux" => "𝑣'𝑇'",
                      "eke" => "(½)|∇ψ|²",
                      "pv" => "𝑃𝑉",
                      "u" => "𝑢",
                     )
    obs2plot = ["u","pv","conc","sf","eke","temp","heatflux",]
    if todo["compute_moments"] == 1
        @show EM.get_Nmem(ens)
        Nmem = EM.get_Nmem(ens)-1
        hist_filenames = [ens.trajs[mem].history for mem=2:2+Nmem-1]
        for obs_name = obs2plot
            # Concentrations 
            moments,mssk,moments_xall,mssk_xall = QG2L.compute_observable_local_moments(hist_filenames, obs_funs[obs_name], sdm)
            JLD2.jldopen(joinpath(resultdir,"moments_mssk_$(obs_name).jld2"),"w") do f
                f["moments"] = moments
                f["mssk"] = mssk
                f["moments_xall"] = moments_xall
                f["mssk_xall"] = mssk_xall
                return
            end
        end
    end

    if todo["plot_moment_map"] == 1
        # define a local observable 
        # compute its statistics, both locally and globally, by reading in the files sequentially and building up location-dependent histograms (or maybe just moments)
        # Examine the maps and manually select some local regions to make local PDFs 
        for obs_name = obs2plot
            moments,mssk = JLD2.jldopen(joinpath(resultdir,"moments_mssk_$(obs_name).jld2"),"r") do f
                return (f["moments"],f["mssk"])
            end

            fig = Figure(size=(1600,800))
            mom_names = [L"\mathrm{mean}",L"\mathrm{standard\ deviation}",L"\mathrm{skewness}",L"\mathrm{excess\ kurtosis}"]
            theta_pivots = [0.0, 0.0, 0.0, 0.0]
            for i_mom = 1:4
                for iz = 1:2
                    sublout = fig[iz,i_mom] = GridLayout()
                    QG2L.plot_summary_statistic!(sublout, mssk[:,:,iz,i_mom], cop, sdm; theta_pivot=theta_pivots[i_mom], title=L"Layer %$(iz) %$(mom_names[i_mom])")
                end
            end
            save(joinpath(resultdir, "moments_$(obs_name).png"), fig)
        end
    end
    if todo["compute_global_histograms"] == 1
        for obs_name = obs2plot
            moments,mssk = JLD2.jldopen(joinpath(resultdir,"moments_mssk_$(obs_name).jld2"),"r") do f
                return (f["moments"],f["mssk"])
            end
            Nmem = min(EM.get_Nmem(ens)-1, 49)
            hist_filenames = [ens.trajs[mem].history for mem=2:2+Nmem-1]
            bin_width = 0.025 * SB.mean(mssk[:,:,:,2])
            bins1,counts1,bins2,counts2 = QG2L.compute_layerwise_histograms(hist_filenames, obs_funs[obs_name], sdm, bin_width)
            JLD2.jldopen(joinpath(resultdir,"$(obs_name)_hists.jld2"),"w") do f
                f["bins1"] = bins1
                f["counts1"] = counts1
                f["bins2"] = bins2
                f["counts2"] = counts2
            end
            # Fit gaussian or exponential distributions
            if (min(minimum(bins1),minimum(bins2)) < 0) && (max(maximum(bins1),maximum(bins2)) > 0)
                mu1 = SB.mean(moments[:,:,1,2])
                sigma1 = sqrt(SB.mean(moments[:,:,1,3]) - mu1^2)
                mu2 = SB.mean(moments[:,:,2,2])
                sigma2 = sqrt(SB.mean(moments[:,:,2,3]) - mu2^2)
                G1 = Dists.Normal(mu1, sigma1)
                G2 = Dists.Normal(mu2, sigma2)
            elseif min(minimum(bins1),minimum(bins2)) >= 0
                scale1 = SB.mean(moments[:,:,1,2])
                scale2 = SB.mean(moments[:,:,2,2])
                G1 = Dists.Exponential(scale1)
                G2 = Dists.Exponential(scale2)
            end
            G1ccdf = Dists.ccdf.(G1, bins1.*bin_width)
            G2ccdf = Dists.ccdf.(G2, bins2.*bin_width)

            fig = Figure(size=(400,800))
            lout = fig[1:2,1:2] = GridLayout()

            ax1 = Axis(lout[1,1], xlabel=obs_labels[obs_name], ylabel=L"CCDF", title="Layer 1", yscale=log10)
            pmf_emp1 = (counts1/sum(counts1))
            ccdf_emp1 = reverse(cumsum(reverse(pmf_emp1)))
            lines!(ax1, (bins1 .+ 0.5)*bin_width, ccdf_emp1, color=:black)
            lines!(ax1, (bins1 .+ 0.5)*bin_width, G1ccdf, color=:black, linestyle=:dash)
            #ylims!(ax1, extrema(filter(!isnan, ccdf_emp1))...)

            ax2 = Axis(lout[2,1], xlabel=obs_labels[obs_name], ylabel="CCDF", title="Layer 2", yscale=log10)
            pmf_emp2 = (counts2/sum(counts2))
            ccdf_emp2 = reverse(cumsum(reverse(pmf_emp2)))
            lines!(ax2, (bins2 .+ 0.5)*bin_width, ccdf_emp2, color=:black)
            lines!(ax2, (bins2 .+ 0.5)*bin_width, G2ccdf, color=:black, linestyle=:dash)
            #ylims!(ax2, extrema(filter(!isnan, ccdf_emp2))...)

            save(joinpath(figdir, "global_ccdfs_$(obs_name).png"), fig)
        end
    end

    
    if todo["compute_extrema"] == 1
        Nmem = min(EM.get_Nmem(ens)-1, 49)
        hist_filenames = [ens.trajs[mem].history for mem=2:2+Nmem-1]
        for obs_name = obs2plot
            println("\ncomputing extrema for $(obs_name)")
            min_vals,max_vals = QG2L.compute_observable_local_extrema(hist_filenames, obs_funs[obs_name], sdm)
            JLD2.jldopen(joinpath(resultdir,"extreme_vals_$(obs_name).jld2"),"w") do f
                f["min_vals"] = min_vals
                f["max_vals"] = max_vals
                return
            end
        end
    end


    if todo["plot_energy"] == 1
        Nmem = min(EM.get_Nmem(ens)-1, 49)
        hist_filenames = [ens.trajs[mem].history for mem=2:2+Nmem-1]
        tgrid = reduce(vcat, (floor(Int, ens.trajs[mem].tphinit/sdm.tu)+1:ens.trajs[mem].tfin for mem=2:2+Nmem-1))
        @show size(tgrid)
        # Plot energy over time 
        energy_fun = (f -> QG2L.obs_fun_mean_energy_density(f, cop, sdm))
        energy = cat((QG2L.histfile2obs(f, energy_fun) for f in hist_filenames)...; dims=4)
        #tgrid = (1:1:size(energy,4))*sdm.tu # TODO correct the time gridding up front
        @show size(energy)
        fig = Figure(size=(1200,800))
        lout = fig[1:2,1:1] = GridLayout()
        for iz = 1:2
            ax = Axis(lout[iz,1], xlabel="time", ylabel="Layer $(iz) energy density", yscale=identity, title="Eddy kinetic energy")
            lines!(ax, tgrid*sdm.tu, vec(energy[:,:,iz,:]), color=:red)
        end
        save(joinpath(figdir,"energy.png"), fig)
        # TODO plot spectrum
    end

    if todo["compute_rough_quantiles"] == 1
        # Compute a few high quantiles as a standard to use across all following extreme value analysis
        Nmem = EM.get_Nmem(ens)
        tspan_phys = [500.0, 2500.0]
        tphinits = [ens.trajs[mem].tphinit for mem=1:Nmem]
        memfirst = findfirst(tphinits .>= tspan_phys[1])
        memlast = findfirst(tphinits .>= tspan_phys[2]) - 1
        hist_filenames = [ens.trajs[mem].history for mem=memfirst:memlast]
        thresh_quants = 1 .- 1 ./ [2, 4, 8, 16, 32, 64]
        Nthresh = length(thresh_quants)
        for obs_name = obs2plot

            obs_val = cat((QG2L.histfile2obs(filename, obs_funs[obs_name]) for filename=hist_filenames)..., dims=4)
            threshes = mapslices(u->SB.quantile(u, thresh_quants), obs_val; dims=4)
            threshes_zm = fill(0.0, (1,sdm.Ny,2,Nthresh))
            for iy = 1:sdm.Ny
                for iz = 1:2
                    threshes_zm[1,iy,iz,:] .= SB.quantile(vec(obs_val[:,iy,iz,:]), thresh_quants)
                end
            end
            @show size(threshes_zm)
            @assert all(threshes_zm[:,:,:,1] .<= threshes_zm[:,:,:,2])
            JLD2.jldopen(joinpath(resultdir,"threshes_$(obs_name).jld2"), "w") do f
                f["threshes"] = threshes
                f["thresh_quants"] = thresh_quants
                f["threshes_zonalmean"] = threshes_zm
            end
        end
    end




    if todo["compute_local_GPD_params"] == 1
        # We need some efficient method to read in data...
        Nmem4locgpd = EM.get_Nmem(ens)-1
        @show Nmem
        hist_filenames = [ens.trajs[mem].history for mem=2:2+Nmem4locgpd-1]
        for obs_name = obs2plot
            println("------------ GPD parameters for $(obs_name) ---------------")
            # TODO decluster separately on each timeseries...kind of a hassle, and maybe not appropriate anyway since neighboring sites are correlated. 
            threshes,threshes_zm,thresh_quants = JLD2.jldopen(joinpath(resultdir,"threshes_$(obs_name).jld2"), "r") do f
                return f["threshes"],f["threshes_zonalmean"],f["thresh_quants"]
            end
            println("About to call QG2L.compute_local_GPD_params...")
            thresh_lo_map,thresh_hi_map = (threshes[:,:,:,i] for i=1:2)
            @show size(thresh_lo_map),size(thresh_hi_map)
            prebuffer,postbuffer,initbuffer = (round(Int,30/sdm.tu) for _=1:3)
            scale,shape = QG2L.compute_local_GPD_params(hist_filenames, obs_funs[obs_name], thresh_hi_map, prebuffer, postbuffer, initbuffer)
            @show size(scale),size(shape)
            println("done calling QG2L.compute_local_GPD_params...")
            JLD2.jldopen(joinpath(resultdir,"gpd_params_$(obs_name).jld2"),"w") do f
                f["scale"] = scale
                f["shape"] = shape
            end
            # same thing but zonally aggregated 
            thresh_lo_zm_map,thresh_hi_zm_map = (threshes_zm[:,:,:,i] for i=1:2)
            function obs_fun_zm(filename)
                obs_val = obs_funs[obs_name](filename)
                Nt = size(obs_val,4)
                obs_val_stacked = zeros((1,sdm.Ny,2,sdm.Nx*Nt))
                for iy = 1:sdm.Ny
                    for iz = 1:2
                        obs_val_stacked[1,iy,iz,:] .= vec(obs_val[:,iy,iz,:]')
                    end
                end
                return obs_val_stacked
            end
            println("For zonal agg: about to call QG2L.compute_local_GPD_params...")
            scale,shape = QG2L.compute_local_GPD_params(hist_filenames, obs_fun_zm, thresh_hi_zm_map, thresh_lo_zm_map)
            println("For zonal agg; Finished calling QG2L.compute_local_GPD_params...")
            JLD2.jldopen(joinpath(resultdir,"gpd_params_$(obs_name)_zm.jld2"),"w") do f
                f["scale"] = scale
                f["shape"] = shape
            end
        end
    end


    if todo["plot_GPD_param_map"] == 1
        for obs_name = obs2plot
            threshes,threshes_zm,thresh_quants = JLD2.jldopen(joinpath(resultdir,"threshes_$(obs_name).jld2"), "r") do f
                return f["threshes"],f["threshes_zonalmean"],f["thresh_quants"]
            end
            params_gpd = JLD2.jldopen(joinpath(resultdir,"gpd_params_$(obs_name).jld2"),"r") do f
                return [f["scale"],f["shape"]]
            end
            @show size(params_gpd[1]),size(params_gpd[2]),size(threshes)
            pushfirst!(params_gpd, threshes[:,:,:,2])
            params_gpd_zm = JLD2.jldopen(joinpath(resultdir,"gpd_params_$(obs_name)_zm.jld2"),"r") do f
                return [f["scale"],f["shape"]]
            end
            pushfirst!(params_gpd_zm, threshes_zm[:,:,:,2])
            fig = Figure(size=(3*480,800))
            thqstr = @sprintf("1/%d",round(Int, 1-1/thresh_quants[2]))
            par_names = [L"threshold $u=$CCDF(%$(thqstr))",L"scale $\sigma$",L"shape $\xi$"]
            theta_pivots = [nothing, 0.0, 0.0]
            for i_param = 1:3
                for iz = 1:2
                    sublout = fig[iz,i_param] = GridLayout()
                    @show size(params_gpd[i_param])
                    QG2L.plot_summary_statistic!(sublout, params_gpd[i_param][:,:,iz], cop, sdm; theta_pivot=theta_pivots[i_param], title=L"Layer %$(iz) %$(par_names[i_param])")
                    lines!(sublout[1,2], vec(params_gpd_zm[i_param][1,:,iz]), sdm.ygrid, color=:blue)
                end
            end
            save(joinpath(figdir,"param_map_gpd_$(obs_name).png"), fig)
            # Plot theoretical maximum according to GPD
            min_vals,max_vals = JLD2.jldopen(joinpath(resultdir,"extreme_vals_$(obs_name).jld2")) do f
                return (f["min_vals"], f["max_vals"])
            end
            gpd_max_fun(thresh,scale,shape) = (shape >= 0 ? Inf : thresh - scale / shape)
            gpd_max_vals = gpd_max_fun.(params_gpd...)
            gpd_zm_max_vals = gpd_max_fun.(params_gpd_zm...)
            fig = Figure(size=(3*480,800))
            for iz = 1:2
                sublout = fig[iz,1] = GridLayout()
                QG2L.plot_summary_statistic!(sublout, max_vals[:,:,iz], cop, sdm; title=L"Layer $%$(iz)$ empirical maxima")
                lines!(sublout[1,2], vec(maximum(max_vals[:,:,iz], dims=1)[1,:]), sdm.ygrid, color=:blue)
                sublout = fig[iz,2] = GridLayout()
                QG2L.plot_summary_statistic!(sublout, gpd_max_vals[:,:,iz], cop, sdm; title=L"Layer $%$(iz)$ maxima according to GPD")
                lines!(sublout[1,2], gpd_zm_max_vals[1,:,iz], sdm.ygrid, color=:blue)
            end
            save(joinpath(figdir,"max_val_map_$(obs_name)_gpd.png"), fig)
        end
    end

    if todo["plot_hovmoller"] == 1
        Nmem = EM.get_Nmem(ens)
        tfins = [ens.trajs[mem].tfin for mem=1:Nmem]
        hovmoller_timespan = cfg.duration_spinup .+ round.(Int, [0.0, 500.0]./sdm.tu)
        if computer == "engaging"
            hovmoller_timespan = round.(Int, [3200.0,3600.0]./sdm.tu)
        end
        stats_timespan = [cfg.duration_spinup+1, tfins[end]]
        @show tfins
        @show cfg.duration_spinup
        memfirst_hov = findfirst(tfins .> hovmoller_timespan[1])
        memlast_hov = findfirst(tfins .>= hovmoller_timespan[2])
        if isnothing(memlast_hov)
            memlast_hov = Nmem
        end
        tinits_hov = tfins[(memfirst_hov-1):(memlast_hov-1)]
        tidx_hov = (hovmoller_timespan[1]-tinits_hov[1]+1):(hovmoller_timespan[2]-tinits_hov[1]+1)
        hist_filenames_hov = [ens.trajs[mem].history for mem=memfirst_hov:memlast_hov]
        memfirst_stats = findfirst(tfins .>= stats_timespan[1]) + 1
        memlast_stats = findfirst(tfins .>= stats_timespan[2])
        if isnothing(memlast_stats)
            memlast_stats = Nmem
        end
        @show memlast_stats
        hist_filenames_stats = [ens.trajs[mem].history for mem=memfirst_stats:memlast_stats]

        # always put in streamfunction as contours
        zonal_mean_sf_fun(hist_filename) = SB.mean(QG2L.obs_fun_sf_hist(hist_filename, sdm, cop); dims=1)
        sf_zm_hov = cat(QG2L.compute_observable_ensemble(hist_filenames_hov, zonal_mean_sf_fun)..., dims=4)[:,:,:,tidx_hov]

        sf_mssk_xall = JLD2.jldopen(joinpath(resultdir,"moments_mssk_sf.jld2"),"r") do f
            return f["mssk_xall"]
        end

        Nsnap = 5
        tgrid = range(hovmoller_timespan[1], hovmoller_timespan[2]; step=1) #range(tfins[memfirst_hov-1]+1, tfins[memlast_hov]; step=1)
        tidx_snap = round.(Int, range(1, length(tgrid); length=Nsnap))
        sf_2d = cat(QG2L.compute_observable_ensemble(hist_filenames_hov, obs_funs["sf"])..., dims=4)[:,:,:,tidx_hov[tidx_snap]]
        #@infiltrate
        for obs_name = obs2plot
            anomaly_cont = true
            anomaly_heat = true
            @show obs_name
            # --------- Plot several snapshots -----------
            fheat_2d = cat(QG2L.compute_observable_ensemble(hist_filenames_hov, obs_funs[obs_name])..., dims=4)[:,:,:,tidx_hov[tidx_snap]]
            @show size(fheat_2d)
            outfile_prefix = joinpath(figdir, "snapshots_$(obs_name)_sf")
            QG2L.plot_snapshots(tgrid[tidx_snap], fheat_2d, sf_2d, sdm, outfile_prefix; fcont_label=obs_labels["sf"], fheat_label=obs_labels[obs_name])

            # Compute first four moments
            zonal_mean_obs_fun(hist_filename) = SB.mean(obs_funs[obs_name](hist_filename), dims=1)
            obs_val_zm_hov = cat(QG2L.compute_observable_ensemble(hist_filenames_hov, zonal_mean_obs_fun)..., dims=4)[:,:,:,tidx_hov]
            @show size(obs_val_zm_hov)

            # TODO instead of recompute zonal means here, read moments from precomputed moments
            obs_mssk_xall = JLD2.jldopen(joinpath(resultdir,"moments_mssk_$(obs_name).jld2"),"r") do f
                return f["mssk_xall"]
            end

            #@infiltrate
            fig = Figure(size=(1200,600))
            for iz = 1:2
                lout = fig[iz,1] = GridLayout()
                QG2L.plot_hovmoller_ydep!(lout, obs_val_zm_hov[1,:,iz,:], sf_zm_hov[1,:,iz,:], obs_mssk_xall[1,:,iz,:], sf_mssk_xall[1,:,iz,:], cop, sdm; tinit=hovmoller_timespan[1], title="Layer $(iz) zonal mean $(obs_labels["sf"]) and $(obs_labels[obs_name])", anomaly_heat=anomaly_heat, anomaly_cont=anomaly_cont)
            end
            save(joinpath(figdir,"hovmoller_$(obs_name).png"), fig)
        end
    end

    if todo["animate"] == 1
        todo_anim = Dict(
                         "conc" =>          1,
                         "sf" =>            1,
                         "eke" =>           1,
                         "heatflux" =>      1,
                         "temp" =>          1,
                         "u" =>             1,
                         "pv" =>            1,
                        )
        # --------- Animate a timespan shortly after spinup --------------
        anim_tspan_ph = [3300.0,3500.0]
        anim_tspan = round.(Int, anim_tspan_ph./sdm.tu)
        Nt = anim_tspan[2] - anim_tspan[1] + 1
        Nmem = EM.get_Nmem(ens)
        tfins = [ens.trajs[mem].tfin for mem=1:Nmem]
        memfirst = findfirst(tfins .>= anim_tspan[1])
        memlast = findfirst(tfins .>= anim_tspan[2])
        @show memfirst,memlast
        hist_filenames = [ens.trajs[mem].history for mem=memfirst:memlast]

        sf_hist = QG2L.read_sf_hist(hist_filenames)
        tgrid = sf_hist.tgrid
        conc_hist = QG2L.read_conc_hist(hist_filenames)

        sf_hist = QG2L.subsample_FlowFieldHistory(sf_hist, anim_tspan[1], anim_tspan[2])
        conc_hist = QG2L.subsample_field_history(conc_hist, tgrid, anim_tspan[1], anim_tspan[2])

        tgrid = sf_hist.tgrid
        fcont = zeros(Float64, (sdm.Nx, sdm.Ny, 2, Nt))
        fheat = zeros(Float64, (sdm.Nx, sdm.Ny, 2, Nt))
        if todo_anim["conc"] == 1
            fheat .= conc_hist 
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(figdir,"anim_sf_conc.mp4")
            QG2L.animate_fields(tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\psi", fheat_label=L"$c$")
        end
        if todo_anim["sf"] == 1
            println("starting to animate sf")
            fheat .= QG2L.obs_fun_sf_hist(tgrid, sf_hist.ox, sdm, cop) #.- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(figdir,"anim_sf_sf.mp4")
            QG2L.animate_fields(tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\psi", fheat_label=L"\psi")
        end
        if todo_anim["eke"] == 1
            println("starting to animate eke")
            fheat .= QG2L.obs_fun_eke_hist(tgrid, sf_hist.ok, sdm, cop)
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(figdir,"anim_sf_eke.mp4")
            QG2L.animate_fields(tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\psi", fheat_label=L"\frac{1}{2}|\nabla\psi|^2")
        end
        if todo_anim["temp"] == 1
            println("starting to animate temp")
            fheat .= QG2L.obs_fun_temperature_hist(tgrid, sf_hist.ok, sdm, cop) #.- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(figdir,"anim_sf_temp.mp4")
            QG2L.animate_fields(tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\psi", fheat_label=L"\psi_{BC}")
        end
        if todo_anim["heatflux"] == 1
            println("starting to animate heatflux")
            fheat .= QG2L.obs_fun_heatflux_hist(tgrid, sf_hist.ok, sdm, cop)
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(figdir,"anim_sf_heatflux.mp4")
            QG2L.animate_fields(tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\psi", fheat_label=L"v'\psi_{BC}")
        end
        if todo_anim["u"] == 1
            u_hist = QG2L.FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
            u_hist.ok .= -cop.Dy .* sf_hist.ok
            QG2L.synchronize_FlowField_k2x!(u_hist)
            fheat .= u_hist.ox
            fheat[:,:,1,:] .+= 1.0 # mean flow 
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(figdir,"anim_sf_u_mem$(Nmem).mp4")
            @show outfile
            QG2L.animate_fields(sf_hist.tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\psi", fheat_label=L"u")
        end
        if todo_anim["pv"] == 1
            println("starting to animate pv")
            pv_hist_ox = QG2L.obs_fun_pv_hist(tgrid, sf_hist.ok, sdm, cop)
            fheat .= pv_hist_ox .+ php.beta*(sdm.ygrid .- sdm.Ly/2)'
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(figdir,"anim_sf_pv.mp4")
            QG2L.animate_fields(tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\psi", fheat_label=L"q")
        end
    end
end


idx_expt = Vector{Int64}([])
if length(ARGS) > 0
    for i_arg = 1:length(ARGS)
        push!(idx_expt, parse(Int, ARGS[i_arg]))
    end
else
    #idx_expt = collect(1:18)
    idx_expt = [0] 
end
for i_expt = idx_expt
    println()
    println("------------------- Starting experiment $i_expt ----------------")
    direct_numerical_simulation_procedure(; i_expt=i_expt, overwrite_expt_setup=true, overwrite_ensemble=false)
    println()
end
