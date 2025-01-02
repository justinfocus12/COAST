include("./Panetta1993.jl")
include("./EnsembleManager.jl")

import .Panetta1993 as P93
import .EnsembleManager as EM
import Printf
import Random
import JLD2
using CairoMakie



function direct_numerical_simulation(; i_expt=nothing)
    todo = Dict(
                "integrate" =>                   1,
                "plot_spaghetti" =>              0,
                "summarize_extremes" =>          0,
                "plot_hovmoller" =>              0,
                "plot_tracer_fluctuations" =>    0,
                "plot_tracer_variance" =>        0,
                "plot_energy" =>                 0,
                "animate" =>                     0,
               )
    sdm,php,cop,pertop = expt_setup(i_expt=i_expt)
    # Find the most unstable mode 
    instab = maximum(real.(cop.evals_ctime), dims=3) .* cop.dealias_mask
    most_unstable_mode = argmax(instab)
    #println("all unstable modes: ")
    #display(instab .> 1e-10)
    tinit = 0.0
    duration_per_chunk = 100.0 
    dt_save = 1.0 

    # ------------------ Set up save-out place --------------
    phpstr = P93.strrep_PhysicalParams(php)
    sdmstr = P93.strrep_SpaceDomain(sdm)
    computer = "jflaptop"
    julia_old = (computer == "jflaptop")
    if computer == "engaging"
        savedir = "/net/bstor002.ib/pog/001/ju26596/jesus_project/TracedPanetta1993/2024-08-26/0/$(phpstr)_$(sdmstr)"
    else
        savedir = "/Users/justinfinkel/Documents/postdoc_mit/computing/tracer_extremes_results/2024-08-26/1/$(phpstr)_$(sdmstr)/DNS"
    end
    mkpath(savedir)

    ensdir = joinpath(savedir,"ensemble_data")
    mkpath(ensdir)
    ensfile = joinpath(ensdir, "ens.jld2")
    resultdir = joinpath(savedir,"analysis")
    mkpath(resultdir)

    

    if todo["integrate"] == 1

        if isfile(ensfile)
            ens = EM.load_Ensemble(ensfile)
        else
            ens = EnsembleManager.Ensemble()
        end

        @show EM.get_nmem(ens)

        rng = Random.MersenneTwister(3718)
        mems2delete = Vector{Int64}([]) #[2,3,4,5]
        EM.clear_Ensemble!(ens, mems2delete)
        @show EM.get_nmem(ens)

        num_chunks = 30

        if EM.get_nmem(ens) == 0
            # Ancestor
            memdir = joinpath(ensdir,"mem1")
            mkpath(memdir)

            sf_init = P93.initialize_FlowField_random_unstable(sdm, cop, rng)
            conc_init = P93.FlowField(sdm.Nx, sdm.Ny) 
            flow_init = P93.FlowState(tinit, sf_init, conc_init)
            pert_init = zeros(Float64, length(pertop.magnitudes))

            init_cond_file = joinpath(memdir, "init_cond.jld2")
            term_cond_file = joinpath(memdir, "term_cond.jld2")
            forcing_file = joinpath(memdir, "pert.jld2")
            history_file = joinpath(memdir, "history.jld2")
            P93.write_state(flow_init, init_cond_file)
            P93.perturb!(flow_init, pert_init, pertop)
            P93.write_perturbation(pert_init, forcing_file)
            tfin_chunk = flow_init.t + duration_per_chunk
            # ---- Integrate ----- 
            flow_fin,sf_hist,sf_hist_the,conc_hist = P93.integrate(flow_init,tfin_chunk,dt_save,cop,sdm,php; nonlinear = true)
            # ----------
            P93.write_state(flow_fin, term_cond_file)
            P93.write_history(sf_hist, conc_hist, history_file)
            traj = EM.Trajectory(flow_init.t, tfin_chunk, init_cond_file, forcing_file, term_cond_file, history_file)
            EM.add_trajectory!(ens, traj)
            EM.save_Ensemble(ens, ensfile)
        end

        # Descendants 
        parent = EM.get_nmem(ens)
        while parent < num_chunks
            child = parent + 1
            # set up directory
            memdir = joinpath(ensdir, "mem$(child)")
            mkpath(memdir)
            # ingest initialization information
            init_cond_file = ens.trajs[parent].term_cond
            flow_init = P93.read_state(ens.trajs[parent].term_cond)
            tfin_chunk = flow_init.t + duration_per_chunk
            term_cond_file = joinpath(memdir, "term_cond.jld2")
            forcing_file = joinpath(memdir, "pert.jld2")
            history_file = joinpath(memdir, "history.jld2")
            # construct forcing
            pert_init = zeros(length(pertop.magnitudes)) 
            P93.perturb!(flow_init, pert_init, pertop)
            flow_fin,sf_hist,sf_hist_the,conc_hist = P93.integrate(flow_init,tfin_chunk,dt_save,cop,sdm,php; nonlinear = true)
            P93.write_state(flow_fin, term_cond_file)
            P93.write_history(sf_hist, conc_hist, history_file)
            P93.write_perturbation(pert_init, forcing_file)
            traj = EM.Trajectory(flow_init.t, tfin_chunk, init_cond_file, forcing_file, term_cond_file, history_file)
            EM.add_trajectory!(ens, traj; parent=parent)
            EM.save_Ensemble(ens, ensfile)
            parent = child
        end
    else
        ens = EM.load_Ensemble(ensfile)
    end


       



end


function small_branching_ensemble(; i_expt=nothing)
    todo = Dict(
                "integrate" =>                   1,
                "plot_spaghetti" =>              1,
                "summarize_extremes" =>          0,
                "plot_hovmoller" =>              0,
                "plot_tracer_fluctuations" =>    0,
                "plot_tracer_variance" =>        0,
                "plot_energy" =>                 0,
                "animate" =>                     0,
               )
    sdm,php,cop,pertop = expt_setup(i_expt=i_expt)
    # Find the most unstable mode 
    instab = maximum(real.(cop.evals_ctime), dims=3) .* cop.dealias_mask
    most_unstable_mode = argmax(instab)
    #println("all unstable modes: ")
    #display(instab .> 1e-10)
    tinit = 0.0
    duration_spinup = 500.0 
    duration_spinoff = 50.0
    dt_save = 1.0 

    # ------------------ Set up save-out place --------------
    phpstr = P93.strrep_PhysicalParams(php)
    sdmstr = P93.strrep_SpaceDomain(sdm)
    computer = "jflaptop"
    julia_old = (computer == "jflaptop")
    if computer == "engaging"
        savedir = "/net/bstor002.ib/pog/001/ju26596/jesus_project/TracedPanetta1993/2024-08-26/0/$(phpstr)_$(sdmstr)"
    else
        savedir = "/Users/justinfinkel/Documents/postdoc_mit/computing/tracer_extremes_results/2024-08-26/1/$(phpstr)_$(sdmstr)"
    end
    mkpath(savedir)

    ensdir = joinpath(savedir,"ensemble_data")
    mkpath(ensdir)
    ensfile = joinpath(ensdir, "ens.jld2")
    resultdir = joinpath(savedir,"analysis")
    mkpath(resultdir)

    

    if todo["integrate"] == 1

        if isfile(ensfile)
            ens = EM.load_Ensemble(ensfile)
        else
            ens = EnsembleManager.Ensemble()
        end

        @show EM.get_nmem(ens)

        rng = Random.MersenneTwister(3718)
        mems2delete = Vector{Int64}([]) #[2,3,4,5]
        EM.clear_Ensemble!(ens, mems2delete)
        @show EM.get_nmem(ens)

        if EM.get_nmem(ens) == 0
            # Ancestor
            memdir = joinpath(ensdir,"mem1")
            mkpath(memdir)

            sf_init = P93.initialize_FlowField_random_unstable(sdm, cop, rng)
            conc_init = P93.FlowField(sdm.Nx, sdm.Ny) 
            flow_init = P93.FlowState(tinit, sf_init, conc_init)
            pert_init = zeros(Float64, length(pertop.magnitudes))

            init_cond_file = joinpath(memdir, "init_cond.jld2")
            term_cond_file = joinpath(memdir, "term_cond.jld2")
            forcing_file = joinpath(memdir, "pert.jld2")
            history_file = joinpath(memdir, "history.jld2")
            P93.write_state(flow_init, init_cond_file)
            P93.perturb!(flow_init, pert_init, pertop)
            P93.write_perturbation(pert_init, forcing_file)
            tfin_chunk = flow_init.t + duration_spinup
            # ---- Integrate ----- 
            flow_fin,sf_hist,sf_hist_the,conc_hist = P93.integrate(flow_init,tfin_chunk,dt_save,cop,sdm,php; nonlinear = true)
            # ----------
            P93.write_state(flow_fin, term_cond_file)
            P93.write_history(sf_hist, conc_hist, history_file)
            traj = EM.Trajectory(flow_init.t, tfin_chunk, init_cond_file, forcing_file, term_cond_file, history_file)
            EM.add_trajectory!(ens, traj)
            EM.save_Ensemble(ens, ensfile)
        end

        # Descendants 
        parent = 1
        for child = setdiff(2:5, 1:EM.get_nmem(ens))
            memdir = joinpath(ensdir, "mem$(child)")
            mkpath(memdir)
            init_cond_file = ens.trajs[parent].term_cond
            flow_init = P93.read_state(ens.trajs[1].term_cond)
            tfin_chunk = flow_init.t + duration_spinoff
            term_cond_file = joinpath(memdir, "term_cond.jld2")
            forcing_file = joinpath(memdir, "pert.jld2")
            history_file = joinpath(memdir, "history.jld2")
            pert_init = child == 2 ? zeros(length(pertop.magnitudes)) : Random.randn(rng, Float64, length(pertop.magnitudes))
            P93.perturb!(flow_init, pert_init, pertop)
            flow_fin,sf_hist,sf_hist_the,conc_hist = P93.integrate(flow_init,tfin_chunk,dt_save,cop,sdm,php; nonlinear = true)
            P93.write_state(flow_fin, term_cond_file)
            P93.write_history(sf_hist, conc_hist, history_file)
            traj = EM.Trajectory(flow_init.t, tfin_chunk, init_cond_file, forcing_file, term_cond_file, history_file)
            EM.add_trajectory!(ens, traj; parent=parent)
            EM.save_Ensemble(ens, ensfile)
        end
    else
        ens = EM.load_Ensemble(ensfile)
    end

    if todo["plot_spaghetti"] == 1
        ix = div(sdm.Nx,2)
        iy = div(sdm.Ny,2)
        dx,dy = sdm.Lx/sdm.Nx,sdm.Ly/sdm.Ny
        x_str = Printf.@sprintf("%.1f", sdm.xgrid[ix]+dx/2)
        y_str = Printf.@sprintf("%.1f", sdm.ygrid[iy]+dy/2)
        obs_fun(sf_hist, conc_hist) = conc_hist.ox[ix,iy,1,:] # Concentration in middle of domain
        Nmem = EM.get_nmem(ens)
        hist_filenames = [ens.trajs[mem].history for mem = 1:Nmem]
        obs_vals = P93.compute_observable_ensemble(hist_filenames, obs_fun)
        fig = Figure(size=(600,200))
        lout = fig[1:1,1:1] = GridLayout()
        ax = Axis(lout[1,1], xlabel="time", ylabel=L"c(%$(x_str),%$(y_str))")
        for mem = 1:Nmem
            tmem = collect(range(ens.trajs[mem].t_init, ens.trajs[mem].t_fin, step=dt_save)[2:end])
            lines!(ax, tmem, obs_vals[mem]) #-obs_vals[2])
            @show obs_vals[mem][1:10]
        end
        xlims!(ax, [400,550])
        file_tail = replace("spaghetti_x$(x_str)_y$(y_str)", "."=>"p") * ".png"
        save(joinpath(resultdir, file_tail), fig)
    end


    if todo["summarize_extremes"] == 1
        # evaluate the local concetrations as a long vector and estimate extreme value statistics
        obsfun(sf_hist, conc_hist) = conc_hist[1,:,:]
        # TODO finish
    end


    if todo["plot_hovmoller"] == 1
        # Zonal velocity
        u_hist = P93.FlowFieldHistory(sf_hist.tgrid, sdm.Nx, sdm.Ny)
        u_hist.ok .= -cop.Dy .* sf_hist.ok
        P93.synchronize_FlowField_k2x!(u_hist)
        outfile = joinpath(savedir,"hov_u.png")
        P93.plot_hovmoller(sf_hist.tgrid, u_hist.ox, sdm, outfile; flabel=L"\mathrm{Zonal velocity}", julia_old=julia_old)
        # Tracer concentration
        outfile = joinpath(savedir,"hov_conc.png")
        P93.plot_hovmoller(conc_hist.tgrid, conc_hist.ox, sdm, outfile; flabel=L"\mathrm{Concentration}", julia_old=julia_old)
    end

    if todo["plot_tracer_fluctuations"] == 1
        duration = sf_hist.tgrid[end] - sf_hist.tgrid[1]
        if duration > 600
            tidxfirst = findfirst(sf_hist.tgrid .> 300)
        else
            tidxfirst = findfirst(sf_hist.tgrid .> sf_hist.tgrid[end]-200)
        end
        # Plot quantiles with latitude
        for iz = 1:2
            outfile = joinpath(savedir, "quants_conc_layer$(iz).png")
            P93.plot_quantiles_latdep(sf_hist.tgrid[tidxfirst:end], conc_hist.ox[:,:,iz,tidxfirst:end], sdm, outfile; flabel=L"c")
        end

        # plot some full PDFs
        for iz = 1:2
            outfile = joinpath(savedir, "fluct_c_layer$(iz).png")
            P93.plot_fluctuations(sf_hist.tgrid[tidxfirst:end], conc_hist.ox[:,:,iz,tidxfirst:end], sdm, outfile; flabel=L"c")
        end
    end



    if todo["plot_tracer_variance"] == 1
        # Plot energy over time 
        conc_meansquare = P93.area_mean_square(conc_hist.ok, sdm.Nx, sdm.Ny)

        fig = Figure(size=(1200,800))
        lout = fig[1:2,1:1] = GridLayout()
        for iz = 1:2
            ax = Axis(lout[iz,1], xlabel="time", ylabel=L"Layer %$(iz) $\mathbb{E}[c^2]$", yscale=identity, title="Statistically steady state")
            lines!(ax, conc_hist.tgrid, vec(conc_meansquare[:,:,iz,:]), color=:red)
        end
        #colsize!(lout, 2, Aspect(1, 1.0))
        save(joinpath(savedir,"conc_meansquare.png"), fig)
        # TODO plot spectrum
    end

    if todo["plot_energy"] == 1
        # Plot energy over time 
        energy = P93.mean_energy_density(sf_hist, cop, sdm)
        energy_the = P93.mean_energy_density(sf_hist_the, cop, sdm)

        tidx_transition = findfirst(vec(energy_the[:,:,1,:] .> maximum(energy[:,:,1,:])))
        @show tidx_transition

        fig = Figure(size=(1200,800))
        lout = fig[1:2,1:2] = GridLayout()
        for iz = 1:2
            ax = Axis(lout[iz,1], xlabel="time", ylabel="Layer $(iz) energy density", title="Early stage", yscale=log10)
            lines!(ax, sf_hist.tgrid[1:tidx_transition], vec(energy_the[:,:,iz,1:tidx_transition]), color=:black, linestyle=:dash, linewidth=3)
            lines!(ax, sf_hist.tgrid[1:tidx_transition], vec(energy[:,:,iz,1:tidx_transition]), color=:red)
            ax = Axis(lout[iz,2], xlabel="time", ylabel="Layer $(iz) energy density", yscale=identity, title="Statistically steady state")
            lines!(ax, sf_hist.tgrid[tidx_transition:end], vec(energy[:,:,iz,tidx_transition:end]), color=:red)
        end
        #colsize!(lout, 2, Aspect(1, 1.0))
        colsize!(lout, 1, Relative(max(1/6,tidx_transition/length(sf_hist.tgrid))))
        save(joinpath(savedir,"energy.png"), fig)
        # TODO plot spectrum
    end

    if todo["animate"] == 1
        todo_anim = Dict(
                         "conc" =>          1,
                         "u" =>             1,
                         "pv" =>            1,
                         "sfmodes" =>       1,
                         "pvflux" =>        1,
                         "heatflux" =>      1,
                        )
        tgrid = sf_hist.tgrid
        fcont = zeros(Float64, (sdm.Nx, sdm.Ny, 2, length(tgrid)))
        fheat = zeros(Float64, (sdm.Nx, sdm.Ny, 2, length(tgrid)))
        if todo_anim["conc"] == 1
            println("starting to animate conc")
            P93.synchronize_FlowField_k2x!(conc_hist)
            fheat .= conc_hist.ox .+ php.grad_tr[2]*(sdm.ygrid .- sdm.Ly/2)' .+ php.grad_tr[1]*(sdm.xgrid .- sdm.Lx/2)
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(savedir,"anim_sf_conc.mp4")
            P93.animate_fields(tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\Psi", fheat_label=L"c", julia_old=julia_old)
        end
        if todo_anim["sfmodes"] == 1
            println("starting to animate sfmodes")
            sfmodes_hist = P93.FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
            for mode = 1:2
                sfmodes_hist.ok[:,:,mode,:] .+= 1/2 * (sf_hist.ok[:,:,1,:] +(-1)^mode*sf_hist.ok[:,:,2,:])
            end
            P93.synchronize_FlowField_k2x!(sfmodes_hist)
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(savedir,"anim_sf_sfmodes.mp4")
            fheat .= sfmodes_hist.ox
            titles = [L"\Psi_{\mathrm{BC}}", L"\Psi_{\mathrm{BT}}"]
            P93.animate_fields(tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\Psi", fheat_label=L"modes", julia_old=julia_old, titles=titles)
        end
        if todo_anim["heatflux"] == 1
            T_hist = P93.FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
            for iz = 1:2
                T_hist.ok[:,:,iz,:] .+= (-1)^iz/2 * (sf_hist.ok[:,:,1,:] - sf_hist.ok[:,:,2,:])
            end
            P93.synchronize_FlowField_k2x!(T_hist)
            v_hist = P93.FlowFieldHistory(sf_hist.tgrid, sdm.Nx, sdm.Ny)
            v_hist.ok .= cop.Dx .* sf_hist.ok
            P93.synchronize_FlowField_k2x!(v_hist)
            fheat .= T_hist.ox .* v_hist.ox
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(savedir,"anim_sf_heatflux.mp4")
            P93.animate_fields(tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\Psi", fheat_label=L"v'T'", julia_old=julia_old)
        end
        if todo_anim["u"] == 1
            println("starting to animate u")
            u_hist = P93.FlowFieldHistory(sf_hist.tgrid, sdm.Nx, sdm.Ny)
            u_hist.ok .= -cop.Dy .* sf_hist.ok
            P93.synchronize_FlowField_k2x!(u_hist)
            fheat .= u_hist.ox 
            fheat[:,:,1,:] .+= 1.0 # mean flow 
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(savedir,"anim_sf_u.mp4")
            P93.animate_fields(tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\Psi", fheat_label=L"u", julia_old=julia_old)
        end
        if todo_anim["pv"] == 1
            println("starting to animate pv")
            pv_hist = P93.FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
            pv_hist.ok .= cop.Lap .* sf_hist.ok
            pv_hist.ok[:,:,1,:] .-= (sf_hist.ok[:,:,1,:] - sf_hist.ok[:,:,2,:])/2
            pv_hist.ok[:,:,2,:] .+= (sf_hist.ok[:,:,1,:] - sf_hist.ok[:,:,2,:])/2
            P93.synchronize_FlowField_k2x!(pv_hist)
            fheat .= pv_hist.ox .+ php.beta*(sdm.ygrid .- sdm.Ly/2)'
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(savedir,"anim_sf_pv.mp4")
            P93.animate_fields(tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\Psi", fheat_label=L"q", julia_old=julia_old)
        end
        if todo_anim["pvflux"] == 1
            println("starting to animate pvflux")
            pv_hist = P93.FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
            pv_hist.ok .= cop.Lap .* sf_hist.ok
            pv_hist.ok[:,:,1,:] .-= (sf_hist.ok[:,:,1,:] - sf_hist.ok[:,:,2,:])/2
            pv_hist.ok[:,:,2,:] .+= (sf_hist.ok[:,:,1,:] - sf_hist.ok[:,:,2,:])/2
            P93.synchronize_FlowField_k2x!(pv_hist)
            v_hist = P93.FlowFieldHistory(sf_hist.tgrid, sdm.Nx, sdm.Ny)
            v_hist.ok .= cop.Dx .* sf_hist.ok
            P93.synchronize_FlowField_k2x!(v_hist)
            fheat .= (pv_hist.ox .+ php.beta*(sdm.ygrid .- sdm.Ly/2)') .* v_hist.ox
            fcont .= sf_hist.ox .- 1.0*(sdm.ygrid .- sdm.Ly/2)'
            outfile = joinpath(savedir,"anim_sf_pvflux.mp4")
            P93.animate_fields(tgrid, fcont, fheat, sdm, outfile; fcont_label=L"\Psi", fheat_label=L"v'q'", julia_old=julia_old)
        end
    end
end






args = Dict()
if length(ARGS) > 0
    args[:i_expt] = parse(Int, ARGS[1])
end
direct_numerical_simulation(; args...)# parse(Int, ARGS[1]))
