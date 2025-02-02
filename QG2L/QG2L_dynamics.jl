function timestep!(
        flow_next::FlowState,
        flob_next::FlowStateObservables,
        flow_prev::FlowState,
        flob_prev::FlowStateObservables,
        cop::ConstantOperators,
        sdm::SpaceDomain,
        php::PhysicalParams,
        ;
        nonlinear::Bool=true,
    )
    if nonlinear
        flow_next.sf.ok .= flob_prev.sf_fwdmap.ok
        #flow_next.conc.ok .= flob_prev.conc_fwdmap.ok
    else
        flow_next.sf.ok .= flob_prev.sf_fwdmap_lin.ok
        #flow_next.conc.ok .= flob_prev.conc_fwdmap_lin.ok
    end
    flow_next.conc .= flob_prev.conc_fwdmap
    #flow_next.sf.ok .= flob_prev.sf.ok + cop.dt*flob_prev.sf_dt.ok
    flow_next.sf.ok .*= cop.dealias_mask
    #flow_next.conc.ok .*= cop.dealias_mask
    synchronize_FlowField_k2x!(flow_next.sf) #, cop)
    #synchronize_FlowField_k2x!(flow_next.conc, cop)
    flow_next.tph = flow_prev.tph + cop.dtph_solve
    # in future, only compute some observables for greater efficiency
    compute_observables!(flob_next, flow_next, cop, sdm, php)

    # ----------- Corrector step -------------
    flow_next.sf.ok .= flob_prev.sf_fwdmap_lin.ok .+ 0.5 .* (flob_prev.sf_fwdmap_nonlin.ok .+ flob_next.sf_fwdmap_nonlin.ok)
    flow_next.sf.ok .*= cop.dealias_mask
    synchronize_FlowField_k2x!(flow_next.sf) #, cop)
    flow_next.conc .= 0.5 .* (flob_prev.conc_fwdmap .+ flob_next.conc_fwdmap)
    #flow_next.conc.ok .= flob_prev.conc_fwdmap_lin.ok .+ 0.5*(flob_prev.conc_fwdmap_nonlin.ok .+ flob_next.conc_fwdmap_nonlin.ok)
    #flow_next.conc.ok .*= cop.dealias_mask
    compute_observables!(flob_next, flow_next, cop, sdm, php)
    # ------------------------------------------

    # Energy diagnosis
    Emode_prev = energy_per_mode(flow_prev.sf, cop, sdm)
    Emode_next = energy_per_mode(flow_next.sf, cop, sdm)
    #println("BENCHMARK 4")
    if false
        println("Emode_prev = ")
        display(Emode_prev .* (abs.(Emode_prev) .> 1e-10))
        println("Emode_next = ")
        display(Emode_next .* (abs.(Emode_next) .> 1e-10))
    end
    if !all(isfinite.(Emode_next))
        println("Emode_prev = ")
        display(Emode_prev .* (abs.(Emode_prev) .> 1e-10))
        println("Emode_next = ")
        display(Emode_next .* (abs.(Emode_next) .> 1e-10))
        error("Infinite energy")
    end
    return
end

function perturb!(flow::FlowState, perts::Vector{Float64}, pertop::PerturbationOperator)
    # pertop transforms pert into a vector
    Nmodes_sf,Nmodes_conc = (length(modes) for modes = (pertop.sf_pert_modes,pertop.conc_pert_modes))
    @show Nmodes_sf,Nmodes_conc
    i_pert_dim = 1
    for p = 1:Nmodes_sf
        amplitude = sqrt(
                         pertop.sf_pert_amplitudes_min[p]^2 * (1-perts[i_pert_dim]) 
                         + pertop.sf_pert_amplitudes_max[p]^2 * perts[i_pert_dim]
                        )
        phase = 2pi*perts[i_pert_dim+1]
        flow.sf.ok .+= (amplitude * exp(1im*phase)) .* pertop.sf_pert_modes[p] 
        i_pert_dim += 2
    end
    synchronize_FlowField_k2x!(flow.sf)
    @infiltrate maximum(abs.(flow.sf.ox)) > 0.6
    for p = 1:Nmodes_conc # TODO use boundary values to set bounds 
        flow.conc .+= ((perts[i_pert_dim] .- 0.5) * pertop.conc_pert_amplitudes_max[p]) .* pertop.conc_pert_modes[p]
        i_pert_dim += 1
    end
    return 
end

    


function integrate(
        flow_init::FlowState,
        tfin::Int64,
        pert_seq::PerturbationSequence,
        pertop::PerturbationOperator,
        cop::ConstantOperators,
        sdm::SpaceDomain,
        php::PhysicalParams,
        ;
        nonlinear::Bool = true,
        verbose::Bool = false,
        # allocations
        sf_the::Union{Nothing,FlowField} = nothing,
        sf_hist::Union{Nothing,FlowFieldHistory} = nothing,
        sf_hist_the::Union{Nothing,FlowFieldHistory} = nothing,
        conc_hist::Union{Nothing,Array{Float64}} = nothing,
    )

    tphinit = flow_init.tph

    flow_prev = FlowState(tphinit, sdm.Nx, sdm.Ny)
    flob_prev = FlowStateObservables(sdm.Nx,sdm.Ny)
    flow_next = FlowState(tphinit, sdm.Nx, sdm.Ny)
    flob_next = FlowStateObservables(sdm.Nx,sdm.Ny)

    copy_FlowState!(flow_prev, flow_init)
    compute_observables!(flob_prev, flow_prev, cop, sdm, php)


    Nt = tfin - floor(Int, tphinit/sdm.tu) #round(Int, (tfin-tinit)/dt_save)
    tgrid = collect((tfin-Nt+1):1:tfin) #range(tinit,tfin,length=Nt+1)[2:end])
    # optionally allocate
    if (isnothing(sf_hist) || isnothing(sf_hist_the) || isnothing(conc_hist))
        sf_hist = FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
        sf_hist_the = FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
        conc_hist = zeros(Float64, (sdm.Nx, sdm.Ny, 2, Nt))
        sf_the = FlowField(sdm.Nx, sdm.Ny)
    else
        sf_hist.tgrid .= tgrid
        sf_hist_the.tgrid .= tgrid
    end

    tph = tphinit
    i_save = 1
    i_print = 1
    dtph_print = 5.0
    i_pert = 1
    while tph < tfin*sdm.tu 
        if (length(pert_seq.ts_ph) >= i_pert) && (tph >= pert_seq.ts_ph[i_pert])
            perturb!(flow_prev, pert_seq.perts[i_pert], pertop)
            compute_observables!(flob_prev, flow_prev, cop, sdm, php)
            i_pert += 1
        end
        timestep!(flow_next, flob_next, flow_prev, flob_prev, cop, sdm, php; nonlinear=nonlinear)
        tph = flow_next.tph

        if tph >= tgrid[i_save]*sdm.tu
            weight_prev = (tph - tgrid[i_save]*sdm.tu)/cop.dtph_solve
            sf_hist.ox[:,:,:,i_save] .= weight_prev * flow_prev.sf.ox + (1-weight_prev) * flow_next.sf.ox
            sf_hist.ok[:,:,:,i_save] .= weight_prev * flow_prev.sf.ok + (1-weight_prev) * flow_next.sf.ok
            conc_hist[:,:,:,i_save] .= weight_prev * flow_prev.conc + (1-weight_prev) * flow_next.conc
            # Vectorize somehow
            for ix = 1:(div(sdm.Nx,2)+1)
                for iy = 1:sdm.Ny
                    sf_hist_the.ok[ix,iy,:,i_save] .= cop.evecs_ctime[ix,iy,:,:] * LA.diagm(0 => exp.(cop.evals_ctime[ix,iy,:] * (tph-tphinit))) * cop.evecs_ctime_inv[ix,iy,:,:] * flow_init.sf.ok[ix,iy,:]
                end
            end
            i_save += 1
        end
        if verbose && (tph-tphinit > i_print*dtph_print)
            println()
            Printf.@printf("tph = %.2e, sf range = (%.2e,%.2e), conc range = (%.2e,%.2e)\n", tph, minimum(flow_next.sf.ox), maximum(flow_next.sf.ox), minimum(flow_next.conc), maximum(flow_next.conc))
            energy = mean_energy_density(flow_next.sf, cop, sdm)
            println("energy = ")
            display(vec(energy))
            i_print += 1
        end
        copy_FlowState!(flow_prev, flow_next)
        copy_FlowStateObservables!(flob_prev, flob_next)
    end
    synchronize_FlowField_k2x!(sf_hist_the)
    return flow_next,sf_hist,sf_hist_the,conc_hist
end

