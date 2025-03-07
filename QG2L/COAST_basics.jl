
function sigmoid(x)
    return 1/(1+exp(-x))
end

function logit(x)
    return log(x) - log1p(-x)
end

function logit_shifted_scaled(c::Float64, c0::Float64)
    z0 = logit((1+c0)/(1+2*c0))
    z = z0 + logit(1/2*((c-1/2)/(c0+1/2) + 1))
    return z
end

function sigmoid_shifted_scaled(z::Float64, c0::Float64)
    z0 = logit((1+c0)/(1+2*c0))
    c = 1/2 + (1+2*c0)*(sigmoid(z-z0)-1/2)
    return c
end

function transcorr(x::Float64, fwd::Bool,; c0=0.1)
    # c denotes correlation (0 <= c <= 1)
    # z denotes transformed correlation (-infty < z < infty)
    if fwd # 
        c = abs(x)
        z = logit_shifted_scaled.(c, c0)
        return z*sign(x)
    else
        z = abs(x)
        c = sigmoid_shifted_scaled.(z, c0)
        return c*sign(x)
    end
end

transcorr(x::Float64, c0::Float64=0.1) = transcorr(x, true; c0=c0)
invtranscorr(x::Float64, c0::Float64=0.1) = transcorr(x, false; c0=c0)


function plot_transcorr(figdir)
    c = collect(range(-1,1,length=200))
    fig = Figure(size=(800,200))
    lout = fig[1,1] = GridLayout()
    ax = Axis(lout[1,1], xlabel="σ⁻¹(ρ; 𝑐₀)", ylabel="Correlation ρ", xgridvisible=false, ygridvisible=false)
    hlines!(ax, [0, 1]; color=:grey79)
    vlines!(ax, 0; color=:grey79)
    c0s = [0.05,0.1,0.2]
    tofcs = hcat((transcorr.(c, c0) for c0=c0s)...)
    z = collect(range(extrema(tofcs)...; length=200))
    sofzs = hcat((sigmoid_shifted_scaled.(z, c0) for c0=c0s)...)
    colors = [:green, :black, :purple]
    for (i_c0,c0) in enumerate(c0s)
        colargs = Dict(:color=>colors[i_c0])
        lines!(ax, tofcs[:,i_c0], c; colargs..., label=@sprintf("𝑐₀ = %.2f", c0))
        lines!(ax, z, sofzs[:,i_c0]; linestyle=(:dash,:dense), colargs...)

    end
    ylims!(ax, ([0,1] .+ [-1,1].*maximum(c0s))...) #+maximum(c0s))
    xlims!(ax, z[1], z[end])
    axislegend(ax; position=:lc, framevisible=false, labelsize=10)
    save(joinpath(figdir,"transcorr.png"), fig)
end



            



mutable struct ConfigCOAST
    lead_time_min::Int64
    lead_time_max::Int64 
    lead_time_inc::Int64
    follow_time::Int64 # how long to simulate after the peak 
    peak_prebuffer_time::Int64 # the time interval over which a chosen peak must be the running maximum
    dtRmax_max::Int64 # how far descendants' max scores are allowed to drift (in either direction)
    num_init_conds_max::Int64
    num_perts_max::Int64
    target_field::String
    target_xPerL::Float64 # as fraction of domain size 
    target_rxPerL::Float64 # as fraction of domain size 
    target_yPerL::Float64 # as fraction of domain size 
    target_ryPerL::Float64 # as fraction of domain size 
    # TODO incorporate PerturbationOperator and target quanttities parameters into this 
end

function ConfigCOAST(
        tu,
        ;
        lead_time_min_ph::Float64 = 2.0,
        lead_time_max_ph::Float64 = 40.0,
        lead_time_inc_ph::Float64 = 2.0, 
        follow_time_ph::Float64 = 20.0,
        peak_prebuffer_time_ph::Float64 = 30.0,
        dtRmax_max_ph::Float64 = 40.0,
        num_init_conds_max::Int64 = 32,
        num_perts_max_per_lead_time::Int64 = 15,
        target_field::String = "conc1",
        target_xPerL::Float64 = 0.5,
        target_rxPerL::Float64 = 1/64,
        target_yPerL::Float64 = 0.5,
        target_ryPerL::Float64 = 1/64,
    )
    lead_time_min = round(Int, lead_time_min_ph/tu)
    lead_time_max = round(Int, lead_time_max_ph/tu)
    lead_time_inc = max(1, round(Int, lead_time_inc_ph/tu))
    follow_time = round(Int, follow_time_ph/tu)
    peak_prebuffer_time = max(1, round(Int, peak_prebuffer_time_ph/tu))
    dtRmax_max = round(Int, dtRmax_max_ph/tu)
    num_lead_times = length(range(lead_time_min,lead_time_max; step=lead_time_inc))
    num_perts_max = num_perts_max_per_lead_time * num_lead_times
    return ConfigCOAST(lead_time_min, lead_time_max, lead_time_inc, follow_time, peak_prebuffer_time, dtRmax_max, num_init_conds_max, num_perts_max, target_field, target_xPerL, target_rxPerL, target_yPerL, target_ryPerL)
end

function strrep_ConfigCOAST_varying_yPerL(cfg::ConfigCOAST)
    s = @sprintf(
                        "tgt%s_x%.2frx%.3f_yallry%.3f", 
                        cfg.target_field, 
                        cfg.target_xPerL, 
                        cfg.target_rxPerL, 
                        cfg.target_ryPerL,
                       ) 
    return replace(s, "."=>"p")
end

function strrep_ConfigCOAST(cfg::ConfigCOAST)
    s = @sprintf(
                        "tgt%s_x%.2frx%.3f_y%.2fry%.3f", 
                        cfg.target_field, 
                        cfg.target_xPerL, 
                        cfg.target_rxPerL, 
                        cfg.target_yPerL, 
                        cfg.target_ryPerL,
                       ) 
    return replace(s, "."=>"p")
end


mutable struct COASTState
    ancestors::Vector{Int64}
    ancestor_init_conds::Vector{String}
    ancestor_init_cond_prehistories::Vector{String}
    dns_peaks::Vector{Float64}
    dns_peak_times::Vector{Int64}
    peak_times_lower_bounds::Vector{Int64}
    peak_times_upper_bounds::Vector{Int64}
    dns_anc_Roft::Vector{Vector{Float64}}
    anc_Roft::Vector{Vector{Float64}} # time-dependent score 
    anc_Rmax::Vector{Float64}
    anc_tRmax::Vector{Int64}
    desc_Roft::Vector{Vector{Vector{Float64}}}
    desc_Rmax::Vector{Vector{Float64}}
    desc_tRmax::Vector{Vector{Int64}}
    desc_tphpert::Vector{Vector{Float64}}
    pert_seq_qmc::Matrix{Float64}
    terminate::Bool
end

function COASTState(pert_dim::Int64)
    ancestors = Vector{Int64}([])
    ancestor_init_conds = Vector{String}([])
    ancestor_init_cond_prehistories = Vector{String}([])
    dns_anc_Roft = Vector{Vector{Float64}}([]) # time-dependent score 
    anc_Roft = Vector{Vector{Float64}}([]) # time-dependent score 
    anc_Rmax = Vector{Float64}([])
    anc_tRmax = Vector{Int64}([])
    desc_Roft = Vector{Vector{Vector{Float64}}}([])
    desc_Rmax = Vector{Vector{Float64}}([])
    desc_tRmax = Vector{Vector{Int64}}([]) # time at 
    desc_tphpert = Vector{Vector{Float64}}([])
    dns_peaks = Vector{Float64}([])
    dns_peak_times = Vector{Int64}([])
    peak_times_lower_bounds = Vector{Int64}([])
    peak_times_upper_bounds = Vector{Int64}([])
    Nsamp = 1024
    pert_seq_qmc = QMC.sample(Nsamp, zeros(Float64, pert_dim), ones(Float64, pert_dim), QMC.LatticeRuleSample())
    if all(pert_seq_qmc[:,1] .== 0.0)
        pert_seq_qmc = pert_seq_qmc[:,2:end]
    end
    #@show U[:,1:6]
    #@show vcat(U,-U)[:,1:6]
    #Usym = reshape(vcat(U,-U), (pert_dim,2*Nsamp))./2
    #@show Usym[:,1:6]
    #Usym .+= 0.5
    #@show Usym[:,1:6]
    ## Transform to Gaussian 
    #pert_seq_qmc = SB.quantile(Dists.Normal(0,1), Usym)
    terminate = false
    return COASTState(
                       ancestors,
                       ancestor_init_conds, ancestor_init_cond_prehistories, 
                       dns_peaks, dns_peak_times, peak_times_lower_bounds, peak_times_upper_bounds, dns_anc_Roft,
                       anc_Roft,anc_Rmax,anc_tRmax,
                       desc_Roft,desc_Rmax,desc_tRmax,
                       desc_tphpert,
                       pert_seq_qmc,
                       terminate
                      )
end

function save_COASTState(coast::COASTState, coastfile::String)
    JLD2.jldopen(coastfile, "w") do f
        for symb = fieldnames(COASTState)
            f[String(symb)] = getfield(coast, symb)
        end
    end
    return
end
function load_COASTState(coastfile::String)
    args = JLD2.jldopen(coastfile, "r") do f
        return [f[String(symb)] for symb=fieldnames(COASTState)]
    end
    return COASTState(args...)
end

function add_ancestor!(coast::COASTState, ens::EM.Ensemble, anc::Int64, cfg::ConfigCOAST, new_obj_val::Vector{Float64})
    # TODO should we specify the peak to boost, based on how the initial condition was set up? 
    push!(coast.ancestors, anc)
    i_anc = length(coast.ancestors)
    push!(coast.anc_Roft, new_obj_val)
    Nt = length(new_obj_val)
    tgrid = (ens.trajs[anc].tfin - Nt) .+ collect(range(1,Nt,step=1))
    it_upcross = coast.peak_times_lower_bounds[i_anc] - tgrid[1] + 1 
    it_downcross = coast.peak_times_upper_bounds[i_anc] - tgrid[1] + 1
    @show it_upcross, it_downcross, tgrid[1]
    it_lower_bound = max(1, it_upcross)
    it_upper_bound = min(Nt, it_downcross-1)
    it_peak = it_lower_bound - 1 + argmax(new_obj_val[it_lower_bound:it_upper_bound])
    tRmax = tgrid[it_peak]
    push!(coast.anc_Rmax, new_obj_val[it_peak])
    push!(coast.anc_tRmax, tRmax)
    push!(coast.desc_Roft, Vector{Int64}([]))
    push!(coast.desc_Rmax, Vector{Int64}([]))
    push!(coast.desc_tRmax, Vector{Int64}([]))
    push!(coast.desc_tphpert, Vector{Float64}([]))
    return
end

function add_descendant!(coast::COASTState, ens::EM.Ensemble, anc::Int64, desc::Int64, new_obj_val::Vector{Float64}, pert_seq::QG2L.PerturbationSequence, dtRmax_max::Int64)
    i_anc = findfirst(coast.ancestors .== anc)
    Nt = length(new_obj_val)
    tinit = ens.trajs[desc].tfin - Nt
    tgrid = (ens.trajs[desc].tfin - Nt) .+ collect(range(1,Nt,step=1))
    push!(coast.desc_Roft[i_anc], new_obj_val)
    it_peak_lolim = max(1, coast.anc_tRmax[i_anc] - dtRmax_max - tinit)
    it_peak_uplim = min(Nt, coast.anc_tRmax[i_anc] + dtRmax_max - tinit)
    it_peak = it_peak_lolim-1 + argmax(new_obj_val[it_peak_lolim:it_peak_uplim])
    tRmax = tgrid[it_peak]
    #it_upcross = coast.peak_times_lower_bounds[i_anc] - tgrid[1] + 1 
    #it_downcross = coast.peak_times_upper_bounds[i_anc] - tgrid[1] + 1
    #it_lower_bound = max(1, it_upcross)
    #it_upper_bound = min(Nt, it_downcross-1)
    #it_peak = it_lower_bound - 1 + argmax(new_obj_val[it_lower_bound:it_upper_bound])
    push!(coast.desc_Rmax[i_anc], new_obj_val[it_peak])
    tRmax = tgrid[it_peak]
    push!(coast.desc_tRmax[i_anc], tRmax)
    push!(coast.desc_tphpert[i_anc], pert_seq.ts_ph[1])
    return
end

function adjust_paths!(coast::COASTState, old_path_part::String, new_path_part::String)
    modify(s) = replace(s, old_path_part=>new_path_part)
    Nanc = length(coast.ancestors)
    for i_anc = 1:Nanc
        coast.ancestor_init_conds[i_anc] = modify(coast.ancestor_init_conds[i_anc])
        coast.ancestor_init_cond_prehistories[i_anc] = modify(coast.ancestor_init_cond_prehistories[i_anc])
    end
end

function adjust_paths!(ens::EM.Ensemble, old_path_part::String, new_path_part::String)
    modify(s) = replace(s, old_path_part=>new_path_part)
    Nmem = EM.get_Nmem(ens)
    for i_mem = 1:Nmem
        ens.trajs[i_mem].init_cond = modify(ens.trajs[i_mem].init_cond)
        ens.trajs[i_mem].forcing = modify(ens.trajs[i_mem].forcing)
        ens.trajs[i_mem].term_cond = modify(ens.trajs[i_mem].term_cond)
        ens.trajs[i_mem].history = modify(ens.trajs[i_mem].history)
    end
end

function upgrade_Ensemble!(immut_ens_file::String, ens_filename::String)
    typemap = Dict("Ensemble"=>EM.ImmutableTrajectoryEnsemble, "Trajectory"=>EM.ImmutableTrajectory)
    immut_ens = JLD2.jldopen(immut_ens_file, "r"; typemap=typemap) do f
        return f["ens"]
    end
    trajs = Vector{EM.Trajectory}([])
    for i_traj = 1:length(immut_ens.trajs)
        traj = EM.Trajectory(
                          (getfield(immut_ens.trajs[i_traj], fieldname) for fieldname = (:tphinit, :tfin, :init_cond, :forcing, :term_cond, :history))...
                         )
        push!(trajs, traj)
    end
    ens = EM.Ensemble(trajs, immut_ens.famtree)
    EM.save_Ensemble(ens, ens_filename)
    return
end





function adjust_scores!(coast::COASTState, ens::EM.Ensemble, cfg::ConfigCOAST, sdm::QG2L.SpaceDomain)
    enforce_causality = false
    for (i_anc,anc) in enumerate(coast.ancestors)
        descendants = Graphs.outneighbors(ens.famtree, anc)
        Nt = length(coast.anc_Roft[i_anc])
        tinit = ens.trajs[anc].tfin - Nt
        for (i_desc,desc) in enumerate(descendants) 
            Roft = coast.desc_Roft[i_anc][i_desc]
            it_pert = round(Int, coast.desc_tphpert[i_anc][i_desc]/sdm.tu)
            it_peak_lolim = max(1, coast.anc_tRmax[i_anc] - cfg.dtRmax_max - tinit)
            if enforce_causality
                it_peak_lolim = max(it_peak_lolim, it_pert - tinit)
            end
            it_peak_uplim = min(Nt, coast.anc_tRmax[i_anc] + cfg.dtRmax_max - tinit)
            # Make sure it's an actual peak 
            it_peak = it_peak_lolim-1 + argmax(Roft[it_peak_lolim:it_peak_uplim])
            if it_peak == it_peak_lolim
                while (it_peak > 1) && (Roft[it_peak] < Roft[it_peak-1])
                    it_peak -= 1
                    it_peak_lolim -= 1
                end
            elseif it_peak == it_peak_uplim
                while (it_peak < Nt) && (Roft[it_peak] < Roft[it_peak+1])
                    it_peak += 1
                    it_peak_uplim += 1
                end
            end

            coast.desc_Rmax[i_anc][i_desc] = Roft[it_peak]
            Nt = length(Roft)
            coast.desc_tRmax[i_anc][i_desc] = ens.trajs[desc].tfin - Nt + it_peak
            if Roft[it_peak] != maximum(Roft[it_peak_lolim:it_peak_uplim])
                @show Roft
                @show it_peak_lolim,it_peak_uplim
                error()
            end
        end
    end
end
    

function obj_fun_COAST_registrar(target_field::String, sdm::QG2L.SpaceDomain, cop::QG2L.ConstantOperators, xPerL::Float64, rxPerL::Float64, yPerL::Float64, ryPerL::Float64)
    if "conc1" == target_field
        iz = 1
        field_fun_from_file = QG2L.obs_fun_conc_hist
        field_fun_from_histories = (sf_hist,conc_hist) -> conc_hist[:,:,iz,:]
    elseif "conc2" == target_field
        iz = 1
        field_fun_from_file = QG2L.obs_fun_conc_hist
        field_fun_from_histories = (sf_hist,conc_hist) -> conc_hist[:,:,iz,:]
    elseif "sf2" == target_field
        iz = 2
        field_fun_from_file = QG2L.obs_fun_sf_hist
        field_fun_from_histories = (sf_hist,conc_hist) -> sf_hist.ox[:,:,iz,:]
    elseif "eke1" == target_field
        iz = 1
        field_fun_from_file = QG2L.obs_fun_eke_hist
        field_fun_from_histories = (sf_hist,conc_hist) -> QG2L.obs_fun_eke_hist(sf_hist.tgrid, sf_hist.ok, sdm, cop)[:,:,iz,:]
    end
    weights = QG2L.horz_avg_filter(sdm.Nx, sdm.Ny, xPerL, rxPerL, yPerL, ryPerL)
    function field_fun_horz_avg_from_file(f::JLD2.JLDFile)
        u = field_fun_from_file(f, sdm, cop)[:,:,iz,:] 
        return vec(sum(u .* weights; dims=[1,2]))
    end
    function field_fun_horz_avg_from_histories(sf_hist::QG2L.FlowFieldHistory,conc_hist::Array{Float64,4})
        u = field_fun_from_histories(sf_hist,conc_hist)
        return vec(sum(u .* weights; dims=[1,2]))
    end
    return field_fun_horz_avg_from_file,field_fun_horz_avg_from_histories

end










function obj_fun_COAST_conc2(
        f::JLD2.JLDFile, 
        sdm::QG2L.SpaceDomain, 
        cop::QG2L.ConstantOperators,
        xPerL::Float64, 
        rxPerL::Float64,
        yPerL::Float64, 
        ryPerL::Float64,
    )
    iz = 2
    conc = QG2L.obs_fun_conc_hist(f, sdm, cop)
    return vec(QG2L.horz_avg(conc[:,:,iz,:], sdm, xPerL, rxPerL, yPerL, ryPerL))
end

function obj_fun_COAST_conc1(
        f::JLD2.JLDFile, 
        sdm::QG2L.SpaceDomain, 
        cop::QG2L.ConstantOperators,
        xPerL::Float64, 
        rxPerL::Float64,
        yPerL::Float64, 
        ryPerL::Float64,
    )
    iz = 1
    conc = QG2L.obs_fun_conc_hist(f, sdm, cop)
    return vec(QG2L.horz_avg(conc[:,:,iz,:], sdm, xPerL, rxPerL, yPerL, ryPerL))
end

function obj_fun_COAST_eke1(
        f::JLD2.JLDFile, 
        sdm::QG2L.SpaceDomain, 
        cop::QG2L.ConstantOperators,
        xPerL::Float64, 
        rxPerL::Float64,
        yPerL::Float64, 
        ryPerL::Float64,
    )
    iz = 1
    eke = QG2L.obs_fun_eke_hist(f, sdm, cop)
    return vec(QG2L.horz_avg(eke[:,:,iz,:], sdm, xPerL, rxPerL, yPerL, ryPerL))
end

function obj_fun_COAST_sf2(
        f::JLD2.JLDFile, 
        sdm::QG2L.SpaceDomain, 
        cop::QG2L.ConstantOperators,
        xPerL::Float64, 
        rxPerL::Float64,
        yPerL::Float64, 
        ryPerL::Float64,
    )
    iz = 2
    sf = QG2L.obs_fun_sf_hist(f, sdm, cop)
    return vec(QG2L.horz_avg(sf[:,:,iz,:], sdm, xPerL, rxPerL, yPerL, ryPerL))
end

function prepare_init_cond_from_dns(
        ens_dns::EM.Ensemble, 
        obj_fun_COAST::Function,
        cfg::ConfigCOAST, 
        sdm::QG2L.SpaceDomain, 
        cop::QG2L.ConstantOperators, 
        php::QG2L.PhysicalParams, 
        pertop::QG2L.PerturbationOperator, 
        init_cond_file::String,
        prehistory_file::String,
        tmin::Int64, # a strict lower bound on the first permissible time index to start searching 
        tmax::Int64,
        thresh_hi::Float64,
        ;
        num_peaks_to_skip = 0
    )
    Nmem = EM.get_Nmem(ens_dns)
    tphinits_dns = [ens_dns.trajs[mem].tphinit for mem=1:Nmem]
    tinits_dns = round.(Int,tphinits_dns./sdm.tu) # These start BEFORE the timeseries of the corresponding member 
    tfins_dns = [ens_dns.trajs[mem].tfin for mem=1:Nmem]
    @show tphinits_dns 
    mem_dns_first = findfirst(tfins_dns .> tmin)
    mem_dns_last = findlast(tinits_dns .< tmax)
    if isnothing(mem_dns_first) || isnothing(mem_dns_last)
        return nothing
    end
    front_clip = tmin - tinits_dns[mem_dns_first] 
    back_clip = max(0, tfins_dns[mem_dns_last] - tmax)
    
    mems_dns = collect(range(mem_dns_first, mem_dns_last, step=1))
    tgrid_dns = collect(tinits_dns[mem_dns_first]+1:1:tfins_dns[mem_dns_last])[front_clip+1:end-back_clip]
    @show mems_dns
    @show tgrid_dns[1:10]
    
    hist_filenames = [ens_dns.trajs[mem].history for mem=mems_dns]
    obj_val_dns = reduce(vcat, QG2L.compute_observable_ensemble(hist_filenames, obj_fun_COAST))[front_clip+1:end-back_clip]
    # TODO include zonal symmetry for better estimate 
    pot = QG2L.peaks_over_threshold(obj_val_dns, thresh_hi, cfg.peak_prebuffer_time, cfg.follow_time, max(cfg.peak_prebuffer_time,cfg.lead_time_max))
    if isnothing(pot)
        return nothing
    end
    peak_vals,peak_tidx,upcross_tidx,downcross_tidx = pot
    it_peak = peak_tidx[num_peaks_to_skip+1]
    it_upcross = upcross_tidx[num_peaks_to_skip+1]
    it_downcross = downcross_tidx[num_peaks_to_skip+1]
    @show it_peak
    @show obj_val_dns[it_peak-2:it_peak+2]
    if !(obj_val_dns[it_peak] >= max(obj_val_dns[it_peak-1],obj_val_dns[it_peak+1]))
        #@infiltrate
        error()
    end
    

    init_time = tgrid_dns[it_peak] - cfg.lead_time_max - 1
    # Find the member whose initial time is the most recent one with respect to init_time
    mem2branch_dns = findlast(tphinits_dns .< init_time*sdm.tu)
    @show mem2branch_dns
    @show init_time
    @show tphinits_dns
    traj = ens_dns.trajs[mem2branch_dns]

    # Run a single auxiliary trajectory to plant a restart at the right time 
    tphinit = traj.tphinit
    @assert tphinit < init_time*sdm.tu
    flow_init = QG2L.read_state(traj.init_cond)
    pert_seq = QG2L.NullPerturbationSequence()
    flow_fin,sf_hist,sf_hist_the,conc_hist = QG2L.integrate(flow_init, init_time, pert_seq, pertop, cop, sdm, php; nonlinear=true)

    QG2L.write_state(flow_fin, init_cond_file)
    QG2L.write_history(sf_hist, conc_hist, prehistory_file)
    # Return the whole objective timeseries expected to be reproduced in the unperturbed control runs 
    tidx_dns_ancestor = (it_peak - cfg.lead_time_max - 1) .+ collect(1:(cfg.lead_time_max + cfg.follow_time))
    return obj_val_dns[it_peak],tgrid_dns[it_peak],tgrid_dns[it_upcross],tgrid_dns[it_downcross],obj_val_dns[tidx_dns_ancestor]
end

function set_sail!(
        # States to mutate
        ens::EM.Ensemble,
        coast::COASTState,
        rng::Random.AbstractRNG,
        obj_fun_COAST_from_file::Function,
        obj_fun_COAST_from_histories::Function,
        # Save-out location
        ensdir::String,
        # Parameters
        cfg::ConfigCOAST,
        cop::QG2L.ConstantOperators,
        pertop::QG2L.PerturbationOperator,
        sdm::QG2L.SpaceDomain,
        php::QG2L.PhysicalParams,
        ;
        # allocations
        sf_the::Union{Nothing,QG2L.FlowField} = nothing,
        sf_hist::Union{Nothing,QG2L.FlowFieldHistory} = nothing,
        sf_hist_the::Union{Nothing,QG2L.FlowFieldHistory} = nothing,
        conc_hist::Union{Nothing,Array{Float64}} = nothing,
    )
    Nmem = EM.get_Nmem(ens)
    mem = Nmem+1
    mem_is_ancestor = (length(coast.ancestors) < cfg.num_init_conds_max)
    memdir = joinpath(ensdir, "mem$(mem)")
    mkpath(memdir)
    #pertdim = length(pertop.camplitudes)
    #pert_dim = length(pertop.sf_pert_amplitudes)
    pert = zeros(Float64, pertop.pert_dim)
    parent = nothing
    pert_seq = QG2L.NullPerturbationSequence()
    if mem_is_ancestor
        i_anc = length(coast.ancestors) + 1
        init_cond_file = coast.ancestor_init_conds[i_anc] #joinpath(ensdir, "init_cond/init_cond_anc$(i_anc).jld2")
    else
        tpert_random = false
        pert_random = false
        desc_per_anc = [Graphs.outdegree(ens.famtree, anc) for anc=coast.ancestors]
        min_num_children = minimum(desc_per_anc)
        i_anc = findfirst(desc_per_anc .== min_num_children)
        i_desc = desc_per_anc[i_anc] + 1
        t_perts = (coast.anc_tRmax[i_anc] - cfg.lead_time_max) .+ range(0, cfg.lead_time_max-cfg.lead_time_min, step=cfg.lead_time_inc)
        parent = coast.ancestors[i_anc]
        if (i_anc > cfg.num_init_conds_max) || ((i_anc == cfg.num_init_conds_max) && (i_desc >= cfg.num_perts_max))
            coast.terminate = true
        end
        # Randomly choose a time to launch
        if (coast.anc_tRmax[i_anc] - cfg.lead_time_max)*sdm.tu <= ens.trajs[parent].tphinit
            @show ens.trajs[parent].tphinit
            @show coast.anc_Roft[i_anc]
            @show ens.trajs[parent].tfin - length(coast.anc_Roft[i_anc]) 
            @error begin
                "Parent could be perturbed before birth. tph_pert >= $((coast.anc_tRmax[i_anc]-cfg.lead_time_max)*sdm.tu) while tphinit of parent is $(ens.trajs[parent].tphinit)"
            end
            error()
        end
        tperts_descs = round.(Int, coast.desc_tphpert[i_anc]/sdm.tu)
        num_siblings_per_tphpert = [sum(tperts_descs .== t_pert) for t_pert=t_perts]
        if tpert_random
            tph_pert = sdm.tu * Random.rand(rng, t_perts)
        else
            tph_pert = sdm.tu * t_perts[argmin(num_siblings_per_tphpert)]
        end
        # Randomly choose a direction to launch 
        if pert_random
            Random.randn!(rng, pert)
        else
            # count how many siblings have the same perturbation time 
            num_siblings_same_tphpert = sum(abs.(coast.desc_tphpert[i_anc] .- tph_pert) .< cfg.lead_time_inc*sdm.tu/10)
            pert .= coast.pert_seq_qmc[:,num_siblings_same_tphpert+1]
        end
        pert_seq = QG2L.PerturbationSequence([tph_pert], [pert])
        init_cond_file = ens.trajs[parent].init_cond #join(ensdir, "mem$(parent)/term_cond.jld2")
    end
    term_cond_file = joinpath(memdir, "term_cond.jld2")
    history_file = joinpath(memdir, "history.jld2")
    pert_seq_file = joinpath(memdir, "pert_seq.jld2")
    QG2L.write_perturbation_sequence(pert_seq, pert_seq_file)
    flow_init = QG2L.read_state(init_cond_file)
    tfin = floor(Int, flow_init.tph/sdm.tu) + cfg.lead_time_max + cfg.follow_time
    # -------------- The integration ---------------------
    println("Starting to integrate member $(mem)")
    flow_next,sf_hist,sf_hist_the,conc_hist = QG2L.integrate(flow_init, tfin, pert_seq, pertop, cop, sdm, php; nonlinear=true, sf_the=sf_the, sf_hist=sf_hist, sf_hist_the=sf_hist_the, conc_hist=conc_hist)
    println("done integrating")
    # ----------------------------------------------------
    println("Starting to write history")
    QG2L.write_history(sf_hist, conc_hist, history_file)
    QG2L.write_state(flow_next, term_cond_file)
    #println("Finished writing history")
    #println("Starting to increment ensemble")
    traj = EM.Trajectory(flow_init.tph, tfin, init_cond_file, pert_seq_file, term_cond_file, history_file)
    EM.add_trajectory!(ens, traj; parent=parent)
    #println("Finished incrementing ensemble")
    #println("Starting to evaluate objective")
    new_obj_val = obj_fun_COAST_from_histories(sf_hist,conc_hist)
    if false
        new_obj_val = JLD2.jldopen(history_file, "r") do f
            return obj_fun_COAST_from_file(f)
        end
    end
    #println("Finished evaluating objective")
    if mem_is_ancestor
        add_ancestor!(coast, ens, mem, cfg, new_obj_val)
        i_anc = length(coast.ancestors)
        @show ens.trajs[mem].tphinit/sdm.tu
        @show coast.anc_Rmax[i_anc]
        @show coast.dns_peaks[i_anc]
        @show coast.anc_tRmax[i_anc]
        @show coast.dns_peak_times[i_anc]
        println("dns_anc_Roft = ")
        @show coast.dns_anc_Roft[i_anc]'
        println("dns_max = $(maximum(coast.dns_anc_Roft[i_anc]))")
        println("coast.anc_Roft = ")
        @show coast.anc_Roft[i_anc]'
        @assert (coast.anc_Rmax[i_anc] == coast.dns_peaks[i_anc])
    else
        add_descendant!(coast, ens, parent, mem, new_obj_val, pert_seq, cfg.dtRmax_max)
    end
    # garbage collection
    if false
        println("Starting garbage collection")
        sf_hist = nothing
        conc_hist = nothing
        sf_hist_the = nothing
        GC.gc()
        println("Finished garbage collection")
    end
end

function desc_by_leadtime(coast::COASTState, i_anc::Int64, leadtime::Int64, sdm::QG2L.SpaceDomain)
    tpert = coast.anc_tRmax[i_anc] - leadtime
    idx_desc = findall(round.(Int, coast.desc_tphpert[i_anc]./sdm.tu) .== tpert)
    return idx_desc
end

zero2nan(p) = replace(p, 0=>NaN)
inf2nan(p) = replace(p, Inf=>NaN)
clipccdfratio(x,ratiomax=1e4) = (1/ratiomax < x < ratiomax ? x : NaN)
clipccdf(x) = (x <= 1e-10 ? NaN : x)
clippdf(x, dlev=0.1) = (x <= 1e-10/dlev ? NaN : x)


function label_target(cfg::ConfigCOAST, sdm::QG2L.SpaceDomain, rsp::String)
    if "1" == rsp
        rspstr = "linear"
    elseif "2" == rsp
        rspstr = "quadratic"
    elseif "e" == rsp
        rspstr = "empirical"
    end
    label = "$(label_target(cfg, sdm)), $(rspstr) 𝑅"
    return label
end

function label_target(cfg::ConfigCOAST, sdm::QG2L.SpaceDomain, scale::Float64, rsp::String)
    if "1" == rsp
        rspstr = "linear"
    elseif "2" == rsp
        rspstr = "quadratic"
    elseif "e" == rsp
        rspstr = "empirical"
    end
    label = "$(label_target(cfg, sdm, scale)), $(rspstr) 𝑅*"
    return label
end

function label_target(cfg::ConfigCOAST, sdm::QG2L.SpaceDomain, scale::Float64)
    scalestr = @sprintf("%.3f", scale)
    label = "$(label_target(cfg,sdm)), scale $(scalestr)"
    return label
end

function label_target(cfg::ConfigCOAST, sdm::QG2L.SpaceDomain)
    N = sdm.Ny
    yN = round(Int, cfg.target_yPerL*N)
    xN = round(Int, cfg.target_xPerL*N)
    ryN = round(Int, cfg.target_ryPerL*N)
    rxN = round(Int, cfg.target_ryPerL*N)
    label = "𝑦₀ = ($yN/$N)𝐿, box radius ($ryN/$N)𝐿"
    return label
end

function label_target(target_ryPerL::Float64, sdm::QG2L.SpaceDomain)
    rxystr = @sprintf("(%d/%d)𝐿",round(Int,target_ryPerL*sdm.Ny),sdm.Ny)
    label = "Box radius $(rxystr)"
    return label
end

function powerofhalfstring(k::Int64)
    symbols = ["(½)","(½)²","(½)³","(½)⁴","(½)⁵","(½)⁶","(½)⁷","(½)⁸","(½)⁹"]
    if 1 <= k <= 9
        return symbols[k]
    end
    return "(½)^$(k)"
end

function sss(k::Int64)
    subscriptstrings = ["₁","₂","₃","₄","₅","₆","₇","₈","₉","₀"]
    return subscriptstrings[mod(k,10)]
end


function label_objective(cfg::ConfigCOAST)
    xstr = @sprintf("%.2f", cfg.target_xPerL)
    rxstr = @sprintf("%.2f", cfg.target_rxPerL)
    ystr = @sprintf("%.2f", cfg.target_yPerL)
    rystr = @sprintf("%.2f", cfg.target_ryPerL)
    locstr = L"x=(%$(xstr)\pm%$(rxstr))L,y=(%$(ystr)\pm%$(rystr))L"
    labels = Dict(
                  "sf2" => L"\Psi(%$(locstr),z=2)",
                  "conc1" => L"c(%$(locstr),z=1)",
                  "conc2" => L"c(%$(locstr),z=2)",
                  "eke1" => L"\frac{1}{2}|\nabla\Psi(%$(locstr),z=1)|^2",
                 )
    short_labels = Dict(
                        "sf2" => L"$\langle\Psi_2\rangle_{\mathrm{box}}$",
                        "conc1" => L"$\langle{c_1}\rangle_{\mathrm{box}}$",
                        "conc2" => L"$\langle{c_2}\rangle_{\mathrm{box}}$",
                        "eke1" => L"$\frac{1}{2}\langle|\nabla\Psi|^2\rangle_{\mathrm{box}}#",
                 )
    return labels[cfg.target_field],short_labels[cfg.target_field]
end


function paramsets()
    target_yPerLs = collect(range(0, 1; length=17)[4:14]) #1/2 .+ [-1/4,-1/8,0,1/8,1/4][1:3]
    target_rs = (1/32) .* sqrt.([1.0, 4.0])
    return target_yPerLs, target_rs
end


function expt_config_COAST(; i_expt=nothing)
    target_yPerLs,target_rs = paramsets()
    vbl_param_arrs = [target_yPerLs, target_rs]
    cartinds = CartesianIndices(tuple((length(arr) for arr in vbl_param_arrs)...))
    if isnothing(i_expt) || i_expt == 0
        ci_expt = CartesianIndex(4,1)
    else
        ci_expt = cartinds[i_expt]
    end
    return (vbl_param_arrs[j][ci_expt[j]] for j=1:length(vbl_param_arrs))
end

function expt_config_COAST_analysis(cfg,pertop)
    leadtimes = collect(range(cfg.lead_time_min,cfg.lead_time_max; step=cfg.lead_time_inc))
    Nleadtime = length(leadtimes)
    

    r2threshes = [0.7] #[0.8,0.7,0.6,0.5]
    Nr2th = length(r2threshes)
    pths = collect(range(1.0, 0.0; length=11)) 
    # for correlation thresholds, focus on the near-zero range too, and disregard negatives 
    corrs = invtranscorr.(collect(range(transcorr(0.01),transcorr(0.99);length=17)))

    # TODO implement mean absolute error as a more-outlier-sensitive alternative to R^2 

    # Parameterize both the input distributions and the response types 
    distns = ("b",) # bump, uniform, gaussian; also add noises in uniform and Gaussian form 
    rsps = ("e","1","2") # empirical, linear model, quadratic model
    mixobjs = Dict(
                   "lt"=>leadtimes, 
                   "r2"=>r2threshes, 
                   "pth"=>pths,
                   "pim"=>pths, 
                   "ei"=>["max"],  # reinterpreted as expected exceedance over threshold 
                   "eot"=>["max"],  # reinterpreted as expected exceedance over threshold 
                   "went"=>["max"],
                   "ent"=>["max"],
                   "max"=>["max"],
                   "globcorr"=>corrs,
                   "contcorr"=>corrs,
                  ) # mixing-related objectives to maximize when choosing a leadtime. Each entry of each list represents a different objective 
    lt2str(lt) = @sprintf("%.2f", lt)

    mixcrit_labels = Dict(
                         "lt"=>"AST", 
                         "r2"=>"𝑅²",
                         "pth"=>"𝑞(μ)",
                         "pim"=>"𝑞(𝑅*)",
                         "ei"=>"𝔼[(Δ𝑅*)₊]",
                         "eot"=>"𝔼[(𝑅*-μ)₊]",
                         "globcorr"=>"ρ[𝑐]",
                         "contcorr"=>"ρ[𝑐(⋅,𝑦₀)]",
                         "ent"=>"𝑆[(𝑅*-μ)₊]", # actually weighted
                         "went"=>"WEntropy",
                        )
    mixobj_labels = Dict(
                         "lt"=>["AST = $(lt2str(lt))" for lt=leadtimes],
                         "r2"=>["𝑅² = $(lt2str(r2))" for r2=r2threshes],
                         "ei"=>["max $(mixcrit_labels["ei"])"],
                         "eot"=>["max $(mixcrit_labels["eot"])"],
                         "pth"=>[@sprintf("𝑞(μ)≈%.2f", pth) for pth=pths],
                         "pim"=>[@sprintf("𝑞(𝑅ₙ*)≈%.2f", pth) for pth=pths],
                         "went"=>["Max. WEnt."],
                         "ent"=>["max $(mixcrit_labels["ent"])"],
                         "globcorr"=>[@sprintf("%s ≈ σ(%.2f)", mixcrit_labels["globcorr"], transcorr(corr)) for corr=corrs],
                         "contcorr"=>[@sprintf("%s ≈ σ(%.2f)", mixcrit_labels["contcorr"], transcorr(corr)) for corr=corrs],
                         # TODO add expected exceedance over threshold (tee or eet or ete)
                        )
    mcs = collect(keys(mixcrit_labels))
    mccolorlist = cgrad(:tab10, length(mcs); categorical=true).colors #CairoMakie.Colors.HSV.(range(0, 360, length(mcs)), 50, 50)
    mixcrit_colors = Dict(mcs[i]=>mccolorlist[i] for i=1:length(mcs))
    i_mode_sf = 1
    Rmin,Rmax = pertop.sf_pert_amplitudes_min[i_mode_sf],pertop.sf_pert_amplitudes_max[i_mode_sf]
    distn_scales = Dict(
                        "b" => Rmin .+ (Rmax-Rmin).*collect(range(0.2, 3.0; step=0.2)),
                        "u" => Rmin .+ (Rmax-Rmin).*[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
                        "g" => Rmin .+ (Rmax-Rmin).*[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
                       )
    fdivnames = ("qrmse","kl","chi2","tv")
    Nboot = 0 #1000
    ccdf_levels = 1 ./ (2 .^ collect(1:15))
    i_thresh_cquantile = 5
    time_ancgen_dns_ph = 4000
    time_ancgen_dns_ph_max = 8000
    time_valid_dns_ph = 16000
    xstride_valid_dns = 1
    adjust_ccdf_per_ancestor = Bool(0)
    return (leadtimes, r2threshes, distns, rsps, mixobjs, mixcrit_labels, mixobj_labels, mixcrit_colors, distn_scales, fdivnames,Nboot,ccdf_levels,time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,i_thresh_cquantile,adjust_ccdf_per_ancestor)
end


function expt_config_metaCOAST_latdep_analysis(; i_expt=nothing)
    target_yPerLs,target_rs = paramsets()
    vbl_param_arrs = [target_rs]
    cartinds = CartesianIndices(tuple((length(arr) for arr in vbl_param_arrs)...))
    if isnothing(i_expt) || i_expt == 0
        ci_expt = CartesianIndex(1,)
    else
        ci_expt = cartinds[i_expt]
    end
    return (vbl_param_arrs[j][ci_expt[j]] for j=1:length(vbl_param_arrs))
end

function regress_lead_dependent_risk_linear_quadratic(coast::COASTState, ens::EM.Ensemble, cfg::ConfigCOAST, sdm::QG2L.SpaceDomain, pertop::QG2L.PerturbationOperator)
    Nmem = EM.get_Nmem(ens)
    Nanc = length(coast.ancestors)
    leadtimes = collect(range(cfg.lead_time_min, cfg.lead_time_max; step=cfg.lead_time_inc))
    Ntpert = length(leadtimes)
    num_perts_max_per_leadtime = div(cfg.num_perts_max, Ntpert)
    coefs_linear = zeros(Float64, (3,Ntpert,Nanc))
    resid_range_linear = zeros(Float64, (2,Ntpert,Nanc))
    residmse_linear = zeros(Float64, (Ntpert,Nanc))
    rsquared_linear = zeros(Float64, (Ntpert,Nanc))
    coefs_quadratic = zeros(Float64, (6,Ntpert,Nanc))
    residmse_quadratic = zeros(Float64, (Ntpert,Nanc))
    rsquared_quadratic = zeros(Float64, (Ntpert,Nanc))
    resid_range_quadratic = zeros(Float64, (2,Ntpert,Nanc))
    hessian_eigvals = zeros(Float64, (2,Ntpert,Nanc))
    hessian_eigvecs = zeros(Float64, (2,2,Ntpert,Nanc))
    i_mode_sf = 1
    Threads.@threads for i_anc = 1:Nanc #(i_anc,anc) in enumerate(coast.ancestors)
        anc = coast.ancestors[i_anc]
        println("Thread $(Threads.threadid()) dealing with ancestor $(i_anc) out of $(Nanc)")
        descendants = Graphs.outneighbors(ens.famtree, anc)
        Xpert = zeros(Float64, (cfg.num_perts_max+1, 2))
        Ypert = zeros(Float64, cfg.num_perts_max+1)
        for (i_leadtime,leadtime) in enumerate(leadtimes)
            tpert = coast.anc_tRmax[i_anc] - leadtime
            idx_desc = desc_by_leadtime(coast, i_anc, leadtime, sdm)
            Ndesc = length(idx_desc)
            Ypert[Ndesc+1] = coast.anc_Rmax[i_anc]
            for (i_desc,desc) in enumerate(descendants[idx_desc])
                pert = QG2L.read_perturbation_sequence(ens.trajs[desc].forcing).perts[1]
                i_pert_dim = 2*i_mode_sf-1
                amplitude = sqrt(
                                 pertop.sf_pert_amplitudes_min[i_mode_sf]^2 * (1-pert[i_pert_dim]) 
                                 + pertop.sf_pert_amplitudes_max[i_mode_sf]^2 * pert[i_pert_dim]
                                )
                phase = 2pi*pert[i_pert_dim+1]
                #@show amplitude,phase*360/(2pi),scores[i_desc]
                Xpert[i_desc,:] .= amplitude .* [cos(phase), sin(phase)]
                Ypert[i_desc] = coast.desc_Rmax[i_anc][idx_desc[i_desc]]
            end
            coefs_linear[:,i_leadtime,i_anc],residmse_linear[i_leadtime,i_anc],rsquared_linear[i_leadtime,i_anc],resid_range_linear[:,i_leadtime,i_anc] = QG2L.linear_regression_2d(Xpert[1:Ndesc+1,:], Ypert[1:Ndesc+1]; intercept=coast.anc_Rmax[i_anc])
            coefs_quadratic[:,i_leadtime,i_anc],residmse_quadratic[i_leadtime,i_anc],rsquared_quadratic[i_leadtime,i_anc],resid_range_quadratic[:,i_leadtime,i_anc] = QG2L.quadratic_regression_2d(Xpert[1:Ndesc+1,:], Ypert[1:Ndesc+1]; intercept=coast.anc_Rmax[i_anc])
            eigs = QG2L.quadratic_regression_2d_eigs(coefs_quadratic[:,i_leadtime,i_anc])
            hessian_eigvals[:,i_leadtime,i_anc] .= eigs[1]
            hessian_eigvecs[:,:,i_leadtime,i_anc] .= eigs[2]
            #display(hessian_eigvals[:,i_leadtime,i_anc])
        end
    end
    println("R2 at initial smallest leadtime")
    @show size(rsquared_linear)
    display(hcat(range(1,Nanc), rsquared_linear[1,:]))
    return (
            coefs_linear,residmse_linear,rsquared_linear,resid_range_linear,
            coefs_quadratic,residmse_quadratic,rsquared_quadratic,resid_range_quadratic,
            hessian_eigvals,hessian_eigvecs
           )
end


function measure_dispersion(coast::COASTState, ens::EM.Ensemble, ens_dns::EM.Ensemble, cfg::ConfigCOAST, sdm::QG2L.SpaceDomain)
    # measure distance in two ways: L2 norm in streamfunction, and L2 norm in concentration
    rmsd = Dict("sf"=>0.0, "conc"=>0.0)
    # choose times from beginning of each member
    Nmem_dns = EM.get_Nmem(ens_dns)
    spinup_dns_ph = 500.0
    tphinits_dns = [ens_dns.trajs[mem].tphinit for mem=1:Nmem_dns]
    firstmem_dns = findfirst(tphinits_dns .>= spinup_dns_ph)
    lastmem_dns = EM.get_Nmem(ens_dns)
    hist_filenames = [ens.dns_trajs[mem].history for mem=1:Nmem_dns]
    sfa,sfb = (zeros(Float64, (sdm.Nx, sdm.Ny, 2)) for _=1:2)
    conca,concb = (zeros(Float64, (sdm.Nx, sdm.Ny, 2)) for _=1:2)
    sfandconc_term(filename) = begin
        sf_term,conc_term = JLD2.jldopen(filename, "r") do f
            return f["sf_hist_ox"][:,:,:,end], f["conc_hist"][:,:,:,end]
        end
        return sf_term,conc_term
    end
    Npairs = 0
    for mema = firstmem_dns:lastmem_dns
        sfa[:],conca[:] = sfandconc_term(hist_filenames[mema])
        for memb = (i1_mem+1):lastmem_dns
            sfb[:],concb[:] = sfandconc_term(hist_filenames[memb])
            rmsd["sf"] += sum((sfa .- sfb).^2)/(sdm.Nx*sdm.Ny*2)
            rmsd["conc"] += sum((conca .- concb).^2)/(sdm.Nx*sdm.Ny*2)
        end
    end
    msqdist(u1,u2) = vec(sum((u1 .- u2).^2; dims=(1,2,3))) ./ (sdm.Nx*sdm.Ny*2)
    leadtimes = collect(range(cfg.lead_time_min, cfg.lead_time_max; step=cfg.lead_time_inc))
    Nleadtime = length(leadtimes)
    Nanc = length(coast.ancestors)
    Nt = cfg.lead_time_max + cfg.follow_time
    msqdist_sf = Vector{Matrix{Float64}}([zeros(Float64, (leadtime+cfg.follow_time,Nanc)) for leadtime=leadtimes])
    msqdist_conc = Vector{Matrix{Float64}}([zeros(Float64, (leadtime+cfg.follow_time,Nanc)) for leadtime=leadtimes])
    sfa,sfb = (zeros(Float64, (sdm.Nx, sdm.Ny, 2, Nt)) for _=1:2)
    conca,concb = (zeros(Float64, (sdm.Nx, sdm.Ny, 2, Nt)) for _=1:2)
    sfandconc_hist(filename) = begin
        sf_term,conc_term = JLD2.jldopen(filename, "r") do f
            return f["sf_hist_ox"], f["conc_hist"]
        end
        return sf_term,conc_term
    end
    for (i_anc,anc) in enumerate(coast.ancestors)
        descendants = Graphs.outneighbors(ens.memgraph, anc)
        for (i_leadtime,leadtime) in enumerate(leadtimes)
            idx_desc = desc_by_leadtime(coast, i_anc, leadtime, sdm)
            Ndesc = length(idx_desc)
            Npairs = div(Ndesc*(Ndesc-1), 2)
            for i_a = 1:Ndesc
                mema = descendants[idx_desc[i_a]]
                hist_filename_a = ens.trajs[mema].history
                sfa[:],conca[:] = sfandconc_hist(hist_filename_a)
                for i_b = i_a+1:Npairs
                    hist_filename_b = ens.trajs[memb].history
                    sfb[:],concb[:] = sfandconc_hist(hist_filename_b)
                    msqdist_sf[i_leadtime][:,i_anc] .+= msqdist(sfa, sfb)[Nt-(cfg.follow_time+leadtime-1):Nt]/Npairs
                    msqdist_conc[i_leadtime][:,i_anc] .+= msqdist(conca, concb)[Nt-(cfg.follow_time+leadtime-1):Nt]/Npairs
                end
            end
        end
    end
    return msqdist_sf,msqdist_conc
end

function plot_contour_dispersion_distribution(
        coast::COASTState, 
        ens::EM.Ensemble, 
        cfg::ConfigCOAST, 
        sdm::QG2L.SpaceDomain,
        cop::QG2L.ConstantOperators, 
        pertop::QG2L.PerturbationOperator,
        contour_dispersion_filename::String,
        idx_anc_2plot::Vector{Int64},
        figdir::String,
    )
    (
     leadtimes,r2threshes,dsts,rsps,mixobjs,
     mixcrit_labels,mixobj_labels,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,i_thresh_cquantile,adjust_ccdf_per_ancestor
    ) = expt_config_COAST_analysis(cfg,pertop)
    globcorr,contcorr,contsymdiff = JLD2.jldopen(contour_dispersion_filename, "r") do f
        return f["globcorr"],f["contcorr"],f["contsymdiff"]
    end
    Nt,Nleadtime,Ndsc,Nanc = size(globcorr) # Ndesc here is per-leadtime
    for i_anc = idx_anc_2plot
        fig = Figure(size=(600,400))
        ylabels = [@sprintf("σ⁻¹(%s)", mixcrit_labels["globcorr"]), @sprintf("σ⁻¹(%s)", mixcrit_labels["contcorr"])]
        lout = fig[1,1] = GridLayout()
        ax_globcorr,ax_contcorr, = [Axis(lout[i,1]; xlabel="𝑡 − 𝑡* (𝑡* = $(@sprintf("%.0f", coast.anc_tRmax[i_anc]/sdm.tu)))", ylabel=ylabels[i], yscale=identity, xgridvisible=false, ygridvisible=false) for i=1:2]
        tgrid_ph = (collect(1:1:Nt) .- cfg.lead_time_max) .* sdm.tu
        for i_leadtime = collect(2:8:Nleadtime)
            leadtime = leadtimes[i_leadtime]
            tidx = (cfg.lead_time_max-leadtime+2):1:Nt
            for i_dsc = 1:Ndsc
                lines!(ax_globcorr,tgrid_ph[tidx],transcorr.((globcorr[tidx,i_leadtime,i_dsc,i_anc])); color=:tomato)
                lines!(ax_contcorr,tgrid_ph[tidx],transcorr.((contcorr[tidx,i_leadtime,i_dsc,i_anc])); color=:tomato)
            end
        end
        for (i_leadtime,leadtime) in enumerate(leadtimes)
            tidx = (cfg.lead_time_max-leadtime+2):1:Nt
            lines!(ax_globcorr, tgrid_ph[tidx], transcorr.((SB.mean(globcorr[:,i_leadtime,:,i_anc]; dims=2)[tidx,1])); color=:black)
            lines!(ax_contcorr, tgrid_ph[tidx], transcorr.((SB.mean(contcorr[:,i_leadtime,:,i_anc]; dims=2)[tidx,1])); color=:black)
            hlines!(ax_globcorr, 0.0; color=:gray, alpha=0.25, linestyle=(:dash,:dense))
            hlines!(ax_contcorr, 0.0; color=:gray, alpha=0.25, linestyle=(:dash,:dense))
        end
        save(joinpath(figdir,"corrs_anc$(i_anc).png"), fig)
    end
end


function compute_contour_dispersion(
        coast::COASTState, 
        ens::EM.Ensemble, 
        cfg::ConfigCOAST, 
        sdm::QG2L.SpaceDomain,
        cop::QG2L.ConstantOperators, 
        pertop::QG2L.PerturbationOperator,
        dns_stats_filename::String, 
        contour_dispersion_filename::String,
        thresh::Float64,
    )
    (
     leadtimes,r2threshes,dsts,rsps,mixobjs,
     mixcrit_labels,mixobj_labels,distn_scales,
     fdivnames,Nboot,ccdf_levels,
     time_ancgen_dns_ph,time_ancgen_dns_ph_max,time_valid_dns_ph,xstride_valid_dns,i_thresh_cquantile,adjust_ccdf_per_ancestor
    ) = expt_config_COAST_analysis(cfg,pertop)
    thresh_cquantile = ccdf_levels[i_thresh_cquantile]
    Nleadtime = length(leadtimes)
    Nanc = length(coast.ancestors)
    Nmem = EM.get_Nmem(ens)
    Ndsc = Nmem - Nanc
    Nt = cfg.follow_time + cfg.lead_time_max

    conc1_zonal_mean = JLD2.jldopen(dns_stats_filename,"r") do f
        iz = 1
        ix = 1
        return f["mssk_xall"][ix:ix,:,iz:iz,1:1]
    end
    iytgt = round(Int, cfg.target_yPerL*sdm.Ny)
    contour_level = conc1_zonal_mean[1,iytgt,1,1]


    # Fast method for reading in concentrations (third dimension for layer is kinda silly)
    dst_inds = CartesianIndices((1:sdm.Nx,1:sdm.Ny,1:1,1:Nt))
    src_inds = CartesianIndices((1:sdm.Nx,1:sdm.Ny,1:1,1:Nt))
    conc1fun!(conc1_onemem::Array{Float64,4},mem::Int64) = begin
        JLD2.jldopen(ens.trajs[mem].history, "r") do f
            copyto!(conc1_onemem, dst_inds, f["conc_hist"], src_inds)
            #conc1_onemem[1:sdm.Nx,1:sdm.Ny,1:Nt] .= f["conc_hist"][:,:,1,:]
        end
        conc1_onemem .-= conc1_zonal_mean
    end
    #Ndsc_per_leadtime = div(cfg.num_perts_max, Nleadtime)
    Ndsc_per_leadtime = div(Ndsc, Nleadtime*Nanc)
    # Various notions of similarity based on contours
    globcorr,contsymdiff,contcorr,= (zeros(Float64, (Nt, Nleadtime, Ndsc_per_leadtime, Nanc)) for _=1:3)

    # pre-allocate arrays for fast in-place correlation calculations 
    # spatially resolved fields
    # (c,i) --> concentration, indicator that concentration is over threshold
    # (x,m,v) --> (dependent on x,y, mean over x,y, variance over x,y)
    # (a,d) --> (ancestor,descendant)
    (cxa,cxd) = (zeros(Float64,(sdm.Nx,sdm.Ny,1,Nt)) for _=1:2)
    (cma,cmd,cma2,cmd2,cmaa,cva,cmdd,cmad) = (zeros(Float64,(1,1,1,Nt)) for _=1:8)
    (ixa,ixd) = (zeros(Float64,(sdm.Nx,1,1,Nt)) for _=1:2)
    (ima,imd,ima2,imd2,imaa,iva,imdd,imad) = (zeros(Float64,(1,1,1,Nt)) for _=1:8)
    #Threads.@threads for i_anc = 1:Nanc
    for i_anc = 1:Nanc

        anc = coast.ancestors[i_anc]
        dscs = Graphs.outneighbors(ens.famtree, anc)

        # Compute the intermediates for the ancestor
        # continuous-valued concentrations
        conc1fun!(cxa,anc)
        SB.mean!(cma, cxa)
        SB.mean!(cmaa, cxa.^2)
        cma2 .= cma.^2
        cva .= cmaa .- cma2
        # Indicators
        ixa .= cxa[1:sdm.Nx,iytgt:iytgt,1:1,:] #contour_level)
        #@infiltrate
        SB.mean!(ima, ixa)
        SB.mean!(imaa, ixa.^2)
        ima2 .= ima.^2
        iva .= imaa .- ima2

        println("Beginning to compute correlations for ancestor $(i_anc), who has $(length(dscs)) total descendants")
        for (i_leadtime,leadtime) in enumerate(leadtimes)
            print("$(i_leadtime), ")
            idx_dsc = desc_by_leadtime(coast, i_anc, leadtime, sdm)
            it1 = cfg.lead_time_max - leadtime
            for i_dsc = 1:Ndsc_per_leadtime
                # Compute the intermediates for the descendant
                dsc = dscs[idx_dsc[i_dsc]]
                conc1fun!(cxd,dsc)
                SB.mean!(cmd, cxd)
                SB.mean!(cmdd, cxd.^2)
                cmd2 .= cmd.^2
                SB.mean!(cmad, cxd.*cxa)
                ixd .= cxd[1:sdm.Nx,iytgt:iytgt,1:1,1:Nt] #contour_level)
                SB.mean!(imd, ixd)
                SB.mean!(imdd, ixd.^2)
                imd2 .= imd.^2
                SB.mean!(imad, ixd.*ixa)

                # Fill in the relevant slice of the array
                globcorr[:,i_leadtime,i_dsc,i_anc] .= ((cmad .- cma.*cmd) ./ sqrt.(cva .* (cmdd .- cmd2)))[1,1,1,:]
                contcorr[:,i_leadtime,i_dsc,i_anc] .= ((imad .- ima.*imd) ./ sqrt.(iva .* (imdd .- imd2)))[1,1,1,:]
                @infiltrate false
                
                #globcorr[:,i_leadtime,i_dsc,i_anc] .= SB.mean(cxd .* cxa; dims=[1,2])[1,1,1,:] .- baseline_globcorr
                #contcorr[:,i_leadtime,i_dsc,i_anc] .= SB.mean(ixd .* ixa; dims=[1,2])[1,1,1,:] .- baseline_contcor
                
                @assert all(-1 .<= globcorr[:,i_leadtime,i_dsc,i_anc] .<= 1)
                @assert all(-1 .<= contcorr[:,i_leadtime,i_dsc,i_anc] .<= 1)
                contsymdiff[:,i_leadtime,i_dsc,i_anc] .= SB.mean(ixa .!= ixd; dims=[1,2])[1,1,1,:]
            end
        end
        println()
    end

    JLD2.jldopen(contour_dispersion_filename, "w") do f
        f["globcorr"] = globcorr
        f["contcorr"] = contcorr
        f["contsymdiff"] = contsymdiff
    end
    #@infiltrate
end
            

