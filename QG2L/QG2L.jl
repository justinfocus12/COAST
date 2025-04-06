module QG2L

import FFTW
import Printf
using Printf: @sprintf
import LinearAlgebra as LA
import StatsBase as SB
import ZernikePolynomials as ZP
import Distributions as Dists
import Extremes as Ext
import LsqFit
import Random
import JLD2
import QuasiMonteCarlo as QMC
using CairoMakie
using Infiltrator: @infiltrate

#
include("./QG2L_typedefs.jl")
include("./QG2L_dynamics.jl")
include("./QG2L_utils.jl")
include("./QG2L_plots.jl")
include("./QG2L_stats_1.jl")
include("./QG2L_stats_2.jl")

function expt_config(; i_expt=nothing)
    # Domain parameters
    Lx,Ly = (2*pi*6 for _=1:2)
    sdm = SpaceDomain(1.0, Lx, 64, Ly, 64)
    # Constant physical parameters
    const_params = Dict(
        :kappa => 0.05,
        :nu_cuberoot => 0.292,
        :tr_bval_lo => 0.0,
        :tr_bval_hi => 1.0,
       )
    # Variable physical parameters
    topo_amps = [0.0, 0.25, 0.5]
    topo_kys = [1, 2]
    betas = [0.25, 0.35]
    vbl_param_arrs = [topo_amps, topo_kys, betas]

    cartinds = CartesianIndices(tuple((length(arr) for arr in vbl_param_arrs)...))
    @show cartinds
    if isnothing(i_expt) | (i_expt == 0)
        ci_expt = CartesianIndex(3,2,1)
        #ci_expt = CartesianIndex(1,2,1)
    else
        ci_expt = cartinds[i_expt]
    end
    @show ci_expt
    # TODO remove redundant indices
    topo_amp,topo_ky,beta = (vbl_param_arrs[j][ci_expt[j]] for j=1:length(vbl_param_arrs))
    

    php = PhysicalParams(; const_params..., beta=beta, topo_ky=topo_ky, topo_amp=topo_amp)

    return (php,sdm)
end

function expt_setup(php, sdm; pert_mag_ox=0.1)
    dtph_solve = 0.025
    cop = ConstantOperators(php, sdm, dtph_solve; implicitude=0.5)
    instab = maximum(real.(cop.evals_ctime), dims=3) .* cop.dealias_mask
    # most-unstable mode
    option = "mumboth"
    if "mumtop" == option
        most_unstable_mode = argmax(instab)
        kx_mum = sdm.kxgrid[most_unstable_mode[1]]
        ky_mum = sdm.kygrid[most_unstable_mode[2]]
        if ky_mum == 0
            kxs = [kx_mum]
            kys = [ky_mum]
            izs = [1]
        else
            kxs = [kx_mum,kx_mum]
            kys = [ky_mum,-ky_mum]
            izs = [1,1]
        end
        pertop = PerturbationOperator_sfphase_onlytop(sdm, kxs, kys, izs)
    elseif "mumboth" == option
        most_unstable_mode = argmax(instab)
        ix,iy = (most_unstable_mode[i] for i=1:2)
        kx_mum = sdm.kxgrid[ix]
        ky_mum = sdm.kygrid[iy]
        which_mode = argmax(real.(cop.evals_ctime[ix,iy,:]))
        kxs = [kx_mum]
        kys = [ky_mum]
        evecs = [cop.evecs_ctime[ix,iy,:,which_mode]]
        pertop = PerturbationOperator_sfphase_bothlayers(sdm, kxs, kys, evecs)
    elseif "mumzon" == option
        kxs = [0]
        kys = [sdm.kygrid[argmax(instab[1,:])]]
        izs = [1]
        pertop = PerturbationOperator_sfphase_onlytop(sdm, kxs, kys, izs)
    end
    @show instab[1,:]
    @show kxs,kys
    @show length(pertop.sf_pert_modes)
    return (cop,pertop)
end

# ------------ Standard observables we use a lot ----------
function obs_fun_pv_hist(f::JLD2.JLDFile, sdm::SpaceDomain, cop::ConstantOperators)
    return obs_fun_pv_hist(f["tgrid"], f["sf_hist_ok"], sdm, cop)
end

function obs_fun_pv_hist(tgrid::Vector{Int64}, sf_hist_ok::Array{ComplexF64,4}, sdm::SpaceDomain, cop::ConstantOperators)
    pv_hist = FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
    pv_hist.ok[:,:,1,:] .= -(sf_hist_ok[:,:,1,:] - sf_hist_ok[:,:,2,:])/2
    pv_hist.ok[:,:,2,:] .= -pv_hist.ok[:,:,1,:]
    pv_hist.ok .+= cop.Lap .* sf_hist_ok
    synchronize_FlowField_k2x!(pv_hist)
    return pv_hist.ox 
end

function obs_fun_eke_hist(f::JLD2.JLDFile, sdm::SpaceDomain, cop::ConstantOperators)
    return obs_fun_eke_hist(f["tgrid"],f["sf_hist_ok"], sdm, cop)
end
function obs_fun_eke_hist(tgrid::Vector{Int64}, sf_hist_ok::Array{ComplexF64,4}, sdm::SpaceDomain, cop::ConstantOperators)
    u_hist = FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
    v_hist = FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
    u_hist.ok .= -cop.Dy .* sf_hist_ok
    synchronize_FlowField_k2x!(u_hist)
    v_hist.ok .= cop.Dx .* sf_hist_ok
    synchronize_FlowField_k2x!(v_hist)
    eke = 0.5 * (u_hist.ox.^2 + v_hist.ox.^2)
    return eke
end

function obs_fun_temperature_hist(f::JLD2.JLDFile, sdm::SpaceDomain, cop::ConstantOperators)
    return obs_fun_temperature_hist(f["tgrid"],f["sf_hist_ok"],sdm,cop)
end
function obs_fun_temperature_hist(tgrid::Vector{Int64}, sf_hist_ok::Array{ComplexF64,4}, sdm::SpaceDomain, cop::ConstantOperators)
    T_hist = FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)

    T_hist.ok .= (sf_hist_ok[:,:,1:1,:] - sf_hist_ok[:,:,2:2,:])/2
    synchronize_FlowField_k2x!(T_hist)
    return T_hist.ox
end

function obs_fun_sf_hist(f::JLD2.JLDFile, sdm::SpaceDomain, cop::ConstantOperators)
    return f["sf_hist_ox"]
end
function obs_fun_sf_hist(tgrid::Vector{Int64}, sf_hist_ox::Array{Float64,4}, sdm::SpaceDomain, cop::ConstantOperators)
    return sf_hist_ox
end

function obs_fun_conc_hist(f::JLD2.JLDFile, sdm::SpaceDomain, cop::ConstantOperators)
    return f["conc_hist"]
end
    
function obs_fun_heatflux_hist(f::JLD2.JLDFile, sdm::SpaceDomain, cop::ConstantOperators)
    return obs_fun_heatflux_hist(f["tgrid"],f["sf_hist_ok"],sdm,cop)
end
function obs_fun_heatflux_hist(tgrid::Vector{Int64}, sf_hist_ok::Array{ComplexF64,4}, sdm::SpaceDomain, cop::ConstantOperators)
    T_hist = FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)

    T_hist.ok .= (sf_hist_ok[:,:,1:1,:] .- sf_hist_ok[:,:,2:2,:])/2
    synchronize_FlowField_k2x!(T_hist)
    vbt_hist = FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny) 
    vbt_hist.ok .= cop.Dx .* (sf_hist_ok[:,:,1:1,:] + sf_hist_ok[:,:,2:2,:])/2
    synchronize_FlowField_k2x!(vbt_hist)

    heatflux_hist = FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny)
    heatflux_hist.ox .= vbt_hist.ox .* T_hist.ox
    return heatflux_hist.ox
end

function obs_fun_meridional_velocity_hist(f::JLD2.JLDFile, sdm::SpaceDomain, cop::ConstantOperators)
    return obs_fun_meridional_velocity_hist(f["tgrid"],f["sf_hist_ok"],sdm,cop)
end
function obs_fun_meridional_velocity_hist(tgrid::Vector{Int64}, sf_hist_ok::Array{ComplexF64,4}, sdm::SpaceDomain, cop::ConstantOperators)
    v_hist = FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny) 
    v_hist.ok .= cop.Dx .* sf_hist_ok
    synchronize_FlowField_k2x!(v_hist)

    return v_hist.ox
end

function obs_fun_mean_energy_density(f::JLD2.JLDFile, cop::ConstantOperators, sdm::SpaceDomain)
    return obs_fun_mean_energy_density(f["tgrid"], f["sf_hist_ok"], cop, sdm)
end

function obs_fun_mean_energy_density(tgrid::Vector{Int64}, sf_hist_ok::Array{ComplexF64,4}, cop::ConstantOperators, sdm::SpaceDomain)
    sf_dx_ok = cop.Dx .* sf_hist_ok
    sf_dy_ok = cop.Dy .* sf_hist_ok
    E = (area_mean_square(sf_dx_ok,sdm.Nx,sdm.Ny) + area_mean_square(sf_dy_ok,sdm.Nx,sdm.Ny))/2
    @show size(E)
    return E
end



end # module QG2L





