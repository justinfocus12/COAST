struct PhysicalParams
    beta::Float64
    kappa::Float64
    raddamp::Float64
    nu::Float64
    topo_ky::Int64
    topo_amp::Float64
    tr_bvals::Vector{Float64}
end

function PhysicalParams(
        ;
        beta = 0.25,
        kappa = 0.05,
        raddamp = 0.0,
        nu_cuberoot = 0.292,
        topo_ky = 1,
        topo_amp = 0.1, # amplitude of topography relative to beta*Ly 
        tr_bval_lo = 0.0,
        tr_bval_hi = 1.0,
    )
    return PhysicalParams(beta,kappa,raddamp,nu_cuberoot^3,topo_ky,topo_amp,[tr_bval_lo,tr_bval_hi])
end


function strrep_PhysicalParams(php::PhysicalParams)
    s = Printf.@sprintf("beta%g_kappa%g_raddamp%g_nuqr%g_topo%.1fsin%dy_bvtr%g%g", php.beta, php.kappa, php.raddamp, php.nu^(1/3), php.topo_amp, php.topo_ky, php.tr_bvals...,)
    s = replace(s, "."=>"p")
    return s
end

struct SpaceDomain
    tu::Float64 # time unit 
    Lx::Float64
    Ly::Float64
    Nx::Int64
    Ny::Int64
    Kx::Int64 
    Ky::Int64
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    kxgrid::Vector{Int64}
    kygrid::Vector{Int64}
    kxgrid_phys::Vector{Float64}
    kygrid_phys::Vector{Float64}
end

function strrep_SpaceDomain(sdm::SpaceDomain)
    s = Printf.@sprintf("L%dx%d_K%dx%d", sdm.Lx/(2pi), sdm.Ly/(2pi), sdm.Kx, sdm.Ky)
    s = replace(s, "."=>"p")
    return s
end

function SpaceDomain(tu,Lx,Nx,Ly,Ny)
    xgrid,kxgrid,kxgrid_phys = build_grids_1d(Lx,Nx)
    ygrid,kygrid,kygrid_phys = build_grids_1d(Ly,Ny)
    Kx = floor(Int,Nx/3) - 1
    Ky = floor(Int,Ny/3) - 1
    return SpaceDomain(tu,Lx,Ly,Nx,Ny,Kx,Ky,xgrid,ygrid,kxgrid,kygrid,kxgrid_phys,kygrid_phys)
end

mutable struct FlowField
    ok::Array{ComplexF64,3} # frequency domain
    ox::Array{Float64,3}  # spatial domain 
end

function FlowField(Nx,Ny)
    fok = zeros(ComplexF64, (div(Nx,2)+1,Ny,2))
    fox = zeros(Float64, (Nx,Ny,2))
    return FlowField(fok,fox)
end

mutable struct FlowFieldHistory
    tgrid::Vector{Int64}
    ok::Array{ComplexF64,4}
    ox::Array{Float64,4}
end


function FlowFieldHistory(tgrid,Nx,Ny)
    Nt = length(tgrid)
    fok = zeros(ComplexF64, (div(Nx,2)+1,Ny,2,Nt))
    fox = zeros(Float64, (Nx,Ny,2,Nt))
    return FlowFieldHistory(tgrid,fok,fox)
end

function concatenate_FlowFieldHistories(ffhists)
    tgrid = vcat((ffhist.tgrid for ffhist in ffhists)...)
    fok = cat((ffhist.ok for ffhist in ffhists)...; dims=4)
    fox = cat((ffhist.ox for ffhist in ffhists)...; dims=4)
    return FlowFieldHistory(tgrid, fok, fox)
end

function subsample_field_history(fhist::Array{Float64,4}, tgrid::Vector{Int64}, tinit::Int64, tfin::Int64)
    it_init = findfirst(tgrid .== tinit)
    it_fin = findfirst(tgrid .== tfin)
    return fhist[:,:,:,it_init:it_fin]
end

function subsample_FlowFieldHistory(ffhist::FlowFieldHistory, tinit::Int64, tfin::Int64)
    it_init = findfirst(ffhist.tgrid .== tinit)
    it_fin = findfirst(ffhist.tgrid .== tfin)
    @show it_init, it_fin
    return FlowFieldHistory(ffhist.tgrid[it_init:it_fin], ffhist.ok[:,:,:,it_init:it_fin], ffhist.ox[:,:,:,it_init:it_fin])
end

mutable struct FlowState
    tph::Float64 # physical time 
    sf::FlowField
    conc::Array{Float64,3}
end

function FlowState(tph::Float64, Nx::Int64, Ny::Int64)
    sf = FlowField(Nx,Ny)
    conc = zeros(Float64,(Nx,Ny,2))
    return FlowState(tph,sf,conc)
end


mutable struct FlowStateObservables
    # Not all fields need to be defined at once; this is meant both for intermediate calculations of tendency, and for post-analysis
    # <f>_d<vbl> means the derivative of <f> with respect to <vbl>
    tph::Float64 # physical time 
    sf::FlowField # streamfunction
    sf_dx::FlowField
    sf_dy::FlowField
    sf_dt::FlowField
    sf_dt_lin::FlowField
    sf_dt_nonlin::FlowField
    sf_fwdmap::FlowField
    sf_fwdmap_lin::FlowField
    sf_fwdmap_nonlin::FlowField
    pv::FlowField # potential vorticity 
    pv_dx::FlowField
    pv_dy::FlowField
    pv_dt::FlowField
    pv_dt_lin::FlowField
    pv_dt_nonlin::FlowField
    jac_sf_pv::FlowField # Jacobian J(sf, pv)

    # finite volume version
    conc::Array{Float64,3}
    conc_fwdmap::Array{Float64,3}
    flux_coefs_xhi::Array{Float64,3} #
    flux_coefs_xlo::Array{Float64,3} #
    flux_coefs_yhi::Array{Float64,3} #
    flux_coefs_ylo::Array{Float64,3} #

    # Spectral version 
    #conc::FlowField # potential vorticity 
    #conc_dx::FlowField
    #conc_dy::FlowField
    #conc_dt::FlowField
    #conc_dt_lin::FlowField
    #conc_dt_nonlin::FlowField
    #conc_fwdmap::FlowField
    #conc_fwdmap_lin::FlowField
    #conc_fwdmap_nonlin::FlowField
    #jac_sf_conc::FlowField # Jacobian J(sf, conc)
end

function FlowStateObservables(Nx,Ny)
    tph = 0.0
    ffs = (FlowField(Nx,Ny) for _ = 1:16)
    conc,conc_fwdmap = (zeros(Float64, (Nx,Ny,2)) for _ = 1:2)
    flux_coefs_xhi = zeros(Float64,(Nx,Ny,2))
    flux_coefs_xlo = zeros(Float64,(Nx,Ny,2))
    flux_coefs_yhi = zeros(Float64,(Nx,Ny,2))
    flux_coefs_ylo = zeros(Float64,(Nx,Ny,2))
    return FlowStateObservables(tph, ffs..., conc, conc_fwdmap, flux_coefs_xhi, flux_coefs_xlo, flux_coefs_yhi, flux_coefs_ylo)
end

struct ConstantOperators
    Dx::Array{ComplexF64,2}
    Dy::Array{ComplexF64,2}
    Dxx::Array{Float64,2}
    Dyy::Array{Float64,2}
    Lap::Array{Float64,2}
    # For the differential equation K * du/dt = L*u + F(u)
    # Transform to the discrete equation u(t+dt) = A*u(t) + B*F(u(t))
    K::Array{Float64,4} # the operator taking streamfunction to potential vorticity
    L::Array{ComplexF64,4}
    Kinv::Array{Float64,4}
    KinvL::Array{ComplexF64,4}
    dtph_solve::Float64
    A::Array{ComplexF64,4}
    B::Array{ComplexF64,4}
    evals_ctime::Array{ComplexF64,3}
    evecs_ctime::Array{ComplexF64,4}
    evecs_ctime_inv::Array{ComplexF64,4}
    evals_dtime::Array{ComplexF64,3}
    evecs_dtime::Array{ComplexF64,4}
    evecs_dtime_inv::Array{ComplexF64,4}
    topography::Array{Float64,3}
    topography_dx::Array{Float64,3}
    topography_dy::Array{Float64,3}
    rfftplan::FFTW.rFFTWPlan
    irfftplan::FFTW.Plan
    dealias_mask::Array{Bool,2}
end

function ConstantOperators(php::PhysicalParams, sdm::SpaceDomain, dtph_solve::Float64; implicitude::Float64=0.5)
    # General form for ODE: K * du/dt = L*u + F
    Nxhalf = div(sdm.Nx,2)+1
    (Dx,Dy) = (zeros(ComplexF64, (Nxhalf,sdm.Ny)) for _=1:2)
    (Dxx,Dyy,Lap) = (zeros(Float64, (Nxhalf,sdm.Ny)) for _=1:3)
    (K,Kinv) = (zeros(Float64, (Nxhalf,sdm.Ny,2,2)) for _=1:2)
    (L,KinvL,A,B) = (zeros(ComplexF64, (Nxhalf,sdm.Ny,2,2)) for _=1:4)

    (topography,topography_dx,topography_dy) = (zeros(Float64, (sdm.Nx,sdm.Ny,2)) for _=1:3)
    topography[:,:,2] .= php.topo_amp * sin.(php.topo_ky * 2*pi*sdm.ygrid/sdm.Ly)'
    topography_dx[:,:,2] .= 0
    topography_dy[:,:,2] .= php.topo_amp * 2*pi*php.topo_ky/sdm.Ly * cos.(php.topo_ky * 2*pi*sdm.ygrid/sdm.Ly)'


    # Perform diagonalization 
    evals_ctime = zeros(ComplexF64, (Nxhalf,sdm.Ny,2))
    evecs_ctime = zeros(ComplexF64, (Nxhalf,sdm.Ny,2,2))
    evecs_ctime_inv = zeros(ComplexF64, (Nxhalf,sdm.Ny,2,2))
    evals_dtime = zeros(ComplexF64, (Nxhalf,sdm.Ny,2))
    evecs_dtime = zeros(ComplexF64, (Nxhalf,sdm.Ny,2,2))
    evecs_dtime_inv = zeros(ComplexF64, (Nxhalf,sdm.Ny,2,2))
    #
    dealias_mask = zeros(Bool, (Nxhalf,sdm.Ny))
    U = [1.0, 0.0] 
    kappa = php.kappa * [0.0, 1.0]

    for iy = 1:sdm.Ny
        ky = sdm.kygrid[iy]
        kyp = sdm.kygrid_phys[iy]
        for ix = 1:Nxhalf
            kx = sdm.kxgrid[ix]
            kxp = sdm.kxgrid_phys[ix]
            dealias_mask[ix,iy] = (abs(kx) <= sdm.Kx) && (abs(ky) <= sdm.Ky) 
            #dealias_mask[ix,iy] = ((kx/sdm.Kx)^2 + (ky/sdm.Ky)^2 <= 1)
            if (kx == ky == 0)
                continue
            end
            Dx[ix,iy] = 1im*kxp
            Dy[ix,iy] = 1im*kyp
            Dxx[ix,iy] = -kxp^2
            Dyy[ix,iy] = -kyp^2
            Lap[ix,iy] = -(kxp^2 + kyp^2)
            for iz = 1:2
                jz = mod(iz,2) + 1
                K[ix,iy,iz,iz] = Lap[ix,iy] - 1/2
                K[ix,iy,iz,jz] = 1/2
            end
            for iz = 1:2
                L[ix,iy,iz,iz] = -Dx[ix,iy]*(php.beta - (-1)^iz*(U[1] - U[2])/2)
                L[ix,iy,iz,:] .-= U[iz]*Dx[ix,iy] * K[ix,iy,iz,:]
            end
            stopcond = false
            Kinv[ix,iy,:,:] .= LA.inv(K[ix,iy,:,:]) 
            stopcond = false && (kx == ky == 20)

            if stopcond
                println("---------before dissipation----------")
                println("Kinv eig = ")
                display(LA.eigen(Kinv[ix,iy,:,:]))
                println("L eig = ")
                display(LA.eigen(L[ix,iy,:,:]))
                println("KinvL eig  = ")
                display(LA.eigen(Kinv[ix,iy,:,:] * L[ix,iy,:,:]))
            end
            for iz = 1:2
                L[ix,iy,iz,iz] -= kappa[iz] * Lap[ix,iy]
                L[ix,iy,iz,iz] -= php.nu * Lap[ix,iy]^3
            end
            KinvL[ix,iy,:,:] .= Kinv[ix,iy,:,:] * L[ix,iy,:,:]
            if stopcond
                println("---------after dissipation----------")
                println("L eig = ")
                display(LA.eigen(L[ix,iy,:,:]))
                println("KinvL eig  = ")
                display(LA.eigen(KinvL[ix,iy,:,:] * L[ix,iy,:,:]))
            end
            if false && (kx == 3) && (ky == 2)
                println("Dx[$(ix),$(iy)] = $(Dx[ix,iy])")
                println("beta = $(php.beta)")
                println("K[$(kx),$(ky)] = ")
                display(K[ix,iy,:,:])
                println("L[$(kx),$(ky)] = ")
                display(L[ix,iy,:,:])
                println("KinvL[$(kx),$(ky)] = ")
                display(KinvL[ix,iy,:,:])
            end
            B[ix,iy,:,:] .= LA.inv(K[ix,iy,:,:] - dtph_solve*implicitude*L[ix,iy,:,:])
            A[ix,iy,:,:] .= B[ix,iy,:,:] * (K[ix,iy,:,:] + dtph_solve*(1-implicitude)*L[ix,iy,:,:])
            B[ix,iy,:,:] .*= dtph_solve
            # Eigenanalysis
            E = LA.eigen(KinvL[ix,iy,:,:])
            evals_ctime[ix,iy,:] .= E.values
            evecs_ctime[ix,iy,:,:] .= E.vectors
            evecs_ctime_inv[ix,iy,:,:] .= LA.inv(E.vectors)
            E = LA.eigen(A[ix,iy,:,:])
            evals_dtime[ix,iy,:] .= E.values
            evecs_dtime[ix,iy,:,:] .= E.vectors
            evecs_dtime_inv[ix,iy,:,:] .= LA.inv(E.vectors)
        end
    end
    if false
        for (ix,iy) in ((2,3),)
            println("A,B,K,L,KinvL at $(ix),$(iy) = ")
            display(A[ix,iy,:,:])
            display(B[ix,iy,:,:])
            display(K[ix,iy,:,:])
            display(L[ix,iy,:,:])
            display(KinvL[ix,iy,:,:])
            display(LA.inv(K[ix,iy,:,:]-dtph_solve*implicitude*L[ix,iy,:,:])*(K[ix,iy,:,:]+dtph_solve*(1-implicitude)*L[ix,iy,:,:]))
        end
    end
    rfftplan  = FFTW.plan_rfft(zeros(Float64, (sdm.Nx,sdm.Ny,2)), [1,2])
    irfftplan = FFTW.plan_irfft(zeros(ComplexF64, (Nxhalf,sdm.Ny,2)), sdm.Nx, [1,2])

    return ConstantOperators(Dx,Dy,Dxx,Dyy,Lap,K,L,Kinv,KinvL,dtph_solve,A,B,evals_ctime,evecs_ctime,evecs_ctime_inv,evals_dtime,evecs_dtime,evecs_dtime_inv,topography,topography_dx,topography_dy,rfftplan,irfftplan,dealias_mask)
end

function compute_fwdmap_tracers_finvol!(
        conc_fwdmap::Array{Float64, 3},
        conc::Array{Float64, 3},
        flux_coefs_xhi::Array{Float64,3},
        flux_coefs_xlo::Array{Float64,3},
        flux_coefs_yhi::Array{Float64,3},
        flux_coefs_ylo::Array{Float64,3},
        u::Array{Float64,3},
        v::Array{Float64,3},
        sdm::SpaceDomain,
        cop::ConstantOperators,
        php::PhysicalParams,
    )
    dx = sdm.Lx/sdm.Nx
    dy = sdm.Ly/sdm.Ny
    dtph = cop.dtph_solve
    U = [1.0, 0.0]
    for iz = 1:2
        for iy = 1:sdm.Ny
            iyhi = (iy == sdm.Ny ? 1 : iy+1)
            iylo = (iy == 1 ? sdm.Ny : iy-1)
            for ix = 1:sdm.Nx # index for the wall 
                ixhi = (ix == sdm.Nx ? 1 : ix+1)
                ixlo = (ix == 1 ? sdm.Nx : ix-1)
                # right
                uhi = 0.5*(u[ix,iy,iz] + u[ixhi,iy,iz]) + U[iz]
                flux_coefs_xhi[ix,iy,iz] = -min(uhi,0) * (dtph/dx)
                cxhi = conc[ixhi,iy,iz]
                # left
                ulo = 0.5*(u[ix,iy,iz] + u[ixlo,iy,iz]) + U[iz]
                flux_coefs_xlo[ix,iy,iz] = max(ulo,0) * (dtph/dx) 
                cxlo = conc[ixlo,iy,iz]
                # above 
                vhi = 0.5*(v[ix,iy,iz] + v[ix,iyhi,iz])
                flux_coefs_yhi[ix,iy,iz] = -min(vhi,0) * (dtph/dy)
                cyhi = (iy == sdm.Ny ? php.tr_bvals[2] : conc[ix,iyhi,iz])
                # below
                vlo = 0.5*(v[ix,iy,iz] + v[ix,iylo,iz])
                flux_coefs_ylo[ix,iy,iz] = max(vlo,0) * (dtph/dy) 
                cylo = (iy == 1 ? php.tr_bvals[1] : conc[ix,iylo,iz])
                # update the map 
                # check if all the coefficients are convex
                @assert all((f[ix,iy,iz]>=0 for f=(flux_coefs_xhi,flux_coefs_xlo,flux_coefs_yhi,flux_coefs_ylo)))
                sumfluxes = flux_coefs_xhi[ix,iy,iz] + flux_coefs_xlo[ix,iy,iz] + flux_coefs_yhi[ix,iy,iz] + flux_coefs_ylo[ix,iy,iz]
                @assert sumfluxes <= 1

                conc_fwdmap[ix,iy,iz] = (
                                         conc[ix,iy,iz] * (
                                                           1 
                                                           - flux_coefs_xhi[ix,iy,iz]
                                                           - flux_coefs_xlo[ix,iy,iz]
                                                           - flux_coefs_yhi[ix,iy,iz]
                                                           - flux_coefs_ylo[ix,iy,iz]
                                                          )
                                         + flux_coefs_xhi[ix,iy,iz] * cxhi
                                         + flux_coefs_xlo[ix,iy,iz] * cxlo
                                         + flux_coefs_yhi[ix,iy,iz] * cyhi
                                         + flux_coefs_ylo[ix,iy,iz] * cylo
                                        )
            end
        end
    end
end

                                             
function compute_observables!(
        flob::FlowStateObservables,
        flow::FlowState,
        cop::ConstantOperators,
        sdm::SpaceDomain,
        php::PhysicalParams,
    )
    flob.tph = flow.tph
    copy_FlowField!(flob.sf, flow.sf)
    flob.conc .= flow.conc
    # Linear tendency
    compute_gridded_matrix_vector_product!(flob.sf_dt_lin.ok, cop.KinvL, flob.sf.ok)
    synchronize_FlowField_k2x!(flob.sf_dt_lin)
    #@show extrema(flob.conc_dt_lin.ox)
    # Nonlinear tendency
    compute_gridded_matrix_vector_product!(flob.pv.ok, cop.K, flob.sf.ok)
    flob.sf_dx.ok .= cop.Dx .* flob.sf.ok
    flob.sf_dy.ok .= cop.Dy .* flob.sf.ok
    flob.pv_dx.ok .= cop.Dx .* flob.pv.ok
    flob.pv_dy.ok .= cop.Dy .* flob.pv.ok
    for symbol in (:pv,:sf_dx,:sf_dy,:pv_dx,:pv_dy)
        synchronize_FlowField_k2x!(getfield(flob,symbol))
    end
    flob.jac_sf_pv.ox .= flob.sf_dx.ox .* (flob.pv_dy.ox .+ cop.topography_dy) .- flob.sf_dy.ox .* (flob.pv_dx.ox .+ cop.topography_dx)
    synchronize_FlowField_x2k!(flob.jac_sf_pv)
    flob.jac_sf_pv.ok .*= cop.dealias_mask
    # Compute PV tendency
    flob.pv_dt_nonlin.ok .= -flob.jac_sf_pv.ok
    synchronize_FlowField_k2x!(flob.pv_dt_nonlin)
    add!(flob.pv_dt, flob.pv_dt_lin, flob.pv_dt_nonlin)
    # Compute SF tendency
    compute_gridded_matrix_vector_product!(flob.sf_dt_nonlin.ok, cop.Kinv, flob.pv_dt_nonlin.ok)
    synchronize_FlowField_k2x!(flob.sf_dt_nonlin)
    add!(flob.sf_dt, flob.sf_dt_lin, flob.sf_dt_nonlin)
    # Compute SF forward-time map 
    compute_gridded_matrix_vector_product!(flob.sf_fwdmap_lin.ok, cop.A, flob.sf.ok)
    synchronize_FlowField_k2x!(flob.sf_fwdmap_lin)
    compute_gridded_matrix_vector_product!(flob.sf_fwdmap_nonlin.ok, cop.B, flob.pv_dt_nonlin.ok)
    synchronize_FlowField_k2x!(flob.sf_fwdmap_nonlin)
    add!(flob.sf_fwdmap, flob.sf_fwdmap_lin, flob.sf_fwdmap_nonlin)
    # Compute Tracer forward-time map 
    compute_fwdmap_tracers_finvol!(flob.conc_fwdmap, flob.conc, flob.flux_coefs_xhi, flob.flux_coefs_xlo, flob.flux_coefs_yhi, flob.flux_coefs_ylo, -flob.sf_dy.ox, flob.sf_dx.ox, sdm, cop, php)
    return

    # spectral version
    #compute_gridded_matrix_vector_product!(flob.conc_fwdmap_lin.ok, cop.A_tr, flob.conc.ok)
    #synchronize_FlowField_k2x!(flob.conc_fwdmap_lin)
    #compute_gridded_matrix_vector_product!(flob.conc_fwdmap_nonlin.ok, cop.B_tr, flob.conc_dt_nonlin.ok)
    #synchronize_FlowField_k2x!(flob.conc_fwdmap_nonlin)
    #add!(flob.conc_fwdmap, flob.conc_fwdmap_lin, flob.conc_fwdmap_nonlin)
end

struct PerturbationOperator
    sf_pert_modes::Vector{Array{ComplexF64,3}}
    sf_pert_amplitudes_min::Vector{Float64}
    sf_pert_amplitudes_max::Vector{Float64}
    conc_pert_modes::Vector{Array{Float64,3}}
    conc_pert_amplitudes_min::Vector{Float64}
    conc_pert_amplitudes_max::Vector{Float64}
    pert_dim::Int64
end

# specialized constructors for PerturbationOperator

function strrep_PerturbationOperator(pertop::PerturbationOperator, sdm::SpaceDomain)
    return "pertdims$(length(pertop.sf_pert_modes))sf$(length(pertop.conc_pert_modes))conc"
end

function PerturbationOperator_sfphase_bothlayers(sdm::SpaceDomain,kxs::Vector{Int64},kys::Vector{Int64},evecs::Vector{Vector{ComplexF64}})
    @assert length(kxs) == length(kys)
    pert_dim_sf = 2*length(kxs) # magnitude and phase
    Nmodes_sf = length(kxs)
    Nxhalf = div(sdm.Nx,2) + 1
    sf_pert_modes = [zeros(ComplexF64, (Nxhalf,sdm.Ny,2)) for _=1:Nmodes_sf]
    sf_pert_amplitudes_min = 0.0 .* ones(pert_dim_sf)
    sf_pert_amplitudes_max = 0.3 .* ones(pert_dim_sf) # real-space typical magnitude of perturbation 
    for (i_pert,(kx,ky)) in enumerate(zip(kxs,kys))
        ikx = findfirst(sdm.kxgrid .== kx)
        iky = findfirst(sdm.kygrid .== ky)
        sf_pert_modes[i_pert][ikx,iky,:] .= sdm.Nx * sdm.Ny .* evecs[i_pert]
    end
    conc_pert_modes = Vector{Array{Float64,3}}([]) #[zeros(Float64, (sdm.Nx,sdm.Ny,2)) for _=1:1]
    conc_pert_amplitudes_min = Vector{Float64}([])
    conc_pert_amplitudes_max = Vector{Float64}([])
    pert_dim_conc = Int64(0)
    return PerturbationOperator(sf_pert_modes,sf_pert_amplitudes_min,sf_pert_amplitudes_max,conc_pert_modes,conc_pert_amplitudes_min,conc_pert_amplitudes_max,pert_dim_sf+pert_dim_conc)
end


function PerturbationOperator_sfphase_onlytop(sdm::SpaceDomain,kxs::Vector{Int64},kys::Vector{Int64},izs::Vector{Int64})
    # TODO add concentration perturbations
    pert_dim_sf = length(kxs)
    @assert length(kxs) == length(kys) == length(izs) == pert_dim_sf
    Nxhalf = div(sdm.Nx,2) + 1
    sf_pert_modes = [zeros(ComplexF64, (Nxhalf,sdm.Ny,2)) for _=1:pert_dim_sf]
    sf_pert_amplitudes_min = 0.1 .* ones(pert_dim_sf) # real-space typical magnitude of perturbation 
    sf_pert_amplitudes_max = 0.5 .* ones(pert_dim_sf) # real-space typical magnitude of perturbation 
    for (i_pert,(kx,ky,iz)) in enumerate(zip(kxs,kys,izs))
        ikx = findfirst(sdm.kxgrid .== kx)
        iky = findfirst(sdm.kygrid .== ky)
        sf_pert_modes[i_pert][ikx,iky,iz] = sdm.Nx * sdm.Ny
    end
    conc_pert_modes = Vector{Array{Float64,3}}([]) #[zeros(Float64, (sdm.Nx,sdm.Ny,2)) for _=1:1]
    conc_pert_amplitudes_max = Vector{Float64}([])
    pert_dim_conc = Int64(0)
    return PerturbationOperator(sf_pert_modes,sf_pert_amplitudes_min,sf_pert_amplitudes_max,conc_pert_modes,conc_pert_amplitudes_min,conc_pert_amplitudes_max,pert_dim_sf+pert_dim_conc)
end

function plot_PerturbationOperator(pertop::PerturbationOperator, sdm::SpaceDomain, pert_seq::Matrix{Float64}, savedir::String)
    num_pert_modes_sf,num_pert_modes_conc = length.((pertop.sf_pert_modes,pertop.conc_pert_modes))
    lblargs = Dict(:xticklabelsize=>10,:xlabelsize=>12,:yticklabelsize=>10,:ylabelsize=>12,)
    for i_mode = 1:num_pert_modes_sf
        fig = Figure(size=(225,400))
        lout = fig[1:2,1:2] = GridLayout()
        mode = FFTW.irfft(pertop.sf_pert_modes[i_mode], sdm.Nx, [1,2])
        for iz=1:2
            ax = Axis(lout[iz,1]; xlabel=L"$x$", ylabel=L"$y$", title=L"$\delta\Psi_{%$(iz)}$", lblargs...)
            img = image!(ax, (0,sdm.Lx), (0,sdm.Ly), mode[:,:,iz]; colormap=:BrBg)
            cbar = Colorbar(lout[iz,2], img, vertical=true)
        end
        colsize!(lout, 1, Relative(8/9))
        save(joinpath(savedir,"pert_sf_mode$(i_mode).png"), fig)
    end
    # Plot only the zonal part for both layers 
    flow = FlowState(0.0, sdm.Nx, sdm.Ny)
    Npert = size(pert_seq, 2)
    fig = Figure(size=(480,80*(Npert)))
    lout = fig[1,1] = GridLayout()
    axs = [Axis(lout[i_pert,1]; xlabel="𝑥/𝐿", title="Streamfunction perturbation δψ(ω⁽ᵐ⁾)", ylabel="𝑚 = $(i_pert)", ylabelrotation=0, xlabelsize=12, xticklabelsize=9, ylabelsize=12, yticklabelsize=9, xticklabelsvisible=(i_pert==Npert), xlabelvisible=(i_pert==Npert), titlevisible=(1==i_pert), titlefont=:regular, xgridvisible=false, ygridvisible=false) for i_pert=1:Npert]
    for i_pert = 1:Npert
        flow.sf.ok .= 0
        flow.sf.ox .= 0
        flow.conc .= 0
        perturb!(flow, pert_seq[:,i_pert], pertop)
        ax = axs[i_pert]
        lines!(ax, sdm.xgrid./sdm.Lx, flow.sf.ox[:,1,1]; color=:deepskyblue, label="𝑧 = 1")
        lines!(ax, sdm.xgrid./sdm.Lx, flow.sf.ox[:,1,2]; color=:sienna, label="𝑧 = 2")
        if i_pert < Npert
            rowgap!(lout, i_pert, 0)
        end
        xlims!(ax, 0, 1)
        ylims!(ax, (2*pertop.sf_pert_amplitudes_max[1]*1.01 .* [-1,1])...)
    end
    leg = Legend(lout[1,2], axs[Npert]; framevisible=true, labelsize=12)
    colsize!(lout, 1, Relative(4/5))
    save(joinpath(savedir,"pert_sf_samples.png"), fig)
        
end



struct PerturbationSequence 
    ts_ph::Vector{Float64}
    perts::Vector{Vector{Float64}}
end

function NullPerturbationSequence()
    ts_ph = Vector{Float64}([])
    perts = Vector{Vector{Float64}}([])
    return PerturbationSequence(ts_ph, perts)
end


