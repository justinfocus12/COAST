# ------------ Utility functions ----------------------
function shortfmt(xrange::Float64)
    @show xrange
    if 0 <= xrange < 0.01
        fmt = "{:.1e}"
    elseif 0.01 <= xrange < 0.1
        fmt = "{:.2f}"
    elseif 0.1 <= xrange < 1
        fmt = "{:.2f}"
    elseif 1 <= xrange < 10000
        fmt = "{:.1f}"
    elseif 10000 <= xrange
        fmt = "{:.1e}"
    end
    @show fmt 
    return fmt 
end

function quantile_sliced(X::Array{Float64}, q::Float64, dims::Integer)
    qslices = mapslices(x->SB.quantile(x, q), X; dims=dims)
    return qslices
end



function any_local_extrema_xperiodic_ynonperiodic(u::Matrix{Float64})
    dxhi = circshift(u, (-1,0)) .- u
    dyhi = circshift(u, (0,-1)) .- u
    dxlo = -circshift(dxhi, (1,0))
    dylo = -circshift(dyhi, (0,1))
    local_min = (dxhi .>= 0) .& (dxlo .>= 0) .& (dyhi .>= 0) .& (dylo .>= 0) .& ((dxhi .> 0) .| (dxlo .> 0) .| (dyhi .> 0) .| (dylo .> 0))
    local_max = (dxhi .<= 0) .& (dxlo .<= 0) .& (dyhi .<= 0) .& (dylo .<= 0) .& ((dxhi .< 0) .| (dxlo .< 0) .| (dyhi .< 0) .| (dylo .< 0))
    any_local_max = any(loc_max[:,2:end-1])
    any_local_min = any(loc_min[:,2:end-1])
    return (any_local_max) | (any_local_min)
end
    




function synchronize_FlowField_k2x!(f::Union{FlowField,FlowFieldHistory})
    f.ox .= FFTW.irfft(f.ok, size(f.ox,1), [1,2])
end
function synchronize_FlowField_x2k!(f::Union{FlowField,FlowFieldHistory})
    f.ok .= FFTW.rfft(f.ox, [1,2])
end

function synchronize_FlowField_k2x!(f::FlowField, cop::ConstantOperators)
    #f.ox .= FFTW.irfft(f.ok, size(f.ox,1), [1,2])
    f.ox .= cop.irfftplan * f.ok
end
function synchronize_FlowField_x2k!(f::FlowField, cop::ConstantOperators)
    FFTW.mul!(f.ok, cop.rfftplan, f.ox)
    #f.ok .= FFTW.rfft(f.ox, [1,2])
end

function add!(fplusg::FlowField, f::FlowField, g::FlowField)
    fplusg.ox .= f.ox + g.ox
    fplusg.ok .= f.ok + g.ok
end

function copy_FlowField!(f::FlowField, g::FlowField)
    f.ox .= g.ox
    f.ok .= g.ok
    return
end

function copy_FlowState!(flow2::FlowState, flow1::FlowState)
    copy_FlowField!(flow2.sf, flow1.sf)
    flow2.conc .= flow1.conc
    flow2.tph = flow1.tph
    return
end

function copy_FlowStateObservables!(flob2::FlowStateObservables, flob1::FlowStateObservables)
    vars_FlowField = (
                      :sf,
                      :sf_dx,
                      :sf_dy,
                      :sf_dt,
                      :sf_dt_lin,
                      :sf_dt_nonlin,
                      :sf_fwdmap,
                      :sf_fwdmap_lin,
                      :sf_fwdmap_nonlin,
                      :pv, # potential vorticity 
                      :pv_dx,
                      :pv_dy,
                      :pv_dt,
                      :pv_dt_lin,
                      :pv_dt_nonlin,
                      :jac_sf_pv, 
                     )
    vars_Array = (
                  :conc,
                  :conc_fwdmap,
                  :flux_coefs_xhi,
                  :flux_coefs_xlo,
                  :flux_coefs_yhi,
                  :flux_coefs_ylo, 
                 )
    flob2.tph = flob1.tph
    for fn = vars_FlowField
        copy_FlowField!(getfield(flob2, fn), getfield(flob1, fn))
    end
    for fn = vars_Array
        getfield(flob2, fn) .= getfield(flob1,fn)
    end
    return
end

function mean_energy_density(sf::Union{FlowField,FlowFieldHistory}, cop::ConstantOperators, sdm::SpaceDomain)
    sf_dx_ok = cop.Dx .* sf.ok
    sf_dy_ok = cop.Dy .* sf.ok
    E = (area_mean_square(sf_dx_ok,sdm.Nx,sdm.Ny) .+ area_mean_square(sf_dy_ok,sdm.Nx,sdm.Ny)) ./ 2
    return E
end

function energy_per_mode(sf::Union{FlowField,FlowFieldHistory}, cop::ConstantOperators, sdm::SpaceDomain)
    return (abs2.(cop.Dx .* sf.ok) .+ abs2.(cop.Dy .* sf.ok)) ./ 2
end

function area_mean_square(fox::Array{Float64})
    Nx,Ny = [size(fox,i) for i=1:2]
    mse = sum(fox.^2, dims=[1,2]) ./ (Nx*Ny)
    return mse
end

function area_mean_square(fok::Array{ComplexF64}, Nx::Int64, Ny::Int64)
    Kx = size(fok,1)
    norm2f_0 = sum(abs2.(selectdim(fok, 1, 1:1)), dims=[1,2])
    norm2f_1 = 2 .* sum(abs2.(selectdim(fok, 1, 2:(Nx-Kx+1))), dims=[1,2])
    norm2f_2 = sum(abs2.(selectdim(fok, 1, (Nx-Kx+2):Kx)), dims=[1,2])
    norm2f_blocks = cat(norm2f_0, norm2f_1, norm2f_2; dims=1)
    #display(norm2f_blocks)
    norm2f = sum(norm2f_blocks, dims=[1,2]) ./ (Nx*Ny)^2
    return norm2f
end

function test_area_mean_square()
    Nx = 11
    Ny = 21
    x = collect(range(0,2pi,Nx+1)[1:end-1])
    y = collect(range(0,2pi,Ny+1)[1:end-1])
    f = FlowField(Nx,Ny)
    rng = Random.MersenneTwister(2038)
    #f.ox .= 3 #sin.(x) #.+ cos.(y)' #.+ sin.(2*x) .+ cos.(2*y)'
    f.ox .= Random.randn(rng, Float64, (Nx,Ny))
    synchronize_FlowField_x2k!(f)
    ams_ok = area_mean_square(f.ok, Nx, Ny)
    @show ams_ok[:,:,1]
    ams_ox = sum(f.ox.^2, dims=[1,2])/(Nx*Ny)
    @show ams_ox[:,:,1]
    @show abs.(ams_ok - ams_ox)
    println("fox = ")
    display(f.ox[:,:,1])
    println("FFT(fox) = ")
    display(FFTW.fft(f.ox[:,:,1], [1,2]))
    println("fok = ")
    display(f.ok[:,:,1])
    return
end



function compute_gridded_matrix_vector_product!(
        # intent(out)
        Au::Array{<:Number,3},
        # intent(in)
        A::Array{<:Number,4},
        u::Array{<:Number,3},
    )
    Au .= 0.0
    for j = 1:size(A,4)
        for i = 1:size(A,3)
            Au[:,:,i] .+= A[:,:,i,j] .* u[:,:,j]
        end
    end
end

function build_grids_1d(L,N)
    xgrid = collect(range(0,L,N+1)[1:end-1])
    Nhalf = div(N,2)
    kgrid = zeros(Int64, N)
    kgrid[1:Nhalf+1] .= 0:Nhalf
    kgrid[Nhalf+2:N] .= -(N-Nhalf-1):-1
    kgrid_phys = kgrid * 2*pi/L
    return xgrid,kgrid,kgrid_phys
end

# --------------- Standard constructors -----------------
function initialize_FlowField_baroclinic(kxs::Vector{Int64}, kys::Vector{Int64}, amplitudes::Vector{Float64}, cop::ConstantOperators, sdm::SpaceDomain)
    sf = FlowField(sdm.Nx, sdm.Ny)
    for iz = 1:2
        ix = mod(abs(kxs[iz]), sdm.Nx) + 1
        iy = mod(kys[iz]*sign(kxs[iz]), sdm.Ny) + 1
        sf.ok[ix,iy,iz] = sdm.Nx*sdm.Ny * amplitudes[iz]/2
    end
    synchronize_FlowField_k2x!(sf, cop)
    return sf
end

function initialize_FlowField_random(amplitude::Float64, sdm::SpaceDomain, cop::ConstantOperators, rng::Random.AbstractRNG)
    sf = FlowField(sdm.Nx, sdm.Ny)
    sf.ok .= Random.randn(rng, ComplexF64, size(sf.ok))
    sf.ok[1,1,:] .= 0
    synchronize_FlowField_k2x!(sf, cop)
    return sf
end

function initialize_FlowField_random_unstable(sdm::SpaceDomain, cop::ConstantOperators, rng::Random.AbstractRNG)
    amplitude = 0.1
    sf = initialize_FlowField_random(amplitude, sdm, cop, rng)
    sf.ok[1,1,:] .= 0
    instab = maximum(real.(cop.evals_ctime), dims=3) .* cop.dealias_mask
    most_unstable_mode = argmax(instab)
    kx = most_unstable_mode[1]
    ky = most_unstable_mode[2]
    @show size(sf.ok)
    @show size(sdm.kxgrid)
    @show size(sdm.kygrid)
    sf.ok .*= (abs.(sqrt.(sdm.kxgrid[1:size(sf.ok, 1)].^2 .+ (sdm.kygrid.^2)') .- sqrt(kx^2 + ky^2)) .< 3)
    synchronize_FlowField_k2x!(sf)
    return sf
end

function test_tendency()

    # ----------------- Define parameters -------------------
    #
    
    sdm = SpaceDomain(2*pi*30, 64, 2*pi*30, 64)
    beta = 0.25
    topo_amp = 0.25 * beta * sdm.Ly

    php = PhysicalParams(; beta=0.25, kappa=0.0, nu_cuberoot=0.0, topo_amp=topo_amp)
    @show php.nu
    dtph_solve = 0.025
    # -------------------------------------------------------
    cop = ConstantOperators(php, sdm, dtph_solve; implicitude=1.0)
    # ------------------ Set up save-out place --------------
    savedir = "/net/bstor002.ib/pog/001/ju26596/jesus_project/Panetta1993/2024-08-16/2"
    mkpath(savedir)
    #
    # ====================================================
    # Single-mode test case 
    # ====================================================
    sf_an = zeros(Float64, (sdm.Nx, sdm.Ny, 2))
    kx_init,ky_init = 23,1 
    kx_init_phys = 2*pi*kx_init/sdm.Lx
    ky_init_phys = 2*pi*ky_init/sdm.Ly
    amplitudes = [1.0,2.0]
    for iz = 1:2
        sf_an[:,:,iz] = amplitudes[iz] * cos.(kx_init_phys*sdm.xgrid .+ ky_init_phys*sdm.ygrid')
    end

    # Reconstruct with Fourier
    sf_co = FlowField(sdm.Nx,sdm.Ny)
    @show size(sf_co.ox), size(sf_co.ok)

    ix = mod(abs(kx_init), sdm.Nx) + 1
    iy = mod(ky_init*sign(kx_init), sdm.Ny) + 1
    for iz = 1:2
        sf_co.ok[ix,iy,iz] = sdm.Nx*sdm.Ny * amplitudes[iz]/2
    end
    synchronize_FlowField_k2x!(sf_co, cop)
    
    # ------- Visualize initial condition ------------
    fig = Figure()
    lout = fig[1:2,1:3] = GridLayout()
    for iz = 1:2
        for (col,sf,label) = ((1,sf_an,"analytic"),(2,sf_co.ox,"computational"),(3,sf_co.ox-sf_an,"co - an"))
            ax = Axis(fig[iz,col], xlabel="x", ylabel="y", title=Printf.@sprintf("SF init %s, range \n (%.2e,%.2e)", label, minimum(sf[:,:,iz]), maximum(sf[:,:,iz])))
            sfmax = maximum(abs.(sf[:,:,iz]))
            levneg = collect(range(-sfmax, 0, 6)[1:end-1])
            levpos = collect(range(0, sfmax, 6)[2:end])
            contneg = contour!(ax, (0,sdm.Lx),(0,sdm.Ly),sf[:,:,iz],levels=levneg,color=:black,linestyle=:dash)
            contpos = contour!(ax, (0,sdm.Lx),(0,sdm.Ly),sf[:,:,iz],levels=levpos,color=:black)
        end
    end
    save(joinpath(savedir,"sf_init_1mode.png"), fig)
    # ------------------------------------------------
    #
    # --------- Compute tendency for initial condition (analytically) --------------
    # K*d(sf)/dt = L*sf
    K = zeros(Float64, (2,2))
    K[1,1] = K[2,2] = -(kx_init_phys^2+ky_init_phys^2+1/2) 
    K[1,2] = K[2,1] = 1/2
    U = [1.0,0.0]
    rhs = zeros(Float64,2)
    rhs .+= kx_init_phys*LA.diagm(0 => php.beta .+ [1,-1].*(U[1]-U[2])/2) * amplitudes 
    rhs .+= kx_init_phys*LA.diagm(0 => U) * K * amplitudes
    println("rhs = ")
    display(rhs)
    println("Kinv = ")
    display(LA.inv(K))

    layer_tendencies = LA.inv(K) * rhs 
    println("layer_tendencies = ")
    display(layer_tendencies)

    sf_dt_an = zeros(Float64, (sdm.Nx, sdm.Ny, 2))
    for iz = 1:2
        sf_dt_an[:,:,iz] .= layer_tendencies[iz] * sin.(kx_init_phys*sdm.xgrid .+ ky_init_phys*sdm.ygrid')
    end

    # ---------------------------------------------------------------
    #
    # ----------- Compute tendency from full routine -------------------------------
    t = 0.0
    flow = FlowState(t, sf_co)
    flob = FlowStateObservables(sdm.Nx, sdm.Ny)
    compute_observables!(flob, flow, cop)

    fig = Figure()
    lout = fig[1:2,1:3] = GridLayout()
    for iz = 1:2
        for (col,field,label) = ((1,sf_dt_an,"analytic"),(2,flob.sf_dt_lin.ox,"computational"),(3,flob.sf_dt_lin.ox-sf_dt_an,"difference"))
            ax = Axis(fig[iz,col], xlabel="x", ylabel="y", title=Printf.@sprintf("SF tendency %s, range \n (%.2e,%.2e)", label, minimum(field[:,:,iz]), maximum(field[:,:,iz])))
            fieldmax = maximum(abs.(field[:,:,iz]))
            levneg = collect(range(-fieldmax, 0, 6)[1:end-1])
            levpos = collect(range(0, fieldmax, 6)[2:end])
            contneg = contour!(ax, (0,sdm.Lx),(0,sdm.Ly),field[:,:,iz],levels=levneg,color=:black,linestyle=:dash)
            contpos = contour!(ax, (0,sdm.Lx),(0,sdm.Ly),field[:,:,iz],levels=levpos,color=:black)
        end
    end
    save(joinpath(savedir,"sf_dt_init_1mode.png"), fig)


    # ===========================================================
    # Two-mode test case 
    # ===========================================================
    kx_init,ky_init = [-1,2],[23,1]
    sf_an = zeros(Float64, (sdm.Nx,sdm.Ny,2))
    kx_init_phys = 2*pi*kx_init/sdm.Lx
    ky_init_phys = 2*pi*ky_init/sdm.Ly
    amplitudes = [1.0,2.0]
    for iz = 1:2
        sf_an[:,:,iz] = amplitudes[iz] * cos.(kx_init_phys[iz]*sdm.xgrid .+ ky_init_phys[iz]*sdm.ygrid')
    end

    # Reconstruct with Fourier
    sf_co = FlowField(sdm.Nx,sdm.Ny)
    for iz = 1:2
        ix = mod(abs(kx_init[iz]), sdm.Nx) + 1
        iy = mod(ky_init[iz]*sign(kx_init[iz]), sdm.Ny) + 1
        sf_co.ok[ix,iy,iz] = sdm.Nx*sdm.Ny * amplitudes[iz]/2
    end
    synchronize_FlowField_k2x!(sf_co, cop)
    
    # ------- Visualize initial condition ------------
    fig = Figure()
    lout = fig[1:2,1:3] = GridLayout()
    for iz = 1:2
        for (col,sf,label) = ((1,sf_an,"analytic"),(2,sf_co.ox,"computational"),(3,sf_co.ox-sf_an,"co - an"))
            ax = Axis(fig[iz,col], xlabel="x", ylabel="y", title=Printf.@sprintf("SF init %s, range \n (%.2e,%.2e)", label, minimum(sf[:,:,iz]), maximum(sf[:,:,iz])))
            sfmax = maximum(abs.(sf[:,:,iz]))
            levneg = collect(range(-sfmax, 0, 6)[1:end-1])
            levpos = collect(range(0, sfmax, 6)[2:end])
            contneg = contour!(ax, (0,sdm.Lx),(0,sdm.Ly),sf[:,:,iz],levels=levneg,color=:black,linestyle=:dash)
            contpos = contour!(ax, (0,sdm.Lx),(0,sdm.Ly),sf[:,:,iz],levels=levpos,color=:black)
        end
    end
    save(joinpath(savedir,"sf_init_2mode.png"), fig)
    # ------------------------------------------------
    #
    # --------- Compute nonlinear tendency for initial condition (analytically) --------------
    #
    pv_dt_nonlin_an = zeros(Float64, (sdm.Nx,sdm.Ny,2))
    sf_dt_nonlin_an = zeros(Float64, (sdm.Nx,sdm.Ny,2))
    method = "3"
    if method == "1"
        pv_dx,pv_dy,sf_dx,sf_dy = (zeros(Float64,(sdm.Nx,sdm.Ny,2)) for _=1:5)
        (sines,cosines) = (zeros(Float64,(sdm.Nx,sdm.Ny,2)) for _=1:2)
        for iz = 1:2
            sines[:,:,iz] .= sin.(broadcast(+, kx_init_phys[iz]*xgrid, ky_init_phys[iz]*ygrid'))
            cosines[:,:,iz] .= cos.(broadcast(+, kx_init_phys[iz]*xgrid, ky_init_phys[iz]*ygrid'))
        end
        for iz = 1:2
            sf_dx[:,:,iz] .= -kx_init_phys[iz] * amplitudes[iz] * sines[:,:,iz]
            sf_dy[:,:,iz] .= -ky_init_phys[iz] * amplitudes[iz] * sines[:,:,iz]
        end
        for iz = 1:2
            jz = mod(iz,2) + 1
            pv_dx[:,:,iz] .= -(kx_init_phys[iz]^2 + ky_init_phys[iz]^2 + 1/2) * sf_dx[:,:,iz] + (1/2)*sf_dx[:,:,jz]
            pv_dy[:,:,iz] .= -(kx_init_phys[iz]^2 + ky_init_phys[iz]^2 + 1/2) * sf_dy[:,:,iz] + (1/2)*sf_dy[:,:,jz]
        end
        J = (sf_dx .* pv_dy - sf_dy .* pv_dx)
        pv_dt_nonlin_an .= -J
        println("Analytical intermediates")
        @show maximum(abs.(real.(J)))
        @show maximum(abs.(real.(sf_dx)))
        @show maximum(abs.(real.(sf_dy)))
        @show maximum(abs.(real.(pv_dx)))
        @show maximum(abs.(real.(pv_dy)))
        @show maximum(abs.(imag.(J)))
    elseif method == "2"
        amps_sines = zeros(Float64, (sdm.Nx,sdm.Ny,2))
        for iz = 1:2
            amps_sines[:,:,iz] .= amplitudes[iz] * sin.(broadcast(+, kx_init_phys[iz]*xgrid, ky_init_phys[iz]*ygrid'))
        end
        pattern = (1/2) * amps_sines[:,:,1] .* amps_sines[:,:,2]
        crossprod = kx_init[2]*ky_init[1] - kx_init[1]*ky_init[2]

        for iz = 1:2
            pv_dt_nonlin_an[:,:,iz] .= - (-1)^iz * pattern * crossprod
        end
    elseif method == "3"
        kx_sum = kx_init_phys[1] + kx_init_phys[2]
        ky_sum = ky_init_phys[1] + ky_init_phys[2]
        kx_diff = kx_init_phys[1] - kx_init_phys[2]
        ky_diff = ky_init_phys[1] - ky_init_phys[2]
        phase_sum = kx_sum*sdm.xgrid .+ ky_sum*sdm.ygrid'
        phase_diff = kx_diff*sdm.xgrid .+ ky_diff*sdm.ygrid'
        cos_sum = cos.(phase_sum)
        cos_diff = cos.(phase_diff)
        amps_prod = amplitudes[1]*amplitudes[2]
        crossprod = kx_init_phys[2]*ky_init_phys[1] - kx_init_phys[1]*ky_init_phys[2]
        pv_dt_sum,pv_dt_diff = (zeros(Float64,(sdm.Nx,sdm.Ny,2)) for _=1:2)
        pv_dt_sum_amps = zeros(Float64,2) 
        pv_dt_diff_amps = zeros(Float64,2)
        for iz = 1:2
            pv_dt_sum_amps[iz] = (-1)^iz/4 * amps_prod * crossprod * (-sdm.Kx <= kx_sum*sdm.Lx/(2*pi) <= sdm.Kx) * (-sdm.Ky <= ky_sum*sdm.Ly/(2*pi) <= sdm.Ky)
            pv_dt_diff_amps[iz] = -(-1)^iz/4 * amps_prod * crossprod * (-sdm.Kx <= kx_diff*sdm.Lx/(2*pi) <= sdm.Kx) * (-sdm.Ky <= ky_diff*sdm.Ly/(2*pi) <= sdm.Ky)
            pv_dt_sum[:,:,iz] = pv_dt_sum_amps[iz] * cos_sum
            pv_dt_diff[:,:,iz] = pv_dt_diff_amps[iz] * cos_diff
        end
        pv_dt_nonlin_an .= pv_dt_sum + pv_dt_diff

        # Now invert to get streamfunction tendency
        # Consider both shifted frequencies
        (K_sum,K_diff) = (zeros(Float64,(2,2)) for _ = 1:2)
        for iz = 1:2
            jz = iz % 2 + 1
            K_sum[iz,iz] = -(kx_sum^2 + ky_sum^2 + 1/2)
            K_sum[iz,jz] = 1/2
            K_diff[iz,iz] = -(kx_diff^2 + ky_diff^2 + 1/2)
            K_diff[iz,jz] = 1/2
        end

        sf_dt_sum_amps = LA.inv(K_sum) * pv_dt_sum_amps
        sf_dt_diff_amps = LA.inv(K_diff) * pv_dt_diff_amps
        for iz = 1:2
            sf_dt_nonlin_an[:,:,iz] .= sf_dt_sum_amps[iz] * cos_sum + sf_dt_diff_amps[iz] * cos_diff
        end
        
    end



    # ---------------------------------------------------------------
    #
    # ----------- Compute tendency from general routine ------------
    tph = 0.0
    flow = FlowState(tph, sf_co)
    compute_observables!(flob, flow, cop)

    @show maximum(abs.(flob.pv_dt_nonlin.ox - pv_dt_nonlin_an))

    fig = Figure()
    lout = fig[1:2,1:3] = GridLayout()
    for iz = 1:2
        for (col,field,label) = ((1,pv_dt_nonlin_an,"analytic"),(2,flob.pv_dt_nonlin.ox,"computational"),(3,flob.pv_dt_nonlin.ox-pv_dt_nonlin_an,"co - an"))
            ax = Axis(fig[iz,col], xlabel="x", ylabel="y", title=Printf.@sprintf("PV tendency %s, range \n (%.2e,%.2e)", label, minimum(field[:,:,iz]), maximum(field[:,:,iz])))
            fieldmax = maximum(abs.(field[:,:,iz]))
            levneg = collect(range(-fieldmax, 0, 6)[1:end-1])
            levpos = collect(range(0, fieldmax, 6)[2:end])
            contneg = contour!(ax, (0,sdm.Lx),(0,sdm.Ly),field[:,:,iz],levels=levneg,color=:black,linestyle=:dash)
            contpos = contour!(ax, (0,sdm.Lx),(0,sdm.Ly),field[:,:,iz],levels=levpos,color=:black)
        end
    end
    save(joinpath(savedir,"pv_dt_nonlin_init_2mode.png"), fig)

    @show maximum(abs.(flob.sf_dt_nonlin.ox - sf_dt_nonlin_an))
    fig = Figure()
    lout = fig[1:2,1:3] = GridLayout()
    for iz = 1:2
        for (col,field,label) = ((1,sf_dt_nonlin_an,"analytic"),(2,flob.sf_dt_nonlin.ox,"computational"),(3,flob.sf_dt_nonlin.ox-sf_dt_nonlin_an,"co - an"))
            ax = Axis(fig[iz,col], xlabel="x", ylabel="y", title=Printf.@sprintf("SF tendency %s, range \n (%.2e,%.2e)", label, minimum(field[:,:,iz]), maximum(field[:,:,iz])))
            fieldmax = maximum(abs.(real.(field[:,:,iz])))
            levneg = collect(range(-fieldmax, 0, 6)[1:end-1])
            levpos = collect(range(0, fieldmax, 6)[2:end])
            contneg = contour!(ax, (0,sdm.Lx),(0,sdm.Ly),real.(field[:,:,iz]),levels=levneg,color=:black,linestyle=:dash)
            contpos = contour!(ax, (0,sdm.Lx),(0,sdm.Ly),real.(field[:,:,iz]),levels=levpos,color=:black)
        end
    end
    save(joinpath(savedir,"sf_dt_nonlin_init_2mode.png"), fig)
end

function write_state(flow::FlowState, warmend_filename::String)
    JLD2.jldopen(warmend_filename,"w") do f
        f["conc"] = flow.conc
        f["sf_ok"] = flow.sf.ok
        f["sf_ox"] = flow.sf.ox
        f["tph"] = flow.tph
    end
end

function read_state(warmend_filename::String)
    flow = JLD2.jldopen(warmend_filename,"r") do f
        tph = f["tph"]
        sf = FlowField(f["sf_ok"], f["sf_ox"])
        conc = f["conc"]
        flow = FlowState(tph, sf, conc)
        return flow
    end
    return flow
end

function write_perturbation_sequence(pert_seq::PerturbationSequence, pert_seq_filename::String)
    JLD2.jldopen(pert_seq_filename,"w") do f
        f["ts_ph"] = pert_seq.ts_ph
        f["perts"] = pert_seq.perts
    end
    return
end

function read_perturbation_sequence(pert_filename::String)
    ts_ph,perts = JLD2.jldopen(pert_filename,"r") do f
        return f["ts_ph"],f["perts"]
    end
    return PerturbationSequence(ts_ph,perts)
end

function write_history(sf_hist::FlowFieldHistory, conc_hist::Array{Float64,4}, history_filename::String)
    JLD2.jldopen(history_filename, "w") do f
        f["tgrid"] = sf_hist.tgrid
        f["sf_hist_ok"] = sf_hist.ok
        f["sf_hist_ox"] = sf_hist.ox
        f["conc_hist"] = conc_hist
    end
    return 
end

function read_history(history_filename::String)
    sf_hist,conc_hist = JLD2.jldopen(history_filename, "r") do f
        sf_hist = FlowFieldHistory(f["tgrid"], f["sf_hist_ok"], f["sf_hist_ox"])
        conc_hist = f["conc_hist"]
        return sf_hist, conc_hist
    end
    return sf_hist, conc_hist
end

function read_sf_hist(history_filename::String)
    sf_hist = JLD2.jldopen(history_filename, "r") do f
        sf_hist = FlowFieldHistory(f["tgrid"], f["sf_hist_ok"], f["sf_hist_ox"])
        return sf_hist
    end
    return sf_hist
end

function read_sf_hist(history_filenames::Vector{String})
    ffhists = (read_sf_hist(hfn) for hfn=history_filenames)
    sf_hist = concatenate_FlowFieldHistories(ffhists)
end

function read_conc_hist(history_filename::String)
    conc_hist = JLD2.jldopen(history_filename, "r") do f
        conc_hist = f["conc_hist"]
        return conc_hist
    end
    return conc_hist
end

function read_conc_hist(history_filenames::Vector{String})
    conc_hists = (read_conc_hist(hfn) for hfn=history_filenames)
    conc_hist = cat(conc_hists...; dims=4)
end


function histfile2obs(filename::String, obs_fun::Function)
    obs_val = JLD2.jldopen(filename, "r") do f
        return obs_fun(f)
    end
    return obs_val
end

# ---------- memory allocation test -----------
#

function compute_observable_ensemble(hist_filenames::Vector{String}, obs_fun::Function)
    # obs_fun must return an array (of whatever shape)
    obs_vals = Vector{Array{Float64}}([])
    for (i_fn,fn) in enumerate(hist_filenames)
        if mod(i_fn,50) == 0
            @show i_fn,length(hist_filenames)
        end
        push!(obs_vals, histfile2obs(fn, obs_fun))
    end
    return obs_vals
end

function compute_observable_histogram(hist_filenames::Vector{String}, obs_fun::Function, bin_width::Float64)
    # Assume the function is scalar-valued
    bins = Vector{Int64}([0])
    counts = Vector{Int64}([0])
    for (i_fn,fn) in enumerate(hist_filenames)
        println("bins counts = ")
        display(hcat(bins, counts)')
        function binned_obs_fun(args...) 
            return round.(Int, obs_fun(args...)/bin_width)
        end
        new_bins = histfile2obs(fn, binned_obs_fun) 
        min_new_bin,max_new_bin = extrema(new_bins)
        @show min_new_bin,max_new_bin
        if min_new_bin < bins[1]
            pushfirst!(counts, zeros(Int64, bins[1]-min_new_bin)...)
            pushfirst!(bins, (min_new_bin:1:bins[1]-1)...)
        end
        if max_new_bin > bins[end]
            push!(counts, zeros(Int64, max_new_bin-bins[end])...)
            push!(bins, (bins[end]+1:1:max_new_bin)...)
        end
        for b in new_bins
            counts[b-bins[1]+1] += 1
        end
    end
    return bins, counts
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

function obs_fun_conc_hist(f::JLD2.JLDFile, sdm::SpaceDomain, cop::ConstantOperators)
    return f["conc_hist_ox"]
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

function obs_fun_zonal_velocity_hist(f::JLD2.JLDFile, sdm::SpaceDomain, cop::ConstantOperators)
    return obs_fun_zonal_velocity_hist(f["tgrid"],f["sf_hist_ok"],sdm,cop)
end
function obs_fun_zonal_velocity_hist(tgrid::Vector{Int64}, sf_hist_ok::Array{ComplexF64,4}, sdm::SpaceDomain, cop::ConstantOperators)
    # not counting background shear 
    u_hist = FlowFieldHistory(tgrid, sdm.Nx, sdm.Ny) 
    u_hist.ok .= -cop.Dy .* sf_hist_ok
    synchronize_FlowField_k2x!(u_hist)
    u_hist.ox[:,:,1,:] .+= 1.0

    return u_hist.ox 
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

function overlap_two_line_segments(a1::Float64,b1::Float64,a2::Float64,b2::Float64)
    #length over overlap between (a1,b1) and (a2,b2)
    # Adjust to two line segments (a,b), (c,d) such that b-a > d-c
    a,b,c,d = (b1-a1 > b2-a2 ? (a1,b1,a2,b2) : (a2,b2,a1,b1))
    if a <= c <= b <= d
        return b - c
    elseif a <= c <= d <= b
        return d - c
    elseif c <= a <= d <= b
        return d - a
    end
    return 0.0
end

function horz_avg_filter(Nx::Int64, Ny::Int64, xPerL::Float64, rxPerL::Float64, yPerL::Float64, ryPerL::Float64)
    y = yPerL
    x = mod(xPerL,1.0)
    ry = ryPerL
    rx = rxPerL
    dy = 1/Ny
    dx = 1/Nx
    weights_x = zeros(Float64, Nx)
    weights_y = zeros(Float64, Ny)
    for ix = 1:Nx
        weights_x[ix] = overlap_two_line_segments((ix-1)*dx+1, ix*dx+1, x-rx+1, x+rx+1)/dx
    end
    if (minimum(weights_x) < 0) || (sum(weights_x) <= 0)
        @show weights_x
        @show x,rx,dx
        error()
    end
    for iy = 1:Ny
        # check if any overlap: top of cell iy greater than bottom of region and vice versa 
        weights_y[iy] = overlap_two_line_segments((iy-1)*dy, iy*dy, y-ry, y+ry)/dy
    end
    if (minimum(weights_y) < 0) || (sum(weights_y) <= 0)
        @show weights_y
        @show y,ry,dy

        error()
    end
    weights = weights_x .* weights_y'
    weights ./= sum(weights)
    return weights
end

function horz_avg(u::Array{Float64}, sdm::SpaceDomain, xPerL::Float64, rxPerL::Float64, yPerL::Float64, ryPerL::Float64)
    # y direction must get clipped atboundaries 
    # All the following in nondimensional units
    weights = horz_avg_filter(sdm.Nx, sdm.Ny, xPerL, rxPerL, yPerL, ryPerL)
    return sum(u .* weights; dims=[1,2])
end

function horz_weighted_sum(u::Array{Float64}, weights::Matrix{Float64})
    return sum(u .* weights; dims=[1,2])
end


function horz_avg_old(u::Array{Float64}, sdm::SpaceDomain, xPerL::Float64, rxPerL::Float64, yPerL::Float64, ryPerL::Float64)
    dxPerL = 1/sdm.Nx
    ixmin = 1 + ceil(Int, (xPerL-rxPerL)/dxPerL)
    ixmax = 1 + floor(Int, (xPerL+rxPerL)/dxPerL)
    idx_x = mod.(collect(ixmin:ixmax).-1, sdm.Nx) .+ 1
    dyPerL = 1/sdm.Ny
    iymin = 1 + ceil(Int, (yPerL-ryPerL)/dyPerL)
    iymax = 1 + floor(Int, (yPerL+ryPerL)/dyPerL)
    idx_y = mod.(collect(iymin:iymax).-1, sdm.Ny) .+ 1
    return SB.mean(
                   selectdim(
                             selectdim(u, 1, idx_x), 
                             2, idx_y
                            ); 
                   dims=[1,2]
                  )
end
