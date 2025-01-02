import FFTW
import Printf
import LinearAlgebra as LA
using CairoMakie

function compute_gridded_matrix_vector_product!(
        # intent(out)
        Au::Array{ComplexF64,3},
        # intent(in)
        A::Array{ComplexF64,4},
        u::Array{ComplexF64,3},
    )
    Au .= 0.0
    for j = 1:size(A,4)
        for i = 1:size(A,3)
            Au[:,:,i] .+= A[:,:,i,j] .* u[:,:,j]
        end
    end
end

function truncate_hifreq!(uh::AbstractArray{ComplexF64}, Kx::Int64, dim::Int64)
    selectdim(uh, dim, (Kx+2):(size(uh,dim)-Kx)) .= 0
end

function truncate_hifreq!(uh::AbstractArray{ComplexF64}, Ks::Vector{Int64}, dims::Vector{Int64})
    for (K,d) in zip(Ks,dims)
        truncate_hifreq!(uh, K, d)
    end
end


function tendency(
        sf_h,
        beta,kappa,raddamp,nu,U,
        Lx,Ly,Nx,Ny,Kx,Ky,
        xgrid,ygrid,kxgrid,kygrid,
        ;
        verbose = false,
    )

    println("---------- yes intendency -------------")
    @show verbose
    Dx = 2*pi*1im/Lx * broadcast(*, kxgrid, transpose(ones(Ny)))
    Dy = 2*pi*1im/Ly * broadcast(*, ones(Nx), transpose(kygrid))
    Lap = Dx.^2 + Dy.^2

    K = zeros(ComplexF64, (Nx,Ny,2,2))
    Kinv = zeros(ComplexF64, (Nx,Ny,2,2))
    KinvL = zeros(ComplexF64, (Nx,Ny,2,2))
    L = zeros(ComplexF64, (Nx,Ny,2,2))
    sf_dt_lin_h = zeros(ComplexF64, (Nx,Ny,2))
    for ix = 1:Nx
        kx = kxgrid[ix]
        for iy = 1:Ny
            ky = kygrid[iy]
            if (kx == ky == 0)
                continue # TODO if radiative damping present 
            end
            for iz = 1:2
                jz = mod(iz,2) + 1
                K[ix,iy,iz,iz] = Lap[ix,iy] - 1/2
                K[ix,iy,iz,jz] = 1/2
            end
            for iz = 1:2
                L[ix,iy,iz,iz] = -Dx[ix,iy]*(beta - (-1)^iz*(U[1]-U[2])/2)
                L[ix,iy,iz,:] .-= U[iz]*Dx[ix,iy]*K[ix,iy,iz,:]
                # dissipation
                L[ix,iy,iz,iz] -= kappa[iz]*Lap[ix,iy]
                L[ix,iy,iz,iz] -= nu*Lap[ix,iy]^3
            end
            Kinv[ix,iy,:,:] .= LA.inv(K[ix,iy,:,:]) 
            KinvL[ix,iy,:,:] .= Kinv[ix,iy,:,:] * L[ix,iy,:,:]
            if false && (kx == 3) && (ky == 2)
                println("Dx[$(ix),$(iy)] = $(Dx[ix,iy])")
                println("beta = $(beta)")
                println("K[$(kx),$(ky)] = ")
                display(K[ix,iy,:,:])
                println("L[$(kx),$(ky)] = ")
                display(L[ix,iy,:,:])
                println("KinvL[$(kx),$(ky)] = ")
                display(KinvL[ix,iy,:,:])
                println("sf_h = ")
                display(sf_h[ix,iy,:])
            end
        end
    end
    compute_gridded_matrix_vector_product!(sf_dt_lin_h, KinvL, sf_h)

    # Nonlinear term
    truncate_hifreq!(sf_h, [Kx,Ky], [1,2])
    (
     sf_dx_h,sf_dx,sf_dy_h,sf_dy,
     pv_h,pv,
     pv_dx_h,pv_dx,pv_dy_h,pv_dy,
     J_h,J
    ) = (zeros(ComplexF64, (Nx,Ny,2)) for _ = 1:12)
    compute_gridded_matrix_vector_product!(pv_h, K, sf_h)
    for iz = 1:2
        pv_dx_h[:,:,iz] .= Dx .* pv_h[:,:,iz]
        pv_dy_h[:,:,iz] .= Dy .* pv_h[:,:,iz]
        sf_dx_h[:,:,iz] .= Dx .* sf_h[:,:,iz]
        sf_dy_h[:,:,iz] .= Dy .* sf_h[:,:,iz]
    end
    pv .= FFTW.ifft(pv_h, [1,2])
    pv_dx .= FFTW.ifft(pv_dx_h, [1,2])
    pv_dy .= FFTW.ifft(pv_dy_h, [1,2])
    sf_dx .= FFTW.ifft(sf_dx_h, [1,2])
    sf_dy .= FFTW.ifft(sf_dy_h, [1,2])

    J = sf_dx .* pv_dy - sf_dy .* pv_dx
    if verbose
        println("Computational intermediates")
        @show maximum(abs.(real.(J)))
        @show maximum(abs.(real.(sf_dx)))
        @show maximum(abs.(real.(sf_dy)))
        @show maximum(abs.(real.(pv_dx)))
        @show maximum(abs.(real.(pv_dy)))
        @show maximum(abs.(imag.(J)))
    end
    J .= real.(J)
    J_h = FFTW.fft(J, [1,2])
    (sf_dt_nonlin_h,pv_dt_nonlin_h) = (zeros(ComplexF64, (Nx,Ny,2)) for _ = 1:2)
    pv_dt_nonlin_h .= -J_h
    compute_gridded_matrix_vector_product!(sf_dt_nonlin_h, Kinv, pv_dt_nonlin_h)

    return sf_dt_lin_h,sf_dt_nonlin_h,pv_dt_nonlin_h
end


function test_tendency()

    # ----------------- Define parameters -------------------
    beta = 0.75
    kappa = 0*[0.0, 0.1] 
    raddamp = 0.0
    nu = 0*0.292^3
    U = [1.0,0.0] 
    Lx = Ly = 2*pi*30
    Nx = Ny = 64
    Kx = Ky = 41
    dt = 0.025
    dt_record = 0.1 
    # -------------------------------------------------------
    # ------------------ Set up save-out place --------------
    savedir = "/net/bstor002.ib/pog/001/ju26596/jesus_project/Panetta1993/2024-08-16/0"
    mkpath(savedir)
    # ------------------ Set up domain ----------------------
    
    xgrid = collect(range(0,Lx,Nx+1)[1:end-1])
    ygrid = collect(range(0,Ly,Nx+1)[1:end-1])
    
    kxgrid = zeros(Int64, Nx)
    kxgrid[1:div(Nx,2)+1] .= 0:div(Nx,2)
    kxgrid[div(Nx,2)+2:Nx] .= -(Nx-div(Nx,2)-1):-1
    
    kygrid = zeros(Int64, Ny)
    kygrid[1:div(Ny,2)+1] .= 0:div(Ny,2)
    kygrid[div(Ny,2)+2:Ny] .= -(Ny-div(Ny,2)-1):-1
    
    tinit,tfin = 0.0, 10.0
    Nt = round(Int,(tfin - tinit)/dt_record)
    tgrid = tinit .+ dt_record .* collect(1:Nt)
    
    # --------------------------------------------------------
    #
    # ====================================================
    # Single-mode test case 
    # ====================================================
    sf_an = zeros(Float64, (Nx,Ny,2))
    kx_init,ky_init = 3,2 
    kx_init_phys = 2*pi*kx_init/Lx
    ky_init_phys = 2*pi*ky_init/Ly
    amplitudes = [1.0,2.0]
    for iz = 1:2
        sf_an[:,:,iz] = amplitudes[iz] * cos.(broadcast(+, kx_init_phys*xgrid, transpose(ky_init_phys*ygrid)))
    end

    # Reconstruct with Fourier
    sf_co_h = zeros(ComplexF64, (Nx,Ny,2))
    for iz = 1:2
        sf_co_h[kx_init+1,ky_init+1,iz] = Nx*Ny * amplitudes[iz]/2
        sf_co_h[end-kx_init+1,end-ky_init+1,iz] = conj(sf_co_h[kx_init+1,ky_init+1,iz])
    end
    sf_co = FFTW.ifft(sf_co_h, [1,2])
    
    # ------- Visualize initial condition ------------
    fig = Figure()
    lout = fig[1:2,1:3] = GridLayout()
    for iz = 1:2
        for (col,sf,label) = ((1,sf_an,"analytic"),(2,sf_co,"computational"),(3,sf_co-sf_an,"co - an"))
            ax = Axis(fig[iz,col], xlabel="x", ylabel="y", title=Printf.@sprintf("SF init %s, range \n (%.2e,%.2e) \n + ((%.2e,%.2e)i", label, minimum(real.(sf[:,:,iz])), maximum(real.(sf[:,:,iz])), minimum(imag.(sf[:,:,iz])), maximum(imag.(sf[:,:,iz]))))
            sfmax = maximum(abs.(real.(sf[:,:,iz])))
            levneg = collect(range(-sfmax, 0, 6)[1:end-1])
            levpos = collect(range(0, sfmax, 6)[2:end])
            contneg = contour!(ax, (0,Lx),(0,Ly),real.(sf[:,:,iz]),levels=levneg,color=:black,linestyle=:dash)
            contpos = contour!(ax, (0,Lx),(0,Ly),real.(sf[:,:,iz]),levels=levpos,color=:black)
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
    rhs = zeros(Float64,2)
    rhs .+= kx_init_phys*LA.diagm(0 => beta .+ [1,-1].*(U[1]-U[2])/2) * amplitudes 
    rhs .+= kx_init_phys*LA.diagm(0 => U) * K * amplitudes
    println("rhs = ")
    display(rhs)

    layer_tendencies = LA.inv(K) * rhs 
    println("layer_tendencies = ")
    display(layer_tendencies)

    sf_dt_an = zeros(Float64, (Nx,Ny,2))
    for iz = 1:2
        sf_dt_an[:,:,iz] .= layer_tendencies[iz] * sin.(broadcast(+, kx_init_phys*xgrid, transpose(ky_init_phys*ygrid)))
    end

    # ---------------------------------------------------------------
    #
    # ----------- Compute tendency from full routine -------------------------------
    sf_dt_lin_co_h,sf_dt_nonlin_co_h,pv_dt_nonlin_co_h = tendency(
        sf_co_h,
        beta,kappa,raddamp,nu,U,
        Lx,Ly,Nx,Ny,Kx,Ky,
        xgrid,ygrid,kxgrid,kygrid,
        ;
        verbose = true
       )
    sf_dt_lin_co = FFTW.ifft(sf_dt_lin_co_h, [1,2])

    @show maximum(abs.(sf_dt_lin_co - sf_dt_an))

    fig = Figure()
    lout = fig[1:2,1:3] = GridLayout()
    for iz = 1:2
        for (col,field,label) = ((1,sf_dt_an,"analytic"),(2,sf_dt_lin_co,"computational"),(3,sf_dt_lin_co-sf_dt_an,"comp - an"))
            ax = Axis(fig[iz,col], xlabel="x", ylabel="y", title=Printf.@sprintf("SF tendency %s, range \n (%.2e,%.2e) \n + ((%.2e,%.2e)i", label, minimum(real.(field[:,:,iz])), maximum(real.(field[:,:,iz])), minimum(imag.(field[:,:,iz])), maximum(imag.(field[:,:,iz]))))
            fieldmax = maximum(abs.(real.(field[:,:,iz])))
            levneg = collect(range(-fieldmax, 0, 6)[1:end-1])
            levpos = collect(range(0, fieldmax, 6)[2:end])
            contneg = contour!(ax, (0,Lx),(0,Ly),real.(field[:,:,iz]),levels=levneg,color=:black,linestyle=:dash)
            contpos = contour!(ax, (0,Lx),(0,Ly),real.(field[:,:,iz]),levels=levpos,color=:black)
        end
    end
    save(joinpath(savedir,"sf_dt_init_1mode.png"), fig)


    # ===========================================================
    # Two-mode test case 
    # ===========================================================
    kx_init,ky_init = [4,1],[1,2]
    sf_an = zeros(Float64, (Nx,Ny,2))
    kx_init_phys = 2*pi*kx_init/Lx
    ky_init_phys = 2*pi*ky_init/Ly
    amplitudes = [1.0,2.0]
    for iz = 1:2
        sf_an[:,:,iz] = amplitudes[iz] * cos.(broadcast(+, kx_init_phys[iz]*xgrid, transpose(ky_init_phys[iz]*ygrid)))
    end

    # Reconstruct with Fourier
    sf_co_h = zeros(ComplexF64, (Nx,Ny,2))
    for iz = 1:2
        sf_co_h[kx_init[iz]+1,ky_init[iz]+1,iz] = Nx*Ny * amplitudes[iz]/2
        sf_co_h[end-kx_init[iz]+1,end-ky_init[iz]+1,iz] = conj(sf_co_h[kx_init[iz]+1,ky_init[iz]+1,iz])
    end
    sf_co = FFTW.ifft(sf_co_h, [1,2])
    
    # ------- Visualize initial condition ------------
    fig = Figure()
    lout = fig[1:2,1:2] = GridLayout()
    for iz = 1:2
        for (col,sf,label) = ((1,sf_an,"analytic"),(2,sf_co,"computational"))
            ax = Axis(fig[iz,col], xlabel="x", ylabel="y", title=Printf.@sprintf("SF init %s, range \n (%.2e,%.2e) \n + ((%.2e,%.2e)i", label, minimum(real.(sf[:,:,iz])), maximum(real.(sf[:,:,iz])), minimum(imag.(sf[:,:,iz])), maximum(imag.(sf[:,:,iz]))))
            sfmax = maximum(abs.(real.(sf[:,:,iz])))
            levneg = collect(range(-sfmax, 0, 6)[1:end-1])
            levpos = collect(range(0, sfmax, 6)[2:end])
            contneg = contour!(ax, (0,Lx),(0,Ly),real.(sf[:,:,iz]),levels=levneg,color=:black,linestyle=:dash)
            contpos = contour!(ax, (0,Lx),(0,Ly),real.(sf[:,:,iz]),levels=levpos,color=:black)
        end
    end
    save(joinpath(savedir,"sf_init_2mode.png"), fig)
    # ------------------------------------------------
    #
    # --------- Compute nonlinear tendency for initial condition (analytically) --------------
    #
    pv_dt_nonlin_an = zeros(Float64, (Nx,Ny,2))
    sf_dt_nonlin_an = zeros(Float64, (Nx,Ny,2))
    method = "3"
    if method == "1"
        pv_dx,pv_dy,sf_dx,sf_dy = (zeros(Float64,(Nx,Ny,2)) for _=1:5)
        (sines,cosines) = (zeros(Float64,(Nx,Ny,2)) for _=1:2)
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
        amps_sines = zeros(Float64, (Nx,Ny,2))
        for iz = 1:2
            amps_sines[:,:,iz] .= amplitudes[iz] * sin.(broadcast(+, kx_init_phys[iz]*xgrid, ky_init_phys[iz]*ygrid'))
        end
        pattern = (1/2) * amps_sines[:,:,1] .* amps_sines[:,:,2]
        crossprod = kx_init[2]*ky_init[1] - kx_init[1]*ky_init[2]

        for iz = 1:2
            pv_dt_nonlin_an[:,:,iz] .= - (-1)^iz * pattern * crossprod
        end
    elseif method == "3"
        kx_bt = kx_init_phys[1] + kx_init_phys[2]
        ky_bt = ky_init_phys[1] + ky_init_phys[2]
        kx_bc = kx_init_phys[1] - kx_init_phys[2]
        ky_bc = ky_init_phys[1] - ky_init_phys[2]
        phase_bt = kx_bt*xgrid .+ ky_bt*ygrid'
        phase_bc = kx_bc*xgrid .+ ky_bc*ygrid'
        cos_bt = cos.(phase_bt)
        cos_bc = cos.(phase_bc)
        amps_prod = amplitudes[1]*amplitudes[2]
        crossprod = kx_init_phys[2]*ky_init_phys[1] - kx_init_phys[1]*ky_init_phys[2]
        pv_dt_bt,pv_dt_bc = (zeros(Float64,(Nx,Ny,2)) for _=1:2)
        pv_dt_bt_amps = zeros(Float64,2) 
        pv_dt_bc_amps = zeros(Float64,2)
        for iz = 1:2
            pv_dt_bt_amps[iz] = (-1)^iz/4 * amps_prod * crossprod * (-Kx <= kx_bt*Lx/(2*pi) <= Kx) * (-Ky <= ky_bt*Ly/(2*pi) <= Ky)
            pv_dt_bc_amps[iz] = -(-1)^iz/4 * amps_prod * crossprod * (-Kx <= kx_bc*Lx/(2*pi) <= Kx) * (-Ky <= ky_bc*Ly/(2*pi) <= Ky)
            pv_dt_bt[:,:,iz] = pv_dt_bt_amps[iz] * cos_bt
            pv_dt_bc[:,:,iz] = pv_dt_bc_amps[iz] * cos_bc
        end
        pv_dt_nonlin_an .= pv_dt_bt + pv_dt_bc

        # Now invert to get streamfunction tendency
        # Consider both shifted frequencies
        (K_bt,K_bc) = (zeros(Float64,(2,2)) for _ = 1:2)
        for iz = 1:2
            jz = iz % 2 + 1
            K_bt[iz,iz] = -(kx_bt^2 + ky_bt^2 + 1/2)
            K_bt[iz,jz] = 1/2
            K_bc[iz,iz] = -(kx_bc^2 + ky_bc^2 + 1/2)
            K_bc[iz,jz] = 1/2
        end

        sf_dt_bt_amps = LA.inv(K_bt) * pv_dt_bt_amps
        sf_dt_bc_amps = LA.inv(K_bc) * pv_dt_bc_amps
        for iz = 1:2
            sf_dt_nonlin_an[:,:,iz] .= sf_dt_bt_amps[iz] * cos_bt + sf_dt_bc_amps[iz] * cos_bc
        end
        

        #sf_dt_nonlin_an = zeros(Float64, (Nx,Ny,2))

    end



    # ---------------------------------------------------------------
    #
    # ----------- Compute tendency from full routine -------------------------------
    sf_dt_lin_co_h,sf_dt_nonlin_co_h,pv_dt_nonlin_co_h = tendency(
        sf_co_h,
        beta,kappa,raddamp,nu,U,
        Lx,Ly,Nx,Ny,Kx,Ky,
        xgrid,ygrid,kxgrid,kygrid,
        verbose = true
       )
    sf_dt_nonlin_co = FFTW.ifft(sf_dt_nonlin_co_h, [1,2])
    pv_dt_nonlin_co = FFTW.ifft(pv_dt_nonlin_co_h, [1,2])

    @show maximum(abs.(pv_dt_nonlin_co - pv_dt_nonlin_an))
    #@show maximum(abs.(sf_dt_nonlin_co - sf_dt_nonlin_an))

    fig = Figure()
    lout = fig[1:2,1:2] = GridLayout()
    for iz = 1:2
        for (col,field,label) = ((1,pv_dt_nonlin_an,"analytic"),(2,pv_dt_nonlin_co,"computational"))
            ax = Axis(fig[iz,col], xlabel="x", ylabel="y", title=Printf.@sprintf("PV tendency %s, range \n (%.2e,%.2e) \n + ((%.2e,%.2e)i", label, minimum(real.(field[:,:,iz])), maximum(real.(field[:,:,iz])), minimum(imag.(field[:,:,iz])), maximum(imag.(field[:,:,iz]))))
            fieldmax = maximum(abs.(real.(field[:,:,iz])))
            levneg = collect(range(-fieldmax, 0, 6)[1:end-1])
            levpos = collect(range(0, fieldmax, 6)[2:end])
            contneg = contour!(ax, (0,Lx),(0,Ly),real.(field[:,:,iz]),levels=levneg,color=:black,linestyle=:dash)
            contpos = contour!(ax, (0,Lx),(0,Ly),real.(field[:,:,iz]),levels=levpos,color=:black)
        end
    end
    save(joinpath(savedir,"pv_dt_nonlin_init_2mode.png"), fig)

    @show maximum(abs.(sf_dt_nonlin_co - sf_dt_nonlin_an))
    fig = Figure()
    lout = fig[1:2,1:2] = GridLayout()
    for iz = 1:2
        for (col,field,label) = ((1,sf_dt_nonlin_an,"analytic"),(2,sf_dt_nonlin_co,"computational"))
            ax = Axis(fig[iz,col], xlabel="x", ylabel="y", title=Printf.@sprintf("SF tendency %s, range \n (%.2e,%.2e) \n + ((%.2e,%.2e)i", label, minimum(real.(field[:,:,iz])), maximum(real.(field[:,:,iz])), minimum(imag.(field[:,:,iz])), maximum(imag.(field[:,:,iz]))))
            fieldmax = maximum(abs.(real.(field[:,:,iz])))
            levneg = collect(range(-fieldmax, 0, 6)[1:end-1])
            levpos = collect(range(0, fieldmax, 6)[2:end])
            contneg = contour!(ax, (0,Lx),(0,Ly),real.(field[:,:,iz]),levels=levneg,color=:black,linestyle=:dash)
            contpos = contour!(ax, (0,Lx),(0,Ly),real.(field[:,:,iz]),levels=levpos,color=:black)
        end
    end
    save(joinpath(savedir,"sf_dt_nonlin_init_2mode.png"), fig)


end

function integrate(
        sf_init_h,
        tinit,tfin,tgrid,dt,
        beta,kappa,raddamp,nu,U,
        Lx,Ly,Nx,Ny,Kx,Ky,
        xgrid,ygrid,kxgrid,kygrid
    )
    Nt = length(tgrid)
    sf_save_h = zeros(ComplexF64, (Nx,Ny,2,Nt))
    sf_h = zeros(ComplexF64, (Nx,Ny,2))
    sf_h .= sf_init_h
    t = tinit
    i_t_save = 1
    const_args = (
                  beta,kappa,raddamp,nu,U,
                  Lx,Ly,Nx,Ny,Kx,Ky,
                  xgrid,ygrid,kxgrid,kygrid
                 )
    while t < tfin
        sf_dt_lin_h_1,sf_dt_nonlin_h_1 = tendency(sf_h, const_args...)
        sf_dt_h_1 = sf_dt_lin_h_1 + sf_dt_nonlin_h_1
        sf_dt_lin_h_2,sf_dt_nonlin_h_2 = tendency(sf_h + (dt/2)*sf_dt_h_1, const_args...)
        sf_dt_h_2 = sf_dt_lin_h_2 + sf_dt_nonlin_h_2
        sf_dt_lin_h_3,sf_dt_nonlin_h_3 = tendency(sf_h + (dt/2)*sf_dt_h_2, const_args...)
        sf_dt_h_3 = sf_dt_lin_h_3 + sf_dt_nonlin_h_3
        sf_dt_lin_h_4,sf_dt_nonlin_h_4 = tendency(sf_h + dt*sf_dt_h_3, const_args...)
        sf_dt_h_4 = sf_dt_lin_h_4 + sf_dt_nonlin_h_4
        sf_h .+= (dt/6) * (sf_dt_h_1 + 2*(sf_dt_h_2 + sf_dt_h_3) + sf_dt_h_4)
        t += dt
        if t > tgrid[i_t_save]
            sf_save_h[:,:,:,i_t_save] .= sf_h
            i_t_save += 1
            println()
            Printf.@printf("t = %.2e, sf_h range = (%.2e,%.2e) + (%.2e,%.2e)i\n", t, minimum(real.(sf_h)), maximum(real.(sf_h)), minimum(imag.(sf_h)), maximum(imag.(sf_h)))
        end
    end
    return sf_save_h
end

function test_integrate()
    # ----------------- Define parameters -------------------
    beta = 0.25
    kappa = [0.0, 0.1] 
    raddamp = 0.0
    nu = 0.292^3
    U = [1.0,0.0] 
    Lx = Ly = 2*pi*30
    Nx = Ny = 64
    Kx = Ky = 41
    dt = 0.025
    dt_record = 0.1 
    # -------------------------------------------------------
    # ------------------ Set up save-out place --------------
    savedir = "/net/bstor002.ib/pog/001/ju26596/jesus_project/Panetta1993/2024-08-15/0"
    mkpath(savedir)
    # ------------------ Set up domain ----------------------
    
    xgrid = collect(range(0,Lx,Nx+1)[1:end-1])
    ygrid = collect(range(0,Ly,Nx+1)[1:end-1])
    
    kxgrid = zeros(Int64, Nx)
    kxgrid[1:div(Nx,2)+1] .= 0:div(Nx,2)
    kxgrid[div(Nx,2)+2:Nx] .= -(Nx-div(Nx,2)-1):-1
    
    kygrid = zeros(Int64, Ny)
    kygrid[1:div(Ny,2)+1] .= 0:div(Ny,2)
    kygrid[div(Ny,2)+2:Ny] .= -(Ny-div(Ny,2)-1):-1
    
    tinit,tfin = 0.0, 2000.0
    Nt = round(Int,(tfin - tinit)/dt_record)
    tgrid = tinit .+ dt_record .* collect(1:Nt)

    sf_init_h = zeros(ComplexF64, (Nx,Ny,2))
    kx_init,ky_init = [4,1],[1,2]
    amplitudes = [1.0,2.0]
    for iz = 1:2
        sf_init_h[kx_init[iz]+1,ky_init[iz]+1,iz] = Nx*Ny * amplitudes[iz]/2
        sf_init_h[end-kx_init[iz]+1,end-ky_init[iz]+1,iz] = conj(sf_init_h[kx_init[iz]+1,ky_init[iz]+1,iz])
    end
    
    sf_save_h = integrate(
        sf_init_h,
        tinit,tfin,tgrid,dt,
        beta,kappa,raddamp,nu,U,
        Lx,Ly,Nx,Ny,Kx,Ky,
        xgrid,ygrid,kxgrid,kygrid
       )
    sf_save = FFTW.ifft(sf_save_h,[1,2])
    @show maximum(abs.(imag.(sf_save)))
    sf_save .= real.(sf_save)


    # Animate
    anim_filename = joinpath(savedir,"sf_anim.mp4")
    fig = Figure()
    framerate = 12
    tidx = round.(Int, range(1, length(tgrid), min(length(tgrid),200)))
    lout = fig[1:2,1:1] = GridLayout()
    axes = collect(Axis(lout[iz,1], xlabel="x", ylabel="y") for iz=1:2)
    for iz = 1:2
        xlims!(axes[iz], (0,Lx))
        ylims!(axes[iz], (0,Ly))
    end
    record(fig,anim_filename, framerate=framerate) do io
        for (i_snap,i_t) in enumerate(tidx)
            objs = []
            for iz = 1:2
                ax = axes[iz]
                sfmax = maximum(abs.(real.(sf_save[:,:,iz,i_t])))
                levneg = collect(range(-sfmax, 0, 6)[1:end-1])
                levpos = collect(range(0, sfmax, 6)[2:end])
                contneg = contour!(ax, (0,Lx),(0,Ly),real.(sf_save[:,:,iz,i_t]),levels=levneg,color=:black,linestyle=:dash)
                push!(objs, (ax,contneg))
                contpos = contour!(ax, (0,Lx),(0,Ly),real.(sf_save[:,:,iz,i_t]),levels=levpos,color=:black)
                push!(objs, (ax,contpos))
            end
            recordframe!(io)
            for obj in objs
                delete!(obj...)
            end
        end
    end
end




test_tendency()
