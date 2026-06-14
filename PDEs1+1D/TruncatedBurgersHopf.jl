using Printf: @sprintf

function tendency!(dt_u_s, usq_s, u_s, kmax)
    usq_s .= 0
    dt_u_s .= 0
    # Fill in the usq_s array
    usq_s[1] = abs2(u_s[1]) + 4*sum(abs2.(u_s[2:kmax+1]))
    for k = 1:kmax
        for m = 1:k-1
            usq_s[k+1] += 2 * u_s[m+1] * u_s[(k-m)+1]
        end
        usq_s[1] += 2 * u_s[1] * u_s[k+1]
        for m = (k+1):kmax
            usq_s[k+1] += 2 * u_s[m+1] * conj(u_s[(m-k)+1])
        end
    end
    dt_u_s .= (-1im/2) .* (0:kmax) .* usq_s
    return
end

function timestep_rk4!(
        u_new, # final output
        urk1,urk2,urk3,urk4, # arguments to tendency!
        dturk1,dturk2,dturk3,dturk4, # tendencies
        usq, # scratch space for computing u2
        u, # initial 
        dt, # timestep
        kmax, # max wavenumber
    )
    urk1 .= u
    tendency!(dturk1, usq, urk1, kmax)
    urk2 .= u .+ (dt/2).*dturk1
    tendency!(dturk2, usq, urk2, kmax)
    urk3 .= u .+ (dt/2).*dturk2
    tendency!(dturk3, usq, urk3, kmax)
    urk4 .= u .+ dt.*dturk3
    tendency!(dturk4, usq, urk4, kmax)
    u_new .= u .+ (dt/6).*(dturk1 .+ 2 .* (dturk2 .+ dturk3) .+ dturk4)
    return
end

function integrate_tbh()
    NF = Float64
    # Simulate the Truncated Burgers-Hopf dynamics, which is a truncation of 
    # u_t + u*u_x = 0
    # to a finite number of Fourier modes. Don't even do FFT on this simplest of demos. 
    # ----------------- Parameters ------------------
    kmax = 16 # maximum wavenumber to retain 
    ks = collect(0:1:kmax)
    # ----------------- Initial conditions ----------
    init_wavenumbers = [1,2,4,8]
    init_amplitudes = [0.5,0.3,0.1,0.05]
    # ----------------- Simulation parameters -------
    dt = 0.01
    Nt = 1000
    Lx = 3.8 # domain width 
    Nx = 128
    # ----------------- Allocate arrays -------------
    uhist = zeros(Complex{NF}, (kmax+1, Nt)) # u in spectral space 
    # scratch space for RK4
    (
     u_old,u_new,
     dt_u,usq,
     # scratch for RK4
     urk1,urk2,urk3,urk4,
     dturk1,dturk2,dturk3,dturk4
    ) = ntuple(_->zeros(Complex{NF}, (kmax+1,)), 12)
    # ----------------- Initialize ------------------
    for (ik,k) in enumerate(init_wavenumbers)
        uhist[1+k,1] = init_amplitudes[ik]
    end
    # ----------------- Integrate -------------------
    u_old .= uhist[:,1]
    for it = 2:Nt
        timestep_rk4!(u_new,
                      urk1,urk2,urk3,urk4,
                      dturk1,dturk2,dturk3,dturk4,
                      usq,
                      u_old,
                      dt,
                      kmax
                     )
        uhist[:,it] .= u_new
        u_old .= u_new
    end
    return uhist
end

function plot_tbh
