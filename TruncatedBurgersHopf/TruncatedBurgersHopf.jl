using Printf: @sprintf

function main()
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
    dt_solve = 0.01
    Nt_solve = 100
    Lx = 3.8 # domain width 
    Nx = 128
    # ----------------- Allocate arrays -------------
    u_s = zeros(Complex{NF}, (kmax+1, Nt_solve)) # u in spectral space 
    dt_u_s,u2_s = ntuple(_->zeros(Complex{NF}, (kmax+1, Nt_solve)), 2)
    # ----------------- Initialize ------------------
    for (ik,k) in enumerate(init_wavenumbers)
        u_s[1+k,1] = init_amplitudes[ik]
    end
    # ----------------- Integrate -------------------
    for it = 2:Nt_solve
        dt_u_s .= 0
        u2_s .= 0
        # Fill in the u2_s array
        u2_s[1] = abs2(u_s[1]) + 4*sum(abs2.(u_s[2:kmax+1]))
        for k = 1:kmax
            for m = 1:k-1
                u2_s[k+1] += 2 * u_s[m+1] * u_s[(k-m)+1]
            end
            u2_s[1] += 2 * u_s[1] * u_s[k+1]
            for m = (k+1):kmax
                u2_s[k+1] += 2 * u_s[m+1] * conj(u_s[(m-k)+1])
            end
        end
        dt_u_s .= 1im .* ks .* u2_s
        # Time step now
    end
end
