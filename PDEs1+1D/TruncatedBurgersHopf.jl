using Printf: @sprintf
using CairoMakie
import Random
using JLD2: jldopen
using Infiltrator: @infiltrate

function tendency!(dtu, usq, u, kmax)
    usq .= 0
    dtu .= 0
    # Fill in the usq array
    usq[1] = abs2(u[1]) + 4*sum(abs2.(u[2:kmax+1]))
    for k = 1:kmax
        for m = 1:k-1
            usq[k+1] += 2 * u[m+1] * u[(k-m)+1]
        end
        usq[k+1] += 2 * u[1] * u[k+1]
        for m = (k+1):kmax
            usq[k+1] += 2 * u[m+1] * conj(u[(m-k)+1])
        end
    end
    dtu .= (-1im/2) .* (0:kmax) .* usq
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

function integrate_tbh(u_init, t_init, dt, Nt)
    NF = Float64
    # Simulate the Truncated Burgers-Hopf dynamics, which is a truncation of 
    # u_t + u*u_x = 0
    # to a finite number of Fourier modes. Don't even do FFT on this simplest of demos. 
    # ----------------- Allocate arrays -------------
    # scratch space for RK4
    kmax = length(u_init) - 1
    (
     u_old,u_new,
     # scratch for RK4
     usq,
     urk1,urk2,urk3,urk4,
     dturk1,dturk2,dturk3,dturk4
    ) = ntuple(_->zeros(Complex{NF}, (kmax+1,)), 11)
    uhist = zeros(Complex{NF}, (kmax+1, Nt)) # u in spectral space 
    thist = t_init .+ (0:Nt-1).*dt
    # ------- Initialize -------
    u_old .= u_init
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
    return uhist, thist
end

function spec2grid(u::Vector{<:Complex{NF}}, Nx::Integer) where NF
    kmax = length(u) - 1
    ug = zeros(NF, Nx)
    ug .+= u[1]
    xs = collect(range(0, 2pi; length=Nx))
    for k = 1:kmax
        ug .+= 2 * real.(u[k+1] .* exp.(1im .* k .* xs))
    end
    return ug
end

function spec2grid(uhist::Matrix{<:Complex{NF}}, Nx::Integer) where NF
    return mapslices(u->spec2grid(u,Nx), uhist; dims=1) 
end

function theme_ax()
    thmax = (;
             xlabelsize=10, ylabelsize=10, titlesize=12,
             xticklabelsize=8, yticklabelsize=8, 
             xlabelfont="Menlo", xticklabelfont="Menlo", ylabelfont="Menlo", yticklabelfont="Menlo", titlefont="Menlo",
             xgridvisible=false, ygridvisible=false,
            )
    return thmax
end

    

function plot_tbh_hov_trace(uhist, thist, outfilename)
    Nk,Nt = size(uhist)
    Nx = 251
    ughist = spec2grid(uhist, Nx)
    uglimits = [-1,1].*maximum(abs.(ughist))
    Rlimits = uglimits
    xs = collect(range(0, 2pi; length=Nx))

    # Trace out an observable 
    x_target = pi
    ix = argmin(abs.(xs .- x_target)) # location where the observable is measured 
    Roft = ughist[ix,:]

    thmax = theme_ax()
    xlimits = (0, 2pi)
    dt = thist[2]-thist[1]
    tlimits = (thist[1]-dt/2, thist[end]+dt/2)
    fig = Figure(size=(600,300))
    lout = fig[1,1] = GridLayout()

    axhov = Axis(lout[1,1]; thmax..., xticklabelsvisible=false, ylabel="𝑥", yticks=([0,pi,2pi], ["0","π", "2π"]), limits=(tlimits,xlimits), title="𝑢(𝑥,𝑡)")
    heatmap!(axhov, thist, xs, ughist'; colormap=:coolwarm, colorrange=uglimits)
    hlines!(axhov, xs[ix]; color=:black, linestyle=(:dash,:dense))

    Rtickvalues = [Rlimits[1], 0, Rlimits[2]]
    Rticklabels = (u->@sprintf("%.1f", u)).(Rtickvalues)
    axRoft = Axis(lout[2,1]; thmax..., ylabel="𝑅(𝑡)", xlabel="𝑡", limits=(tlimits, tuple(Rlimits...)), yticks=(Rtickvalues, Rticklabels), )
    lines!(axRoft, thist, Roft; color=:black)
    linkxaxes!(axhov, axRoft)
    rowsize!(lout, 1, Relative(4/5))
    rowgap!(lout, 1, 10)
    save(outfilename, fig)
end

function main()
    todo = Dict{String,Bool}(
                             "spinup" =>           1,
                             "plot_spinup" =>      1,
                            )
    # ---------------- Output directories -----------
    dirout_base = "/Users/justinfinkel/Documents/postdoc_mit/computing/COAST_results/PDEs1+1D/2026-06-14/1"
    mkpath(dirout_base)
    dirout_data = joinpath(dirout_base, "data")
    dirout_plot = joinpath(dirout_base, "plot")
    mkpath.([dirout_data,dirout_plot])
    # ----------------- Parameters ------------------
    kmax = 8 # maximum wavenumber to retain 
    ks = collect(0:1:kmax)
    # ----------------- Simulation parameters -------
    dt = 0.01
    # ----------------- Ensemble parameters ---------
    duration_spinup = 15.0
    duration_spinon = 6.0 
    # ----------------- spinup --------------
    if todo["spinup"]
        rng = Random.MersenneTwister(48401)
        init_wavenumbers = [1,2] #,4,8]
        init_amplitudes = [0.5,0.3] #,0.1,0.05]
        init_phases = 2pi .* rand(rng, Float64, length(init_amplitudes)) 
        u_init = zeros(ComplexF64, (kmax+1,))
        for ik = 1:length(init_wavenumbers)
            u_init[init_wavenumbers[ik]+1] = init_amplitudes[ik] * exp(1im * init_phases[ik])
        end
        u_init[1] = 1/3 
        t_init = 0.0
        Nt_spinup = round(Int64, duration_spinup/dt)
        uhist_spinup,thist_spinup = integrate_tbh(u_init, t_init, dt, Nt_spinup)
        jldopen(joinpath(dirout_data,"spinup.jld2"), "w") do f
            f["uhist"] = uhist_spinup
            f["thist"] = thist_spinup
        end
    end
    if todo["plot_spinup"]
        uhist,thist = jldopen(joinpath(dirout_data,"spinup.jld2"),"r") do f
            f["uhist"], f["thist"]
        end
        plot_tbh_hov_trace(uhist, thist, joinpath(dirout_plot, "spinup.png"))
    end

    #if todo["spinoff"]
    #    u_init = uhist[:,end]
    #end
    return uhist, thist
end

uhist, thist = main()


