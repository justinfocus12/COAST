using Printf: @sprintf
using CairoMakie
import Random
using JLD2: jldopen
using Infiltrator: @infiltrate
using Statistics: quantile, median, mean

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

function write_history(uhist::Matrix,thist::Vector,filename::String)
    jldopen(filename,"w") do f
        f["uhist"] = uhist
        f["thist"] = thist
    end
    return 
end
function read_history(filename::String)
    uhist,thist= jldopen(filename,"r") do f
        f["uhist"], f["thist"]
    end
    return uhist,thist
end

function compute_peaks_over_threshold(Rhist::Vector{NF}, excprob::Real, buffer::Integer) where NF<:Real
    Nt = length(Rhist)
    Rspeak = Vector{NF}([])
    itspeak = Vector{Integer}([])
    threshold = quantile(Rhist, 1-excprob)
    itprecluster = 0
    incluster = false
    valley_length = 0
    for it = 1:Nt
        if Rhist[it] > threshold
            if !incluster
                itprecluster = it - 1
            end
            incluster = true 
            valley_length = 0
        else
            valley_length += 1
            if valley_length > buffer
                if incluster
                    itpeaknew = itprecluster + argmax(Rhist[itprecluster+1:it])
                    push!(itspeak, itpeaknew)
                    push!(Rspeak, Rhist[itpeaknew])
                end
                incluster = false
            end
        end
    end
    return threshold, Rspeak, itspeak
end

function compute_extreme_statistics(Rfile::String, Rpeakfile::String, threshold::NFr, buffer::NFt) where {NFr<:Real,NFt<:Real}
    Rhist,thist = jldopen(Rfile,"r") do f
        return f["Rhist"], f["thist"]
    end
    dt = thist[2] - thist[1]
    buffer_integer = round(Int64, buffer/dt)
    threshold,Rspeak,itspeak = compute_peaks_over_threshold(Rhist, threshold, buffer)
    jldopen(Rpeakfile,"w") do f
        f["thist"] = thist
        f["threshold"] = threshold
        f["Rspeak"] = Rspeak
        f["itspeak"] = itspeak
    end

function compute_intensity(ufile::String, Rfile::String)
    uhist,thist = read_history(ufile)
    Rhist = intensity(uhist)
    jldopen(Rfile, "w") do f
        f["Rhist"] = Rhist_spinon
        f["thist"] = thist_spinon
    end
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
    thist = collect(t_init .+ (0:Nt-1).*dt)
    # ------- Initialize -------
    u_old .= u_init
    uhist[:,1] .= u_init
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
function padbounds(v::Vector{NF},padfrac::Real) where NF <: Real
    vmin,vmax = extrema(v)
    vrange = vmax - vmin
    vmin -= padfrac*vrange
    vmax += padfrac*vrange
    return (vmin,vmax)
end

function intensity(u::Vector,x::NF=pi) where NF <: Real
    kmax = length(u)-1
    return sum(2*real(u[1]) .+ real.(u[2:kmax+1] .* exp.(1im .* (1:kmax) .* x)))
end
function intensity(uhist::Matrix,x::NF=pi) where NF <: Real
    return mapslices(u->intensity(u,x), uhist; dims=1)[1,:]
end

function plot_tbh_trace_pdf(uhist, thist, tspan_plot_timeseries, tspan_plot_peaks, threshold, outfilename; Rbins::Union{Vector,Nothing}=nothing)
    # Plot a limited-time trace of the observable
    Rhist = intensity(uhist)
    if isnothing(Rbins)
        Rmin,Rmax = padbounds(Rhist,0.01)
        Rmin,Rmax = extrema(Rhist)
        Rbineds = collect(range(Rmin, Rmax; length=31))
    end
    Rbinlos = Rbineds[1:end-1]
    Rbinmids = (Rbineds[1:end-1] .+ Rbineds[2:end])./2
    Nbin = length(Rbinlos)
    ccdf_counts = sum(Rhist .> Rbineds'; dims=1)[1,:]
    @assert ccdf_counts[end] == 0
    ccdf = ccdf_counts./ccdf_counts[1]
    ccdfmin = minimum(filter(c->c>0, ccdf))
    pdf = (ccdf[1:end-1] .- ccdf[2:end]) ./ diff(Rbineds)
    pdfmin = (ccdf[1:end-1] .- ccdf[2:end]) ./ diff(Rbineds)
    
    itfirst_timeseries = searchsortedfirst(thist, tspan_plot_timeseries[1])
    itlast_timeseries = searchsortedlast(thist, tspan_plot_timeseries[2])
    itfirst_peaks = searchsortedfirst(thist, tspan_plot_peaks[1])
    itlast_peaks = searchsortedlast(thist, tspan_plot_peaks[2])

    ccdfmin = minimum(filter(c->c>0, ccdf))
    @show threshold

    thmax = theme_ax()
    fig = Figure(size=(400,500))
    lout = fig[1,1] = GridLayout()

    ax1 = Axis(lout[1,1]; thmax..., limits=((tspan_plot_timeseries[1],tspan_plot_timeseries[2]),(Rmin,Rmax)), xlabel="𝑡", ylabel="𝑅(𝑥(𝑡))")
    lines!(ax1, thist[itfirst_timeseries:itlast_timeseries], Rhist[itfirst_timeseries:itlast_timeseries]; color=:black)

    ax2 = Axis(lout[1,2]; thmax..., title="PDF", ylabelvisible=false, yticklabelsvisible=false, xscale=log10, xticklabelrotation=-pi/2)
    lines!(ax2, replace(pdf, 0=>NaN), Rbinmids; color=:black)

    ax3 = Axis(lout[2,1]; thmax..., limits=((tspan_plot_peaks[1],tspan_plot_peaks[2]),(threshold,Rmax)), xlabel="𝑡", ylabel="𝑅(𝑥(𝑡))")
    scatter!(ax3, thist[itfirst_peaks:itlast_peaks], Rhist[itfirst_peaks:itlast_peaks]; color=:black, markersize=2)

    ax4 = Axis(lout[2,2]; thmax..., limits=((ccdfmin/2,excprob),(threshold,Rmax)), title="CCDF", ylabelvisible=false, yticklabelsvisible=false, xscale=log10, xticklabelrotation=-pi/2, xticks=(xticklabels=[ccdfmin,(ccdfmin+excprob)/2,excprob]; (xticklabels,(x->@sprintf("%.0E", x)).(xticklabels))))
    lines!(ax4, replace(ccdf, 0=>NaN), Rbineds; color=:black)
e
    linkyaxes!(ax1,ax2)
    linkyaxes!(ax3,ax4)
    colsize!(lout, 1, Relative(4/5))
    colgap!(lout, 1, 0)

    save(outfilename, fig)
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
                             "spinon" =>           1,
                             "potspinon" =>        1,
                             "plot_spinon" =>      1,
                             "spinoff" =>          1,
                             "plot_spinoffs" =>    1,
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
    duration_spinon = 200.0 
    duration_spinoff = 6.0 
    N_dsc = 3
    # ------------- Target parameters -----------
    excprob = 0.01
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
        Nt_spinup = round(Int64, duration_spinup/dt) + 1
        uhist_spinup,thist_spinup = integrate_tbh(u_init, t_init, dt, Nt_spinup)
        write_history(uhist_spinup,thist_spinup,joinpath(dirout_data,"spinup.jld2"))
    end
    if todo["plot_spinup"]
        uhist,thist = read_history(joinpath(dirout_data,"spinup.jld2"))
        plot_tbh_hov_trace(uhist, thist, joinpath(dirout_plot, "spinup.png"))
    end
    if todo["spinon"]
        uhist_spinup,thist_spinup = read_history(joinpath(dirout_data,"spinup.jld2"))
        u_init = uhist_spinup[:,end]
        t_init = thist_spinup[end]
        Nt_spinon = round(Int64, duration_spinon/dt) + 1
        uhist_spinon,thist_spinon = integrate_tbh(u_init, t_init, dt, Nt_spinon)
        write_history(uhist_spinon,thist_spinon,joinpath(dirout_data,"spinon.jld2"))
    end
    if todo["compute_intensity_spinon"]
        compute_intensity(joinpath(dirout_data,"spinon.jld2"), joinpath(dirout_data,"spinonR.jld2"))
    end
    if todo["potspinon"]
        compute_extreme_statistics(joinpath.(dirout_data, ["spinonR.jld2","spinonextstats.jld2"])..., excprob, buffer)
    end
    if todo["plot_spinon"]
        uhist,thist = read_history(joinpath(dirout_data,"spinon.jld2"))
        plot_tbh_hov_trace(uhist, thist, joinpath(dirout_plot, "spinon.png"))
        plot_tbh_trace_pdf(uhist, thist, [thist[1],thist[1]+30], [thist[1],thist[end]], joinpath(dirout_plot, "Rhist_spinon.png"))
    end

    if todo["spinoff"]
        Nt_spinoff = round(Int64, duration_spinoff/dt) + 1
        rng = Random.MersenneTwister(90193)
        uhist_spinon,thist_spinon = read_history(joinpath(dirout_data,"spinon.jld2"))
        for i_dsc = 1:N_dsc
            phase_shift = rand(rng, Float64, kmax) .* (2pi/10)
            u_init = uhist_spinon[:,end] 
            u_init[2:kmax+1] .*= exp.(1im * phase_shift)
            t_init = thist_spinon[end]
            uhist_spinoff,thist_spinoff = integrate_tbh(u_init, t_init, dt, Nt_spinoff)
            write_history(uhist_spinoff, thist_spinoff, joinpath(dirout_data,"spinoff_dsc$(i_dsc).jld2"))
        end
    end

    if todo["plot_spinoffs"]
        for i_dsc = 1:N_dsc
            uhist_spinoff,thist_spinoff = read_history(joinpath(dirout_data,"spinoff_dsc$(i_dsc).jld2"))
            plot_tbh_hov_trace(uhist_spinoff, thist_spinoff, joinpath(dirout_plot,"spinoff_dsc$(i_dsc).png"))
        end

    end
    return 
end

main()


