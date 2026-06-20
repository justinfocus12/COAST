using Printf: @sprintf
using CairoMakie
import Random
import Extremes
using JLD2: jldopen, @load
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

function compute_peaks_over_threshold(Rhist::Vector{NF}, threshold::Real, buffer::Integer) where NF<:Real
    Nt = length(Rhist)
    Rspeak = Vector{NF}([])
    itspeak = Vector{Integer}([])
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
    return Rspeak, itspeak
end

function cquantile_gpd(excprob::Real, logscale::Real, shape::Real; loc::Real=0.0)
    return loc + exp(logscale)/shape * (1/excprob^shape - 1)
end

function survival_fun_gpd(x::Real, logscale::Real, shape::Real; loc::Real=0.0)
    return (1 + shape * (x - loc)/exp(logscale))^(-1/shape)
end


function compute_intensity_statistics(Rfile::String, Rstatsfile::String, excprob_approx::NFR, Nbin::Integer, peakbuffer::NFt) where {NFR<:Real,NFt<:Real}
    Rhist,thist = jldopen(Rfile,"r") do f
        return f["Rhist"], f["thist"]
    end
    dt = thist[2] - thist[1]
    Nt = length(thist)
    buffer = round(Int64, peakbuffer/dt)
    Rmin,Rmax = padbounds(Rhist,0.01)

    Rbineds = collect(range(Rmin, Rmax; length=Nbin+1))
    Rbinlos = Rbineds[1:Nbin]

    Rccdf = sum(Rhist .> Rbinlos'; dims=1)[1,:] ./ Nt
    i_bin_thresh = searchsortedfirst(Rccdf, excprob_approx; lt=(x,y)->(x>=y))
    threshold = Rbinlos[i_bin_thresh]
    excprob = Rccdf[i_bin_thresh]

    # Now start dealing with severities, i.e. Peaks, called S
    Ss,Sits = compute_peaks_over_threshold(Rhist, threshold, buffer)
    Npeak = length(Ss)

    Sbinlos_unif = Rbinlos[i_bin_thresh:Nbin]
    Sccdf_empest_unifbins = sum(Ss .> Sbinlos_unif'; dims=1)[1,:] ./ Npeak
    # Fit GPD and rearrange tail bins to have uniformly spaced quantiles 
    GPD = Extremes.gpfitpwm(Ss .- threshold)
    gpdlogscale,gpdshape = GPD.θ̂
    Sccdf_gpdest_unifbins = survival_fun_gpd.(Sbinlos_unif, gpdlogscale, gpdshape; loc=threshold)
    Sccdf_gpdest_gpdbins = collect(range(excprob, 0; length=Nbin-i_bin_thresh+2)[1:end-1])
    Sbinlos_gpd = cquantile_gpd.(Sccdf_gpdest_gpdbins, gpdlogscale, gpdshape; loc=threshold)
    Sccdf_empest_gpdbins = sum(Ss .> Sbinlos_gpd'; dims=1)[1,:] ./ Npeak
    Sccdf_gpdest_gpdbins = survival_fun_gpd.(Sbinlos_gpd, gpdlogscale, gpdshape; loc=threshold) # should be same, uniformly spaced
    
    # Estmate GEV and GPD parameters for the tail, and make an empirical histogram wth evnly spaced bins in estimated probability space. 
    jldopen(Rstatsfile,"w") do f
        f["thist"] = thist
        f["Rbinlos"] = Rbinlos
        f["Rccdf"] = Rccdf
        f["threshold"] = threshold
        f["i_bin_thresh"] = i_bin_thresh
        f["Ss"] = Ss
        f["Sits"] = Sits 
        f["excprob"] = excprob
        f["gpdlogscale"] = gpdlogscale
        f["gpdshape"] = gpdshape
        f["Sbinlos_unif"] = Sbinlos_unif
        f["Sccdf_empest_unifbins"] = Sccdf_empest_unifbins
        f["Sccdf_gpdest_unifbins"] = Sccdf_gpdest_unifbins
        f["Sccdf_gpdest_gpdbins"] = Sccdf_gpdest_gpdbins
        f["Sbinlos_gpd"] = Sbinlos_gpd
        f["Sccdf_empest_gpdbins"] = Sccdf_empest_gpdbins
    end
end

function compute_intensity(ufile::String, Rfile::String)
    uhist,thist = read_history(ufile)
    Rhist = intensity(uhist)
    jldopen(Rfile, "w") do f
        f["Rhist"] = Rhist
        f["thist"] = thist
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
    uatx = sum(2*real(u[1]) .+ real.(u[2:kmax+1] .* exp.(1im .* (1:kmax) .* x)))
    return exp.(uatx)
end
function intensity(uhist::Matrix,x::NF=pi) where NF <: Real
    return mapslices(u->intensity(u,x), uhist; dims=1)[1,:]
end

function plot_boosts(ufilename::String, Rfilename::String, Rstatsfilename::String, dirout_data::String, dirout_plot::String, sim_params::NamedTuple, boost_params::NamedTuple)
    (; astmin, astmax, aststep, Ndsc, bst, kspert, maxdphase, maxdtpeak) = boost_params
    (; dt,) = sim_params
    @load ufilename uhist thist
    @load Rfilename Rhist
    @load Rstatsfilename Sits Ss threshold Sbinlos_unif Sccdf_empest_unifbins

    NFu = typeof(uhist[1,1])
    NFreal = typeof(real(uhist[1,1]))
    NFt = typeof(thist[1])
    Nanc = length(Sits)
    ianc = 2
    bsNt = round(Integer, bst/dt)
    asts = collect(range(astmin, astmax; step=aststep))
    asNts = round.(Integer, asts./dt)
    Nast = findlast(asNts .< Sits[ianc])
    maxdNtpeak = round(Integer, maxdtpeak/dt)
    for iast = 1:Nast
        ast = asts[iast]
        asNt = asNts[iast]

        (Sits[ianc]-asNt <= 0) && continue

        Rhist_dscs = zeros(NFreal, (asNt+bsNt+1,Ndsc))
        thist_dscs = zeros(NFt, (asNt+bsNt+1,Ndsc))
        dSit_dscs = zeros(Integer, (Ndsc,))
        S_dscs = zeros(NFreal, (Ndsc,))
        Rdsc_at_Sitancs = zeros(NFreal, (Ndsc,))
        jldopen(joinpath(dirout_data, "boosts.jld2"), "r") do ff
            anckey = "ianc$(ianc)"
            astkey = "iast$(iast)"
            for idsc = 1:Ndsc
                dsckey = "idsc$(idsc)"
                aadkey = joinpath(anckey,astkey,dsckey)
                Rhist_dscs[:,idsc] .= ff[joinpath(aadkey,"Rhist")]
                S_dscs[idsc] = ff[joinpath(aadkey,"S_dsc")]
                thist_dscs[:,idsc] .= ff[joinpath(aadkey,"thist")]
                dSit_dscs[idsc] = ff[joinpath(aadkey,"dSit_dsc")]
                Rdsc_at_Sitancs[idsc] = ff[joinpath(aadkey,"Rdsc_at_Sitanc")]
            end
        end
        Sbinwidths = vcat(diff(Sbinlos_unif), maximum(Ss)-Sbinlos_unif[end])
        Spdf = -diff(vcat(Sccdf_empest_unifbins, 0)) ./ Sbinwidths
        Sccdf_boost = sum(S_dscs .> Sbinlos_unif'; dims=1)[1,:]./Ndsc
        Spdf_boost = -diff(vcat(Sccdf_boost, 0)) ./ Sbinwidths
        Spdfmin = minimum(filter(p->p>0, vcat(Spdf, Spdf_boost)))
        Spdfmax = maximum(vcat(Spdf, Spdf_boost))

        Rlimits = (minimum(Rhist), 2*Sbinlos_unif[end]-Sbinlos_unif[end-1])
        
        thmax = theme_ax()
        fig = Figure(size=(400,300))
        lout = fig[1,1] = GridLayout()

        ax1 = Axis(lout[1,1]; thmax..., xlabel="𝑡", ylabel="𝑅(𝑥(𝑡))", limits=((-asts[end], bst,),Rlimits))
        ax2 = Axis(lout[1,2]; thmax..., xlabel="PDF", ylabel="𝑅*",ylabelvisible=false, yticklabelsvisible=false, limits=((Spdfmin,Spdfmax),Rlimits))

        tancmax = thist[Sits[ianc]]
        for idsc = 1:Ndsc
            lines!(ax1, thist_dscs[:,idsc].-tancmax, Rhist_dscs[:,idsc]; color=:red, linestyle=:solid)
            scatter!(ax1, dt*dSit_dscs[idsc], S_dscs[idsc]; marker=:star6, color=:red)
        end
        lines!(ax1, dt.*collect(-(asNts[Nast]):bsNt), Rhist[Sits[ianc].+(-asNts[Nast]:bsNt)]; color=:black, linestyle=(:dash,:dense))
        scatter!(ax1, 0, Ss[ianc]; marker=:star6, color=:black)
        hlines!(ax1, threshold; color=:grey79)

        lines!(ax2, Spdf, Sbinlos_unif.+Sbinwidths./2; color=:black, )
        lines!(ax2, Spdf_boost, Sbinlos_unif.+Sbinwidths./2; color=:red, )
        hlines!(ax1, threshold; color=:grey79)
        hlines!(ax2, threshold; color=:grey79)


        linkyaxes!(ax1, ax2)
        colsize!(lout, 1, Relative(5/6))

        save(joinpath(dirout_plot, "boosts_ianc$(ianc)_iast$(iast).png"), fig)
    end
    return
end



function boost_peaks(ufilename::String, Rstatsfilename::String, dirout_data::String, sim_params::NamedTuple, boost_params::NamedTuple; overwrite_boosts::Bool=false)

    (; astmin, astmax, aststep, Ndsc, bst, kspert, maxdphase, maxdtpeak) = boost_params
    (; dt,) = sim_params
    @load Rstatsfilename Sits
    #(; Sits, ) = jldopen(Rstatsfilename, "r") do f; return NamedTuple(Symbol(key)=>f[key] for key in keys(f)); end
    @load ufilename uhist thist
    #uhist, thist = jldopen(ufilename, "r") do f; return f["uhist"],f["thist"]; end
    NFu = typeof(uhist[1,1])
    NFreal = typeof(real(uhist[1,1]))
    Nanc = length(Sits)
    asts = collect(range(astmin, astmax; step=aststep))
    Nast = length(asts)
    bsNt = round(Integer, bst/dt)
    maxdNtpeak = round(Integer, maxdtpeak/dt)
    boostfilename = joinpath(dirout_data,"boosts.jld2") 
    overwrite_boosts && rm(boostfilename)
    for ianc = 1:2 #Nanc
        anckey = "ianc$(ianc)"
        for iast = 1:Nast
            astkey = "iast$(iast)"
            ast = asts[iast]
            asNt = round(Integer,ast/dt)
            it_init_dsc = Sits[ianc] - asNt
            it_init_dsc <=0 && continue
            t_init_dsc = thist[it_init_dsc]
            u_init_dsc = zeros(NFu, size(uhist,1)) #uhist[:,it_init_dsc]
            rng = Random.MersenneTwister(27395)
            jldopen(boostfilename, "a+") do ff
                Ndsc_done = (anckey in keys(ff) && astkey in keys(ff[anckey])) ? length(keys(ff[anckey][astkey])) : 0
                for idsc = (Ndsc_done)+1:Ndsc
                    dsckey = "idsc$(idsc)"
                    aadkey = joinpath(anckey,astkey,dsckey)
                    ast = asts[iast]
                    asNt = round(Integer,ast/dt)
                    it_init_dsc = Sits[ianc] - asNt
                    t_init_dsc = thist[it_init_dsc]
                    u_init_dsc .= uhist[:,it_init_dsc]
                    u_init_dsc[kspert.+1] .*= exp.((2pi*1im*maxdphase) .* (2*rand(rng, NFreal)-1))
                    uhist_dsc,thist_dsc = integrate_tbh(u_init_dsc, t_init_dsc, dt, asNt+bsNt+1) 
                    Rhist_dsc = intensity(uhist_dsc)
                    S_dsc,dSit_dsc = findmax(Rhist_dsc[(asNt+1).+(-maxdNtpeak:maxdNtpeak)])
                    dSit_dsc -= (maxdNtpeak+1)
                    ff[joinpath(aadkey,"thist")] = thist_dsc
                    ff[joinpath(aadkey,"uhist")] = uhist_dsc
                    ff[joinpath(aadkey,"Rhist")] = Rhist_dsc
                    ff[joinpath(aadkey,"S_dsc")] = S_dsc
                    ff[joinpath(aadkey,"dSit_dsc")] = dSit_dsc
                    ff[joinpath(aadkey,"Rdsc_at_Sitanc")] = Rhist_dsc[asNt+1]
                end
            end
        end
    end
end







function plot_tbh_trace_pdf(ufilename, Rfilename, Rstatsfilename, outfilename)
    uhist,thist = read_history(ufilename)
    Rhist = jldopen(Rfilename,"r") do f; return f["Rhist"]; end

    (; Rbinlos, Rccdf, threshold, i_bin_thresh, Ss, Sits, excprob, Sbinlos_unif, Sccdf_empest_unifbins, Sccdf_gpdest_unifbins) = jldopen(Rstatsfilename, "r") do f; return NamedTuple(Symbol(key)=>f[key] for key in keys(f)); end
    Npeak = length(Ss)
    tspan_plot_R = thist[Sits[div(Npeak,2)]] .+ [-20,20] #[thist[Sits[1]],thist[1]+30]
    tspan_plot_S = thist[Sits[div(Npeak,2)]] .+ [-100,100] #[thist[1],thist[1]+100]
    plot_tbh_trace_pdf(thist, Rhist, Ss, Sits,
                       Rbinlos, Rccdf,
                       Sbinlos_unif, Sccdf_empest_unifbins, Sccdf_gpdest_unifbins, 
                       tspan_plot_R,tspan_plot_S,threshold,
                       outfilename)
end

function plot_tbh_trace_pdf(
        thist, Rhist, Ss, Sits,
        Rbinlos, Rccdf,
        Sbinlos_unif, Sccdf_empest_unifbins, Sccdf_gpdest_unifbins,
        tspan_plot_R, tspan_plot_S, threshold, 
        outfilename)
    # Plot a limited-time trace of the observable
    NbinR = length(Rbinlos)
    Rmin,Rmax = padbounds(Rhist,0.01)
    Rbinwidths = vcat(diff(Rbinlos), Rbinlos[NbinR]-Rbinlos[NbinR-1])./2
    Rbinmids = Rbinlos .+ Rbinwidths./2
    Rpdf = vcat(-diff(Rccdf),Rccdf[end]) ./ Rbinwidths
    Rpdfmin = minimum(filter(c->c>0, Rpdf))
    Rpdfmax = maximum(Rpdf)

    NbinS = length(Sbinlos_unif)
    Sbinwidths = vcat(diff(Sbinlos_unif), Sbinlos_unif[NbinS]-Sbinlos_unif[NbinS-1])./2
    Sbinmids = Sbinlos_unif .+ Sbinwidths./2
    Sccdfmin = min((minimum(filter(c->c>0, Sccdf)) for Sccdf=(Sccdf_empest_unifbins, Sccdf_gpdest_unifbins))...)
    
    itfirstR = searchsortedfirst(thist, tspan_plot_R[1])
    itlastR = searchsortedlast(thist, tspan_plot_S[2])
    iSfirst = searchsortedfirst(thist[Sits], tspan_plot_S[1])
    iSlast = searchsortedlast(thist[Sits], tspan_plot_S[2])

    thmax = theme_ax()
    fig = Figure(size=(400,500))
    lout = fig[1,1] = GridLayout()

    ax1 = Axis(lout[1,1]; thmax..., limits=((tspan_plot_R[1],tspan_plot_R[2]),(Rmin,Rmax)), xlabel="𝑡", ylabel="𝑅(𝑥(𝑡))", title="Intensity")
    hlines!(ax1, threshold; color=:grey79)
    lines!(ax1, thist[itfirstR:itlastR], Rhist[itfirstR:itlastR]; color=:black)

    ax2 = Axis(lout[1,2]; thmax..., limits=((Rpdfmin,Rpdfmax),(Rmin,Rmax)), title="PDF", ylabelvisible=false, yticklabelsvisible=false, xscale=log10, xticklabelrotation=-pi/2)
    lines!(ax2, replace(Rpdf, 0=>NaN), Rbinmids; color=:black)

    ax3 = Axis(lout[2,1]; thmax..., limits=((tspan_plot_S[1],tspan_plot_S[2]),(threshold,Rmax)), xlabel="𝑡*", ylabel="𝑅*", title="Severities")
    lines!(ax3, thist[Sits[iSfirst]:Sits[iSlast]], Rhist[Sits[iSfirst]:Sits[iSlast]]; color=:black, linewidth=1)
    scatter!(ax3, thist[Sits[iSfirst:iSlast]], Ss[iSfirst:iSlast]; color=:red, markersize=5, marker=:star6)

    ax4 = Axis(lout[2,2]; thmax..., limits=((Sccdfmin/2,1.0),(threshold,Rmax)), title="CCDF", ylabelvisible=false, yticklabelsvisible=false, xscale=log10, xticklabelrotation=-pi/2, xticks=(xticklabels=[Sccdfmin,sqrt(Sccdfmin*1.0),1.0]; (xticklabels,(x->@sprintf("%.0E", x)).(xticklabels))))
    lines!(ax4, replace(Sccdf_empest_unifbins, 0=>NaN), Sbinlos_unif; color=:black, linewidth=2, linestyle=(:dash,:dense))
    lines!(ax4, replace(Sccdf_gpdest_unifbins, 0=>NaN), Sbinlos_unif; color=:red)

    linkyaxes!(ax1,ax2)
    linkyaxes!(ax3,ax4)
    colsize!(lout, 1, Relative(4/5))
    colgap!(lout, 1, 0)

    save(outfilename, fig)
end
    
function plot_tbh_hov_trace(ufilename::String, outfilename::String)
    uhist,thist = read_history(ufilename)
    plot_tbh_hov_trace(uhist,thist,outfilename)
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
    tlimits = (thist[1]-dt/2, min(thist[end],thist[1]+30)+dt/2)
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
                             "spinup" =>                    0,
                             "plot_spinup" =>               0,
                             "spinon" =>                    0,
                             "plot_spinon" =>               0,
                             "compute_intensity" =>         0,
                             "compute_intensity_stats" =>   0,
                             "plot_intensity" =>            0,
                             "plot_intensity_stats" =>      0,
                             "spinoff" =>                   0,
                             "plot_spinoffs" =>             0,
                             "boost" =>                     0,
                             "plot_boosts" =>               1,
                            )
    # ---------------- Output directories -----------
    dirout_base = "/Users/justinfinkel/Documents/postdoc_mit/computing/COAST_results/PDEs1+1D/2026-06-17/3"
    mkpath(dirout_base)
    dirout_data = joinpath(dirout_base, "data")
    dirout_plot = joinpath(dirout_base, "plot")
    mkpath.([dirout_data,dirout_plot])
    # ----------------- Parameters ------------------
    kmax = 8 # maximum wavenumber to retain 
    ks = collect(0:1:kmax)
    # ----------------- Simulation parameters -------
    dt = 0.01
    sim_params = (; dt)
    # ----------------- Ensemble parameters ---------
    duration_spinup = 15.0
    duration_spinon = 5000.0 
    duration_spinoff = 6.0 
    Ndsc_spinoff = 3

    boost_params = (; astmin=1/4, astmax=8.0, bst=1.0, aststep=1/4, Ndsc=24, maxdphase=1/32, kspert=[kmax,], maxdtpeak=1/8)
    overwrite_boosts = true

    # ------------- Target parameters -----------
    excprob_approx = 0.01
    peakbuffer = 5.0
    Nbin = 60
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
        plot_tbh_hov_trace(joinpath(dirout_data,"spinup.jld2"), joinpath(dirout_plot, "spinup.png"))
    end
    if todo["spinon"]
        uhist_spinup,thist_spinup = read_history(joinpath(dirout_data,"spinup.jld2"))
        u_init = uhist_spinup[:,end]
        t_init = thist_spinup[end]
        Nt_spinon = round(Int64, duration_spinon/dt) + 1
        uhist_spinon,thist_spinon = integrate_tbh(u_init, t_init, dt, Nt_spinon)
        write_history(uhist_spinon,thist_spinon,joinpath(dirout_data,"spinon.jld2"))
    end
    if todo["compute_intensity"]
        compute_intensity(joinpath(dirout_data,"spinon.jld2"), joinpath(dirout_data,"spinonR.jld2"))
    end
    if todo["plot_spinon"]
        plot_tbh_hov_trace(joinpath(dirout_data,"spinon.jld2"),joinpath(dirout_plot, "spinon_hov_trace.png"))
    end
    if todo["compute_intensity_stats"]
        compute_intensity_statistics(joinpath.(dirout_data, ["spinonR.jld2","spinonRstats.jld2"])..., excprob_approx, Nbin, peakbuffer)
    end
    if todo["plot_intensity_stats"]
        plot_tbh_trace_pdf(joinpath.(dirout_data,["spinon.jld2","spinonR.jld2","spinonRstats.jld2"])..., joinpath(dirout_plot, "Rstats_spinon.png"))
    end

    if todo["spinoff"]
        Nt_spinoff = round(Int64, duration_spinoff/dt) + 1
        rng = Random.MersenneTwister(90193)
        uhist_spinon,thist_spinon = read_history(joinpath(dirout_data,"spinon.jld2"))
        kspert = [kmax,]
        for i_dsc = 1:Ndsc_spinoff
            phase_shift = rand(rng, Float64, length(kspert)) .* (2pi/1000)
            u_init = uhist_spinon[:,end] 
            u_init[kspert.+1] .*= exp.(1im * phase_shift)
            t_init = thist_spinon[end]
            uhist_spinoff,thist_spinoff = integrate_tbh(u_init, t_init, dt, Nt_spinoff)
            write_history(uhist_spinoff, thist_spinoff, joinpath(dirout_data,"spinoff_dsc$(i_dsc).jld2"))
        end
    end

    if todo["plot_spinoffs"]
        for i_dsc = 1:Ndsc_spinoff
            uhist_spinoff,thist_spinoff = read_history(joinpath(dirout_data,"spinoff_dsc$(i_dsc).jld2"))
            plot_tbh_hov_trace(joinpath(dirout_data,"spinoff_dsc$(i_dsc).jld2"), joinpath(dirout_plot,"spinoff_dsc$(i_dsc).png"))
        end
    end

    if todo["boost"]
        boost_peaks(
                    joinpath(dirout_data,"spinon.jld2"),
                    joinpath(dirout_data,"spinonRstats.jld2"),
                    dirout_data,
                    sim_params,
                    boost_params,
                    ;
                    overwrite_boosts=overwrite_boosts,
                   )
    end
    if todo["plot_boosts"]
        plot_boosts(joinpath(dirout_data,"spinon.jld2"), joinpath(dirout_data, "spinonR.jld2"), joinpath(dirout_data,"spinonRstats.jld2"), dirout_data, dirout_plot, sim_params, boost_params)
    end
    return 
end

main()


