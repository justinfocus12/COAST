
function astcolors()
    return Dict(
                "TotEnt" => :goldenrod,
                "ThrEnt" => :orangered,
                "XclEnt" => :dodgerblue2,
                "XclEntOne" => :skyblue2,
                "astunif" => :firebrick,
               )
end

function get_themes()
    theme_ax = (xticklabelsize=8, yticklabelsize=8, xlabelsize=10, ylabelsize=10, xgridvisible=false, ygridvisible=false, titlefont="Menlo", ylabelfont="Menlo", xlabelfont="Menlo", xticklabelfont="Menlo", yticklabelfont="Menlo", titlesize=10)
    theme_leg = (labelsize=8, framevisible=true, labelfont="Menlo", titlefont="Menlo")
    return theme_ax,theme_leg
end

function poweroftwostring(k::Int64)
    symbols = ["2⁰","2¹","2²","2³","2⁴","2⁵","2⁶","2⁷","2⁸","2⁹","2¹⁰","2¹¹","2¹²","2¹³","2¹⁴","2¹⁵","2¹⁶","2¹⁷","2¹⁸"]
    if 0 <= k <= length(symbols)-1
        return symbols[k+1]
    end
    return "2^$(k)"
end

function powerofhalfstring(k::Int64)
    symbols = ["(½)⁰","(½)¹","(½)²","(½)³","(½)⁴","(½)⁵","(½)⁶","(½)⁷","(½)⁸","(½)⁹","(½)¹⁰","(½)¹¹","(½)¹²","(½)¹³","(½)¹⁴","(½)¹⁵","(½)¹⁶","(½)¹⁷","(½)¹⁸"]
    if 0 <= k <= length(symbols)-1
        return symbols[k+1]
    end
    return "(½)^$(k)"
end


function supscr(k::Int64)
    sss_onedigit = ["⁰","¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹"]
    kdigs = reverse(digits(abs(k)))
    ss = join([sss_onedigit[kd+1] for kd=kdigs])
    if k < 0
        ss = "⁻" * ss
    end
    return ss
end

function scinot2(x::Number)
    powerof2 = floor(Int,log2(x))
    coeff = round(Int64,x/2^powerof2)
    return @sprintf("%d×2%s",coeff,supscr(powerof2))
end
    

function hatickvals(ylo,yhi)
    nlg_first = ceil(Int, nlg1m(ylo))
    nlg_last = floor(Int, nlg1m(yhi))
    nlgs = unique(round.(Int, range(nlg_first, nlg_last; length=3)))
    tickvals = nlg1m_inv.(nlgs)
    ticklabs = ["1−2^(−$(tv))" for tv=tickvals] 
    return (tickvals,ticklabs)
end

Makie.inverse_transform(nlg1m) = nlg1m_inv
Makie.defaultlimits(::typeof(nlg1m)) = (nlg1m_inv(2.0), nlg1m_inv(8.0))
Makie.defined_interval(::typeof(nlg1m)) = Makie.OpenInterval(0.0,1.0) 

