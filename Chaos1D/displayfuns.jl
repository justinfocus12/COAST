function poweroftwostring(k::Int64)
    symbols = ["2⁰","2¹","2²","2³","2⁴","2⁵","2⁶","2⁷","2⁸","2⁹","2¹⁰","2¹¹","2¹²","2¹³","2¹⁴","2¹⁵","2¹⁶","2¹⁷","2¹⁸"]
    if 0 <= k <= length(symbols)-1
        return symbols[k+1]
    end
    return "2^$(k)"
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

