
function van_der_corput(N)
    # Generate the first N points of the van der corput sequence 
    max_bit_length = floor(Int, 1+log2(N))
    xs = zeros(Float64, N)
    n = 1
    for bit_length = 1:max_bit_length
        for k = 1:(2^(bit_length-1))
            if n <= N
                xs[n] = (2*k-1)/(2^bit_length)
            end
            n += 1
        end
    end
    return xs
end

function empirical_ccdf(x::Vector{<:Number})
    N = length(x)
    order = sortperm(x)
    ccdf = (collect(range(N, 1; step=-1)) .- 0.5)./N
    return x[order], ccdf
end

function compute_empirical_ccdf(xs::Vector{Float64}, bin_lower_edges::Vector{Float64})
    @assert all(diff(bin_lower_edges) .> 0)
    @assert length(xs) > 0
    ccdf = sum(Float64, xs .> bin_lower_edges'; dims=1)[1,:] ./ length(xs)
    return ccdf
end

function compute_conditional_entropy_proxy(xs::Vector{Float64}, bin_lower_edges::Vector{Float64})
    ccdf = compute_empirical_ccdf(xs, bin_lower_edges) #sum(Float64, xs .> bin_lower_edges'; dims=1)[1,:]
    pmf = vcat(-diff(ccdf), ccdf[end])
    condent = -sum(xlog2x.(pmf)) + xlog2x(ccdf[1])
    @assert condent >= 0
    return condent 
end

function compute_thresholded_entropy(xs::Vector{Float64}, bin_lower_edges::Vector{Float64})
    pmf = compute_empirical_ccdf(xs, bin_lower_edges) #sum(Float64, xs .> bin_lower_edges'; dims=1)[1,:]
    pmf[1:end-1] .-= pmf[2:end]
    if length(xs) > 0
        pmf ./= length(xs)
    end
    entropy = -sum(xlog2x.(pmf))
    return entropy
end


function ccdf2pmf(ccdf::Vector{Float64})
    pmf = vcat(-diff(ccdf), ccdf[end])
    return pmf
end

function ccdf2pdf(ccdf::Vector{Float64}, bin_edges::Vector{Float64})
    return ccdf2pmf(ccdf) ./ diff(bin_edges)
end

function chi2div(ccdf_truth::Vector{Float64}, ccdf_approx::Vector{Float64})
    pmf_truth = ccdf2pmf(ccdf_truth)
    pmf_approx = ccdf2pmf(ccdf_approx)
    return sum((pmf_truth .- pmf_approx).^2 ./ pmf_truth)
end

function hellingerdist(ccdf_truth::Vector{Float64}, ccdf_approx::Vector{Float64})
    pmf_truth = ccdf2pmf(ccdf_truth)
    pmf_approx = ccdf2pmf(ccdf_approx)
    return sum((sqrt.(pmf_truth) .- sqrt.(pmf_approx)).^2)
end

function wassersteindist(ccdf_truth::Vector{Float64}, ccdf_approx::Vector{Float64})
    return sum(abs.(ccdf2pmf(ccdf_truth) .- ccdf2pmf(ccdf_approx)))
end

function xlog2x(x::Float64)
    return xlogx(x)/log(2)
end
function xlog2y(x::Float64, y::Float64)
    return xlogy(x,y)/log(2)
end

function kldiv(ccdf_truth::Vector{Float64}, ccdf_approx::Vector{Float64})
    pmf_truth, pmf_approx = map(ccdf2pmf, (ccdf_truth, ccdf_approx))
    kl = sum(xlog2y.(pmf_approx, pmf_approx./pmf_truth))
    return kl
end
function nlg1m(x::Number) 
    return -log1p(-x)/log(2) # log_2(1/(1-x))
end
function nlg1m_inv(y::Number) # 1 - 1/(2^y)
    return -expm1(-y*log(2))
end
