module EnsembleMod

import JLD2
import Graphs

struct ImmutableTrajectory
    tphinit::Number  # physical time unit 
    tfin::Number
    init_cond::String
    forcing::String
    term_cond::String
    history::String
end

mutable struct Trajectory
    tphinit::Number  # physical time unit 
    tfin::Number
    init_cond::String
    forcing::String
    term_cond::String
    history::String
end

mutable struct ImmutableTrajectoryEnsemble 
    trajs::Vector{ImmutableTrajectory}
    famtree::Graphs.SimpleGraphs.SimpleDiGraph{Int64}
end

mutable struct Ensemble 
    trajs::Vector{Trajectory}
    famtree::Graphs.SimpleGraphs.SimpleDiGraph{Int64}
end

function Ensemble()
    trajs = Vector{Trajectory}([])
    famtree = Graphs.SimpleDiGraph()
    return Ensemble(trajs, famtree)
end

function clear_Ensemble!(ens::Ensemble, mems2delete::Vector{Int64})
    Graphs.rem_vertices!(ens.famtree, mems2delete)
    deleteat!(ens.trajs, mems2delete)
    return
end

function get_Nmem(ens::Ensemble)
    return Graphs.nv(ens.famtree)
end

function add_trajectory!(ens::Ensemble, traj::Trajectory; parent=nothing)
    Graphs.add_vertex!(ens.famtree)
    push!(ens.trajs, traj)
    if !isnothing(parent)
        child = Graphs.nv(ens.famtree)
        Graphs.add_edge!(ens.famtree, parent, child)
    end
end

function save_Ensemble(ens::Ensemble, filename::String)
    JLD2.jldopen(filename, "w") do f
        f["ens"] = ens
    end
    return
end

function load_Ensemble(filename::String)
    ens = JLD2.jldopen(filename, "r") do f
        return f["ens"]
    end
    return ens
end

function load_ImmutableTrajectoryEnsemble(filename::String)
    ens = JLD2.jldopen(filename, "r"; typemap="Ensemble"=>ImmutableTrajectoryEnsemble) do f
        return f["ens"]
    end
    return ens
end

function compile_ancestry(ens::Ensemble, v::Int64)
    # chain together a whole historical trajectory line ending at the descendant (TODO add option to limit how far back it goes)
    lineage = Vector{Int64}([v])
    parents = Graphs.inneighbors(v)
    npar = length(parents)
    while npar > 0 
        @assert npar == 1
        v = parents[1]
        pushfirst!(lineage, v)
        parents = Graphs.inneighbors(v)
        npar = length(parents)
    end 
    return lineage
end



end # module Ensemble
