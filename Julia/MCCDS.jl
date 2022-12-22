using MKL  # MKL needs to be imported as first package
using LinearAlgebra
# using LoopVectorization
using BenchmarkTools
using Random
using HDF5
# using Distributed
# addprocs(6)
# @everywhere using SharedArrays

println("Threads: ",Threads.nthreads())
# println("Workers: ",nworkers())

function BrownianMotion(T::S,N::Int,M::Int) where {S <: AbstractFloat}
    dt = T / (N-1)
    W = zeros(S,M,N)
    W[:,2:end] = sqrt(dt).*randn(S,M,N-1)
    return cumsum(W,dims=2)
end

function mainSerial(Wt::AbstractArray{T,2},Mt::AbstractArray{T,2},
                    t::AbstractArray{T,1},batch::Int,LGD::T,annuity::T,
                    rRange::Tuple{T,T},sRange::Tuple{T,T},rhoRange::Tuple{T,T},x0Range::Tuple{T,T},
                    rng::AbstractRNG) where {T <:AbstractFloat}
    N,M = size(Wt)
    r = Array{T}(undef,batch)
    sigma = Array{T}(undef,batch)
    rho = Array{T}(undef,batch)
    x0 = Array{T}(undef,batch)
    payLeg = zeros(T,batch)
    protLeg = zeros(T,batch)
    @inbounds for iB = 1:batch
        r[iB] = rRange[1]+rand(rng,T,1)[1]*(rRange[2]-rRange[1])
        sigma[iB] = sRange[1]+rand(rng,T,1)[1]*(sRange[2]-sRange[1])
        rho[iB] = rhoRange[1]+rand(rng,T,1)[1]*(rhoRange[2]-rhoRange[1])
        x0[iB] = x0Range[1]+rand(rng,T,1)[1]*(x0Range[2]-x0Range[1])
        beta = (r[iB] -(sigma[iB] ^2)/T(2))/sigma[iB]  
        bt = exp.(-r[iB].*t)
        @inbounds for iM = 1:M
            df = T(0.0)
            nom = T(0.0)
            @inbounds for iT=2:N
                Xt =  x0[iB] + beta * t[iT] + sqrt(1 - rho[iB]) * Wt[iT,iM] + sqrt(rho[iB]) * Mt[iT,iM]
                if Xt <= T(0.0) && iT<N
                    nom=bt[iT+1]
                    break 
                end
                df +=  bt[iT]
            end
            denom= df
            protLeg[iB]+= nom
            payLeg[iB]+= denom
        end
    end
    protLeg .*= (LGD/M)
    payLeg .*= (annuity/M)
    return hcat(r,sigma,rho,x0,protLeg./payLeg,protLeg,payLeg)
end


const dtype = Float32

const T = dtype(5.0)

const total = 2^16
const batch = 2^10  # =1024
const samples = Int64(total/batch)
const annuity = dtype(.25)
const N = Int64(T/annuity) + 1
const M = 10^5
const LGD = dtype(0.6)

const rRange = (dtype(.1),dtype(.2))
const sRange = (dtype(.01),dtype(.99))
const rhoRange = (dtype(.01),dtype(.99))
const x0Range = (dtype(0.0),dtype(6.0))

const t = collect(dtype,LinRange(0.0,T,N))

@show batch
@show typeof(rRange)

Wt = copy(BrownianMotion(T,N,M)')  # copy is important otherwise wrong memory layout
Mt = copy(BrownianMotion(T,N,M)')

# rngs = [MersenneTwister(i) for i in 1: Threads.nthreads()];
# @btime Threads.@threads for _ = 1:2^16
#     mainSerial(Wt,Mt,t,1,LGD,annuity,rRange,sRange,rhoRange,x0Range,rngs[Threads.threadid()])
# end  # 39.18 s

# @btime Threads.@threads for _ = 1:2^6
#     mainSerial(Wt,Mt,t,2^10,LGD,annuity,rRange,sRange,rhoRange,x0Range,rngs[Threads.threadid()])
# end  # 41.4 s

# @btime Threads.@threads for _ = 1:2^10
#     mainSerial(Wt,Mt,t,2^6,LGD,annuity,rRange,sRange,rhoRange,x0Range,rngs[Threads.threadid()])
# end # 40.02

out = collect((zeros(dtype,0,7) for _ = 1:Threads.nthreads()))  # each thread gets its own output list to avoid locking
rngs = [MersenneTwister(i) for i in 1: Threads.nthreads()];
# rngs = [Xoshiro(i) for i in 1: Threads.nthreads()];
@time begin
    Threads.@threads for i = 1:samples
        out[Threads.threadid()]=vcat(out[Threads.threadid()],mainSerial(Wt,Mt,t,batch,LGD,annuity,rRange,sRange,rhoRange,x0Range,rngs[Threads.threadid()]))
    end
    data = reduce(vcat,out)    
end  # 42 s

