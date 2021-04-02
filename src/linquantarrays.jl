"""Struct that holds the quantised array as UInts with an additional
field for the min, max of the original range."""
struct LinQuantArray{T,N} <: AbstractArray{Unsigned,N}
    A::Array{T,N}       # array of UInts
    min::Float64        # offset min, max
    max::Float64
end

Base.size(QA::LinQuantArray) = size(QA.A)
Base.getindex(QA::LinQuantArray,i...) = getindex(QA.A,i...)
Base.eltype(Q::LinQuantArray{T,N}) where {T,N} = T

"""Quantise an array linearly into a LinQuantArray."""
function LinQuantization(::Type{T},A::AbstractArray) where {T<:Unsigned}
    all(isfinite.(A)) || throw(DomainError("Linear quantization only in (-∞,∞)"))

    Amin = Float64(minimum(A))              # minimum of value range
    Amax = Float64(maximum(A))              # maximum of value range

    if Amin == Amax
        Δ = 0.0                                 # set to zero for no range
    else
        Δ = (2^(sizeof(T)*8)-1)/(Amax-Amin)     # inverse spacing
    end

    Q = similar(A,T)                        # preallocate

    # map minimum to 0x0, maximum to 0xff...ff
    @inbounds for i in eachindex(Q)
        Q[i] = round((A[i]-Amin)*Δ)
    end

    return LinQuantArray{T,ndims(Q)}(Q,Amin,Amax)
end

# define 8, 16, 24 and 32 bit
LinQuant8Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt8,A)
LinQuant16Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt16,A)
LinQuant24Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt24,A)
LinQuant32Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt32,A)

"""De-quantise a LinQuantArray into floats."""
function Base.Array{T}(n::Integer,Q::LinQuantArray) where {T<:AbstractFloat}
    Qmin = Q.min                # min as Float64
    Qmax = Q.max                # max as Float64
    Δ = (Qmax-Qmin)/(2^n-1)     # linear spacing

    A = similar(Q,T)

    @inbounds for i in eachindex(A)
        # convert Q[i]::UInt to Float64 via *
        # then to T through =
        A[i] = Qmin + Q[i]*Δ
    end

    return A
end

# define default conversions for 8, 16, 24 and 32 bit
Base.Array{T}(Q::LinQuantArray{UInt8,N}) where {T,N} = Array{T}(8,Q)
Base.Array{T}(Q::LinQuantArray{UInt16,N}) where {T,N} = Array{T}(16,Q)
Base.Array{T}(Q::LinQuantArray{UInt24,N}) where {T,N} = Array{T}(24,Q)
Base.Array{T}(Q::LinQuantArray{UInt32,N}) where {T,N} = Array{T}(32,Q)

Base.Array(Q::LinQuantArray{UInt8,N}) where N = Array{Float32}(8,Q)
Base.Array(Q::LinQuantArray{UInt16,N}) where N = Array{Float32}(16,Q)
Base.Array(Q::LinQuantArray{UInt24,N}) where N = Array{Float32}(24,Q)
Base.Array(Q::LinQuantArray{UInt32,N}) where N = Array{Float64}(32,Q)

# one quantization per layer
"""Linear quantization independently for every element along dimension
dim in array A. Returns a Vector{LinQuantArray}."""
function LinQuantArray(::Type{TUInt},A::AbstractArray{T,N},dim::Int) where {TUInt,T,N}
    @assert dim <= N   "Can't quantize a $N-dimensional array in dim=$dim"
    n = size(A)[dim]
    L = Vector{LinQuantArray}(undef,n)
    t = [if j == dim 1 else Colon() end for j in 1:N]
    for i in 1:n
        t[dim] = i
        L[i] = LinQuantization(TUInt,A[t...])    
    end
    return L
end

# for 8,16,24 and 32 bit
LinQuant8Array(A::AbstractArray{T,N},dim::Int) where {T,N} = LinQuantArray(UInt8,A,dim)
LinQuant16Array(A::AbstractArray{T,N},dim::Int) where {T,N} = LinQuantArray(UInt16,A,dim)
LinQuant24Array(A::AbstractArray{T,N},dim::Int) where {T,N} = LinQuantArray(UInt24,A,dim)
LinQuant32Array(A::AbstractArray{T,N},dim::Int) where {T,N} = LinQuantArray(UInt32,A,dim)

"""Undo the linear quantisation independently along one dimension, and returns
an array whereby the dimension always comes last. Hence, might be permuted compared
to the uncompressed array."""
function Base.Array{T}(L::Vector{LinQuantArray}) where T
    N = ndims(L[1])
    n = length(L)
    s = size(L[1])
    t = axes(L[1])
    A = Array{T,N+1}(undef,s...,length(L))
    for i in 1:n
        A[t...,i] = Array{T}(L[i])
    end
    return A
end

