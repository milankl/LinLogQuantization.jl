
const Option{T} = Union{T,Nothing}

"""Struct that holds the quantised array as UInts with an additional
field for the min, max of the original range."""
struct LinQuantArray{T,N} <: AbstractArray{Integer,N}
    A::Array{T,N}       # array of UInts
    min::Float64        # offset min, max
    max::Float64
end

Base.size(QA::LinQuantArray) = size(QA.A)
Base.getindex(QA::LinQuantArray,i...) = getindex(QA.A,i...)
Base.eltype(Q::LinQuantArray{T,N}) where {T,N} = T

"""

        LinQuantization(::Type{T}, A::AbstractArray; extrema::Tuple = extrema(A)) where {T<:Integer}

Quantise an array linearly into a LinQuantArray.

# Arguments
- `T`: the type of the quantised array
- `A`: the array to quantise
- `extrema`: the minimum and maximum of the range, defaults to `extrema(A)`.

# Returns
- a LinQuantArray{T} with the quantised array and the minimum and maximum of the original range.
"""

function LinQuantization(
    ::Type{T},
    A::AbstractArray,
    extrema::Nothing
) where {T<:Integer}
    all(isfinite.(A)) || throw(DomainError("Linear quantization only in (-∞,∞)"))

    # range of values in A
    Amin, Amax  = Float64(minimum(A)), Float64(maximum(A))    

    # minimum and maximum representable value of type T
    Tmin, Tmax = Float64(typemin(T)), Float64(typemax(T))

    # inverse spacing, set to zero for no range
    Δ⁻¹ = Amin == Amax ? zero(Float64) : (Tmax-Tmin)/(Amax-Amin)

    Q = similar(A,T)                        # preallocate

    # map minimum to typemin(T), maximum to typemax(t)
    @inbounds for i in eachindex(Q)
        Q[i] = round((A[i]-Amin)*Δ⁻¹ + Tmin)
    end

    return LinQuantArray{T,ndims(Q)}(Q,Amin,Amax)
end

function LinQuantization(
    ::Type{T},
    A::AbstractArray,
    extrema::Tuple = extrema(A),
) where {T<:Integer}
    all(isfinite.(A)) || throw(DomainError("Linear quantization only in (-∞,∞)"))

    # minimum-maximum range of values
    Amin, Amax = Float64.(extrema)    
    
    # minimum and maximum representable value of type T
    Tmin, Tmax = Float64(typemin(T)), Float64(typemax(T))

    # inverse spacing, set to zero for no range
    Δ⁻¹ = Amin == Amax ? zero(Float64) : (Tmax-Tmin)/(Amax-Amin)
    
    # preallocate
    Q = similar(A, T)                        

    # map minimum to typemin(T), maximum to typemax(t)
    # clamp to [Amin,Amax] removing out-of-range values
    @inbounds for i in eachindex(Q)
        Q[i] = round(T, clamp((A[i]-Amin)*Δ⁻¹ + Tmin, Tmin, Tmax))
    end

    return LinQuantArray{T,ndims(Q)}(Q,Amin,Amax)
end

# define for unsigned integers of  8, 16, 24 and 32 bit 
LinQuant8Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt8,A)
LinQuant16Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt16,A)
LinQuant24Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt24,A)
LinQuant32Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt32,A)

# define for unsigned integers of  8, 16, 24 and 32 bit 
LinQuantUInt8Array(A::AbstractArray{T,N}, e::Option{Tuple}) where {T,N} = LinQuantization(UInt8,A,e)
LinQuantUInt16Array(A::AbstractArray{T,N}, e::Option{Tuple}) where {T,N} = LinQuantization(UInt16,A,e)
LinQuantUInt24Array(A::AbstractArray{T,N}, e::Option{Tuple}) where {T,N} = LinQuantization(UInt24,A,e)
LinQuantUInt32Array(A::AbstractArray{T,N}, e::Option{Tuple}) where {T,N} = LinQuantization(UInt32,A,e)

# define for signed integers of  8, 16, 24 and 32 bit
LinQuantInt8Array(A::AbstractArray{T,N}, e::Option{Tuple}) where {T,N} = LinQuantization(Int8,A,e)
LinQuantInt16Array(A::AbstractArray{T,N}, e::Option{Tuple}) where {T,N} = LinQuantization(Int16,A,e)
LinQuantInt24Array(A::AbstractArray{T,N}, e::Option{Tuple}) where {T,N} = LinQuantization(Int24,A,e)
LinQuantInt32Array(A::AbstractArray{T,N}, e::Option{Tuple}) where {T,N} = LinQuantization(Int32,A,e)


"""De-quantise a LinQuantArray into floats."""
function Base.Array{U}(n::Integer, Q::LinQuantArray) where {U<:AbstractFloat}
    Qmin = Q.min                     # min of original Array as Float64
    Qmax = Q.max                     # max of original Array as Float64
    Tmin = Float64(typemin(Q.A[1]))  # min representable in type as Float64
    Tmax = Float64(typemax(Q.A[1]))  # max representable in type as Float64
    Δ = (Qmax-Qmin)/(Tmax-Tmin)          # linear spacing

    A = similar(Q,U)

    @inbounds for i in eachindex(A)
        # convert Q[i]::Integer to Float64 via *
        # then to T through =
        A[i] = Qmin + (Q[i] - Tmin)*Δ
    end

    return A
end

# define default conversions for unsigned 8, 16, 24 and 32 bit
Base.Array{T}(Q::LinQuantArray{UInt8,N}) where {T,N} = Array{T}(8,Q)
Base.Array{T}(Q::LinQuantArray{UInt16,N}) where {T,N} = Array{T}(16,Q)
Base.Array{T}(Q::LinQuantArray{UInt24,N}) where {T,N} = Array{T}(24,Q)
Base.Array{T}(Q::LinQuantArray{UInt32,N}) where {T,N} = Array{T}(32,Q)

# define default conversions for signed 8, 16, 24 and 32 bit
Base.Array(Q::LinQuantArray{Int8,N}) where N = Array{Float32}(8,Q)
Base.Array(Q::LinQuantArray{Int16,N}) where N = Array{Float32}(16,Q)
Base.Array(Q::LinQuantArray{Int24,N}) where N = Array{Float32}(24,Q)
Base.Array(Q::LinQuantArray{Int32,N}) where N = Array{Float64}(32,Q)

# one quantization per layer
"""Linear quantization independently for every element along dimension
dim in array A. Returns a Vector{LinQuantArray}."""
function LinQuantArray(
    ::Type{TInteger},
    A::AbstractArray{T,N},
    dim::Int,
    extrema::Option{Tuple} = nothing
) where {TInteger,T,N}
    @assert dim <= N   "Can't quantize a $N-dimensional array in dim=$dim"
    n = size(A)[dim]
    L = Vector{LinQuantArray}(undef,n)
    t = [if j == dim 1 else Colon() end for j in 1:N]
    for i in 1:n
        t[dim] = i
        L[i] = LinQuantization(TInteger,A[t...], extrema)    
    end
    return L
end

# for unsigned integers  8,16,24 and 32 bit
LinQuant8Array(A::AbstractArray{T,N},dim::Int) where {T,N} = LinQuantArray(UInt8,A,dim)
LinQuant16Array(A::AbstractArray{T,N},dim::Int) where {T,N} = LinQuantArray(UInt16,A,dim)
LinQuant24Array(A::AbstractArray{T,N},dim::Int) where {T,N} = LinQuantArray(UInt24,A,dim)
LinQuant32Array(A::AbstractArray{T,N},dim::Int) where {T,N} = LinQuantArray(UInt32,A,dim)

# for unsigned integers  8,16,24 and 32 bit
LinQuantUInt8Array(A::AbstractArray{T,N},dim::Int,e::Option{Tuple}) where {T,N} = LinQuantArray(UInt8,A,dim,e)
LinQuantUInt16Array(A::AbstractArray{T,N},dim::Int,e::Option{Tuple}) where {T,N} = LinQuantArray(UInt16,A,dim,e)
LinQuantUInt24Array(A::AbstractArray{T,N},dim::Int,e::Option{Tuple}) where {T,N} = LinQuantArray(UInt24,A,dim,e)
LinQuantUInt32Array(A::AbstractArray{T,N},dim::Int,e::Option{Tuple}) where {T,N} = LinQuantArray(UInt32,A,dim,e)

# for signed integers of 8,16,24 and 32 bit
LinQuantInt8Array(A::AbstractArray{T,N},dim::Int,e::Option{Tuple}) where {T,N} = LinQuantArray(Int8,A,dim,e)
LinQuantInt16Array(A::AbstractArray{T,N},dim::Int,e::Option{Tuple}) where {T,N} = LinQuantArray(Int16,A,dim,e)
LinQuantInt24Array(A::AbstractArray{T,N},dim::Int,e::Option{Tuple}) where {T,N} = LinQuantArray(Int24,A,dim,e)
LinQuantInt32Array(A::AbstractArray{T,N},dim::Int,e::Option{Tuple}) where {T,N} = LinQuantArray(Int32,A,dim,e)


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

