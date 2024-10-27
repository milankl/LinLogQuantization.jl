const Option{T} = Union{T,Nothing}

"""
    LinQuantArray{T,N}
Struct that holds the quantised array as UInts with an additional
field for the min, max of the original range.
"""
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
    A::AbstractArray;
    extrema::Option{Tuple}=nothing
) where {T<:Integer}
    # range of values in A
    Amin, Amax  = isnothing(extrema) ? Float64.(Base.extrema(A)) : Float64.(extrema)

    # guard against infinite values
    (isfinite(Amin) && isfinite(Amax)) || throw(DomainError("Linear quantization only in (-∞,∞)"))

    # minimum and maximum representable value of type T
    Tmin, Tmax = Float64(typemin(T)), Float64(typemax(T))

    # inverse spacing, set to zero for no range
    Δ⁻¹ = Amin == Amax ? zero(Float64) : (Tmax-Tmin)/(Amax-Amin)

    Q = similar(A,T)                        # preallocate

    if isnothing(extrema) # range defaults to extrema(A)
        # map minimum to typemin(T), maximum to typemax(t)
        @inbounds for i in eachindex(Q)
            Q[i] = round((A[i]-Amin)*Δ⁻¹ + Tmin)
        end
    else 
        # map minimum to typemin(T), maximum to typemax(t)
        # clamp to [Tmin,Tmax] removing out-of-range values
        @inbounds for i in eachindex(Q)
            Q[i] = round(T, clamp((A[i]-Amin)*Δ⁻¹ + Tmin, Tmin, Tmax))
        end
    end

    return LinQuantArray{T,ndims(Q)}(Q,Amin,Amax)
end

function LinQuantArray{U}(A::AbstractArray{T,N}; extrema::Option{Tuple}=nothing) where {U<:Integer,T,N}
    LinQuantization(U,A; extrema=extrema)
end

# keep compatibility: shortcuts for unsigned integers of 8, 16, 24 and 32 bit
LinQuant8Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt8,A)
LinQuant16Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt16,A)
LinQuant24Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt24,A)
LinQuant32Array(A::AbstractArray{T,N}) where {T,N} = LinQuantization(UInt32,A)


"""
    Array{U}(Q::LinQuantArray) where {U<:AbstractFloat}

De-quantise a LinQuantArray into floats.

# Arguments
- `U`: the type of the de-quantised array
- `Q`: the LinQuantArray to de-quantise

# Returns
- an array of type U with the de-quantised values.
"""
function Base.Array{U}(Q::LinQuantArray) where {U<:AbstractFloat}
    Qmin = Q.min                     # min of original Array as Float64
    Qmax = Q.max                     # max of original Array as Float64
    T = eltype(Q)
    Tmin = Float64(typemin(T))       # min representable in type as Float64
    Tmax = Float64(typemax(T))       # max representable in type as Float64
    Δ = (Qmax-Qmin)/(Tmax-Tmin)      # linear spacing

    A = similar(Q,U)

    @inbounds for i in eachindex(A)
        # convert Q[i]::Integer to Float64 via *
        # then to T through =
        A[i] = Qmin + (Q[i] - Tmin)*Δ
    end

    return A
end

# default conversions for unsigned 8, 16, 24 and 32 bit
Base.Array(Q::LinQuantArray{UInt8,N}) where {N} = Array{Float32}(Q)
Base.Array(Q::LinQuantArray{UInt16,N}) where {N} = Array{Float32}(Q)
Base.Array(Q::LinQuantArray{UInt24,N}) where {N} = Array{Float32}(Q)
Base.Array(Q::LinQuantArray{UInt32,N}) where {N} = Array{Float64}(Q)

# default conversions for signed 8, 16, 24 and 32 bit
Base.Array(Q::LinQuantArray{Int8,N}) where N = Array{Float32}(Q)
Base.Array(Q::LinQuantArray{Int16,N}) where N = Array{Float32}(Q)
Base.Array(Q::LinQuantArray{Int24,N}) where N = Array{Float32}(Q)
Base.Array(Q::LinQuantArray{Int32,N}) where N = Array{Float64}(Q)


"""
    LinQuantArray(TInteger, A, dim; extrema=nothing)

Linear quantization independently for every element along dimension
dim in array A.

# Arguments
- `TInteger`: the type of the quantised array
- `A`: the array to quantise
- `dim`: the dimension along which to quantise
- `extrema`: the minimum and maximum of the range, defaults to `nothing`.

# Returns
- a Vector{LinQuantArray} with the quantised array and the minimum and maximum of the original range.
"""
function LinQuantArray(
    ::Type{TInteger},
    A::AbstractArray{T,N},
    dim::Int;
    extrema::Option{Tuple} = nothing
) where {TInteger,T,N}
    @assert dim <= N   "Can't quantize a $N-dimensional array in dim=$dim"
    n = size(A)[dim]
    L = Vector{LinQuantArray}(undef,n)
    t = [if j == dim 1 else Colon() end for j in 1:N]
    for i in 1:n
        t[dim] = i
        L[i] = LinQuantization(TInteger,A[t...]; extrema=extrema)    
    end
    return L
end

function LinQuantArray{U}(
    A::AbstractArray{T,N},
    dim::Int; 
    extrema::Option{Tuple}=nothing
) where {U<:Integer,T,N} 
    LinQuantArray(U,A,dim;extrema=extrema)
end

# keep compatibility: shortcuts for unsigned integers of 8, 16, 24 and 32 bit
LinQuant8Array(A::AbstractArray{T,N},dim::Int) where {T,N} = LinQuantArray(UInt8,A,dim)
LinQuant16Array(A::AbstractArray{T,N},dim::Int) where {T,N} = LinQuantArray(UInt16,A,dim)
LinQuant24Array(A::AbstractArray{T,N},dim::Int) where {T,N} = LinQuantArray(UInt24,A,dim)
LinQuant32Array(A::AbstractArray{T,N},dim::Int) where {T,N} = LinQuantArray(UInt32,A,dim)



"""
    Array{U}(L::Vector{LinQuantArray}) where {U<:AbstractFloat}

Undo the linear quantisation independently along one dimension, and returns
an array whereby the dimension always comes last. Hence, might be permuted compared
to the uncompressed array.
"""
function Base.Array{U}(L::Vector{LinQuantArray}) where {U<:AbstractFloat}
    N = ndims(L[1])
    n = length(L)
    s = size(L[1])
    t = axes(L[1])
    A = Array{U,N+1}(undef,s...,length(L))
    for i in 1:n
        A[t...,i] = Array{U}(L[i])
    end
    return A
end
