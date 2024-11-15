"""
    LogQuantArray{T,N}
Struct that holds the quantised array as UInts with an additional
field for the min, max of the original range.
"""
struct LogQuantArray{T,N} <: AbstractArray{Unsigned,N}
    A::Array{T,N}       # array of UInts
    min::Float64        # min of value range
    max::Float64        # max of value range
end

Base.size(QA::LogQuantArray) = size(QA.A)
Base.getindex(QA::LogQuantArray, i...) = getindex(QA.A, i...)
Base.eltype(Q::LogQuantArray{T,N}) where {T,N} = T

"""Return minimum, ignoring negatives and zeroes."""
function minpos(A::AbstractArray{T}) where {T}
    o = zero(T)
    mi = foldl((x, y) -> y > o ? min(x, y) : x, A; init=typemax(T))
    mi == typemax(T) && return o
    return mi
end

"""

    LogQuantization(::Type{T}, A::AbstractArray; extrema::Tuple = extrema(A)) where {T<:Integer}

Quantise an array logarithmically into a LogQuantArray.

# Arguments
- `T`: the type of the quantised array
- `A`: the array to quantise
- `extrema`: the minimum and maximum of the range, defaults to `extrema(A)`.

# Returns
- a LogQuantArray{T} with the quantised array and the minimum and maximum of the original range.
"""
function LogQuantization(
    ::Type{T},
    A::AbstractArray,
    round_nearest_in::Symbol=:linspace
) where {T<:Unsigned}

    (any(A .< zero(eltype(A))) || ~all(isfinite.(A))) &&
        throw(DomainError("Logarithmic  quantization only for positive&zero entries."))

    # min/max of non-zero entries
    mi = Float64(minpos(A))
    logmin = log(Float64(mi))
    logmax = log(Float64(maximum(A)))

    # throw error in case the range is zero.
    if logmin == logmax
        Δ⁻¹ = 0.0
        c = 0.0
    else
        # inverse log spacing
        # map min to 1 and max to ff..., reserve 0 for 0.
        Δ⁻¹ = (2^(sizeof(T) * 8) - 2) / (logmax - logmin)

        # shift to round-to-nearest in lin or log-space
        if round_nearest_in == :linspace
            c = 1 / 2 - Δ⁻¹ * log(mi * (exp(1 / Δ⁻¹) + 1) / 2)
        elseif round_nearest_in == :logspace
            c = -logmin * Δ⁻¹
        else
            throw(ArgumentError("Round-to-nearest either :linspace or :logspace"))
        end
    end

    # preallocate output
    Q = similar(A, T)

    @inbounds for i in eachindex(A)
        # store 0 as 0x00...
        # store positive numbers via convert to logpacking as 0x1-0xff..
        Q[i] = iszero(A[i]) ? zero(T) : convert(T, round(c + Δ⁻¹ * log(Float64(A[i])))) + one(T)
    end

    return LogQuantArray{T,ndims(Q)}(Q, Float64(logmin), Float64(logmax))
end


"""De-quantise a LogQuantArray into floats."""
function Base.Array{T}(n::Integer, Q::LogQuantArray) where {T<:AbstractFloat}
    Qlogmin = Q.min                 # log(min::Float64)
    Qlogmax = Q.max                 # log(max::Float64)

    # spacing in logspace ::Float64
    Δ = (Qlogmax - Qlogmin) / (2^n - 2)   # -2 as 0x00.. is reserved for 0

    A = similar(Q, T)                # preallocate

    @inbounds for i in eachindex(A)
        # 0x0 is unpack as 0
        # exp in Float64 then convert to T at assignment =
        A[i] = iszero(Q[i]) ? zero(T) : A[i] = exp(Qlogmin + (Q[i] - 1) * Δ)
    end

    return A
end

Base.Array{T}(Q::LogQuantArray{UInt8,N}) where {T,N} = Array{T}(8, Q)
Base.Array{T}(Q::LogQuantArray{UInt16,N}) where {T,N} = Array{T}(16, Q)
Base.Array{T}(Q::LogQuantArray{UInt24,N}) where {T,N} = Array{T}(24, Q)
Base.Array{T}(Q::LogQuantArray{UInt32,N}) where {T,N} = Array{T}(32, Q)

Base.Array(Q::LogQuantArray{UInt8,N}) where {N} = Array{Float32}(8, Q)
Base.Array(Q::LogQuantArray{UInt16,N}) where {N} = Array{Float32}(16, Q)
Base.Array(Q::LogQuantArray{UInt24,N}) where {N} = Array{Float32}(24, Q)
Base.Array(Q::LogQuantArray{UInt32,N}) where {N} = Array{Float64}(32, Q)


"""
    LogQuantArray(::Type{TUInt},A::AbstractArray{T,N};dims::Int) where {TUInt<:Unsigned,T,N}
Logarithmic quantization independently for every element along dimension
`dims` in array `A`.

# Returns
-  a Vector{LogQuantArray}.
"""
function LogQuantArray(::Type{TUInt}, A::AbstractArray{T,N}; dims::Int) where {TUInt<:Unsigned,T,N}
    @assert dims <= N "Can't quantize a $N-dimensional array in dimension=$dims"
    n = size(A)[dims]
    L = Vector{LogQuantArray}(undef, n)
    t = [
        if j == dims
            1
        else
            Colon()
        end for j in 1:N
    ]
    for i in 1:n
        t[dims] = i
        L[i] = LogQuantization(TUInt, A[t...])
    end
    return L
end

function LogQuantArray{U}(
    A::AbstractArray{T,N};
    dims::Option{Int}=nothing
) where {U<:Unsigned,T,N} 
    isnothing(dims) ? LogQuantization(U,A) : LogQuantArray(U,A;dims=dims)
end

# keep compatibility: shortcuts for unsigned integers of 8, 16, 24 and 32-bit
function LogQuant8Array(A::AbstractArray{T,N}, rn::Symbol=:linspace; dims::Option{Int}=nothing) where {T,N}
    isnothing(dims) ? LogQuantization(UInt8, A, rn) : LogQuantArray(UInt8, A; dims=dims)
end

function LogQuant16Array(A::AbstractArray{T,N}, rn::Symbol=:linspace; dims::Option{Int}=nothing) where {T,N}
    isnothing(dims) ? LogQuantization(UInt16, A, rn) : LogQuantArray(UInt16, A; dims=dims)
end

function LogQuant24Array(A::AbstractArray{T,N}, rn::Symbol=:linspace; dims::Option{Int}=nothing) where {T,N}
    isnothing(dims) ? LogQuantization(UInt24, A, rn) : LogQuantArray(UInt24, A; dims=dims)
end

function LogQuant32Array(A::AbstractArray{T,N}, rn::Symbol=:linspace; dims::Option{Int}=nothing) where {T,N} 
    isnothing(dims) ? LogQuantization(UInt32, A, rn) : LogQuantArray(UInt32, A; dims=dims)
end


"""Undo the logarithmic quantisation independently along one dimension, and returns
an array whereby the dimension always comes last. Hence, might be permuted compared
to the uncompressed array."""
function Base.Array{T}(L::Vector{LogQuantArray}) where {T}
    N = ndims(L[1])
    n = length(L)
    s = size(L[1])
    t = axes(L[1])
    A = Array{T,N + 1}(undef, s..., length(L))
    for i in 1:n
        A[t..., i] = Array{T}(L[i])
    end
    return A
end