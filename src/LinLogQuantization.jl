module LinLogQuantization

    const Option{T} = Union{T,Nothing}

    export  LinQuantArray, LogQuantArray,
    LinQuant8Array, LinQuant16Array, LinQuant24Array, LinQuant32Array,
    LogQuant8Array, LogQuant16Array, LogQuant24Array, LogQuant32Array,
    minpos

    # enable UInt24 and Int24 support
    import BitIntegers
    BitIntegers.@define_integers 24

    export UInt24, Int24

    include("linquantarrays.jl")
    include("logquantarrays.jl")

end
