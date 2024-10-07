module LinLogQuantization

    export  LinQuantArray, LogQuantArray,
    LinQuant8Array, LinQuant16Array, LinQuant24Array, LinQuant32Array,
    LogQuant8Array, LogQuant16Array, LogQuant24Array, LogQuant32Array,
    minpos

    # enable UInt24 support
    import BitIntegers
    BitIntegers.@define_integers 24

    export UInt24

    include("linquantarrays.jl")
    include("logquantarrays.jl")

end
