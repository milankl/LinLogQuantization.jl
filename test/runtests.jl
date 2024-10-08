using LinLogQuantization
using Test


@testset "minpos" begin
    @test minpos([0,0,0,0]) == 0
    @test minpos([0,0,1,0]) == 1
    
    A = randn(100)
    @test minpos(A) > 0
    @test minpos(A) == minimum(A[A.>0])

    A = rand(100) .+ 1
    @test minpos(A) == minimum(A)
end

@testset "Linear quantization" begin

    @testset "Default extrema" begin 
        for T in [Float64, Float32, Float16]
            for s in [(100,),
                        (10,20),
                        (13,14,15),
                        (23,17,12,5)]
            
                A = rand(T,s...)

                for U in [
                            UInt8,
                            UInt16,
                            UInt24,
                            UInt32,
                            Int8,
                            Int16,
                            Int24,
                            Int32
                        ]  
                    
                    # initial conversion is not reversible
                    # due to rounding errors
                    A2 = Array{T}(LinQuantArray{U}(A))

                    # then test whether back&forth conversion is reversible
                    @test A2 == Array{T}(LinQuantArray{U}(A2))
                end
            end
        end
    end

    @testset "Custom extrema" begin 
        for ext in [(0.1, 0.6), (0.22, 0.75), (0.3, 0.5)]
            for T in [Float64, Float32, Float16]
                for s in [(100,),
                            (10,20),
                            (13,14,15),
                            (23,17,12,5)]
                
                    A = rand(T,s...)

                    for U in [
                                UInt8,
                                UInt16,
                                UInt24,
                                UInt32,
                                Int8,
                                Int16,
                                Int24,
                                Int32
                            ]  
                        
                        # initial conversion is not reversible
                        # due to rounding errors
                        A2 = Array{T}(LinQuantArray{U}(A; extrema=ext))

                        # then test whether back&forth conversion is reversible
                        @test A2 == Array{T}(LinQuantArray{U}(A2; extrema=ext))
                        @test Base.extrema(A2) == ext
                    end
                end
            end
        end
    end
end

@testset "Log quantization" begin
    for T in [Float64,Float32,Float16]
        for s in [(100,),
                    (10,20),
                    (13,14,15),
                    (23,17,12,5)]
        
            A = rand(T,s...)

            for LogQ in [LogQuant8Array,
                        LogQuant16Array,
                        LogQuant24Array,
                        LogQuant32Array]
                
                for rn in [:linspace,:logspace]
                
                    # initial conversion is not reversible
                    # due to rounding errors
                    A2 = Array{T}(LogQ(A,rn))

                    # then test whether back&forth conversion is reversible
                    @test A2 == Array{T}(LogQ(A2,rn))
                end
            end
        end
    end
end

@testset "LinQuant along dimension" begin
    @testset "Default extrema" begin
        A = rand(Float32,10,20,30,40)
        Q = LinQuant32Array(A,4)
        @test A ≈ Array{Float32}(Q)

        Q = LinQuant24Array(A,4)
        @test A ≈ Array{Float32}(Q)

        Q = LinQuant16Array(A,4)
        @test A ≈ Array{Float32}(Q)

        Q = LinQuant8Array(A,4)
        @test all(isapprox.(A,Array{Float32}(Q),atol=1e-1))

        A = rand(Float32,10,20,30,40)
        Q = LinQuantArray{Int32}(A,4)
        @test A ≈ Array{Float32}(Q)

        Q = LinQuantArray{Int24}(A,4)
        @test A ≈ Array{Float32}(Q)

        Q = LinQuantArray{Int16}(A,4)
        @test A ≈ Array{Float32}(Q)

        Q = LinQuantArray{Int8}(A,4)
        @test all(isapprox.(A,Array{Float32}(Q),atol=1e-1))
    end 

    @testset "Custom extrema" begin


    end
end

@testset "LogQuant along dimension" begin
    A = rand(Float32,10,20,30,40)
    Q = LogQuant32Array(A,4)
    @test A ≈ Array{Float32}(Q)

    Q = LogQuant24Array(A,4)
    @test A ≈ Array{Float32}(Q)

    Q = LogQuant16Array(A,4)
    @test A ≈ Array{Float32}(Q)

    Q = LogQuant8Array(A,4)
    @test all(isapprox.(A,Array{Float32}(Q),atol=1e-1))
end

@testset "Linear quantization will all 0" begin
    for T in [Float64,Float32,Float16]
        for s in [(100,),
                    (10,20),
                    (3,4,5),
                    (6,7,8,9)]
        
            A = zeros(T,s...)

            for U in [
                        UInt8,
                        UInt16,
                        UInt24,
                        UInt32,
                        Int8,
                        Int16,
                        Int24,
                        Int32
                    ]  
                        
                # initial conversion is not reversible
                # due to rounding errors
                A2 = Array{T}(LinQuantArray{U}(A))

                # then test whether back&forth conversion is reversible
                @test A2 == Array{T}(LinQuantArray{U}(A2))
            end
        end
    end
end

@testset "Log quantization will all 0" begin
    for T in [Float64,Float32,Float16]
        for s in [(100,),
                    (10,20),
                    (3,4,5),
                    (6,7,8,9)]
        
            A = zeros(T,s...)

            for LogQ in [LogQuant8Array,
                LogQuant16Array,
                LogQuant24Array,
                LogQuant32Array]
                
                # initial conversion is not reversible
                # due to rounding errors
                A2 = Array{T}(LogQ(A))

                # then test whether back&forth conversion is reversible
                @test A2 == Array{T}(LogQ(A2))
            end
        end
    end
end

@testset "Linear quantization will all c" begin
    for T in [Float64,Float32,Float16]
        for s in [(100,),
                    (10,20),
                    (3,4,5),
                    (6,7,8,9)]
        
            A = zeros(T,s...) .+ T(randn())

            for U in [
                        UInt8,
                        UInt16,
                        UInt24,
                        UInt32,
                        Int8,
                        Int16,
                        Int24,
                        Int32
                    ]  
                
                # initial conversion is not reversible
                # due to rounding errors
                A2 = Array{T}(LinQuantArray{U}(A))

                # then test whether back&forth conversion is reversible
                @test A2 == Array{T}(LinQuantArray{U}(A2))
            end
        end
    end
end

@testset "Log quantization will all c" begin
    for T in [Float64,Float32,Float16]
        for s in [(100,),
                    (10,20),
                    (3,4,5),
                    (6,7,8,9)]
        
            A = zeros(T,s...) .+ T(rand())

            for LogQ in [LogQuant8Array,
                LogQuant16Array,
                LogQuant24Array,
                LogQuant32Array]
                
                # initial conversion is not reversible
                # due to rounding errors
                A2 = Array{T}(LogQ(A))

                # then test whether back&forth conversion is reversible
                @test A2 == Array{T}(LogQ(A2))
            end
        end
    end
end