using LinLogQuantization
using Test

@testset "Linear quantization" begin
    for T in [Float64,Float32,Float16]
        for s in [(100,),
                    (10,20),
                    (13,14,15),
                    (23,17,12,5)]
        
            A = rand(T,s...)

            for LinQ in [LinQuant8Array,
                        LinQuant16Array,
                        LinQuant24Array,
                        LinQuant32Array]
                
                # initial conversion is not reversible
                # due to rounding errors
                A2 = Array{T}(LinQ(A))

                # then test whether back&forth conversion is reversible
                @test A2 == Array{T}(LinQ(A2))
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