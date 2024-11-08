# To run the benchmarks, you need to install BenchmarkTools and Printf  
using LinLogQuantization
using BenchmarkTools
using Printf

"""
Calculate rounded throughput in MB/s
"""
function calculate_throughput(size_mb::Float64, time_ms::Float64)
    # Round time to nearest multiple of 5ms
    rounded_time = round(time_ms/5) * 5
    # Calculate throughput and round to nearest 100 MB/s
    throughput = size_mb/(rounded_time/1000)
    return round(throughput/100) * 100
end

"""
Benchmark linear quantization for a given array size
"""
function benchmark_linear_quantization(n::Int)
    A = rand(Float64, n)
    size_mb = sizeof(A)/1000^2
    
    println("\nBenchmarking Linear Quantization (Array size: $(size_mb) MB)...")
    
    # Store results
    results = Dict{String, Dict{String, Dict{String, Int}}}()
    
    # Test integers ordered by bit size
    for with_extrema in ["default", "custom"]
        results[with_extrema] = Dict{String, Dict{String, Int}}()

        for I in [UInt8, Int8, UInt16, Int16, UInt24, Int24, UInt32, Int32]
            type_name = string(I)
            results[with_extrema][type_name] = Dict{String, Int}()

            # Compression
            if with_extrema == "custom"
                ext = (0.2, 0.8)  # Example extrema values
                b_comp = @benchmark LinQuantArray{$I}($A; extrema=$ext)
                compressed = LinQuantArray{I}(A; extrema=ext)
            else
                b_comp = @benchmark LinQuantArray{$I}($A)
                compressed = LinQuantArray{I}(A)
            end
            
            time_comp = mean(b_comp.times) / 1e6  # Convert ns to ms and compute mean
            results[with_extrema][type_name]["compression"] = Int(calculate_throughput(size_mb, time_comp))
            
            # Decompression
            b_decomp = @benchmark Array{Float64}($compressed)
            time_decomp = mean(b_decomp.times) / 1e6  # Convert ns to ms and compute mean

            results[with_extrema][type_name]["decompression"] = Int(calculate_throughput(size_mb, time_decomp))
        end
    end

    results
end

"""
Benchmark log quantization for a given array size
"""
function benchmark_log_quantization(n::Int)
    A = rand(Float64, n)
    size_mb = sizeof(A)/1000^2
    
    println("\nBenchmarking Log Quantization (Array size: $(size_mb) MB)...")
    
    # Store results
    results = Dict{String, Dict{String, Dict{String, Int}}}()
    
    for method in [:linspace, :logspace]
        results[string(method)] = Dict{String, Dict{String, Int}}()
        
        for Q in [LogQuant8Array, LogQuant16Array, LogQuant24Array, LogQuant32Array]
            type_name = string(Q)
            results[string(method)][type_name] = Dict{String, Int}()
            
            # Compression
            b_comp = @benchmark $Q($A, $method)
            time_comp = mean(b_comp.times) / 1e6  # Convert ns to ms and compute mean
            speed_comp = calculate_throughput(size_mb, time_comp)
            results[string(method)][type_name]["compression"] = Int(speed_comp)
            
            # Create a compressed array for decompression benchmark
            compressed = Q(A, method)
            
            # Decompression
            b_decomp = @benchmark Array{Float64}($compressed)
            time_decomp = mean(b_decomp.times) / 1e6  # Convert ns to ms and compute mean
            speed_decomp = calculate_throughput(size_mb, time_decomp)
            results[string(method)][type_name]["decompression"] = Int(speed_decomp)
        end
    end
    
    results
end

"""
Print markdown table for linear quantization results
""" 
function print_lin_results(results)
    # Print markdown table with reordered types
    println("\n| Method           | UInt8    | Int8     | UInt16    | Int16     | UInt24   | Int24    | UInt32   | Int32    |")
    println("| ---------------- | -------: | -------: | --------: | --------: | -------: | -------: | -------: | -------: |")
    
    for with_extrema in ["default", "custom"]
        println("| $(with_extrema == "default" ? "**Default extrema**" : "**Custom extrema**") |")
        # Print compression results
        print("| compression      |")
        for type in ["UInt8", "Int8", "UInt16", "Int16", "UInt24", "Int24", "UInt32", "Int32"]

            @printf(" %4d MB/s |", results[with_extrema][type]["compression"])
        end
        println()
    
        # Print decompression results
        print("| decompression    |")
        for type in ["UInt8", "Int8", "UInt16", "Int16", "UInt24", "Int24", "UInt32", "Int32"]
            @printf(" %4d MB/s |", results[with_extrema][type]["decompression"])
        end
        println()
    end
    println("\n")
end

"""
Print markdown table for logarithmic quantization results
"""
function print_log_results(results)
    # Print markdown table
    println("\n| Method           | UInt8    | UInt16    | UInt24   | UInt32   |")
    println("| ---------------- | -------: | --------: | -------: | -------: |")
    
    for method in ["linspace", "logspace"]
        println("| **$(method)** |")
        # Print compression results
        print("| compression      |")
        for type in ["LogQuant8Array", "LogQuant16Array", "LogQuant24Array", "LogQuant32Array"]
            @printf(" %4d MB/s |", results[method][type]["compression"])
        end
        println()
        
        # Print decompression results
        print("| decompression    |")
        for type in ["LogQuant8Array", "LogQuant16Array", "LogQuant24Array", "LogQuant32Array"]
            @printf(" %4d MB/s |", results[method][type]["decompression"])
        end
        println()
    end
    println()
end


# Example usage:
#results_linear = benchmark_linear_quantization(10_000_000) 
#print_lin_results(results_linear)   

results_log = benchmark_log_quantization(10_000_000);
print_log_results(results_log)
