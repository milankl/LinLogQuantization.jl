# LinLogQuantization.jl
[![CI](https://github.com/milankl/LinLogQuantization.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/milankl/LinLogQuantization.jl/actions/workflows/CI.yml)


Linear and logarithmic quantization for Julia arrays into 8, 16, 24 or 32-bit integers.
Linear quantization is available into both unsigned (`UInt8, UInt16, UInt24, UInt32`) and singed (`Int8, Int16, Int24, Int32`) 
integers. Logarithmic quantization is available for unsigned integers. 

## Table of Contents

- [Introduction](#Introduction)
- [Usage](#Usage)
- [Theory](#Theory)
- [Benchmarking](#Benchmarking)
- [Installation](#Installation)

## Introduction 

Quantization is a lossy compression method that divides the range of values in
an array of type `U<:Number` into equi-distant quanta and encodes those into values of 
type `T<:Integer`, which will range from `typemin(T)` to `typemax(T)`. For 
`T<:Unsigned`, this range will be `[0, 2^n-1]`, and for `T<:Signed`, this range 
will be `[-2^(n-1), 2^(n-1)-1]`, where `n` is the number of bits.
The quanta are either equi-distant in
linear space or in logarithmic space, which has a denser encoding for
values close to the minimum in trade-off with a less dense encoding close
to the maximum. 

Linear quantization takes values in (-∞,∞) (no `NaN` or `Inf`), while logarithmic quantization
is only supported for values in [0,∞).

## Usage

### Linear quantization

Linear quantization of n-dimensional arrays (any number format that can be
converted to `Float64` is supported, including `Float32, Float16`)
into both signed and unsigned 8, 16, 24 or 32-bit integers is achieved via
```julia
julia> A = rand(Float32, 1000)
julia> L = LinQuantArray{UInt8}(A)
1000-element LinQuantArray{UInt8,1}:
 0xc2
 0x19
 0x3e
 0x5b
    ⋮
```
and similarly with `LinQuantArray{T}` with `T` being `UInt8, UInt16, UInt24, UInt32, Int8, Int16, Int24`, or `Int32`.
Aliases exist for quantization into unsigned integers, namely: `LinQuant8Array, LinQuant16Array, LinQuant24Array, LinQuant32Array`.

```julia
julia> L2 = LinQuant8Array(A)
1000-element LinQuantArray{UInt8,1}:
 0xc2
 0x19
 0x3e
 0x5b
    ⋮
```

Decompression via: 
```julia
julia> Array(L)
1000-element Array{Float32,1}:
 0.76074356
 0.09858093
 0.24355145
 0.357177
    ⋮
```
`Array{T}()` optionally takes a type parameter `T` such that decompression to
other number formats than the default `Float32` is possible
```julia
julia> Array{Float16}(L)
1000-element Array{Float16,1}:
 0.76074356
 0.09858093
 0.24355145
 0.357177
    ⋮
```
For linear quantization you can also specify custom extrema instead of using the default 
minimum and maximum values of the array to encode. To do so, you can use the `extrema` keyword 
argument, which expects a tuple of min, max:
```julia 
julia> A = rand(Float32, (10, 10))
julia> L = LinQuantArray{Int16}(A; extrema=(0.3, 0.6))
julia> A2 = Array{Float32}(L)
10×10 Matrix{Float32}:
 0.3       0.6       0.3       …  0.6       0.3       0.309247
 0.503781  0.6       0.3          0.3       0.6       0.6
 0.3       0.3       0.575972     0.6       0.6       0.6
 0.384633  0.6       0.6          0.3       0.3       0.6
 0.479625  0.313481  0.6          0.3       0.6       0.393147
 0.6       0.6       0.3       …  0.3       0.6       0.6
 0.445077  0.6       0.411673     0.3       0.545823  0.419716
 0.3       0.6       0.479657     0.517707  0.32639   0.385713
 0.349673  0.6       0.3          0.6       0.3       0.6
 0.383639  0.6       0.6          0.6       0.6       0.6
```

which effectively clamps the data distribution into the range determined by `extrema`.


### Logarithmic quantization

In a similar way, `LogQuant8Array, LogQuant16Array, LogQuant24Array, LogQuant32Array`
compress an n-dimensional array (non-negative elements only) via logarithmic quantization.
```julia
julia> A = rand(Float32, 100, 100)
julia> A[1,1] = 0    # zero is possible
julia> L = LogQuant16Array(A)
100×100 LogQuantArray{UInt16,2}:
 0x0000  0xf22d  0xfdf6  0xf3e8  0xf775  …  
 0xe3dc  0xfdc0  0xedb5  0xed47  0xee5b     
 0xde3d  0xbe58  0xb541  0xf573  0x9885     
 0xf38b  0xfefe  0xea2f  0xfbb6  0xf0d2     
 0xd0d2  0xfe1f  0xff60  0xf6cd  0xec26        
 0xffa6  0xe621  0xf14d  0xfb2c  0xf50c  …  
 0xfcb7  0xe6fb  0xf237  0xecd5  0xfb0a     
 0xe4ed  0xf86f  0xf83d  0xff86  0xb686     
      ⋮                                  ⋱
```
Exception occurs for 0, which is mapped to `0x0` as shown in the first element.
`Ox1` to `0xff...ff` are then the available bitpatterns to encode the range from `minimum(A)`
to `maximum(A)` logarithmically (`extrema` keyword currently not available).
By default the rounding mode for logarithmic quantization (also linear quantization)
is round-to-nearest in linear space. Alternatively, a second argument can be either
`:linspace` or `:logspace`, which allows for round-to-nearest in logarithmic space.
See a derivation of this below.
Decompression as with linear quantization via the `Array()` function.

#### Quantizing Along a Specific Dimension

In some cases, you may want to quantize an array along a specific dimension independently. This is useful when each slice along that dimension represents a separate dataset or when the range of values varies significantly across slices. You can achieve this by specifying the `dims` keyword argument in the `LinQuantArray` function.

For example, to quantize a 3D array along the second dimension:

```julia
julia> L1 = LinQuantArray{UInt8}(A, dims=2)

julia> L2 = LogQuantArray{UInt8}(A, dims=2)
```

Each element in the resulting vector `L` is a `LinQuantArray` that represents a slice of the original array along the specified dimension. This allows for independent quantization of each slice.

## Theory

### Linear quantization

To compress an array `A` into a type `T`, the minimum and maximum of both the array and the type are obtained
```julia
Amin = minimum(A)
Amax = maximum(A)

Tmin = typemin(T)
Tmax = typemax(T)
```
which allows the calculation of `Δ⁻¹`, the inverse of the spacing between two
quantums
```julia
Δ⁻¹ = (Tmax - Tmin)/(Amax - Amin)
```
For every element `a` in `A` the corresponding quantum `q` which is closest in linear space
is calculated via
```julia
q = T(round((A[i]-Amin)*Δ⁻¹ + Tmin))
```
where `round` is the round-to-nearest-integer function and `T` the conversion
function to the chosen integer type. Consequently, an array of all `q` and `Amin, Amax`
have to be stored to allow for decompression, which is obtained by reversing the conversion
from `a` to `q`. Note that the rounding error is introduced as the `round` function cannot
be inverted.

### Logarithmic quantization

Logarithmic quantization distributes the quanta logarithmically, such that
more bitpatterns are reserved for values close to the minimum and fewer close to
the maximum in `A`. Logarithmic quantization can be generalised to negative values
by introducing a sign-bit, however, we limit our application here to non-negative
values. We obtain the minimum and maximum value in `A` as follows
```julia
Alogmin = log(minpos(A))
Alogmax = log(maximum(A))
```
where zeros are ignored in the `minpos` function, which instead returns the smallest
positive value. The inverse spacing `Δ` is then
```julia
Δ⁻¹ = (2^n-2)/(logmax-logmin)
```
Note, that only `2^n-1` (and not 2^n as for linear quantization) bitpatterns
are used to resolve the range between minimum and maximum, as we want to reserve
the bitpattern `0x000000` for zero. The corresponding quantum `q` for `a` in
`A` is then
```julia
q = T(round(c + Δ⁻¹*log(a))) + 0x1
```
unless `a=0` in which case `q=0x000000`. The constant `c` can be set as `-Alogmin*Δ⁻¹`
such that we obtain essentially the same compression function as for linear quantization,
except that every element `a` in `A` is converted to their logarithm first. However,
rounding to nearest in logarithmic space will therefore be achieved, which is a
biased rounding mode, that has a bias away from zero. We can correct this
bias by using instead
```julia
c = 1/2 - Δ⁻¹*log(minimum(A)*(exp(1/Δ⁻¹)+1)/2)
```
which yields round-to-nearest in linear space. See next section.

### Round to nearest in linear or logarithmic space

For a logarithmic integer system with base `b` (i.e. only `0, b, b², b³,...`
are representable), we have
```julia
log_b(1) = 0
log_b(√b) = 0.5
log_b(b) = 1
log_b(√b³) = 1.5
log_b(b²) = 2
```
such that `q*√b` is always halfway between two representable numbers `q, q2` in
logarithmic space, which will be the threshold for round up or down in the `round`
function. `q*√b` is not halfway in linear space, which is always at
`q + (q*b - q)/2`. For simplicity we can set `q=1`, and for `b=2` we find that
```julia
√2 = 1.41... != 1.5 = 1 + (2-1)/2
```
Round-to-nearest in log-space therefore rounds also the values between 1.41... and 1.5
to 2, which will introduce an away-from-zero bias. As halfway in log-space is reached
by multiplication with `√b`, this can be corrected to halfway in linear space
by adding a constant `c_b` in log-space, such that conversion from halfway in linear
space, i.e. `1+(b-1)/2` should yield halfway in log-space, i.e. 0.5  
```julia
c_b + log_b(1+(b-1)/2) = 0.5
```
So, for `b=2` we have `c_b = 0.5 - log2(1.5) ≈ -0.085`. Hence, a small number will
be subtracted before rounding is applied to reduce the away-from-zero bias.

![](https://github.com/milankl/LinLogquantization.jl/blob/main/figs/round_logquant.png)

**Figure A1.** Schematic to illustrate round-to-nearest in linear vs logarithmic
space for logarithmic number systems.

We now generalise the logarithmic system, such that the distance `dlog = 1/Δ⁻¹` between
two representable numbers (i.e. quanta) is not necessarily 1 (in log-space) and
we allow for an offset as done in the logarithmic quantization. Let `min` be the
offset (i.e. the minimum of the uncompressed array) and `dlin` the spacing between
the first two representable quanta `min, q2`. Then the logarithm of halfway in
linear space, `log_b(min + dlin/2)`, should map to `0.5`.
```julia
c_b + (log_b(min + dlin/2) - log_b(min))/dlog = 0.5
```
With `dlin = b^(log_b(min) + dlog) - min` this can be transformed into
```julia
c_b = 1/2 - 1/dlog*log_b((b^dlog + 1)/2)
```
and combined with the offset correction `-log_b(min)*Δ⁻¹` to form either
```julia
c = -log(min)*Δ⁻¹,   (round-to-nearest in log-space)
c = 1/2 - Δ⁻¹*log(minimum(A)*(exp(1/Δ⁻¹)+1)/2)    (round-to-nearest in linear-space)
```
with `b = ℯ`, so that only the natural logarithm has to be computed for every
element in the uncompressed array.
 
## Benchmarking

Approximate throughputs in a MacBook Pro M2, 24GB RAM are (via `@benchmark`)

### Linear quantization
| Method           | UInt8    | Int8     | UInt16    | Int16     | UInt24   | Int24    | UInt32   | Int32    |
| ---------------- | -------: | -------: | --------: | --------: | -------: | -------: | -------: | -------: |
| **Default extrema** |
| compression      | 1500 MB/s | 1500 MB/s | 1500 MB/s | 1500 MB/s |  100 MB/s |  100 MB/s | 1500 MB/s | 1500 MB/s |
| decompression    | 16000 MB/s | 8000 MB/s | 16000 MB/s | 8000 MB/s | 8000 MB/s | 8000 MB/s | 16000 MB/s | 16000 MB/s |
| **Custom extrema** |
| compression      | 8000 MB/s | 8000 MB/s | 8000 MB/s | 8000 MB/s |    100 MB/s |    100 MB/s | 8000 MB/s | 8000 MB/s |
| decompression    | 16000 MB/s | 8000 MB/s | 16000 MB/s | 8000 MB/s |    8000 MB/s |    8000 MB/s | 16000 MB/s | 16000 MB/s |


### Logarithmic quantization
| Method           | UInt8    | UInt16    | UInt24   | UInt32   |
| ---------------- | -------: | --------: | -------: | -------: |
| **linspace** |
| compression      |  800 MB/s |  800 MB/s |  100 MB/s |  700 MB/s |
| decompression    | 2000 MB/s | 2000 MB/s | 2000 MB/s | 2000 MB/s |
| **logspace** |
| compression      |  800 MB/s |  800 MB/s |  100 MB/s |  700 MB/s |
| decompression    | 2000 MB/s | 2000 MB/s | 2000 MB/s | 2000 MB/s |

24-bit quantization is via `UInt24`  and `Int24` from the `BitIntegers` package,
which introduces a drastic slow-down.

## Installation

LinLogQuantization.jl is registered, so just do
```julia
julia>] add LinLogQuantization
```
where `]` opens the package manager
