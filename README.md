# LinLogQuantization.jl
[![CI](https://github.com/milankl/LinLogQuantization.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/milankl/LinLogQuantization.jl/actions/workflows/CI.yml)

Linear and logarithmic quantisation for Julia arrays into 8, 16, 24 or 32-bit.
Quantisation is a lossy compression method that divides the range of values in
an array in equi-distant quantums and encodes those from 0 to `2^n-1` where
`n` is the number of bits available. The quantums are either equi-distant in
linear space or in logarithmic space, which has a denser encoding for
values close to the minimum in trade-off with a less dense encoding close
to the maximum. 

Linear quantization takes values in (-∞,∞) (no `NaN` or `Inf`) logarithmic quantization
is only supported for values in [0,∞).

## Usage: Linear quantization

Linear quantisation of n-dimensional arrays (any number format that can be
converted to `Float64` is supported, including `Float32, Float16`)
into 8, 16, 24 or 32 bit is achieved via
```julia
julia> A = rand(Float32,1000)
julia> L = LinQuant8Array(A)
1000-element LinQuantArray{UInt8,1}:
 0xc2
 0x19
 0x3e
 0x5b
    ⋮
```
and similarly with `LinQuant16Array, LinQuant24Array, LinQuant32Array`.
Decompression via
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
other number formats than the default `Float32` is possible involves a rounding
error which follows a round-to-nearest in linear space.

### Logarithmic quantisation

In a similar way, `LogQuant8Array, LogQuant16Array, LogQuant24Array, LogQuant32Array`
compresses an n-dimensional array (non-negative elements only) via logarithmic quantisation.
```julia
julia> A = rand(Float32,100,100)
julia> A[1,1] = 0
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
Exception occurs for 0, which is mapped to `0x0`.
`Ox1` to `0xff...ff` are then the available bitpatterns to encode the range from `minimum(A)`
to `maximum(A)` logarithmically. By default the rounding mode for logarithmic quantisation
is round-to-nearest in linear space. Alternatively, a second argument can be either
`:linspace` or `:logspace`, which allows for round-to-nearest in logarithmic space.
Decompression as with linear quantisation via the `Array()` function.

## Theory

To compress an array `A`, the minimum and maximum is obtained
```julia
Amin = minimum(A)
Amax = maximum(A)
```
which allows the calculation of `Δ`, the inverse of the spacing between two
quantums
```julia
Δ = 2^(n-1)/(Amax-Amin)
```
where `n` is the number of bits used for quantisation. For every
element `a` in `A` the corresponding quantum `q` which is closest in linear space
is calculated via
```julia
q = T(round((a-Amin)*Δ))
```
where `round` is the round-to-nearest function for integers and `T` the conversion
function to 24-bit unsigned integers `UInt24` (or `UInt8, UInt16` for other choices
of `n`). Consequently, an array of all `q` and `Amin,Amax` have to be stored to
allow for decompression, which is obtained by reversing the conversion from `a`
to `q`. Note that the rounding error is introduced as the `round` function cannot
be inverted.

Logarithmic quantisation distributes the quantums logarithmically, such that
more bitpatterns are reserved for values close to the minimum and fewer close to
the maximum in `A`. Logarithmic quantisation can be generalised to negative values
by introducing a sign-bit, however, we limit our application here to non-negative
values. We obtain the minimum and maximum value in `A` as follows
```julia
Alogmin = log(minpos(A))
Alogmax = log(maximum(A))
```
where zeros are ignored in the `minpos` function, which instead returns the smallest
positive value. The inverse spacing `Δ` is then
```julia
Δ = 2^(n-2)/(logmax-logmin)
```
Note, that only `2^(n-1)` (and not 2^n as for linear quantisation) bitpatterns
are used to resolve the range between minimum and maximum, as we want to reserve
the bitpattern `0x000000` for zero. The corresponding quantum `q` for `a`
`A` is then
```julia
q = T(round(c + Δ*log(a)))+0x1
```
unless `a=0` in which case `q=0x000000`. The constant `c` can be set as `-Alogmin*Δ`
such that we obtain essentially the same compression function as for linear quantisation,
except that every element `a` in `A` is converted to their logarithm first. However,
rounding to nearest in logarithmic space will therefore be achieved, which is a
biased rounding mode, that has a bias away from zero. We can correct this
round-to-nearest in logarithmic space rounding mode with
```julia
c = 1/2 - Δ*log(minimum(A)*(exp(1/Δ)+1)/2)
```
which yields round-to-nearest in linear space. See next section.

## Round to nearest in linear or logarithmic space

For a logarithmic integer system with base `b` (i.e. only `0,b,b²,b³,...`
are representable), for example, we have
```julia
log_b(1) = 0
log_b(√b) = 0.5
log_b(b) = 1
log_b(√b³) = 1.5
log_b(b²) = 2
```
such that `q*√b` is always halfway between two representable numbers `q,q2` in
logarithmic space, which will be the threshold for round up or down in the `round`
function. `q*√b` is not halfway in linear space, which is always at
`q + (q*b - q)/2`. For simplicity we can set `q=1`, and for `b=2` we find that
```julia
√2 = 1.41... != 1.5 = 1 + (2-1)/2
```
Round-to-nearest in log-space therefore rounds the values between 1.41... and 1.5
to 2, which will introduce an away-from-zero bias. As halfway in log-space is reached
by multiplication with `√b`, this can be corrected to halfway in linear space
by adding a constant `c_b` in log-space, such that conversion from halfway in linear
space, i.e. `1+(b-1)/2` should yield halway in log-space, i.e. 0.5  
```julia
c_b + log_b(1+(b-1)/2) = 0.5
```
So, for `b=2` we have `c_b = 0.5 - log2(1.5) ≈ -0.085`. Hence, a small number will
be subtracted before rounding is applied to reduce the away-from-zero bias.

![](https://github.com/milankl/LinLogQuantization.jl/blob/master/figs/round_logquant.png)

**Figure A1.** Schematic to illustrate round-to-nearest in linear vs logarithmic
space for logarithmic number systems.

We now generalise the logarithmic system, such that the distance `dlog = 1/Δ` between
two representable numbers (i.e. quantums) is not necessarily 1 (in log-space) and
we allow for an offset as done in the logarithmic quantisation. Let `min` be the
offset (i.e. the minimum of the uncompressed array) and `dlin` the spacing between
the first two representable quantums `min,q2`. Then the logarithm of halfway in
linear space, `log_b(min + dlin/2)`, should map to `0.5`.
```julia
c_b + (log_b(min + dlin/2) - log_b(min))/dlog = 0.5
```
With `dlin = b^(log_b(min) + dlog) - min` this can be transformed into
```julia
c_b = 1/2 - 1/dlog*log_b((b^dlog + 1)/2)
```
and combined with the offset correction `-log_b(min)*Δ` to form either
```julia
c = -log(min)*Δ,   (round-to-nearest in log-space)
c = 1/2 - Δ*log(minimum(A)*(exp(1/Δ)+1)/2)    (round-to-nearest in linear-space)
```
with `b = ℯ`, so that only the natural logarithm has to be computed for every
element in the uncompressed array.

## Benchmarking

Approximate throughputs are (via `@btime`)

| Method               | 8 bit    | 16 bit    | 24 bit   |    32 bit|
| -------------------- | -------: | --------: | -------: | -------: |
| **Linear** |
| compression   | 1350 MB/s| 1350 MB/s | 50 MB/s  | 1350 MB/s|
| decompression | 4700 MB/s| 4700 MB/s | 4000 MB/s| 3600 MB/s| 
| **Logarithmic** |
| decompression  |  250 MB/s|   250 MB/s|  250 MB/s|  500 MB/s|
| compression    |  285 MB/s|   285 MB/s|   40 MB/s|  285 MB/s|

24-bit quantisation is via `UInt24` from the `BitIntegers` package,
which introduces a drastic slow-down.

## Installation

Open the package manager via `]` then
```julia
julia> add https://github.com/milankl/LinLogQuantization.jl
```
The package is not yet registered in the Julia registry.
