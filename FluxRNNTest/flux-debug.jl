using Distributions, Zygote, LinearAlgebra

g1(x) = pdf(MvNormal(I(3)), x)
gradient(g1, [0,0,0]) # no error
g2(x) = pdf(MvNormal(1*I(3)), x)
gradient(g2, [0,0,0]) # ERROR: DimensionMismatch: x and y are of different lengths!

ρ1 = MvNormal(I(3))
ρ2 = MvNormal(1. * I(3))
ρ1 == ρ2 # true
g1(x) = pdf(ρ1, x)
gradient(g1, [0,0,0]) # no error
g2(x) = pdf(ρ2, x)
gradient(g2, [0,0,0]) # no error

