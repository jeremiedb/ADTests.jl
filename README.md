# AD Tests

## Dense layer with exp activation

### Zygote.jl

`/zygote/dense.jl`

```julia
#  47.057 ms (28 allocations: 105.00 MiB)
@btime zygote_test($m, $x);
```

## Tracker.jl

`/tracker/dense.jl`

```julia
#  66.045 ms (140 allocations: 107.02 MiB)
@btime tracker_test($m, $x);
```

## Yota.jl

`/yota/dense.jl`

```julia
# 43.431 ms (77 allocations: 89.01 MiB)
@btime val, grads = yota_test($yota_loss, $m, $x);
```

### AutoGrad.jl

`/autograd/dense.jl`

```julia
#  37.210 ms (154 allocations: 81.01 MiB)
@btime autograd_test($x, $w1, $b1);
```

## Nabla.jl

Stack overflow crash on rand().

## Diffractor.jl

Requires Julia v1.10

## Manuel forward-backwards

`/manual/dense.jl`

```julia
# 42.532 ms (34 allocations: 81.00 MiB)
@btime grads_a = g_custom_A($x, $ps);
# 38.850 ms (29 allocations: 49.00 MiB)
@btime grads_b = g_custom_B($x, $ps);
# 26.699 ms (19 allocations: 17.00 MiB)
@btime grads_c = g_custom_C($x, $ps);
```

How realistic is it to achieve cutom backward pass optimisations? 
For example, reusing forward pass allocations.

!["AD-allocs"](AD-allocs.png)
