# using Revise
using Random: seed!
using Enzyme
using NNlib
using Base.Threads
using Base.Threads: @threads
using LinearAlgebra

function my_conv(x::AbstractArray{T,5}, w::AbstractArray{T,5}, cdims::ConvDims) where {T}

    y = similar(
        x,
        promote_type(T, T),
        NNlib.output_size(cdims)...,
        NNlib.channels_out(cdims),
        size(x, 5),
    )

    my_conv!(y, x, w, cdims)
end

function my_conv!(
    out::AbstractArray{T,5},
    in1::AbstractArray{T,5},
    in2::AbstractArray{T,5},
    cdims::C;
    kwargs...,
) where {T,C<:ConvDims}
    x_cs = Iterators.partition(1:size(in1, 4), NNlib.channels_in(cdims) ÷ NNlib.groupcount(cdims))
    w_cs = Iterators.partition(1:size(in2, 5), NNlib.channels_out(cdims) ÷ NNlib.groupcount(cdims))
    cdims2 = NNlib.basetype(C)(
        cdims,
        G = 1,
        C_in = NNlib.channels_in(cdims) ÷ NNlib.groupcount(cdims),
        C_out = NNlib.channels_out(cdims) ÷ NNlib.groupcount(cdims),
    )

    Threads.@sync for (xc, wc) in zip(x_cs, w_cs)
        x = @view in1[ntuple(i -> i == 4 ? xc : Colon(), 5)...]
        w = @view in2[ntuple(i -> i == 5 ? wc : Colon(), 5)...]
        y = @view out[ntuple(i -> i == 4 ? wc : Colon(), 5)...]
        Threads.@spawn my_conv_im2col!(y, x, w, cdims2; kwargs...)
    end

    return out
end


function kernel_index(w, h, d, cdims::ConvDims)
    NNlib.flipkernel(cdims) && return (w, h, d)
    kernel_w, kernel_h, kernel_d = NNlib.kernel_size(cdims)
    return (kernel_w - w + 1, kernel_h - h + 1, kernel_d - d + 1)
end

function my_conv_im2col!(
    y::AbstractArray{T,5},
    x::AbstractArray{T,5},
    w::AbstractArray{T,5},
    cdims::DenseConvDims;
    col::AbstractArray{T,3} = similar(x, NNlib.im2col_dims(cdims)),
    alpha::T = T(1),
    beta::T = T(0),
) where {T}
    M = prod(NNlib.output_size(cdims))
    N = NNlib.channels_out(cdims)
    K = prod(NNlib.kernel_size(cdims)) * NNlib.channels_in(cdims)

    @threads for batch_idx = 1:size(x, 5)
        # col_slice is a thread-local workspace
        col_slice = view(col, :, :, threadid())

        my_im2col!(col_slice, view(x, :, :, :, :, batch_idx), cdims)

        GC.@preserve col_slice w y begin
            col_ptr = pointer(col_slice)
            w_ptr = pointer(w)
            y_ptr = pointer(y, (batch_idx - 1) * M * N + 1)
            NNlib.gemm!(Val(false), Val(false), M, N, K, alpha, col_ptr, w_ptr, beta, y_ptr)
        end
    end
    return y
end

function my_im2col!(
    col::AbstractArray{T,2},
    x::AbstractArray{T,4},
    cdims::ConvDims,
) where {T}

    # Extract those nice, compile-time constant type parameters from `cdims`.
    width, height, depth = NNlib.input_size(cdims)
    kernel_w, kernel_h, kernel_d = NNlib.kernel_size(cdims)
    C_in = NNlib.channels_in(cdims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = NNlib.padding(cdims)
    dil_w, dil_h, dil_d = NNlib.dilation(cdims)
    stride_w, stride_h, stride_d = NNlib.stride(cdims)
    out_width, out_height, out_depth = NNlib.output_size(cdims)

    # Reshape col for easy access.
    col_reshaped = reshape(col, (
        # Output resolution
        out_width,
        out_height,
        out_depth,

        # By input patch size
        kernel_w,
        kernel_h,
        kernel_d,
        C_in,
    ))

    padded_regions, central_region = NNlib.calc_padding_regions(cdims)

    # A helper function to project from output (w, h) to input (input_w, input_h)
    @inline project(idx, stride, pad) = (idx - 1) * stride - pad + 1

    # We begin by copying the central region of the image which requires no padding at all.
    # Eliminating the branches of the fully generalized version below gives us a nice
    # speedup on the majority of the data.
    @inbounds for c = 1:C_in
        # Unpack "central region"
        w_region, h_region, d_region = central_region

        for kd = 1:kernel_d,
            kh = 1:kernel_h,
            kw = 1:kernel_w,
            d in d_region,
            h in h_region,
            w in w_region

            input_kd = project(d, stride_d, pad_d_lo) + (kd - 1) * dil_d
            input_kh = project(h, stride_h, pad_h_lo) + (kh - 1) * dil_h
            input_kw = project(w, stride_w, pad_w_lo) + (kw - 1) * dil_w
            kidxs = kernel_index(kw, kh, kd, cdims)

            xval::T = x[input_kw, input_kh, input_kd, c]
            col_reshaped[w, h, d, kidxs..., c] = xval
        end
    end


    # For each "padded region", we run the fully general version
    @inbounds for (w_region, h_region, d_region) in padded_regions
        for c = 1:C_in,
            d in d_region,
            h in h_region,
            w in w_region,
            kd = 1:kernel_d,
            kh = 1:kernel_h,
            kw = 1:kernel_w

            input_kd = project(d, stride_d, pad_d_lo) + (kd - 1) * dil_d
            input_kh = project(h, stride_h, pad_h_lo) + (kh - 1) * dil_h
            input_kw = project(w, stride_w, pad_w_lo) + (kw - 1) * dil_w

            kidxs = kernel_index(kw, kh, kd, cdims)

            out_of_bounds = (
                input_kd <= 0 ||
                input_kd > depth ||
                input_kh <= 0 ||
                input_kh > height ||
                input_kw <= 0 ||
                input_kw > width
            )
            if out_of_bounds
                col_reshaped[w, h, d, kidxs..., c] = T(0)
                continue
            end

            # Copy the data over
            xval::T = x[input_kw, input_kh, input_kd, c]
            col_reshaped[w, h, d, kidxs..., c] = xval
        end
    end
end

#############################################
# im2col algo
#############################################
w = randn(Float32, 3, 3, 3, 5, 7);
dw = zero(w);
x = randn(Float32, (3, 3, 3, 5, 8));
cdims = DenseConvDims(
    size(x), size(w); stride = (1,1,1), padding = (0,0,0), dilation = (1,1,1), flipkernel=false, groups=1)


my_conv(x, w, cdims);
loss(x, w, cdims) = sum(my_conv(x, w, cdims))
@time loss(x, w, cdims);
grads = Enzyme.autodiff(Reverse, loss, Const(x), Duplicated(w, dw), Const(cdims));

@code_warntype my_conv(x, w, cdims);
@code_lowered my_conv(x, w, cdims)
@code_typed my_conv(x, w, cdims)
@code_llvm my_conv(x, w, cdims)
@code_native my_conv(x, w, cdims)
