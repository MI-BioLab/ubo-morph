// CuPy CUDA kernels for the morphing backend.
// Kernel arguments intentionally use flat POD arrays so launches avoid temporary host-side structures.

// Inverse affine sampling: each output pixel maps to source coordinates and bilinearly samples with a constant border.
__device__ __forceinline__ float constant_sample(
    const unsigned char* image,
    const int row,
    const int col,
    const int channel,
    const int H,
    const int W
) {
    if (row < 0 || row >= H || col < 0 || col >= W) return 0.0f;
    return (float)image[(row * W + col) * 3 + channel];
}

extern "C" __global__ void affine_warp_uint8(
    const unsigned char* __restrict__ image,
    const int H,
    const int W,
    const int output_H,
    const int output_W,
    const float m00,
    const float m01,
    const float m02,
    const float m10,
    const float m11,
    const float m12,
    unsigned char* __restrict__ output
) {
    const long long pixel =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long pixel_count = (long long)output_H * output_W;
    if (pixel >= pixel_count) return;
    const int row = (int)(pixel / output_W);
    const int col = (int)(pixel - (long long)row * output_W);
    const float source_col = m00 * col + m01 * row + m02;
    const float source_row = m10 * col + m11 * row + m12;
    const int row0 = (int)floorf(source_row);
    const int col0 = (int)floorf(source_col);
    const float row_fraction = source_row - row0;
    const float col_fraction = source_col - col0;
    const long long base = pixel * 3;

    for (int channel = 0; channel < 3; ++channel) {
        const float value00 =
            constant_sample(image, row0, col0, channel, H, W);
        const float value01 =
            constant_sample(image, row0, col0 + 1, channel, H, W);
        const float value10 =
            constant_sample(image, row0 + 1, col0, channel, H, W);
        const float value11 =
            constant_sample(image, row0 + 1, col0 + 1, channel, H, W);
        const float value =
            (1.0f - row_fraction) *
                ((1.0f - col_fraction) * value00 + col_fraction * value01) +
            row_fraction *
                ((1.0f - col_fraction) * value10 + col_fraction * value11);
        output[base + channel] =
            (unsigned char)min(max(__float2int_rn(value), 0), 255);
    }
}

// Exact area resampling: each thread integrates the source pixels covered by one output pixel's footprint.
extern "C" __global__ void area_resize_uint8(
    const unsigned char* __restrict__ image,
    const int H,
    const int W,
    const int output_H,
    const int output_W,
    unsigned char* __restrict__ output
) {
    const long long pixel =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long pixel_count = (long long)output_H * output_W;
    if (pixel >= pixel_count) return;
    const int output_row = (int)(pixel / output_W);
    const int output_col =
        (int)(pixel - (long long)output_row * output_W);
    const float scale_y = (float)H / output_H;
    const float scale_x = (float)W / output_W;
    const float start_y = output_row * scale_y;
    const float end_y = (output_row + 1) * scale_y;
    const float start_x = output_col * scale_x;
    const float end_x = (output_col + 1) * scale_x;
    const int first_row = (int)floorf(start_y);
    const int last_row = min((int)ceilf(end_y) - 1, H - 1);
    const int first_col = (int)floorf(start_x);
    const int last_col = min((int)ceilf(end_x) - 1, W - 1);
    float sums[3] = {0.0f, 0.0f, 0.0f};

    for (int row = first_row; row <= last_row; ++row) {
        const float row_weight =
            fminf(end_y, row + 1.0f) - fmaxf(start_y, (float)row);
        for (int col = first_col; col <= last_col; ++col) {
            const float col_weight =
                fminf(end_x, col + 1.0f) - fmaxf(start_x, (float)col);
            const float weight = row_weight * col_weight;
            const long long source = ((long long)row * W + col) * 3;
            for (int channel = 0; channel < 3; ++channel) {
                sums[channel] += weight * image[source + channel];
            }
        }
    }
    const float inverse_area = 1.0f / (scale_x * scale_y);
    const long long destination = pixel * 3;
    for (int channel = 0; channel < 3; ++channel) {
        output[destination + channel] = (unsigned char)min(
            max(__float2int_rn(sums[channel] * inverse_area), 0),
            255
        );
    }
}

// Hull rasterization: flattened triangle bounding boxes turn irregular geometry into a compact parallel work list.
// Each ten-float descriptor stores three vertices followed by its clipped box;
// offsets map a global work item back to the descriptor that owns its box.
extern "C" __global__ void rasterize_convex_hull(
    const float* __restrict__ descriptors,
    const long long* __restrict__ offsets,
    const int triangle_count,
    const int W,
    const long long work_count,
    unsigned char* __restrict__ output
) {
    const long long work_index =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (work_index >= work_count) return;

    int low = 0;
    int high = triangle_count;
    while (low + 1 < high) {
        const int middle = low + (high - low) / 2;
        if (offsets[middle] <= work_index) {
            low = middle;
        } else {
            high = middle;
        }
    }
    const float* descriptor = descriptors + low * 10;
    const int box_x = (int)descriptor[6];
    const int box_y = (int)descriptor[7];
    const int box_width = (int)descriptor[8];
    const long long local = work_index - offsets[low];
    const int col = box_x + (int)(local % box_width);
    const int row = box_y + (int)(local / box_width);
    const float px = (float)col;
    const float py = (float)row;

    const float ax = descriptor[0], ay = descriptor[1];
    const float bx = descriptor[2], by = descriptor[3];
    const float cx = descriptor[4], cy = descriptor[5];
    const float w0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    const float w1 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
    const float w2 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
    if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
        output[row * W + col] = 1;
    }
}

// Non-overlapping triangle warp: rasterize only target bounding boxes, then barycentrically map covered pixels back to the source.
// The sixteen-float descriptor is [target vertices, inverse affine map, box x/y/width].
__device__ __forceinline__ bool is_top_left(
    const float x0,
    const float y0,
    const float x1,
    const float y1
) {
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    return (dy < 0.0f) || (dy == 0.0f && dx > 0.0f);
}

__device__ __forceinline__ float reflect101(float value, const int size) {
    if (size <= 1) return 0.0f;
    const float period = 2.0f * (size - 1);
    value = fmodf(fabsf(value), period);
    return value > size - 1 ? period - value : value;
}

extern "C" __global__ void warp_triangle_boxes(
    const unsigned char* __restrict__ image,
    const float* __restrict__ descriptors,
    const long long* __restrict__ offsets,
    const int triangle_count,
    const int H,
    const int W,
    const long long work_count,
    unsigned char* __restrict__ output
) {
    const long long work_index =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (work_index >= work_count) return;

    int low = 0;
    int high = triangle_count;
    while (low + 1 < high) {
        const int middle = low + (high - low) / 2;
        if (offsets[middle] <= work_index) {
            low = middle;
        } else {
            high = middle;
        }
    }
    const int triangle = low;
    const float* descriptor = descriptors + triangle * 16;
    const int box_x = (int)descriptor[12];
    const int box_y = (int)descriptor[13];
    const int box_width = (int)descriptor[14];
    const long long local = work_index - offsets[triangle];
    const int col = box_x + (int)(local % box_width);
    const int row = box_y + (int)(local / box_width);

    const float ax = descriptor[0], ay = descriptor[1];
    const float bx = descriptor[2], by = descriptor[3];
    const float cx = descriptor[4], cy = descriptor[5];
    const float px = (float)col, py = (float)row;

    const float w0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    const float w1 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
    const float w2 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
    const bool image_boundary =
        col == 0 || col == W - 1 || row == 0 || row == H - 1;
    const bool inside =
        (w0 > 0.0f ||
         (w0 == 0.0f && (image_boundary || is_top_left(ax, ay, bx, by)))) &&
        (w1 > 0.0f ||
         (w1 == 0.0f && (image_boundary || is_top_left(bx, by, cx, cy)))) &&
        (w2 > 0.0f ||
         (w2 == 0.0f && (image_boundary || is_top_left(cx, cy, ax, ay))));
    if (!inside) return;

    float sc = descriptor[6] * col + descriptor[7] * row + descriptor[8];
    float sr = descriptor[9] * col + descriptor[10] * row + descriptor[11];
    sr = reflect101(sr, H);
    sc = reflect101(sc, W);

    const int r0 = (int)floorf(sr), r1 = min(r0 + 1, H - 1);
    const int c0 = (int)floorf(sc), c1 = min(c0 + 1, W - 1);
    const float dr = sr - r0, dc = sc - c0;
    const long long output_index = ((long long)row * W + col) * 3;
    for (int ch = 0; ch < 3; ++ch) {
        const float v00 =
            (float)image[((long long)r0 * W + c0) * 3 + ch];
        const float v01 =
            (float)image[((long long)r0 * W + c1) * 3 + ch];
        const float v10 =
            (float)image[((long long)r1 * W + c0) * 3 + ch];
        const float v11 =
            (float)image[((long long)r1 * W + c1) * 3 + ch];
        const float value =
            (1.0f - dr) * ((1.0f - dc) * v00 + dc * v01)
            +        dr  * ((1.0f - dc) * v10 + dc * v11);
        output[output_index + ch] =
            (unsigned char)min(max(__float2int_rn(value), 0), 255);
    }
}

// Overlap fallback: first establish deterministic triangle ownership, then sample once per output pixel from that ownership map.
// Ownership uses the largest triangle index, matching the sequential overwrite order
// without making the final image depend on atomic write timing.
__device__ __forceinline__ float fallback_reflect101(
    float value,
    const int size
) {
    if (size <= 1) return 0.0f;
    const float period = 2.0f * (size - 1);
    value = fmodf(fabsf(value), period);
    return value > size - 1 ? period - value : value;
}

extern "C" __global__ void build_box_membership(
    const float* __restrict__ descriptors,
    const long long* __restrict__ offsets,
    const int triangle_count,
    const int W,
    const long long work_count,
    int* __restrict__ membership
) {
    const long long work_index =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (work_index >= work_count) return;
    int low = 0;
    int high = triangle_count;
    while (low + 1 < high) {
        const int middle = low + (high - low) / 2;
        if (offsets[middle] <= work_index) low = middle;
        else high = middle;
    }
    const float* descriptor = descriptors + low * 16;
    const int box_x = (int)descriptor[12];
    const int box_y = (int)descriptor[13];
    const int box_width = (int)descriptor[14];
    const long long local = work_index - offsets[low];
    const int col = box_x + (int)(local % box_width);
    const int row = box_y + (int)(local / box_width);
    const float px = (float)col, py = (float)row;
    const float ax = descriptor[0], ay = descriptor[1];
    const float bx = descriptor[2], by = descriptor[3];
    const float cx = descriptor[4], cy = descriptor[5];
    const float w0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    const float w1 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
    const float w2 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
    if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
        atomicMax(&membership[(long long)row * W + col], low);
    }
}

extern "C" __global__ void sample_box_membership(
    const unsigned char* __restrict__ image,
    const float* __restrict__ descriptors,
    const int* __restrict__ membership,
    const int H,
    const int W,
    unsigned char* __restrict__ output
) {
    const long long pixel =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long pixel_count = (long long)H * W;
    if (pixel >= pixel_count) return;
    const int triangle = membership[pixel];
    const long long output_index = pixel * 3;
    if (triangle < 0) {
        output[output_index] = 0;
        output[output_index + 1] = 0;
        output[output_index + 2] = 0;
        return;
    }
    const int row = (int)(pixel / W);
    const int col = (int)(pixel - (long long)row * W);
    const float* descriptor = descriptors + triangle * 16;
    float source_col =
        descriptor[6] * col + descriptor[7] * row + descriptor[8];
    float source_row =
        descriptor[9] * col + descriptor[10] * row + descriptor[11];
    source_row = fallback_reflect101(source_row, H);
    source_col = fallback_reflect101(source_col, W);
    const int row0 = (int)floorf(source_row);
    const int row1 = min(row0 + 1, H - 1);
    const int col0 = (int)floorf(source_col);
    const int col1 = min(col0 + 1, W - 1);
    const float row_fraction = source_row - row0;
    const float col_fraction = source_col - col0;
    for (int channel = 0; channel < 3; ++channel) {
        const float value00 =
            image[((long long)row0 * W + col0) * 3 + channel];
        const float value01 =
            image[((long long)row0 * W + col1) * 3 + channel];
        const float value10 =
            image[((long long)row1 * W + col0) * 3 + channel];
        const float value11 =
            image[((long long)row1 * W + col1) * 3 + channel];
        const float value =
            (1.0f - row_fraction) *
                ((1.0f - col_fraction) * value00 +
                 col_fraction * value01) +
            row_fraction *
                ((1.0f - col_fraction) * value10 +
                 col_fraction * value11);
        output[output_index + channel] =
            (unsigned char)min(max(__float2int_rn(value), 0), 255);
    }
}

// Blend: combine two uint8 images in one pass, using either a supplied per-pixel alpha or a scalar factor.
extern "C" __global__ void blend_uint8(
    const unsigned char* __restrict__ image1,
    const unsigned char* __restrict__ image2,
    const float* __restrict__ alpha,
    const long long pixel_count,
    const float factor,
    const int use_alpha,
    unsigned char* __restrict__ output
) {
    const long long pixel =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;

    const float foreground = use_alpha ? alpha[pixel] : 1.0f - factor;
    const float background = 1.0f - foreground;
    const long long base = pixel * 3;
    for (int channel = 0; channel < 3; ++channel) {
        const float value =
            foreground * image1[base + channel] +
            background * image2[base + channel];
        output[base + channel] =
            (unsigned char)min(max(__float2int_rn(value), 0), 255);
    }
}

// HLS conversion: keep OpenCV's 8-bit BGR/HLS encoding while moving conversion arithmetic into one pixel kernel.
extern "C" __global__ void bgr_to_hls(
    const unsigned char* __restrict__ input,
    const long long pixel_count,
    unsigned char* __restrict__ output
) {
    const long long pixel =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const long long base = pixel * 3;
    const float blue = input[base] / 255.0f;
    const float green = input[base + 1] / 255.0f;
    const float red = input[base + 2] / 255.0f;
    const float maximum = fmaxf(fmaxf(red, green), blue);
    const float minimum = fminf(fminf(red, green), blue);
    const float delta = maximum - minimum;
    const float lightness = (maximum + minimum) * 0.5f;

    float hue = 0.0f;
    float saturation = 0.0f;
    if (delta > 0.0f) {
        const float denominator = 1.0f - fabsf(2.0f * lightness - 1.0f);
        saturation = denominator > 0.0f ? delta / denominator : 0.0f;
        if (maximum == red) {
            hue = fmodf((green - blue) / delta, 6.0f);
        } else if (maximum == green) {
            hue = (blue - red) / delta + 2.0f;
        } else {
            hue = (red - green) / delta + 4.0f;
        }
    }
    int hue_byte = __float2int_rn(hue * 30.0f) % 180;
    if (hue_byte < 0) hue_byte += 180;
    output[base] = (unsigned char)hue_byte;
    output[base + 1] = (unsigned char)min(
        max(__float2int_rn(lightness * 255.0f), 0),
        255
    );
    output[base + 2] = (unsigned char)min(
        max(__float2int_rn(saturation * 255.0f), 0),
        255
    );
}

extern "C" __global__ void hls_to_bgr(
    const unsigned char* __restrict__ input,
    const long long pixel_count,
    unsigned char* __restrict__ output
) {
    const long long pixel =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const long long base = pixel * 3;
    const float hue = input[base] / 30.0f;
    const float lightness = input[base + 1] / 255.0f;
    const float saturation = input[base + 2] / 255.0f;
    const float chroma =
        (1.0f - fabsf(2.0f * lightness - 1.0f)) * saturation;
    const float intermediate =
        chroma * (1.0f - fabsf(fmodf(hue, 2.0f) - 1.0f));
    const int sector = (int)floorf(hue);

    const float red =
        sector == 0 || sector == 5 ? chroma :
        sector == 1 || sector == 4 ? intermediate : 0.0f;
    const float green =
        sector == 1 || sector == 2 ? chroma :
        sector == 0 || sector == 3 ? intermediate : 0.0f;
    const float blue =
        sector == 3 || sector == 4 ? chroma :
        sector == 2 || sector == 5 ? intermediate : 0.0f;
    const float offset = lightness - chroma * 0.5f;

    output[base] = (unsigned char)min(
        max(__float2int_rn((blue + offset) * 255.0f), 0),
        255
    );
    output[base + 1] = (unsigned char)min(
        max(__float2int_rn((green + offset) * 255.0f), 0),
        255
    );
    output[base + 2] = (unsigned char)min(
        max(__float2int_rn((red + offset) * 255.0f), 0),
        255
    );
}

// Histogram matching: accumulate channel histograms in block-local shared memory, derive CDF lookups, then apply them in parallel.
// Histogram storage is channel-major with 256 input bins followed by 256 reference bins;
// the lookup kernel turns those counts into a monotone byte-to-byte transfer function.
extern "C" __global__ void build_histograms(
    const unsigned char* __restrict__ image,
    const unsigned char* __restrict__ mask,
    const unsigned char* __restrict__ reference,
    const unsigned char* __restrict__ reference_mask,
    const long long input_pixel_count,
    const long long reference_pixel_count,
    const int channels,
    unsigned int* __restrict__ histograms
) {
    extern __shared__ unsigned int local_histograms[];
    const int bin_count = channels * 2 * 256;
    for (int index = threadIdx.x; index < bin_count; index += blockDim.x) {
        local_histograms[index] = 0;
    }
    __syncthreads();

    for (
        long long pixel = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        pixel < input_pixel_count;
        pixel += (long long)gridDim.x * blockDim.x
    ) {
        const long long base = pixel * channels;
        if (mask[pixel]) {
            for (int channel = 0; channel < channels; ++channel) {
                atomicAdd(
                    &local_histograms[(channel * 2) * 256 + image[base + channel]],
                    1U
                );
            }
        }
    }
    for (
        long long pixel = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        pixel < reference_pixel_count;
        pixel += (long long)gridDim.x * blockDim.x
    ) {
        const long long base = pixel * channels;
        if (reference_mask[pixel]) {
            for (int channel = 0; channel < channels; ++channel) {
                atomicAdd(
                    &local_histograms[
                        (channel * 2 + 1) * 256 + reference[base + channel]
                    ],
                    1U
                );
            }
        }
    }
    __syncthreads();

    for (int index = threadIdx.x; index < bin_count; index += blockDim.x) {
        atomicAdd(&histograms[index], local_histograms[index]);
    }
}

extern "C" __global__ void build_histogram_lookup(
    const unsigned int* __restrict__ histograms,
    const int channels,
    unsigned char* __restrict__ lookup
) {
    const int channel = blockIdx.x;
    if (channel >= channels || threadIdx.x != 0) return;
    const unsigned int* input = histograms + channel * 512;
    const unsigned int* reference = input + 256;

    unsigned long long input_total = 0;
    unsigned long long reference_total = 0;
    for (int value = 0; value < 256; ++value) {
        input_total += input[value];
        reference_total += reference[value];
    }
    if (input_total == 0 || reference_total == 0) {
        for (int value = 0; value < 256; ++value) {
            lookup[channel * 256 + value] = (unsigned char)value;
        }
        return;
    }

    unsigned long long input_cumulative = 0;
    unsigned long long reference_cumulative = reference[0];
    int reference_value = 0;
    for (int input_value = 0; input_value < 256; ++input_value) {
        input_cumulative += input[input_value];
        while (
            reference_value < 255 &&
            reference_cumulative * input_total <
                input_cumulative * reference_total
        ) {
            ++reference_value;
            reference_cumulative += reference[reference_value];
        }
        lookup[channel * 256 + input_value] =
            (unsigned char)reference_value;
    }
}

extern "C" __global__ void apply_histogram_lookup(
    const unsigned char* __restrict__ image,
    const unsigned char* __restrict__ mask,
    const unsigned char* __restrict__ lookup,
    const long long pixel_count,
    const int channels,
    unsigned char* __restrict__ output
) {
    const long long pixel =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const long long base = pixel * channels;
    for (int channel = 0; channel < channels; ++channel) {
        const unsigned char value = image[base + channel];
        output[base + channel] = mask[pixel]
            ? lookup[channel * 256 + value]
            : value;
    }
}

// Feathering: derive the alpha semantics from mask, erosion, and distance fields while compositing the final image in one pass.
// Interior eroded pixels stay opaque; only the transition band consults the distance field,
// while image-border pixels remain background to preserve the CPU edge convention.
extern "C" __global__ void feather_blend(
    const unsigned char* __restrict__ image,
    const unsigned char* __restrict__ background,
    const unsigned char* __restrict__ mask,
    const unsigned char* __restrict__ eroded,
    const float* __restrict__ distance,
    const int H,
    const int W,
    const float transition,
    unsigned char* __restrict__ output
) {
    const long long pixel =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long pixel_count = (long long)H * W;
    if (pixel >= pixel_count) return;
    const int row = (int)(pixel / W);
    const int col = (int)(pixel - (long long)row * W);

    float alpha = 0.0f;
    if (mask[pixel]) {
        if (eroded[pixel]) {
            alpha = 1.0f;
        } else if (row != 0 && row != H - 1 && col != 0 && col != W - 1) {
            alpha = fminf(fmaxf((distance[pixel] - 1.0f) / transition, 0.0f), 1.0f);
        }
    }
    const float background_alpha = 1.0f - alpha;
    const long long base = pixel * 3;
    for (int channel = 0; channel < 3; ++channel) {
        const float value =
            alpha * image[base + channel] +
            background_alpha * background[base + channel];
        output[base + channel] =
            (unsigned char)min(max(__float2int_rn(value), 0), 255);
    }
}

