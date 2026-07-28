// CuPy CUDA kernels for the morphing backend.
// Kernel arguments intentionally use flat POD arrays so launches avoid temporary host-side structures.

/**
 * Reads one channel from a three-channel image, returning a constant zero outside the image.
 *
 * Args:
 *     image: Contiguous row-major uint8 input [H][W][3] with interleaved channels.
 *     row: Scalar row index.
 *     col: Scalar column index.
 *     channel: Scalar channel index in [0, 2].
 *     H: Scalar image height and row bound.
 *     W: Scalar image width, row stride, and column bound.
 *
 * Returns:
 *     float: The selected sample, or 0.0f when (row, col) is out of bounds.
 */
__device__ __forceinline__ float constant_sample(
    const unsigned char* image,
    const int row,
    const int col,
    const int channel,
    const int H,
    const int W
) {
    if (row < 0 || row >= H || col < 0 || col >= W) {
        return 0.0f;
    }
    return (float)image[(row * W + col) * 3 + channel];
}

/**
 * Bilinearly samples a three-channel image at a continuous coordinate using a constant-zero border.
 *
 * Args:
 *     image: Contiguous row-major uint8 input [H][W][3] with interleaved channels.
 *     H: Scalar image height and row bound.
 *     W: Scalar image width, row stride, and column bound.
 *     row: Continuous row coordinate.
 *     col: Continuous column coordinate.
 *     channel: Scalar channel index in [0, 2].
 *
 * Returns:
 *     float: The bilinearly interpolated channel value, including zero-weighted out-of-bounds neighbors.
 */
__device__ __forceinline__ float bilinear_interpolation(
    const unsigned char* image,
    const int H,
    const int W,
    const float row,
    const float col,
    const int channel
) {
    const int row0 = (int)floorf(row);
    const int col0 = (int)floorf(col);
    const float row_fraction = row - row0;
    const float col_fraction = col - col0;
    const float value00 = constant_sample(image, row0, col0, channel, H, W);
    const float value01 = constant_sample(image, row0, col0 + 1, channel, H, W);
    const float value10 = constant_sample(image, row0 + 1, col0, channel, H, W);
    const float value11 = constant_sample(image, row0 + 1, col0 + 1, channel, H, W);
    return (1.0f - row_fraction) * ((1.0f - col_fraction) * value00 + col_fraction * value01)
        + row_fraction * ((1.0f - col_fraction) * value10 + col_fraction * value11);
}

/**
 * Inverse-warps a three-channel image with bilinear interpolation and a constant-zero source border.
 *
 * Args:
 *     image: Contiguous row-major uint8 input [H][W][3] with interleaved channels.
 *     H: Scalar source height and row bound.
 *     W: Scalar source width, row stride, and column bound.
 *     output_H: Scalar output height and row count.
 *     output_W: Scalar output width and row stride.
 *     m00: Scalar element (0, 0) of the row-major 2x3 output-to-source affine matrix.
 *     m01: Scalar element (0, 1) of the row-major 2x3 output-to-source affine matrix.
 *     m02: Scalar element (0, 2) of the row-major 2x3 output-to-source affine matrix.
 *     m10: Scalar element (1, 0) of the row-major 2x3 output-to-source affine matrix.
 *     m11: Scalar element (1, 1) of the row-major 2x3 output-to-source affine matrix.
 *     m12: Scalar element (1, 2) of the row-major 2x3 output-to-source affine matrix.
 *     output: Contiguous row-major uint8 output [output_H][output_W][3] with interleaved channels.
 *
 * Returns:
 *     void: Results are written to output, one three-channel pixel per valid thread.
 */
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
    const long long pixel = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long pixel_count = (long long) output_H * output_W;
    // Early exit for threads that are out of bounds
    if (pixel >= pixel_count) {
        return;
    }

    // Find 2D coordinates of the output pixel and compute the corresponding source coordinates
    const int row = (int) (pixel / output_W);
    const int col = (int) (pixel - (long long) row * output_W);
    const float source_col = m00 * col + m01 * row + m02;
    const float source_row = m10 * col + m11 * row + m12;
    const long long base = pixel * 3;

    for (int channel = 0; channel < 3; channel++) {
        const float value = bilinear_interpolation(image, H, W, source_row, source_col, channel);
        output[base + channel] =
            (unsigned char)min(max(__float2int_rn(value), 0), 255);
    }
}

/**
 * Resizes a three-channel image by integrating source-pixel coverage over each output-pixel footprint.
 *
 * Args:
 *     image: Contiguous row-major uint8 input [H][W][3] with interleaved channels.
 *     H: Scalar source height and row bound.
 *     W: Scalar source width, row stride, and column bound.
 *     output_H: Scalar output height used to derive the vertical sampling scale.
 *     output_W: Scalar output width and row stride.
 *     output: Contiguous row-major uint8 output [output_H][output_W][3] with interleaved channels.
 *
 * Returns:
 *     void: Area-averaged pixels are written to output.
 */
extern "C" __global__ void area_resize_uint8(
    const unsigned char* __restrict__ image,
    const int H,
    const int W,
    const int output_H,
    const int output_W,
    unsigned char* __restrict__ output
) {
    const long long pixel = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long pixel_count = (long long) output_H * output_W;
    // Early exit for threads that are out of bounds
    if (pixel >= pixel_count) {
        return;
    }

    const int output_row = (int) (pixel / output_W);
    const int output_col = (int) (pixel - (long long) output_row * output_W);
    const float scale_y = (float) H / output_H;
    const float scale_x = (float) W / output_W;
    const float start_y = output_row * scale_y;
    const float end_y = (output_row + 1) * scale_y;
    const float start_x = output_col * scale_x;
    const float end_x = (output_col + 1) * scale_x;
    const int first_row = (int) floorf(start_y);
    const int last_row = min((int) ceilf(end_y) - 1, H - 1);
    const int first_col = (int) floorf(start_x);
    const int last_col = min((int) ceilf(end_x) - 1, W - 1);
    float sums[3] = {0.0f, 0.0f, 0.0f};

    for (int row = first_row; row <= last_row; row++) {
        const float row_weight = fminf(end_y, row + 1.0f) - fmaxf(start_y, (float) row);
        for (int col = first_col; col <= last_col; col++) {
            const float col_weight = fminf(end_x, col + 1.0f) - fmaxf(start_x, (float) col);
            const float weight = row_weight * col_weight;
            const long long source = ((long long) row * W + col) * 3;
            for (int channel = 0; channel < 3; channel++) {
                sums[channel] += weight * image[source + channel];
            }
        }
    }
    const float inverse_area = 1.0f / (scale_x * scale_y);
    const long long destination = pixel * 3;
    for (int channel = 0; channel < 3; channel++) {
        output[destination + channel] = (unsigned char)min(
            max(__float2int_rn(sums[channel] * inverse_area), 0),
            255
        );
    }
}

/**
 * Resolves a flattened bounding-box work item to its descriptor and image coordinates.
 *
 * Args:
 *     descriptors: Contiguous row-major float descriptor input with descriptor_stride values per row.
 *     offsets: Contiguous int64 input [descriptor_count + 1] containing exclusive prefix sums of box areas.
 *     descriptor_count: Scalar number of descriptor rows searched in offsets.
 *     descriptor_stride: Scalar number of float values in each descriptor row.
 *     box_offset: Scalar index of box x within a descriptor; box y and width follow it.
 *     work_index: Scalar flattened index into the concatenated descriptor bounding boxes.
 *     descriptor_index: Optional scalar output receiving the descriptor row; may be null when unused.
 *     row: Scalar output receiving the resolved image row.
 *     col: Scalar output receiving the resolved image column.
 *
 * Returns:
 *     const float*: Pointer to the owning descriptor row within descriptors.
 */
__device__ __forceinline__ const float* resolve_box_work_item(
    const float* descriptors,
    const long long* offsets,
    const int descriptor_count,
    const int descriptor_stride,
    const int box_offset,
    const long long work_index,
    int* descriptor_index,
    int* row,
    int* col
) {
    int low = 0;
    int high = descriptor_count;
    while (low + 1 < high) {
        const int middle = low + (high - low) / 2;
        if (offsets[middle] <= work_index) {
            low = middle;
        } else {
            high = middle;
        }
    }
    const float* descriptor = descriptors + low * descriptor_stride;
    const int box_x = (int)descriptor[box_offset];
    const int box_y = (int)descriptor[box_offset + 1];
    const int box_width = (int)descriptor[box_offset + 2];
    const long long local = work_index - offsets[low];
    if (descriptor_index) {
        *descriptor_index = low;
    }
    *col = box_x + (int)(local % box_width);
    *row = box_y + (int)(local / box_width);
    return descriptor;
}

/**
 * Tests whether a pixel lies inside a counter-clockwise triangle under the requested edge-ownership policy.
 *
 * Args:
 *     descriptor: Contiguous float input whose first six values are ax, ay, bx, by, cx, cy.
 *     row: Scalar pixel row used as the point's y coordinate.
 *     col: Scalar pixel column used as the point's x coordinate.
 *     inclusive_edges: Scalar flag accepting all boundary edges when true; otherwise applies the top-left rule.
 *
 * Returns:
 *     bool: True when the pixel is inside the triangle and owns any boundary edge under the selected policy.
 */
__device__ __forceinline__ bool is_inside_triangle(
    const float* descriptor,
    const int row,
    const int col,
    const bool inclusive_edges
) {
    const float ax = descriptor[0], ay = descriptor[1];
    const float bx = descriptor[2], by = descriptor[3];
    const float cx = descriptor[4], cy = descriptor[5];
    const float px = (float)col, py = (float)row;
    const float w0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    const float w1 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
    const float w2 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
    if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) {
        return false;
    }
    if (inclusive_edges) {
        return true;
    }

    const float ab_dx = bx - ax, ab_dy = by - ay;
    const float bc_dx = cx - bx, bc_dy = cy - by;
    const float ca_dx = ax - cx, ca_dy = ay - cy;
    const bool ab_top_left = ab_dy < 0.0f || (ab_dy == 0.0f && ab_dx > 0.0f);
    const bool bc_top_left = bc_dy < 0.0f || (bc_dy == 0.0f && bc_dx > 0.0f);
    const bool ca_top_left = ca_dy < 0.0f || (ca_dy == 0.0f && ca_dx > 0.0f);
    return (w0 > 0.0f || ab_top_left) &&
        (w1 > 0.0f || bc_top_left) &&
        (w2 > 0.0f || ca_top_left);
}

/**
 * Rasterizes counter-clockwise triangles into a binary mask from a compact list of bounding-box pixels.
 *
 * Args:
 *     descriptors: Row-major float input [triangle_count][10]: ax, ay, bx, by, cx, cy, box x, box y,
 *         box width, box height.
 *     offsets: Contiguous int64 input [triangle_count + 1] containing exclusive prefix sums of box areas.
 *     triangle_count: Scalar descriptor-row count used to search offsets.
 *     W: Scalar mask width and output row stride.
 *     work_count: Scalar total bounding-box area, equal to offsets[triangle_count].
 *     output: Contiguous row-major uint8 in/out mask [height][W], zero-initialized before the kernel runs.
 *
 * Returns:
 *     void: Covered mask elements are set to 1; all other elements remain unchanged.
 */
extern "C" __global__ void rasterize_convex_hull(
    const float* __restrict__ descriptors,
    const long long* __restrict__ offsets,
    const int triangle_count,
    const int W,
    const long long work_count,
    unsigned char* __restrict__ output
) {
    const long long work_index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    // Early exit for threads that are out of bounds
    if (work_index >= work_count) {
        return;
    }

    int row;
    int col;
    const float* descriptor = resolve_box_work_item(
        descriptors, offsets, triangle_count, 10, 6, work_index, 0, &row, &col
    );
    if (is_inside_triangle(descriptor, row, col, true)) {
        output[row * W + col] = 1;
    }
}

/**
 * Maps a continuous coordinate into an image extent using OpenCV-style BORDER_REFLECT_101.
 *
 * Args:
 *     value: Scalar coordinate that may lie outside the valid extent.
 *     size: Scalar extent length that defines the reflection period.
 *
 * Returns:
 *     float: The coordinate reflected into [0, size - 1], or 0.0f when size is at most one.
 */
__device__ __forceinline__ float reflect101(float value, const int size) {
    if (size <= 1) {
        return 0.0f;
    }
    const float period = 2.0f * (size - 1);
    value = fmodf(fabsf(value), period);
    return value > size - 1 ? period - value : value;
}

/**
 * Reflects a source coordinate, bilinearly samples all three channels, and writes one output pixel.
 *
 * Args:
 *     image: Contiguous row-major uint8 input [H][W][3] with interleaved channels.
 *     H: Scalar image height and source row bound.
 *     W: Scalar image width, source row stride, and column bound.
 *     source_row: Continuous source row reflected with BORDER_REFLECT_101 before sampling.
 *     source_col: Continuous source column reflected with BORDER_REFLECT_101 before sampling.
 *     output_index: Scalar byte offset of the destination pixel in output.
 *     output: Contiguous row-major uint8 output with three interleaved channels per pixel.
 *
 * Returns:
 *     void: Three rounded and clamped channel values are written at output_index.
 */
__device__ __forceinline__ void write_reflected_bilinear_pixel(
    const unsigned char* image,
    const int H,
    const int W,
    float source_row,
    float source_col,
    const long long output_index,
    unsigned char* output
) {
    source_row = reflect101(source_row, H);
    source_col = reflect101(source_col, W);
    for (int channel = 0; channel < 3; ++channel) {
        const float value = bilinear_interpolation(image, H, W, source_row, source_col, channel);
        output[output_index + channel] =
            (unsigned char)min(max(__float2int_rn(value), 0), 255);
    }
}

/**
 * Rasterizes non-overlapping target triangles and bilinearly samples their affine-mapped source pixels.
 *
 * Args:
 *     image: Contiguous row-major uint8 input [H][W][3] with interleaved channels.
 *     descriptors: Row-major float input [triangle_count][16]: target ax, ay, bx, by, cx, cy; inverse affine
 *         m00, m01, m02, m10, m11, m12; then box x, box y, box width, box height.
 *     offsets: Contiguous int64 input [triangle_count + 1] containing exclusive prefix sums of box areas.
 *     triangle_count: Scalar descriptor-row count used to search offsets.
 *     H: Scalar image height and row bound.
 *     W: Scalar image width and row stride.
 *     work_count: Scalar total bounding-box area, equal to offsets[triangle_count].
 *     output: Contiguous row-major uint8 output [H][W][3], zero-initialized for uncovered pixels.
 *
 * Returns:
 *     void: Each covered target pixel receives one interleaved, bilinearly sampled channel triplet.
 */
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
    const long long work_index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (work_index >= work_count) {
        return;
    }

    int row;
    int col;
    const float* descriptor = resolve_box_work_item(
        descriptors, offsets, triangle_count, 16, 12, work_index, 0, &row, &col
    );
    const bool image_boundary =
        col == 0 || col == W - 1 || row == 0 || row == H - 1;
    if (!is_inside_triangle(descriptor, row, col, image_boundary)) {
        return;
    }

    const float source_col = descriptor[6] * col + descriptor[7] * row + descriptor[8];
    const float source_row = descriptor[9] * col + descriptor[10] * row + descriptor[11];
    const long long output_index = ((long long)row * W + col) * 3;
    write_reflected_bilinear_pixel(image, H, W, source_row, source_col, output_index, output);
}

/**
 * Assigns each covered pixel to the highest-index overlapping triangle for deterministic ownership.
 *
 * Args:
 *     descriptors: Row-major float input [triangle_count][16]: six target coordinates, six inverse-affine
 *         coefficients, then box x, box y, box width, box height.
 *     offsets: Contiguous int64 input [triangle_count + 1] containing exclusive prefix sums of box areas.
 *     triangle_count: Scalar descriptor-row count used to search offsets.
 *     W: Scalar membership-map width and row stride.
 *     work_count: Scalar total bounding-box area, equal to offsets[triangle_count].
 *     membership: Contiguous row-major int32 in/out map [height][W], initialized to -1 and updated atomically.
 *
 * Returns:
 *     void: Covered entries retain the largest triangle index, matching sequential overwrite order.
 */
extern "C" __global__ void build_box_membership(
    const float* __restrict__ descriptors,
    const long long* __restrict__ offsets,
    const int triangle_count,
    const int W,
    const long long work_count,
    int* __restrict__ membership
) {
    const long long work_index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (work_index >= work_count) {
        return;
    }
    int triangle;
    int row;
    int col;
    const float* descriptor = resolve_box_work_item(
        descriptors, offsets, triangle_count, 16, 12, work_index, &triangle, &row, &col
    );
    if (is_inside_triangle(descriptor, row, col, true)) {
        atomicMax(&membership[(long long)row * W + col], triangle);
    }
}

/**
 * Samples one source pixel for every output pixel according to a precomputed triangle ownership map.
 *
 * Args:
 *     image: Contiguous row-major uint8 input [H][W][3] with interleaved channels.
 *     descriptors: Row-major float input [triangle_count][16]: six target coordinates, six inverse-affine
 *         coefficients, then box x, box y, box width, box height.
 *     membership: Contiguous row-major int32 input [H][W]; -1 means no triangle owns the pixel.
 *     H: Scalar image height used for source reflection and output bounds.
 *     W: Scalar image width and row stride for image and membership buffers.
 *     output: Contiguous row-major uint8 output [H][W][3] with interleaved channels.
 *
 * Returns:
 *     void: Owned pixels are sampled bilinearly; unowned pixels are written as three zero bytes.
 */
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
    if (pixel >= pixel_count) {
        return;
    }
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
    const float source_col =
        descriptor[6] * col + descriptor[7] * row + descriptor[8];
    const float source_row =
        descriptor[9] * col + descriptor[10] * row + descriptor[11];
    write_reflected_bilinear_pixel(image, H, W, source_row, source_col, output_index, output);
}

/**
 * Alpha-composites one interleaved three-channel pixel and writes rounded, clamped uint8 values.
 *
 * Args:
 *     foreground: Contiguous row-major uint8 foreground input [pixel_count][3].
 *     background: Contiguous row-major uint8 background input [pixel_count][3].
 *     pixel: Scalar pixel index shared by foreground, background, and output.
 *     foreground_alpha: Scalar foreground weight; the background weight is 1.0f - foreground_alpha.
 *     output: Contiguous row-major uint8 output [pixel_count][3] with interleaved channels.
 *
 * Returns:
 *     void: The three composited channel values are written to output at pixel.
 */
__device__ __forceinline__ void composite_three_channel_pixel(
    const unsigned char* foreground,
    const unsigned char* background,
    const long long pixel,
    const float foreground_alpha,
    unsigned char* output
) {
    const float background_alpha = 1.0f - foreground_alpha;
    const long long base = pixel * 3;
    for (int channel = 0; channel < 3; ++channel) {
        const float value =
            foreground_alpha * foreground[base + channel] +
            background_alpha * background[base + channel];
        output[base + channel] =
            (unsigned char)min(max(__float2int_rn(value), 0), 255);
    }
}

/**
 * Blends two three-channel images using either per-pixel alpha values or a single scalar factor.
 *
 * Args:
 *     image1: Contiguous row-major uint8 input [pixel_count][3] with interleaved channels.
 *     image2: Contiguous row-major uint8 input [pixel_count][3] with the same layout as image1.
 *     alpha: Contiguous float32 input [pixel_count]; read only when use_alpha is nonzero.
 *     pixel_count: Scalar number of pixels in each image and alpha entries when alpha is enabled.
 *     factor: Scalar image2 weight when use_alpha is zero; image1 then uses 1.0f - factor.
 *     use_alpha: Scalar flag selecting the alpha buffer instead of factor.
 *     output: Contiguous row-major uint8 output [pixel_count][3] with interleaved channels.
 *
 * Returns:
 *     void: Rounded and clamped blended channel values are written to output.
 */
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
    if (pixel >= pixel_count) {
        return;
    }

    const float foreground = use_alpha ? alpha[pixel] : 1.0f - factor;
    composite_three_channel_pixel(image1, image2, pixel, foreground, output);
}

/**
 * Converts OpenCV-style 8-bit BGR pixels to OpenCV-style 8-bit HLS pixels.
 *
 * Args:
 *     input: Contiguous row-major uint8 input [pixel_count][3] with interleaved B, G, R channels.
 *     pixel_count: Scalar number of three-channel pixels in input and output.
 *     output: Contiguous row-major uint8 output [pixel_count][3] with interleaved H, L, S channels.
 *
 * Returns:
 *     void: Converted HLS pixels are written to output, with hue encoded in [0, 179].
 */
extern "C" __global__ void bgr_to_hls(
    const unsigned char* __restrict__ input,
    const long long pixel_count,
    unsigned char* __restrict__ output
) {
    const long long pixel =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
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

/**
 * Converts OpenCV-style 8-bit HLS pixels to OpenCV-style 8-bit BGR pixels.
 *
 * Args:
 *     input: Contiguous row-major uint8 input [pixel_count][3] with interleaved H, L, S channels.
 *     pixel_count: Scalar number of three-channel pixels in input and output.
 *     output: Contiguous row-major uint8 output [pixel_count][3] with interleaved B, G, R channels.
 *
 * Returns:
 *     void: Rounded and clamped BGR pixels are written to output.
 */
extern "C" __global__ void hls_to_bgr(
    const unsigned char* __restrict__ input,
    const long long pixel_count,
    unsigned char* __restrict__ output
) {
    const long long pixel =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
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

/**
 * Accumulates one channel of masked input and reference histograms per block using a fixed shared buffer.
 *
 * Args:
 *     image: Contiguous row-major uint8 input [input_pixel_count][channels] with interleaved channels.
 *     mask: Contiguous uint8 input [input_pixel_count], with nonzero entries selecting image pixels.
 *     reference: Contiguous row-major uint8 input [reference_pixel_count][channels].
 *     reference_mask: Contiguous uint8 input [reference_pixel_count], selecting reference pixels.
 *     input_pixel_count: Scalar number of pixels represented by image and mask.
 *     reference_pixel_count: Scalar number of pixels represented by reference and reference_mask.
 *     channels: Scalar channel count, innermost image/reference stride, and valid blockIdx.y range.
 *     histograms: Contiguous uint32 in/out buffer [channels][2][256], channel-major with input bins first;
 *         callers zero-initialize it before blocks atomically merge their local counts.
 *
 * Returns:
 *     void: Histogram counts are atomically accumulated into histograms.
 */
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
    __shared__ unsigned int local_histograms[2 * 256];
    const int channel = blockIdx.y;
    if (channel >= channels) {
        return;
    }
    for (int index = threadIdx.x; index < 2 * 256; index += blockDim.x) {
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
            atomicAdd(&local_histograms[image[base + channel]], 1U);
        }
    }
    for (
        long long pixel = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        pixel < reference_pixel_count;
        pixel += (long long)gridDim.x * blockDim.x
    ) {
        const long long base = pixel * channels;
        if (reference_mask[pixel]) {
            atomicAdd(&local_histograms[256 + reference[base + channel]], 1U);
        }
    }
    __syncthreads();

    unsigned int* channel_histograms = histograms + channel * 2 * 256;
    for (int index = threadIdx.x; index < 2 * 256; index += blockDim.x) {
        atomicAdd(&channel_histograms[index], local_histograms[index]);
    }
}

/**
 * Builds one monotone byte-to-byte histogram-matching lookup table per channel.
 *
 * Args:
 *     histograms: Contiguous channel-major uint32 input [channels][2][256], with input bins before reference bins.
 *     channels: Scalar number of histogram rows and lookup tables.
 *     lookup: Contiguous row-major uint8 output [channels][256], indexed by channel then input byte value.
 *
 * Returns:
 *     void: Each block's first thread writes one channel lookup, or the identity map for an empty histogram.
 */
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

/**
 * Applies channel-specific histogram lookup tables to masked pixels and copies unmasked values unchanged.
 *
 * Args:
 *     image: Contiguous row-major uint8 input [pixel_count][channels] with interleaved channels.
 *     mask: Contiguous uint8 input [pixel_count], with nonzero entries selecting transformed pixels.
 *     lookup: Contiguous row-major uint8 input [channels][256], indexed by channel then source byte value.
 *     pixel_count: Scalar number of pixels represented by image, mask, and output.
 *     channels: Scalar channel count and innermost image/output stride.
 *     output: Contiguous row-major uint8 output [pixel_count][channels] with interleaved channels.
 *
 * Returns:
 *     void: Mapped or copied channel values are written to output.
 */
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

/**
 * Feather-blends an image over a background using mask erosion and a distance-based transition band.
 *
 * Args:
 *     image: Contiguous row-major uint8 foreground [H][W][3] with interleaved channels.
 *     background: Contiguous row-major uint8 background [H][W][3] with the same layout as image.
 *     mask: Contiguous row-major uint8 selection mask [H][W]; nonzero pixels may receive foreground.
 *     eroded: Contiguous row-major uint8 eroded mask [H][W]; nonzero pixels receive alpha 1.
 *     distance: Contiguous row-major float32 distance field [H][W], sampled only in the transition band.
 *     H: Scalar height shared by all image, mask, distance, and output buffers.
 *     W: Scalar width and row stride shared by all image, mask, distance, and output buffers.
 *     transition: Scalar distance range used to normalize transition-band alpha.
 *     output: Contiguous row-major uint8 result [H][W][3] with interleaved channels.
 *
 * Returns:
 *     void: Rounded and clamped feathered pixels are written to output.
 */
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
    composite_three_channel_pixel(image, background, pixel, alpha, output);
}
