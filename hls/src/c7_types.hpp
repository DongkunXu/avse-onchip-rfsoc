// c7_types.hpp — fixed-point types for the C7 (Conv-TasNet-style) HLS fit check.
// int16 activations/weights (ap_fixed<16,*>), wide accumulator. Adapted from the reference
// project's common/types.hpp (precision locked at int16 per DECISIONS D-3).
#ifndef C7_TYPES_HPP
#define C7_TYPES_HPP

#include <ap_fixed.h>
#include <ap_int.h>

namespace c7 {

// Activations / feature maps: ±64 range, ~10-bit effective precision (matches reference data_t).
typedef ap_fixed<16, 7, AP_TRN, AP_SAT> data_t;
// Conv / linear (MAC) weight operands after BN folding.
typedef ap_fixed<16, 5> wgt_t;
// Inline BN / in_norm affine (s,b), PReLU slopes and folded conv biases. WIDE on purpose: the in_norm
// fold scale reaches ~102 on low-variance encoder channels (overflows wgt_t's +-16), and these are
// per-channel constants, not the systolic-array operands the int16 lock (D-3) targets. See DECISIONS D-18.
typedef ap_fixed<32, 16> bn_t;
// Wide accumulator (DSP48E2 native), saturating.
typedef ap_fixed<48, 22, AP_TRN, AP_SAT> acc_t;
// External PCM sample (I/O boundary): ±1.0.
typedef ap_fixed<16, 1> sample_t;

namespace cfg {
// Task / window
constexpr int T      = 19200;   // 1.2 s @ 16 kHz
constexpr int L      = 32;      // encoder/decoder kernel (samples)
constexpr int STRIDE = 16;      // L/2
constexpr int T_LAT  = 1201;    // Conv1d(T, k=L, s=STRIDE, pad=STRIDE) -> floor(T/STRIDE)+1 = 1201
                                // (single latent resolution — no U-Net skips)
// Channels
constexpr int N   = 128;        // encoder filters / latent channels
constexpr int B   = 64;         // bottleneck channels
constexpr int H   = 128;        // TCN conv channels
constexpr int X   = 5;          // dilated blocks per repeat (dilation 1,2,4,8,16)
constexpr int R   = 2;          // repeats
constexpr int NBLK = X * R;     // 10 TCN blocks
constexpr int KD  = 3;          // depthwise kernel
}  // namespace cfg

}  // namespace c7

#endif  // C7_TYPES_HPP
