// tb_audio_core.cpp — C-sim cross-check: the synthesizable c7_audio_top must reproduce the fixed-point
// emulator (tools/c7_fixedpoint.py) for the same audio_in + video_embed. Golden vectors from
// tools/dump_hls_vectors.py. Proves the HLS audio core == the emulator that produced the 4.98 dB number.
#include "c7_types.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>

using namespace c7;

extern "C" void c7_audio_top(const sample_t *audio_in, const data_t *video_embed, sample_t *audio_out);

#ifndef VECPATH
#define VECPATH "G:/phD_Projects/AVSE-OnChip-RFSoC/hls/tb/vectors_audio.txt"
#endif

int main() {
    std::ifstream f(VECPATH);
    if (!f) { printf("ERROR: cannot open %s\n", VECPATH); return 1; }
    int nw, T, T_LAT, B;
    f >> nw >> T >> T_LAT >> B;
    printf("tb_audio_core: nw=%d T=%d T_LAT=%d B=%d\n", nw, T, T_LAT, B);

    const int VE = B * T_LAT;
    std::vector<sample_t> ain(T), aout(T);
    std::vector<data_t>   vemb(VE);
    std::vector<float>    golden(T);

    double worst = 0.0;
    int fails = 0;
    for (int wn = 0; wn < nw; wn++) {
        float v;
        for (int i = 0; i < T;  i++) { f >> v; ain[i]  = (sample_t)v; }
        for (int i = 0; i < VE; i++) { f >> v; vemb[i] = (data_t)v;  }
        for (int i = 0; i < T;  i++) { f >> v; golden[i] = v; }       // read golden

        c7_audio_top(ain.data(), vemb.data(), aout.data());

        double mx = 0.0, se = 0.0, sg = 0.0;
        int amax = 0;
        for (int i = 0; i < T; i++) {
            double o = (double)aout[i], g = (double)golden[i];
            double d = std::fabs(o - g);
            if (d > mx) { mx = d; amax = i; }
            se += d * d; sg += g * g;
        }
        double rmsd = std::sqrt(se / T), rmsg = std::sqrt(sg / T);
        double rel = rmsd / rmsg, lsb = mx / 0.001953125;   // data_t LSB = 2^-9
        printf("  win%d: rel_rms=%.3e  max|diff|=%.3e (%.1f data_t LSB @i=%d: hls=%.5f gold=%.5f) rms=%.3e\n",
               wn, rel, mx, lsb, amax, (double)aout[amax], (double)golden[amax], rmsg);
        if (rel > worst) worst = rel;
        if (rel > 1e-2) fails++;   // faithful = aggregate rel-rms within 1% (float-emu vs fixed-pt jitter ~1 LSB)
    }
    printf("RESULT: worst rel_rms=%.3e over %d windows -> %s\n",
           worst, nw, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
