// tb_avse.cpp — end-to-end C-sim cross-check: the monolithic c7_avse_top (video encoder + audio core)
// must reproduce the fixed-point emulator for the same audio + raw video. Golden from dump_hls_vectors.py.
#include "c7_types.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>

using namespace c7;

extern "C" void c7_avse_top(const sample_t *audio_in, const data_t *video_in, sample_t *audio_out);

#ifndef VECPATH
#define VECPATH "G:/phD_Projects/AVSE-OnChip-RFSoC/hls/tb/vectors_full.txt"
#endif

int main() {
    std::ifstream f(VECPATH);
    if (!f) { printf("ERROR: cannot open %s\n", VECPATH); return 1; }
    int nw, T, TF, IN;
    f >> nw >> T >> TF >> IN;
    printf("tb_avse: nw=%d T=%d TF=%d IN=%d\n", nw, T, TF, IN);

    const int VID = TF * IN * IN;
    std::vector<sample_t> ain(T), aout(T);
    std::vector<data_t>   vin(VID);
    std::vector<float>    golden(T);

    double worst = 0.0;
    int fails = 0;
    for (int wn = 0; wn < nw; wn++) {
        float v;
        for (int i = 0; i < T;   i++) { f >> v; ain[i] = (sample_t)v; }
        for (int i = 0; i < VID; i++) { f >> v; vin[i] = (data_t)v;  }   // [0,1] -> data_t (= emulator's input quant)
        for (int i = 0; i < T;   i++) { f >> v; golden[i] = v; }

        c7_avse_top(ain.data(), vin.data(), aout.data());

        double mx = 0.0, se = 0.0, sg = 0.0; int amax = 0;
        for (int i = 0; i < T; i++) {
            double o = (double)aout[i], g = (double)golden[i], d = std::fabs(o - g);
            if (d > mx) { mx = d; amax = i; }
            se += d * d; sg += g * g;
        }
        double rmsd = std::sqrt(se / T), rmsg = std::sqrt(sg / T), rel = rmsd / rmsg;
        printf("  win%d: rel_rms=%.3e  max|diff|=%.3e (%.1f LSB @i=%d: hls=%.5f gold=%.5f) rms=%.3e\n",
               wn, rel, mx, mx / 0.001953125, amax, (double)aout[amax], (double)golden[amax], rmsg);
        if (rel > worst) worst = rel;
        if (rel > 1e-2) fails++;
    }
    printf("RESULT: worst rel_rms=%.3e over %d windows -> %s\n", worst, nw, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
