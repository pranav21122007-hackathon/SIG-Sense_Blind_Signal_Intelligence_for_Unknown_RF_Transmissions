#!/usr/bin/env python3
import os
import time
import numpy as np
from gnuradio import gr, blocks, digital, channels, fec

class GroundTruthGenerator(gr.top_block):
    def __init__(self, output_filename, snr_dB=10.0, inter_cols=8, num_samples=100000):
        super().__init__("Ground Truth RF Dataset Generator")

        # --- Parameters ---
        samp_rate = 200000
        sps = 4
        inter_rows = 16
        block_len = inter_cols * inter_rows
        noise_volt = 10.0 ** (-snr_dB / 20.0)

        # --- FEC Encoder Object ---
        # Rate 1/2, K=7, Polynomials [109, 117]
        enc_cc = fec.cc_encoder_make(
            block_len * 8, 7, 2, [109, 117], 0, fec.CC_STREAMING, False
        )
        
        # Extended Encoder wrapper (Removed invalid 'threading_type' argument)
        encoder_blk = fec.extended_encoder(
            encoder_obj_list=enc_cc,
            threading=0,
            puncpat='11'
        )

        # --- Signal Processing Blocks ---
        random_bits = np.random.default_rng(1234).integers(
    0, 2, block_len * 100, dtype=np.uint8
)

        src = blocks.vector_source_b(random_bits.tolist(), repeat=True)
        head = blocks.head(gr.sizeof_gr_complex, num_samples)
        
        # Interleaver blocks
        
        transposer = blocks.matrix_interleaver(
            gr.sizeof_char,
            inter_rows,
            inter_cols,
            False
        )
        

        # Modulator & Channel
        qpsk = digital.constellation_qpsk().base()
        mod = digital.generic_mod(
            constellation=qpsk,
            differential=False,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=0.35
        )
        chan = channels.channel_model(
            noise_voltage=noise_volt, 
            frequency_offset=0.0, 
            epsilon=1.0
        )
        sink = blocks.file_sink(gr.sizeof_gr_complex, output_filename, False)

        # --- Connections ---
        self.connect(src, encoder_blk)
        self.connect(encoder_blk, transposer)
        self.connect(transposer, mod)
        self.connect(mod, chan)
        self.connect(chan, head)
        self.connect(head, sink)

def main():
    output_dir = "./rf_dataset"
    os.makedirs(output_dir, exist_ok=True)

    # 10 Parametric Variations (SNR dB, Interleaver Columns)
    variations = [
        (-5.0, 4), (-5.0, 8),
        (0.0, 4),  (0.0, 8),  (0.0, 16),
        (5.0, 8),  (5.0, 16),
        (10.0, 8), (10.0, 16),
        (15.0, 16)
    ]

    print(f"[*] Starting batch generation of {len(variations)} IQ variants...")

    for idx, (snr, cols) in enumerate(variations):
        filename = os.path.join(output_dir, f"qpsk_snr{int(snr)}dB_cols{cols}_idx{idx}.fc32")
        print(f"[{idx+1}/{len(variations)}] Generating: SNR={snr}dB, Interleaver Cols={cols} -> {filename}")
        
        tb = GroundTruthGenerator(output_filename=filename, snr_dB=snr, inter_cols=cols)
        tb.run()
        tb.wait()

    print("[+] Dataset generation complete.")

if __name__ == "__main__":
    main()