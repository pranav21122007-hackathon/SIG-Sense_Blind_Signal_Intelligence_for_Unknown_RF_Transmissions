import os
import json
import math
import numpy as np
from gnuradio import gr, blocks, digital, channels, filter

def conv_encode_k7_r12(bits):
    """
    Standard NASA/Voyager Rate 1/2, K=7 Convolutional Encoder.
    Generators in octal: G1 = 133_8 (0x5B / 109), G2 = 171_8 (0x79 / 79)
    """
    g1 = [1, 0, 1, 1, 0, 1, 1]  # 133 octal
    g2 = [1, 1, 1, 1, 0, 0, 1]  # 171 octal
    
    reg = [0] * 7
    encoded = []
    
    for b in bits:
        reg = [b] + reg[:-1]
        out1 = sum(r * g for r, g in zip(reg, g1)) % 2
        out2 = sum(r * g for r, g in zip(reg, g2)) % 2
        encoded.extend([out1, out2])
        
    return np.array(encoded, dtype=np.uint8)

def matrix_interleave(bits, rows, cols):
    """
    Classic Block/Matrix Interleaver:
    Fills row-by-row, reads out column-by-column.
    """
    block_size = rows * cols
    # Pad to integer multiple of block_size if necessary
    pad_len = (block_size - (len(bits) % block_size)) % block_size
    if pad_len > 0:
        bits = np.pad(bits, (0, pad_len), mode='constant', constant_values=0)
    
    num_blocks = len(bits) // block_size
    interleaved = []
    
    for i in range(num_blocks):
        block = bits[i * block_size : (i + 1) * block_size].reshape((rows, cols))
        interleaved.extend(block.T.flatten())
        
    return np.array(interleaved, dtype=np.uint8)


class GroundTruthGenerator(gr.top_block):
    def __init__(self, filename, snr_db, interleaved_bits, sps=4, num_samples=300000):
        super(GroundTruthGenerator, self).__init__("GroundTruthGenerator")

        samp_rate = 200000
        symbol_rate = int(samp_rate / sps)

        # 1. Stream the pre-encoded, interleaved bits
        src = blocks.vector_source_b(interleaved_bits.tolist(), repeat=True)

        # 2. Pack 2 bits per QPSK symbol
        pack_qpsk = blocks.pack_k_bits_bb(2)

        # 3. Chunks to Symbols (Gray-coded QPSK constellation)
        qpsk_constellation = [
            -0.707 - 0.707j,  # 00
            -0.707 + 0.707j,  # 01
             0.707 - 0.707j,  # 10
             0.707 + 0.707j   # 11
        ]
        mapper = digital.chunks_to_symbols_bc(qpsk_constellation, 1)

        # 4. Pulse Shaping Filter (RRC)
        rrc_taps = filter.firdes.root_raised_cosine(
            1.0, samp_rate, symbol_rate, 0.35, 11 * sps
        )
        pulse_shaper = filter.interp_fir_filter_ccf(sps, rrc_taps)

        # 5. Calibrated AWGN Channel
        noise_pwr = 10.0 ** (-snr_db / 10.0)
        noise_volts = math.sqrt(noise_pwr * (sps / 2.0))
        channel = channels.channel_model(
            noise_voltage=noise_volts,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0 + 0.0j],
            noise_seed=int(np.random.randint(0, 65535))
        )

        # 6. Truncate to fixed length and save to raw .IQ (complex64)
        head = blocks.head(gr.sizeof_gr_complex, num_samples)
        sink = blocks.file_sink(gr.sizeof_gr_complex, filename, False)

        # Connections
        self.connect(src, pack_qpsk, mapper, pulse_shaper, channel, head, sink)


def run_batch():
    output_dir = "./synthetic_dataset"
    os.makedirs(output_dir, exist_ok=True)

    snr_levels = [0, 5, 10, 15, 20]
    interleaver_widths = [4, 8]  # rows are fixed at 16

    configs = []
    for col in interleaver_widths:
        for snr in snr_levels:
            configs.append({"snr_db": snr, "cols": col, "rows": 16})

    metadata_log = []

    # Generate a fixed sequence of raw random bits
    np.random.seed(42)
    raw_bits = np.random.randint(0, 2, size=50000, dtype=np.uint8)

    # Encode with Rate 1/2, K=7
    encoded_bits = conv_encode_k7_r12(raw_bits)

    for i, cfg in enumerate(configs):
        iq_filename = os.path.join(output_dir, f"signal_idx{i}_snr{cfg['snr_db']}dB_cols{cfg['cols']}.iq")
        print(f"Generating variant {i+1}/10: SNR={cfg['snr_db']} dB, Columns={cfg['cols']} -> {iq_filename}")

        # Interleave with the specific matrix dimensions
        interleaved_bits = matrix_interleave(encoded_bits, cfg['rows'], cfg['cols'])

        tb = GroundTruthGenerator(
            filename=iq_filename,
            snr_db=cfg['snr_db'],
            interleaved_bits=interleaved_bits,
            num_samples=300000
        )
        tb.run()
        tb.stop()
        tb.wait()

        metadata_log.append({
            "file": os.path.basename(iq_filename),
            "snr_db": cfg['snr_db'],
            "interleaver_rows": cfg['rows'],
            "interleaver_cols": cfg['cols'],
            "fec_type": "Conv_K7_R1/2",
            "poly": [109, 79],
            "modulation": "QPSK",
            "baud_rate": 50000,
            "sample_rate": 200000,
            "dtype": "complex64"
        })

    with open(os.path.join(output_dir, "ground_truth_manifest.json"), "w") as f:
        json.dump(metadata_log, f, indent=4)

    print("\nSuccess: All 10 variants generated cleanly in ./synthetic_dataset/")

if __name__ == "__main__":
    run_batch()