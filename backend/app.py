import sys
from pathlib import Path

# Add project root directory to Python's search path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Now your stage imports will work cleanly:
from ingestion.rate_extraction import run_stage_1
from classifier.modulation_pipeline import run_stage_2
from sync_gnuradio.sync import run_stage_3
# ... other imports

# backend/main.py

import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Import adapted stage runners
from ingestion.rate_extraction import run_stage_1
from classifier.modulation_pipeline import run_stage_2
from sync_gnuradio.sync import run_stage_3
from deinterleaver.deinterleaving import run_stage_4_adapted
from fec.fec import run_stage_5
from framemining.framemining import run_stage_6_adapted

app = FastAPI(title="RFGenius Analysis Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def process_signal(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_file = os.path.join(tmp_dir, file.filename)
        with open(input_file, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 1. Ingestion: load .wav or .iq, estimate fs/rs, compute PSD & waterfall
        fc32_file = os.path.join(tmp_dir, "stream.fc32")
        stage1 = run_stage_1(input_file, fc32_file)
        
        # 2. Classifier: cumulants & neural classification
        stage2 = run_stage_2(stage1["iq_data"])
        
        # 3. Sync & Slicing: Costas loop + Gardner TED -> Demodulated Bits
        stage3 = run_stage_3(
            fc32_file_path=fc32_file,
            fs=stage1["fs"],
            rs=stage1["rs"],
            mod_str=stage2["class"],
            output_dir=tmp_dir
        )
        
        # 4. Deinterleaver: detect column width & invert matrix permutation
        stage4 = run_stage_4_adapted(stage3["bits"])
        
        # 5. FEC: Viterbi / Reed-Solomon parity syndrome decoder
        stage5 = run_stage_5(stage4["output_bits"])
        
        # 6. Framemining: preamble correlation & annotated hex dump
        stage6 = run_stage_6_adapted(stage5["decoded_bits"])
        
        # Construct unified frontend contract payload
        return {
            "fs": stage1["fs"],
            "rs": stage1["rs"],
            "waterfall": stage1["waterfall"],
            "psd": stage1["psd"],
            "constellation": {
                "raw": stage3["raw_points"],
                "synced": stage3["synced_points"]
            },
            "modulation": {
                "class": stage2["class"],
                "confidence": stage2["confidence"],
                "cumulants": stage2["cumulants"]
            },
            "interleaver": {
                "type": stage4["interleaver_type"],
                "width": stage4["detected_width"],
                "confidence": stage4["confidence"],
                "rank_profile": stage4["rank_profile"]
            },
            "fec": {
                "type": stage5["type"],
                "params": stage5["params"],
                "confidence": stage5["confidence"]
            },
            "frames": {
                "length": stage6["frame_length"],
                "preamble_matches": stage6["preamble_matches"],
                "hex_dump": stage6["hex_dump"][:128] # Display first 128 rows in hex viewer
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
