#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil


def run_model1(suffix):
    print("→ Running Model 1 pipeline…")
    # invoke your existing script (must be in same dir)
    subprocess.check_call([sys.executable, "model1_pipeline.py"])
    # move & rename its outputs
    os.makedirs("trained_models", exist_ok=True)
    # pipeline writes w1_10k_noextctx.pkl and predictions1_noextctx.wtag
    shutil.copy("w1_10k_noextctx.pkl", f"trained_models/weights_1.pkl")
    shutil.move("predictions1_noextctx.wtag",
                f"comp_m1_{suffix}.wtag")
    print("  → comp_m1_{}.wtag".format(suffix))
    print("  → trained_models/weights_1.pkl\n")

def run_model2(suffix):
    print("→ Running Model 2 pipeline…")
    subprocess.check_call([sys.executable, "model2_pipeline.py"])
    os.makedirs("trained_models", exist_ok=True)
    # pipeline writes weights_5k_selftrain.pkl and predictions2_final.wtag
    shutil.copy("weights_5k_selftrain.pkl", f"trained_models/weights_2.pkl")
    shutil.move("predictions2_final.wtag",
                f"comp_m2_{suffix}.wtag")
    print("  → comp_m2_{}.wtag".format(suffix))
    print("  → trained_models/weights_2.pkl\n")

def main():
    ids = "207476763_322409376"
    run_model1(ids)
    run_model2(ids)

    print("Done. You now have:")
    print("  trained_models/weights_1.pkl")
    print("  trained_models/weights_2.pkl")
    print(f"  comp_m1_{ids}.wtag")
    print(f"  comp_m2_{ids}.wtag")

if __name__ == "__main__":
    main()
