import pickle
import os
import zipfile

from inference import tag_all_test

# unzip into the current directory
OUTPUT_DIRECTORY_PATH = "."

def unzip_directory(zip_path, output_path="."):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)

def main():
    IDS = "207476763_322409376"

    # Model 1
    m1_path = "weights_1.pkl"
    # m1_path = os.path.join("trained_models", "weights_1.pkl")
    with open(m1_path, "rb") as f:
        (opt_params1, f2i1) = pickle.load(f)
    w1 = opt_params1[0]
    out1 = f"comp_m1_{IDS}.wtag"             # relative path
    print(f"[Model1] tagging → {out1}")
    tag_all_test("data/comp1.words", w1, f2i1, out1, tagged=False)

    # Model 2
    m2_path = "weights_2.pkl"
    # m2_path = os.path.join("trained_models", "weights_2.pkl")
    with open(m2_path, "rb") as f:
        (opt_params2, f2i2) = pickle.load(f)
    w2 = opt_params2[0]
    out2 = f"comp_m2_{IDS}.wtag"
    print(f"[Model2] tagging → {out2}")
    tag_all_test("data/comp2.words", w2, f2i2, out2, tagged=False)

if __name__ == "__main__":
    # zip_file_path = "HW1_207476763_322409376.zip"
    # unzip_directory(zip_file_path, OUTPUT_DIRECTORY_PATH)
    main()
