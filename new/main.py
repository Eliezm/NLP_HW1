from pos_memm.pos_memm import POS_MEMM, analyze_results, save_comp_results, data_preprocessing

def main():
    # ── edit these paths to wherever your files really live ──
    TRAIN_PATH = "data/train.wtag"
    TEST_PATH  = "data/test.wtag"
    COMP_PATH  = "data/comp.words"
    RESULTS_DIR = "results"

    # ── pick your hyper‐parameters ──
    REGULARIZATION      = 5e-05
    SPELLING_THRESHOLD  = 5
    WORD_THRESHOLD      = 3
    MODE                = "complex"
    USE_106_107         = True
    VERBOSITY           = 0
    PARALLEL            = False

    m = POS_MEMM()
    # 1) train
    print("=== TRAINING ===")
    m.train(
        TRAIN_PATH,
        regularization      = REGULARIZATION,
        mode                = MODE,
        spelling_threshold  = SPELLING_THRESHOLD,
        word_count_threshold= WORD_THRESHOLD,
        use_106_107         = USE_106_107,
        verbosity           = VERBOSITY,
        log_path            = None
    )
    m.save_model(RESULTS_DIR)

    # 2) test
    print("\n=== TESTING ===")
    m.test(
        TEST_PATH,
        end                 = 0,
        parallel            = PARALLEL,
        verbosity           = VERBOSITY,
        save_results_to_file= RESULTS_DIR,
        log_path            = None
    )

    # 3) analyze
    print("\n=== ANALYSIS ===")
    PRED_PATH = f"{RESULTS_DIR}/predictions.txt"
    analyze_results(PRED_PATH, TEST_PATH, TRAIN_PATH, RESULTS_DIR)

    # 4) (optional) build comp output
    print("\n=== COMPARE ===")
    _,_,comp_sents,_ = data_preprocessing(COMP_PATH,'comp')
    all_tagged, _ = m.predict_parallel(comp_sents)  # or .predict()
    save_comp_results(all_tagged, RESULTS_DIR, COMP_PATH)

    print("\nDone.  Everything’s in", RESULTS_DIR)

if __name__ == "__main__":
    main()
