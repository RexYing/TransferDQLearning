class config():
    input_path = "../project_autoencoder/results/pj_nature/frames/"

    # output config
    output_path  = "results/ae_train/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"

    # model and training config
    q_scope           = "q_0"
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 250000
    batch_size         = 32
    state_history      = 4
    lr           = 0.01
    epoch_num    = 5

