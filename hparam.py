import tensorflow as tf

hparams = tf.contrib.training.HParams(
    # Audio:
    num_mels=80,
    n_fft=2048,
    sample_rate=16000,
    win_length=800,
    hop_length=200,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    fmin=0,
    fmax=8000,

    # train
    learning_rate_decay_way='linear', #cosine or linear
    initial_learning_rate=0.0001,
    final_learning_rate=1e-7,
    clip_norm=1.0,
    train_steps=2000000,
    save_model_every=2000,
    summary_interval=1,
    logdir_root='./logdir',
    warmup_steps=80000,
    decay_rate=0.5,

    # network
    sample_size=200 * 80,                # sample size
    batch_size=2,                       # batch size

)
