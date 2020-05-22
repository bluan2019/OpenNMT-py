python  train.py -data data/en2zh -save_model models/en2zh -world_size 6 -gpu_ranks 0 1 2 3 4 5 \
        -layers 4 -rnn_size 512 -word_vec_size 512 -batch_type tokens -batch_size 2048\
        -max_generator_batches 32 -normalization tokens -dropout 0.1 -accum_count 4 \
        -max_grad_norm 0 -optim adam -encoder_type transformer -decoder_type transformer \
        -position_encoding -param_init 0 -warmup_steps 16000 -learning_rate 2 -param_init_glorot \
        -decay_method noam -label_smoothing 0.1 -adam_beta2 0.998 -report_every 1000
