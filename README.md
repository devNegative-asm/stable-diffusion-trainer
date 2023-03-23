Steps
1. Create your aspect buckets with `create-aspect.py`
2. Create your config.json with keys: 
 * "DATA_PATH" (path to directory with image files and txt captions with the same filename)
 * "BUCKETS" (output of `create-aspect.py`)
 * "SHUFFLE_CAPTIONS_AFTER" (when processing tags, keep the first n tags in the same order)
3. register the environment variable SD_TRAINER_CONFIG_FILE to the path to the config.json file
4. run `export-buckets.py` with the desired training batch size, saving its result as a .pt file
5. run `encoder.py` using the buckets from `export-buckets.py` to convert your training images into latent space
6. fine tune using `finetuner.py`

Example fine tuning command:
```
SD_TRAINER_CONFIG_FILE=config.json python finetuner.py --model ../model/sd2_1 --run_name "test" --dataset buckets.pt --lr 1e-6 --epochs 1 --use_ema --ucg 0.01 --use_8bit_adam --save_steps 4500 --reshuffle_tags --image_log_steps 500 --image_log_amount 3 --lr_scheduler cosine_with_restarts --lr_num_cycles 5 --lr_min_scale 0.2 --use_xformers --train_text_encoder --extended_mode_chunks 3 --clip_penultimate
```