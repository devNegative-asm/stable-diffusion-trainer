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