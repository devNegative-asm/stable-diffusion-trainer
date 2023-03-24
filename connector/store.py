import copy
from datetime import datetime
import json
import random
import re
from scipy.interpolate import interp1d
import torch
import tqdm
from typing import Dict, List, Generator, Tuple
import typing
import os
import imagesize
###############
# setup: config.json
###############
with open(os.environ['SD_TRAINER_CONFIG_FILE'], "r") as config_file:
    config = json.load(config_file)

shuffle_after_n_captions = config["SHUFFLE_CAPTIONS_AFTER"]
aspect_f = config["BUCKETS"]
aspect = list(zip(aspect_f["bucket_ratios"], aspect_f["buckets"]))
all_files = [os.path.join(config["DATA_PATH"], filename) for filename in os.listdir(config["DATA_PATH"])]
all_latent_files = [instance for instance in all_files if instance.endswith(".latent")]
all_image_files = [instance for instance in all_files if instance.endswith(".png") or instance.endswith(".jpg") or instance.endswith(".webp")]
all_caption_files = [instance for instance in all_files if instance.endswith(".txt")]
print(f"found {len(all_caption_files)} caption files.")
print(f"found {len(all_image_files)} image files.")
print(f"found {len(all_latent_files)} latent files.")

assert len(all_image_files) == len(all_caption_files)
assert len(all_image_files) > 0

###############
# Tensor Silliness
###############
class StringArray:
    def __init__(self, strings : typing.List[str], encoding : typing.Literal['ascii', 'utf_16_le', 'utf_32_le'] = 'utf_16_le'):
        strings = list(strings)
        self.encoding = encoding
        self.multiplier = dict(ascii = 1, utf_16_le = 2, utf_32_le = 4)[encoding]
        self.data = torch.ByteTensor(torch.ByteStorage.from_buffer(''.join(strings).encode(encoding)))
        self.cumlen = torch.LongTensor(list(map(len, strings))).cumsum(dim = 0).mul_(self.multiplier)
        assert int(self.cumlen[-1]) == len(self.data), f'[{encoding}] is not enough to hold characters, use a larger character class'

    def __getitem__(self, i):
        return bytes(self.data[(self.cumlen[i - 1] if i >= 1 else 0) : self.cumlen[i]]).decode(self.encoding)

    def __len__(self):
        return len(self.cumlen)
    
    def tolist(self):
        data_bytes, cumlen = bytes(self.data), self.cumlen.tolist()
        return [data_bytes[0:cumlen[0]].decode(self.encoding)] + [data_bytes[start:end].decode(self.encoding) for start, end in zip(cumlen[:-1], cumlen[1:])]

###############
# Dataset Preparation
###############
def find_aspect(width: int, height: int) -> Tuple[int,int]:
    ratio = width / height
    ratio = min(aspect, key=lambda x:abs(x[0]-ratio))
    return (ratio[1][0], ratio[1][1])

def generate_images() -> Generator[Tuple[str, str, float], None, None]:
    for image_file in all_image_files:
        width, height = imagesize.get(image_file)
        yield ('data_source', ".".join(os.path.basename(image_file).split(".")[:-1]), find_aspect(width, height), (width, height))

###############
# ImageStore
###############
class ImageStore:
    def __init__(self) -> None:
        self.image_files = []
        self.image_files.extend(tqdm.tqdm(generate_images(), total=len(all_image_files)))

    def __len__(self) -> int:
        return len(self.image_files)

    # iterator returns height/width of images and their index in the store
    def entries_iterator(self) -> Generator[Tuple[Tuple[int, int], int], None, None]:
        for f in range(len(self)):
            yield self.image_files[f][3], f

class AspectBucket:
    def __init__(self, store: ImageStore,
                 batch_size: int,
                 max_ratio: float = 2):

        self.batch_size = batch_size
        self.total_dropped = 0

        if max_ratio <= 0:
            self.max_ratio = float('inf')
        else:
            self.max_ratio = max_ratio

        self.store = store
        self.buckets = []
        self._bucket_ratios = []
        self._bucket_interp = None
        self.bucket_data: Dict[tuple, List[int]] = dict()
        self.init_buckets()
        self.fill_buckets()

    def init_buckets(self):
        self.buckets = aspect_f["buckets"]

        # cache the bucket ratios and the interpolator that will be used for calculating the best bucket later
        # the interpolator makes a 1d piecewise interpolation where the input (x-axis) is the bucket ratio,
        # and the output is the bucket index in the self.buckets array
        # to find the best fit we can just round that number to get the index
        self._bucket_ratios = aspect_f["bucket_ratios"]
        self._bucket_interp = interp1d(self._bucket_ratios, list(range(len(self.buckets))), assume_sorted=True,
                                       fill_value=None)

        # convert buckets from lists (from the json) to tuples
        self.buckets = list(map(lambda x: tuple(x), self.buckets))

        for b in self.buckets:
            self.bucket_data[b] = []

    def get_batch_count(self):
        return sum(len(b) // self.batch_size for b in self.bucket_data.values())

    def get_bucket_info(self):
        return json.dumps({ "buckets": self.buckets, "bucket_ratios": self._bucket_ratios })

    def get_batch_iterator(self) -> Generator[Tuple[Tuple[int, int, int]], None, None]:
        """
        Generator that provides batches where the images in a batch fall on the same bucket

        Each element generated will be:
            (index, w, h)

        where each image is an index into the dataset
        :return:
        """
        max_bucket_len = max(len(b) for b in self.bucket_data.values())
        index_schedule = list(range(max_bucket_len))
        random.shuffle(index_schedule)

        bucket_len_table = {
            b: len(self.bucket_data[b]) for b in self.buckets
        }

        bucket_schedule = []
        for i, b in enumerate(self.buckets):
            bucket_schedule.extend([i] * (bucket_len_table[b] // self.batch_size))

        random.shuffle(bucket_schedule)

        bucket_pos = {
            b: 0 for b in self.buckets
        }

        total_generated_by_bucket = {
            b: 0 for b in self.buckets
        }

        for bucket_index in bucket_schedule:
            b = self.buckets[bucket_index]
            i = bucket_pos[b]
            bucket_len = bucket_len_table[b]

            batch = []
            while len(batch) != self.batch_size:
                # advance in the schedule until we find an index that is contained in the bucket
                k = index_schedule[i]
                if k < bucket_len:
                    entry = self.bucket_data[b][k]
                    batch.append(entry)

                i += 1

            total_generated_by_bucket[b] += self.batch_size
            bucket_pos[b] = i
            yield [(idx, *b) for idx in batch]

    def fill_buckets(self):
        entries = self.store.entries_iterator()
        total_dropped = 0

        for entry, index in tqdm.tqdm(entries, total=len(self.store)):
            if not self._process_entry(entry, index):
                total_dropped += 1

        for b, values in self.bucket_data.items():
            # shuffle the entries for extra randomness and to make sure dropped elements are also random
            random.shuffle(values)

            # make sure the buckets have an exact number of elements for the batch
            to_drop = len(values) % self.batch_size
            self.bucket_data[b] = list(values[:len(values) - to_drop])
            total_dropped += to_drop

        self.total_dropped = total_dropped

    def _process_entry(self, entry: Tuple[int, int], index: int) -> bool:
        aspect = entry[0] / entry[1] # width / height

        if aspect > self.max_ratio or (1 / aspect) > self.max_ratio:
            return False

        best_bucket = self._bucket_interp(aspect)

        if best_bucket is None:
            return False

        bucket = self.buckets[round(float(best_bucket))]

        self.bucket_data[bucket].append(index)

        return True
###############
# Regex & Setup
###############
redacted_regex = re.compile("hi. the name of this regex matched a regex, and I don't know what it's supposed to do anyway :P")
###############
# AspectDataset
###############
class AspectDataset(torch.utils.data.Dataset):
    def __init__(self, bucket: AspectBucket, ucg: float = 0.1):
        self.ucg = ucg
        self.data_source_name = StringArray(map(lambda x:x[0], bucket.store.image_files))
        self.data_source_id = StringArray(map(lambda x:x[1], bucket.store.image_files))
        self.len = len(bucket.store)

    def __len__(self):
        return self.len

    def __getitem__(self, item: Tuple[int, int, int]):
        return_dict = {'latent': None, 'captions': None}
        #image = self.data[item[0]]
        source_name = self.data_source_name[item[0]]
        source_id = self.data_source_id[item[0]]

        f = open(os.path.join(config["DATA_PATH"], source_id + ".latent"), 'rb')
        return_dict['latent'] = f.read()
        f.close()

        f = open(os.path.join(config["DATA_PATH"], source_id + ".txt"), 'r')
        captions = [tag.strip() for tag in f.read().strip().split(',')]
        f.close()

        header_captions = captions[:shuffle_after_n_captions]
        tail_captions = captions[shuffle_after_n_captions:]
        random.shuffle(tail_captions)
        caption_file = header_captions + tail_captions

        return_dict['captions'] = caption_file
        return_dict['source_name'] = source_name
        return_dict['source_id'] = source_id

        return (
            copy.deepcopy(return_dict["latent"]),
            copy.deepcopy(return_dict["captions"]),
            copy.deepcopy(return_dict["source_name"]),
            copy.deepcopy(return_dict["source_id"])
        )
###############
# AspectBucketSampler
###############     
class AspectBucketSampler(torch.utils.data.Sampler):
    def __init__(self, bucket: AspectBucket, dataset: AspectDataset):
        super().__init__(None)
        self.bucket = bucket
        self.sampler = dataset

    def __iter__(self):
        # return batches as they are and let accelerate distribute
        indices = self.bucket.get_batch_iterator()
        return iter(indices)

    def __len__(self):
        return self.bucket.get_batch_count()
