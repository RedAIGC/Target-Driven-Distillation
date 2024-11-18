import torch
import itertools
import json
import math
from typing import Iterable, List, Optional, Union
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t

import torchvision.transforms.functional as TF
from torchvision.transforms import RandomHorizontalFlip
from torch.utils.data import Dataset, Sampler, default_collate
from torchvision import transforms

import webdataset as wds
from braceexpand import braceexpand
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)
import os

from torch.utils.data import DataLoader
# from diffusers.training_utils import resolve_interpolation_mode

import chardet
import random

# Adjust for your dataset
WDS_JSON_WIDTH = "width"  # original_width for LAION
WDS_JSON_HEIGHT = "height"  # original_height for LAION
MIN_SIZE = 512  # ~960 for LAION, ideal: 1024 if the dataset contains large images


# from torchvision.transforms import ToPILImage
from diffusers.utils import make_image_grid

def resolve_interpolation_mode(interpolation_type: str):
    """
    Maps a string describing an interpolation function to the corresponding torchvision `InterpolationMode` enum. The
    full list of supported enums is documented at
    https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.functional.InterpolationMode.

    Args:
        interpolation_type (`str`):
            A string describing an interpolation method. Currently, `bilinear`, `bicubic`, `box`, `nearest`,
            `nearest_exact`, `hamming`, and `lanczos` are supported, corresponding to the supported interpolation modes
            in torchvision.

    Returns:
        `torchvision.transforms.InterpolationMode`: an `InterpolationMode` enum used by torchvision's `resize`
        transform.
    """
    # if not is_torchvision_available():
    #     raise ImportError(
    #         "Please make sure to install `torchvision` to be able to use the `resolve_interpolation_mode()` function."
    #     )

    if interpolation_type == "bilinear":
        interpolation_mode = transforms.InterpolationMode.BILINEAR
    elif interpolation_type == "bicubic":
        interpolation_mode = transforms.InterpolationMode.BICUBIC
    elif interpolation_type == "box":
        interpolation_mode = transforms.InterpolationMode.BOX
    elif interpolation_type == "nearest":
        interpolation_mode = transforms.InterpolationMode.NEAREST
    elif interpolation_type == "nearest_exact":
        interpolation_mode = transforms.InterpolationMode.NEAREST_EXACT
    elif interpolation_type == "hamming":
        interpolation_mode = transforms.InterpolationMode.HAMMING
    elif interpolation_type == "lanczos":
        interpolation_mode = transforms.InterpolationMode.LANCZOS
    else:
        raise ValueError(
            f"The given interpolation mode {interpolation_type} is not supported. Currently supported interpolation"
            f" modes are `bilinear`, `bicubic`, `box`, `nearest`, `nearest_exact`, `hamming`, and `lanczos`."
        )

    return interpolation_mode


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f

def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext) :param lcase: convert suffixes to
    lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample

def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples



class WebdatasetFilter:
    def __init__(self, min_size=MIN_SIZE, max_pwatermark=0.5):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark

    def __call__(self, x):
        try:
            if "json" in x:
                x_json = json.loads(x["json"])
                filter_size = (x_json.get(WDS_JSON_WIDTH, 0.0) or 0.0) >= self.min_size and x_json.get(
                    WDS_JSON_HEIGHT, 0
                ) >= self.min_size
                filter_watermark = (x_json.get("pwatermark", 0.0) or 0.0) <= self.max_pwatermark
                return filter_size and filter_watermark
            else:
                return False
        except Exception:
            return False
        

class SDXLText2ImageDataset:
    def __init__(
        self,
        train_shards_path_or_url: Union[str, List[str]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        resolution: int = 1024,
        interpolation_type: str = "bilinear",
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        use_fix_crop_and_size: bool = False,
        random_flip: bool = False,
        
    ):
        
        # train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
        if os.path.isdir(train_shards_path_or_url):
            tar_list = os.listdir(train_shards_path_or_url)
            temp_train_shards_path_or_url = [os.path.join(train_shards_path_or_url, f) for f in tar_list if f.endswith('.tar')] #and f.startswith('cog_')]
        train_shards_path_or_url = temp_train_shards_path_or_url
        # flatten list using itertools
        # train_shards_path_or_url = list(itertools.chain.from_iterable(temp_train_shards_path_or_url))

        def get_orig_size(json):
            if use_fix_crop_and_size:
                return (resolution, resolution)
            else:
                return (int(json.get(WDS_JSON_WIDTH, 0.0)), int(json.get(WDS_JSON_HEIGHT, 0.0)))

        interpolation_mode = resolve_interpolation_mode(interpolation_type)

        def transform(example):
            # resize image
            image = example["image"]
            image = TF.resize(image, resolution, interpolation=interpolation_mode)
            
            # random flip
            if random_flip and random.random()<0.5:
                image = TF.hflip(image)
            example["orig_size"] = (image.size[1], image.size[0])
            # get crop coordinates and crop image
            c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
            image = TF.crop(image, c_top, c_left, resolution, resolution)
            
            image = TF.to_tensor(image)            
            image = TF.normalize(image, [0.5], [0.5])

            example["image"] = image
            example["crop_coords"] = (c_top, c_left) if not use_fix_crop_and_size else (0, 0)
            return example
        
        def decode(example):
            try:
                text = example["text"]
            
                text_encoding = chardet.detect(text)["encoding"]
            
                example["text"] = text.decode(text_encoding)
                #print(example["text"])
            except:
                example['text'] = ""
            return example

        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.rename(
                image="jpg;png;jpeg;webp", text="text;txt;caption;prompt", handler=wds.warn_and_stop
            ),
            wds.map(filter_keys({"image", "text"})),
            wds.map(transform),
            wds.map(decode),
            wds.to_tuple("image", "text", "orig_size", "crop_coords"),
        ]

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            # wds.select(WebdatasetFilter(min_size=MIN_SIZE)),
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        

        
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        
        
        # add meta-data to dataloader instance for convenience
        
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        
    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader
    
