import enum
import os
import warnings
from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import transformers
from PIL import Image
from transformers.image_utils import make_list_of_images

from .model.image_processing_phi3_v import (Phi3VImageProcessor,
                                            calc_hd_transform_size)


class ImageHelperBase:

    @dataclass
    class ShapeInfo:
        pad_width: int 
        pad_height: int 
        width: int 
        height: int 
        pad_left: int = 0 
        pad_top: int = 0

        @property
        def width_no_pad(self):
            return 0

    def map_bbox_square(self, shape_info: ShapeInfo, bbox: tuple[float, float, float, float]):
        x1, y1, x2, y2 = bbox
        x1 = (x1 * shape_info.width + shape_info.pad_left) / shape_info.pad_width
        y1 = (y1 * shape_info.height + shape_info.pad_top) / shape_info.pad_height
        x2 = (x2 * shape_info.width + shape_info.pad_left) / shape_info.pad_width
        y2 = (y2 * shape_info.height + shape_info.pad_top) / shape_info.pad_height
        return (x1, y1, x2, y2)
    
    @torch.no_grad()
    def preprocess(self, image: Image, return_shape_info=False):
        raise NotImplementedError

    def image_expand2square(self, pil_img: Image, max_edge_length: int=None):
        """
        return:
        - new image:
        """
        width, height = pil_img.size
        if width == height:
            return pil_img, self.ShapeInfo(width, height, width, height)
        if max_edge_length is None:
            max_edge_length = max(width, height)

        if width > height:
            if width > max_edge_length:
                ratio = max_edge_length / width
                new_width = max_edge_length
                new_height = int(height * ratio)
                pil_img = pil_img.resize(
                    (new_width, new_height), Image.LANCZOS)
                width, height = new_width, new_height
            else:
                max_edge_length = width

            pad_left = 0
            pad_top = (width - height) // 2
        else:
            if height > max_edge_length:
                ratio = max_edge_length / height
                new_width = int(width * ratio)
                new_height = max_edge_length
                pil_img = pil_img.resize(
                    (new_width, new_height), Image.LANCZOS)
                width, height = new_width, new_height
            else:
                max_edge_length = height

            pad_left = (height - width) // 2
            pad_top = 0

        result = Image.new(pil_img.mode, (max_edge_length, max_edge_length))
        result.paste(pil_img, (pad_left, pad_top))
        return result, self.ShapeInfo(max_edge_length, max_edge_length, width, height, pad_left, pad_top)

    def load_PIL_image(self, 
        image_root: str,
        image_name: str = None,
        do_convert_rgb: bool = True,
    ):
        """
        if image_root is None: 
        """
        if image_root is None:
            if image_name is None:
                return None
            image_path = image_name
        elif image_name is None:
            image_path = image_root
        else:
            image_path = os.path.join(image_root, image_name)

        try:
            img = Image.open(image_path)
            if do_convert_rgb:
                img = img.convert("RGB")
            return img
        except Image.DecompressionBombError as e:
            print(e, image_path)
            return None

    def load_image(
        self,
        image_root: str,
        image_name: str = None,
        return_shape_info=False,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Load image;

        Example:
        ```
        helper.load_image("path/to/image")
        ```
        or 
        ```
        helper.load_image("path/to/image", "image.jpg")
        ```
        """
        image = self.load_PIL_image(image_root, image_name, do_convert_rgb)
        if image is None:
            if return_shape_info:
                return None, None
            else:
                return None
        return self.preprocess(
            image,
            return_shape_info=return_shape_info,
            do_convert_rgb=do_convert_rgb,
            **kwargs,
        )

class ImageHelperWithProcessor(ImageHelperBase):

    def __init__(self, processor: transformers.CLIPImageProcessor, image_aspect_ratio: str = "pad"):
        self.processor = processor
        self.image_aspect_ratio = image_aspect_ratio

    def __call__(self, image_path: str, return_shape_info=False):
        return self.load_image(image_path, return_shape_info=return_shape_info)

    @torch.no_grad()
    def preprocess(self, image: Image, return_shape_info=False, **kwargs):
        if self.processor is None:
            # convert to torch.Tensor
            return torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32)

        size = self.processor.crop_size
        max_edge_length = max(size["height"], size["width"])

        if self.image_aspect_ratio == "pad":
            image, sinfo = self.image_expand2square(image, max_edge_length)
            image = self.processor.preprocess(image, return_tensors='pt')[
                'pixel_values'][0]
        else:
            sinfo = None
            image = self.processor.preprocess(image, return_tensors='pt')[
                'pixel_values'][0]
        
        if return_shape_info:
            return image, sinfo
        else:
            return image

class ImageHelperWithPhi3Processor(ImageHelperBase):
    def __init__(self, processor: Phi3VImageProcessor| str):
        if isinstance(processor, str):
            processor = Phi3VImageProcessor.from_pretrained(processor)
        self.processor = processor
        self.img_size = 336 # fix value; do not change
    
    def image_expand2square(self, pil_img: Image, max_edge_length=None):
        max_edge_length = self.img_size if max_edge_length is None else max_edge_length
        return super().image_expand2square(pil_img, max_edge_length)
    
    def drop_empty_crops(self, batch_out: dict[str, torch.Tensor], return_torch_tensor: bool = True):
        """
        Since the size of images are different, after padding and cropping, 
        there will be some empty crops, which is fill with zeros and 
        do not have any pixel informations.
        Therefore, these empty crops could be dropped.

        Required:
            - batch_out: the output from self.processor(...)
        Return:
            - pixel_values: shape: (K, 3, 336, 336)
                The cropped images.
                Where K = \sum_i num_crops_i, the sum of the number of valid crops for each image.
            - crop_index: shape: (K,)
                The index of the image that the crop belongs to.
                Values are in range [0, N), where N is the number of images in the batch.
                If crop_index[i] = j, then the crop i belongs to the image j.
        """
        pixel_values = batch_out["pixel_values"]
        image_sizes = batch_out["image_sizes"]

        out_pixel_values = []
        crops_index = []
        for i, img_size in enumerate(image_sizes):
            h, w = img_size
            h_crop = h // self.img_size
            w_crop = w // self.img_size
            num_crops = h_crop * w_crop + 1

            px = pixel_values[i, 0:num_crops] # shape: (num_crops, 3, 336, 336)
            out_pixel_values.append(px)
            crops_index.extend([i] * num_crops)
        
        if return_torch_tensor:
            return {
                "pixel_values": torch.cat(out_pixel_values, dim=0), # shape: (K, 3, 336, 336)
                "crop_index": torch.tensor(crops_index), # shape: (K,)
            }
        else:
            return {
                "pixel_values": out_pixel_values, # shape: (N, num_crops, 3, 336, 336)
                "crop_index": crops_index, # shape: (K,)
            }
    
    def calc_hd_image_size(self, image: Image):
        width, height = image.size
        return calc_hd_transform_size(width, height, self.processor.num_crops)
    
    def calc_hd_num_image_tokens(self, image: Image):
        w, h = self.calc_hd_image_size(image)
        return self.calc_hd_num_image_tokens_by_size(w, h)
    
    def calc_hd_num_image_tokens_by_size(self, w, h):
        return int((h//self.img_size)*(w//self.img_size) + 1) * (144 + 1)

    @torch.no_grad()
    def preprocess(self, image: Image, pad_image: Image = None, return_shape_info=False, **kwargs):
        """
        can Recive list of tensors,
        """
        image = make_list_of_images(image)

        if pad_image is None:
            global_images, sinfo = [], []
            for im in image:
                # 336 is the fix number, improve later
                im, _s = self.image_expand2square(im, 336)
                global_images.append(im)
                sinfo.append(_s)
        else:
            global_images = make_list_of_images(pad_image)
            if return_shape_info:
                warnings.warn("pad_image is not None, return_shape_info will be ignored")
            return_shape_info = False

        out = self.processor.preprocess(
            image,
            global_images=global_images,
            **kwargs,
        )

        # update num_img_tokens
        image_sizes = out["image_sizes"]
        # TODO: this is fixed, improve later
        # Each partial image is size 336x336. with CLIP H/14, there will be 24x24 tokens. 
        # We downsample the tokens by 4, so there will be 12x12 = 144 tokens.
        # A special '\n' token will be append to the end of each image. 
        # So the total number of tokens for each image is 144 + 1 = 145.
        num_img_tokens = [self.calc_hd_num_image_tokens_by_size(w, h) for h, w in image_sizes]
        out["num_img_tokens"] = num_img_tokens

        if return_shape_info:
            return out, sinfo
        else:
            return out

class DebugUtil:
    class Status(enum.Enum):
        Running = 0
        Quite = 1
        Continue = 2
        Exit = 3
        QuiteContinue = 4

        def is_continue(input):
            return input == DebugUtil.Status.Continue or input == DebugUtil.Status.QuiteContinue

        def is_quite(input):
            return input == DebugUtil.Status.Quite or input == DebugUtil.Status.QuiteContinue

    def __init__(
        self,
        print_values=True,
        print_l2=True,
        print_histogram=True,
        debug_cached_root_path="./__hidden/debug_cache",
    ):
        self.status = self.Status.Running  # type: DebugUtil.Status
        self.print_values = print_values
        self.print_l2 = print_l2
        self.print_histogram = print_histogram
        self.cached_root = debug_cached_root_path
        os.makedirs(debug_cached_root_path, exist_ok=True)

    def cached_path_join(self, fname: str):
        return os.path.join(self.cached_root, fname)

    def enable_print_values(self, val=True):
        self.print_values = val
        return self

    def enable_print_l2(self, val=True):
        self.print_l2 = val
        return self

    def enable_print_histogram(self, val=True):
        self.print_histogram = val
        return self

    @torch.no_grad()
    def fft(self, x: torch.Tensor, name: str = None, dim: int = -1, dump_file_name: str = None):

        if self.status == self.Status.Quite or self.status == self.Status.QuiteContinue:
            # quite mode; do not print anything
            return self

        if name is not None:
            print("\033[32m", name, "\033[0m")
        if not isinstance(x, torch.Tensor):
            print(x)
            return self

        x = x.float()

        # compute fourier transform
        fft_x = torch.fft.fft(x.reshape(-1, x.shape[-1]), dim=dim)
        fft_abs = torch.abs(fft_x)

        print("FFT:")
        print("FFT abs:", fft_abs.data.cpu().numpy())
        if dump_file_name is not None:
            plt.figure(figsize=(8, 4))
            plt.plot(fft_abs.data.cpu().numpy().T)
            plt.savefig(self.cached_path_join(dump_file_name))
            plt.close()

        return self

    @torch.no_grad()
    def print(self, x: torch.Tensor, name: str = None, num_bins=10, dump_file_name: str = None, decimals=None):
        if self.status == self.Status.Quite or self.status == self.Status.QuiteContinue:
            # quite mode; do not print anything
            return self

        if name is not None:
            print("\033[32m", name, "\033[0m")
        if not isinstance(x, torch.Tensor):
            print(x)
            return self

        # Print the basic details of the tensor
        print("Tensor Type:", type(x).__name__)
        print("Data Type:", x.dtype)
        print("Device:", x.device)
        print("Shape:", x.shape)
        if x.numel() == 1:
            print("Value:", x.item())
            return self

        x = x.float()

        decimals = 2 if decimals is None else decimals
        his_formater = "{:." + str(decimals) + "f}"

        if self.print_histogram:
            # Print statistical measures
            histogram = torch.histc(x, bins=num_bins)
            bin_edges = torch.linspace(x.min(), x.max(), steps=num_bins + 1)

            print("Histogram:")
            for i, count in enumerate(histogram):
                bin_l = his_formater.format(bin_edges[i])
                bin_r = his_formater.format(bin_edges[i+1])
                print(f"{bin_l.rjust(6)}, {bin_r.rjust(6)} | {int(count.item())}")

        print(" std:", x.std().item())
        print("Mean:", x.mean().item())

        # Print the tensor's values
        if self.print_values:
            values = x.data.cpu().numpy()
            if decimals is not None:
                print("Values:")
                with np.printoptions(precision=decimals, suppress=True):
                    print(values)
            else:
                print("Values:\n", values)

        if self.print_l2:
            print("Norm L2:", torch.norm(x, dim=-1).data.cpu().numpy())

        if dump_file_name is not None:
            if dump_file_name.endswith(".pt") or dump_file_name.endswith(".torch"):
                torch.save(x, self.cached_path_join(dump_file_name))
            elif dump_file_name.endswith(".npy"):
                np.save(self.cached_path_join(
                    dump_file_name), x.data.cpu().numpy())
            elif dump_file_name.endswith(".npz"):
                np.savez(self.cached_path_join(
                    dump_file_name), x=x.data.cpu().numpy())
            elif dump_file_name.endswith(".txt"):
                np.savetxt(self.cached_path_join(dump_file_name),
                           x.data.cpu().numpy(), fmt="%.5f")
            else:
                raise Exception("Unsupported file format")

        return self

    def dump_img(
        self,
        image: Union[np.ndarray, torch.Tensor],
        dump_file_name: str = None,
        auto_correct_mean_std: bool = True,
        image_preprocess_mean=[0.4814546, 0.4578275, 0.4082107],
        image_preprocess_std=[0.2686295, 0.2613025, 0.2757711],
        padding=2,
        pad_value=0,
    ):
        if self.status == self.Status.Quite or self.status == self.Status.QuiteContinue:
            # quite mode; do not print anything
            return self

        # save the images to disk, with justified mean and variance

        if dump_file_name is None:
            # sorry, we don't support display image here
            return self

        if isinstance(image, np.ndarray):
            # to pytorch
            image = torch.from_numpy(image)

        # if image.dtype is integer;  save to file
        output_file_path = self.cached_path_join(dump_file_name)

        if image.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            torchvision.utils.save_image(image, output_file_path)
            return self

        # if image.dtype is float;  save to file
        # check image value range not in [0, 1]

        if auto_correct_mean_std and (image.min() < 0 or image.max() > 1):
            # try to adjust the mean and variance
            image_preprocess_mean = torch.Tensor(
                image_preprocess_mean,
            ).to(image.device)
            image_preprocess_std = torch.Tensor(
                image_preprocess_std,
            ).to(image.device)
            if image.ndim == 3:
                image_preprocess_mean = image_preprocess_mean.view(3, 1, 1)
                image_preprocess_std = image_preprocess_std.view(3, 1, 1)
            elif image.ndim == 4:
                image_preprocess_mean = image_preprocess_mean.view(1, 3, 1, 1)
                image_preprocess_std = image_preprocess_std.view(1, 3, 1, 1)

            image = (image * image_preprocess_std) + image_preprocess_mean

        torchvision.utils.save_image(
            image, output_file_path, normalize=True, padding=padding, pad_value=pad_value)
        return self
    
    def dump_attentions(self, attentions: torch.Tensor, dump_file_name: str):
        """
        attentions: [batch, num_heads, num_patches, num_patches]

        dump_file_name: str 
        """

        batch_size, num_heads, num_patches, _ = attentions.shape

        attn = attentions.reshape(-1, 1, num_patches, num_patches)
        attn = torch.log10(1000.0*attn + 1.0)
        attn = (attn - attn.min()) / (attn.max() - attn.min())

        torchvision.utils.save_image(
            attn,
            self.cached_path_join(dump_file_name),
            nrow=num_heads,
        )

        print(f"dump attentions to {dump_file_name}")

        return self

    def pause(self, message="press enter to continue"):
        if self.status.is_continue():
            return self

        while True:
            x = input(f"{message} >>> ")
            x = x.lower()
            if x in ["exit", "bye", "x"]:
                self.status = self.Status.Exit
                exit(0)
            elif x in ["continue", "c"]:
                if self.status == self.Status.Quite or self.status == self.Status.QuiteContinue:
                    self.status = self.Status.QuiteContinue
                else:
                    self.status = self.Status.Continue
                break
            elif x in ["quite", "q"]:
                self.status = self.Status.Quite
                break
            elif x in ["qc", "cq"]:
                self.status = self.Status.QuiteContinue
                break
            elif x in ["help", "h"]:
                print("- [enter]:\n\tcontinue; to next loop")
                print("- 'exit', 'bye', 'x':\n\texit the program")
                print(
                    "- 'continue', 'c':\n\tcontinue. This pause will be skip from now on.")
                print("- 'quite', 'q': quite mode, do not print anyting except '>>>'")
                print("- 'qc', 'cq': quite and continue")
                print("- 'h', 'help':\n\tshow this help message")
            else:
                break

        return self

    def exit(self, ret_code=0):
        exit(ret_code)


GlobalDebugUtil = DebugUtil()
