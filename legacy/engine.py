"""Upscaling engine wrapping Real-ESRGAN for manwha image processing."""

from .config import DirectoryConfig, EngineConfig
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
from realesrgan import RealESRGANer
import cv2


class Engine:
    """Loads the Real-ESRGAN model and exposes upscaling with optional post-sharpening."""

    def __init__(self, engine_config: EngineConfig, directory_config: DirectoryConfig):
        self.engine_config = engine_config
        self.directory_config = directory_config

        if engine_config.cudnn_benchmark:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True

        # Build the RRDBNet architecture and load weights via position-based mapping
        model_path=str(self.directory_config.model_path)
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = self._load_model(net, loadnet)
        net.load_state_dict(state_dict, strict=False)
        net.eval().to('cuda')
        
        self.upsampler = RealESRGANer(scale=4, model_path=model_path, model=net, tile=self.engine_config.tile, tile_pad=10, pre_pad=0, half=self.engine_config.half, gpu_id=0)

    def _load_model(self, net, loadnet):
        """Map checkpoint weights onto the model by position rather than key name.

        Some checkpoints use legacy key names that don't match the current
        RRDBNet architecture. Positional mapping handles these mismatches.
        """
        model_state = net.state_dict()

        # Unwrap common checkpoint wrappers
        if isinstance(loadnet, dict) and 'params' in loadnet:
            weights = loadnet['params']
        elif isinstance(loadnet, dict) and 'params_ema' in loadnet:
            weights = loadnet['params_ema']
        else:
            weights = loadnet

        raw_keys = list(weights.keys())
        target_keys = list(model_state.keys())

        if len(raw_keys) != len(target_keys):
            print(f"Warning: Key count mismatch! File: {len(raw_keys)}, Model: {len(target_keys)}")

        new_state_dict = {}
        for i in range(min(len(raw_keys), len(target_keys))):
            new_state_dict[target_keys[i]] = weights[raw_keys[i]]

        return new_state_dict

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Upscale an image, choosing direct or chunked strategy based on aspect ratio."""
        height, width = image.shape[:2]

        if height / width < self.engine_config.ratio_threshold:
            upscaled_image = self._direct_upscale(image)
        else:
            # Tall images are split into overlapping chunks to avoid VRAM limits
            upscaled_image = self._chunked_upscale(image)

        if upscaled_image is not None:
            return self._post_sharpen(upscaled_image)

        raise RuntimeError("Upscaling Failed")

    def _direct_upscale(self, image: np.ndarray) -> np.ndarray:
        """Upscale the full image in a single pass."""
        upscaled, _ = self.upsampler.enhance(image, outscale=self.engine_config.scale)
        return upscaled

    def _chunked_upscale(self, image: np.ndarray) -> np.ndarray:
        """Upscale a tall image by processing overlapping horizontal strips and blending seams."""
        original_tile = self.upsampler.tile_size
        self.upsampler.tile_size = 0  # Disable internal tiling while we manage chunks manually

        chunk_size = self.engine_config.chunk_size
        overlap = self.engine_config.overlap
        height = image.shape[0]

        upscale_img_chunks = []
        for start_y in range(0, height, chunk_size - overlap):
            end_y = min(start_y + chunk_size, height)
            chunk = image[start_y:end_y, :]
            upscaled, _ = self.upsampler.enhance(chunk, outscale=self.engine_config.scale)
            upscale_img_chunks.append(upscaled)

        # Stitch chunks with linear alpha blending over the overlap region to hide seams
        overlap_upscaled = overlap * self.engine_config.scale
        pieces = [upscale_img_chunks[0][:-overlap_upscaled]]

        for i in range(1, len(upscale_img_chunks)):
            overlap_A = upscale_img_chunks[i - 1][-overlap_upscaled:]
            overlap_B = upscale_img_chunks[i][:overlap_upscaled]

            actual_overlap = min(overlap_A.shape[0], overlap_B.shape[0])
            overlap_A = overlap_A[-actual_overlap:]
            overlap_B = overlap_B[:actual_overlap]

            blend = np.linspace(0, 1, actual_overlap)[:, np.newaxis, np.newaxis]
            pieces.append(overlap_A * (1 - blend) + overlap_B * blend)

            # Append the non-overlapping body of every chunk except the last
            if i < len(upscale_img_chunks) - 1:
                pieces.append(upscale_img_chunks[i][overlap_upscaled:-overlap_upscaled])
            else:
                pieces.append(upscale_img_chunks[i][overlap_upscaled:])

        self.upsampler.tile_size = original_tile
        return np.concatenate(pieces, axis=0)

    def _post_sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking to recover detail softened by upscaling."""
        if self.engine_config.sharpen_strength == 0:
            return image

        blurred = cv2.GaussianBlur(image, (0, 0), self.engine_config.sharpen_radius)
        sharpened = cv2.addWeighted(image, 1 + self.engine_config.sharpen_strength, blurred, -self.engine_config.sharpen_strength, 0)

        clipped_image = np.clip(sharpened, 0, 255).astype(np.uint8)

        comparison_mask = (clipped_image[:, :] < 15)

        pixel_comparison = np.all(comparison_mask, axis=2)

        clipped_image[pixel_comparison] = 0

        return clipped_image