import argparse, os, sys, zipfile
import numpy as np
from PIL import Image
import logging
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import torch
import time
import shutil

## Hard Code Directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_EXTRACT = os.path.join(ROOT_DIR, 'TempExtract')
LOG_DIR = os.path.join(ROOT_DIR, 'Logs')
UPSCALED_DIR = os.path.join(ROOT_DIR, 'upscaled')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', '4x-UltraSharp.pth')
CHAPTER_FOLDER = os.path.join(ROOT_DIR, 'The Legend of the Northern Blade')

## Get the chapter names.
def get_chapter_name(cbz_path):
    return os.path.splitext(os.path.basename(cbz_path))[0]

## Wipes temp extract folder
def wipe_temp_extract():
    ## Check if the Directory Exists
    if os.path.isdir(TEMP_EXTRACT):
        counter = 0
        for file in os.listdir(TEMP_EXTRACT):
            file_path = os.path.join(TEMP_EXTRACT, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    counter += 1
            except FileNotFoundError:
                print(f"Error {file_path} not found")
        
        print(f"Sucessfully deleted {counter} files from TempExtract")
    else:
        os.mkdir("TempExtract")
        dir_path = os.path.join(ROOT_DIR, 'TempExtract')

        if os.path.exists(dir_path):
            dirName = os.path.dirname(dir_path)
            print(f'Sucessfully created {dirName} at {dir_path}')
    
    tempIsEmpty = len([f for f in os.listdir(TEMP_EXTRACT) if os.path.isfile(os.path.join(TEMP_EXTRACT, f))]) == 0
    print(f"TempExtract is {'Empty' if tempIsEmpty else 'not empty'}")

## Takes in the path of a cbz file and adds it to temp extract.
def populate_temp_extract(file_path):
    with zipfile.ZipFile(file_path) as z:
        file_list = z.namelist()
        filtered_files = [f for f in file_list if f.endswith(('.jpg', '.png'))]
        extracted_files = []

        counter = 1
        for original_name in filtered_files:
            z.extract(original_name, path=TEMP_EXTRACT)   
            extracted_files.append(original_name)

        if not extracted_files:
            sys.exit("No valid images found in cbz")

        for original_name in sorted(extracted_files):
            old_path = os.path.join(TEMP_EXTRACT, original_name)
            new_name = f"{counter:03d}.jpg"
            new_path = os.path.join(TEMP_EXTRACT, new_name)

            os.rename(old_path, new_path)
            counter += 1
        
        print(f"Extracted and renamed {counter - 1} images")
        return True

## Process chapter
def process_chapter(cbz_path, scale, upsampler, chapter_temp_dir):
    ## Wipe all files from the temp extract folder.
    wipe_temp_extract()

    ## Populate the temp extract
    populate_temp_extract(cbz_path)

    files = sorted([f for f in os.listdir(TEMP_EXTRACT) if f.lower().endswith('.jpg')])

    for f in files:
        input_path = os.path.join(TEMP_EXTRACT, f)
        output_filename = f.replace('.jpg', '.png')
        output_path = os.path.join(chapter_temp_dir, output_filename)
        upscale_Image(upsampler, scale, input_path, output_path)

def sharpen_input(img):
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

def upscale_Image(upsampler, scale, input_path, output_path):
    img_np = cv2.imread(input_path)
    
    img_h = len(img_np)
    img_w = len(img_np[0])

    chunk_size = 720
    overlap = 128

    up_chunks = []
    for start_y in range(0, img_h, chunk_size - overlap):
        end_y = min(start_y + chunk_size, img_h)
        
        chunk = img_np[start_y:end_y, :]

        chunk_to_process = sharpen_input(chunk.astype(np.uint8))

        output, _ = upsampler.enhance(chunk_to_process.astype(np.uint8), outscale=scale)

        math_upscale = cv2.resize(chunk, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        final_output = cv2.addWeighted(output, 0.8, math_upscale, 0.2, 0)
        
        overlap_scaled = int(overlap * scale)

        if start_y > 0:
            cut_off = overlap_scaled // 2
            output = output[cut_off:, :]

        if end_y < img_h:
            cut_off_bottom = overlap_scaled // 2
            output = output[:-cut_off_bottom, :]
            
        up_chunks.append(final_output)

    final_img = np.concatenate(up_chunks, axis=0)
    cv2.imwrite(os.path.join(output_path), final_img)

## This function is modeled after checkyKeys of basicsr util
## I did not want to deal with version incompatibilities.
def loadModel(net, loadnet):
    # 1. Prepare for mapping
    model_state = net.state_dict()
    # Ensure we are looking at the root of the loaded file
    if isinstance(loadnet, dict) and 'params' in loadnet:
        weights = loadnet['params']
    elif isinstance(loadnet, dict) and 'params_ema' in loadnet:
        weights = loadnet['params_ema']
    else:
        weights = loadnet

    # 2. Position-Based Brute Force Mapper
    # This ignores the name (e.g. "model.1.sub.0") and matches by order
    new_state_dict = {}
    raw_keys = list(weights.keys())
    target_keys = list(model_state.keys())

    if len(raw_keys) != len(target_keys):
        print(f"Warning: Key count mismatch! File: {len(raw_keys)}, Model: {len(target_keys)}")

    for i in range(min(len(raw_keys), len(target_keys))):
        new_state_dict[target_keys[i]] = weights[raw_keys[i]]
    
    return new_state_dict

def process_all_chapters(scale):
    cbz_files = sorted([f for f in os.listdir(CHAPTER_FOLDER) if f.lower().endswith('.cbz')])

    ## Logger Setup
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, 'upscale_errors.log'),
        level=logging.INFO,  # Change to INFO to log times (non-errors)
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Set-Up RealESRGAN
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    loadnet = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    state_dict = loadModel(net, loadnet)
    net.load_state_dict(state_dict, strict=False)
    net.eval().to('cuda')
    upsampler = RealESRGANer(scale=4, model_path=MODEL_PATH, model=net, tile=0, tile_pad=10, pre_pad=0, half=True, gpu_id=0)
    upscale_Image(upsampler, scale, os.path.join(TEMP_EXTRACT, '001.jpg'), os.path.join(UPSCALED_DIR, '001.png'))
    total_start = time.time()

    # for f in cbz_files:
    #     cbz_path = os.path.join(CHAPTER_FOLDER, f)
    #     chapter_name = get_chapter_name(cbz_path)
    #     chapter_temp_dir = os.path.join(UPSCALED_DIR, chapter_name + '_temp')
    #     os.makedirs(chapter_temp_dir, exist_ok=True)

    #     chapter_start = time.time()

    #     process_chapter(cbz_path, scale, upsampler, chapter_temp_dir)

    #     output_cbz_path = os.path.join(UPSCALED_DIR, chapter_name + '_upscaled.cbz')
    #     zip_to_cbz(chapter_temp_dir, output_cbz_path)
    #     shutil.rmtree(chapter_temp_dir)

    #     chapter_time = time.time() - chapter_start
    #     logging.info(f"Chapter {chapter_name} upscale time: {chapter_time:.2f} sec")

    # total_time = time.time() - total_start
    # logging.info(f"Full run time for all chapters: {total_time:.2f} sec")
        
## Zip the upscaled PNGs to CBZ
def zip_to_cbz(chapter_temp_dir, output_cbz_path):
    files = sorted([f for f in os.listdir(chapter_temp_dir) if f.lower().endswith('.png')])
    with zipfile.ZipFile(output_cbz_path, 'w') as zip_ref:
        for f in files:
            zip_ref.write(os.path.join(chapter_temp_dir, f), f)
    print(f"Zipped upscaled chapter to {output_cbz_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manwha HD Upscaler MVP")
    parser.add_argument('--scale', type = int, default = 2, help = "Upscale factor (2 or 4)")
    args = parser.parse_args()

    if not os.path.isfile(MODEL_PATH):
        sys.exit("Model Missing")

    os.makedirs(LOG_DIR, exist_ok = True)

    process_all_chapters(args.scale)
