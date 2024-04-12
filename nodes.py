import os
import cv2
import random
import shutil
import subprocess

import torch
import torchaudio
import folder_paths

import numpy as np
from PIL import Image

from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

from .configs.default import get_cfg_defaults
from .core.networks.diffusion_net import DiffusionNet
from .core.networks.diffusion_util import NoisePredictor, VarianceSchedule
from .core.utils import (
    crop_src_image,
    get_pose_params,
    get_video_style_clip,
    get_wav2vec_audio_window,
)
from .generators.utils import get_netG, render_video


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

# Our any instance wants to be a wildcard string
any = AnyType("*")


def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


@torch.no_grad()
def get_diff_net(cfg, device):
    diff_net = DiffusionNet(
        cfg=cfg,
        net=NoisePredictor(cfg),
        var_sched=VarianceSchedule(
            num_steps=cfg.DIFFUSION.SCHEDULE.NUM_STEPS,
            beta_1=cfg.DIFFUSION.SCHEDULE.BETA_1,
            beta_T=cfg.DIFFUSION.SCHEDULE.BETA_T,
            mode=cfg.DIFFUSION.SCHEDULE.MODE,
        ),
    )
    checkpoint = torch.load(cfg.INFERENCE.CHECKPOINT, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    diff_net_dict = {
        k[9:]: v for k, v in model_state_dict.items() if k[:9] == "diff_net."
    }
    diff_net.load_state_dict(diff_net_dict, strict=True)
    diff_net.eval()

    return diff_net


@torch.no_grad()
def inference_one_video(
    cfg,
    audio_path,
    style_clip_path,
    pose_path,
    output_path,
    diff_net,
    device,
    max_audio_len=None,
    sample_method="ddim",
    ddim_num_step=10,
):
    audio_raw = audio_data = np.load(audio_path)

    if max_audio_len is not None:
        audio_raw = audio_raw[: max_audio_len * 50]
    gen_num_frames = len(audio_raw) // 2

    audio_win_array = get_wav2vec_audio_window(
        audio_raw,
        start_idx=0,
        num_frames=gen_num_frames,
        win_size=cfg.WIN_SIZE,
    )

    audio_win = torch.tensor(audio_win_array).to(device)
    audio = audio_win.unsqueeze(0)

    # the second parameter is "" because of bad interface design...
    style_clip_raw, style_pad_mask_raw = get_video_style_clip(
        style_clip_path, "", style_max_len=256, start_idx=0
    )

    style_clip = style_clip_raw.unsqueeze(0).to(device)
    style_pad_mask = (
        style_pad_mask_raw.unsqueeze(0).to(device)
        if style_pad_mask_raw is not None
        else None
    )

    gen_exp_stack = diff_net.sample(
        audio,
        style_clip,
        style_pad_mask,
        output_dim=cfg.DATASET.FACE3D_DIM,
        use_cf_guidance=cfg.CF_GUIDANCE.INFERENCE,
        cfg_scale=cfg.CF_GUIDANCE.SCALE,
        sample_method=sample_method,
        ddim_num_step=ddim_num_step,
    )
    gen_exp = gen_exp_stack[0].cpu().numpy()

    pose_ext = pose_path[-3:]
    pose = None
    pose = get_pose_params(pose_path)
    # (L, 9)

    selected_pose = None
    if len(pose) >= len(gen_exp):
        selected_pose = pose[: len(gen_exp)]
    else:
        selected_pose = pose[-1].unsqueeze(0).repeat(len(gen_exp), 1)
        selected_pose[: len(pose)] = pose

    gen_exp_pose = np.concatenate((gen_exp, selected_pose), axis=1)
    np.save(output_path, gen_exp_pose)
    return output_path


def cv_frame_generator(video):
    try:
        video_cap = cv2.VideoCapture(video)
        if not video_cap.isOpened():
            raise ValueError(f"{video} could not be loaded with cv.")
        # set video_cap to look at start_index frame
        total_frame_count = 0
        total_frames_evaluated = -1
        frames_added = 0
        base_frame_time = 1/video_cap.get(cv2.CAP_PROP_FPS)
        width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        prev_frame = None
        target_frame_time = base_frame_time
        yield (width, height, target_frame_time)
        time_offset=target_frame_time - base_frame_time
        while video_cap.isOpened():
            if time_offset < target_frame_time:
                is_returned = video_cap.grab()
                # if didn't return frame, video has ended
                if not is_returned:
                    break
                time_offset += base_frame_time
            if time_offset < target_frame_time:
                continue
            time_offset -= target_frame_time
            # if not at start_index, skip doing anything with frame
            total_frame_count += 1
            total_frames_evaluated += 1

            # opencv loads images in BGR format (yuck), so need to convert to RGB for ComfyUI use
            # follow up: can videos ever have an alpha channel?
            # To my testing: No. opencv has no support for alpha
            unused, frame = video_cap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # convert frame to comfyui's expected format
            # TODO: frame contains no exif information. Check if opencv2 has already applied
            frame = np.array(frame, dtype=np.float32) / 255.0
            if prev_frame is not None:
                inp  = yield prev_frame
                if inp is not None:
                    #ensure the finally block is called
                    return
            prev_frame = frame
            frames_added += 1

        if prev_frame is not None:
            yield prev_frame
    finally:
        video_cap.release()
    
class DreamTalk:
    @classmethod
    def INPUT_TYPES(s):
        audio_extensions = ["wav", "mp3", "flac"]
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        style_clip_dir = get_ext_dir('data/style_clip/3DMM')
        style_clip_files = [f for f in os.listdir(style_clip_dir) if os.path.isfile(os.path.join(style_clip_dir, f))]
        pose_dir = get_ext_dir('data/pose')
        pose_files = [f for f in os.listdir(pose_dir) if os.path.isfile(os.path.join(pose_dir, f))]
        return {"required": {
                    "image": ("IMAGE", ),
                    "audio": (sorted(files), ),
                    "style_clip": (sorted(style_clip_files), {"default": "M030_front_neutral_level1_001.mat"}),
                    "pose": (sorted(pose_files),  {"default": "RichardShelby_front_neutral_level1_001.mat"}),
                    "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "max_gen_len": ("INT", {"default": 1000, "min": 1, "max": 10000000000, "step": 1}),
                    "img_crop": ("BOOLEAN", {"default": True},),
                     },}

    CATEGORY = "Dreamtalk"

    RETURN_TYPES = ("IMAGE", "INT", "INT", )
    RETURN_NAMES = ("images", "count", "frame_rate", )
    FUNCTION = "inference"


    def inference(self, image, audio, style_clip, pose, cfg_scale, max_gen_len, img_crop):
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available.")
        device = torch.device("cuda")

        cfg = get_cfg_defaults()
        cfg.CF_GUIDANCE.SCALE = cfg_scale

        checkpoint_path = os.path.join(get_ext_dir('checkpoints'), 'denoising_network.pth')
        if os.path.isfile(checkpoint_path):
            cfg.INFERENCE.CHECKPOINT = checkpoint_path
        else:
            cfg.INFERENCE.CHECKPOINT = os.path.join(folder_paths.models_dir, 'dreamtalk/denoising_network.pth')
        
        renderer_path = os.path.join(get_ext_dir('checkpoints'), 'renderer.pt')
        if os.path.isfile(renderer_path):
            cfg.INFERENCE.RENDERER = renderer_path
        else:
            cfg.INFERENCE.RENDERER = os.path.join(folder_paths.models_dir, 'dreamtalk/renderer.pt')

        cfg.freeze()

        if not os.path.isfile(cfg.INFERENCE.CHECKPOINT):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="cncbec/dreamtalk", allow_patterns=["denoising_network.pth"], local_dir=os.path.dirname(cfg.INFERENCE.CHECKPOINT), local_dir_use_symlinks=False)
        if not os.path.isfile(cfg.INFERENCE.RENDERER):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="cncbec/dreamtalk", allow_patterns=["renderer.pt"], local_dir=os.path.dirname(cfg.INFERENCE.RENDERER), local_dir_use_symlinks=False)

        output_name = ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        tmp_dir = folder_paths.get_temp_directory()
        os.makedirs(tmp_dir, exist_ok=True)

        # get audio in 16000Hz
        wav_path = os.path.join(folder_paths.get_input_directory(), audio)
        wav_16k_path = os.path.join(tmp_dir, f"{output_name}_16K.wav")
        command = f"ffmpeg -y -i {wav_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {wav_16k_path}"
        subprocess.run(command.split())

        # get wav2vec feat from audio
        wav2vec_processor = Wav2Vec2Processor.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        )

        wav2vec_model = (
            Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
            .eval()
            .to(device)
        )

        speech_array, sampling_rate = torchaudio.load(wav_16k_path)
        audio_data = speech_array.squeeze().numpy()
        inputs = wav2vec_processor(
            audio_data, sampling_rate=16_000, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            audio_embedding = wav2vec_model(
                inputs.input_values.to(device), return_dict=False
            )[0]

        audio_feat_path = os.path.join(tmp_dir, f"{output_name}_wav2vec.npy")
        np.save(audio_feat_path, audio_embedding[0].cpu().numpy())

        # get src image
        for (_, img) in enumerate(image):
            img = 255. * img.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            src_img_path = os.path.join(tmp_dir, f"{output_name}_src_img.png")
            cropped_img_path = os.path.join(tmp_dir, f"{output_name}_cropped_img.png")
            img.save(src_img_path)
            break

        if img_crop:
            crop_src_image(src_img_path, cropped_img_path, 0.4)
        else:
            shutil.copy(src_img_path, cropped_img_path)

        style_clip_path = os.path.join(get_ext_dir('data/style_clip/3DMM'), style_clip)
        pose_path = os.path.join(get_ext_dir('data/pose'), pose)

        with torch.no_grad():
            # get diff model and load checkpoint
            diff_net = get_diff_net(cfg, device).to(device)
            # generate face motion
            face_motion_path = os.path.join(tmp_dir, f"{output_name}_facemotion.npy")
            inference_one_video(
                cfg,
                audio_feat_path,
                style_clip_path,
                pose_path,
                face_motion_path,
                diff_net,
                device,
                max_audio_len=max_gen_len,
            )
            # get renderer
            renderer_path = cfg.INFERENCE.RENDERER
            renderer_conf_path = os.path.join(get_ext_dir('generators'), 'renderer_conf.yaml')
            renderer = get_netG(renderer_path, device, renderer_conf_path)
            # render video
            output_video_path = os.path.join(tmp_dir, f"{output_name}.mp4")
            render_video(
                renderer,
                cropped_img_path,
                face_motion_path,
                wav_16k_path,
                output_video_path,
                device,
                fps=25,
                no_move=False,
            )

        gen = cv_frame_generator(output_video_path)
        (width, height, target_frame_time) = next(gen)
        width = int(width)
        height = int(height)
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
        if len(images) == 0:
            raise RuntimeError("No frames generated")
        return (images, len(images), 25)

NODE_CLASS_MAPPINGS = {
    "D_DreamTalk": DreamTalk,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_DreamTalk": "Dream Talk",
}