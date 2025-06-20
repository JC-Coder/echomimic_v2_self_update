import os
import random
from pathlib import Path
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image, UnidentifiedImageError
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
from src.utils.util import save_videos_grid
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2
from moviepy.editor import VideoFileClip, AudioFileClip

import gradio as gr
from datetime import datetime
from torchao.quantization import quantize_, int8_weight_only
import gc
import glob

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import threading
from fastapi.staticfiles import StaticFiles

total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
print(f"\033[32mCUDA版本：{torch.version.cuda}\033[0m")
print(f"\033[32mPytorch版本：{torch.__version__}\033[0m")
print(f"\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m")
print(f"\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m")
print(f"\033[32m精度：float16\033[0m")
dtype = torch.float16
if torch.cuda.is_available():
    device = "cuda"
else:
    print("cuda not available, using cpu")
    device = "cpu"

ffmpeg_path = os.getenv("FFMPEG_PATH")
if ffmpeg_path is None:
    print(
        "please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=./ffmpeg-4.4-amd64-static"
    )
elif ffmpeg_path not in os.getenv("PATH"):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


def load_image(refimg, width, height):
    try:
        if isinstance(refimg, str) and os.path.exists(refimg):
            img = Image.open(refimg)
        elif isinstance(refimg, np.ndarray):
            img = Image.fromarray(refimg)
        elif isinstance(refimg, Image.Image):
            img = refimg
        else:
            raise ValueError(f"Unsupported image input type: {type(refimg)}")
        return img.resize((width, height))
    except (UnidentifiedImageError, ValueError, OSError) as e:
        print(f"Error loading image: {e}")
        raise ValueError(
            "Failed to load the uploaded image. Please check the file format and try again."
        )


def generate(
    image_input,
    audio_input,
    pose_input,
    width,
    height,
    length,
    steps,
    sample_rate,
    cfg,
    fps,
    context_frames,
    context_overlap,
    quantization_input,
    seed,
):
    width = int(width)
    height = int(height)
    length = int(length)
    steps = int(steps)
    sample_rate = int(sample_rate)
    fps = int(fps)
    context_frames = int(context_frames)
    context_overlap = int(context_overlap)
    if seed is not None:
        try:
            seed = int(seed)
        except Exception:
            seed = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("outputs")
    save_dir.mkdir(exist_ok=True, parents=True)

    ############# model_init started #############
    ## vae init
    vae = AutoencoderKL.from_pretrained("./pretrained_weights/sd-vae-ft-mse").to(
        device, dtype=dtype
    )
    if quantization_input:
        quantize_(vae, int8_weight_only())
        print("使用int8量化")

    ## reference net init
    reference_unet = UNet2DConditionModel.from_pretrained(
        "./pretrained_weights/sd-image-variations-diffusers",
        subfolder="unet",
        use_safetensors=False,
    ).to(dtype=dtype, device=device)
    reference_unet.load_state_dict(
        torch.load("./pretrained_weights/reference_unet.pth", weights_only=True)
    )
    if quantization_input:
        quantize_(reference_unet, int8_weight_only())

    ## denoising net init
    if os.path.exists("./pretrained_weights/motion_module.pth"):
        print("using motion module")
    else:
        exit("motion module not found")
        ### stage1 + stage2
    denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
        "./pretrained_weights/sd-image-variations-diffusers",
        "./pretrained_weights/motion_module.pth",
        subfolder="unet",
        unet_additional_kwargs={
            "use_inflated_groupnorm": True,
            "unet_use_cross_frame_attention": False,
            "unet_use_temporal_attention": False,
            "use_motion_module": True,
            "cross_attention_dim": 384,
            "motion_module_resolutions": [1, 2, 4, 8],
            "motion_module_mid_block": True,
            "motion_module_decoder_only": False,
            "motion_module_type": "Vanilla",
            "motion_module_kwargs": {
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                "temporal_position_encoding": True,
                "temporal_position_encoding_max_len": 32,
                "temporal_attention_dim_div": 1,
            },
        },
    ).to(dtype=dtype, device=device)
    denoising_unet.load_state_dict(
        torch.load("./pretrained_weights/denoising_unet.pth", weights_only=True),
        strict=False,
    )

    # pose net init
    pose_net = PoseEncoder(
        320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)
    ).to(dtype=dtype, device=device)
    pose_net.load_state_dict(
        torch.load("./pretrained_weights/pose_encoder.pth", weights_only=True)
    )

    ### load audio processor params
    audio_processor = load_audio_model(
        model_path="./pretrained_weights/audio_processor/tiny.pt", device=device
    )

    ############# model_init finished #############
    sched_kwargs = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "linear",
        "clip_sample": False,
        "steps_offset": 1,
        "prediction_type": "v_prediction",
        "rescale_betas_zero_snr": True,
        "timestep_spacing": "trailing",
    }
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_net,
        scheduler=scheduler,
    )

    pipe = pipe.to(device, dtype=dtype)

    if seed is not None and seed > -1:
        generator = torch.manual_seed(seed)
    else:
        seed = random.randint(100, 1000000)
        generator = torch.manual_seed(seed)

    inputs_dict = {
        "refimg": image_input,
        "audio": audio_input,
        "pose": pose_input,
    }

    print("Pose:", inputs_dict["pose"])
    print("Reference:", inputs_dict["refimg"])
    print("Audio:", inputs_dict["audio"])

    save_name = f"{save_dir}/{timestamp}"

    ref_image_pil = load_image(inputs_dict["refimg"], width, height)
    audio_clip = AudioFileClip(inputs_dict["audio"])

    length = min(
        length, int(audio_clip.duration * fps), len(os.listdir(inputs_dict["pose"]))
    )

    start_idx = 0

    pose_list = []
    for index in range(start_idx, start_idx + length):
        tgt_musk = np.zeros((width, height, 3)).astype("uint8")
        tgt_musk_path = os.path.join(inputs_dict["pose"], "{}.npy".format(index))
        detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
        imh_new, imw_new, rb, re, cb, ce = detected_pose["draw_pose_params"]
        im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
        im = np.transpose(np.array(im), (1, 2, 0))
        tgt_musk[rb:re, cb:ce, :] = im

        tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert("RGB")
        pose_list.append(
            torch.Tensor(np.array(tgt_musk_pil))
            .to(dtype=dtype, device=device)
            .permute(2, 0, 1)
            / 255.0
        )

    poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
    audio_clip = AudioFileClip(inputs_dict["audio"])

    audio_clip = audio_clip.set_duration(length / fps)
    video = pipe(
        ref_image_pil,
        inputs_dict["audio"],
        poses_tensor[:, :, :length, ...],
        width,
        height,
        length,
        steps,
        cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=context_frames,
        fps=fps,
        context_overlap=context_overlap,
        start_idx=start_idx,
    ).videos

    final_length = min(video.shape[2], poses_tensor.shape[2], length)
    video_sig = video[:, :, :final_length, :, :]

    save_videos_grid(
        video_sig,
        save_name + "_woa_sig.mp4",
        n_rows=1,
        fps=fps,
    )

    video_clip_sig = VideoFileClip(
        save_name + "_woa_sig.mp4",
    )
    video_clip_sig = video_clip_sig.set_audio(audio_clip)
    video_clip_sig.write_videofile(
        save_name + "_sig.mp4", codec="libx264", audio_codec="aac", threads=2
    )
    video_output = save_name + "_sig.mp4"
    seed_text = gr.update(visible=True, value=seed)
    return video_output, seed_text


def list_outputs():
    files = sorted(glob.glob("outputs/*.mp4"), reverse=True)
    return files


with gr.Blocks() as demo:
    gr.Markdown(
        """
            <div>
                <h2 style="font-size: 30px;text-align: center;">EchoMimicV2</h2>
            </div>
            <div style="text-align: center;">
                <a href="https://github.com/antgroup/echomimic_v2">🌐 Github</a> |
                <a href="https://arxiv.org/abs/2411.10061">📜 arXiv </a>
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ 该演示仅供学术研究和体验使用。
            </div>
            
            """
    )
    with gr.Column():
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    image_input = gr.Image(
                        label="图像输入（自动缩放）",
                        type="filepath",
                        sources=["upload"]
                    )
                    audio_input = gr.Audio(label="音频输入", type="filepath")
                    pose_input = gr.Textbox(
                        label="姿态输入（目录地址）",
                        placeholder="请输入姿态数据的目录地址",
                        value="assets/halfbody_demo/pose/fight",
                    )
                with gr.Group():
                    with gr.Row():
                        width = gr.Number(
                            label="宽度（默认768，请选择默认值）", value=768
                        )
                        height = gr.Number(
                            label="高度（默认768，请选择默认值）", value=768
                        )
                        length = gr.Number(label="视频长度，推荐120）", value=120)
                    with gr.Row():
                        steps = gr.Number(label="步骤（默认30）", value=30)
                        sample_rate = gr.Number(
                            label="采样率（默认16000）", value=16000
                        )
                        cfg = gr.Number(label="cfg（推荐2.5）", value=2.5, step=0.1)
                    with gr.Row():
                        fps = gr.Number(label="帧率（默认24）", value=24)
                        context_frames = gr.Number(
                            label="上下文框架（默认12）", value=12
                        )
                        context_overlap = gr.Number(
                            label="上下文重叠（默认3）", value=3
                        )
                    with gr.Row():
                        quantization_input = gr.Checkbox(
                            label="int8量化（推荐显存12G的用户开启，并使用不超过5秒的音频）",
                            value=False,
                        )
                        seed = gr.Number(label="种子(-1为随机)", value=-1)
                generate_button = gr.Button("🎬 生成视频")
            with gr.Column():
                video_output = gr.Video(label="输出视频")
                seed_text = gr.Textbox(label="种子", interactive=False, visible=False)
        with gr.Row():
            gr.Markdown("**所有已生成视频：**")
            refresh_button = gr.Button("🔄 刷新列表")
        with gr.Row():
            files_list = gr.Files(label="输出文件列表", file_count="multiple")

        def update_files():
            return list_outputs()

        refresh_button.click(fn=update_files, inputs=[], outputs=[files_list])
        # Also update the list after generation
        generate_button.click(
            fn=update_files, inputs=[], outputs=[files_list], queue=False
        )

        gr.Examples(
            examples=[
                [
                    "EMTD_dataset/ref_imgs_by_FLUX/man/0003.png",
                    "assets/halfbody_demo/audio/chinese/fighting.wav",
                ],
                [
                    "EMTD_dataset/ref_imgs_by_FLUX/woman/0033.png",
                    "assets/halfbody_demo/audio/chinese/good.wav",
                ],
                [
                    "EMTD_dataset/ref_imgs_by_FLUX/man/0010.png",
                    "assets/halfbody_demo/audio/chinese/news.wav",
                ],
                [
                    "EMTD_dataset/ref_imgs_by_FLUX/man/1168.png",
                    "assets/halfbody_demo/audio/chinese/no_smoking.wav",
                ],
                [
                    "EMTD_dataset/ref_imgs_by_FLUX/woman/0057.png",
                    "assets/halfbody_demo/audio/chinese/ultraman.wav",
                ],
                [
                    "EMTD_dataset/ref_imgs_by_FLUX/man/0001.png",
                    "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav",
                ],
                [
                    "EMTD_dataset/ref_imgs_by_FLUX/woman/0077.png",
                    "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav",
                ],
            ],
            inputs=[image_input, audio_input],
            label="预设人物及音频",
        )

    generate_button.click(
        generate,
        inputs=[
            image_input,
            audio_input,
            pose_input,
            width,
            height,
            length,
            steps,
            sample_rate,
            cfg,
            fps,
            context_frames,
            context_overlap,
            quantization_input,
            seed,
        ],
        outputs=[video_output, seed_text],
    )

def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return destination

# FastAPI app
fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
fastapi_app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@fastapi_app.post("/generate")
async def generate_api(
    image_input: UploadFile = File(...),
    audio_input: UploadFile = File(...),
    pose_input: str = Form(...),
    width: int = Form(768),
    height: int = Form(768),
    length: int = Form(120),
    steps: int = Form(30),
    sample_rate: int = Form(16000),
    cfg: float = Form(2.5),
    fps: int = Form(24),
    context_frames: int = Form(12),
    context_overlap: int = Form(3),
    quantization_input: bool = Form(False),
    seed: int = Form(-1),
):
    # Save uploaded files to temp
    image_path = f"temp/{datetime.now().strftime('%Y%m%d_%H%M%S')}_image.png"
    audio_path = f"temp/{datetime.now().strftime('%Y%m%d_%H%M%S')}_audio.wav"
    os.makedirs("temp", exist_ok=True)
    save_upload_file(image_input, image_path)
    save_upload_file(audio_input, audio_path)
    # Call generate
    try:
        video_output, _ = generate(
            image_path,
            audio_path,
            pose_input,
            width,
            height,
            length,
            steps,
            sample_rate,
            cfg,
            fps,
            context_frames,
            context_overlap,
            quantization_input,
            seed,
        )
        return FileResponse(video_output, media_type="video/mp4", filename=os.path.basename(video_output))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@fastapi_app.get("/outputs")
async def list_outputs_api():
    files = sorted(glob.glob("outputs/*.mp4"), reverse=True)
    return {"files": [os.path.basename(f) for f in files]}

# Threaded FastAPI runner
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    threading.Thread(target=run_fastapi, daemon=True).start()
    demo.launch(
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7860,  # Default Gradio port
        share=True,  # Create a public URL
        inbrowser=False,  # Don't try to open browser
    )
