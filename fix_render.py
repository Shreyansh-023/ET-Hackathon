#!/usr/bin/env python3
"""
Fix black video issue by using FFmpeg filter_complex to overlay images on timeline.
"""
import json
import subprocess
from pathlib import Path

def main():
    job_id = "20260323-182428-16353efd"
    job_dir = Path("jobs") / job_id
    manifest_path = job_dir / "blender" / "blender_scene_manifest.json"
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    fps = manifest["fps"]
    width, height = 1920, 1080
    
    # Output paths
    output_dir = (job_dir / "renders").resolve()
    voiceover = (job_dir / "audio" / "voiceover.wav").resolve()
    subtitles_path = (job_dir / "audio" / "subtitles.srt").resolve()
    output_video = output_dir / "final.mp4"
    
    # Build filter_complex with overlaid images
    input_files = []
    scenes = manifest.get("scenes", [])
    
    # Add image inputs
    for scene in scenes:
        img = scene.get("image_path", "")
        if img:
            input_files.append("-loop")
            input_files.append("1")
            input_files.append("-i")
            input_files.append(img)
    
    # Build filter chain with proper overlay syntax
    filter_parts = []
    filter_parts.append("color=black:s=1920x1080:d=90[bg]")
    
    # Scale each image input
    for idx in range(len(scenes)):
        filter_parts.append(
            f"[{idx+1}]scale=w=1920:h=1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black[img{idx}]"
        )
    
    # Chain overlays with proper padding between each
    prev = "[bg]"
    for idx, scene in enumerate(scenes):
        start = scene.get("start_frame", 0) / fps
        end = scene.get("end_frame", 0) / fps
        next_pad = f"[tmp{idx}]" if idx < len(scenes) - 1 else "[v]"
        filter_parts.append(
            f"{prev}[img{idx}]overlay=enable='between(t\\,{start}\\,{end})':x=0:y=0{next_pad}"
        )
        prev = next_pad
    
    filter_complex = ";".join(filter_parts)
    
    # Build FFmpeg command to escape subtitle path properly
    sub_path = str(subtitles_path.resolve().as_posix())
    if len(sub_path) >= 2 and sub_path[1] == ":":
        sub_path = f"{sub_path[0]}\\:{sub_path[2:]}"
    
    cmd = [
        "ffmpeg",
        "-y",
        *input_files,
        "-i", str(voiceover),
        "-filter_complex", filter_complex,
        "-vf", f"subtitles='{sub_path}'",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "192k",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-movflags", "+faststart",
        str(output_video),
    ]
    
    print(f"Building image-composited video with FFmpeg...")
    result = subprocess.run(cmd, cwd=str(job_dir.parent), capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"SUCCESS: Video rebuilt: {output_video}")
        print(f"File size: {output_video.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(f"ERROR: FFmpeg failed:\n{result.stderr}")
        print(f"Command: {' '.join(cmd)}")
        exit(1)

if __name__ == "__main__":
    main()
