#!/usr/bin/env python3
"""
Simplest possible fix: Use FFmpeg to create video from images with proper timing.
"""
import json
import subprocess
import tempfile
from pathlib import Path

def main():
    job_id = "20260323-182428-16353efd"
    job_dir = Path("jobs") / job_id
    manifest_path = job_dir / "blender" / "blender_scene_manifest.json"
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    fps = manifest["fps"]
    scenes = manifest["scenes"]
    voiceover = (job_dir / "audio" / "voiceover.wav").resolve()
    subtitles_path = (job_dir / "audio" / "subtitles.srt").resolve()
    output_video = job_dir / "renders" / "final.mp4"
    
    # Create concat demux list
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = Path(f.name)
        for scene in scenes:
            img_path = scene["image_path"]
            duration = (scene["end_frame"] - scene["start_frame"]) / fps
            f.write(f"file '{img_path}'\n")
            f.write(f"duration {duration}\n")
    
    try:
        # Step 1: Create video from image sequence
        cmd_video = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            str(output_video),
        ]
        
        print("Step 1: Creating video from images...")
        result = subprocess.run(cmd_video, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR in video creation: {result.stderr}")
            return
        
        # Step 2: Add audio using -shortest and proper mapping
        temp_with_audio = output_video.parent / "temp_with_audio.mp4"
        cmd_audio = [
            "ffmpeg", "-y",
            "-i", str(output_video),
            "-i", str(voiceover),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            str(temp_with_audio),
        ]
        
        print("Step 2: Adding audio...")
        result = subprocess.run(cmd_audio, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR in audio mux: {result.stderr}")
            return
        
        # Step 3: Add subtitles and final processing
        sub_path = str(subtitles_path.as_posix())
        if len(sub_path) >= 2 and sub_path[1] == ":":
            sub_path = f"{sub_path[0]}\\:{sub_path[2:]}"
        
        final_output = str(output_video)
        cmd_final = [
            "ffmpeg", "-y",
            "-i", str(temp_with_audio),
            "-vf", f"subtitles='{sub_path}'",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "192k",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-movflags", "+faststart",
            final_output,
        ]
        
        print("Step 3: Adding subtitles and finalizing...")
        result = subprocess.run(cmd_final, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR in final processing: {result.stderr}")
            return
        
        # Cleanup
        temp_with_audio.unlink()
        
        size_mb = output_video.stat().st_size / 1024 / 1024
        print(f"SUCCESS: Video completed: {output_video}")
        print(f"Size: {size_mb:.2f} MB")
        
    finally:
        concat_file.unlink()

if __name__ == "__main__":
    main()
