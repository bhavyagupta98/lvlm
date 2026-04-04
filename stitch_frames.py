#!/usr/bin/env python3
"""
Stitch frame images into a video to verify vehicle movement and test execution.
"""

import argparse
import logging
from pathlib import Path
from typing import List
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stitch_frames_to_video(image_dir: Path, output_path: Path, fps: int = 5, codec: str = 'mp4v'):
    """
    Stitch frame images into an MP4 video.
    
    Args:
        image_dir: Directory containing frame_*.jpg files
        output_path: Path to save output video
        fps: Frames per second (default 20 matches CARLA sync)
        codec: Video codec ('mp4v' for MP4, 'MJPG' for AVI, 'avc1' for H.264)
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not installed. Install with: pip install opencv-python")
        return False
    
    # Find all frame images
    frame_files = sorted(image_dir.glob('frame_*.jpg'))
    
    if not frame_files:
        logger.error(f"No frame images found in {image_dir}")
        return False
    
    logger.info(f"Found {len(frame_files)} frames")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        logger.error(f"Failed to read first frame: {frame_files[0]}")
        return False
    
    height, width = first_frame.shape[:2]
    logger.info(f"Frame dimensions: {width}x{height}")
    
    # Try multiple codecs (fallback approach for macOS compatibility)
    codecs_to_try = [
        ('mp4v', '.mp4'),  # Standard MP4
        ('avc1', '.mp4'),  # H.264 (widely compatible)
        ('H264', '.mp4'),  # Alternative H.264
        ('xvid', '.avi'),  # XVID codec
        ('MJPG', '.avi'),  # Motion JPEG (most compatible)
    ]
    
    out = None
    successful_codec = None
    successful_ext = None
    
    for codec_name, file_ext in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            test_output = str(output_path).replace('.mp4', file_ext).replace('.avi', file_ext)
            
            logger.info(f"Trying codec: {codec_name}")
            out = cv2.VideoWriter(test_output, fourcc, fps, (width, height))
            
            if out.isOpened():
                logger.info(f"✓ Successfully opened video writer with codec: {codec_name}")
                successful_codec = codec_name
                successful_ext = file_ext
                output_path = Path(test_output)
                break
            else:
                logger.debug(f"Failed to open with {codec_name}, trying next...")
                out = None
        except Exception as e:
            logger.debug(f"Codec {codec_name} failed: {e}")
            continue
    
    if out is None or not out.isOpened():
        logger.error(f"Failed to open video writer with any codec. Available codecs may be limited.")
        logger.info("Install ffmpeg: brew install ffmpeg")
        return False
    
    # Write frames
    for i, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            logger.warning(f"Failed to read frame {i}: {frame_path}")
            continue
        
        # Ensure correct size
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(frame_files)} frames")
    
    out.release()
    logger.info(f"✓ Video saved to {output_path}")
    logger.info(f"  Codec: {successful_codec}, Duration: {len(frame_files) / fps:.1f}s at {fps} FPS")
    return True


def stitch_frames_to_gif(image_dir: Path, output_path: Path, fps: int = 5, downsample: int = 1):
    """
    Stitch frame images into an animated GIF.
    
    Args:
        image_dir: Directory containing frame_*.jpg files
        output_path: Path to save output GIF
        fps: Frames per second
        downsample: Skip frames (e.g., downsample=2 uses every 2nd frame)
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("PIL not installed. Install with: pip install pillow")
        return False
    
    # Find and load frames
    frame_files = sorted(image_dir.glob('frame_*.jpg'))[::downsample]
    
    if not frame_files:
        logger.error(f"No frame images found in {image_dir}")
        return False
    
    logger.info(f"Found {len(frame_files)} frames (after downsampling by {downsample})")
    
    frames = []
    for i, frame_path in enumerate(frame_files):
        try:
            img = Image.open(frame_path)
            frames.append(img)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Loaded {i + 1}/{len(frame_files)} frames")
        except Exception as e:
            logger.warning(f"Failed to load frame {i}: {frame_path} - {e}")
            continue
    
    if not frames:
        logger.error("No frames loaded")
        return False
    
    # Create GIF
    duration = int(1000 / fps)  # milliseconds per frame
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    logger.info(f"✓ GIF saved to {output_path}")
    logger.info(f"  Duration: {len(frames) * duration / 1000:.1f}s at {fps} FPS")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stitch frame images into video or GIF"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/Users/bhavya/Desktop/images",
        help="Images directory (default: /Users/bhavya/Desktop/images)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output filename (default: <scenario_name>.mp4 or .gif)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory for generated video/gif files"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["mp4", "gif", "both"],
        default="mp4",
        help="Output format: mp4, gif, or both (default: mp4)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second (default: 5)"
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsample for GIF (e.g., 2 = every 2nd frame, default: 1)"
    )
    
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return False
    
    # Check if there are frame images directly in this directory
    frame_files = list(images_dir.glob('frame_*.jpg'))
    
    if frame_files:
        # Process frames directly in this directory
        logger.info(f"Found {len(frame_files)} frame images in {images_dir}")
        
        # Create output directory
        output_dir = Path(args.output_dir) if args.output_dir else images_dir / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output filename
        if args.output_file:
            scenario_name = args.output_file.replace('.mp4', '').replace('.gif', '')
        else:
            scenario_name = images_dir.name
        
        success = True
        
        # Generate requested formats
        if args.format in ["mp4", "both"]:
            mp4_path = output_dir / f"{scenario_name}.mp4"
            logger.info(f"Creating MP4 video: {mp4_path}")
            if not stitch_frames_to_video(images_dir, mp4_path, fps=args.fps):
                success = False
        
        if args.format in ["gif", "both"]:
            gif_path = output_dir / f"{scenario_name}.gif"
            logger.info(f"Creating GIF: {gif_path}")
            if not stitch_frames_to_gif(images_dir, gif_path, fps=args.fps, downsample=args.downsample):
                success = False
        
        if success:
            logger.info(f"\n✓ All outputs saved to {output_dir}")
        else:
            logger.error("Some operations failed")
        
        return success
    
    else:
        # Try to process subdirectories
        scenario_dirs = [d for d in images_dir.iterdir() if d.is_dir() and d.name != "videos"]
        
        if not scenario_dirs:
            logger.error(f"No frame images or scenario subdirectories found in {images_dir}")
            return False
        
        logger.info(f"Found {len(scenario_dirs)} scenario directories")
        
        # Create output directory
        output_dir = Path(args.output_dir) if args.output_dir else images_dir / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success = True
        
        # Process each scenario
        for image_dir in scenario_dirs:
            scenario_name = image_dir.name
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {scenario_name}")
            logger.info(f"{'='*60}")
            
            # Generate requested formats
            if args.format in ["mp4", "both"]:
                mp4_path = output_dir / f"{scenario_name}.mp4"
                logger.info(f"Creating MP4 video: {mp4_path}")
                if not stitch_frames_to_video(image_dir, mp4_path, fps=args.fps):
                    success = False
            
            if args.format in ["gif", "both"]:
                gif_path = output_dir / f"{scenario_name}.gif"
                logger.info(f"Creating GIF: {gif_path}")
                if not stitch_frames_to_gif(image_dir, gif_path, fps=args.fps, downsample=args.downsample):
                    success = False
        
        if success:
            logger.info(f"\n✓ All outputs saved to {output_dir}")
        else:
            logger.error("Some operations failed")
        
        return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
