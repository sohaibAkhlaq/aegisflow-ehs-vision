import imageio
import sys

def convert_webp_to_mp4(webp_path, mp4_path, fps=30):
    print(f"Reading {webp_path}...")
    try:
        reader = imageio.get_reader(webp_path)
        print("Initializing MP4 writer...")
        writer = imageio.get_writer(mp4_path, fps=fps, codec='libx264', quality=8)
        
        count = 0
        for frame in reader:
            writer.append_data(frame)
            count += 1
            if count % 50 == 0:
                print(f"Converted {count} frames...")
        
        writer.close()
        print(f"Successfully converted {count} frames to {mp4_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_webp.py input.webp output.mp4")
        sys.exit(1)
    
    convert_webp_to_mp4(sys.argv[1], sys.argv[2])
