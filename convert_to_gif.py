from moviepy import VideoFileClip

video_path = "Traffic Simulation for Traffic Flow Control.mp4"

clip = VideoFileClip(video_path)

# 🔥 pick a moving section (VERY IMPORTANT)
clip = clip.subclipped(2, 12)   # skip first 2 seconds

# 🔥 resize for performance
clip = clip.resized(width=900)

# 🔥 smoother animation
clip.write_gif("demo.gif", fps=25)

print("Better GIF created!")