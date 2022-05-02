from importlib.resources import path
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

back_hand_path = "../Cut-data/data-v2/backhand"
fore_hand_path = "../Cut-data/data-v2/forehand"
serve_path = "../Cut-data/data-v2/serve"
# warmup_path = "../Cut-data/warmup"

# demo_path = "../Video-data/cut-data/demo-data/"

# print(os.path.abspath(back_hand_path))

def get_list_path(data_path):
    back_hand_list_path = []
    # fore_hand_list_path = []
    # serve_list_path = []

    for file in os.listdir(data_path):
        path = os.path.join(os.path.abspath(data_path), file)
        back_hand_list_path.append(path)
        
    return back_hand_list_path

def concate_video(video1, video2):
    clip_1 = VideoFileClip(video1)
    clip_2 = VideoFileClip(video2)
    final_clip = concatenate_videoclips([clip_1,clip_2])
    # final_clip.write_videofile("final.mp4")
    return final_clip

def create_video_label(video_list):
    clip = concate_video(video_list[0], video_list[1])
    for i in range(2, len(video_list)):
        tmp_clip =  VideoFileClip(video_list[i])
        clip = concatenate_videoclips([clip, tmp_clip])

    return clip

# print("main")

if __name__ == "__main__":
    # print("main")
    back_hand_list = get_list_path(back_hand_path)
    fore_hand_list = get_list_path(fore_hand_path)
    serve_list = get_list_path(serve_path)
    # warmup_list = get_list_path(warmup_path)

    # demo_list = get_list_path("C:\\Users\\Administrator\\Desktop\\demo")

    back_hand_video = create_video_label(back_hand_list)
    back_hand_video.write_videofile("./video/video-v2/back_hand.mp4")

    # fore_hand_video = create_video_label(fore_hand_list)
    # fore_hand_video.write_videofile("./video/video-v2/fore_hand.mp4")

    # serve_video = create_video_label(serve_list)
    # serve_video.write_videofile("./video/video-v2/serve.mp4")

    # warmup_video = create_video_label(warmup_list)
    # warmup_video.write_videofile("./warmup.mp4")

    # demo_video = create_video_label(demo_list)
    # demo_video.write_videofile("./demo.mp4")
