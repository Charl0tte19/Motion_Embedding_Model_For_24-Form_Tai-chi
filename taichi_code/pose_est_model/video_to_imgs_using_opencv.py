import cv2
import argparse
import os
# opencv取得的fps還可以，但取得的total_frame有可能很不準，所以用它來算duration也有可能很不準
# ffmpeg取得的fps也可以，duration(秒數)也可以，由此兩算出來的total_frame也可以
# ffmpeg -i 00_00_40_1605.mp4 -filter:v fps=10 -an 00_00_40_1605_fps10.mp4 產生的fps10影片，和原本的影片，速度是一樣的，只是total_frame數不同，比如說3010，因此現在的total_frame數會是原本的1/3
# 我測試的結果 ffmpeg -i 00_00_40_1605.mp4 -r 10 -an 00_00_40_1605_fps10.mp4 產生的fps10影片，效果和-vf的差不多
# ffmpeg -i 00_00_40_1605_fps10.mp4 -filter:v setpts=2*PTS -an 00_00_40_1605_fps10_speed2.mp4 產生的fps10_speed2，速度會是fps100p也會變成fps10影片的兩倍
# FFmpeg中的setpts濾鏡用於調整PTS（Presentation Timestamp），從而改變影片的播放速度。使用setpts=2*PTS會使每個影格的時間戳記變成原來的2倍，導致播放速度減半。但如果你想保持幀數不變而只是拉長影片的播放時間，你還需要調整幀率（fps）。
# 總之，直接用change_video_fps_and_speed.sh吧
def main(args):
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(total_frames, frame_width, frame_height, video_fps)

    frame_ith = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("Finish or Error!!")
            break

        cv2.imwrite(os.path.join(args.save_folder,f"img_{frame_ith:04}.jpg"), frame)
        frame_ith += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, default='./example/test.mp4',
                        help='input video path')
 
    parser.add_argument('--save-folder', type=str, default='imgs',
                        help='folder path where result images will be saved')

    args = parser.parse_args()

    main(args)