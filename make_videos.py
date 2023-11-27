# %%
import os
import cv2
import argparse
import tqdm
import pandas as pd
# %%
# 共同研究のためパスは公開しない
PATH_DB = "/mnt/..."
PATH_SAVE = "/mnt/..."
PATH_IMG = "/mnt/..."
# %%
def make_videos(farm_ID, day, df):
    video_save_dir = os.path.join(PATH_SAVE, farm_ID)
    os.makedirs(video_save_dir, exist_ok=True)
    video_save_path = video_save_dir + "/" + day + ".mp4"
    height, width = df["height"][int(farm_ID)], df["width"][int(farm_ID)]
    print(height, width)
    codec = "H264"
    fourcc = cv2.VideoWriter_fourcc(*codec)    
    writer = cv2.VideoWriter(video_save_path, fourcc, 10,  (width, height))
    img_root_dir = os.path.join(PATH_IMG, farm_ID, farm_ID + "-" + day)
    for hour in tqdm.tqdm(os.listdir(img_root_dir)):
        filenames = os.listdir(os.path.join(img_root_dir, hour))
        filenames = sorted(filenames)
        for i, filename in enumerate(filenames):
            if i % 20 == 0:
                img = cv2.imread(os.path.join(img_root_dir, hour, filename))
                text = filename[-18:-14] + "/" + filename[-14:-12] + "/" + filename[-12:-10] + "-" + filename[-10:-8] + ":" + filename[-8:-6] + "-" + filename[-6:-4]
                cv2.putText(img=img, text=text, org=(15,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0,0,255), thickness=2)
                writer.write(img)
    writer.release()
# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm_ID", type=str)
    parser.add_argument("--day", type=str)
    args = parser.parse_args()
    csv_path = os.path.join(PATH_DB, "farm_info_selected_1.csv")
    df = pd.read_csv(csv_path, index_col=0)
    make_videos(farm_ID=args.farm_ID, day=args.day, df=df)
# %%
if __name__ == "__main__":
    main()
