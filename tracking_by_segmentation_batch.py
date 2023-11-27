# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mmcv
import mmengine
import cv2
import torch
import tqdm
import argparse
import warnings
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample

warnings.simplefilter('ignore')
# %%
PATH_CHECKPOINT = "/mnt/..." # 学習済みモデルのconfigファイルとcheckpointファイルのpath(固定)
PATH_IMG_ROOT = "/mnt/..." # 入力画像データのpath(固定)
PATH_MASKS_ROOT = "/mnt/..." # トラッキングに必要な分娩房マスク画像のpath(固定)
PATH_DB = "/mnt/..." # 各種テーブルcsvデータのpath(固定)
PATH_SAVE_IMGS_ROOT = "/mnt/..." # 切り取ったbbox画像の出力path
PARH_SAVE_CSV_ROOT = "/mnt/..." # BBOXの座標・確信度スコアを記録するcsvファイルの出力path
PATH_SAVE_VIDEOS_ROOT = "/mnt/..." # トラッキング確認用の可視化動画データの出力path
# %%
class MyDataset(Dataset):
    def __init__(self, path_root, filenames):
        self.path_root = path_root
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path_img = os.path.join(self.path_root, self.filenames[idx])
        img_rgb = mmcv.imread(path_img, channel_order="rgb")
        img_bgr = mmcv.imread(path_img)
        filename = self.filenames[idx]
        return img_rgb, img_bgr, filename
# %%
def custom_collate(batch):
    return list(zip(*batch))
# %%
def get_img_and_bboxes(img, result, mask_array, model, df, visualizer, thr): # 画像1枚を入力して、可視化画像1枚とBBOXの座標・確信度スコアを出力する関数。提案アルゴリズムの部分。
    labels = result.pred_instances.labels.cpu().numpy()
    if len(labels) == 0:
        return img, [], []
    select_list = [] # 各出力BBOXの重複部分の面積を格納するリスト
    for i in range(len(labels)): #出力された各BBOXについて判定を行う
        if df[1][labels[i]] == "animal" and result.pred_instances.scores[i] > thr: # 「動物」かつ「確信度が指定した閾値以上」のBBOXのみを採用
            
            cow_binary_image = result.pred_instances.masks.cpu().numpy()[i] # 対象BBOXのピクセル領域を取得
            cow_array = np.where(cow_binary_image, 100, 0) # (True, False)を(100,0)に変換
            sum_array = cow_array + mask_array # 分娩房領域マスクと加算
            unique_values, counts = np.unique(sum_array, return_counts=True) # 加算した画像は(0,100,200)の3つの値のみで構成されている
            pixel_num = 0
            for value, count in zip(unique_values, counts):
                if value == 200: # 200の部分が物体領域と分娩房領域の重複部分
                    pixel_num = count # 重複部分の面積を取得
                    break
            select_list.append(pixel_num)
        else:
            select_list.append(0)
    max_num = max(select_list) # 各重複領域の最大値を取得(例：各bboxの重複面積が[0,0,10000,20000,0]なら20000)
    max_index = select_list.index(max_num) # 最大値のbboxのうち、最初のbboxのみを選択し、そのインデックスを取得(例：各bboxの重複面積が[20000,0,10000,20000,0]なら0番目)
    if max_num > 0: # 全部0、では無い場合
        select_list = [1 if i == max_index else 0 for i in range(len(select_list))] # 最大値は1、最大値以外は0(例：各bboxの重複面積が[20000,0,10000,20000,0]なら[1,0,0,0,0])
    else: # 全部0の場合は、対象牛が映っていないと判定し、何も出力しない
        select_list = [0 for i in range(len(select_list))]
    select_list = torch.tensor(select_list)
    
    data_sample = DetDataSample()
    pred_instances = InstanceData()
    for att in ["bboxes", "labels", "masks", "scores"]:
        exec("pred_instances." + att + "=result.pred_instances." + att + "[select_list == 1]")
    data_sample.pred_instances = pred_instances
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=data_sample,
        draw_gt = None,
        wait_time=0,
        pred_score_thr=0,
    )
    scores = result.pred_instances.scores[select_list == 1]
    bboxes = result.pred_instances.bboxes[select_list == 1]
    return visualizer.get_image(), scores, bboxes
# %%
def main(): # 入力画像から、(1)クロップ済み画像、(2)bbox情報のcsv、(3)確認用の動画、を出力する
    # 引数処理（農場ID、日付）
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm_ID", type=str)
    parser.add_argument("--day", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config_name", type=str, default="mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco.py")
    parser.add_argument("--checkpoint_name", type=str, default="mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth")
    parser.add_argument("--thr", type=float, default=0.1, help="対象牛判定のクラス確信度の閾値(高いほど厳しく検出されにくい)")
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    
    # モデルと可視化の設定
    config_file = os.path.join(PATH_CHECKPOINT, args.config_name)
    checkpoint_file = os.path.join(PATH_CHECKPOINT, args.checkpoint_name)
    register_all_modules()
    model = init_detector(config_file, checkpoint_file, device=args.device)
    df_coco = pd.read_csv(os.path.join(PATH_DB, "coco_labels.csv"), header=None)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.mask_color="red"
    visualizer.bbox_color="red"
    visualizer.alpha=0.3 # 画像とマスクの濃さのパラメータ。値が大きすぎるととマスクが濃すぎて何の物体が隠れているか分からなくなる。
    
    # 分娩房マスクの読み込み
    path_mask_img = os.path.join(PATH_MASKS_ROOT, args.farm_ID + ".png")
    image = cv2.imread(path_mask_img)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = gray_image > 1
    mask_array = np.where(binary_image, 100, 0) # (True, False)を(100,0)に変換
    
    # 動画の設定(全体)
    df_farm_info = pd.read_csv(os.path.join(PATH_DB, "farm_info.csv"), index_col=0)
    height, width = df_farm_info["縦幅"][int(args.farm_ID)], df_farm_info["横幅"][int(args.farm_ID)]
    codec = "H264"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    # 入力画像と保存pathの設定
    path_imgs_root_day = os.path.join(PATH_IMG_ROOT, args.farm_ID, args.farm_ID + "-" + args.day)
    hours = os.listdir(path_imgs_root_day)
    path_save_videos_root = os.path.join(PATH_SAVE_VIDEOS_ROOT, args.farm_ID, args.day)
    os.makedirs(path_save_videos_root, exist_ok=True)
    path_save_csv_root = os.path.join(PARH_SAVE_CSV_ROOT, args.farm_ID, args.day)
    os.makedirs(path_save_csv_root, exist_ok=True)
    
    no_bbox_count = 0
    bbox_count = 0
    for hour in tqdm.tqdm(hours, leave=False): # 1時間分の動画ずつ実行
        # 動画の設定(詳細)
        path_save_videos = os.path.join(path_save_videos_root, hour.split("-")[1] + ".mp4")
        writer = cv2.VideoWriter(path_save_videos, fourcc, 10,  (width, height))
        
        # csvの設定
        path_save_csv = os.path.join(path_save_csv_root, hour.split("-")[1] + ".csv")
        df_bbox = pd.DataFrame()
        new_columns = ["img_name", "score", "x_min", "x_max", "y_min", "y_max"]
        for col in new_columns:
            df_bbox[col] = None
        df_bbox = df_bbox.set_index("img_name")
        # bbox画像保存pathの設定
        path_save_imgs_root = os.path.join(PATH_SAVE_IMGS_ROOT, args.farm_ID, args.day, hour.split("-")[1])
        os.makedirs(path_save_imgs_root, exist_ok=True)
        
        # 各画像名の取得
        path_imgs_root_hour = os.path.join(path_imgs_root_day, hour)
        filenames = os.listdir(path_imgs_root_hour)
        filenames = sorted(filenames)
        dataset_inference = MyDataset(path_imgs_root_hour, filenames)
        dataloader_inference = DataLoader(dataset_inference, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
        for batch in tqdm.tqdm(dataloader_inference, leave=False):
            imgs_rgb, imgs_bgr, filenames = batch
            results = inference_detector(model, imgs=imgs_rgb)
            for index in range(len(results)):
                img_visualized, scores, bboxes = get_img_and_bboxes(img=imgs_rgb[index], result=results[index], mask_array=mask_array, model=model, df=df_coco, visualizer=visualizer, thr=args.thr)
                if len(scores) > 0: # 対象牛が映っている場合
                    x_min, y_min, x_max, y_max = bboxes[0].cpu().numpy()
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    score = scores[0].cpu().numpy()
                    df_bbox.loc[filenames[index].split("-")[1].split(".")[0]] = [score, x_min, x_max, y_min, y_max]
                    img_cropped = imgs_bgr[index][y_min:y_max, x_min:x_max, :]
                    path_save_img = os.path.join(path_save_imgs_root, filenames[index].split("-")[1])
                    cv2.imwrite(path_save_img, img_cropped)
                    bbox_count += 1
                else: # 対象牛が映っていない場合
                    no_bbox_count += 1
                img_visualized = cv2.cvtColor(img_visualized, cv2.COLOR_BGR2RGB)
                writer.write(img_visualized)
        writer.release() # 動画を出力
        df_bbox.to_csv(path_save_csv) # bbox情報のcsvを出力
        # break
    print(f"bbox : {bbox_count}, No bbox : {no_bbox_count}") # 対象牛が映っている枚数と映っていない枚数を表示
# %%
if __name__ == "__main__":
    main()