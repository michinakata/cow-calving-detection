# %%
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from copy.models import *
from dataset import CowDataset
from utils import write_event
# %%
PATH_RESULT_OUTPUT = "/mnt/iot-qnap3/nakata/cow-calving-detection/Feature_Extraction/result_output/"
# %%
def main():
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=20,
        help="Number of images in each mini-batch",
    )
    parser.add_argument(
        "--epoch_size",
        "-e",
        type=int,
        default=20,
        help="Number of sweeps over the dataset to train",
    )
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        default=-1,
        help="GPU ID (negative value indicates CPU)",
    )
    parser.add_argument(
        "--train_filepath", "-tf", type=str, help="Filepath (.csv) to train dataset"
    )
    parser.add_argument(
        "--valid_filepath", "-vf", type=str, help="Filepath (.csv) to valid dataset"
    )
    parser.add_argument(
        "--lr", "-lr", type=float, default=1e-4, help="init learning rate"
    )
    parser.add_argument(
        "--patience", "-pa", type=int, default=1, help="cnt of patience"
    )
    parser.add_argument(
        "--max_lr_changes",
        "-mlc",
        type=int,
        default=1,
        help="max cnt of learning rate changed",
    )
    parser.add_argument(
        "--hist", "-hi", type=int, default=0, help="use histgram normalization"
    )
    parser.add_argument("--aug", "-ag", type=int, default=0, help="use augmentation")
    parser.add_argument("--parallel", "-pr", type=int, default=0, help="use parallel")
    parser.add_argument(
        "--use_bn", "-ub", type=int, default=0, help="use batch normalization"
    )
    parser.add_argument("--result_path", type=str)
    args = parser.parse_args()

    print("GPU: {}".format(args.gpu))
    print("# Minibatch-size: {}".format(args.batch_size))
    print("# epoch: {}".format(args.epoch_size))
    print("")
    
    # result_pathの設定
    
    """
    モデル出力パスの生成
    """
    output_root = Path(args.result_path)
    output_root.mkdir(parents = True, exist_ok=True) #ディレクトリが存在しない場合のみ作成
    (output_root / "params.json").write_text(
        json.dumps(vars(args), indent=4, sort_keys=True)
    )
    """
    モデルパスの指定
    """
    model_path = os.path.join(args.result_path, "{}-model.pt")
    best_model_path = os.path.join(args.result_path, "best-model.pt")

    """
    modelの定義, マルチタスクCNN
    """
    # model = SpatialMultitask(args.use_bn)
    # model = Nakata_Net_2(args.use_bn)
    # model = ViT_EfficientNet(args.use_bn)
    # model = ViT_DenseNet(args.use_bn)
    model = ViT_Pytorch_DenseNet(args.use_bn)
    """
    deviceの定義
    """
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    """
    modelをdevice(gpu or cpu)に切り替え
    """
    model = model.to(device)
    
    """
    学習 GPU並列化
    """
    # 並列化 CUDA_VISIBLE_DEVICESで複数GPU番号を指定する必要がある．
    if args.parallel:
        model = torch.nn.DataParallel(model)

    """
    学習 最適化optimizer
    """
    #    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                milestones=[int(args.epoch_size/2), int(args.epoch_size*3/4)], gamma=0.1)

    """
    訓練データ
    """
    train_dataset = CowDataset(
        args.train_filepath, augmentation=args.aug, histgram=args.hist
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    """
    検証データ
    """
    valid_dataset = CowDataset(
        args.valid_filepath, augmentation=False, histgram=args.hist
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    """
    学習時のパラメータ
    """
    patience = args.patience #patienceの回数validation lossが下がったら学習率を減衰する
    lr = args.lr #学習率
    max_lr_changes = args.max_lr_changes #学習率を変更する最大回数
    lr_changes = 0  #学習率を変更した回数
    lr_reset_epoch = 1 #
    valid_losses = [] #検証時のロスを記録する配列
    best_valid_loss = float("inf")

    """
    並列化する場合
    """
    if args.parallel:
        #def save(ep): torch.save({"model": ,...,"epoch": ep,...,"best_valid_loss": ,...}, str(model_path).format(ep))と同じ
        save = lambda ep: torch.save(
            {
                "model": model.module.state_dict(),
                "epoch": ep,
                "best_valid_loss": best_valid_loss,
            },
            str(model_path).format(ep),
        )
    else: #並列化しない場合
        #def save(ep): torch.save({"model": ,...,"epoch": ep,...,"best_valid_loss": ,...}, str(model_path).format(ep))と同じ
        save = lambda ep: torch.save(
            {
                "model": model.state_dict(),
                "epoch": ep,
                "best_valid_loss": best_valid_loss,
            },
            str(model_path).format(ep),
        )

    head_weights, tail_weights = train_dataset.get_weight()

    """
    最適化関数
    """
    #    head_criterion = nn.MSELoss()
    #    tail_criterion = nn.MSELoss()
    # head_criterion = nn.BCELoss(torch.Tensor(head_weights[:-1]).view(3, 3).cuda()) #二値分類
    # tail_criterion = nn.BCELoss(torch.Tensor(tail_weights[:-1]).view(3, 3).cuda()) #二値分類
    head_criterion = nn.BCELoss(torch.Tensor(head_weights[:-1]).view(3, 3).to(device)) #二値分類
    tail_criterion = nn.BCELoss(torch.Tensor(tail_weights[:-1]).view(3, 3).to(device)) #二値分類
    pose_criterion = nn.CrossEntropyLoss() #多クラス分類

    log = output_root.joinpath("train.log").open("at", encoding="utf8")

    for epoch in range(1, args.epoch_size + 1):
        #tqdm進捗bar
        tq = tqdm(total=(args.epoch_size or len(train_loader) * args.batch_size))
        #進捗barに説明文を設定
        tq.set_description(f"Epoch {epoch}, lr {lr}")
        """
        訓練段階
        訓練データ(train_loader)
        学習の流れ
        ・DataLoaderからデータ読み込み
        ・ロスを計算(頭部,尾部,姿勢,頭部・尾部同時発生)
        ・誤差逆伝播
        ・Lossの表示
        """
        total_loss = [0, 0, 0] #頭部ロス, 尾部ロス, 姿勢ロス
        total_size = 0 #
        model.train() #マルチタスクCNN modelを学習モードに切り替える
        
        for batch_idx, (_, data, head_lbl, tail_lbl, pose_lbl) in enumerate(
            train_loader
        ):
            """
            _: 画像名
            data: 画像FloatTensorの値
            head_lbl: 頭部
            tail_lbl: 尾部
            pose_lbl: 姿勢
            """
            #CUDAで扱える形に変換
            # data, head_lbl, tail_lbl, pose_lbl = (
            #     data.cuda(),
            #     head_lbl.cuda(),
            #     tail_lbl.cuda(),
            #     pose_lbl.cuda(),
            # )
            data, head_lbl, tail_lbl, pose_lbl = (
                data.to(device),
                head_lbl.to(device),
                tail_lbl.to(device),
                pose_lbl.to(device),
            )
            
            optimizer.zero_grad() #計算した勾配の初期化
            """
            output[0]: 
            output[1]: 
            マルチタスクCNNにTensor dataを入力する
            """
            output = model(data)
            
            #頭部のロス, head_criterion(予測出力,頭部), バイナリクロスエントロピー
            loss_head = head_criterion(
                output[1][:, 0, :, :], head_lbl[:, :-1].view(-1, 3, 3)
            )
            #尾部のロス, tail_criterion(予測出力,尾部), バイナリクロスエントロピー
            loss_tail = tail_criterion(
                output[1][:, 1, :, :], tail_lbl[:, :-1].view(-1, 3, 3)
            )
            #姿勢のロス, pose_criterion(予測出力,姿勢情報), クロスエントロピー
            loss_pose = pose_criterion(output[0], pose_lbl)
            #首と尾の位置を表す2つのヒートマップの要素和であり,同じ場所に首と尾の位置が存在しないという制約項
            #output[1][:, 0, :, :]:頭部のロス, output[1][:, 1, :, :]:尾部のロス　に関して積の和を取る
            #積の和が0以上の場合,尾と首位置が重なってしまっているため,損失が拡大する
            #cooccur: 同時に起こる
            #data.size(0): テンソルの第一軸の次元数
            loss_cooccur = torch.sum(
                (output[1][:, 0, :, :] * output[1][:, 1, :, :])
            ) / data.size(0)

            #全ロスの合計(頭部,尾部,姿勢ロス,頭部尾部同時発生ロス)
            loss = loss_head + loss_tail + loss_pose + loss_cooccur
            """
            バックプロパゲーション(誤差逆伝播)
            """
            loss.backward() #誤差逆伝播
            optimizer.step() #重みの更新を行う, 学習率と最適化手法に基づいて重みを更新
            total_size += data.size(0) #テンソルの第一軸の次元数

            for i, x in enumerate(["head", "tail", "pose"]):
                exec("total_loss[{}] += loss_{}.item()".format(i, x))

            """
            Lossの表示
            """
            if batch_idx % (len(train_loader) // 2) == 0:
                print(
                    "\nHEAD Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        total_loss[0] / total_size,
                    )
                )
                print(
                    "TAIL Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        total_loss[1] / total_size,
                    )
                )
                print(
                    "POSE Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        total_loss[2] / total_size,
                    )
                )
            tq.update(data.size(0))

        write_event(log, "train", epoch, loss=total_loss)


        """
        検証段階
        """
        model.eval() #modelを評価用に変える
        tq.close() #tqdmをwith文で使わない場合は
        total_loss = [0, 0, 0] #頭部ロス, 尾部ロス, 姿勢ロス
        total_size = 0 #total_size:
        correct = [0, 0, 0] #正解ラベル
        """
        検証データ(valid_loader)
        with torch.no_grad()
        自動微分での勾配計算は無効になる
        勾配更新しない
        検証の流れ
        ・DataLoaderからのデータ(imgname,imgFloatTensor,head,tail,pose)の読み込み
        ・モデルにテンソルを入力
        ・ロスを計算(頭部,尾部,姿勢,頭部・尾部同時発生)
        ・total_loss: [頭部ロス,尾部ロス,姿勢ロス]
        """
        with torch.no_grad():
            for batch_idx, (_, data, head_lbl, tail_lbl, pose_lbl) in tqdm(
                enumerate(valid_loader)
            ):
                """
                検証データラベル
                _: 画像名
                data: 画像FloatTensorの値
                head_lbl: 頭部
                tail_lbl: 尾部
                pose_lbl: 姿勢
                """
                # data, head_lbl, tail_lbl, pose_lbl = (
                #     data.cuda(),
                #     head_lbl.cuda(),
                #     tail_lbl.cuda(),
                #     pose_lbl.cuda(),
                # )
                data, head_lbl, tail_lbl, pose_lbl = (
                data.to(device),
                head_lbl.to(device),
                tail_lbl.to(device),
                pose_lbl.to(device),
                )
                """
                検証データに対する予測値
                output[0]: 姿勢
                output[1][:, 1, :, :]: 頭部
                output[1][:, 1, :, :]: 尾部
                """
                output = model(data)

                #頭部のロス, head_criterion(予測出力,頭部)
                loss = head_criterion(
                    output[1][:, 0, :, :], head_lbl[:, :-1].view(-1, 3, 3)
                )
                total_loss[0] += loss.item() #loss.item()でlossの要素を取得する
                #尾部のロス, head_criterion(予測出力,尾部)
                loss = tail_criterion(
                    output[1][:, 1, :, :], tail_lbl[:, :-1].view(-1, 3, 3)
                )
                total_loss[1] += loss.item() #loss.item()でlossの要素を取得する
                #姿勢のロス, head_criterion(予測出力,姿勢)
                loss = pose_criterion(output[0], pose_lbl)
                total_loss[2] += loss.item() #loss.item()でlossの要素を取得する
                
                """
                頭部予測
                """
                head_predicted = (
                    output[1][:, 0, :, :].view(-1, 9).max(1, keepdim=True)[1]
                )
                # 9マスすべてconfidenceが0.5以下のとき garbageクラスを予測したこととする
                idx = np.where((output[1][:, 0, :, :].view(-1, 9) < 0.5).sum(1).to('cpu').detach().numpy().copy() == 0)
                #head_predictedを9にする
                head_predicted[idx] = 9
                correct[0] += (
                    head_predicted.view(-1)
                    .eq(head_lbl[:, :].argmax(dim=1))
                    .sum()
                    .item()
                )

                """
                尾部予測
                """
                tail_predicted = (
                    output[1][:, 1, :, :].view(-1, 9).max(1, keepdim=True)[1]
                )
                # 9マスすべてconfidenceが0.5以下のとき garbageクラスを予測したこととする
                #.to('cpu').detach().numpy().copy(): TorchTensorをcpuに変換
                idx = np.where((output[1][:, 1, :, :].view(-1, 9) < 0.5).sum(1).to('cpu').detach().numpy().copy() == 0)
                tail_predicted[idx] = 9
                # 尾位置の予測値と尾位置の正解ラベルが一致している数をカウント
                correct[1] += (
                    tail_predicted.view(-1)
                    .eq(tail_lbl[:, :].argmax(dim=1))
                    .sum()
                    .item()
                )

                """
                姿勢予測
                """
                state_predicted = output[0].max(1, keepdim=True)[1] #姿勢予測テンソル
                #姿勢予測テンソルを平坦化し正解ラベルと同じ数を合計する
                correct[2] += (state_predicted.flatten() == pose_lbl).sum().item()

                total_size += data.size(0) #テンソルの第一軸の次元数

        val_loss = np.array(total_loss) / len(valid_loader) #ロス
        val_acc = np.array(correct) / total_size #正解数をtotal_sizeで割り,正解率を求める.

        valid_losses.append(sum(val_loss))

        write_event(
            log, "valid", epoch, loss=list(val_loss), head_tail_pose_acc=list(val_acc)
        )
        #マルチタスクCNN modelの保存
        save(epoch)

        #best_modelの保存
        if sum(val_loss) < best_valid_loss: #lossがbest_valid_lossよりも小さい場合
            #best_valid_lossの更新 
            best_valid_loss = sum(val_loss) 
            #best_modelの保存
            shutil.copy(str(model_path).format(epoch), str(best_model_path)) #
        elif (
            patience
            and epoch - lr_reset_epoch > patience
            and min(valid_losses[-patience:]) > best_valid_loss
        ):
            """
            学習率の変更
            """
            lr_changes += 1 #学習率変更回数インクリメント
            if lr_changes > max_lr_changes: #学習率の変更回数が上限を超えたら学習終了
                break
            lr /= 5 #学習率の変更
            print(f"lr updated to {lr}")
            lr_reset_epoch = epoch #lr_resnet_epochの更新
            #  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            #optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr)

        print(
            "\nHEAD Validation Epoch: {} Avg loss: {:.6f}, Avg acc: {:.6f}".format(
                epoch, val_loss[0], val_acc[0]
            )
        )
        print(
            "TAIL Validation Epoch: {} Avg loss: {:.6f}, Avg acc: {:.6f}".format(
                epoch, val_loss[1], val_acc[1]
            )
        )
        print(
            "POSE Validation Epoch: {} Avg loss: {:.6f}, Avg acc: {:.6f}".format(
                epoch, val_loss[2], val_acc[2]
            )
        )


#        if args.pararell:
#            torch.save(model.module.state_dict(), os.path.join(args.output,
#                    'ep{}_valloss{}-{}-{}_valacc{}-{}-{}.model'.format(epoch,
#                        val_loss[0], val_loss[1], val_loss[2],
#                        val_acc[0], val_acc[1], val_acc[2])))
#        else:
#            torch.save(model.state_dict(), os.path.join(args.output,
#                    'ep{}_valloss{}-{}-{}_valacc{}-{}-{}.model'.format(epoch,
#                        val_loss[0], val_loss[1], val_loss[2],
#                        val_acc[0], val_acc[1], val_acc[2])))


if __name__ == "__main__":
    main()
