#!/usr/bin/env bash

# 変更するパラメータはDYASとFARM_IDのみ
PATH_ROOT="/mnt/..." # 変更不要
PATH_VIDEOS="/mnt/..." # 変更不要

START_DATE="20220713"
END_DATE="20220720" # この日は含まれない
FARM_ID="..." # 共同研究のため公開しない

# 開始日から終了日までの日付を生成しループ
CURRENT_DATE=$START_DATE
echo $FARM_ID
while [[ $CURRENT_DATE != $END_DATE ]]; do
    echo $CURRENT_DATE
    # 作成したディレクトリに.tarファイルを解凍し画像に変換
    echo "tarファイル解凍中"
    pv ${PATH_ROOT}/${FARM_ID}/${FARM_ID}-${CURRENT_DATE}.tar | tar -xzf - -C ${PATH_ROOT}/${FARM_ID}/
    # tarファイルを削除
    echo "tarファイル削除中"
    rm $PATH_ROOT/${FARM_ID}/${FARM_ID}-${CURRENT_DATE}.tar
    # pythonスクリプトを実行し画像を動画に変換
    echo "動画変換中"
    python3 /mnt/.../make_videos.py --farm_ID $FARM_ID \
                                                                                        --day $CURRENT_DATE \
    # 次の日付を生成
    CURRENT_DATE=$(date -d "$CURRENT_DATE + 1 day" +%Y%m%d)
done
