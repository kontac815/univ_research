import pandas as pd

def process_hpo_files():
    # 1. ファイルの読み込み
    try:
        df_master = pd.read_csv("../data/archived/hpo_master_all_jp.weblio.with_llm_jp.v2.csv")
        df_symptom = pd.read_csv("../data/archived/HPO_symptom_depth_leq3_with_jp.merged.clean_llm.csv")
        df_def = pd.read_csv("../data/hpo_master_with_def.csv")
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。ファイル名を確認してください。\n{e}")
        return

    # 2. マスターファイルから指定カラムを抽出
    # ユーザー指定の "2" は category_v2 であると推定して処理します
    target_columns = [
        "HPO_ID", "name_en", "jp_final", "category", "sub_category", "depth", 
        "self_reportable", "parents", "children"
    ]
    
    # 念のため、存在するカラムのみを選択（エラー回避）
    existing_cols = [c for c in target_columns if c in df_master.columns]
    df_base = df_master[existing_cols].copy()

    # HPO_ID をインデックスに設定（検索・更新を高速化するため）
    df_base.set_index("HPO_ID", inplace=True)

    # 3. HPO_symptom_depth_leq3... の jp_final で上書き
    # 更新用のデータを作成（HPO_IDとjp_finalのみ）
    if "jp_final" in df_symptom.columns:
        df_update = df_symptom[["HPO_ID", "jp_final"]].copy()
        df_update.set_index("HPO_ID", inplace=True)
        
        # NaN（空白）でない値のみで上書きを実行
        # updateメソッドはインデックス(HPO_ID)が一致する行のデータを更新します
        df_update = df_update.dropna(subset=["jp_final"])
        df_base.update(df_update)
        print("jp_finalの上書き処理が完了しました。")
    else:
        print("警告: 更新用ファイルに jp_final カラムが見つかりませんでした。")

    # 4. hpo_master_with_def の definition_ja を列として追加
    # インデックスを列に戻してマージの準備
    df_base.reset_index(inplace=True)

    if "definition_ja" in df_def.columns:
        # 必要な列のみ抽出
        df_def_subset = df_def[["HPO_ID", "definition_ja"]].copy()
        
        # 左外部結合 (Left Join) で定義を追加
        # df_baseにあるHPO_IDは全て残し、定義がある場合のみ紐付けます
        df_final = pd.merge(df_base, df_def_subset, on="HPO_ID", how="left")
        print("definition_jaの列追加が完了しました。")
    else:
        df_final = df_base
        print("警告: 定義用ファイルに definition_ja カラムが見つかりませんでした。")

    # 5. 結果の保存
    output_filename = "hpo_master_processed_result.csv"
    # 日本語文字化け防止のため utf-8-sig を使用
    df_final.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"処理完了: {output_filename} として保存しました。")

if __name__ == "__main__":
    process_hpo_files()