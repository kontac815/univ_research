import pandas as pd

# 入力ファイルパス
IN_PATH = "../data/hpo_master_with_def_short.csv"

# 出力ファイルパス
OUT_DEPTH_GE3 = "../data/HPO_depth_ge3.csv"
OUT_DEPTH_GE3_SELF = "../data/HPO_depth_ge3_self_reportable.csv"

# CSV を読み込み
df = pd.read_csv(IN_PATH)

# ① depth >= 3 のものを抽出
df_depth_ge3 = df[df["depth"] >= 3.0]

# ② depth >= 3 かつ self_reportable == 1 のものを抽出
#   （self_reportable が 1/0 のフラグだと仮定）
df_depth_ge3_self = df[(df["depth"] >= 3.0) & (df["self_reportable"] == 1.0)]

# CSV として保存
df_depth_ge3.to_csv(OUT_DEPTH_GE3, index=False)
df_depth_ge3_self.to_csv(OUT_DEPTH_GE3_SELF, index=False)

print(f"depth>=3: {len(df_depth_ge3)} 行を書き出しました -> {OUT_DEPTH_GE3}")
print(f"depth>=3 & self_reportable==1: {len(df_depth_ge3_self)} 行を書き出しました -> {OUT_DEPTH_GE3_SELF}")
