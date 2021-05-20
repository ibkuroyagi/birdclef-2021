import pandas as pd
from tqdm import tqdm


def relabel_name(df1, source_sec=20, target_sec=5):
    tmp_df = pd.DataFrame(columns=df1.columns)
    for i, items in enumerate(tqdm(df1.values)):
        start = int(items[0].split("/")[-1].split(".")[0].split("_")[1])
        end = int(items[0].split("/")[-1].split(".")[0].split("_")[2])
        for sec in range(start, end, 5):
            path = (
                items[0]
                .replace(f"{source_sec}sec", f"{target_sec}sec")
                .replace(f"_{start}", f"_{sec}")
                .replace(f"_{end}", f"_{sec+5}")
            )
            tmp_df = tmp_df.append(
                [path, items[1], items[2], items[3]],
                ["path", "birds", "dataset", "fold"],
            )
    return tmp_df


a = pd.read_csv("dump/relabel20sec/b0_mixup2/relabel.csv")
b = pd.read_csv("dump/relabel5sec/b0_mixup2/relabel.csv")
df2 = relabel_name(a)
df2.to_csv("df2.csv", index=False)
tmp1 = a[a["birds"] != "nocall"]
tmp2 = b[b["birds"] != "nocall"]
tmp3 = df2[df2["birds"] != "nocall"]
for i in range(5):
    cnt1 = tmp1["fold"] == i
    cnt2 = tmp2["fold"] == i
    cnt3 = tmp3["fold"] == i
    print(f"{i}: a:{sum(cnt1)}, b:{sum(cnt2)}, df2:{sum(cnt3)}")
