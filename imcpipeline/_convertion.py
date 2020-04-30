import os
from pathlib import Path as P
import pandas as pd


df = pd.read_csv('metadata/annotation.csv').set_index('acquisition_date')
df.index = df.index.astype(str)

cols = ['roi_number', 'sample_number', 'panorama_number', 'roi_names']
# get roi_number to be the second label
rois = (
    df[cols[0]]
    .str.split(',')
    .apply(pd.Series).stack()
    .reset_index(level=1, drop=True)
    .rename(cols[0]))

rescaled = list()
for col in cols[1:]:
    rescaled.append(
        df[col]
        .str.split(',')
        .apply(pd.Series)
        .stack()
        .reset_index(level=1, drop=True)
        .rename(col)
        .to_frame()
        .set_index(rois, append=True)
        .squeeze())

df2 = pd.DataFrame(rescaled).T.reset_index(level=1)
df2 = df2.join(df.drop(cols, axis=1))
df2['roi_name'] = df2.index.astype(str) + "-" + df2['roi_number']

df2 = df2.set_index(['roi_name'], append=True).reorder_levels([1, 0], axis=0)
df2.to_csv('metadata/annotation.per_roi.csv')


# Rename files to use only the sample/roi name
for sample, row in df.dropna(subset=['acquisition_name']).iterrows():
    p = P("processed") / sample / "tiffs"
    if not p.is_dir():
        continue
    files = pd.Series([q.parts[-1] for q in p.iterdir()])
    files = files[files.str.contains(row['acquisition_name'])].sort_values()
    ex = files.str.extract(r"^(" + row['acquisition_name'] + r")_.*_r(\d+)_.*?_ac(_\w+.\w+)")
    repl = sample + '-' + ex[1].astype(str) + ex[2]
    for f, t in zip(files.tolist(), repl):
        print(p / f, p / t)
        os.rename(p / f, p / t)

    # Rename file endings
    cmds = [
        f"rename 's/_ilastik_s2_Probabilities_mask.tiff/_full_mask.tiff/g' {p}/*.tiff",
        f"rename 's/_ilastik_s2_Probabilities_NucMask.tiff/_full_nucmask.tiff/g' {p}/*.tiff",
        f"rename 's/_ilastik_s2_Probabilities.tiff/_Probabilities.tiff/g' {p}/*.tiff"]
    for cmd in cmds:
        os.system(cmd)

    # rename ometiff folder (or better find a way to have channel metadata more accessible)
    p = (P("processed") / sample / "ometiff").absolute()
    for d in p.iterdir():
        print(d, p / sample)
        os.rename(d, p / sample)
        d = p / sample

        # and files (not touching the ROI names for now)
        for f in d.iterdir():
            print(f, str(f).replace(row.acquisition_name, sample))
            os.rename(f, str(f).replace(row.acquisition_name, sample))
