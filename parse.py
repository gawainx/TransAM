import json
import sys

if __name__ == '__main__':

    DATASET = sys.argv[1]
    assert DATASET in ['wiki', 'nell']
    with (
        open(f'datasets/{DATASET}/path_graph') as pgfp,
        open(f'datasets/{DATASET}/openke/train2id.txt', 'w') as trfp,
        open(f'datasets/{DATASET}/openke/entity2id.txt', 'w') as efp,
        open(f'datasets/{DATASET}/openke/relation2id.txt', 'w') as rfp,

    ):
        e2id = json.load(open(f'datasets/{DATASET}/ent2ids'))
        r2id = json.load(open(f'datasets/{DATASET}/relation2ids'))
        efp.write(f"{len(e2id)}\n")
        rfp.write(f"{len(r2id)}\n")
        lines = pgfp.readlines()
        trfp.write(f'{len(lines) * 2}\n')
        for e, idx in e2id.items():
            efp.write(f"{e} {idx}\n")
        for r, idx in r2id.items():
            rfp.write(f"{r} {idx}\n")
        for line in lines:
            s, r, o = line.rstrip().rsplit('\t')
            trfp.write(f"{e2id[s]}\t{e2id[o]}\t{r2id[r]}\n")
            trfp.write(f"{e2id[o]}\t{e2id[s]}\t{r2id[f'{r}_inv']}\n")
