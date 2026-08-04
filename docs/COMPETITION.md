# Competition brief

Distilled from the official competition page. Live figures come from `tools/kaggle_status.py`;
the archived page scrape lives at `docs/source/Biohub.md` and is stale as of 2026-07-03.

## Task

Detect cells in each timepoint of a 3D + time zebrafish embryo microscopy volume, link them across
time, and identify divisions — producing a tracking graph of nodes (detections) and edges (temporal
links).

## Timeline

| Date | Event |
| --- | --- |
| 2026-06-29 | Start |
| 2026-09-22 | Entry deadline / team merger deadline |
| 2026-09-29 | Final submission deadline (23:59 UTC) |

## Prizes

$60,000 total — 1st $18,000, 2nd $12,000, 3rd $8,000, 4th $6,000, 5th $6,000, 6th $5,000, 7th $5,000.

## Data

Each sample is a short 3D+time video stored as a **Zarr v3** volume.

- Array at path `0/`, shape `(T, Z, Y, X)`, typically `(100, 64, 256, 256)`, `uint16`.
- One chunk per timepoint: `(1, 64, 256, 256)`, blosc/zstd. Chunk for timepoint `t` is at
  `0/c/{t}/0/0/0`. Metadata in `0/zarr.json`.
- **Physical voxel scale: z = 1.625, y = x = 0.40625 µm/voxel.** Anisotropic — z is 4× coarser.

Ground truth (training only) is a `.geff` directory, also Zarr v3:

- `nodes/ids` — node ID array
- `nodes/props/{t,z,y,x}/values` — integer centroid coordinates in voxels
- `edges/ids` — `(N, 2)` array of `(source_id, target_id)`

**Annotations are sparse** — not every cell in every frame is labelled. The
`estimated_number_of_nodes` field in the `.geff` metadata estimates the true total cell count per
sample, and it is what the node-count penalty is measured against.

### Sample inventory (counted from the mounted dataset)

| Split | Samples | Embryos |
| --- | --- | --- |
| train | 199 | `44b6` (71), `6bba` (128) |
| test (visible) | 4 | copies from train — no ground truth |

Folder names are `{embryo_id}_{field_of_view}`. **Train and test are embryo-disjoint**; the hidden
test set is swapped in at rerun and is roughly the size of the training set (~199 samples). Public LB
is 29% of it, private LB the remaining 71%.

Total dataset size: 87.61 GB, 24,886 files, CC0 licensed.

## Evaluation

```
score = adj_edge_jaccard + 0.1 * division_jaccard
```

Node matching is optimal bipartite assignment on scaled centroid distance, max 7.0 µm, using the
physical voxel scale above. Full analysis of the scorer's exploitable structure:
[METRIC_ANALYSIS.md](METRIC_ANALYSIS.md). Reference implementation:
[royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition).

Scores can legitimately exceed 1.0 because the node-count adjustment is unbounded above.

## Submission format

CSV named `submission.csv`, with node rows and edge rows grouped by dataset:

```
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6,node,1,0,32,128,128,-1,-1
1,44b6,node,2,1,33,130,125,-1,-1
2,6bba,edge,-1,-1,-1,-1,-1,1,2
```

- **Node rows**: `row_type=node`, with `node_id,t,z,y,x` as integer voxel centroids; `source_id` and
  `target_id` set to `-1`.
- **Edge rows**: `row_type=edge`, with `source_id`/`target_id` referencing node IDs; `node_id,t,z,y,x`
  set to `-1`.
- `id` is a required throwaway consecutive index.
- `dataset` must match the test folder name without `.zarr`.
- **Every dataset in the test set must appear.**

`src/biocell/submission.py` writes this format and `validate_submission()` checks the failure modes
that silently waste a submission slot.

## Code requirements

- Notebooks only. CPU or GPU notebook ≤ 12 hours runtime.
- **Internet access disabled** at scoring time — dependencies and weights must be attached as Kaggle
  datasets.
- Freely and publicly available external data and pre-trained models are allowed.
- Output must be named `submission.csv`.
- 5 submissions per day.

## Citation

Thibaut Goldsborough, Jordão Bragantini, Xiang Zhao, Gordon Leary, Teun Huijben, Ilan da Silva
Theodoro, Kyle Harrington, Chi-Li Chiu, Walter Reade, María Cruz, and Loïc A. Royer. *Biohub - Cell
Tracking During Development.* Kaggle, 2026.
