# ArtiBench

The benchmark from the ArtiAtlas paper.

## The protocol

| Dimension | `results.json` key | What it evaluates |
|-----------|---|-------------------|
| **Object Geometry** | `whole_mesh` | The assembled whole-object mesh |
| **Part Decomposition** | `part_meshes` | Per-link geometry: link decomposition, part count |
| **Articulation** | `articulation` | Joint parameters, collision-free motion, articulated-state geometry |

| Suite | Objects | 3D ground truth | What runs |
|-------|--------:|:---------------:|-----------|
| `category_rich` | 200 | yes | everything: self-quality, C1 joint parameters, C2 part geometry, image alignment, VLM judge |
| `in_the_wild` | 100 | no | self-quality, image alignment, VLM judge — C1/C2 are skipped automatically |

## Install

```bash
pip install -e ".[alignment,vlm]"

# Each compiles against the torch already in the environment, so install them
# after the line above and with build isolation off.
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"   # geometry metrics
pip install --no-build-isolation git+https://github.com/ashawkey/cubvh                            # eval_self only
pip install --no-build-isolation "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"  # optional
```

`cubvh` is CUDA-only.
Without `pointnet2_ops`, ULIP-2 and Uni3D fall back to a slower CPU sampler.

| Requirement | |
|---|---|
| Blender | **4.2.2 LTS** on `PATH`, or at `$ARTIBENCH_BLENDER_BIN` / `$BLENDER_BIN`. Published numbers used this build; another shifts the image metrics |
| API key | `GOOGLE_API_KEY` or `GEMINI_API_KEY`, for the VLM judge only (`--no-vlm` skips it) |

⚠️ **If the first render appears to hang, set `ARTIBENCH_BLENDER_DEVICE=CPU`.** Cycles GPU
initialization can stall indefinitely on a contended node: Blender sits at `Loading render kernels`
without ever failing.

`bash scripts/smoke_test.sh` scores the two bundled objects against a stored fixture.

## Submission format

One directory per object, named by the suite's object id, each holding a **`mobility.urdf`** and the
meshes it points at (`.obj`, `.ply` or `.stl`, paths relative to the URDF).

- **Orientation.** Forward-kinematics at zero joint state must leave the object **Z-up, facing −X**,
  at the scale of your input mesh. The suite hands you meshes Y-up, +Z front, so this is a change
  of frame.
- **Structure.** A tree — no link with two or more parents — with every articulated joint
  `revolute` / `prismatic` / `continuous`. `revolute` and `prismatic` carry a `<limit>`; a
  `continuous` joint is read as [0, 2π] whatever limit you give it, and a `revolute` joint spanning
  a full turn is reclassified `continuous`.

The canonical form also has a geometry-free `base` link with a single `fixed` joint beneath it
carrying the global orientation, but you do not have to produce it. `evaluate.sh` runs
`check_urdf_conventions.py --enable-urdf-fixing` before scoring: it inserts `base` when missing, and
collapses any other `fixed` joint into its parent, composing the transform so absolute poses are
preserved. The inserted joint is identity, so the orientation above still has to be baked into your
submission. A non-tree graph, a self-loop or a cycle is reported and left alone; it surfaces as
`kinematic_tree_valid_rate`.

Link and joint names are yours to choose. Your parts and joints are matched to the ground truth
geometrically — Hungarian assignment on rest-pose bounding-box centers — so nothing is parsed out
of a name.

```bash
python scripts/check_urdf_conventions.py --root my_predictions                    # dry run
python scripts/check_urdf_conventions.py --root my_predictions --enable-urdf-fixing
```

That audits structure only. **It does not check orientation, and neither does anything else.** A
rotated submission passes every local check, then scores near-worst on the geometry metrics —
`rest_overall_cd` jumps by orders of magnitude, `axis_error_rad_mean` saturates near π/2 — while
`type_accuracy`, `count_accuracy` and CLIP barely move.

[`tutorial.ipynb`](tutorial.ipynb) works through all of this in code, down to what a 90° error
costs.

## Get a suite

Download and unzip from the
[Google Drive folder](https://drive.google.com/drive/folders/1Nzw9Bdl00Zybwxy3LPRm0IP12Z78VIFo).

| Archive | Size | What it is |
|---|--:|---|
| `category_rich.zip` | 0.93 GiB | The whole `category_rich` suite: inputs and 3D ground truth. |
| `in_the_wild.zip` | 0.50 GiB | The `in_the_wild` suite: reference images, manifest, and everything the evaluator reads. |
| `in_the_wild-meshes.zip` | 3.00 GiB | The `whole_mesh/` inputs for the same objects, for methods that consume a mesh. |

```bash
export ARTIBENCH_DATA_ROOT=/scratch/artibench
mkdir -p "$ARTIBENCH_DATA_ROOT"
unzip category_rich.zip -d "$ARTIBENCH_DATA_ROOT"
```

Verify a download with `python -m artibench.data verify <suite>`.

`in_the_wild` is split along the same line as `--input-modality`: `in_the_wild.zip` is what every
method needs, and `in_the_wild-meshes.zip` adds the mesh inputs that only a `mesh` method reads.
**Both unzip into the same `in_the_wild/` tree.** `files.sha256` covers the whole suite, so a
`verify` run after only the first archive reports every `whole_mesh/` entry as missing.

```
$ARTIBENCH_DATA_ROOT/category_rich/
├── manifest.json                    # entry point; paths inside are relative
├── images/<oid>/                    # reference views + per-view part-label EXRs
├── whole_mesh/<oid>/mesh.obj        # input for whole-mesh methods
├── part_meshes/<oid>/               # input for part-mesh methods
├── cached_link_point_clouds/<oid>/  # optional, skips GT re-sampling
├── gt/<oid>/                        # mobility.urdf + textured_objs/ + annotations
├── files.sha256
└── VERSION
```

`in_the_wild` has no `gt/` and no `cached_link_point_clouds/`. Each suite's `manifest.json` is also
published beside the archives, and `assets/ids/category_rich_200.txt` and
`assets/ids/in_the_wild_100.txt` list the object ids without downloading anything.

## Run the full benchmark

```bash
export ARTIBENCH_DATA_ROOT=/scratch/artibench     # where you unzipped
export GOOGLE_API_KEY=...                         # or GEMINI_API_KEY

bash scripts/evaluate.sh \
    --suite category_rich \
    --pred-dir /path/to/my_predictions \
    --input-modality image \
    --out-dir out/mymodel
```

Any object you do not supply is counted as missing rather than skipped, so the denominator stays at
200. Report the coverage `evaluate.sh` prints alongside your numbers; the `completeness` block in
`results.json` records which ids were absent.

⚠️ **The repair pass edits your URDFs in place**, keeping a `<name>_backup.urdf` beside each one.
Run against a copy, or pass `--no-check-urdf`, if you need your originals untouched.

`--input-modality` is a property of **your method**, not of the data: `mesh` if it consumes the
ground-truth mesh, `image` if it reconstructs at arbitrary scale and placement. It selects how
predictions are aligned before the geometry metrics, so numbers computed under the two settings are
not comparable: state which one you used alongside your results.

## Get the numbers you report

```bash
python scripts/score_run.py --suite category_rich --run out/mymodel          # a table
python scripts/score_run.py --suite category_rich --run out/mymodel --json   # or JSON
```

Prints your method's value for each published column, under the column name the paper uses.

## Outputs

`scripts/evaluate.sh` writes two report directories under `--out-dir`:

```
out/mymodel/
├── self_quality/    # watertightness, part overlap, kinematic-tree validity
└── paired/          # C1, C2, alignment, VLM judge
```

When a per-sample metric cannot be computed it is **absent** from that row rather than filled with `0`,
`-1` or `inf`. `metric_counts` in `results.json` gives you the denominator for every metric.
