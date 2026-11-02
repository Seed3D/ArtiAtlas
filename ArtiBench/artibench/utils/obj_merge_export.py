# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import trimesh


_TX_RE_NEWMTL = re.compile(r"^\s*newmtl\s+(\S+)")
_TX_RE_MAP = re.compile(r"^\s*map_\w+\s+(.+)$")


def load_obj_raw_preserve(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    with path.open() as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                toks = line[2:].strip().split()
                if len(toks) == 3:
                    faces.append([int(t.split("/")[0]) - 1 for t in toks])
                else:
                    faces.append([int(t.split("/")[0]) - 1 for t in toks[:3]])
                    for idx in range(2, len(toks) - 1):
                        faces.append(
                            [
                                int(toks[0].split("/")[0]) - 1,
                                int(toks[idx].split("/")[0]) - 1,
                                int(toks[idx + 1].split("/")[0]) - 1,
                            ]
                        )
    return (
        np.asarray(verts, dtype=float).reshape(-1, 3),
        np.asarray(faces, dtype=int).reshape(-1, 3),
    )


def load_model_from_path(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".obj":
        return load_obj_raw_preserve(path)

    mesh = trimesh.load(path.as_posix(), process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.geometry.values())
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh asset: {path}")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _sha1_of_file(path: Path, chunk: int = 1 << 16) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def _parse_mtl_blocks(mtl_paths: List[Path]) -> Dict[str, List[str]]:
    blocks: Dict[str, List[str]] = {}
    for mtl in mtl_paths:
        if not mtl.exists():
            continue
        with mtl.open() as fh:
            cur_name: str | None = None
            cur_lines: List[str] = []
            for line in fh:
                match = _TX_RE_NEWMTL.match(line)
                if match:
                    if cur_name is not None:
                        blocks[cur_name] = cur_lines
                    cur_name, cur_lines = match.group(1), [line]
                else:
                    cur_lines.append(line)
            if cur_name is not None:
                blocks[cur_name] = cur_lines
    return blocks


def _material_has_texture(block: List[str]) -> bool:
    return any(_TX_RE_MAP.match(line) is not None for line in block)


def _prune_duplicate_backfaces_prefer_textured(
    face_tokens: List[List[str]],
    face_mats: List[str | None],
    faces_np: np.ndarray,
    mat_blocks: Dict[str, List[str]],
) -> tuple[List[List[str]], List[str | None], np.ndarray, int]:
    """Drop exact duplicate triangles when only some duplicates carry textures.

    PartNet-Mobility occasionally stores a textured screen face together with a
    coplanar untextured duplicate using the same position indices but opposite
    winding. Viewers then often show the plain duplicate on top. When a
    duplicate group mixes textured and non-textured materials, keep the
    textured faces and drop the non-textured duplicates.
    """
    if len(face_tokens) <= 1:
        return face_tokens, face_mats, faces_np, 0

    textured_materials = {
        name for name, block in mat_blocks.items() if _material_has_texture(block)
    }
    if not textured_materials:
        return face_tokens, face_mats, faces_np, 0

    by_pos_tri: Dict[tuple[int, int, int], List[int]] = {}
    for face_idx, tri in enumerate(np.asarray(faces_np, dtype=np.int64)):
        key = tuple(sorted(int(v) for v in tri.tolist()))
        by_pos_tri.setdefault(key, []).append(face_idx)

    drop_indices: set[int] = set()
    for face_indices in by_pos_tri.values():
        if len(face_indices) < 2:
            continue

        textured = [
            idx for idx in face_indices if face_mats[idx] in textured_materials
        ]
        non_textured = [
            idx for idx in face_indices if face_mats[idx] not in textured_materials
        ]
        if textured and non_textured:
            drop_indices.update(non_textured)

    if not drop_indices:
        return face_tokens, face_mats, faces_np, 0

    keep_indices = [idx for idx in range(len(face_tokens)) if idx not in drop_indices]
    pruned_tokens = [face_tokens[idx] for idx in keep_indices]
    pruned_mats = [face_mats[idx] for idx in keep_indices]
    pruned_faces = np.asarray(faces_np[keep_indices], dtype=faces_np.dtype)
    return pruned_tokens, pruned_mats, pruned_faces, len(drop_indices)


def _copy_and_rewrite_textures(
    block: List[str],
    obj_dir: Path,
    tex_dir: Path,
    output_path: Path,
    tex_seen: Dict[str, str],
) -> List[str]:
    out: List[str] = []
    for line in block:
        match = _TX_RE_MAP.match(line)
        if match is None:
            out.append(line)
            continue

        tex_path = (obj_dir / match.group(1)).resolve()
        if not tex_path.exists():
            out.append(line)
            continue

        digest = _sha1_of_file(tex_path)
        new_name = f"{digest}{tex_path.suffix.lower()}"
        if digest not in tex_seen:
            shutil.copyfile(tex_path, tex_dir / new_name)
            tex_seen[digest] = new_name

        prefix = line.split(match.group(1), 1)[0]
        rel_path = os.path.relpath(tex_dir, output_path.parent)
        out.append(f"{prefix}{rel_path}/{new_name}\n")
    return out


def combine_meshes_and_save(
    verts: Sequence[np.ndarray],
    faces: Sequence[np.ndarray],
    model_paths: Sequence[Union[str, Path]],
    output_base: Union[str, Path],
    texture_dir: Optional[Union[str, Path]] = None,
    overwrite_colors: Optional[Sequence[Optional[Tuple[float, float, float, float]]]] = None,
    verbose: bool = True,
) -> None:
    n_models = len(model_paths)
    if not (len(verts) == len(faces) == n_models):
        raise ValueError("verts, faces and model_paths must be the same length")
    if overwrite_colors is not None and len(overwrite_colors) != n_models:
        raise ValueError("overwrite_colors must match model count")

    output_base = Path(output_base)
    out_obj = output_base.with_suffix(".obj")
    out_mtl = output_base.with_suffix(".mtl")
    tex_dir = Path(texture_dir) if texture_dir is not None else out_obj.parent / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)

    obj_buf: List[str] = [f"mtllib {out_mtl.name}\n"]
    mtl_buf: List[str] = []
    tex_seen: Dict[str, str] = {}

    v_ofs = 0
    vt_ofs = 0
    vn_ofs = 0

    for mesh_idx, (verts_np, faces_np, src_raw) in enumerate(zip(verts, faces, model_paths)):
        src_path = Path(src_raw)
        ext = src_path.suffix.lower()
        color_override = None if overwrite_colors is None else overwrite_colors[mesh_idx]

        if verts_np.ndim != 2 or verts_np.shape[1] != 3:
            raise ValueError("verts entries must be shaped (N, 3)")

        if ext == ".obj":
            lines = src_path.read_text().splitlines()

            vt_lines: List[str] = []
            vn_lines: List[str] = []
            face_tokens: List[List[str]] = []
            face_mats: List[str | None] = []
            mtl_files: List[str] = []
            cur_mat: str | None = None
            explicit_usemtl = False

            for line in lines:
                if line.startswith("mtllib"):
                    mtl_files.extend(line.split()[1:])
                elif line.startswith("vt "):
                    vt_lines.append(line)
                elif line.startswith("vn "):
                    vn_lines.append(line)
                elif line.startswith("usemtl"):
                    cur_mat = line.split()[1]
                    explicit_usemtl = True
                elif line.startswith("f "):
                    toks = line[2:].strip().split()
                    if len(toks) == 3:
                        face_tokens.append(toks)
                        face_mats.append(cur_mat)
                    else:
                        face_tokens.append(toks[:3])
                        face_mats.append(cur_mat)
                        for tri_idx in range(2, len(toks) - 1):
                            face_tokens.append([toks[0], toks[tri_idx], toks[tri_idx + 1]])
                            face_mats.append(cur_mat)

            # NOTE: this assertion relies on load_obj_raw_preserve and the fan- triangulation above using the same decomposition logic.
            # If either is changed the other must be updated in lockstep.
            if len(face_tokens) != faces_np.shape[0]:
                raise ValueError(
                    f"{src_path.name}: faces array has {faces_np.shape[0]} rows but OBJ contains {len(face_tokens)} triangles"
                )

            if color_override is not None:
                r, g, b, a = color_override
                mat_name = f"{mesh_idx}_override"
                mtl_buf.append(
                    f"newmtl {mat_name}\n"
                    f"Kd {r:.6f} {g:.6f} {b:.6f}\n"
                    f"Ka {r:.6f} {g:.6f} {b:.6f}\n"
                    f"d {a:.6f}\n"
                    f"Ns 10.0\n\n"
                )
                face_mats = [mat_name] * len(face_tokens)
                name_map: Dict[str, str] = {}
            else:
                mtl_paths = [src_path.parent / name for name in mtl_files]
                mat_blocks = _parse_mtl_blocks(mtl_paths)
                name_map = {}

                if explicit_usemtl:
                    face_tokens, face_mats, faces_np, num_pruned = _prune_duplicate_backfaces_prefer_textured(
                        face_tokens,
                        face_mats,
                        faces_np,
                        mat_blocks,
                    )
                    if verbose and num_pruned > 0:
                        print(
                            f"[obj-merge] pruned {num_pruned} non-textured duplicate faces from {src_path.name}"
                        )

                if not explicit_usemtl:
                    if len(mat_blocks) == 1:
                        sole = next(iter(mat_blocks))
                        mat_name = f"{mesh_idx}_{sole}"
                        rewritten = _copy_and_rewrite_textures(
                            mat_blocks[sole], src_path.parent, tex_dir, out_mtl, tex_seen
                        )
                        rewritten[0] = f"newmtl {mat_name}\n"
                        mtl_buf.extend(rewritten if rewritten[-1].endswith("\n") else rewritten + ["\n"])
                    else:
                        mat_name = f"{mesh_idx}_implicit"
                        mtl_buf.append(f"newmtl {mat_name}\nKd 0.8 0.8 0.8\nNs 10\n\n")
                    face_mats = [mat_name] * len(face_tokens)
                else:
                    used = {mat for mat in face_mats if mat}
                    for old_name in used:
                        if old_name not in mat_blocks:
                            continue
                        new_name = f"{mesh_idx}_{old_name}"
                        name_map[old_name] = new_name
                        rewritten = _copy_and_rewrite_textures(
                            mat_blocks[old_name], src_path.parent, tex_dir, out_mtl, tex_seen
                        )
                        rewritten[0] = f"newmtl {new_name}\n"
                        mtl_buf.extend(rewritten if rewritten[-1].endswith("\n") else rewritten + ["\n"])

            obj_buf.extend(
                f"v {vx:.6f} {vy:.6f} {vz:.6f}\n" for vx, vy, vz in verts_np.astype(float)
            )
            obj_buf.extend(line + "\n" for line in vt_lines)
            obj_buf.extend(line + "\n" for line in vn_lines)

            last_mat = None
            for face_idx, toks in enumerate(face_tokens):
                mat_here = name_map.get(face_mats[face_idx], face_mats[face_idx])
                if mat_here != last_mat:
                    obj_buf.append(f"usemtl {mat_here}\n")
                    last_mat = mat_here

                tri = []
                for corner_idx, tok in enumerate(toks):
                    parts = tok.split("/")
                    v_new = int(faces_np[face_idx, corner_idx]) + 1 + v_ofs
                    vt_new = None
                    vn_new = None
                    if len(parts) >= 2 and parts[1]:
                        vt_new = int(parts[1]) + vt_ofs
                    if len(parts) == 3 and parts[2]:
                        vn_new = int(parts[2]) + vn_ofs

                    if vt_new is None and vn_new is None:
                        tri.append(f"{v_new}")
                    elif vt_new is not None and vn_new is None:
                        tri.append(f"{v_new}/{vt_new}")
                    elif vt_new is None and vn_new is not None:
                        tri.append(f"{v_new}//{vn_new}")
                    else:
                        tri.append(f"{v_new}/{vt_new}/{vn_new}")
                obj_buf.append("f " + " ".join(tri) + "\n")

            vt_len = len(vt_lines)
            vn_len = len(vn_lines)
        else:
            mat_name = f"{mesh_idx}_default"
            if color_override is not None:
                r, g, b, a = color_override
            else:
                r = g = b = 0.8
                a = 1.0
            mtl_buf.append(
                f"newmtl {mat_name}\n"
                f"Kd {r:.6f} {g:.6f} {b:.6f}\n"
                f"Ka {r:.6f} {g:.6f} {b:.6f}\n"
                f"d {a:.6f}\n"
                f"Ns 10.0\n\n"
            )
            obj_buf.extend(
                f"v {vx:.6f} {vy:.6f} {vz:.6f}\n" for vx, vy, vz in verts_np.astype(float)
            )
            obj_buf.append(f"usemtl {mat_name}\n")
            for tri in faces_np:
                obj_buf.append("f " + " ".join(str(int(idx) + 1 + v_ofs) for idx in tri) + "\n")
            vt_len = 0
            vn_len = 0

        v_ofs += int(verts_np.shape[0])
        vt_ofs += vt_len
        vn_ofs += vn_len

    out_obj.write_text("".join(obj_buf), encoding="utf-8")
    out_mtl.write_text("".join(mtl_buf), encoding="utf-8")
    if verbose:
        print(f"[obj-merge] wrote {out_obj} and {out_mtl.name} ({len(tex_seen)} unique textures)")


def export_combined_obj_from_instances(
    instances: Iterable[dict],
    output_obj_path: Path,
    *,
    texture_dir: Path | None = None,
    verbose: bool = False,
) -> Path:
    if output_obj_path.suffix.lower() != ".obj":
        raise ValueError(f"Expected .obj output path, got: {output_obj_path}")

    verts_list: List[np.ndarray] = []
    faces_list: List[np.ndarray] = []
    model_paths: List[Path] = []

    for instance in instances:
        mesh_path = Path(instance["mesh_path"])
        transform = np.asarray(instance["transform"], dtype=np.float64)
        verts, faces = load_model_from_path(mesh_path)
        verts_world = trimesh.transform_points(verts, transform)
        verts_list.append(np.asarray(verts_world, dtype=np.float64))
        faces_list.append(np.asarray(faces, dtype=np.int64))
        model_paths.append(mesh_path)

    if not verts_list:
        raise RuntimeError(f"No mesh instances found for {output_obj_path}")

    resolved_texture_dir = texture_dir
    if resolved_texture_dir is None:
        resolved_texture_dir = output_obj_path.parent / "textures"
    combine_meshes_and_save(
        verts=verts_list,
        faces=faces_list,
        model_paths=model_paths,
        output_base=output_obj_path.with_suffix(""),
        texture_dir=resolved_texture_dir,
        verbose=verbose,
    )
    return output_obj_path


def export_part_meshes_obj_dir(
    part_entries: Sequence[dict],
    output_dir: Path,
    *,
    manifest_name: str = "manifest.json",
    verbose: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    texture_dir = output_dir / "textures"
    parts_manifest: list[dict] = []

    for entry in part_entries:
        part_id = int(entry["part_id"])
        instances = list(entry.get("instances", []))
        if not instances:
            continue

        obj_name = f"part_{part_id:03d}"
        obj_path = export_combined_obj_from_instances(
            instances=instances,
            output_obj_path=output_dir / f"{obj_name}.obj",
            texture_dir=texture_dir,
            verbose=verbose,
        )
        parts_manifest.append(
            {
                "part_id": part_id,
                "obj_path": obj_path.name,
                "mtl_path": obj_path.with_suffix(".mtl").name,
                "num_instances": len(instances),
            }
        )

    manifest = {
        "format": "artibench_part_meshes_obj_v1",
        "texture_dir": "textures",
        "num_parts": len(parts_manifest),
        "parts": parts_manifest,
    }
    manifest_path = output_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path
