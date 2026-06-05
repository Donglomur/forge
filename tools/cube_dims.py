"""cube_dims.py  --  find cube1's REAL size in the DexCanvas parquet.

The render currently GUESSES the cube size from fingertip spread. If the dataset
stores the true object dimensions, we should use them. This dumps every size-like
field and the full object_info block so we can wire the real number in.

  python tools/cube_dims.py ~/dexcanvas/mocap_ver0.1.parquet [row]
"""
import sys
import numpy as np
import pandas as pd

SIZE_WORDS = ("size", "scale", "dim", "extent", "bbox", "bound", "half", "width",
              "height", "depth", "length", "radius", "mesh", "asset", "shape", "object_name", "name")


def short(v):
    a = None
    try: a = np.asarray(v)
    except Exception: pass
    if isinstance(v, (str, bytes)): return repr(v)
    if isinstance(v, (int, float)): return str(v)
    if a is not None and a.dtype != object:
        if a.size <= 12: return f"{a.dtype}{a.shape} = {np.round(a, 4).tolist()}"
        return (f"{a.dtype}{a.shape}  min={a.min():.4f} max={a.max():.4f} "
                f"first={np.round(a.ravel()[:6], 4).tolist()}")
    if hasattr(v, "__len__"):
        try: return f"{type(v).__name__} len={len(v)} first={v[0]!r}"
        except Exception: pass
    return f"{type(v).__name__}"


def walk(d, pre=""):
    hits = []
    if isinstance(d, dict):
        for k, v in d.items():
            key = pre + str(k)
            if any(w in str(k).lower() for w in SIZE_WORDS) and not isinstance(v, dict):
                hits.append((key, short(v)))
            hits += walk(v, key + ".")
    return hits


def dump_objectinfo(d, out, pre=""):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                if str(k).lower() == "object_info":
                    out.append("  [object_info] full contents:")
                    for kk, vv in v.items():
                        out.append(f"      {kk}: {short(vv)}")
                dump_objectinfo(v, out, pre + str(k) + ".")
            else:
                dump_objectinfo(v, out, pre + str(k) + ".")


def main():
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    row = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    df = pd.read_parquet(sys.argv[1])
    print(f"parquet rows={len(df)}  top-level columns={list(df.columns)}")
    rec = df.iloc[row].to_dict()

    print("=" * 60)
    print("SIZE-LIKE FIELDS anywhere in the row:")
    print("=" * 60)
    hits = walk(rec)
    if not hits:
        print("  (none found by keyword)")
    for k, v in hits:
        print(f"  {k}: {v}")

    print("=" * 60)
    out = []
    dump_objectinfo(rec, out)
    print("\n".join(out) if out else "  (no object_info block found)")

    # positional bound of the object over time (NOT size, just a sanity check)
    print("=" * 60)
    print("object position travel (sanity bound, not the size):")
    def find_pos(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if str(k).lower() in ("position", "pose") and not isinstance(v, dict):
                    try:
                        a = np.asarray(v, float)
                        if a.size >= 3: return a
                    except Exception: pass
                r = find_pos(v)
                if r is not None: return r
        return None
    p = find_pos(rec)
    if p is not None:
        a = np.asarray(p, float)
        a = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1)
        xyz = a[:, :3]
        print(f"  frames={a.shape[0]}  x[{xyz[:,0].min():.3f},{xyz[:,0].max():.3f}]"
              f"  y[{xyz[:,1].min():.3f},{xyz[:,1].max():.3f}]"
              f"  z[{xyz[:,2].min():.3f},{xyz[:,2].max():.3f}]")
    else:
        print("  (no object position array found)")
    print("=" * 60)
    print("  >> paste the SIZE-LIKE FIELDS and [object_info] sections above")


if __name__ == "__main__":
    main()
