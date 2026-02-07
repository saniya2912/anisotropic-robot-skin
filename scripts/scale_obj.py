import re
from pathlib import Path

IN_PATH = Path("models/assets/skin.obj")
OUT_PATH = Path("models/assets/skin_scaled.obj")

# Change this:
# - try 0.01 if CAD units were cm
# - try 0.001 if CAD units were mm
SCALE = 0.01

# Also shift Z so the minimum Z becomes 0 (prevents starting inside the floor)
SHIFT_TO_GROUND = True

v_re = re.compile(r"^v\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*$")

def main():
    lines = IN_PATH.read_text().splitlines()

    verts = []
    for ln in lines:
        m = v_re.match(ln)
        if m:
            x, y, z = map(float, m.groups())
            verts.append((x, y, z))

    if not verts:
        raise RuntimeError("No vertices found (no 'v x y z' lines).")

    # scale
    scaled = [(x*SCALE, y*SCALE, z*SCALE) for x, y, z in verts]

    # shift so min z = 0
    if SHIFT_TO_GROUND:
        min_z = min(z for _, _, z in scaled)
        scaled = [(x, y, z - min_z) for x, y, z in scaled]

    # rewrite file
    out_lines = []
    vi = 0
    for ln in lines:
        m = v_re.match(ln)
        if m:
            x, y, z = scaled[vi]
            out_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
            vi += 1
        else:
            out_lines.append(ln)

    OUT_PATH.write_text("\n".join(out_lines) + "\n")

    # print bounds for sanity
    xs = [p[0] for p in scaled]
    ys = [p[1] for p in scaled]
    zs = [p[2] for p in scaled]
    print("Wrote:", OUT_PATH)
    print(f"Bounds X: {min(xs):.4f} .. {max(xs):.4f}  (size {max(xs)-min(xs):.4f})")
    print(f"Bounds Y: {min(ys):.4f} .. {max(ys):.4f}  (size {max(ys)-min(ys):.4f})")
    print(f"Bounds Z: {min(zs):.4f} .. {max(zs):.4f}  (size {max(zs)-min(zs):.4f})")

if __name__ == "__main__":
    main()
