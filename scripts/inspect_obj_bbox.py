from pathlib import Path
import argparse
import math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    args = ap.parse_args()

    path = Path(args.obj)
    groups = {}
    current = "default"
    groups[current] = []

    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("g "):
                current = line[2:].strip() or "unnamed"
                groups.setdefault(current, [])
            elif line.startswith("v "):
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                groups[current].append((x, y, z))

    def bbox(pts):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        mn = (min(xs), min(ys), min(zs))
        mx = (max(xs), max(ys), max(zs))
        size = (mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2])
        center = ((mx[0]+mn[0])/2, (mx[1]+mn[1])/2, (mx[2]+mn[2])/2)
        return mn, mx, size, center

    all_pts = [p for pts in groups.values() for p in pts]
    mn, mx, size, center = bbox(all_pts)
    print("ALL bbox min:", mn, "max:", mx)
    print("ALL size:", size, "center:", center)

    for g, pts in groups.items():
        if not pts:
            continue
        mn, mx, size, center = bbox(pts)
        print(f"\nGroup '{g}':")
        print("  min:", mn, "max:", mx)
        print("  size:", size, "center:", center)

if __name__ == "__main__":
    main()
