"""GIF -> Lottie by tracing every frame into vector shapes (flipbook style).

Each GIF frame becomes one shape layer that is visible for exactly one frame.
Vectors stay crisp at any size, but motion is not keyframed.

Built for flat-colour art (cartoons, emoji, pixel art), not photos. Per frame:
  1. upscale 3x, snap pixels to the GIF's real colours (auto-detected; edge blends
     and white matte fringe resolve to a neighbouring colour)
  2. trace one mask per colour, stacked like screen-print layers
  3. trace line art separately as a top "ink" layer (thin dark strokes survive)
  4. small round blobs (halftone dots) become true circles
  5. simplify stair-stepped outlines and refit them as smooth curves

usage: python gif_to_lottie.py in.gif out.lottie [--step 2] [--help for knobs]
"""
import argparse, json, os, re, zipfile
from multiprocessing import Pool
from collections import Counter
import numpy as np
from scipy import ndimage
from PIL import Image
import vtracer

PATH_RE = re.compile(r'<path d="([^"]+)" fill="#([0-9A-Fa-f]{6})"(?: transform="translate\(([-\d.]+),([-\d.]+)\)")?')
TOK_RE = re.compile(r'[MLCZ]|-?\d*\.?\d+(?:e-?\d+)?')


def r(x):
    return round(x, 1)


def parse_d(d, ox, oy, scale=1.0):
    """SVG path data (absolute M/L/C/Z) -> list of Lottie bezier shapes."""
    toks = TOK_RE.findall(d)
    shapes, cur, i = [], None, 0

    def num():
        nonlocal i
        v = float(toks[i]); i += 1
        return v

    def pt():
        return [num() + ox, num() + oy]

    def close():
        if cur and len(cur["v"]) > 1:
            # last vertex duplicates the first: fold its in-tangent onto vertex 0
            if cur["v"][-1] == cur["v"][0]:
                cur["i"][0] = cur["i"].pop()
                cur["v"].pop(); cur["o"].pop()
            shapes.append(cur)

    cmd = None
    while i < len(toks):
        if toks[i] in "MLCZ":
            cmd = toks[i]; i += 1
        if cmd == "M":
            close()
            p = pt()
            cur = {"v": [p], "i": [[0, 0]], "o": [[0, 0]]}
            cmd = "L"  # implicit lineto after moveto
        elif cmd == "L":
            p = pt()
            cur["v"].append(p); cur["i"].append([0, 0]); cur["o"].append([0, 0])
        elif cmd == "C":
            c1, c2, p = pt(), pt(), pt()
            last = cur["v"][-1]
            cur["o"][-1] = [c1[0] - last[0], c1[1] - last[1]]
            cur["v"].append(p); cur["i"].append([c2[0] - p[0], c2[1] - p[1]]); cur["o"].append([0, 0])
        elif cmd == "Z":
            close(); cur = None
    close()

    out = []
    for s in shapes:
        sh = smooth_shape(s, scale) if SMOOTH else {
            "c": True,
            "v": [[r(a * scale), r(b * scale)] for a, b in s["v"]],
            "i": [[r(a * scale), r(b * scale)] for a, b in s["i"]],
            "o": [[r(a * scale), r(b * scale)] for a, b in s["o"]]}
        if sh:
            out.append(sh)
    return out


SMOOTH = 0.0  # simplify tolerance in output px; set from --smooth
CORNER = np.cos(np.radians(55))  # turns sharper than 55 deg stay sharp corners


def flatten(s, scale, per_curve=4):
    """Bezier shape -> closed polyline in output px."""
    v, ii, oo = (np.array(s[k], dtype=np.float64) * scale for k in ("v", "i", "o"))
    pts = []
    for k in range(len(v)):
        p0, p3 = v[k], v[(k + 1) % len(v)]
        p1, p2 = p0 + oo[k], p3 + ii[(k + 1) % len(v)]
        if not (oo[k].any() or ii[(k + 1) % len(v)].any()):
            pts.append(p0)
            continue
        for t in np.arange(per_curve) / per_curve:
            pts.append((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)
    return np.array(pts)


def rdp(pts, eps):
    """Ramer-Douglas-Peucker on an open polyline (iterative). Returns kept indices."""
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = pts[b] - pts[a]
        rel = pts[a + 1:b] - pts[a]
        L = np.hypot(*seg)
        d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / L if L > 1e-9 else np.hypot(rel[:, 0], rel[:, 1])
        m = d.argmax()
        if d[m] > eps:
            keep[a + 1 + m] = True
            stack += [(a, a + 1 + m), (a + 1 + m, b)]
    return keep


def smooth_shape(s, scale):
    """Trace output follows pixel stair-steps. Flatten, simplify (RDP), then rebuild
    as smooth Catmull-Rom curves, leaving sharp turns as real corners."""
    pts = flatten(s, scale)
    if len(pts) < 3:
        return None
    far = np.hypot(*(pts - pts[0]).T).argmax()  # split the loop into two open runs
    ring = np.vstack([pts, pts[:1]])
    keep = np.zeros(len(pts), bool)
    keep[:far + 1] |= rdp(ring[:far + 1], SMOOTH)
    keep[far:] |= rdp(ring[far:], SMOOTH)[:-1]
    v = pts[keep]
    if len(v) < 3:
        return None
    prev, nxt = np.roll(v, 1, 0), np.roll(v, -1, 0)
    din, dout = v - prev, nxt - v
    lin, lout = np.hypot(*din.T), np.hypot(*dout.T)
    cos = (din * dout).sum(1) / np.maximum(lin * lout, 1e-9)
    chord = nxt - prev
    lc = np.maximum(np.hypot(*chord.T), 1e-9)[:, None]
    o = chord / lc * lout[:, None] / 3
    i = -chord / lc * lin[:, None] / 3
    sharp = cos < CORNER
    o[sharp] = 0
    i[sharp] = 0
    rr = lambda a: [[r(x), r(y)] for x, y in a]
    return {"c": True, "v": rr(v), "i": rr(i), "o": rr(o)}


def static(v):
    return {"a": 0, "k": v}


def svg_to_groups(svg, scale=1.0):
    groups = []
    for d, hexc, tx, ty in PATH_RE.findall(svg):
        rgb = [int(hexc[k:k + 2], 16) / 255 for k in (0, 2, 4)]
        items = [{"ty": "sh", "ks": static(s)} for s in parse_d(d, float(tx or 0), float(ty or 0), scale)]
        items.append({"ty": "fl", "c": static([round(c, 4) for c in rgb] + [1]), "o": static(100), "r": 1})
        items.append({"ty": "tr", "p": static([0, 0]), "a": static([0, 0]), "s": static([100, 100]),
                      "r": static(0), "o": static(100)})
        groups.append({"ty": "gr", "it": items})
    # SVG paints later paths on top; Lottie paints earlier shapes on top
    return groups[::-1]


def pick_palette(im, n, min_dist, min_share=3e-5):
    """Colours ranked by how often they form solid 2x2 blocks. Flat fills (even 2px
    halftone dots) do; anti-alias blends and matte fringe are 1px bands along edges
    and almost never do, so they drop out without colour-specific rules.
    Near-duplicates merge; colours under `min_share` of all solid blocks are noise."""
    counts = Counter()
    for k in range(0, im.n_frames, max(1, im.n_frames // 40)):
        im.seek(k)
        a = np.asarray(im.convert("RGBA"), dtype=np.int64)
        key = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
        key[a[..., 3] <= 127] = -1
        inner = key[:-1, :-1]
        same = (inner == key[1:, :-1]) & (inner == key[:-1, 1:]) & (inner == key[1:, 1:])
        vals, cnt = np.unique(inner[same & (inner >= 0)], return_counts=True)
        counts.update(dict(zip(vals.tolist(), cnt.tolist())))
    pal = []
    total = sum(counts.values())
    for v, cnt in counts.most_common():
        if cnt < min_share * total:
            break
        c = (v >> 16, (v >> 8) & 255, v & 255)
        if all(sum((a - b) ** 2 for a, b in zip(c, p)) >= min_dist ** 2 for p in pal):
            pal.append(c)
            if len(pal) == n:
                break
    return np.array(pal, dtype=np.int32)


def snap(frame, pal, exact=15, passes=3, ink_bias=0.5, fringe=1):
    """Snap pixels to the palette. Pixels already close to a palette color keep it;
    anti-alias blends (far from every palette color) take whichever colour of their
    confident 3x3 neighbours they're closest to, so edges don't grow 1px rings of a
    third color but thin dark outlines survive.
    `fringe` px inside the silhouette are matte blends (cut out of a white background);
    they become ink wherever ink touches them."""
    a = np.asarray(frame, dtype=np.int32)
    h, w = a.shape[:2]
    n = len(pal)
    d2 = ((a[..., None, :3] - pal[None, None]) ** 2).sum(-1)
    idx = d2.argmin(-1)
    d2 = d2.astype(np.float64)
    d2[..., pal.sum(-1).argmin()] *= ink_bias   # only affects blends; exact pixels already chosen
    opaque = a[..., 3] > 127
    idx[~opaque] = n  # transparent is its own class
    ink = pal.sum(-1).argmin()
    near_clear = ~opaque
    for _ in range(fringe):
        near_clear = near_clear | np.pad(near_clear, 1)[:-2, 1:-1] | np.pad(near_clear, 1)[2:, 1:-1]             | np.pad(near_clear, 1)[1:-1, :-2] | np.pad(near_clear, 1)[1:-1, 2:]
    matte = opaque & near_clear
    d2[matte, ink] = 0
    # half-transparent and matte-fringe pixels have unreliable colour, so they're blends too
    conf = ((d2.min(-1) <= exact ** 2) & (a[..., 3] >= 250) & ~matte) | ~opaque
    for _ in range(passes + fringe):
        onehot = np.zeros((h + 2, w + 2, n + 1), np.int32)
        onehot[1:-1, 1:-1][conf] = np.eye(n + 1, dtype=np.int32)[idx[conf]]
        votes = sum(onehot[y:y + h, x:x + w] for y in range(3) for x in range(3))
        fix = ~conf & (votes.sum(-1) > 0)
        cand = np.where(votes[fix][:, :n] > 0, d2[fix], np.inf)
        best = cand.argmin(-1)
        idx[fix] = np.where(np.isinf(cand.min(-1)), n, best)  # only transparent neighbours
        conf = conf | fix
    return idx  # palette index per pixel, len(pal) = transparent


def trace_mask(mask, color, speckle, detail, scale):
    """Trace one boolean mask into Lottie groups filled with `color`."""
    h, w = mask.shape
    if not mask.any():
        return []
    px = np.where(mask[..., None], 0, 255).astype(np.uint8).repeat(3, -1)
    px = np.concatenate([px, np.full((h, w, 1), 255, np.uint8)], -1).reshape(-1, 4)
    svg = vtracer.convert_pixels_to_svg(
        [tuple(p) for p in px.tolist()], size=(w, h), colormode="binary", mode="spline",
        filter_speckle=speckle, corner_threshold=60, length_threshold=detail,
        splice_threshold=45, path_precision=2)
    # one group per colour: traced regions never overlap, and even-odd fill keeps
    # islands inside holes correct, so this saves a fill+transform per blob
    items = [{"ty": "sh", "ks": static(sh)}
             for d, _, tx, ty in PATH_RE.findall(svg)
             for sh in parse_d(d, float(tx or 0), float(ty or 0), scale)]
    if not items:
        return []
    g = fill_group(items, color)
    g["it"][-2]["r"] = 2  # even-odd
    return [g]


def fill_group(items, color):
    rgb = [round(int(v) / 255, 4) for v in color]
    return {"ty": "gr", "it": items + [
        {"ty": "fl", "c": static(rgb + [1]), "o": static(100), "r": 1},
        {"ty": "tr", "p": static([0, 0]), "a": static([0, 0]), "s": static([100, 100]),
         "r": static(0), "o": static(100)}]}


def pop_dots(own, max_d, scale):
    """Small round blobs (halftone dots) -> Lottie ellipses. Far smaller than traced
    beziers and perfectly round. Removes them from `own` in place."""
    lab, n = ndimage.label(own)
    els = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(bw, bh) > max_d or max(bw, bh) > 2 * min(bw, bh):
            continue
        blob = lab[sl] == k
        area = blob.sum()
        if not 0.55 <= area / (bw * bh) <= 1.0:  # a disc fills ~0.785 of its box
            continue
        ys, xs = np.nonzero(blob)
        g = np.sqrt(area / (np.pi / 4 * bw * bh))  # resize box so the ellipse keeps the blob's area
        els.append({"ty": "el", "d": 1,
                    "p": static([r((xs.mean() + sl[1].start + 0.5) * scale), r((ys.mean() + sl[0].start + 0.5) * scale)]),
                    "s": static([r(bw * g * scale), r(bh * g * scale)])})
        own[sl][blob] = False
    return els


def trace_layers(idx, pal, speckle, detail, scale, dot_max=0):
    """Screen-print style: one traced mask per palette colour, most common colour at the
    bottom. A layer extends a little under the colours painted above it (no hairline
    gaps between layers) and fills holes those colours sit in, so thin lines and small
    details stay intact without being traced again in every layer below them."""
    n = len(pal)
    opaque = idx < n
    ring = max(1, round(1 / scale))  # one source pixel
    groups = []
    for k, c in enumerate(pal):
        own = idx == k
        if not own.any():
            continue
        above = opaque & (idx > k)
        els = pop_dots(own, dot_max, scale) if dot_max else []
        mask = own | (ndimage.binary_dilation(own, iterations=ring) & above)
        holes, nh = ndimage.label(~mask)
        if nh:
            # a hole made only of higher colours gets painted over anyway: fill it
            bad = np.bincount(holes[~above & ~mask], minlength=nh + 1)
            mask |= (holes > 0) & (bad[holes] == 0)
        layer = trace_mask(mask, c, speckle, detail, scale) if mask.any() else []
        if els:
            layer = [fill_group(els, c)] + layer
        groups = layer + groups  # later colours paint on top
    return groups


def ink_mask(frame, ratio, max_chroma=0.2, radius=6, max_width=3):
    """Line art as its own layer. Rendered outlines are thin anti-aliased strokes that
    rarely reach full ink, so palette snapping loses them. Instead: ink is any
    low-saturation pixel noticeably darker than the brightest pixel within `radius`.
    Ink is thin: a grey region with any part wider than `max_width` px is a fill
    (e.g. a grey beak), not a line, unless it's near-black."""
    f = np.asarray(frame, dtype=np.float64)
    opaque = f[..., 3] > 127
    lum = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]) / 255
    chroma = (f[..., :3].max(-1) - f[..., :3].min(-1)) / 255
    lum[~opaque] = 0
    local_max = lum.copy()
    for _ in range(radius):
        p = np.pad(local_max, 1)
        local_max = np.maximum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
    black = opaque & (lum < 0.2)
    grey = opaque & (lum < local_max * ratio) & (chroma < max_chroma) & ~black
    thick = ndimage.binary_opening(grey, iterations=max_width // 2 + 1)
    lab, n = ndimage.label(grey)
    fills = np.unique(lab[thick])
    grey &= ~np.isin(lab, fills[fills > 0])
    return black | grey


def trace_frame(job):
    """Trace one GIF frame into Lottie shapes (runs in a worker process)."""
    path, n, pal, a = job
    global SMOOTH
    SMOOTH = a.smooth
    im = Image.open(path)
    im.seek(n)
    w, h = im.size
    f = im.convert("RGBA")
    if a.upscale > 1:
        f = f.resize((w * a.upscale, h * a.upscale), Image.LANCZOS)
    u = a.upscale
    if pal is None:
        svg = vtracer.convert_pixels_to_svg(
            list(f.getdata()), size=f.size, colormode="color", hierarchical="stacked",
            mode="spline", filter_speckle=a.speckle * u, color_precision=6,
            layer_difference=16, corner_threshold=60, length_threshold=a.detail * u,
            splice_threshold=45, path_precision=2)
        return svg_to_groups(svg, 1 / u)
    shapes = trace_layers(snap(f, pal, fringe=u), pal, a.speckle * u, a.detail * u, 1 / u, round(a.dots * u))
    if a.ink:
        ink = pal[pal.sum(-1).argmin()]
        shapes = trace_mask(ink_mask(f, a.ink, radius=2 * u, max_width=3 * u), ink,
                            a.speckle * u, a.detail * u, 1 / u) + shapes
    return shapes


def main():
    ap = argparse.ArgumentParser(description="Trace a GIF into a Lottie (.json or .lottie).")
    ap.add_argument("gif"); ap.add_argument("out", help="out.json, or out.lottie for the zipped dotLottie")
    ap.add_argument("--colors", type=int, default=16, help="max palette size; real colours are auto-detected (0 = raw vtracer)")
    ap.add_argument("--upscale", type=int, default=3, help="trace at Nx resolution for finer lines")
    ap.add_argument("--speckle", type=int, default=2, help="drop blobs smaller than N source px")
    ap.add_argument("--ink", type=float, default=0.75, help="line-art layer darkness ratio (0 = off)")
    ap.add_argument("--dots", type=float, default=5, help="round blobs up to N source px become circles (0 = off)")
    ap.add_argument("--smooth", type=float, default=0.4, help="simplify + smooth tolerance in px (0 = off)")
    ap.add_argument("--detail", type=float, default=4.0, help="min segment length in px (higher = smaller file)")
    ap.add_argument("--step", type=int, default=1, help="keep every Nth frame (2 = animate on twos)")
    ap.add_argument("--min-dist", type=int, default=28, help="palette colours closer than this merge")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="frames traced in parallel")
    a = ap.parse_args()

    im = Image.open(a.gif)
    w, h = im.size
    pal = pick_palette(im, a.colors, a.min_dist) if a.colors else None
    if pal is not None:
        print(f"palette ({len(pal)}):", [tuple(int(v) for v in c) for c in pal])
    frames = list(range(0, im.n_frames, a.step))
    durs = []
    for k in range(im.n_frames):
        im.seek(k)
        durs.append(im.info.get("duration") or 100)  # GIF players treat 0 as ~100ms
    with Pool(min(a.jobs, len(frames))) as pool:
        traced = pool.map(trace_frame, [(a.gif, n, pal, a) for n in frames])

    layers, t = [], 0
    for n, shapes in zip(frames, traced):
        d = sum(durs[n:n + a.step])  # a held drawing covers the frames it skips
        layers.append({"ddd": 0, "ind": len(layers) + 1, "ty": 4, "nm": f"frame {n}", "sr": 1,
                       "ks": {"o": static(100), "r": static(0), "p": static([0, 0, 0]),
                              "a": static([0, 0, 0]), "s": static([100, 100, 100])},
                       "ao": 0, "shapes": shapes, "ip": t, "op": t + d, "st": 0, "bm": 0})
        t += d

    # timeline in ms (fr=1000) so every GIF delay is an exact integer span
    name = os.path.splitext(os.path.basename(a.out))[0]
    lottie = {"v": "5.7.4", "fr": 1000, "ip": 0, "op": t, "w": w, "h": h, "nm": name,
              "ddd": 0, "assets": [], "layers": layers}
    data = json.dumps(lottie, separators=(",", ":"))
    if a.out.endswith(".lottie"):
        with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.writestr("manifest.json", json.dumps({"version": "1", "generator": "gif_to_lottie",
                                                    "animations": [{"id": name}]}))
            z.writestr(f"animations/{name}.json", data)
    else:
        with open(a.out, "w") as fh:
            fh.write(data)
    print(f"{a.out}: {len(layers)} frames, {t} ms, {os.path.getsize(a.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
