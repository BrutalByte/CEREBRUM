"""Generate all raster image assets for the CEREBRUM marketing site."""
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent

BG       = (10,  10,  15)
SURFACE  = (17,  17,  24)
INDIGO   = (99,  102, 241)
CYAN     = (34,  211, 238)
WHITE    = (241, 245, 249)
DIMWHITE = (148, 163, 184)

def _indigo_alpha(a): return (*INDIGO, a)
def _cyan_alpha(a):   return (*CYAN,   a)


def draw_graph(draw, cx, cy, scale=1.0, alpha_mul=1.0):
    """Draw the CEREBRUM graph logo (query→path→answer) centered at cx,cy."""
    s = scale
    nodes = [
        (cx,        cy - 18*s),   # top   — query
        (cx - 16*s, cy),           # left
        (cx + 16*s, cy),           # right
        (cx,        cy + 18*s),   # bottom — answer
        (cx - 28*s, cy - 8*s),    # far left
        (cx + 28*s, cy - 8*s),    # far right
    ]
    edges = [
        (0, 1, INDIGO), (0, 2, INDIGO),
        (1, 3, CYAN),   (2, 3, CYAN),
        (0, 4, INDIGO), (0, 5, INDIGO),
        (1, 4, INDIGO), (2, 5, INDIGO),
    ]
    for (i, j, col) in edges:
        a = int(80 * alpha_mul)
        draw.line([nodes[i], nodes[j]], fill=(*col, a), width=max(1, int(1.5*s)))

    radii   = [int(6*s), int(5*s), int(5*s), int(7*s), int(4*s), int(4*s)]
    colors  = [INDIGO,   INDIGO,   INDIGO,   CYAN,     INDIGO,   INDIGO  ]
    alphas  = [220,      180,      180,      255,      140,      140     ]
    for (x, y), r, col, a in zip(nodes, radii, colors, alphas):
        a = int(a * alpha_mul)
        bbox = [x-r, y-r, x+r, y+r]
        draw.ellipse(bbox, fill=(*col, a))
    # answer node highlight ring
    ax, ay = nodes[3]
    r = radii[3]
    draw.ellipse([ax-r-2, ay-r-2, ax+r+2, ay+r+2],
                 outline=(*CYAN, int(160*alpha_mul)), width=2)


# ─── favicon 32×32 ────────────────────────────────────────────────────────────
def make_favicon_32():
    img = Image.new("RGBA", (32, 32), (0,0,0,0))
    d   = ImageDraw.Draw(img, "RGBA")
    # circle background
    d.ellipse([0,0,31,31], fill=(*BG, 255))
    d.ellipse([0,0,31,31], outline=(*INDIGO, 180), width=1)
    # tiny graph
    nodes = [(16,6),(8,16),(24,16),(14,26)]
    edges = [(0,1,INDIGO),(0,2,INDIGO),(1,3,CYAN),(2,3,CYAN)]
    for i,j,col in edges:
        d.line([nodes[i],nodes[j]], fill=(*col,120), width=1)
    radii  = [3,2,2,3]
    colors = [INDIGO,INDIGO,INDIGO,CYAN]
    for (x,y),r,col in zip(nodes,radii,colors):
        d.ellipse([x-r,y-r,x+r,y+r], fill=(*col,230))
    img.save(OUT / "favicon-32x32.png")
    print("favicon-32x32.png")

# ─── favicon 16×16 ────────────────────────────────────────────────────────────
def make_favicon_16():
    img = Image.new("RGBA", (16, 16), (0,0,0,0))
    d   = ImageDraw.Draw(img, "RGBA")
    d.ellipse([0,0,15,15], fill=(*BG, 255))
    d.ellipse([0,0,15,15], outline=(*INDIGO, 180), width=1)
    nodes = [(8,3),(4,8),(12,8),(7,13)]
    edges = [(0,1,INDIGO),(0,2,INDIGO),(1,3,CYAN),(2,3,CYAN)]
    for i,j,col in edges:
        d.line([nodes[i],nodes[j]], fill=(*col,120), width=1)
    for (x,y),col in zip(nodes,[INDIGO,INDIGO,INDIGO,CYAN]):
        r = 2 if col==CYAN else 1
        d.ellipse([x-r,y-r,x+r,y+r], fill=(*col,230))
    img.save(OUT / "favicon-16x16.png")
    print("favicon-16x16.png")

# ─── apple-touch-icon 180×180 ─────────────────────────────────────────────────
def make_apple_touch():
    sz  = 180
    img = Image.new("RGBA", (sz, sz), BG)
    d   = ImageDraw.Draw(img, "RGBA")
    # rounded rect bg
    d.rounded_rectangle([0,0,sz-1,sz-1], radius=36, fill=(*SURFACE,255),
                         outline=(*INDIGO,80), width=2)
    draw_graph(d, sz//2, sz//2, scale=3.5)
    img.save(OUT / "apple-touch-icon.png")
    print("apple-touch-icon.png")

# ─── OG image 1200×630 ────────────────────────────────────────────────────────
def make_og_image():
    W, H = 1200, 630
    img  = Image.new("RGBA", (W, H), BG)
    d    = ImageDraw.Draw(img, "RGBA")

    # subtle grid
    for x in range(0, W, 60):
        d.line([(x,0),(x,H)], fill=(255,255,255,6))
    for y in range(0, H, 60):
        d.line([(0,y),(W,y)], fill=(255,255,255,6))

    # top indigo accent bar
    d.rectangle([0,0,W,4], fill=(*INDIGO,255))

    # graph illustration — right side
    draw_graph(d, 920, 315, scale=7.0, alpha_mul=0.85)

    # faint glow under answer node
    for r in range(80, 0, -10):
        a = int(15 * (1 - r/80))
        d.ellipse([920-r, 415-r, 920+r, 415+r], fill=(*CYAN, a))

    # ── text ──────────────────────────────────────────────────────────────────
    try:
        font_xl  = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",  72)
        font_lg  = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",  42)
        font_md  = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",    28)
        font_sm  = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",    22)
        font_tag = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",  20)
    except OSError:
        font_xl = font_lg = font_md = font_sm = font_tag = ImageFont.load_default()

    # CEREBRUM wordmark
    d.text((80, 80), "CEREBRUM", font=font_xl, fill=WHITE)
    # indigo underline
    d.rectangle([80, 165, 500, 169], fill=(*INDIGO, 255))

    # tagline
    d.text((80, 190), "Training-Free Knowledge Graph Reasoning", font=font_md, fill=DIMWHITE)

    # stat badges
    badges = [
        ("60.4%",  "3-hop H@1",       INDIGO),
        ("90.7%",  "3-hop H@10",      CYAN),
        ("Zero",   "Training needed", (16,185,129)),   # emerald
    ]
    bx = 80
    for val, label, col in badges:
        bw = 180
        d.rounded_rectangle([bx, 260, bx+bw, 340], radius=10,
                             fill=(*col, 25), outline=(*col, 120), width=1)
        d.text((bx+bw//2, 278), val,   font=font_lg, fill=(*col,255), anchor="mt")
        d.text((bx+bw//2, 323), label, font=font_sm, fill=DIMWHITE,   anchor="mb")
        bx += bw + 20

    # bullet points
    bullets = [
        "Every answer is a verifiable graph path",
        "No LLM required — no hallucinations",
        "Works on any knowledge base out-of-the-box",
    ]
    by = 390
    for b in bullets:
        d.ellipse([80, by+7, 90, by+17], fill=(*CYAN,200))
        d.text((104, by), b, font=font_sm, fill=DIMWHITE)
        by += 42

    # bottom bar
    d.rectangle([0, H-48, W, H], fill=(*SURFACE,255))
    d.text((80, H-34),   "github.com/BrutalByte/CEREBRUM",
           font=font_tag, fill=(*INDIGO,200))
    d.text((W-80, H-34), "AGPL-3.0 • Commercial license available",
           font=font_tag, fill=(*DIMWHITE,150), anchor="ra")

    img.save(OUT / "og-image.png")
    print("og-image.png")

# ─── logo 512×512 (for PWA manifest etc.) ─────────────────────────────────────
def make_logo_512():
    sz  = 512
    img = Image.new("RGBA", (sz, sz), (0,0,0,0))
    d   = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([0,0,sz-1,sz-1], radius=sz//5,
                         fill=(*SURFACE,255), outline=(*INDIGO,120), width=3)
    draw_graph(d, sz//2, sz//2, scale=10.0)
    img.save(OUT / "logo-512.png")
    print("logo-512.png")

if __name__ == "__main__":
    make_favicon_16()
    make_favicon_32()
    make_apple_touch()
    make_og_image()
    make_logo_512()
    print("All assets generated.")
