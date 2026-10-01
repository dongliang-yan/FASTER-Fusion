"""Draws the app icon: the swept sector, the vessel in colour flow, and the
magenta SIVV arc, on the navy of the rail. Writes icon.png (and .icns/.ico)."""
import math, os, subprocess, sys
from PIL import Image, ImageDraw, ImageFilter

S = 1024
here = os.path.dirname(os.path.abspath(__file__))
res = os.path.join(here, "..", "faster_fusion", "resources")
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([40, 40, S - 40, S - 40], radius=200, fill=(26, 36, 56, 255))
apex = (S / 2, 230)
R = 620
a0, a1 = math.radians(90 - 32), math.radians(90 + 32)
fan = [apex] + [(apex[0] + R * math.cos(a), apex[1] + R * math.sin(a))
                for a in [a0 + (a1 - a0) * i / 60 for i in range(61)]]
d.polygon(fan, fill=(92, 104, 128, 255))
d.ellipse([apex[0] - 60, apex[1] - 60, apex[0] + 60, apex[1] + 60], fill=(210, 215, 225, 255))
# vessel (colour flow)
v = Image.new("RGBA", (S, S), (0, 0, 0, 0)); dv = ImageDraw.Draw(v)
dv.ellipse([420, 560, 610, 700], fill=(245, 150, 20, 255))
dv.ellipse([455, 590, 575, 670], fill=(253, 215, 40, 255))
img.alpha_composite(v.filter(ImageFilter.GaussianBlur(6)))
# SIVV constant-range arc
r = 420
box = [apex[0] - r, apex[1] - r, apex[0] + r, apex[1] + r]
d.arc(box, 90 - 32, 90 + 32, fill=(255, 64, 255, 255), width=34)
os.makedirs(res, exist_ok=True)
img.save(os.path.join(res, "icon.png"))
img.save(os.path.join(here, "icon.ico"), sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
if sys.platform == "darwin":
    iset = os.path.join(here, "icon.iconset"); os.makedirs(iset, exist_ok=True)
    for n in (16, 32, 128, 256, 512):
        img.resize((n, n), Image.LANCZOS).save(os.path.join(iset, f"icon_{n}x{n}.png"))
        img.resize((2 * n, 2 * n), Image.LANCZOS).save(os.path.join(iset, f"icon_{n}x{n}@2x.png"))
    subprocess.run(["iconutil", "-c", "icns", iset, "-o", os.path.join(here, "icon.icns")], check=True)
print("icon written")
