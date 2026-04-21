"""把 308A8585.JPG 压缩 + 抠主体 + 像素化，输出 idle sprite sheet。
参考 zjy_idle_sheet.png 的风格：10 帧 × 128×128，PNG 带 alpha。

抠图思路：
1. 先只用"高饱和蓝"找到校服这块最大连通域 → 得到躯干 bbox
2. 把 bbox 向上延伸一段作为头部区域，在里头找黑发 + 肤色
3. 红领巾用颜色阈值补充
4. 所有 mask 只在"躯干 bbox + 头部扩展区"之内生效，bbox 外一律透明
"""
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from scipy import ndimage
import numpy as np
import math

SRC = "308A8585.JPG"
COMPRESSED = "zjy2_photo.jpg"
CLEAN = "zjy2_clean.png"
SHEET = "zjy2_idle_sheet.png"

FRAMES = 10
FRAME = 128
SHEET_W = FRAMES * FRAME
PIXEL_SCALE = 5

# === 1. 压缩 + 方向矫正 ===========================================
im = Image.open(SRC)
im = ImageOps.exif_transpose(im)
if im.width > im.height:
    im = im.rotate(-90, expand=True)
im.thumbnail((1600, 1200), Image.LANCZOS)
im.convert("RGB").save(COMPRESSED, quality=85, optimize=True)
print(f"[1/4] 压缩完成 → {COMPRESSED}  {im.size}")

rgb = np.array(im.convert("RGB"), dtype=np.int16)
R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
H, W = R.shape

# === 2a. 先锁定校服（高饱和蓝）=====================================
# 饱和蓝：B 比 R 高很多、比 G 也高、自身够亮但别白；同时把 R+G+B 的灰度拉开
strong_blue = (B - R > 45) & (B - G > 15) & (B > 100) & (R < 180)
# 连通域找最大那块，认定为上衣+短裤整体
labels, n = ndimage.label(strong_blue)
if n == 0:
    raise SystemExit("没检测到蓝衣，调阈值")
sizes = ndimage.sum(strong_blue, labels, range(1, n + 1))
main = np.argmax(sizes) + 1
shirt_mask = labels == main
ys, xs = np.where(shirt_mask)
sy0, sy1 = ys.min(), ys.max()
sx0, sx1 = xs.min(), xs.max()
shirt_h = sy1 - sy0
shirt_w = sx1 - sx0

# 头部区域：从躯干顶端向上 1.1×衣高，再在左右各加 25% 防切肩
head_top = max(0, sy0 - int(shirt_h * 1.1))
head_bot = sy0 + int(shirt_h * 0.15)
head_x0 = max(0, sx0 - int(shirt_w * 0.25))
head_x1 = min(W, sx1 + int(shirt_w * 0.25))

# 总 bbox = 躯干 + 头部 + 两侧手臂延伸
bx0 = max(0, sx0 - int(shirt_w * 0.30))
bx1 = min(W, sx1 + int(shirt_w * 0.30))
by0 = head_top
by1 = min(H, sy1 + int(shirt_h * 0.05))

# === 2b. 头部：黑发 + 肤色 =========================================
hair = (R < 75) & (G < 75) & (B < 85)
# 严格肤色：带饱和度过滤，排除阳光下的白/花的暖黄
chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
skin = ((R >= 140) & (R <= 215)                 # 适中亮度，排除过曝白
        & (R > G) & (G > B)                     # 橙偏红
        & (R - B >= 25) & (R - B <= 90)
        & (chroma >= 18) & (chroma <= 70))      # 饱和度在皮肤合理范围

bbox_mask = np.zeros_like(shirt_mask)
bbox_mask[by0:by1, bx0:bx1] = True
head_region = np.zeros_like(shirt_mask)
head_region[head_top:head_bot, head_x0:head_x1] = True
hair = hair & head_region
skin = skin & bbox_mask

# === 2c. 红领巾 ====================================================
red = (R - G > 40) & (R - B > 30) & (R > 130) & bbox_mask

# === 2d. 合并 + 形态学清理 =========================================
# 肤色只保留"靠近蓝衣/头发"的像素，避免背景白墙被当肤色
shirt_or_hair = shirt_mask | hair
proximity = ndimage.binary_dilation(shirt_or_hair, iterations=30)  # 30 像素内
skin = skin & proximity

fg = (shirt_mask | hair | skin | red) & bbox_mask
# 再做一次"只留大块"过滤，清掉头顶/边缘残留的小云团
lab2, n2 = ndimage.label(fg)
if n2 > 0:
    sz2 = ndimage.sum(fg, lab2, range(1, n2 + 1))
    # 保留 top 3 大块（躯干、头、可能分离的手）
    order = np.argsort(sz2)[::-1]
    keep_ids = set((order[:3] + 1).tolist())
    # 并且尺寸必须 > 总面积的 2%
    min_size = 0.02 * fg.sum()
    keep_ids = {k for k in keep_ids if sz2[k - 1] > min_size}
    fg = np.isin(lab2, list(keep_ids))

mask_img = Image.fromarray((fg.astype(np.uint8)) * 255, "L")
mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
mask_img = mask_img.filter(ImageFilter.MinFilter(5))
mask_img = mask_img.filter(ImageFilter.MaxFilter(5))
mask_img = mask_img.filter(ImageFilter.GaussianBlur(0.6))
mask_np = np.array(mask_img)

# 再取最大连通域，确保背景漏网之鱼被剔除
lab, ncc = ndimage.label(mask_np > 90)
if ncc > 0:
    ssz = ndimage.sum(mask_np > 90, lab, range(1, ncc + 1))
    keep = np.argmax(ssz) + 1
    mask_np = np.where(lab == keep, mask_np, 0).astype(np.uint8)

# 裁剪到人物 bbox
ys, xs = np.where(mask_np > 60)
x0, x1 = xs.min(), xs.max()
y0, y1 = ys.min(), ys.max()
pad = 10
x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)

crop_rgb = np.array(im.convert("RGB"))[y0:y1, x0:x1]
crop_mask = mask_np[y0:y1, x0:x1]
rgba = np.dstack([crop_rgb, crop_mask]).astype(np.uint8)
clean = Image.fromarray(rgba, "RGBA")
clean.save(CLEAN)
print(f"[2/4] 抠图完成 → {CLEAN}  {clean.size}")

# === 3. 像素化 =====================================================
# 目标像素尺寸（低分辨率"像素画"版本），再用 NEAREST 放大到 128 高
LOW_H = 44                                    # 低分辨率角色高度 ≈ 44 像素
aspect = clean.width / clean.height
LOW_W = max(1, int(round(LOW_H * aspect)))

# 先把 alpha 单独保存并二值化
_, _, _, a0 = clean.split()
a0 = np.array(a0)
a0 = np.where(a0 > 80, 255, 0).astype(np.uint8)
clean_rgba = np.dstack([np.array(clean.convert("RGB")), a0])
# 把背景位置统一填成纯黑，避免 LANCZOS 时白色/黄色糊进头发边
clean_rgba[..., :3][a0 == 0] = 0
clean = Image.fromarray(clean_rgba, "RGBA")

low = clean.resize((LOW_W, LOW_H), Image.LANCZOS)
# 量化到 18 色，加强像素画感
low_rgb = low.convert("RGB").quantize(colors=18, method=Image.MEDIANCUT, dither=0).convert("RGBA")
low_rgb.putalpha(low.split()[-1])
# 二值化 alpha
la = np.array(low_rgb.split()[-1])
la = np.where(la > 110, 255, 0).astype(np.uint8)
la_arr = np.array(low_rgb); la_arr[..., 3] = la
low_rgb = Image.fromarray(la_arr, "RGBA")

# 放大到 frame 高度（NEAREST 保持硬边）
scale = FRAME / LOW_H
pixelated = low_rgb.resize((int(LOW_W * scale), FRAME), Image.NEAREST)
print(f"[3/4] 像素化完成 → frame size {pixelated.size}")

# === 4. 拼 sprite sheet ============================================
sheet = Image.new("RGBA", (SHEET_W, FRAME), (0, 0, 0, 0))
pw, ph = pixelated.size
if pw > FRAME or ph > FRAME:
    s = min(FRAME / pw, FRAME / ph)
    pixelated = pixelated.resize((max(1, int(pw * s)), max(1, int(ph * s))), Image.NEAREST)

bw, bh = pixelated.size
for i in range(FRAMES):
    bob = int(round(math.sin(i / FRAMES * 2 * math.pi) * 1))
    x = (FRAME - bw) // 2
    y = FRAME - bh + bob
    sheet.paste(pixelated, (i * FRAME + x, y), pixelated)
    if 3 <= i <= 7:
        strength = [0, 0, 0, 60, 120, 160, 120, 60, 0, 0][i]
        blush = Image.new("RGBA", (FRAME, FRAME), (0, 0, 0, 0))
        d = ImageDraw.Draw(blush)
        face_cy = y + int(bh * 0.18)
        r = 5
        for cx in (x + int(bw * 0.38), x + int(bw * 0.62)):
            d.ellipse([cx - r, face_cy - r, cx + r, face_cy + r],
                      fill=(220, 70, 70, strength))
        sheet.alpha_composite(blush, (i * FRAME, 0))

sheet.save(SHEET)
print(f"[4/4] Sprite sheet 完成 → {SHEET}  {sheet.size}")
