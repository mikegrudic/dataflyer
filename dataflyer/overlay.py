"""Dev overlay: renders text HUD in the top-right corner."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

SHADER_DIR = Path(__file__).parent / "shaders"


def _load_shader(name):
    return (SHADER_DIR / name).read_text()


class DevOverlay:
    def __init__(self, ctx):
        self.ctx = ctx
        self.enabled = False
        self._tex = None
        self._vao = None
        self._prog = ctx.program(
            vertex_shader=_load_shader("text.vert"),
            fragment_shader=_load_shader("text.frag"),
        )
        # Quad VBO: position + UV, will be updated per frame
        self._vbo = ctx.buffer(reserve=6 * 4 * 4)  # 6 verts * 4 floats * 4 bytes
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, "2f 2f", "in_position", "in_uv")],
        )
        self._font = self._get_font(14)

    def _get_font(self, size):
        try:
            import matplotlib.font_manager as fm
            path = fm.findfont(fm.FontProperties(family="monospace"))
            if path:
                return ImageFont.truetype(path, size)
        except Exception:
            pass
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    def update(self, lines, fb_width, fb_height):
        """Render text lines to a texture and position it in the top-right."""
        if not self.enabled or not lines:
            return

        # Render text to PIL image
        text = "\n".join(lines)
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.multiline_textbbox((0, 0), text, font=self._font)
        tw = bbox[2] - bbox[0] + 16
        th = bbox[3] - bbox[1] + 12

        img = Image.new("RGBA", (tw, th), (0, 0, 0, 180))
        draw = ImageDraw.Draw(img)
        draw.multiline_text((8, 6), text, fill=(0, 255, 0, 255), font=self._font)

        # Upload as texture
        data = img.tobytes()
        if self._tex is not None:
            self._tex.release()
        self._tex = self.ctx.texture((tw, th), 4, data=data)
        self._tex.filter = (0x2601, 0x2601)  # GL_LINEAR

        # Position quad in top-right corner (NDC coordinates)
        # NDC: x in [-1, 1], y in [-1, 1], top-right is (1, 1)
        px_w = tw / fb_width * 2  # width in NDC
        px_h = th / fb_height * 2  # height in NDC
        x1 = 1.0 - px_w - 0.01  # small margin from right edge
        x2 = 1.0 - 0.01
        y1 = 1.0 - px_h - 0.01  # small margin from top
        y2 = 1.0 - 0.01

        # Two triangles, UV flipped vertically (PIL origin is top-left)
        verts = np.array([
            x1, y1, 0, 1,
            x2, y1, 1, 1,
            x1, y2, 0, 0,
            x2, y1, 1, 1,
            x2, y2, 1, 0,
            x1, y2, 0, 0,
        ], dtype=np.float32)
        self._vbo.write(verts.tobytes())

    def render(self):
        if not self.enabled or self._tex is None:
            return
        self.ctx.enable(0x0BE2)  # GL_BLEND
        self.ctx.blend_func = (0x0302, 0x0303)  # SRC_ALPHA, ONE_MINUS_SRC_ALPHA
        self._tex.use(location=0)
        self._prog["u_texture"].value = 0
        self._vao.render(vertices=6)
        self.ctx.disable(0x0BE2)

    def release(self):
        for attr in ("_tex", "_vbo", "_vao", "_prog"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
