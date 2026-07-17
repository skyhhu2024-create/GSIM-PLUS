"""Rebuild and package the nine figures used by the ESSD manuscript."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(
    __import__("os").environ.get("GSIM_PLUS_PROJECT_DIR", SCRIPT_DIR.parent)
).resolve()
PAPER_DIR = PROJECT_DIR / "111-paper"
OUTDIR = PAPER_DIR / "ESSD_Revised_Figures"
TARGET_WIDTH = 4252  # 180 mm at 600 dpi
GAP = 36


def run(script: Path) -> None:
    print(f"Running {script.name} ...")
    subprocess.run([sys.executable, str(script)], check=True, cwd=script.parent)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def resize_width(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def panel_font(size: int = 68):
    windows_dir = __import__("os").environ.get("WINDIR")
    candidates = []
    if windows_dir:
        windows_fonts = Path(windows_dir) / "Fonts"
        candidates.extend(
            [
                windows_fonts / "timesbi.ttf",
                windows_fonts / "timesbd.ttf",
                windows_fonts / "times.ttf",
            ]
        )
    candidates.extend(
        [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def label_panel(image: Image.Image, label: str, x: int = 46, y: int = 30) -> None:
    ImageDraw.Draw(image).text((x, y), label, fill="black", font=panel_font())


def save_png(image: Image.Image, name: str) -> None:
    image.save(OUTDIR / name, dpi=(600, 600), compress_level=4)


def compose_fig1() -> None:
    panel_a = resize_width(load_rgb(PAPER_DIR / "Fig1a_all_stations.png"), TARGET_WIDTH)
    panel_b = resize_width(load_rgb(PAPER_DIR / "Fig1c_histogram.png"), (TARGET_WIDTH - GAP) // 2)
    panel_c = resize_width(load_rgb(PAPER_DIR / "Fig1d_cdf.png"), (TARGET_WIDTH - GAP) // 2)
    panel_d = resize_width(load_rgb(PAPER_DIR / "Fig1b_anchor_target.png"), TARGET_WIDTH)

    middle_height = max(panel_b.height, panel_c.height)
    total_height = panel_a.height + middle_height + panel_d.height + 2 * GAP
    canvas = Image.new("RGB", (TARGET_WIDTH, total_height), "white")
    canvas.paste(panel_a, (0, 0))
    middle_y = panel_a.height + GAP
    canvas.paste(panel_b, (0, middle_y))
    canvas.paste(panel_c, (panel_b.width + GAP, middle_y))
    bottom_y = middle_y + middle_height + GAP
    canvas.paste(panel_d, (0, bottom_y))

    label_panel(canvas, "(a)")
    label_panel(canvas, "(b)", y=middle_y + 20)
    label_panel(canvas, "(c)", x=panel_b.width + GAP + 36, y=middle_y + 20)
    label_panel(canvas, "(d)", y=bottom_y + 20)
    save_png(canvas, "Fig1.png")

    component_map = {
        "Fig1a.pdf": "Fig1a_all_stations.pdf",
        "Fig1b.pdf": "Fig1c_histogram.pdf",
        "Fig1c.pdf": "Fig1d_cdf.pdf",
        "Fig1d.pdf": "Fig1b_anchor_target.pdf",
    }
    for target, source in component_map.items():
        shutil.copy2(PAPER_DIR / source, OUTDIR / target)


def compose_fig5() -> None:
    sources = [
        PAPER_DIR / "Fig8a_koppen_NSE.png",
        PAPER_DIR / "Fig8b_koppen_KGE.png",
        PAPER_DIR / "Fig8d_koppen_Bias.png",
    ]
    panels = [resize_width(load_rgb(path), TARGET_WIDTH) for path in sources]
    total_height = sum(panel.height for panel in panels) + GAP * (len(panels) - 1)
    canvas = Image.new("RGB", (TARGET_WIDTH, total_height), "white")

    y = 0
    for label, panel in zip(["(a)", "(b)", "(c)"], panels):
        canvas.paste(panel, (0, y))
        label_panel(canvas, label, y=y + 24)
        y += panel.height + GAP
    save_png(canvas, "Fig5.png")

    for target, source in [
        ("Fig5a.pdf", "Fig8a_koppen_NSE.pdf"),
        ("Fig5b.pdf", "Fig8b_koppen_KGE.pdf"),
        ("Fig5c.pdf", "Fig8d_koppen_Bias.pdf"),
    ]:
        shutil.copy2(PAPER_DIR / source, OUTDIR / target)


def copy_figure(number: int, png: Path, pdf: Path) -> None:
    shutil.copy2(png, OUTDIR / f"Fig{number}.png")
    shutil.copy2(pdf, OUTDIR / f"Fig{number}.pdf")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if "--assemble-only" not in sys.argv:
        scripts = [
            SCRIPT_DIR / "Fig1_global_station_overview.py",
            SCRIPT_DIR / "GRDC_CrossValidation" / "plot_GRDC_final.py",
            SCRIPT_DIR / "plot_taylor_professional.py",
            SCRIPT_DIR / "Fig8_koppen_boxplot.py",
            SCRIPT_DIR / "plot_lowflow_boundary_figures.py",
            SCRIPT_DIR / "Fig9d_dual_cdf.py",
            SCRIPT_DIR / "Fig9_product_quality.py",
            SCRIPT_DIR / "Fig10_timeseries.py",
        ]
        for script in scripts:
            run(script)

    compose_fig1()
    compose_fig5()
    grdc = PAPER_DIR / "GRDC独立验证"
    copy_figure(2, grdc / "Fig_GRDC_timeseries.png", grdc / "Fig_GRDC_timeseries.pdf")
    copy_figure(3, grdc / "Fig_GRDC_scatter.png", grdc / "Fig_GRDC_scatter.pdf")
    copy_figure(4, grdc / "Fig_GRDC_taylor_v2.png", grdc / "Fig_GRDC_taylor_v2.pdf")
    copy_figure(6, PAPER_DIR / "Fig11_why_002_threshold.png", PAPER_DIR / "Fig11_why_002_threshold.pdf")
    copy_figure(7, PAPER_DIR / "Fig9d_completeness_dual_cdf.png", PAPER_DIR / "Fig9d_completeness_dual_cdf.pdf")
    copy_figure(8, PAPER_DIR / "Fig9c_quality_map.png", PAPER_DIR / "Fig9c_quality_map.pdf")
    copy_figure(9, PAPER_DIR / "Fig10_timeseries.png", PAPER_DIR / "Fig10_timeseries.pdf")

    print(f"Submission figures written to: {OUTDIR}")


if __name__ == "__main__":
    main()
