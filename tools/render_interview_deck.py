"""Render the final interview deck to per-slide PNGs for visual review."""

from pathlib import Path

import aspose.slides as slides


DECK = Path("docs/presentation/FunRec_Interview_Story_Final.pptx")
OUTPUT = Path("docs/presentation/FunRec_Interview_Story_Final_rendered")


def main():
    if not DECK.exists():
        raise FileNotFoundError(DECK)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    with slides.Presentation(str(DECK)) as presentation:
        for index, slide in enumerate(presentation.slides, 1):
            image = slide.get_image(1.5, 1.5)
            image.save(
                str(OUTPUT / f"slide-{index:02d}.png"),
                slides.ImageFormat.PNG,
            )
    print(f"Rendered {len(presentation.slides)} slides to {OUTPUT}")


if __name__ == "__main__":
    main()
