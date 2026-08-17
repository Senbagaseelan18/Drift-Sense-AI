import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRAM-44
# Multi Die DRAM With Three Dark Structures
# ============================================================


SEED = 20260814

rng = np.random.default_rng(SEED)


OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_44"


SIZE = 10000

REF_SIZE = 1000

DIE = 3000


# target black structure

TARGET = (
    6500,
    5200
)


# ============================================================
# HELPERS
# ============================================================


def line(img, x1, y1, x2, y2, v, w):
    cv2.line(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        int(v),
        int(w),
        lineType=cv2.LINE_AA
    )



def circle(img, x, y, r, v):
    cv2.circle(
        img,
        (int(x), int(y)),
        int(r),
        int(v),
        -1,
        lineType=cv2.LINE_AA
    )


# ============================================================
# DRAM DIE
# ============================================================


def create_dram_die(img, x, y):
    cv2.rectangle(
        img,
        (x, y),
        (x + DIE, y + DIE),
        20,
        -1
    )

    cv2.rectangle(
        img,
        (x + 100, y + 100),
        (x + DIE - 100, y + DIE - 100),
        50,
        8
    )

    for row, yy in enumerate(
        range(
            y + 250,
            y + DIE - 250,
            120
        )
    ):
        offset = 0
        if row % 2:
            offset = 60

        for xx in range(
            x + 250 + offset,
            x + DIE - 250,
            120
        ):
            circle(
                img,
                xx,
                yy,
                14,
                210
            )
            circle(
                img,
                xx,
                yy,
                4,
                250
            )

    for yy in range(
        y + 200,
        y + DIE - 200,
        400
    ):
        line(
            img,
            x + 150,
            yy,
            x + DIE - 150,
            yy,
            70,
            5
        )


# ============================================================
# DARK STRUCTURES
# ============================================================


def dark_rectangle(img, x, y):
    cv2.rectangle(
        img,
        (x, y),
        (x + 220, y + 220),
        5,
        -1
    )
    cv2.rectangle(
        img,
        (x - 15, y - 15),
        (x + 235, y + 235),
        60,
        4
    )



def dark_irregular(img, x, y):
    pts = np.array(
        [
            [x, y],
            [x + 260, y + 30],
            [x + 220, y + 240],
            [x + 40, y + 200]
        ],
        np.int32
    )
    cv2.fillPoly(
        img,
        [pts],
        5
    )



def add_dark_features(img):
    dark_rectangle(
        img,
        1800,
        2200
    )
    dark_irregular(
        img,
        7200,
        1800
    )
    dark_rectangle(
        img,
        TARGET[0],
        TARGET[1]
    )


# ============================================================
# SEARCH SCENE
# ============================================================


def create_scene():
    img = np.full(
        (
            SIZE,
            SIZE
        ),
        15,
        dtype=np.uint8
    )

    positions = [
        (200, 200),
        (3500, 200),
        (6800, 200),
        (200, 3500),
        (3500, 3500),
        (6800, 3500)
    ]

    for p in positions:
        create_dram_die(
            img,
            p[0],
            p[1]
        )

    add_dark_features(img)
    return img


# ============================================================
# SEM EFFECT
# ============================================================


def reference_sem(img):
    out = cv2.GaussianBlur(
        img,
        (3, 3),
        0.3
    )
    out = out.astype(float)
    out += rng.normal(
        0,
        1,
        img.shape
    )
    return np.clip(
        out,
        0,
        255
    ).astype(np.uint8)



def search_sem(img):
    out = cv2.GaussianBlur(
        img,
        (5, 5),
        1
    ).astype(float)

    ripple = np.sin(
        np.arange(
            img.shape[0]
        ) / 12
    )
    out += ripple[:, None] * 3

    halo = cv2.GaussianBlur(
        rng.normal(
            0,
            20,
            img.shape
        ),
        (121, 121),
        0
    )
    out += halo
    out += rng.normal(
        0,
        3,
        img.shape
    )
    return np.clip(
        out,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# MAIN
# ============================================================


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    scene = create_scene()

    rx = TARGET[0] - 400
    ry = TARGET[1] - 400

    reference = scene[
        ry:
        ry + REF_SIZE,
        rx:
        rx + REF_SIZE
    ]

    search = cv2.resize(
        scene,
        (1000, 1000),
        interpolation=cv2.INTER_AREA
    )

    reference = reference_sem(reference)
    search = search_sem(search)

    cv2.imwrite(
        str(
            OUT / "reference_100x.png"
        ),
        reference
    )

    cv2.imwrite(
        str(
            OUT / "search_10x.png"
        ),
        search
    )

    gt = {
        "pair": "dram_44",
        "architecture": "multi_die_dram_three_dark_structures",
        "target_structure": TARGET,
        "reference_origin": [
            rx,
            ry
        ],
        "scale_ratio": 10
    }

    with open(
        OUT / "ground_truth.json",
        "w"
    ) as f:
        json.dump(
            gt,
            f,
            indent=4
        )

    print("DRAM-44 COMPLETE")


if __name__ == "__main__":
    main()
