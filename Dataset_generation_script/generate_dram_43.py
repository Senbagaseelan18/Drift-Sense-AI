import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRAM-43
# Line Collapse + Contact Shift Defect
# ============================================================


SEED = 20260813

rng = np.random.default_rng(SEED)


OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_43"


SIZE = 10000

REF_SIZE = 1000


# defect position

DEFECT_X = 5200
DEFECT_Y = 4700


# ============================================================
# DRAW HELPERS
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
# BASE DRAM STRUCTURE
# ============================================================


def create_structure():
    img = np.full(
        (
            SIZE,
            SIZE
        ),
        18,
        dtype=np.uint8
    )

    # vertical wordlines
    for x in range(
        300,
        SIZE - 300,
        70
    ):
        line(
            img,
            x,
            300,
            x,
            SIZE - 300,
            100,
            5
        )

    # contacts
    for row, y in enumerate(
        range(
            500,
            SIZE - 500,
            120
        )
    ):
        offset = 0
        if row % 2:
            offset = 35

        for x in range(
            350 + offset,
            SIZE - 350,
            140
        ):
            circle(
                img,
                x,
                y,
                18,
                220
            )

    return img


# ============================================================
# DEFECT CREATION
# ============================================================


def add_line_collapse(img):
    # merge two lines
    x = DEFECT_X
    line(
        img,
        x,
        DEFECT_Y - 500,
        x + 45,
        DEFECT_Y + 500,
        150,
        18
    )

    # melted region
    cv2.circle(
        img,
        (
            x,
            DEFECT_Y
        ),
        80,
        145,
        -1
    )



def add_contact_shift(img):
    # erase original
    cv2.circle(
        img,
        (
            DEFECT_X + 120,
            DEFECT_Y + 120
        ),
        25,
        18,
        -1
    )

    # shifted contact
    cv2.circle(
        img,
        (
            DEFECT_X + 165,
            DEFECT_Y + 95
        ),
        18,
        230,
        -1
    )


# ============================================================
# SEM PHYSICS
# ============================================================


def reference_sem(img):
    out = cv2.GaussianBlur(
        img,
        (3, 3),
        0.25
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

    # scan drift
    drift = np.sin(
        np.arange(
            img.shape[0]
        ) / 18
    )
    out += drift[:, None] * 3

    # charging cloud
    cloud = cv2.GaussianBlur(
        rng.normal(
            0,
            15,
            img.shape
        ),
        (101, 101),
        0
    )
    out += cloud

    # detector noise
    out += rng.normal(
        0,
        3,
        img.shape
    )

    # hot pixels
    mask = rng.random(
        img.shape
    ) < 0.002
    out[mask] += 40

    return np.clip(
        out,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 60)
    print("DRIFT-SENSE DRAM-43")
    print("=" * 60)

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    scene = create_structure()

    # add fabrication defects
    add_line_collapse(scene)
    add_contact_shift(scene)

    # reference around defect
    ref_x = DEFECT_X - 500
    ref_y = DEFECT_Y - 500

    reference = scene[
        ref_y:
        ref_y + REF_SIZE,
        ref_x:
        ref_x + REF_SIZE
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
        "pair": "dram_43",
        "architecture": "dense_vertical_dram_line_field",
        "defect": [
            "line_collapse",
            "contact_shift"
        ],
        "defect_nm": [
            DEFECT_X,
            DEFECT_Y
        ],
        "reference_origin": [
            ref_x,
            ref_y
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

    print("DRAM-43 COMPLETE")


if __name__ == "__main__":
    main()
