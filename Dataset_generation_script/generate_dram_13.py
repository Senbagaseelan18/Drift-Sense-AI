#!/usr/bin/env python3

"""
DRIFT-SENSE DRAM-13 Generator

Structure:
    Curved wordline DRAM array
    Vertical bitlines
    Circular storage contacts
    Elongated capacitor pillars

Output:

results/
└── dram_13/
    ├── reference_100x.png
    ├── search_10x.png
    └── ground_truth.json
"""

from pathlib import Path
import json
import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

SEED = 20260822

PHYSICAL_SIZE = 10000

REFERENCE_SIZE = 1000

SEARCH_SIZE = 1000

TARGET_X = 4300
TARGET_Y = 4500


OUTPUT = (
    Path(__file__).resolve().parents[1]
    /
    "results" / "generated_dataset_images" / "dram_13"
)


# ============================================================
# DRAM PARAMETERS
# ============================================================

CELL_PITCH_X = 150
CELL_PITCH_Y = 150


DIE_SIZE = 2200
DIE_GAP = 300
MARGIN = 200


# ============================================================
# DRAW HELPERS
# ============================================================

def line(
    img,
    p1,
    p2,
    color,
    width
):

    cv2.line(
        img,
        tuple(map(int,p1)),
        tuple(map(int,p2)),
        int(color),
        int(width),
        lineType=cv2.LINE_AA
    )


def circle(
    img,
    x,
    y,
    r,
    value
):

    cv2.circle(
        img,
        (int(x),int(y)),
        int(r),
        int(value),
        -1,
        lineType=cv2.LINE_AA
    )


# ============================================================
# SINGLE DRAM CELL
# ============================================================

def draw_cell(
    img,
    x,
    y,
    rng
):

    # curved vertical wordline

    pts=[]

    for t in np.linspace(-1,1,20):

        px=x+20*t

        py=y+70*t

        pts.append(
            (
                px,
                py
            )
        )

    for a,b in zip(
        pts[:-1],
        pts[1:]
    ):

        line(
            img,
            a,
            b,
            55,
            5
        )


    # horizontal bitline

    line(
        img,
        (
            x-45,
            y
        ),
        (
            x+45,
            y
        ),
        70,
        7
    )


    # elongated storage contact

    cv2.ellipse(
        img,
        (
            int(x+18),
            int(y-5)
        ),
        (
            18,
            35
        ),
        -20,
        0,
        360,
        170,
        -1
    )


    # contact dot

    circle(
        img,
        x+18,
        y-30,
        10,
        220
    )


# ============================================================
# DIE
# ============================================================

def draw_die(
    img,
    x0,
    y0,
    rng
):

    # dark die background

    cv2.rectangle(
        img,
        (
            x0,
            y0
        ),
        (
            x0+DIE_SIZE,
            y0+DIE_SIZE
        ),
        25,
        -1
    )


    # DRAM cells

    row=0

    y=y0+120


    while y < y0+DIE_SIZE-100:

        x=x0+100

        while x < x0+DIE_SIZE-100:

            draw_cell(
                img,
                x,
                y,
                rng
            )

            x+=CELL_PITCH_X


        y+=CELL_PITCH_Y
        row+=1



    # die border

    cv2.rectangle(
        img,
        (
            x0+20,
            y0+20
        ),
        (
            x0+DIE_SIZE-20,
            y0+DIE_SIZE-20
        ),
        45,
        8
    )



# ============================================================
# UNIQUE FEATURE
# ============================================================

def draw_landmark(img):

    cx = TARGET_X+500
    cy = TARGET_Y+500


    # remove one contact region

    cv2.rectangle(
        img,
        (
            cx-80,
            cy-80
        ),
        (
            cx+80,
            cy+80
        ),
        28,
        -1
    )


    # three normal contacts

    circle(
        img,
        cx-45,
        cy-30,
        14,
        230
    )

    circle(
        img,
        cx+45,
        cy-30,
        14,
        230
    )

    circle(
        img,
        cx-45,
        cy+40,
        14,
        230
    )


    # missing lower right contact


    # abnormal bridge

    line(
        img,
        (
            cx-50,
            cy
        ),
        (
            cx+50,
            cy
        ),
        110,
        8
    )



# ============================================================
# NOISE
# ============================================================

def add_noise(
    img,
    rng
):

    out=img.astype(
        np.float32
    )


    # SEM charging gradient

    gradient=np.linspace(
        -4,
        4,
        out.shape[1]
    )

    out += gradient



    # scan variation

    rows=rng.normal(
        0,
        1.5,
        out.shape[0]
    )

    out += rows[:,None]



    # detector noise

    out += rng.normal(
        0,
        2,
        out.shape
    )


    # shot noise

    out += rng.poisson(
        1.5,
        out.shape
    )


    return np.clip(
        out,
        0,
        255
    ).astype(
        np.uint8
    )



# ============================================================
# GENERATION
# ============================================================

def main():

    print("="*70)
    print("DRIFT-SENSE DRAM-13 GENERATOR")
    print("="*70)


    rng=np.random.default_rng(
        SEED
    )


    scene=np.full(
        (
            PHYSICAL_SIZE,
            PHYSICAL_SIZE
        ),
        20,
        dtype=np.uint8
    )


    # 4x4 DRAM dies

    for r in range(4):

        for c in range(4):

            x=(
                MARGIN+
                c*(DIE_SIZE+DIE_GAP)
            )

            y=(
                MARGIN+
                r*(DIE_SIZE+DIE_GAP)
            )


            draw_die(
                scene,
                x,
                y,
                rng
            )



    # unique pattern

    draw_landmark(
        scene
    )



    # reference crop

    reference=scene[
        TARGET_Y:
        TARGET_Y+REFERENCE_SIZE,

        TARGET_X:
        TARGET_X+REFERENCE_SIZE
    ]


    # search image

    search=cv2.resize(
        scene,
        (
            SEARCH_SIZE,
            SEARCH_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )


    # imaging degradation

    reference=cv2.GaussianBlur(
        reference,
        (3,3),
        0.3
    )


    search=cv2.GaussianBlur(
        search,
        (5,5),
        0.8
    )


    search=add_noise(
        search,
        rng
    )



    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )


    cv2.imwrite(
        str(
            OUTPUT/
            "reference_100x.png"
        ),
        reference
    )


    cv2.imwrite(
        str(
            OUTPUT/
            "search_10x.png"
        ),
        search
    )


    gt={

        "pair":
        "dram_13",

        "architecture":
        "curved_wordline_contact_dram",

        "target_origin_nm":
        [
            TARGET_X,
            TARGET_Y
        ],

        "search_center_px":
        [
            TARGET_X/10+50,
            TARGET_Y/10+50
        ],

        "noise":
        [
            "charging_gradient",
            "scan_variation",
            "shot_noise",
            "detector_noise"
        ]

    }


    with open(
        OUTPUT/
        "ground_truth.json",
        "w"
    ) as f:

        json.dump(
            gt,
            f,
            indent=4
        )


    print()
    print("DRAM-13 GENERATED")
    print()
    print(
        OUTPUT
    )



if __name__=="__main__":

    main()