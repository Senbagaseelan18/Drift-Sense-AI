import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRAM-50
# Advanced Multi Region DRAM Benchmark
# Defect:
# Partial Capacitor Collapse + Void
# ============================================================


SEED = 20260820

rng = np.random.default_rng(SEED)


OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_50"


SIZE = 10000

REF_SIZE = 1000


DEFECT_X = 5600

DEFECT_Y = 5400


# ============================================================
# HELPERS
# ============================================================


def line(img,x1,y1,x2,y2,val,w):

    cv2.line(
        img,
        (int(x1),int(y1)),
        (int(x2),int(y2)),
        int(val),
        int(w),
        lineType=cv2.LINE_AA
    )



def circle(img,x,y,r,val):

    cv2.circle(
        img,
        (int(x),int(y)),
        int(r),
        int(val),
        -1,
        lineType=cv2.LINE_AA
    )


# ============================================================
# DRAM REGIONS
# ============================================================


def vertical_region(img,x,y,w,h):


    for xx in range(
        x,
        x+w,
        55
    ):

        line(
            img,
            xx,
            y,
            xx,
            y+h,
            100,
            5
        )


    for yy in range(
        y+40,
        y+h,
        100
    ):

        for xx in range(
            x+20,
            x+w,
            110
        ):

            circle(
                img,
                xx,
                yy,
                10,
                220
            )



def capacitor_region(img,x,y,w,h):


    pitch=140


    for row,yy in enumerate(
        range(
            y,
            y+h,
            pitch
        )
    ):


        offset=0

        if row%2:

            offset=70


        for xx in range(
            x+offset,
            x+w,
            pitch
        ):


            circle(
                img,
                xx,
                yy,
                18,
                235
            )


def routing_region(img,x,y,w,h):


    for yy in range(
        y,
        y+h,
        80
    ):

        line(
            img,
            x,
            yy,
            x+w,
            yy,
            80,
            5
        )


# ============================================================
# DEFECT
# ============================================================


def add_collapse_void(img,x,y):


    # partial collapse shadow


    cv2.ellipse(
        img,
        (
            int(x),
            int(y)
        ),
        (
            55,
            35
        ),
        20,
        0,
        360,
        45,
        -1
    )


    # void opening


    circle(
        img,
        x+10,
        y-5,
        22,
        8
    )


    # irregular residue


    for i in range(10):

        dx=rng.integers(
            -40,
            40
        )

        dy=rng.integers(
            -40,
            40
        )


        circle(
            img,
            x+dx,
            y+dy,
            4,
            120
        )


# ============================================================
# CREATE SCENE
# ============================================================


def create_scene():


    img=np.full(
        (
            SIZE,
            SIZE
        ),
        18,
        dtype=np.uint8
    )


    # Region 1

    vertical_region(
        img,
        300,
        300,
        2800,
        3000
    )


    # Region 2

    capacitor_region(
        img,
        3500,
        300,
        3000,
        3000
    )


    # Region 3

    routing_region(
        img,
        300,
        3800,
        8500,
        1200
    )


    # Region 4

    capacitor_region(
        img,
        700,
        6000,
        3000,
        3000
    )


    # Region 5

    vertical_region(
        img,
        5000,
        6000,
        3500,
        3000
    )


    add_collapse_void(
        img,
        DEFECT_X,
        DEFECT_Y
    )


    return img


# ============================================================
# SEM PHYSICS
# ============================================================


def reference_sem(img):


    out=cv2.GaussianBlur(
        img,
        (3,3),
        0.3
    ).astype(float)


    out+=rng.normal(
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


    out=cv2.GaussianBlur(
        img,
        (5,5),
        1
    ).astype(float)


    # focus gradient


    gradient=np.linspace(
        -12,
        12,
        img.shape[1]
    )


    out+=gradient


    # charging field


    charging=cv2.GaussianBlur(
        rng.normal(
            0,
            20,
            img.shape
        ),
        (151,151),
        0
    )


    out+=charging


    # row jitter


    out+=(
        rng.normal(
            0,
            2,
            img.shape[0]
        )[:,None]
    )


    # hot pixels


    mask=rng.random(
        img.shape
    )<0.003


    out[mask]+=50


    # poisson noise


    out+=rng.poisson(
        4,
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


    print("="*60)
    print("DRIFT-SENSE DRAM-50")
    print("="*60)


    OUT.mkdir(
        parents=True,
        exist_ok=True
    )


    scene=create_scene()


    ref_x=DEFECT_X-500

    ref_y=DEFECT_Y-500


    reference=scene[
        ref_y:
        ref_y+REF_SIZE,

        ref_x:
        ref_x+REF_SIZE
    ]


    search=cv2.resize(
        scene,
        (1000,1000),
        interpolation=cv2.INTER_AREA
    )


    reference=reference_sem(reference)

    search=search_sem(search)


    cv2.imwrite(
        str(
            OUT/"reference_100x.png"
        ),
        reference
    )


    cv2.imwrite(
        str(
            OUT/"search_10x.png"
        ),
        search
    )


    gt={

        "pair":
        "dram_50",

        "architecture":
        "advanced_multi_region_dram",

        "defect":
        "partial_capacitor_collapse_with_void",

        "defect_position":
        [
            DEFECT_X,
            DEFECT_Y
        ],

        "reference_origin":
        [
            ref_x,
            ref_y
        ],

        "scale_ratio":
        10

    }


    with open(
        OUT/"ground_truth.json",
        "w"
    ) as f:

        json.dump(
            gt,
            f,
            indent=4
        )


    print("DRAM-50 COMPLETE")
    print(OUT)


if __name__=="__main__":

    main()
