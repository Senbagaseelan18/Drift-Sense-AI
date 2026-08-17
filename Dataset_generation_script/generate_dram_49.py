import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRAM-49
# Stacked Capacitor Ring DRAM
# Defect: Missing Ring Segment
# ============================================================


SEED = 20260819

rng = np.random.default_rng(SEED)


OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_49"


SIZE = 10000

REF_SIZE = 1000


DEFECT_X = 5200

DEFECT_Y = 4800



# ============================================================
# DRAW HELPERS
# ============================================================


def ring(img,x,y,r,value,width):

    cv2.circle(
        img,
        (int(x),int(y)),
        int(r),
        int(value),
        int(width),
        lineType=cv2.LINE_AA
    )



def line(img,x1,y1,x2,y2,v,w):

    cv2.line(
        img,
        (int(x1),int(y1)),
        (int(x2),int(y2)),
        int(v),
        int(w),
        lineType=cv2.LINE_AA
    )


# ============================================================
# CAPACITOR ARRAY
# ============================================================


def create_ring_array(img):

    pitch=180


    for row,y in enumerate(
        range(
            500,
            SIZE-500,
            pitch
        )
    ):


        offset=0

        if row%2:

            offset=90


        for x in range(
            500+offset,
            SIZE-500,
            pitch
        ):


            # outer capacitor ring

            ring(
                img,
                x,
                y,
                35,
                220,
                8
            )


            # inner node

            ring(
                img,
                x,
                y,
                12,
                245,
                3
            )


# ============================================================
# ROUTING CHANNELS
# ============================================================


def create_routing(img):


    for y in range(
        900,
        SIZE,
        1000
    ):

        line(
            img,
            300,
            y,
            SIZE-300,
            y,
            70,
            12
        )


    for x in range(
        1000,
        SIZE,
        1200
    ):

        line(
            img,
            x,
            300,
            x,
            SIZE-300,
            80,
            8
        )


# ============================================================
# DEFECT
# ============================================================


def add_missing_segment(img,x,y):


    # erase part of ring


    cv2.rectangle(
        img,
        (
            x-45,
            y-45
        ),
        (
            x+15,
            y+15
        ),
        20,
        -1
    )


    # small rough edge


    for i in range(5):

        dx=rng.integers(
            -20,
            20
        )

        dy=rng.integers(
            -20,
            20
        )


        cv2.circle(
            img,
            (
                x+dx,
                y+dy
            ),
            5,
            35,
            -1
        )


# ============================================================
# SCENE
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


    create_ring_array(img)

    create_routing(img)

    add_missing_segment(
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
        .25
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


    out=img.astype(float)


    # astigmatism style blur


    kx=cv2.GaussianBlur(
        out,
        (7,3),
        1
    )


    ky=cv2.GaussianBlur(
        out,
        (3,7),
        1
    )


    out=(kx+ky)/2


    # charging spots


    charge=cv2.GaussianBlur(
        rng.normal(
            0,
            15,
            img.shape
        ),
        (151,151),
        0
    )


    out+=charge


    # multiplicative noise


    out*=rng.normal(
        1,
        0.02,
        img.shape
    )


    # scan drift


    out+=(
        np.sin(
            np.arange(
                img.shape[0]
            )/20
        )[:,None]
        *2
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


    print("DRIFT-SENSE DRAM-49")


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
        "dram_49",

        "architecture":
        "stacked_capacitor_ring_dram",

        "defect":
        "missing_ring_segment",

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


    print("DRAM-49 COMPLETE")
    print(OUT)


if __name__=="__main__":

    main()
