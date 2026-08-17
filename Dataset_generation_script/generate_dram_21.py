import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-21
# Multi Die DRAM + Dual Via Defect Localization
# ============================================================


SEED = 20260831

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_21"
)


SCENE_SIZE = 10000

REFERENCE_SIZE = 1000



# ============================================================
# DIE PARAMETERS
# ============================================================


DIE_SIZE = 2200

DIE_GAP = 300

MARGIN = 250


# Two identical black via defects

DEFECTS = [

    # reference target

    (1,2),

    # distractor

    (3,1)

]

TARGET = (1,2)



# ============================================================
# DRAW HELPERS
# ============================================================


def rect(img,x1,y1,x2,y2,val):

    cv2.rectangle(
        img,
        (int(x1),int(y1)),
        (int(x2),int(y2)),
        int(val),
        -1
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
# DRAM CELL ARRAY
# ============================================================


def generate_dram_cells(
        img,
        x,
        y,
        size
):


    pitch=55


    for row,yy in enumerate(
        range(
            y,
            y+size,
            pitch
        )
    ):


        for xx in range(
            x,
            x+size,
            pitch
        ):


            radius = (
                7+
                rng.normal(
                    0,
                    0.8
                )
            )


            circle(
                img,
                xx,
                yy,
                radius,
                210
            )


            # contact highlight


            circle(
                img,
                xx-2,
                yy-2,
                2,
                245
            )



# ============================================================
# VIA DEFECT
# ============================================================


def create_via_defect(
        img,
        x,
        y
):


    # dark isolation opening

    rect(
        img,
        x-55,
        y-55,
        x+55,
        y+55,
        30
    )


    # black central via


    circle(
        img,
        x,
        y,
        30,
        15
    )


    # small metal rim


    cv2.circle(
        img,
        (
            int(x),
            int(y)
        ),
        38,
        80,
        4
    )



# ============================================================
# CREATE DIE
# ============================================================


def create_die(
        img,
        x,
        y,
        defect=False
):


    # die background


    rect(
        img,
        x,
        y,
        x+DIE_SIZE,
        y+DIE_SIZE,
        20
    )


    # die boundary


    cv2.rectangle(
        img,
        (
            x+50,
            y+50
        ),
        (
            x+DIE_SIZE-50,
            y+DIE_SIZE-50
        ),
        55,
        8
    )


    generate_dram_cells(
        img,
        x+180,
        y+180,
        DIE_SIZE-360
    )



    if defect:

        create_via_defect(
            img,
            x+700,
            y+700
        )



# ============================================================
# SEARCH FIELD
# ============================================================


def generate_scene():


    img=np.full(
        (
            SCENE_SIZE,
            SCENE_SIZE
        ),
        18,
        dtype=np.uint8
    )


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


            create_die(
                img,
                x,
                y,
                (r,c) in DEFECTS
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
    )


    out=out.astype(float)


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
    )


    out=out.astype(float)


    # SEM dose

    out += rng.poisson(
        3,
        img.shape
    )


    # horizontal charging

    out += np.linspace(
        -10,
        10,
        img.shape[1]
    )


    # scan variation

    out += (
        rng.normal(
            0,
            2,
            img.shape[0]
        )[:,None]
    )


    # detector grain

    out += rng.normal(
        0,
        2,
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


    print("="*70)

    print(
        "DRIFT-SENSE DRAM-21 GENERATOR"
    )

    print("="*70)



    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )



    print("[1] Creating DRAM die field")


    scene=generate_scene()



    # target reference position


    ref_x=(
        MARGIN+
        TARGET[1]*(DIE_SIZE+DIE_GAP)
        +350
    )


    ref_y=(
        MARGIN+
        TARGET[0]*(DIE_SIZE+DIE_GAP)
        +350
    )



    print("[2] Extracting reference")


    reference=scene[
        ref_y:
        ref_y+REFERENCE_SIZE,

        ref_x:
        ref_x+REFERENCE_SIZE
    ]



    print("[3] Creating search image")


    search=cv2.resize(
        scene,
        (
            1000,
            1000
        ),
        interpolation=cv2.INTER_AREA
    )



    reference=reference_sem(
        reference
    )


    search=search_sem(
        search
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
        "dram_21",

        "architecture":
        "multi_die_dram_contact_array_with_two_via_anomalies",

        "target_defect":
        [
            TARGET[0],
            TARGET[1]
        ],

        "reference_origin_nm":
        [
            ref_x,
            ref_y
        ],

        "search_bbox_px":
        [
            ref_x/10,
            ref_y/10,
            100,
            100
        ],

        "scale_ratio":
        10

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
    print("DRAM-21 COMPLETE")
    print(OUTPUT)



if __name__=="__main__":

    main()