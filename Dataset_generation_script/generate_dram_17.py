import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-17
# Diagonal Capacitor DRAM + Local Top Left Structure
# ============================================================


SEED = 20260827

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_17"
)


SCENE_SIZE = 10000

REFERENCE_SIZE = 1000



# ============================================================
# DRAM PARAMETERS
# ============================================================


PITCH = 150


# special structure location
# TOP LEFT REGION

FEATURE_X = 1800
FEATURE_Y = 1800



# ============================================================
# DRAW HELPERS
# ============================================================


def line(
        img,
        x1,
        y1,
        x2,
        y2,
        value,
        width
):

    cv2.line(
        img,
        (
            int(x1),
            int(y1)
        ),
        (
            int(x2),
            int(y2)
        ),
        int(value),
        int(width),
        lineType=cv2.LINE_AA
    )




def ellipse(
        img,
        x,
        y,
        rx,
        ry,
        angle,
        value
):

    cv2.ellipse(
        img,
        (
            int(x),
            int(y)
        ),
        (
            int(rx),
            int(ry)
        ),
        angle,
        0,
        360,
        int(value),
        -1,
        lineType=cv2.LINE_AA
    )



def rect(
        img,
        x1,
        y1,
        x2,
        y2,
        value
):

    cv2.rectangle(
        img,
        (
            int(x1),
            int(y1)
        ),
        (
            int(x2),
            int(y2)
        ),
        int(value),
        -1
    )



# ============================================================
# DIAGONAL DRAM ARRAY
# ============================================================


def generate_diagonal_array(img):


    margin = 500


    rows = 0


    for y in range(
        margin,
        SCENE_SIZE-margin,
        PITCH
    ):


        offset = 0


        if rows % 2:

            offset = PITCH//2



        for x in range(
            margin+offset,
            SCENE_SIZE-margin,
            PITCH
        ):


            # curved/slanted capacitor


            angle = -25


            ellipse(
                img,
                x,
                y,
                28,
                65,
                angle,
                185
            )


            # contact

            ellipse(
                img,
                x+8,
                y-28,
                12,
                12,
                0,
                230
            )



            # diagonal connection


            line(
                img,
                x-45,
                y+80,
                x+30,
                y-80,
                70,
                5
            )


        rows += 1



# ============================================================
# UNIQUE STRUCTURE
# ============================================================


def generate_feature(img):


    cx = FEATURE_X
    cy = FEATURE_Y



    # dark isolation region

    rect(
        img,
        cx-180,
        cy-150,
        cx+180,
        cy+150,
        35
    )



    # central metal bridge

    rect(
        img,
        cx-70,
        cy-70,
        cx+120,
        cy+70,
        210
    )



    # vertical connection


    rect(
        img,
        cx-25,
        cy-220,
        cx+25,
        cy-70,
        190
    )



    # missing corner

    rect(
        img,
        cx+40,
        cy+20,
        cx+130,
        cy+100,
        35
    )



    # small defect contact

    ellipse(
        img,
        cx-100,
        cy-100,
        20,
        20,
        0,
        230
    )



# ============================================================
# CREATE PHYSICAL SCENE
# ============================================================


def create_scene():


    img=np.full(
        (
            SCENE_SIZE,
            SCENE_SIZE
        ),
        25,
        dtype=np.uint8
    )


    generate_diagonal_array(
        img
    )


    generate_feature(
        img
    )


    return img



# ============================================================
# SEM IMAGING
# ============================================================


def reference_sem(img):


    out=cv2.GaussianBlur(
        img,
        (3,3),
        0.35
    )


    out=out.astype(float)


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


    out=cv2.GaussianBlur(
        img,
        (5,5),
        0.9
    )


    out=out.astype(float)



    # SEM dose noise

    out += rng.poisson(
        3,
        img.shape
    )



    # charging

    out += np.linspace(
        -8,
        8,
        img.shape[1]
    )



    # scan variation

    out += (
        rng.normal(
            0,
            1.5,
            img.shape[0]
        )
        [:,None]
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
    print("DRIFT-SENSE DRAM-17 GENERATOR")
    print("="*70)



    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )


    print("[1] Generating diagonal DRAM structure...")


    scene=create_scene()



    print("[2] Extracting reference...")


    # crop around top-left structure

    ref_x=FEATURE_X-500

    ref_y=FEATURE_Y-500



    reference=scene[
        ref_y:
        ref_y+REFERENCE_SIZE,

        ref_x:
        ref_x+REFERENCE_SIZE
    ]



    print("[3] Creating search image...")


    search=cv2.resize(
        scene,
        (
            1000,
            1000
        ),
        interpolation=cv2.INTER_AREA
    )



    print("[4] Applying SEM physics...")


    reference=reference_sem(
        reference
    )


    search=search_sem(
        search
    )



    print("[5] Saving...")


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
        "dram_17",

        "architecture":
        "diagonal_capacitor_dram_with_local_structure",

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
    print("DRAM-17 COMPLETE")
    print(OUTPUT)



if __name__=="__main__":

    main()