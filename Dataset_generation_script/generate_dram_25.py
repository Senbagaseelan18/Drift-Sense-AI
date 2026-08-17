import cv2
import json
import numpy as np
from pathlib import Path



# ============================================================
# DRIFT-SENSE DRAM-25
# Single Die DRAM Missing Hole Localization
# ============================================================


SEED = 20260903

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_25"
)


SCENE_SIZE = 10000

REFERENCE_SIZE = 1000



# missing hole location

DEFECT_X = 5200

DEFECT_Y = 4600



PITCH = 180



# ============================================================
# DRAW HELPERS
# ============================================================


def circle(img,x,y,r,value):

    cv2.circle(
        img,
        (
            int(x),
            int(y)
        ),
        int(r),
        int(value),
        -1,
        lineType=cv2.LINE_AA
    )



# ============================================================
# DRAM ARRAY
# ============================================================


def generate_dram_array(img):


    margin=600


    row=0


    for y in range(
        margin,
        SCENE_SIZE-margin,
        PITCH
    ):


        offset=0


        if row%2:

            offset=PITCH//2



        for x in range(
            margin+offset,
            SCENE_SIZE-margin,
            PITCH
        ):


            # normal capacitor hole


            circle(
                img,
                x,
                y,
                35,
                210
            )


            # inner contrast


            circle(
                img,
                x,
                y,
                12,
                245
            )


        row+=1



# ============================================================
# MISSING CONTACT DEFECT
# ============================================================


def create_missing_hole(img):


    # erase contact


    cv2.circle(
        img,
        (
            DEFECT_X,
            DEFECT_Y
        ),
        42,
        35,
        -1
    )


    # small collapse shadow


    cv2.circle(
        img,
        (
            DEFECT_X+8,
            DEFECT_Y+5
        ),
        18,
        20,
        -1
    )



# ============================================================
# CREATE PHYSICAL DIE
# ============================================================


def create_scene():


    img=np.full(
        (
            SCENE_SIZE,
            SCENE_SIZE
        ),
        18,
        dtype=np.uint8
    )


    generate_dram_array(
        img
    )


    create_missing_hole(
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
        0.25
    )


    out=out.astype(float)


    out+=rng.normal(
        0,
        0.8,
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
        0.8
    )


    out=out.astype(float)


    # mild SEM dose noise


    out+=rng.poisson(
        2,
        img.shape
    )


    # small illumination drift


    out+=np.linspace(
        -5,
        5,
        img.shape[1]
    )


    # detector variation


    out+=rng.normal(
        0,
        1.5,
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
        "DRIFT-SENSE DRAM-25 GENERATOR"
    )

    print("="*70)



    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )



    print("[1] Creating single DRAM die")


    scene=create_scene()



    print("[2] Extracting reference")


    ref_x=DEFECT_X-500

    ref_y=DEFECT_Y-500



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
        "dram_25",

        "architecture":
        "single_die_contact_array_missing_hole",

        "defect_position_nm":
        [
            DEFECT_X,
            DEFECT_Y
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
    print("DRAM-25 COMPLETE")
    print(OUTPUT)



if __name__=="__main__":

    main()