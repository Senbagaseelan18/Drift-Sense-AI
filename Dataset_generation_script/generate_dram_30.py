import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRAM-30
# FULL ACTIVE DIE SEM STRUCTURE
# No empty regions
# Dense multi-layer DRAM architecture
# ============================================================


SEED = 20260908

rng=np.random.default_rng(SEED)


OUT=Path(__file__).resolve().parents[1]/"results" / "generated_dataset_images" / "dram_30"


SIZE=10000

REF_SIZE=1000



# ============================================================
# DRAW FUNCTIONS
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
# FULL DRAM PATTERN
# ============================================================


def vertical_bitlines(img):


    for x in range(
        200,
        SIZE-200,
        45
    ):

        line(
            img,
            x,
            200,
            x,
            SIZE-200,
            100,
            3
        )


        for y in range(
            250,
            SIZE-200,
            90
        ):

            circle(
                img,
                x,
                y,
                9,
                215
            )





def horizontal_wordlines(img):


    for y in range(
        300,
        SIZE-200,
        70
    ):

        line(
            img,
            150,
            y,
            SIZE-150,
            y,
            80,
            4
        )





def contact_mesh(img):


    for y in range(
        400,
        SIZE-300,
        130
    ):


        offset=0

        if (y//130)%2:

            offset=60


        for x in range(
            300+offset,
            SIZE-300,
            120
        ):


            circle(
                img,
                x,
                y,
                16,
                230
            )


            circle(
                img,
                x,
                y,
                5,
                255
            )





def isolation_stripes(img):


    for x in range(
        1000,
        SIZE,
        2500
    ):

        line(
            img,
            x,
            100,
            x,
            SIZE-100,
            50,
            12
        )



    for y in range(
        2000,
        SIZE,
        2500
    ):

        line(
            img,
            100,
            y,
            SIZE-100,
            y,
            55,
            12
        )





# ============================================================
# CREATE DIE
# ============================================================


def create_scene():


    img=np.full(
        (
            SIZE,
            SIZE
        ),
        20,
        dtype=np.uint8
    )


    # base dense structures

    vertical_bitlines(img)

    horizontal_wordlines(img)

    contact_mesh(img)

    isolation_stripes(img)



    return img



# ============================================================
# SEM PHYSICS
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
        0.9
    )


    out=out.astype(float)



    # vertical SEM charging

    charge=np.sin(
        np.arange(
            img.shape[1]
        )/30
    )


    out+=charge*3



    # detector variation

    out+=rng.normal(
        0,
        2.5,
        img.shape
    )



    # dose noise

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


    print(
        "DRIFT-SENSE DRAM-30"
    )


    OUT.mkdir(
        parents=True,
        exist_ok=True
    )


    scene=create_scene()



    # reference from internal active area

    ref_x=4200

    ref_y=4300



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
        "dram_30",

        "architecture":
        "full_active_die_multilayer_dram",

        "reference_origin_nm":
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


    print("DONE")
    print(OUT)



if __name__=="__main__":

    main()