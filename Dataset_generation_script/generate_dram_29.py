import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRAM-29
# Single Die Multi-Region Line Architecture
# ============================================================


SEED = 20260907

rng = np.random.default_rng(SEED)


OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_29"


SIZE = 10000

REF_SIZE = 1000



# ============================================================
# DRAW HELPERS
# ============================================================


def line(img,x1,y1,x2,y2,v,w):

    cv2.line(
        img,
        (int(x1),int(y1)),
        (int(x2),int(y2)),
        int(v),
        int(w),
        lineType=cv2.LINE_AA
    )



def rect(img,x1,y1,x2,y2,v):

    cv2.rectangle(
        img,
        (x1,y1),
        (x2,y2),
        int(v),
        -1
    )



# ============================================================
# STRUCTURE 1
# VERTICAL BITLINE REGION
# ============================================================


def vertical_region(
        img,
        x,
        y,
        w,
        h,
        pitch
):


    for xx in range(
        x,
        x+w,
        pitch
    ):

        line(
            img,
            xx,
            y,
            xx,
            y+h,
            120,
            5
        )


        # small landing pads

        for yy in range(
            y+50,
            y+h,
            130
        ):

            cv2.circle(
                img,
                (
                    xx,
                    yy
                ),
                10,
                220,
                -1
            )





# ============================================================
# STRUCTURE 2
# HORIZONTAL WORDLINE REGION
# ============================================================


def horizontal_region(
        img,
        x,
        y,
        w,
        h,
        pitch
):


    for yy in range(
        y,
        y+h,
        pitch
    ):

        line(
            img,
            x,
            yy,
            x+w,
            yy,
            100,
            5
        )


    for xx in range(
        x+40,
        x+w,
        100
    ):

        for yy in range(
            y+20,
            y+h,
            90
        ):

            cv2.circle(
                img,
                (
                    xx,
                    yy
                ),
                8,
                230,
                -1
            )





# ============================================================
# CREATE DIE
# ============================================================


def create_die():


    img=np.full(
        (
            SIZE,
            SIZE
        ),
        18,
        dtype=np.uint8
    )


    # outer die boundary

    cv2.rectangle(
        img,
        (200,200),
        (9800,9800),
        45,
        8
    )



    # top left dense region

    vertical_region(
        img,
        500,
        500,
        3500,
        3800,
        55
    )



    # middle routing lines

    horizontal_region(
        img,
        500,
        4700,
        8500,
        1700,
        75
    )



    # bottom region

    vertical_region(
        img,
        5200,
        6800,
        3500,
        2200,
        80
    )



    # isolation separators


    line(
        img,
        4500,
        300,
        4500,
        9500,
        60,
        20
    )


    line(
        img,
        300,
        6500,
        9500,
        6500,
        60,
        20
    )


    return img



# ============================================================
# SEM EFFECT
# ============================================================


def reference_sem(img):

    out=cv2.GaussianBlur(
        img,
        (3,3),
        0.4
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


    # detector noise

    out+=rng.normal(
        0,
        3,
        img.shape
    )


    # horizontal scan variation

    out+=(
        np.sin(
            np.arange(
                img.shape[0]
            )[:,None]/15
        )*2
    )


    # dose noise

    out+=rng.poisson(
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


    print(
        "Generating DRAM-29"
    )


    die=create_die()



    # reference from transition region

    ref_x=4300

    ref_y=4300



    reference=die[
        ref_y:
        ref_y+REF_SIZE,

        ref_x:
        ref_x+REF_SIZE
    ]



    search=cv2.resize(
        die,
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

        "pair":"dram_29",

        "architecture":
        "single_die_multi_region_line_structure",

        "reference_contains":
        "vertical_horizontal_transition",

        "origin_nm":
        [
            ref_x,
            ref_y
        ],

        "scale":
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