import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-26
# Natural DRAM Die Corner Localization
#
# No artificial defect
# Localization using:
#   - die boundary
#   - scribe lines
#   - neighbouring die corner
#   - DRAM density variation
# ============================================================


SEED = 20260904

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_26"
)


SCENE_SIZE = 10000

REFERENCE_SIZE = 1000



# ============================================================
# WAFER PARAMETERS
# ============================================================


DIE_SIZE = 2200

DIE_GAP = 300

MARGIN = 250



# reference taken from this die corner

TARGET_DIE = (1,1)



# ============================================================
# DRAW HELPERS
# ============================================================


def line(img,x1,y1,x2,y2,val,width):

    cv2.line(
        img,
        (int(x1),int(y1)),
        (int(x2),int(y2)),
        int(val),
        int(width),
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
# REALISTIC DRAM DIE PATTERNS
# ============================================================


def dense_dram_array(
        img,
        x,
        y,
        size,
        pitch
):


    for row,yy in enumerate(
        range(
            y,
            y+size,
            pitch
        )
    ):


        offset=0

        if row%2:

            offset=pitch//2



        for xx in range(
            x+offset,
            x+size,
            pitch
        ):

            circle(
                img,
                xx,
                yy,
                7,
                215
            )


            circle(
                img,
                xx,
                yy,
                2,
                245
            )





def wordline_dram(
        img,
        x,
        y,
        size
):


    pitch=55


    for yy in range(
        y,
        y+size,
        pitch
    ):

        line(
            img,
            x,
            yy,
            x+size,
            yy,
            95,
            4
        )


    for xx in range(
        x+20,
        x+size,
        70
    ):

        for yy in range(
            y+30,
            y+size,
            100
        ):

            circle(
                img,
                xx,
                yy,
                8,
                220
            )





def mixed_dram(
        img,
        x,
        y,
        size
):

    half=size//2


    dense_dram_array(
        img,
        x,
        y,
        half,
        55
    )


    wordline_dram(
        img,
        x+half,
        y,
        half
    )





# ============================================================
# CREATE SINGLE DIE
# ============================================================


def create_die(
        img,
        x,
        y,
        style
):


    # substrate

    cv2.rectangle(
        img,
        (
            x,
            y
        ),
        (
            x+DIE_SIZE,
            y+DIE_SIZE
        ),
        20,
        -1
    )


    # die border

    cv2.rectangle(
        img,
        (
            x+80,
            y+80
        ),
        (
            x+DIE_SIZE-80,
            y+DIE_SIZE-80
        ),
        55,
        8
    )



    ax=x+180
    ay=y+180

    size=DIE_SIZE-360



    if style==0:

        dense_dram_array(
            img,
            ax,
            ay,
            size,
            55
        )


    elif style==1:

        dense_dram_array(
            img,
            ax,
            ay,
            size,
            70
        )


    elif style==2:

        wordline_dram(
            img,
            ax,
            ay,
            size
        )


    else:

        mixed_dram(
            img,
            ax,
            ay,
            size
        )



# ============================================================
# CREATE WAFER SEARCH IMAGE
# ============================================================


def create_scene():


    img=np.full(
        (
            SCENE_SIZE,
            SCENE_SIZE
        ),
        15,
        dtype=np.uint8
    )


    patterns=[

        0,1,2,3,
        1,2,3,0,
        2,3,0,1,
        3,0,1,2

    ]


    k=0


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
                patterns[k]
            )


            k+=1



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
        1
    )


    out=out.astype(float)


    # lower SEM dose

    out+=rng.poisson(
        3,
        img.shape
    )


    # scan variation

    out+=(
        rng.normal(
            0,
            1.5,
            img.shape[0]
        )[:,None]
    )


    # illumination drift

    out+=np.linspace(
        -8,
        8,
        img.shape[1]
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
        "DRIFT-SENSE DRAM-26 GENERATOR"
    )

    print("="*70)



    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )



    print("[1] Creating DRAM wafer")


    scene=create_scene()



    # reference contains two die corners

    ref_x=(
        MARGIN+
        TARGET_DIE[1]*(DIE_SIZE+DIE_GAP)
        +1500
    )


    ref_y=(
        MARGIN+
        TARGET_DIE[0]*(DIE_SIZE+DIE_GAP)
        +1500
    )



    print("[2] Extracting corner reference")


    reference=scene[
        ref_y:
        ref_y+REFERENCE_SIZE,

        ref_x:
        ref_x+REFERENCE_SIZE
    ]



    print("[3] Creating search SEM")


    search=cv2.resize(
        scene,
        (
            1000,
            1000
        ),
        interpolation=cv2.INTER_AREA
    )



    reference=reference_sem(reference)

    search=search_sem(search)



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
        "dram_26",

        "architecture":
        "natural_multi_die_dram_corner_matching",

        "target_die":
        TARGET_DIE,

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
    print("DRAM-26 COMPLETE")
    print(OUTPUT)



if __name__=="__main__":

    main()