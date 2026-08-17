import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-24
# Staircase Wordline + Bitline DRAM SEM Generator
# ============================================================


SEED = 20260902

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_24"
)


SCENE_SIZE = 10000

REFERENCE_SIZE = 1000



# ============================================================
# DIE PARAMETERS
# ============================================================


DIE_SIZE = 2200

GAP = 320

MARGIN = 250



TARGET_DIE = (2,1)

DISTRACTOR_DIE = (0,3)



# ============================================================
# DRAW FUNCTIONS
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
        (int(x1),int(y1)),
        (int(x2),int(y2)),
        int(value),
        int(width),
        lineType=cv2.LINE_AA
    )



def circle(
        img,
        x,
        y,
        radius,
        value
):

    cv2.circle(
        img,
        (int(x),int(y)),
        int(radius),
        int(value),
        -1,
        lineType=cv2.LINE_AA
    )



# ============================================================
# DRAM STRUCTURES
# ============================================================


def vertical_bitline_die(
        img,
        x,
        y,
        size
):

    pitch=45


    # vertical bitlines

    for xx in range(
        x,
        x+size,
        pitch
    ):

        line(
            img,
            xx,
            y,
            xx,
            y+size,
            110,
            5
        )


    # landing contacts

    for yy in range(
        y+30,
        y+size,
        75
    ):

        for xx in range(
            x+20,
            x+size,
            90
        ):

            circle(
                img,
                xx,
                yy,
                8,
                220
            )





def staircase_wordline_die(
        img,
        x,
        y,
        size
):


    pitch=60


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
            120,
            6
        )


        # repeated contacts

        for xx in range(
            x+40,
            x+size,
            100
        ):

            circle(
                img,
                xx,
                yy+15,
                9,
                220
            )





def honeycomb_die(
        img,
        x,
        y,
        size
):


    pitch=75

    row=0


    for yy in range(
        y,
        y+size,
        pitch
    ):


        offset=0


        if row%2:

            offset=38


        for xx in range(
            x+offset,
            x+size,
            pitch
        ):


            circle(
                img,
                xx,
                yy,
                13,
                220
            )


        row+=1





def transition_die(
        img,
        x,
        y,
        size
):


    half=size//2


    # routing area


    for yy in range(
        y,
        y+half,
        80
    ):

        line(
            img,
            x,
            yy,
            x+half,
            yy,
            100,
            5
        )



    # memory area

    honeycomb_die(
        img,
        x+half,
        y,
        half
    )



# ============================================================
# SMALL PROCESS FEATURES
# ============================================================


def add_target_feature(
        img,
        x,
        y
):


    # collapsed via opening

    cv2.rectangle(
        img,
        (
            x-55,
            y-55
        ),
        (
            x+55,
            y+55
        ),
        25,
        -1
    )


    circle(
        img,
        x,
        y,
        30,
        15
    )


    # asymmetric metal residue

    line(
        img,
        x-40,
        y+45,
        x+70,
        y-30,
        220,
        9
    )





def add_distractor_feature(
        img,
        x,
        y
):


    cv2.rectangle(
        img,
        (
            x-45,
            y-45
        ),
        (
            x+45,
            y+45
        ),
        35,
        -1
    )


    circle(
        img,
        x,
        y,
        25,
        20
    )


    line(
        img,
        x-60,
        y,
        x+60,
        y,
        210,
        8
    )



# ============================================================
# CREATE DIE
# ============================================================


def create_die(
        img,
        x,
        y,
        style,
        target=False,
        distractor=False
):


    # die background

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



    # boundary

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
        10
    )



    ax=x+220
    ay=y+220

    size=DIE_SIZE-440



    if style==0:

        vertical_bitline_die(
            img,
            ax,
            ay,
            size
        )


    elif style==1:

        staircase_wordline_die(
            img,
            ax,
            ay,
            size
        )


    elif style==2:

        honeycomb_die(
            img,
            ax,
            ay,
            size
        )


    else:

        transition_die(
            img,
            ax,
            ay,
            size
        )



    if target:

        add_target_feature(
            img,
            x+750,
            y+750
        )


    if distractor:

        add_distractor_feature(
            img,
            x+700,
            y+700
        )



# ============================================================
# SEARCH FIELD
# ============================================================


def create_scene():


    img=np.full(
        (
            SCENE_SIZE,
            SCENE_SIZE
        ),
        16,
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
                c*(DIE_SIZE+GAP)
            )


            y=(
                MARGIN+
                r*(DIE_SIZE+GAP)
            )



            create_die(
                img,
                x,
                y,
                patterns[k],
                (r,c)==TARGET_DIE,
                (r,c)==DISTRACTOR_DIE
            )


            k+=1



    return img



# ============================================================
# SEM IMAGE FORMATION
# ============================================================


def reference_sem(img):


    out=cv2.GaussianBlur(
        img,
        (3,3),
        0.35
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
        1.0
    )


    out=out.astype(float)


    # electron dose variation

    out+=rng.poisson(
        3,
        img.shape
    )


    # charging gradient

    out+=np.linspace(
        -10,
        10,
        img.shape[1]
    )


    # scan noise

    out+=(
        rng.normal(
            0,
            2,
            img.shape[0]
        )[:,None]
    )


    # detector variation

    out+=rng.normal(
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
    print("DRIFT-SENSE DRAM-24 GENERATOR")
    print("="*70)



    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )



    print("[1] Creating realistic DRAM field")


    scene=create_scene()



    ref_x=(
        MARGIN+
        TARGET_DIE[1]*(DIE_SIZE+GAP)
        +400
    )


    ref_y=(
        MARGIN+
        TARGET_DIE[0]*(DIE_SIZE+GAP)
        +400
    )



    print("[2] Extracting reference")


    reference=scene[
        ref_y:
        ref_y+REFERENCE_SIZE,

        ref_x:
        ref_x+REFERENCE_SIZE
    ]



    print("[3] Generating search SEM")


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



    ground_truth={

        "pair":
        "dram_24",

        "architecture":
        "staircase_wordline_bitline_dram",

        "target_die":
        TARGET_DIE,

        "distractor_die":
        DISTRACTOR_DIE,

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
            ground_truth,
            f,
            indent=4
        )



    print("DRAM-24 COMPLETE")
    print(OUTPUT)



if __name__=="__main__":

    main()