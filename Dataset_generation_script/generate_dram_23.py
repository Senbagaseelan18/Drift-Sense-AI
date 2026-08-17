import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-23
# Multi Architecture DRAM Dies + Two Local Structures
# ============================================================


SEED = 20260901

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_23"
)


SCENE_SIZE = 10000

REFERENCE_SIZE = 1000


# ============================================================
# DIE PARAMETERS
# ============================================================


DIE_SIZE = 2200

DIE_GAP = 320

MARGIN = 250



# Target and distractor dies

TARGET_DIE = (1,2)

DISTRACTOR_DIE = (3,1)



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
# DRAM DIE PATTERNS
# ============================================================


def dense_contact_die(
        img,
        x,
        y,
        size
):

    pitch=45


    for yy in range(
        y,
        y+size,
        pitch
    ):

        for xx in range(
            x,
            x+size,
            pitch
        ):

            circle(
                img,
                xx,
                yy,
                8,
                220
            )



def fin_die(
        img,
        x,
        y,
        size
):

    pitch=45


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


    for yy in range(
        y+30,
        y+size,
        90
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
                7,
                220
            )





def stripe_die(
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
            130,
            5
        )


    for yy in range(
        y+20,
        y+size,
        100
    ):

        for xx in range(
            x+30,
            x+size,
            90
        ):

            circle(
                img,
                xx,
                yy,
                8,
                210
            )





def sparse_die(
        img,
        x,
        y,
        size
):

    pitch=100

    row=0


    for yy in range(
        y,
        y+size,
        pitch
    ):

        offset=0

        if row%2:
            offset=50


        for xx in range(
            x+offset,
            x+size,
            pitch
        ):

            circle(
                img,
                xx,
                yy,
                14,
                220
            )

        row+=1





def mixed_die(
        img,
        x,
        y,
        size
):

    half=size//2


    fin_die(
        img,
        x,
        y,
        half
    )


    sparse_die(
        img,
        x+half,
        y,
        half
    )



# ============================================================
# UNIQUE STRUCTURES
# ============================================================


def add_target_structure(
        img,
        x,
        y
):


    # small black via


    cv2.circle(
        img,
        (
            int(x),
            int(y)
        ),
        35,
        20,
        -1
    )


    # metal rim


    cv2.circle(
        img,
        (
            int(x),
            int(y)
        ),
        45,
        90,
        4
    )


    # asymmetric bridge

    line(
        img,
        x-50,
        y+40,
        x+60,
        y-30,
        220,
        10
    )




def add_distractor_structure(
        img,
        x,
        y
):


    cv2.circle(
        img,
        (
            int(x),
            int(y)
        ),
        32,
        25,
        -1
    )


    cv2.circle(
        img,
        (
            int(x),
            int(y)
        ),
        42,
        100,
        4
    )


    line(
        img,
        x-60,
        y,
        x+60,
        y,
        200,
        8
    )



# ============================================================
# CREATE DIE
# ============================================================


def create_die(
        img,
        x,
        y,
        pattern,
        target=False,
        distractor=False
):


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
        10
    )


    ax=x+220
    ay=y+220

    size=DIE_SIZE-440



    if pattern==0:

        dense_contact_die(
            img,
            ax,
            ay,
            size
        )


    elif pattern==1:

        fin_die(
            img,
            ax,
            ay,
            size
        )


    elif pattern==2:

        stripe_die(
            img,
            ax,
            ay,
            size
        )


    elif pattern==3:

        sparse_die(
            img,
            ax,
            ay,
            size
        )


    else:

        mixed_die(
            img,
            ax,
            ay,
            size
        )



    if target:

        add_target_structure(
            img,
            x+720,
            y+720
        )



    if distractor:

        add_distractor_structure(
            img,
            x+700,
            y+700
        )



# ============================================================
# SEARCH SCENE
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

        4,0,3,1,

        2,4,1,0,

        3,1,4,2

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
                patterns[k],
                (r,c)==TARGET_DIE,
                (r,c)==DISTRACTOR_DIE
            )


            k+=1



    return img



# ============================================================
# SEM EFFECT
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
        1
    )


    out=out.astype(float)


    out+=rng.poisson(
        3,
        img.shape
    )


    out+=np.linspace(
        -10,
        10,
        img.shape[1]
    )


    out+=(
        rng.normal(
            0,
            2,
            img.shape[0]
        )[:,None]
    )


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
    print("DRIFT-SENSE DRAM-23 GENERATOR")
    print("="*70)



    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )


    print("[1] Creating multi-pattern DRAM field")


    scene=create_scene()



    ref_x=(
        MARGIN+
        TARGET_DIE[1]*(DIE_SIZE+DIE_GAP)
        +350
    )


    ref_y=(
        MARGIN+
        TARGET_DIE[0]*(DIE_SIZE+DIE_GAP)
        +350
    )



    print("[2] Extracting reference")


    reference=scene[
        ref_y:
        ref_y+REFERENCE_SIZE,

        ref_x:
        ref_x+REFERENCE_SIZE
    ]



    print("[3] Creating search")


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
        "dram_23",

        "architecture":
        "multi_die_dram_with_two_local_structures",

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
            gt,
            f,
            indent=4
        )


    print()
    print("DRAM-23 COMPLETE")
    print(OUTPUT)



if __name__=="__main__":

    main()