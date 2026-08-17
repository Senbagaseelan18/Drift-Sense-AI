import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-20
# Dual Layer DRAM Array + Routing Window Structure
# ============================================================


SEED = 20260830

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_20"
)


SCENE_SIZE = 10000

REFERENCE_SIZE = 1000



# ============================================================
# DIE PARAMETERS
# ============================================================


DIE_SIZE = 2200

DIE_GAP = 320

MARGIN = 250



# target die

TARGET_ROW = 1
TARGET_COL = 1



# ============================================================
# DRAW HELPERS
# ============================================================


def line(
        img,
        x1,
        y1,
        x2,
        y2,
        val,
        width
):

    cv2.line(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        int(val),
        int(width),
        lineType=cv2.LINE_AA
    )



def circle(
        img,
        x,
        y,
        r,
        val
):

    cv2.circle(
        img,
        (int(x), int(y)),
        int(r),
        int(val),
        -1,
        lineType=cv2.LINE_AA
    )



# ============================================================
# DRAM PATTERNS
# ============================================================


def hex_contact_array(
        img,
        x,
        y,
        size
):

    pitch = 75

    row = 0


    for yy in range(
        y,
        y+size,
        pitch
    ):

        offset = 0

        if row % 2:
            offset = pitch//2


        for xx in range(
            x+offset,
            x+size,
            pitch
        ):

            circle(
                img,
                xx,
                yy,
                12,
                220
            )


            # inner SEM highlight

            circle(
                img,
                xx-2,
                yy-2,
                4,
                245
            )


        row += 1



def double_wordline_array(
        img,
        x,
        y,
        size
):


    pitch = 65


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
            90,
            5
        )


        line(
            img,
            x,
            yy+18,
            x+size,
            yy+18,
            45,
            3
        )


        for xx in range(
            x+30,
            x+size,
            90
        ):

            circle(
                img,
                xx,
                yy+10,
                8,
                220
            )




def fin_channel_array(
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
            120,
            5
        )


    for yy in range(
        y+30,
        y+size,
        80
    ):

        for xx in range(
            x+20,
            x+size,
            100
        ):

            circle(
                img,
                xx,
                yy,
                6,
                220
            )




def mixed_density_array(
        img,
        x,
        y,
        size
):


    half=size//2


    # dense half

    hex_contact_array(
        img,
        x,
        y,
        half
    )


    # open half

    for xx in range(
        x+half,
        x+size,
        90
    ):

        line(
            img,
            xx,
            y,
            xx,
            y+size,
            100,
            5
        )


    for yy in range(
        y+40,
        y+size,
        100
    ):

        circle(
            img,
            x+half+40,
            yy,
            10,
            210
        )



# ============================================================
# SMALL PROCESS FEATURE
# ============================================================


def add_feature(
        img,
        x,
        y
):


    # remove one normal contact area


    cv2.rectangle(
        img,
        (
            x-70,
            y-70
        ),
        (
            x+70,
            y+70
        ),
        35,
        -1
    )


    # asymmetric bridge


    cv2.rectangle(
        img,
        (
            x-35,
            y-80
        ),
        (
            x+35,
            y+60
        ),
        215,
        -1
    )


    circle(
        img,
        x+75,
        y+30,
        12,
        230
    )



# ============================================================
# DIE CREATION
# ============================================================


def create_die(
        img,
        x,
        y,
        style,
        feature=False
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
        22,
        -1
    )


    # die border


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

        hex_contact_array(
            img,
            ax,
            ay,
            size
        )


    elif style==1:

        double_wordline_array(
            img,
            ax,
            ay,
            size
        )


    elif style==2:

        fin_channel_array(
            img,
            ax,
            ay,
            size
        )


    else:

        mixed_density_array(
            img,
            ax,
            ay,
            size
        )



    if feature:

        add_feature(
            img,
            x+720,
            y+720
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
        2,0,3,1,
        3,2,1,0,
        1,3,0,2

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
                (
                    r==TARGET_ROW
                    and
                    c==TARGET_COL
                )
            )


            k+=1



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



    # SEM dose

    out+=rng.poisson(
        3,
        img.shape
    )


    # charging

    out+=np.linspace(
        -9,
        9,
        img.shape[1]
    )



    # raster noise

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

    print(
        "DRIFT-SENSE DRAM-20 GENERATOR"
    )

    print("="*70)



    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )


    print("[1] Generating multi architecture DRAM field")


    scene=create_scene()



    # target reference


    ref_x=(
        MARGIN+
        TARGET_COL*(DIE_SIZE+DIE_GAP)
        +450
    )


    ref_y=(
        MARGIN+
        TARGET_ROW*(DIE_SIZE+DIE_GAP)
        +450
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



    print("[4] SEM simulation")


    reference=reference_sem(
        reference
    )


    search=search_sem(
        search
    )



    print("[5] Saving")



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
        "dram_20",

        "architecture":
        "dual_layer_dram_array_with_routing_windows",

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
    print("DRAM-20 COMPLETE")
    print(OUTPUT)



if __name__=="__main__":

    main()