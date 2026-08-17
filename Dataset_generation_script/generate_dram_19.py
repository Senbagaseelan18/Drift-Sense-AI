import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-19
# Hierarchical DRAM Bank SEM Generator
# ============================================================


SEED = 20260829

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_19"
)


SCENE_SIZE = 10000

REFERENCE_SIZE = 1000



# ============================================================
# PARAMETERS
# ============================================================


BANK_SIZE = 2200

BANK_GAP = 350

MARGIN = 250



# unique feature

FEATURE_BANK_X = 2

FEATURE_BANK_Y = 1



# ============================================================
# DRAW FUNCTIONS
# ============================================================


def draw_line(img,x1,y1,x2,y2,val,width):

    cv2.line(
        img,
        (int(x1),int(y1)),
        (int(x2),int(y2)),
        int(val),
        int(width),
        lineType=cv2.LINE_AA
    )



def draw_dot(img,x,y,r,val):

    cv2.circle(
        img,
        (int(x),int(y)),
        int(r),
        int(val),
        -1,
        lineType=cv2.LINE_AA
    )



# ============================================================
# BANK PATTERNS
# ============================================================


def dense_bank(img,x,y,size):


    pitch=38


    for xx in range(
        x,
        x+size,
        pitch
    ):

        draw_line(
            img,
            xx,
            y,
            xx,
            y+size,
            80,
            4
        )


    for yy in range(
        y+20,
        y+size,
        55
    ):

        for xx in range(
            x+15,
            x+size,
            pitch
        ):

            draw_dot(
                img,
                xx,
                yy,
                7,
                215
            )





def stripe_bank(img,x,y,size):


    pitch=45


    for yy in range(
        y,
        y+size,
        pitch
    ):


        draw_line(
            img,
            x,
            yy,
            x+size,
            yy,
            140,
            5
        )


    for yy in range(
        y+20,
        y+size,
        80
    ):

        for xx in range(
            x+30,
            x+size,
            75
        ):

            draw_dot(
                img,
                xx,
                yy,
                8,
                220
            )





def checker_bank(img,x,y,size):


    pitch=70

    row=0


    for yy in range(
        y,
        y+size,
        pitch
    ):

        offset=0

        if row%2:
            offset=35


        for xx in range(
            x+offset,
            x+size,
            pitch
        ):

            draw_dot(
                img,
                xx,
                yy,
                11,
                220
            )


        row+=1




def routing_bank(img,x,y,size):


    # mixed routing

    for i in range(
        0,
        size,
        100
    ):

        draw_line(
            img,
            x,
            y+i,
            x+size,
            y+i,
            120,
            6
        )


    for i in range(
        0,
        size,
        130
    ):

        draw_line(
            img,
            x+i,
            y,
            x+i,
            y+size,
            75,
            3
        )



# ============================================================
# UNIQUE SMALL STRUCTURE
# ============================================================


def add_feature(img,x,y):


    # missing region

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


    # bright bridge

    cv2.rectangle(
        img,
        (
            x-20,
            y-60
        ),
        (
            x+25,
            y+60
        ),
        220,
        -1
    )


    # side contact

    draw_dot(
        img,
        x+65,
        y,
        12,
        230
    )



# ============================================================
# CREATE BANK
# ============================================================


def create_bank(img,x,y,kind,feature=False):


    # dark bank area


    cv2.rectangle(
        img,
        (
            x,
            y
        ),
        (
            x+BANK_SIZE,
            y+BANK_SIZE
        ),
        25,
        -1
    )


    # boundary

    cv2.rectangle(
        img,
        (
            x+40,
            y+40
        ),
        (
            x+BANK_SIZE-40,
            y+BANK_SIZE-40
        ),
        55,
        8
    )


    ax=x+220
    ay=y+220

    size=BANK_SIZE-440



    if kind==0:

        dense_bank(
            img,
            ax,
            ay,
            size
        )

    elif kind==1:

        stripe_bank(
            img,
            ax,
            ay,
            size
        )

    elif kind==2:

        checker_bank(
            img,
            ax,
            ay,
            size
        )

    else:

        routing_bank(
            img,
            ax,
            ay,
            size
        )



    if feature:

        add_feature(
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
        18,
        dtype=np.uint8
    )


    bank_types=[

        0,1,2,3,
        1,2,0,3,
        2,3,1,0,
        3,0,2,1

    ]


    index=0



    for r in range(4):

        for c in range(4):


            x=(
                MARGIN+
                c*(BANK_SIZE+BANK_GAP)
            )


            y=(
                MARGIN+
                r*(BANK_SIZE+BANK_GAP)
            )


            create_bank(
                img,
                x,
                y,
                bank_types[index],
                (
                    r==FEATURE_BANK_Y
                    and
                    c==FEATURE_BANK_X
                )
            )


            index+=1



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

    print("DRIFT-SENSE DRAM-19 GENERATOR")

    print("="*70)



    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )



    print("[1] Creating hierarchical DRAM banks")


    scene=create_scene()



    # reference around special feature


    ref_x=(
        MARGIN+
        FEATURE_BANK_X*(BANK_SIZE+BANK_GAP)
        +250
    )


    ref_y=(
        MARGIN+
        FEATURE_BANK_Y*(BANK_SIZE+BANK_GAP)
        +250
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
        "dram_19",

        "architecture":
        "hierarchical_multi_bank_dram",

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


    print("DONE")
    print(OUTPUT)



if __name__=="__main__":

    main()