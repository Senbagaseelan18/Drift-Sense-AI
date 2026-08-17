import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-22
# Layered DRAM Mesh + Contact Collapse Defect
# ============================================================


SEED = 20260832

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_22"
)


SCENE_SIZE = 10000

REFERENCE_SIZE = 1000



# ============================================================
# DIE PARAMETERS
# ============================================================


DIE_SIZE = 2200

GAP = 350

MARGIN = 250



# target and distractor

TARGET = (2,2)

DISTRACTOR = (1,3)



# ============================================================
# DRAW HELPERS
# ============================================================


def draw_line(
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



def draw_circle(
        img,
        x,
        y,
        r,
        value
):

    cv2.circle(
        img,
        (int(x),int(y)),
        int(r),
        int(value),
        -1,
        lineType=cv2.LINE_AA
    )



# ============================================================
# DRAM PATTERNS
# ============================================================


def vertical_fin_mesh(
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

        draw_line(
            img,
            xx,
            y,
            xx,
            y+size,
            95,
            5
        )


    for yy in range(
        y+30,
        y+size,
        80
    ):

        for xx in range(
            x+15,
            x+size,
            90
        ):

            draw_circle(
                img,
                xx,
                yy,
                7,
                220
            )




def cross_grid(
        img,
        x,
        y,
        size
):


    pitch=80


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
            130,
            5
        )


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
        y+30,
        y+size,
        pitch
    ):

        for xx in range(
            x+30,
            x+size,
            pitch
        ):

            draw_circle(
                img,
                xx,
                yy,
                9,
                220
            )





def sparse_capacitor(
        img,
        x,
        y,
        size
):


    pitch=120

    row=0


    for yy in range(
        y,
        y+size,
        pitch
    ):

        offset=0

        if row%2:

            offset=60


        for xx in range(
            x+offset,
            x+size,
            pitch
        ):

            draw_circle(
                img,
                xx,
                yy,
                16,
                220
            )


        row+=1





def mixed_region(
        img,
        x,
        y,
        size
):


    half=size//2


    vertical_fin_mesh(
        img,
        x,
        y,
        half
    )


    sparse_capacitor(
        img,
        x+half,
        y,
        half
    )


# ============================================================
# DEFECT
# ============================================================


def add_collapse_defect(
        img,
        x,
        y,
        rotate=False
):


    # remove local area

    cv2.rectangle(
        img,
        (
            x-60,
            y-60
        ),
        (
            x+60,
            y+60
        ),
        25,
        -1
    )


    if rotate:


        cv2.rectangle(
            img,
            (
                x-20,
                y-80
            ),
            (
                x+50,
                y+20
            ),
            220,
            -1
        )


    else:


        cv2.rectangle(
            img,
            (
                x-70,
                y-20
            ),
            (
                x+20,
                y+50
            ),
            220,
            -1
        )



    draw_circle(
        img,
        x+80,
        y-40,
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
        pattern,
        defect=False,
        rotated=False
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



    ax=x+200
    ay=y+200

    size=DIE_SIZE-400



    if pattern==0:

        vertical_fin_mesh(
            img,
            ax,
            ay,
            size
        )


    elif pattern==1:

        cross_grid(
            img,
            ax,
            ay,
            size
        )


    elif pattern==2:

        sparse_capacitor(
            img,
            ax,
            ay,
            size
        )


    else:

        mixed_region(
            img,
            ax,
            ay,
            size
        )



    if defect:

        add_collapse_defect(
            img,
            x+700,
            y+700,
            rotated
        )



# ============================================================
# SCENE
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



    patterns=[

        0,1,2,3,
        1,2,3,0,
        2,3,1,0,
        3,0,2,1

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
                (r,c)==TARGET or (r,c)==DISTRACTOR,
                (r,c)==DISTRACTOR
            )


            k+=1



    return img



# ============================================================
# SEM NOISE
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

    print("DRIFT-SENSE DRAM-22 GENERATOR")

    print("="*70)



    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )



    print("[1] Creating DRAM mesh field")

    scene=create_scene()



    ref_x=(
        MARGIN+
        TARGET[1]*(DIE_SIZE+GAP)
        +350
    )


    ref_y=(
        MARGIN+
        TARGET[0]*(DIE_SIZE+GAP)
        +350
    )



    print("[2] Reference crop")


    reference=scene[
        ref_y:
        ref_y+REFERENCE_SIZE,

        ref_x:
        ref_x+REFERENCE_SIZE
    ]



    print("[3] Search SEM")


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
            OUTPUT/"reference_100x.png"
        ),
        reference
    )


    cv2.imwrite(
        str(
            OUTPUT/"search_10x.png"
        ),
        search
    )



    gt={

        "pair":"dram_22",

        "architecture":
        "layered_dram_mesh_with_contact_collapse",

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

        "scale_ratio":10

    }



    with open(
        OUTPUT/"ground_truth.json",
        "w"
    ) as f:

        json.dump(
            gt,
            f,
            indent=4
        )



    print("DRAM-22 COMPLETE")
    print(OUTPUT)



if __name__=="__main__":

    main()