import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRAM-48
# Filled Multi Die DRAM Wafer
# Target: Dark Trench Defect
# ============================================================


SEED = 20260818

rng = np.random.default_rng(SEED)


OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_48"


SCENE = 12000

DIE = 2500

GAP = 400

REF = 1000



TARGET_DIE = (2,1)


# ============================================================
# DRAW HELPERS
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



def dot(img,x,y,r,val):

    cv2.circle(
        img,
        (int(x),int(y)),
        int(r),
        int(val),
        -1,
        lineType=cv2.LINE_AA
    )


# ============================================================
# DIFFERENT DRAM DIE TYPES
# ============================================================


def dense_contact_die(img,x,y):

    pitch=70


    for row,yy in enumerate(
        range(
            y+150,
            y+DIE-150,
            pitch
        )
    ):

        offset=0

        if row%2:
            offset=pitch//2


        for xx in range(
            x+150+offset,
            x+DIE-150,
            pitch
        ):


            dot(
                img,
                xx,
                yy,
                10,
                220
            )


            dot(
                img,
                xx,
                yy,
                3,
                250
            )



def vertical_bitline_die(img,x,y):


    for xx in range(
        x+150,
        x+DIE-150,
        55
    ):


        draw_line(
            img,
            xx,
            y+150,
            xx,
            y+DIE-150,
            120,
            4
        )


        for yy in range(
            y+250,
            y+DIE-250,
            100
        ):


            dot(
                img,
                xx,
                yy,
                8,
                230
            )



def horizontal_wordline_die(img,x,y):


    for yy in range(
        y+150,
        y+DIE-150,
        65
    ):


        draw_line(
            img,
            x+150,
            yy,
            x+DIE-150,
            yy,
            100,
            4
        )


    for yy in range(
        y+220,
        y+DIE-200,
        130
    ):

        for xx in range(
            x+200,
            x+DIE-200,
            120
        ):

            dot(
                img,
                xx,
                yy,
                12,
                220
            )



def mixed_die(img,x,y):


    dense_contact_die(
        img,
        x,
        y
    )


    for yy in range(
        y+300,
        y+DIE-200,
        500
    ):

        draw_line(
            img,
            x+100,
            yy,
            x+DIE-100,
            yy,
            60,
            12
        )


# ============================================================
# DEFECT
# ============================================================


def add_trench_defect(img,x,y):


    # remove local contacts

    cv2.rectangle(
        img,
        (
            x-80,
            y-120
        ),
        (
            x+80,
            y+120
        ),
        30,
        -1
    )


    # dark trench

    cv2.line(
        img,
        (
            x,
            y-130
        ),
        (
            x,
            y+130
        ),
        5,
        18
    )


    # rough process residue

    for i in range(20):

        dx=rng.integers(
            -100,
            100
        )

        dy=rng.integers(
            -150,
            150
        )


        dot(
            img,
            x+dx,
            y+dy,
            3,
            80
        )


# ============================================================
# CREATE SCENE
# ============================================================


def create_scene():


    img=np.full(
        (
            SCENE,
            SCENE
        ),
        25,
        dtype=np.uint8
    )


    types=[
        dense_contact_die,
        vertical_bitline_die,
        horizontal_wordline_die,
        mixed_die
    ]


    idx=0


    locations=[]


    for r in range(4):

        for c in range(4):


            x=300+c*(DIE+GAP)

            y=300+r*(DIE+GAP)


            # die border

            cv2.rectangle(
                img,
                (
                    x,
                    y
                ),
                (
                    x+DIE,
                    y+DIE
                ),
                60,
                10
            )


            func=types[idx%4]


            func(
                img,
                x,
                y
            )


            locations.append(
                (
                    x,
                    y
                )
            )


            idx+=1


    # target die

    tx=locations[
        TARGET_DIE[0]*4+TARGET_DIE[1]
    ]


    defect_x=tx[0]+1300

    defect_y=tx[1]+1300


    add_trench_defect(
        img,
        defect_x,
        defect_y
    )


    return img, defect_x, defect_y


# ============================================================
# SEM EFFECTS
# ============================================================


def reference_noise(img):


    out=cv2.GaussianBlur(
        img,
        (3,3),
        .25
    ).astype(float)


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



def search_noise(img):


    out=cv2.GaussianBlur(
        img,
        (5,5),
        1
    ).astype(float)


    h,w=img.shape


    # illumination field

    field=cv2.resize(
        rng.normal(
            0,
            12,
            (80,80)
        ),
        (w,h),
        interpolation=cv2.INTER_CUBIC
    )


    out+=field


    # scan drift

    out+=(
        np.sin(
            np.arange(h)/15
        )[:,None]
        *3
    )


    # detector variation

    out*=rng.normal(
        1,
        0.02,
        img.shape
    )


    # poisson

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


    print("DRAM-48 GENERATION")


    OUT.mkdir(
        parents=True,
        exist_ok=True
    )


    scene,dx,dy=create_scene()


    # reference from defect region

    rx=dx-500

    ry=dy-500


    reference=scene[
        ry:
        ry+REF,

        rx:
        rx+REF
    ]


    search=cv2.resize(
        scene,
        (
            1000,
            1000
        ),
        interpolation=cv2.INTER_AREA
    )


    reference=reference_noise(reference)

    search=search_noise(search)


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

        "pair":"dram_48",

        "architecture":
        "multi_die_filled_dram_wafer",

        "defect":
        "dark_trench",

        "defect_location_nm":
        [
            int(dx),
            int(dy)
        ],

        "reference_origin_nm":
        [
            int(rx),
            int(ry)
        ],

        "scale_ratio":10

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
