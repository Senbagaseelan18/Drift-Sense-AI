import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-18
# Multi Die DRAM SEM Localization Generator
# ============================================================


SEED = 20260828

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_18"
)


SCENE = 10000

REF_SIZE = 1000



# ============================================================
# DIE SETTINGS
# ============================================================


DIE = 2200

GAP = 300

MARGIN = 250



# unique structure location

TARGET_DIE_X = 3

TARGET_DIE_Y = 2



# ============================================================
# HELPERS
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
# DIE PATTERNS
# ============================================================


def dense_contact(img,x,y,size):

    pitch=45

    for yy in range(y,y+size,pitch):

        for xx in range(x,x+size,pitch):

            dot(
                img,
                xx,
                yy,
                8,
                220
            )



def horizontal_array(img,x,y,size):

    pitch=55

    for yy in range(y,y+size,pitch):

        line(
            img,
            x,
            yy,
            x+size,
            yy,
            150,
            5
        )

    for yy in range(y+20,y+size,pitch):

        for xx in range(x+30,x+size,80):

            dot(
                img,
                xx,
                yy,
                7,
                210
            )




def stagger_array(img,x,y,size):

    pitch=70

    row=0

    for yy in range(y,y+size,pitch):

        offset=0

        if row%2:
            offset=35


        for xx in range(x+offset,x+size,pitch):

            dot(
                img,
                xx,
                yy,
                10,
                215
            )

        row+=1




def fin_array(img,x,y,size):

    pitch=45

    for xx in range(x,x+size,pitch):

        line(
            img,
            xx,
            y,
            xx,
            y+size,
            130,
            4
        )


    for yy in range(y+40,y+size,90):

        for xx in range(x+20,x+size,90):

            dot(
                img,
                xx,
                yy,
                6,
                220
            )




def defect_array(img,x,y,size):

    pitch=55

    for yy in range(y,y+size,pitch):

        for xx in range(x,x+size,pitch):

            if rng.random()>0.08:

                dot(
                    img,
                    xx,
                    yy,
                    8,
                    220
                )



# ============================================================
# SMALL DETECTION FEATURE
# ============================================================


def add_landmark(img,x,y):


    # small asymmetric process anomaly

    dot(
        img,
        x,
        y,
        14,
        230
    )


    dot(
        img,
        x+25,
        y,
        14,
        230
    )


    # missing contact

    cv2.rectangle(
        img,
        (
            x-10,
            y-40
        ),
        (
            x+45,
            y+40
        ),
        40,
        -1
    )



# ============================================================
# CREATE DIE
# ============================================================


def create_die(img,x,y,kind,landmark=False):


    cv2.rectangle(
        img,
        (x,y),
        (x+DIE+x if False else x+DIE,y+DIE+y),
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
            x+DIE-40,
            y+DIE-40
        ),
        55,
        8
    )


    ax=x+220
    ay=y+220

    size=DIE-440



    if kind==0:
        dense_contact(img,ax,ay,size)

    elif kind==1:
        horizontal_array(img,ax,ay,size)

    elif kind==2:
        stagger_array(img,ax,ay,size)

    elif kind==3:
        fin_array(img,ax,ay,size)

    else:
        defect_array(img,ax,ay,size)



    if landmark:

        add_landmark(
            img,
            x+650,
            y+650
        )



# ============================================================
# SEARCH FIELD
# ============================================================


def generate_scene():


    img=np.full(
        (
            SCENE,
            SCENE
        ),
        15,
        dtype=np.uint8
    )


    kinds=[

        0,1,2,3,
        4,0,2,1,
        3,4,1,0,
        2,3,4,1

    ]


    k=0


    for r in range(4):

        for c in range(4):


            x=MARGIN+c*(DIE+GAP)

            y=MARGIN+r*(DIE+GAP)


            create_die(
                img,
                x,
                y,
                kinds[k],
                (
                    r==TARGET_DIE_Y
                    and
                    c==TARGET_DIE_X
                )
            )


            k+=1



    return img



# ============================================================
# SEM EFFECT
# ============================================================


def sem_reference(img):

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


    return np.clip(out,0,255).astype(np.uint8)



def sem_search(img):

    out=cv2.GaussianBlur(
        img,
        (5,5),
        0.9
    )


    out=out.astype(float)


    out+=rng.poisson(
        3,
        img.shape
    )


    out+=np.linspace(
        -8,
        8,
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


    return np.clip(out,0,255).astype(np.uint8)



# ============================================================
# MAIN
# ============================================================


def main():

    print("="*60)
    print("DRIFT-SENSE DRAM-18")
    print("="*60)


    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )


    scene=generate_scene()



    # reference around landmark

    ref_x=(
        MARGIN+
        TARGET_DIE_X*(DIE+GAP)
        +400
    )


    ref_y=(
        MARGIN+
        TARGET_DIE_Y*(DIE+GAP)
        +400
    )



    reference=scene[
        ref_y:ref_y+REF_SIZE,
        ref_x:ref_x+REF_SIZE
    ]



    search=cv2.resize(
        scene,
        (1000,1000),
        interpolation=cv2.INTER_AREA
    )


    reference=sem_reference(reference)

    search=sem_search(search)



    cv2.imwrite(
        str(OUTPUT/"reference_100x.png"),
        reference
    )


    cv2.imwrite(
        str(OUTPUT/"search_10x.png"),
        search
    )



    gt={

        "pair":"dram_18",

        "architecture":
        "multi_die_dram_with_unique_process_feature",

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
        ]

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


    print("Generated:")
    print(OUTPUT)



if __name__=="__main__":

    main()