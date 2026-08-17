import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-28
#
# Folded Bitline DRAM Architecture
#
# Different from previous:
# - no circular contact array
# - no die corner matching
# - no missing defect
#
# Localization:
# folded bitline phase + isolation boundary
# ============================================================


SEED = 20260906

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_28"
)


SIZE = 10000

REF_SIZE = 1000



DIE = 2400

GAP = 400

MARGIN = 300



# ============================================================
# DRAW
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



def ellipse(img,x,y,rx,ry,val):

    cv2.ellipse(
        img,
        (int(x),int(y)),
        (int(rx),int(ry)),
        0,
        0,
        360,
        int(val),
        -1
    )



# ============================================================
# FOLDED BITLINE CELL
# ============================================================


def folded_array(
        img,
        x,
        y,
        size,
        pitch
):


    rows=size//pitch


    for r in range(rows):


        yy=y+r*pitch


        # alternating folded bitline direction

        if r%2==0:


            for xx in range(
                x,
                x+size,
                pitch
            ):


                line(
                    img,
                    xx,
                    yy-25,
                    xx,
                    yy+25,
                    100,
                    3
                )


                ellipse(
                    img,
                    xx,
                    yy,
                    12,
                    18,
                    220
                )


        else:


            for xx in range(
                x+pitch//2,
                x+size,
                pitch
            ):


                line(
                    img,
                    xx,
                    yy-25,
                    xx,
                    yy+25,
                    120,
                    3
                )


                ellipse(
                    img,
                    xx,
                    yy,
                    12,
                    18,
                    235
                )





# ============================================================
# DRAM BLOCK
# ============================================================


def create_block(
        img,
        x,
        y,
        mode
):


    cv2.rectangle(
        img,
        (x,y),
        (
            x+DIE,
            y+DIE
        ),
        25,
        -1
    )


    cv2.rectangle(
        img,
        (x+80,y+80),
        (
            x+DIE-80,
            y+DIE-80
        ),
        55,
        6
    )


    if mode==0:

        folded_array(
            img,
            x+200,
            y+200,
            DIE-400,
            55
        )


    elif mode==1:

        folded_array(
            img,
            x+200,
            y+200,
            DIE-400,
            75
        )


    elif mode==2:

        folded_array(
            img,
            x+200,
            y+200,
            DIE-400,
            45
        )


    else:

        folded_array(
            img,
            x+200,
            y+200,
            DIE-400,
            65
        )



# ============================================================
# CREATE WAFER
# ============================================================


def create_scene():


    img=np.full(
        (
            SIZE,
            SIZE
        ),
        15,
        dtype=np.uint8
    )


    patterns=[
        0,1,2,
        3,0,1,
        2,3,0
    ]


    k=0


    for r in range(3):

        for c in range(3):


            x=MARGIN+c*(DIE+GAP)

            y=MARGIN+r*(DIE+GAP)


            create_block(
                img,
                x,
                y,
                patterns[k]
            )


            k+=1


    return img



# ============================================================
# SEM
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
        0.7,
        img.shape
    )


    return np.clip(out,0,255).astype(np.uint8)





def search_sem(img):


    out=cv2.GaussianBlur(
        img,
        (5,5),
        0.9
    )


    out=out.astype(float)


    # vertical charging

    gradient=np.linspace(
        -8,
        8,
        img.shape[0]
    )

    out+=gradient[:,None]



    # detector stripe

    out+=rng.normal(
        0,
        2,
        img.shape
    )


    out+=rng.poisson(
        3,
        img.shape
    )


    return np.clip(out,0,255).astype(np.uint8)



# ============================================================
# MAIN
# ============================================================


def main():


    print("DRIFT-SENSE DRAM-28")


    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )


    scene=create_scene()



    # reference crosses block boundary

    ref_x=(
        MARGIN+
        DIE+
        GAP//2
    )


    ref_y=(
        MARGIN+
        600
    )



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

        "pair":"dram_28",

        "architecture":
        "folded_bitline_dram",

        "reference":
        "block_boundary_context",

        "reference_origin_nm":
        [
            ref_x,
            ref_y
        ],

        "scale_ratio":
        10

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


    print("DONE")
    print(OUTPUT)



if __name__=="__main__":

    main()