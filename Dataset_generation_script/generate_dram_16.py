import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-16
# Contact Array + Local Defect Landmark
# ============================================================


SEED = 20260826

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_16"
)



SCENE_SIZE = 10000

REFERENCE_SIZE = 1000



# ============================================================
# DRAM ARRAY PARAMETERS
# ============================================================


PITCH = 110


CONTACT_RADIUS = 18



# special structure location

FEATURE_X = 5200

FEATURE_Y = 4700




# ============================================================
# DRAW HELPERS
# ============================================================


def circle(
        img,
        x,
        y,
        r,
        value
):

    cv2.circle(
        img,
        (
            int(x),
            int(y)
        ),
        int(r),
        int(value),
        -1,
        lineType=cv2.LINE_AA
    )



def rectangle(
        img,
        x1,
        y1,
        x2,
        y2,
        value
):

    cv2.rectangle(
        img,
        (
            int(x1),
            int(y1)
        ),
        (
            int(x2),
            int(y2)
        ),
        int(value),
        -1
    )



# ============================================================
# NORMAL DRAM CONTACT ARRAY
# ============================================================


def generate_contact_array(img):


    margin=400


    for y in range(
        margin,
        SCENE_SIZE-margin,
        PITCH
    ):


        for x in range(
            margin,
            SCENE_SIZE-margin,
            PITCH
        ):


            # small fabrication variation

            radius = (
                CONTACT_RADIUS
                +
                rng.normal(
                    0,
                    1.5
                )
            )


            circle(
                img,
                x,
                y,
                radius,
                185
            )


            # small highlight

            circle(
                img,
                x-2,
                y-2,
                radius*0.35,
                220
            )



# ============================================================
# SPECIAL STRUCTURE
# ============================================================


def generate_local_feature(img):


    cx=FEATURE_X

    cy=FEATURE_Y



    # dark isolation square

    rectangle(
        img,
        cx-160,
        cy-160,
        cx+160,
        cy+160,
        35
    )


    # bright defect / process mark

    rectangle(
        img,
        cx-90,
        cy-90,
        cx+90,
        cy+90,
        210
    )



    # diagonal missing line

    cv2.line(
        img,
        (
            cx-70,
            cy-70
        ),
        (
            cx+70,
            cy+70
        ),
        60,
        8
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
        25,
        dtype=np.uint8
    )



    generate_contact_array(
        img
    )


    generate_local_feature(
        img
    )


    return img



# ============================================================
# SEM IMAGING
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
        0.9
    )


    out=out.astype(float)


    # SEM dose noise

    out+=rng.poisson(
        3,
        img.shape
    )


    # charging gradient

    out+=np.linspace(
        -8,
        8,
        img.shape[1]
    )


    # scan variation

    out+=(
        rng.normal(
            0,
            1.5,
            img.shape[0]
        )
        [:,None]
    )


    # detector noise

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
        "DRIFT-SENSE DRAM-16 GENERATOR"
    )

    print("="*70)



    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )


    print("[1] Creating DRAM contact wafer")


    scene=create_scene()



    print("[2] Extracting reference")


    ref_x=4700

    ref_y=4200



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



    print("[4] Applying SEM noise")


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
        "dram_16",

        "architecture":
        "uniform_contact_array_with_local_process_feature",

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