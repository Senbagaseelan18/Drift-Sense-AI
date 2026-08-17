import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-14
# Realistic DRAM SEM Localization Dataset Generator
# ============================================================


SEED = 20260824

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_14"
)


# ============================================================
# PHYSICAL PARAMETERS
# ============================================================


SCENE_SIZE = 10000

REFERENCE_SIZE = 1000


DIE_SIZE = 2200

STREET = 300

MARGIN = 300



# ============================================================
# DRAW HELPERS
# ============================================================


def draw_line(
        img,
        p1,
        p2,
        value,
        width
):

    cv2.line(
        img,
        tuple(map(int,p1)),
        tuple(map(int,p2)),
        int(value),
        int(width),
        lineType=cv2.LINE_AA
    )



def draw_contact(
        img,
        x,
        y,
        radius,
        value
):

    cv2.ellipse(
        img,
        (
            int(x),
            int(y)
        ),
        (
            int(radius),
            int(radius*1.25)
        ),
        0,
        0,
        360,
        int(value),
        -1
    )



# ============================================================
# DRAM ARRAY TYPES
# ============================================================


def dense_array(
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

            draw_contact(
                img,
                xx,
                yy,
                7,
                220
            )




def sparse_array(
        img,
        x,
        y,
        size
):

    pitch=65


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

            if rng.random()>0.15:

                draw_contact(
                    img,
                    xx,
                    yy,
                    9,
                    220
                )





def rotated_array(
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

        for xx in range(
            x,
            x+size,
            pitch
        ):

            shift=(yy%100)//5


            draw_contact(
                img,
                xx+shift,
                yy,
                8,
                210
            )





def defect_array(
        img,
        x,
        y,
        size
):

    pitch=50


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


            if rng.random()>0.06:

                draw_contact(
                    img,
                    xx,
                    yy,
                    8,
                    215
                )



# ============================================================
# CREATE SINGLE DIE
# ============================================================


def create_die(
        scene,
        x,
        y,
        style
):


    # substrate

    cv2.rectangle(
        scene,
        (
            x,
            y
        ),
        (
            x+DIE_SIZE,
            y+DIE_SIZE
        ),
        25,
        -1
    )


    # die boundary

    cv2.rectangle(
        scene,
        (
            x+50,
            y+50
        ),
        (
            x+DIE_SIZE-50,
            y+DIE_SIZE-50
        ),
        60,
        8
    )



    array_x=x+250
    array_y=y+250

    array_size=DIE_SIZE-500



    if style=="dense":

        dense_array(
            scene,
            array_x,
            array_y,
            array_size
        )


    elif style=="sparse":

        sparse_array(
            scene,
            array_x,
            array_y,
            array_size
        )


    elif style=="rotated":

        rotated_array(
            scene,
            array_x,
            array_y,
            array_size
        )


    else:

        defect_array(
            scene,
            array_x,
            array_y,
            array_size
        )



    # array boundary

    cv2.rectangle(
        scene,
        (
            array_x-20,
            array_y-20
        ),
        (
            array_x+array_size+20,
            array_y+array_size+20
        ),
        45,
        5
    )



# ============================================================
# CREATE FULL SEARCH SCENE
# ============================================================


def create_search_scene():


    scene=np.full(
        (
            SCENE_SIZE,
            SCENE_SIZE
        ),
        18,
        dtype=np.uint8
    )


    styles=[

        "dense",
        "sparse",
        "rotated",

        "defect",
        "dense",
        "sparse",

        "rotated",
        "defect",
        "dense"

    ]


    index=0


    for row in range(3):

        for col in range(3):


            xpos=(
                MARGIN+
                col*(DIE_SIZE+STREET)
            )


            ypos=(
                MARGIN+
                row*(DIE_SIZE+STREET)
            )


            create_die(
                scene,
                xpos,
                ypos,
                styles[index]
            )


            index+=1


    return scene



# ============================================================
# SEM SIMULATION
# ============================================================


def reference_sem(img):


    img=cv2.GaussianBlur(
        img,
        (3,3),
        0.4
    )


    img=img.astype(float)


    img+=rng.normal(
        0,
        1,
        img.shape
    )


    return np.clip(
        img,
        0,
        255
    ).astype(np.uint8)





def search_sem(img):


    img=cv2.GaussianBlur(
        img,
        (5,5),
        0.9
    )


    img=img.astype(float)


    # dose noise

    img+=rng.poisson(
        3,
        img.shape
    )



    # charging

    gradient=np.linspace(
        -8,
        8,
        img.shape[1]
    )


    img+=gradient



    # scan lines

    rows=rng.normal(
        0,
        2,
        img.shape[0]
    )


    img+=rows[:,None]



    # detector grain

    img+=rng.normal(
        0,
        2,
        img.shape
    )


    return np.clip(
        img,
        0,
        255
    ).astype(np.uint8)




# ============================================================
# MAIN
# ============================================================


def main():


    print("="*70)
    print("DRIFT-SENSE DRAM-14 GENERATOR")
    print("="*70)


    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )


    print("[1] Generating DRAM wafer...")


    scene=create_search_scene()



    # choose reference from middle dense die region

    ref_x=4300

    ref_y=4300



    print("[2] Extracting reference...")


    reference=scene[
        ref_y:
        ref_y+REFERENCE_SIZE,

        ref_x:
        ref_x+REFERENCE_SIZE
    ]



    print("[3] Creating 10x search...")


    search=cv2.resize(
        scene,
        (
            1000,
            1000
        ),
        interpolation=cv2.INTER_AREA
    )



    print("[4] SEM degradation...")


    reference=reference_sem(
        reference
    )


    search=search_sem(
        search
    )



    print("[5] Saving...")


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
        "dram_14",

        "architecture":
        "multi_die_dram_sem",

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

        "scale":
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
    print("DONE")
    print(
        OUTPUT
    )



if __name__=="__main__":

    main()