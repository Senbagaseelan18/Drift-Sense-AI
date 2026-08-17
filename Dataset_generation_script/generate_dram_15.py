import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRIFT-SENSE DRAM-15
# Trench / Fin Array DRAM Synthetic Generator
# ============================================================


SEED = 20260825

rng = np.random.default_rng(SEED)


OUTPUT = (
    Path(__file__).resolve().parents[1] /
    "results" / "generated_dataset_images" / "dram_15"
)


# Physical SEM field

SCENE_SIZE = 10000

REFERENCE_SIZE = 1000


# ============================================================
# DIE GEOMETRY
# ============================================================


DIE_SIZE = 2200

STREET = 300

MARGIN = 300



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



def draw_contact(
        img,
        x,
        y,
        rx,
        ry,
        value
):

    cv2.ellipse(
        img,
        (
            int(x),
            int(y)
        ),
        (
            int(rx),
            int(ry)
        ),
        0,
        0,
        360,
        int(value),
        -1
    )



# ============================================================
# DRAM ARCHITECTURES
# ============================================================


def trench_dense(
        img,
        x,
        y,
        size
):

    pitch = 38


    # vertical trench array

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
            70,
            5
        )


    # small capacitor nodes

    for yy in range(
        y+20,
        y+size,
        55
    ):

        for xx in range(
            x+20,
            x+size,
            pitch
        ):

            draw_contact(
                img,
                xx,
                yy,
                6,
                14,
                210
            )





def staggered_capacitor(
        img,
        x,
        y,
        size
):

    pitch = 65


    row=0


    for yy in range(
        y+40,
        y+size,
        pitch
    ):


        offset = 0

        if row%2:
            offset = pitch//2


        for xx in range(
            x+offset,
            x+size,
            pitch
        ):


            draw_contact(
                img,
                xx,
                yy,
                10,
                16,
                220
            )


        row+=1





def open_pitch(
        img,
        x,
        y,
        size
):


    pitch=90


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
            55,
            6
        )


    for yy in range(
        y+20,
        y+size,
        pitch
    ):


        draw_contact(
            img,
            x+40,
            yy,
            12,
            20,
            200
        )





def defect_trench(
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


        if rng.random()>0.08:

            draw_line(
                img,
                xx,
                y,
                xx,
                y+size,
                65,
                5
            )


    for yy in range(
        y+20,
        y+size,
        55
    ):

        for xx in range(
            x+20,
            x+size,
            pitch
        ):


            if rng.random()>0.12:

                draw_contact(
                    img,
                    xx,
                    yy,
                    7,
                    14,
                    210
                )



# ============================================================
# CREATE DIE
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
        22,
        -1
    )



    # die frame

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
        55,
        10
    )



    ax=x+220
    ay=y+220

    size=DIE_SIZE-440



    if style=="dense":

        trench_dense(
            scene,
            ax,
            ay,
            size
        )


    elif style=="stagger":

        staggered_capacitor(
            scene,
            ax,
            ay,
            size
        )


    elif style=="open":

        open_pitch(
            scene,
            ax,
            ay,
            size
        )


    else:

        defect_trench(
            scene,
            ax,
            ay,
            size
        )



    # array boundary

    cv2.rectangle(
        scene,
        (
            ax-20,
            ay-20
        ),
        (
            ax+size+20,
            ay+size+20
        ),
        45,
        5
    )



# ============================================================
# FULL SEARCH FIELD
# ============================================================


def generate_scene():


    scene=np.full(
        (
            SCENE_SIZE,
            SCENE_SIZE
        ),
        15,
        dtype=np.uint8
    )


    patterns=[

        "dense",
        "stagger",
        "open",

        "defect",
        "dense",
        "stagger",

        "open",
        "defect",
        "dense"

    ]


    k=0


    for r in range(3):

        for c in range(3):


            x=(
                MARGIN+
                c*(DIE_SIZE+STREET)
            )


            y=(
                MARGIN+
                r*(DIE_SIZE+STREET)
            )


            create_die(
                scene,
                x,
                y,
                patterns[k]
            )

            k+=1


    return scene



# ============================================================
# SEM PHYSICS
# ============================================================


def reference_noise(img):


    out=cv2.GaussianBlur(
        img,
        (3,3),
        0.35
    )


    out=out.astype(float)


    out += rng.normal(
        0,
        1,
        out.shape
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
        0.85
    )


    out=out.astype(float)


    # lower SEM dose

    out += rng.poisson(
        2,
        out.shape
    )


    # charging gradient

    out += np.linspace(
        -7,
        7,
        out.shape[1]
    )


    # scan drift

    out += (
        rng.normal(
            0,
            1.5,
            out.shape[0]
        )
        [:,None]
    )


    # detector grain

    out += rng.normal(
        0,
        2,
        out.shape
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
    print("DRIFT-SENSE DRAM-15 GENERATOR")
    print("="*70)



    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )


    print("[1] Creating trench DRAM wafer...")


    scene=generate_scene()



    # Choose crop containing die edge + array

    ref_x=3600
    ref_y=3600



    print("[2] Extracting reference...")


    reference=scene[
        ref_y:
        ref_y+REFERENCE_SIZE,

        ref_x:
        ref_x+REFERENCE_SIZE
    ]



    print("[3] Creating search SEM...")


    search=cv2.resize(
        scene,
        (
            1000,
            1000
        ),
        interpolation=cv2.INTER_AREA
    )



    print("[4] Applying SEM noise...")


    reference=reference_noise(
        reference
    )


    search=search_noise(
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
        "dram_15",

        "architecture":
        "trench_fin_array_dram",

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
    print("DRAM-15 COMPLETE")
    print(OUTPUT)



if __name__=="__main__":

    main()