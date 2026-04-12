import torch
from ultralytics.models.sam import sam3
from ultralytics.models.sam import SAM3SemanticPredictor
import glob
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
#import syntheticexamplestuff 
from gmphd import Gmphd,GmphdComponent
#Helper functions
def extract_measurements(results):
    """
    Parse the SAM3 results into a list of measurements dicts

    Each measurement will hold:
        z: 2x1 [cx,cy] centroid in pixel space
        conf: a scalar detection confidence in [0,1]

    Confidence is not folded into R here. It is passed through to the PHD likelihood so the update step can weight each gaussian accordingly.
    """
    measurements = []
    for result in results:
        if result.masks is None:
            continue
        masks = result.masks.data.cpu().numpy()
        confs = (result.boxes.conf.cpu().numpy() 
                if result.boxes is not None else np.ones(len(masks)))

        for mask,conf in zip(masks,confs):
            ys,xs = np.where(mask > MIN_DETECTION_CONF)
            if len(xs) ==0:
                continue
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            measurements.append({"z":np.array([cx,cy,0.0,0.0]),"conf":float(conf)})
    return measurements


def build_birth_gmm(measurements, birth_weight, P_birth):
    """Spawn a birth component at each measurement location each frame"""
    birth_gmm = []
    for m in measurements:
        z=m["z"]
        #born with zero velocity since we don't know it yet
        loc = np.array([z[0],z[1],0.0,0.0])
        birth_gmm.append(GmphdComponent(
                weight=birth_weight,
                mean=loc,
                cov = P_birth.copy()
        ))
    return birth_gmm

if __name__ == "__main__":
    # Initialize predictor
    overrides = dict(conf=0.25, task="segment", mode="predict", model="sam3.pt", half=True, save=True)
    predictor = SAM3SemanticPredictor(overrides=overrides)


    #print("Please input filepath for video analysis")
    #"~/data/M3OT/1/rgb/train/1-01/img1/000001.PNG"
    #filepath = input()

    import os
    image_paths = glob.glob(os.path.expanduser("~/data/M3OT/1/rgb/train/1-01/img1/000001.PNG"))
    print(f"Found {len(image_paths)} images: {image_paths}")
    #image_paths = glob.glob("~/data/M3OT/1/rgb/train/1-01/img1/000001.PNG")
    image_area = 1920 * 1080
    # Detection Thresholds
    MASK_PIXEL_THRESHOLD = 0.5 # Only want to really accept things that we are some level of confident in. 50% feels reasonable
    MIN_DETECTION_CONF   = 0.5

    #Motion Model: CV and White noise accelertation
    dt = 1
    #Define the state transition matrix to be x = [x,y,vx,vy]
    F = np.array([[1,0,dt,0],
                [0,1,0,dt],
                [0,0,1,0],
                [0,0,0,1]], dtype=np.float64)


    q = 75
    Q = q * np.array([
        [dt**3/3, 0,        dt**2/2, 0],
        [0,        dt**3/3, 0,        dt**2/2],
        [dt**2/2, 0,        dt,        0],
        [0,        dt**2/2, 0,        dt]
    ],dtype=np.float64)

    #Measurement Model: Observing [x,y] only
    H = np.hstack((np.eye(2), np.zeros((2,2))))
    r = 5.0
    R = r*np.eye(2)

    #GM-PHD parameters
    birth_prob              = 0.1
    survival_prob           = 0.975
    detect_prob             = 0.95
    clutter_total = 5 #Defines clutter per frame
    clutter_intensity = clutter_total/image_area
    bias                    = 2 #Defines the bias towards false positives over missed detections

    m0 = np.array([0.0,0.0,0.0,0.0]) #x,y,Vx,Vy
    P0 = 10000 * np.eye(4)


    #birth_gmm = [{"weight": birth_prob, "mean": m0.copy(), "cov": P0.copy()}]
    birth_gmm = [GmphdComponent(weight=birth_prob, mean=m0.copy(), cov=P0.copy())]
    gmphd_filter = Gmphd(birth_gmm,birth_prob,survival_prob,detect_prob, F, Q, H, R, clutter_intensity)


    for path in image_paths: 
        try: 
            #Open image
            with Image.open(path) as img:
                predictor.set_image(img)
                results = predictor(text=["vehicles"])
                # ── SAM3 debug ────────────────────────────────────────────────────
                print(f"Number of results: {len(results)}")
                for i, r in enumerate(results):
                    print(f"  Result {i}:")
                    print(f"    masks:  {r.masks.data.shape if r.masks is not None else 'None'}")
                    print(f"    boxes:  {r.boxes.data.shape if r.boxes is not None else 'None'}")
                    print(f"    confs:  {r.boxes.conf.cpu().numpy() if r.boxes is not None else 'None'}")

                measurements = extract_measurements(results)

                print(f"Measurements extracted: {len(measurements)}")
                for m in measurements:
                    print(f"  z={m['z']}, conf={m['conf']:.3f}")

                for r in results:
                    r.show()
                gmphd_filter.birthgmm = build_birth_gmm(measurements, birth_prob, P0)
                gmphd_filter.update(measurements)
                gmphd_filter.prune_targets()
                targets = gmphd_filter.extractstate()

                print(f"[{path}] {len(measurements)} measurements → {len(targets)} targets")
        except Exception as e:
            print(f"Error processing {path}: {e}")



