# face_score

Detect faces in video stream and score based on criteria:

- Size
- Shape
- Pose
- Sharpness (canny)

Combine scores and filter faces based on

- Size
- Shape
- Pose
- Sharpness (canny)
- Position tracking (filter stationary faces during timeout)

Send faces matching the criteria over network

Save debug frames for seeing scores

### TODOs

- face_score_config class for configuring face_score
- Distribute frames to be processed over network for multi processing
- Collect faces in same position using tracking, send best scoring faces in a configured period
- Create venv, handle dependencies & configure setup tools
- CI Pipeline
- Unit & perf tests

## Example Output

![faces](https://user-images.githubusercontent.com/7614296/216850694-077c04ae-a554-4ea0-855b-b9104d793ff6.jpg)

<details>
    <summary>logs</summary>

    -0 13.76 discard face:      {'discard': ['size'], 'final': 46.5, 'shape': 96.0, 'pose': 76.0, 'size': 0.5, 'size_raw': (36, 36), 'canny_raw': 33.65, 'canny': 115.0}
    -1 13.76 discard face:      {'discard': ['size'], 'final': 40.0, 'shape': 92.0, 'pose': 73.0, 'size': 0.5, 'size_raw': (36, 36), 'canny_raw': 13.18, 'canny': 84.0}
    -2 16.27 discard face:      {'discard': ['canny'], 'final': 46.75, 'shape': 75.0, 'pose': 50.0, 'size': 0.85, 'size_raw': (155, 155), 'canny_raw': 3.63, 'canny': 50.0}
    -3 17.52 discard face:      {'discard': ['size'], 'final': 41.5, 'shape': 92.0, 'pose': 60.0, 'size': 0.5, 'size_raw': (75, 75), 'canny_raw': 20.17, 'canny': 110.0}
    -4 18.77 discard face:      {'discard': ['size', 'pose'], 'final': 28.0, 'shape': 93.0, 'pose': 44.0, 'size': 0.5, 'size_raw': (75, 75), 'canny_raw': 8.07, 'canny': 50.0}
    -5 18.77 discard face:      {'discard': ['size'], 'final': 41.5, 'shape': 96.0, 'pose': 65.0, 'size': 0.5, 'size_raw': (62, 62), 'canny_raw': 18.97, 'canny': 100.0}
    -6 22.52 discard face:      {'discard': ['pose'], 'final': 51.0, 'shape': 74.0, 'pose': 41.0, 'size': 1.0, 'size_raw': (223, 223), 'canny_raw': 5.1, 'canny': 50.0}
    -7 23.77 discard face:      {'discard': ['size'], 'final': 38.0, 'shape': 71.0, 'pose': 51.0, 'size': 0.5, 'size_raw': (89, 89), 'canny_raw': 26.82, 'canny': 115.0}
    -8 25.02 discard face:      {'discard': ['pose'], 'final': 36.5, 'shape': 73.0, 'pose': 46.0, 'size': 0.5, 'size_raw': (108, 107), 'canny_raw': 21.25, 'canny': 110.0}
    #0 26.28 good    face:      {'discard': [], 'final': 43.5, 'shape': 75.0, 'pose': 72.0, 'size': 0.5, 'size_raw': (108, 107), 'canny_raw': 28.84, 'canny': 115.0}
    -9 26.28 discard face:      {'discard': ['pose'], 'final': 34.5, 'shape': 95.0, 'pose': 45.0, 'size': 0.5, 'size_raw': (107, 108), 'canny_raw': 12.82, 'canny': 86.0}
    -10 27.53 discard face:     {'discard': ['size', 'tracking'], 'final': 41.0, 'shape': 72.0, 'pose': 62.0, 'size': 0.5, 'size_raw': (89, 89), 'canny_raw': 31.16, 'canny': 115.0}
    #1 27.53 good    face:      {'discard': [], 'final': 41.0, 'shape': 77.0, 'pose': 72.0, 'size': 0.5, 'size_raw': (107, 107), 'canny_raw': 19.07, 'canny': 100.0}
    -11 28.78 discard face:     {'discard': ['size', 'tracking'], 'final': 37.5, 'shape': 72.0, 'pose': 52.0, 'size': 0.5, 'size_raw': (89, 89), 'canny_raw': 22.86, 'canny': 110.0}
    -12 28.78 discard face:     {'discard': ['size'], 'final': 42.0, 'shape': 77.0, 'pose': 69.0, 'size': 0.5, 'size_raw': (74, 74), 'canny_raw': 20.49, 'canny': 110.0}
    -13 30.03 discard face:     {'discard': ['size'], 'final': 41.0, 'shape': 97.0, 'pose': 52.0, 'size': 0.5, 'size_raw': (43, 43), 'canny_raw': 35.86, 'canny': 115.0}
    -14 31.28 discard face:     {'discard': ['size'], 'final': 48.0, 'shape': 96.0, 'pose': 82.0, 'size': 0.5, 'size_raw': (74, 75), 'canny_raw': 34.05, 'canny': 115.0}
    -15 31.28 discard face:     {'discard': ['size'], 'final': 40.5, 'shape': 77.0, 'pose': 59.0, 'size': 0.5, 'size_raw': (52, 52), 'canny_raw': 35.36, 'canny': 115.0}
    -16 32.53 discard face:     {'discard': ['size'], 'final': 46.5, 'shape': 94.0, 'pose': 76.0, 'size': 0.5, 'size_raw': (89, 90), 'canny_raw': 27.38, 'canny': 115.0}
    -17 33.78 discard face:     {'discard': ['size'], 'final': 42.0, 'shape': 73.0, 'pose': 67.0, 'size': 0.5, 'size_raw': (62, 63), 'canny_raw': 38.0, 'canny': 115.0}
    -18 33.78 discard face:     {'discard': ['size'], 'final': 48.0, 'shape': 95.0, 'pose': 82.0, 'size': 0.5, 'size_raw': (89, 90), 'canny_raw': 38.55, 'canny': 115.0}
    -19 36.29 discard face:     {'discard': ['pose'], 'final': 35.5, 'shape': 73.0, 'pose': 46.0, 'size': 0.5, 'size_raw': (107, 107), 'canny_raw': 16.73, 'canny': 106.0}
    -20 38.79 discard face:     {'discard': ['shape'], 'final': 80.08, 'shape': 60.0, 'pose': 68.0, 'size': 1.04, 'size_raw': (186, 186), 'canny_raw': 16.66, 'canny': 106.0}
    -21 42.54 discard face:     {'discard': ['shape', 'pose'], 'final': 52.0, 'shape': 69.0, 'pose': 46.0, 'size': 1.0, 'size_raw': (223, 223), 'canny_raw': 7.83, 'canny': 50.0}
    #2 43.79 good    face:      {'discard': [], 'final': 79.12, 'shape': 95.0, 'pose': 84.0, 'size': 0.86, 'size_raw': (154, 155), 'canny_raw': 18.15, 'canny': 102.0}
    -22 45.05 discard face:     {'discard': ['tracking'], 'final': 70.52, 'shape': 96.0, 'pose': 63.0, 'size': 0.86, 'size_raw': (154, 155), 'canny_raw': 19.42, 'canny': 99.0}
    #3 46.30 good    face:      {'discard': [], 'final': 83.2, 'shape': 77.0, 'pose': 59.0, 'size': 1.04, 'size_raw': (186, 185), 'canny_raw': 20.35, 'canny': 110.0}
    -23 47.55 discard face:     {'discard': ['tracking'], 'final': 68.85, 'shape': 77.0, 'pose': 62.0, 'size': 0.85, 'size_raw': (155, 155), 'canny_raw': 20.89, 'canny': 110.0}
    -24 48.80 discard face:     {'discard': ['tracking', 'pose'], 'final': 70.72, 'shape': 96.0, 'pose': 47.0, 'size': 1.04, 'size_raw': (186, 186), 'canny_raw': 13.98, 'canny': 80.0}
    -25 48.80 discard face:     {'discard': ['size'], 'final': 46.5, 'shape': 95.0, 'pose': 76.0, 'size': 0.5, 'size_raw': (43, 43), 'canny_raw': 52.13, 'canny': 115.0}
    -26 50.05 discard face:     {'discard': ['tracking', 'pose'], 'final': 78.0, 'shape': 94.0, 'pose': 47.0, 'size': 1.04, 'size_raw': (186, 186), 'canny_raw': 17.56, 'canny': 104.0}
    -27 51.30 discard face:     {'discard': ['pose'], 'final': 72.96, 'shape': 95.0, 'pose': 49.0, 'size': 0.96, 'size_raw': (129, 129), 'canny_raw': 17.76, 'canny': 103.0}
    #4 52.55 good    face:      {'discard': [], 'final': 68.0, 'shape': 92.0, 'pose': 54.0, 'size': 0.85, 'size_raw': (155, 155), 'canny_raw': 15.43, 'canny': 109.0}
    -28 53.80 discard face:     {'discard': ['tracking', 'pose'], 'final': 60.32, 'shape': 93.0, 'pose': 48.0, 'size': 1.04, 'size_raw': (186, 186), 'canny_raw': 7.74, 'canny': 50.0}
    -29 55.05 discard face:     {'discard': ['pose'], 'final': 54.08, 'shape': 72.0, 'pose': 44.0, 'size': 1.04, 'size_raw': (186, 185), 'canny_raw': 8.77, 'canny': 50.0}
    -30 56.31 discard face:     {'discard': ['tracking', 'pose'], 'final': 54.08, 'shape': 73.0, 'pose': 45.0, 'size': 1.04, 'size_raw': (186, 186), 'canny_raw': 7.78, 'canny': 50.0}
    #5 57.56 good    face:      {'discard': [], 'final': 62.400000000000006, 'shape': 95.0, 'pose': 52.0, 'size': 1.04, 'size_raw': (186, 186), 'canny_raw': 9.8, 'canny': 50.0}
    -31 58.81 discard face:     {'discard': ['tracking'], 'final': 63.440000000000005, 'shape': 96.0, 'pose': 53.0, 'size': 1.04, 'size_raw': (186, 185), 'canny_raw': 9.12, 'canny': 50.0}
    -32 61.31 discard face:     {'discard': ['tracking', 'pose'], 'final': 59.0, 'shape': 96.0, 'pose': 49.0, 'size': 1.0, 'size_raw': (223, 223), 'canny_raw': 7.6, 'canny': 50.0}
    -33 72.57 discard face:     {'discard': ['pose'], 'final': 54.0, 'shape': 70.0, 'pose': 49.0, 'size': 1.0, 'size_raw': (223, 223), 'canny_raw': 7.72, 'canny': 50.0}
    #6 78.83 good    face:      {'discard': [], 'final': 64.31, 'shape': 93.0, 'pose': 51.0, 'size': 1.09, 'size_raw': (267, 268), 'canny_raw': 6.26, 'canny': 50.0}
    #7 82.58 good    face:      {'discard': [], 'final': 68.67, 'shape': 94.0, 'pose': 57.99999999999999, 'size': 1.09, 'size_raw': (267, 267), 'canny_raw': 8.15, 'canny': 50.0}
    -34 83.83 discard face:     {'discard': ['tracking'], 'final': 60.0, 'shape': 74.0, 'pose': 61.0, 'size': 1.0, 'size_raw': (223, 223), 'canny_raw': 8.17, 'canny': 50.0}
    -35 85.08 discard face:     {'discard': ['tracking'], 'final': 64.0, 'shape': 94.0, 'pose': 61.0, 'size': 1.0, 'size_raw': (223, 223), 'canny_raw': 9.05, 'canny': 50.0}
    -36 90.09 discard face:     {'discard': ['shape'], 'final': 79.04, 'shape': 69.0, 'pose': 73.0, 'size': 1.04, 'size_raw': (186, 186), 'canny_raw': 12.29, 'canny': 89.0}
    #8 91.34 good    face:      {'discard': [], 'final': 75.0, 'shape': 71.0, 'pose': 68.0, 'size': 1.0, 'size_raw': (223, 223), 'canny_raw': 12.76, 'canny': 86.0}
    -37 92.59 discard face:     {'discard': ['shape'], 'final': 79.04, 'shape': 68.0, 'pose': 67.0, 'size': 1.04, 'size_raw': (186, 186), 'canny_raw': 11.38, 'canny': 93.0}
    -38 93.84 discard face:     {'discard': ['tracking'], 'final': 72.0, 'shape': 71.0, 'pose': 62.0, 'size': 1.0, 'size_raw': (223, 223), 'canny_raw': 12.35, 'canny': 88.0}
    -39 95.09 discard face:     {'discard': ['tracking', 'shape'], 'final': 75.92, 'shape': 68.0, 'pose': 72.0, 'size': 1.04, 'size_raw': (186, 186), 'canny_raw': 13.83, 'canny': 81.0}
    -40 116.37 discard face:    {'discard': ['size'], 'final': 45.0, 'shape': 95.0, 'pose': 71.0, 'size': 0.5, 'size_raw': (74, 75), 'canny_raw': 32.76, 'canny': 115.0}
    #9 131.38 good    face:     {'discard': [], 'final': 65.0, 'shape': 71.0, 'pose': 76.0, 'size': 1.0, 'size_raw': (223, 223), 'canny_raw': 6.39, 'canny': 50.0}
    #10 132.63 good    face:    {'discard': [], 'final': 79.57000000000001, 'shape': 97.0, 'pose': 79.0, 'size': 1.09, 'size_raw': (268, 268), 'canny_raw': 9.94, 'canny': 50.0}
    #11 133.88 good    face:    {'discard': [], 'final': 82.84, 'shape': 73.0, 'pose': 64.0, 'size': 1.09, 'size_raw': (267, 268), 'canny_raw': 11.14, 'canny': 94.0}
    -41 135.13 discard face:    {'discard': ['tracking'], 'final': 75.21000000000001, 'shape': 77.0, 'pose': 81.0, 'size': 1.09, 'size_raw': (267, 268), 'canny_raw': 9.87, 'canny': 50.0}
    -42 136.39 discard face:    {'discard': ['canny', 'pose'], 'final': 54.0, 'shape': 74.0, 'pose': 40.0, 'size': 1.08, 'size_raw': (321, 321), 'canny_raw': 2.93, 'canny': 50.0}
    -43 137.64 discard face:    {'discard': ['canny', 'pose'], 'final': 54.50000000000001, 'shape': 72.0, 'pose': 40.0, 'size': 1.09, 'size_raw': (268, 267), 'canny_raw': 4.74, 'canny': 50.0}
    #12 138.89 good    face:    {'discard': [], 'final': 64.31, 'shape': 74.0, 'pose': 57.99999999999999, 'size': 1.09, 'size_raw': (267, 267), 'canny_raw': 8.66, 'canny': 50.0}
    -44 140.14 discard face:    {'discard': ['tracking'], 'final': 75.21000000000001, 'shape': 78.0, 'pose': 78.0, 'size': 1.09, 'size_raw': (268, 268), 'canny_raw': 8.61, 'canny': 50.0}
    -45 141.39 discard face:    {'discard': ['canny', 'pose'], 'final': 58.24, 'shape': 93.0, 'pose': 44.0, 'size': 1.04, 'size_raw': (185, 186), 'canny_raw': 2.02, 'canny': 50.0}
    -46 142.64 discard face:    {'discard': ['pose'], 'final': 36.98, 'shape': 70.0, 'pose': 26.0, 'size': 0.86, 'size_raw': (155, 154), 'canny_raw': 8.15, 'canny': 50.0}

</details>
