# Decision and scope log

One line per decision: date, what was decided, and why. Newest at the bottom.
This log feeds the methods and reflection sections of the final report.

| Date | Decision | Why |
| --- | --- | --- |
| 17 Nov | The app runs as a local web app: the FastAPI server runs on the user's machine and serves the page and the API | The model needs the GPU in our laptops; no hosting cost; graders start it from the README (M12) |
| 17 Nov | Clicks first (M4), brush strokes later as a Should (S4) | Agreed with the lecturer: once clicks work, strokes and boxes are an easy extension |
| 17 Nov | No DICOM loading | Agreed with the lecturer: not needed; DICOM can be converted to NIfTI with dcm2niix |
| 17 Nov | SAM2 installed from Meta's GitHub, not the PyPI "sam2" package | The PyPI package is a third-party upload, not Meta's; we run what we cite |
| 17 Nov | Every volume is reoriented to RAS on load (nibabel as_closest_canonical); export maps masks back to the original affine | Viewer and model always see the same axis directions; one reverse step at export (M7) |
| 22 Nov | The volume is sent to the browser as int16 when the values fit, otherwise float32; /info says which ("dtype") | The Ernie T1 has values up to 65535 with decimals, which int16 cannot hold. Changes the "16-bit integers" line in the scoping document |
| 22 Nov | Until /data exists, the front end loads frontend/dev/<name>.bin written by frontend/tools/make_dev_volume.py | The WebGL view can be built without waiting for the backend; switching to /data needs no front-end change |
| 22 Nov | The axial view keeps the orientation of the backend's PNG slices (x to the right, anterior at the top) | Same picture as before; checking left/right against ITK-SNAP is planned for week 2 (M1) |
