Staff extraction using a Visual Transformer.

Using the cvcmuscima dataset available here:

https://pages.cvc.uab.es/cvcmuscima/index_database.html

this trains a visual transformer to detect staves. Two versions are available:

- vit.py is the raw code,
- light.py uses lightning ai 

In case you wonder, this works really well! Above 95% precision and recall on the pixels of the staff lines themselves.

Issue with opencv-python:
# sudo apt-get istall fonts-dejavu
# ln -s /usr/share/fonts/truetype/dejavu \
  ~/.venv/lib/python3.12/site-packages/cv2/qt/fonts