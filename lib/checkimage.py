from PIL import Image

rel = Image.open('../dataset/deepglobe/train/104_sat.jpg')

mask = Image.open('../dataset/deepglobe/train/104_mask.png')

mask = mask.convert("P")

mask.putpalette([
    0, 0, 0, # black background
    255, 0, 0, # index 1 is red
    255, 255, 0, # index 2 is yellow
    255, 153, 0, # index 3 is orange
])

mask.show()
