import numpy as np
import math
from PIL import Image, ImageDraw

minimap = Image.open("assets\\images\\Minimap.png")
minimap_drawing = ImageDraw.Draw(minimap)
heightmap : np.ndarray = np.load("data\\heightmap.npy")

target_x, target_y = (1104, 1041)
targetHeight = heightmap[target_y, target_x]

heightLoss = 1.5 # Higher values = closer markers
endHeight_threshold = 0.05 # Lower values = fewer but more accurate pullout points
glider_pullout_height = 100 # Higher Values = furthur markers

validPoints = 0
pixelDistances = []

for y, row in enumerate(heightmap):
    for x, pixel_height in enumerate(row):
        if np.isnan(pixel_height):
            continue

        # Get the distance from the current pixel to the target pixel
        pixelDistance = math.sqrt((target_x - x)**2 + (target_y - y)**2)

        endHeight = pixel_height + glider_pullout_height - pixelDistance * heightLoss

        if abs(endHeight - targetHeight) <= endHeight_threshold:
            minimap_drawing.circle((x, y), 3, fill="red", outline="black", width=1)
            validPoints += 1
            pixelDistances.append(pixelDistance)
            


print(f"\nFound {validPoints} valid pullout points\n")

print(f"Min distance: {min(pixelDistances):.2f}")
print(f"Max distance: {max(pixelDistances):.2f}")
print(f"Mean distance: {np.mean(pixelDistances):.2f}")
print(f"Median distance: {np.median(pixelDistances):.2f}")
print(f"Std deviation: {np.std(pixelDistances):.2f}\n")

# minimap.show()
minimap.save("assets\\saves\\dropmap_save.png")