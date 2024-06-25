import matplotlib.pyplot as plt

def visualize_bounding_box(image, bbox):
    x1, y1, x2, y2 = bbox
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2))
    plt.show()
