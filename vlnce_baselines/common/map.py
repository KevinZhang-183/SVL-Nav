import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import rotate
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
import networkx as nx
import copy


def points_to_occ_map_centered(xz_points, voxel_size=0.05, map_size=(400, 400)):

    H, W = map_size
    image = np.zeros((H, W), dtype=np.uint8)

    center = np.array([W // 2, H // 2])

    pixel_coords = (xz_points / voxel_size).astype(int) + center

    mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H)
    )
    pixel_coords = pixel_coords[mask]

    for x, z in pixel_coords:
        image[z, x] = 255 

    return image

def polar_sample(num_angles=60, num_radii=20, max_radius=3.0):

    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    radii = np.linspace(0, max_radius, num_radii + 1)[1:] 
    points = []
    for r in radii:
        for theta in angles:
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            points.append([x, z])
    return np.array(points)

def fill_points_to_shape(image, min_area=30):
    kernel = np.ones((7, 7), np.uint8) 
    dilated = cv2.dilate(image, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    filtered = np.zeros_like(image)
    for i in range(1, num_labels): 
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(image)

    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    return filled

def smooth_edges(image, blur_kernel=(75,75), binary_thresh=127):

    blurred = cv2.GaussianBlur(image, blur_kernel, 0)
    
    _, smoothed = cv2.threshold(blurred, binary_thresh, 255, cv2.THRESH_BINARY)
    
    return smoothed

def get_max_region(img):
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        largest_component = np.zeros_like(binary)
    else:
        largest_contour = max(contours, key=cv2.contourArea)

        largest_component = np.zeros_like(binary)
        cv2.drawContours(largest_component, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    return largest_component

def find_endpoints_and_branchpoints(binary_image):
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)

    binary = (binary_image == 255).astype(np.uint8)

    neighbor_count = convolve(binary, kernel, mode='constant')

    ys, xs = np.where(binary == 1)

    endpoints = []
    branchpoints = []

    for y, x in zip(ys, xs):
        n = neighbor_count[y, x]
        if n == 1:
            endpoints.append((x, y)) 
        # elif n >= 3:
        #     branchpoints.append((x, y))

    return endpoints, branchpoints

def merge_close_points(points, eps=5.0):
    if len(points) == 0:
        return points

    merged_points = []
    visited = set()

    for i, (y1, x1) in enumerate(points):
        if i in visited:
            continue

        close_points = []
        for j, (y2, x2) in enumerate(points):
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist < eps:
                close_points.append([y2, x2])
                visited.add(j)

        if close_points:
            avg_y = int(np.mean([p[0] for p in close_points]))
            avg_x = int(np.mean([p[1] for p in close_points]))
            merged_points.append([avg_y, avg_x])

    return np.array(merged_points)

def pixel_to_world_coords(pixel_coords, voxel_size=0.05, map_size=(400, 400)):

    H, W = map_size
    center = np.array([W // 2, H // 2])
    offset = pixel_coords - center
    world_coords = offset * voxel_size 
    return world_coords

def compute_relative_positions(points, self_pos):

    self_x, self_z = self_pos[0], self_pos[2] if len(self_pos) == 3 else self_pos[1]
    
    relative_coords = points - np.array([self_x, self_z])
    delta_x, delta_z = relative_coords[:, 0], relative_coords[:, 1]
    
    distances = np.sqrt(delta_x**2 + delta_z**2)
    
    angles_rad = np.arctan2(delta_x, -delta_z) 
    angles_rad = np.where(angles_rad < 0, angles_rad + 2 * np.pi, angles_rad) 
    
    angles_deg = np.degrees(angles_rad)

    angles_rad = 2 * np.pi - angles_rad
    angles_deg = 360 - angles_deg
    
    return {
        "relative_coords": relative_coords,
        "distances": distances,
        "angles_rad": angles_rad,
        "angles_deg": angles_deg
    }

def filter_close_points(points, min_dist=1.0):
    selected = []
    for p in points:
        if np.linalg.norm(p - np.array([0,0])) > min_dist:
            selected.append(p)
    return np.array(selected)



def filter_skeleton_within_radius(skeleton, origin_pixel, max_distance_m=1.5, voxel_size=0.05):

    max_dist_pixel = int(max_distance_m / voxel_size)
    H, W = skeleton.shape
    Y, X = np.ogrid[:H, :W]

    cx, cy = origin_pixel
    dist_sq = (X - cx)**2 + (Y - cy)**2
    mask = dist_sq <= max_dist_pixel**2

    skeleton_filtered = np.zeros_like(skeleton)
    skeleton_filtered[mask] = skeleton[mask]
    skeleton_filtered[skeleton_filtered != 255] = 0  
    return skeleton_filtered

def get_structure_wp(pcd, position=[0,0,0], voxel_size=0.03, map_size=(1024,1736), clamp_dist=(1,2)):
    # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    try:
        points = np.asarray(pcd.points)
    except:
        points = pcd

    wall_mask = (points[:, 1] < -0)# & (points[:, 1] > -1.6)
    wall_points = points[wall_mask]

    xz_points = wall_points[:, [0, 2]] 
    points = polar_sample(max_radius=1)
    xz_points = np.concatenate([xz_points, points])

    occ_map = points_to_occ_map_centered(xz_points, voxel_size=voxel_size, map_size=map_size)
    filled_contour = fill_points_to_shape(occ_map)
    filled_contour = smooth_edges(filled_contour, blur_kernel=(75,75))
    filled_contour = get_max_region(filled_contour)
    navi_area = copy.deepcopy(filled_contour)

    skeleton = skeletonize(filled_contour // 255).astype(np.uint8) * 255
    # filled_contour[skeleton == 255] = 128

    skeleton_filter = filter_skeleton_within_radius(skeleton, (int(map_size[1]/2), int(map_size[0]/2)), clamp_dist[1], voxel_size)
    if skeleton_filter.sum() == 0:
        skeleton_filter = skeleton
    filled_contour[skeleton == 255] = 128

    end_points, branch_points = find_endpoints_and_branchpoints(skeleton_filter)
    merged_wp = end_points+branch_points
    merged_wp = merge_close_points(end_points+branch_points, eps=10.0)

    wp_world = pixel_to_world_coords(merged_wp, voxel_size=voxel_size, map_size=map_size)
    wp_world = filter_close_points(wp_world, 1)

    wp = compute_relative_positions(wp_world, position)

    for point in merged_wp:
        cv2.circle(filled_contour, (point[0], point[1]), radius=10, color=128, thickness=-1)
    cv2.circle(filled_contour, (int(map_size[1]/2), int(map_size[0]/2)), radius=10, color=64, thickness=-1)
    # Image.fromarray(filled_contour).save("ego_map.png")
    # Image.fromarray(occ_map).save("occ_map.png")
    # Image.fromarray(skeleton).save("skeleton.png")
    # Image.fromarray(skeleton_filter).save("skeleton_filter.png")
    # Image.fromarray(navi_area).save("navi_area.png")

    return wp['angles_rad'], wp['distances'], navi_area


if __name__ == "__main__":
    # Optional: only needed for local debugging with a .ply file; not required for get_structure_wp at runtime.
    import open3d as o3d

    pcd = o3d.io.read_point_cloud("wp.ply")
    wp_world = get_structure_wp(pcd)
