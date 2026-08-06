print("Script started")
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import time
import os
import math
import sys

model_path = 'models/hand_landmarker.task'

hand_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1)
landmarker = HandLandmarker.create_from_options(options)

cap = None
for i in range(3):
    print("Checking camera setup...")
    cap = cv2.VideoCapture(i)
    time.sleep(2)
    if cap.isOpened():
        break
    else:
        cap.release()
if not cap or not cap.isOpened():
    print("Failed to open camera.")
    exit()
print("Camera opened successfully.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Tracking', 640, 480)
cv2.moveWindow('Hand Tracking', 50, 50)

mesh_window_x = 750
mesh_window_y = 50

import glob
model_path = None
faces = None
colors = None


obj_dir = 'obj_models'
obj_files = [f for f in os.listdir(obj_dir) if f.lower().endswith('.obj')] if os.path.exists(obj_dir) else []

print("[DEBUG] Model selection complete, proceeding to main loop setup.")
using_obj = False
using_stl = False
using_ply = False
using_cube = False
print("Checking for OBJ/STL/PLY/Cube model...")
if obj_files:
    print("OBJ file found: ", obj_files[0])
    try:
        import trimesh
        from PIL import Image
        obj_path = os.path.join(obj_dir, obj_files[0])
        mesh = trimesh.load(obj_path, force='mesh')
        vertices = mesh.vertices
        faces_idx = mesh.faces
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        center = (min_coords + max_coords) / 2
        scale = 1.0 / max(max_coords - min_coords)
        vertices = (vertices - center) * scale
        poly_faces = [vertices[face] for face in faces_idx]
        colors = None
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
            tex_img = mesh.visual.material.image
            uv = mesh.visual.uv
            tex_np = np.array(tex_img)
            face_colors = []
            for face in faces_idx:
                uv_face = uv[face]
                h, w = tex_np.shape[0], tex_np.shape[1]
                px = (uv_face[:, 0] * w).astype(int)
                py = ((1 - uv_face[:, 1]) * h).astype(int)
                px = np.clip(px, 0, w - 1)
                py = np.clip(py, 0, h - 1)
                samples = tex_np[py, px]
                avg_color = np.mean(samples[:, :3], axis=0) / 255.0
                face_colors.append(tuple(avg_color))
            colors = face_colors
        elif hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
            colors = [tuple(c) for c in colors]
        else:
            colors = [(0.7, 0.7, 0.7)] * len(poly_faces)
        faces = poly_faces
        using_obj = True
    except Exception as e:
        print(f"Error loading OBJ file: {e}")
        print("Falling back to STL/PLY/Cube.")
        using_obj = False
if not using_obj:
    stl_candidates = []
    if os.path.exists('models'):
        stl_candidates += [('models', f) for f in os.listdir('models') if f.lower().endswith('.stl')]
    if os.path.exists('stl_models'):
        stl_candidates += [('stl_models', f) for f in os.listdir('stl_models') if f.lower().endswith('.stl')]
    if stl_candidates:
        found_dir, found_file = stl_candidates[0]
        print("STL file found: ", found_file, "in", found_dir)
        from stl import mesh as npmesh
        model_path = os.path.join(found_dir, found_file)
        mesh_data = npmesh.Mesh.from_file(model_path)
        points = mesh_data.vectors.reshape(-1, 3)
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        center = (min_coords + max_coords) / 2
        scale = 1.0 / max(max_coords - min_coords)
        mesh_data.vectors = (mesh_data.vectors - center) * scale
        faces = mesh_data.vectors
        colors = 'cyan'
        using_stl = True
    else:
        ply_dir = 'ply_models'
        ply_files = [f for f in os.listdir(ply_dir) if f.lower().endswith('.ply')] if os.path.exists(ply_dir) else []
        if ply_files:
            print("PLY file found: ", ply_files[0])
            try:
                import trimesh
                ply_path = os.path.join(ply_dir, ply_files[0])
                mesh = trimesh.load(ply_path)
                vertices = mesh.vertices
                faces = mesh.faces
                min_coords = vertices.min(axis=0)
                max_coords = vertices.max(axis=0)
                center = (min_coords + max_coords) / 2
                scale = 1.0 / max(max_coords - min_coords)
                vertices = (vertices - center) * scale
                poly_faces = [vertices[face] for face in faces]
                colors = [(0.5, 0.0, 0.0)] * len(faces)
                faces = poly_faces
                using_ply = True
            except Exception as e:
                print(f"Error loading PLY file: {e}")
                print("Falling back to cube.")
                using_ply = False
        if not using_stl and not using_ply:
            print("No OBJ, STL or PLY found, using cube.")
            vertices = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ])
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[3], vertices[0], vertices[4], vertices[7]]
            ]
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
            using_cube = True

def make_axes_equal(ax):
    extents = np.array([getattr(ax, f"get_{dim}lim")() for dim in ('x', 'y', 'z')])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    ax.set_xlim(centers[0] - r, centers[0] + r)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] - r, centers[2] + r)

def zoom_axes(ax, factor):
    def shrink(lim):
        center = (lim[0] + lim[1]) / 2.0
        half = (lim[1] - lim[0]) / 2.0
        new_half = half * factor
        return (center - new_half, center + new_half)
    ax.set_xlim(shrink(ax.get_xlim()))
    ax.set_ylim(shrink(ax.get_ylim()))
    ax.set_zlim(shrink(ax.get_zlim()))

def render_solid_mesh(faces, colors=None, title='3D Mesh'):
    if faces is None or len(faces) == 0:
        print('[WARN] No faces to render.')
        return None
    plt.ion()
    fig = plt.figure(title)
    ax = fig.add_subplot(111, projection='3d')
    facecolors = None
    if isinstance(colors, list) and len(colors) == len(faces):
        facecolors = colors
    elif colors is not None:
        facecolors = [colors] * len(faces)
    else:
        facecolors = [(0.7, 0.7, 0.7)] * len(faces)

    poly = Poly3DCollection(faces, facecolors=facecolors, linewidths=0.15, edgecolors='k')
    poly.set_facecolor(facecolors)
    ax.add_collection3d(poly)
    all_verts = np.vstack([np.asarray(f).reshape(-1, 3) for f in faces])
    ax.auto_scale_xyz(all_verts[:,0], all_verts[:,1], all_verts[:,2])
    make_axes_equal(ax)
    ax.set_axis_off()
    fig.canvas.draw()
    plt.pause(0.001)
    return fig, ax

mesh_fig = None
mesh_ax = None
try:
    if faces is not None:
        print('[INFO] Rendering 3D mesh in a separate window...')
        res = render_solid_mesh(faces, colors)
        if res is not None:
            mesh_fig, mesh_ax = res
            print(f"[INFO] 3D renderer ready: fig={mesh_fig}, ax={mesh_ax}")
            try:
                manager = mesh_fig.canvas.manager
                if hasattr(manager, 'window'):
                    manager.window.wm_geometry(f'+{mesh_window_x}+{mesh_window_y}')
            except Exception as e:
                print(f"[WARN] Could not position 3D window: {e}")
            try:
                zoom_axes(mesh_ax, 0.65)
                base_limits = (
                    mesh_ax.get_xlim(), mesh_ax.get_ylim(), mesh_ax.get_zlim()
                )
            except Exception as e:
                print(f"[WARN] Could not apply initial zoom: {e}")
        else:
            mesh_fig = mesh_ax = None
except Exception as e:
    print(f'[ERROR] Failed to render mesh: {e}')

azim = mesh_ax.azim if mesh_ax is not None and hasattr(mesh_ax, 'azim') else -60
elev = mesh_ax.elev if mesh_ax is not None and hasattr(mesh_ax, 'elev') else 30
prev_hand_x = None
prev_hand_y = None
initial_pinch = None
base_limits = None
rotate_sens = 720.0
zoom_sens = 1.8


frame_count = 0
print("Entering main loop...")
print("[DEBUG] Entered main loop, starting frame read loop.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from camera.")
        break
    else:
        print("[INFO] Frame read from camera.")
    mouse_control = False
    mouse_x, mouse_y = None, None
    error_msgs = []
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[ERROR] cvtColor failed: {e}")
        break
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    except Exception as e:
        print(f"[ERROR] MediaPipe image creation failed: {e}")
        break
    try:
        hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_count)
    except Exception as e:
        print(f"[ERROR] MediaPipe hand detection failed: {e}")
        break
    if hand_landmarker_result.hand_landmarks:
        hand_landmarks = hand_landmarker_result.hand_landmarks[0]
        index_tip = hand_landmarks[8]
        thumb_tip = hand_landmarks[4]
        mouse_control = True
        x = int(index_tip.x * frame.shape[1])
        y = int(index_tip.y * frame.shape[0])
        tx = int(thumb_tip.x * frame.shape[1])
        ty = int(thumb_tip.y * frame.shape[0])
        mouse_x, mouse_y = x, y
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        cv2.circle(frame, (tx, ty), 10, (255, 0, 0), -1)
        cv2.line(frame, (x, y), (tx, ty), (255, 255, 0), 2)
    else:
        error_msgs.append("[WARN] No hand detected in frame.")
        prev_hand_x = None
        prev_hand_y = None
        initial_pinch = None

    if mouse_control and mouse_x is not None and mouse_y is not None:
        try:
            if mesh_ax is not None:
                nx = index_tip.x
                ny = index_tip.y
                if prev_hand_x is None:
                    prev_hand_x = nx
                    prev_hand_y = ny
                dx = nx - prev_hand_x
                dy = ny - prev_hand_y
                prev_hand_x = nx
                prev_hand_y = ny

                azim += dx * rotate_sens
                elev += dy * rotate_sens * 0.5
                mesh_ax.view_init(elev, azim)
                print(f"[DEBUG] 3D view updated: azim={azim:.1f}, elev={elev:.1f}")

                try:
                    thumb_tip = hand_landmarks[4]
                    pinch = math.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2 + (getattr(index_tip, 'z', 0) - getattr(thumb_tip, 'z', 0)) ** 2)
                except Exception:
                    pinch = None

                scale = 1.0
                if pinch is not None:
                    if initial_pinch is None:
                        initial_pinch = pinch
                        base_limits = (
                            mesh_ax.get_xlim(), mesh_ax.get_ylim(), mesh_ax.get_zlim()
                        )
                    else:
                        if pinch <= 0:
                            pinch = 1e-6
                        scale = 1.0 + (initial_pinch - pinch) * zoom_sens
                        scale = max(0.25, min(3.0, scale))
                        xlim, ylim, zlim = base_limits
                        def scale_limits(lim):
                            c = (lim[0] + lim[1]) / 2.0
                            half = (lim[1] - lim[0]) / 2.0
                            new_half = half * scale
                            return (c - new_half, c + new_half)
                        mesh_ax.set_xlim(scale_limits(xlim))
                        mesh_ax.set_ylim(scale_limits(ylim))
                        mesh_ax.set_zlim(scale_limits(zlim))

                if pinch is not None:
                    line_width = int(np.clip(2 + abs(scale - 1.0) * 8, 2, 12))
                    cv2.line(frame, (x, y), (tx, ty), (0, 255, 255), line_width)
                else:
                    cv2.line(frame, (x, y), (tx, ty), (0, 255, 255), 2)

                try:
                    mesh_fig.canvas.draw()
                    mesh_fig.canvas.flush_events()
                except Exception:
                    plt.pause(0.001)
        except Exception as e:
            print(f"[ERROR] 3D control update failed: {e}")

    if error_msgs:
        for msg in error_msgs:
            print(msg)

    print("[DEBUG] About to call cv2.imshow")
    try:
        cv2.imshow('Hand Tracking', frame)
        print("[DEBUG] cv2.imshow called successfully")
    except Exception as e:
        print(f"[ERROR] cv2.imshow failed: {e}")
        break
    print("[DEBUG] About to call cv2.waitKey")
    try:
        key = cv2.waitKey(1)
        print("[DEBUG] cv2.waitKey called successfully")
    except Exception as e:
        print(f"[ERROR] cv2.waitKey failed: {e}")
        break
    if key & 0xFF == ord('q'):
        print("[INFO] Quit requested by user.")
        break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
landmarker.close()
print("Hand tracking and 3D control stopped.")
