import numpy as np
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import cv2
import open3d as o3d
import sys
from gt import LineMesh
import argparse


navyblue = [0.0, 0.0, 0.5]
tan = [0.82, 0.71, 0.55]
output_list = ['00017161_6d67f7e9138241b9ac13beef_trimesh_005_9',
    '00026810_ba32175d3cc1445cbad08b2b_trimesh_004_15',
    '00037132_48846ad2b02a4eb696f12645_trimesh_000_16',
    '00070553_501f6288a67e415e9f4dab97_trimesh_000_22',
    '00082067_eb3e392a54bf015889ede4dd_trimesh_000_2',
    '00052220_06b1539d43ea426a999c2485_trimesh_000_6']

k = 0
parser = argparse.ArgumentParser()
parser.add_argument('--index', default=0, type=int)
args = parser.parse_args()
junctions_color = [[72/255.0,209/255.0,204/255.0],[0,0,0],[1.0,0,1.0]]

def custom_draw_geometry_with_rotation(spheres,pcd):
    def rotate_view(vs):
        global k
        # image = vs.capture_screen_float_buffer(False)
        # image = np.array(image).astype("float")
        # # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # image = cv2.resize(image, (640, 480))
        # plt.imsave("{}/ours/{:#04}.jpg".format(args.index,k),\
        #             np.asarray(image), dpi = 1)
        ctr = vs.get_view_control()
        ctr.rotate(10.0, 0.0)
        if k==208:
            sys.exit(0)
        k = k+ 1
        return False

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=960, left=0, top=0)
    for p in pcd:
        vis.add_geometry(p)
    for s in spheres:
        vis.add_geometry(s)
    vis.get_render_option().background_color = [255.0/255.0,255.0/255.0,255.0/255.0]
    vis.register_animation_callback(rotate_view)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    name = output_list[args.index]
    if not osp.isdir(osp.join(str(args.index),'ours')):
        os.mkdir(osp.join(str(args.index),'ours'))
    with open('resluts/results.json') as _:
        res_list = json.load(_)
    for r in res_list:
        if r['fname'].rstrip('.png') == name:
            fname = r['fname']
            obj_idx = fname[:8]
            junctions_pred = np.array(r['junctions_cc_refined'],dtype=np.float)
            junctions_label = np.array(r['junctions_label_pred'],dtype=int)
            idx_lines_for_junctions_visi = np.array(r['idx_lines_for_junctions_visi'])
            idx_lines_for_junctions_hidden = np.array(r['idx_lines_for_junctions_hidden'])
            idx_lines_for_junctions = idx_lines_for_junctions_visi
            idx_lines_for_junctions = np.concatenate([idx_lines_for_junctions_visi,idx_lines_for_junctions_hidden],axis=0)
            num_visi = len(idx_lines_for_junctions_visi)
            num_hidden = len(idx_lines_for_junctions_hidden)
            colors = []
            for x in range(num_visi):
                colors.append(navyblue)
            for x in range(num_hidden):
                colors.append(tan)
            spheres = []
            for i, junc in enumerate(junctions_pred):
                label = junctions_label[i]
                if label != 2:
                    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    label = junctions_label[i]
                    mesh_sphere.paint_uniform_color(junctions_color[label - 1])
                    mesh_sphere.translate(junc)
                    spheres.append(mesh_sphere)
            points = np.array(junctions_pred)
            line_mesh1 = LineMesh(points, idx_lines_for_junctions, colors, radius=0.005)
            line_mesh1_geoms = line_mesh1.cylinder_segments
            custom_draw_geometry_with_rotation(spheres, [*line_mesh1_geoms])


