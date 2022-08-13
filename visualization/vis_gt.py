import numpy as np
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import cv2
import open3d as o3d
import time
import sys
import argparse


output_list = ['00017161_6d67f7e9138241b9ac13beef_trimesh_005_9',
    '00026810_ba32175d3cc1445cbad08b2b_trimesh_004_15',
    '00037132_48846ad2b02a4eb696f12645_trimesh_000_16',
    '00070553_501f6288a67e415e9f4dab97_trimesh_000_22',
    '00082067_eb3e392a54bf015889ede4dd_trimesh_000_2',
    '00052220_06b1539d43ea426a999c2485_trimesh_000_6']

junctions_color = [[72/255.0,209/255.0,204/255.0],[0.0,0.0,0.0],[1.0,0,1.0]]

parser = argparse.ArgumentParser()
parser.add_argument('--index', default=0, type=int)
args = parser.parse_args()


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, -1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            #import pdb;pdb.set_trace()
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)



navyblue = [0.0, 0.0, 0.5]
tan = [0.82, 0.71, 0.55]
k = 0

def custom_draw_geometry_with_rotation(spheres,pcd):
    def rotate_view(vs):
        global k
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
    path = output_list[args.index]
    if not osp.isdir(str(args.index)):
        os.mkdir(str(args.index))
    if not osp.isdir(osp.join(str(args.index),'wireframe')):
        os.mkdir(osp.join(str(args.index),'wireframe'))
    obj_idx = path[:8]
    j_path = osp.join('outputs',obj_idx,path+'.json')
    with open(j_path) as _:
        data = json.load(_)
    junctions_cc = data['junctions_cc']
    junctions_label = np.array(data['junction_label'],dtype=int)
    lines_visible = np.array(data['lines_positive_visible'])
    lines_hidden = np.array(data['lines_positive_hidden'])
    lines = np.concatenate([lines_visible,lines_hidden],axis=0).tolist()
    num_visi = len(lines_visible)
    num_hidden = len(lines_hidden) 
    colors = []
    for x in range(num_visi):
        colors.append(navyblue)
    for x in range(num_hidden):
        colors.append(tan)
    spheres = []
    for i,junc in enumerate(junctions_cc):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        label = junctions_label[i]
        mesh_sphere.paint_uniform_color(junctions_color[label-1])
        mesh_sphere.translate(junc)
        spheres.append(mesh_sphere)

    points = np.array(junctions_cc)
    line_mesh1 = LineMesh(points, lines, colors, radius=0.005)
    line_mesh1_geoms = line_mesh1.cylinder_segments
    custom_draw_geometry_with_rotation(spheres,[*line_mesh1_geoms])
 

