import argparse
import copy

import numpy as np
import torch
import trimesh
from tqdm import tqdm
from src import (Tuple)

def load_poses(path):
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = torch.from_numpy(c2w).float()
        poses.append(c2w)
    return poses

def cull_mesh(input_mesh,traj,output_mesh,intrin:Tuple):
    """
        input_mesh: str
        traj: str or  List(torch.tensor)
        output_mesh: str
        intrin:Tuple(H,W,fx,fy,cx,cy)
    """
    H, W, fx, fy, cx, cy = intrin
    if type(traj) == str:
        poses = load_poses(traj)
    else:
        poses = []
        for pose in traj:
            c2w = pose.clone()
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            poses.append(c2w)

    n_imgs = len(poses)
    mesh = trimesh.load(input_mesh, process=False)
    pc = mesh.vertices
    faces = mesh.faces

    # delete mesh vertices that are not inside any camera's viewing frustum
    whole_mask = np.ones(pc.shape[0]).astype(bool)
    for i in tqdm(range(0, n_imgs, 1)):
        c2w = poses[i]
        points = pc.copy()
        points = torch.from_numpy(points).cuda()
        w2c = c2w.inverse()
        K = torch.from_numpy(
            np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
        ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
        homo_points = torch.cat(
            [points, ones], dim=1).reshape(-1, 4, 1).cuda().float()
        cam_cord_homo = w2c@homo_points
        cam_cord = cam_cord_homo[:, :3]

        cam_cord[:, 0] *= -1
        uv = K.float()@cam_cord.float()
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.float().squeeze(-1).cpu().numpy()
        edge = 0
        mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W -
                                                   edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)
        whole_mask &= ~mask
    pc = mesh.vertices
    faces = mesh.faces
    face_mask = whole_mask[mesh.faces].all(axis=1)
    unseen_mesh = copy.deepcopy(mesh)
    mesh.update_faces(~face_mask)
    mesh.export(output_mesh)
    unseen_mesh.update_faces(face_mask)
    unseen_pcd = unseen_mesh.vertices
    np.save(output_mesh.replace('.ply', '_pc_unseen.npy'), unseen_pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments to cull the mesh.'
    )

    parser.add_argument('--input_mesh', type=str,
                        help='path to the mesh to be culled')
    parser.add_argument('--traj', type=str,  help='path to the trajectory')
    parser.add_argument('--output_mesh', type=str,  help='path to the output mesh')
    args = parser.parse_args()
    intrin = (680.,1200.,600.,600.,599.5,339.5)
    cull_mesh(input_mesh=args.input_mesh,traj=args.traj,output_mesh=args.output_mesh,intrin=intrin)