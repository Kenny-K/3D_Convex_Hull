import numpy as np
import open3d
import argparse
from collections import defaultdict
import time

class Facet:
    def __init__(self, point3, inner) -> None:
        self.p1 = point3[0]   # registered point [4, ] :=  [point_num | xyz coor]
        self.p2 = point3[1]
        self.p3 = point3[2]
        self.inner = inner  # [3, ] center point coordinate
        self.norm = self.get_norm()
        self.visibility = True

    def get_norm(self):
        vec_1 = self.p3[1:] - self.p2[1:]
        vec_2 = self.p1[1:] - self.p3[1:]
        norm = np.cross(vec_1,vec_2)
        inner_vec = self.inner - self.p1[1:]
        if np.dot(norm, inner_vec) < 0:
            norm = -norm    # Force all norm vector to point towards inner side
        return norm

    def check_coplanar(self, pt):
        vec = pt[1:] - self.p1[1:]
        return np.dot(vec,self.norm) == 0

class ConvexHull:
    def __init__(self,init_points) -> None:
        self.points = init_points       # [V, 3]
        self.center_coor = np.mean(self.points,axis=0)[1:]
        assert not Facet(init_points[:3],self.center_coor).check_coplanar(init_points[3]), print('Coplanar points:\n', init_points)
        self.facets = [Facet(init_points[:3],self.center_coor),
                    Facet(init_points[[0,2,3]],self.center_coor),
                    Facet(init_points[[1,2,3]],self.center_coor),
                    Facet(init_points[[0,1,3]],self.center_coor)]
        self.edges = defaultdict(list)
        self.init_edges()

    def init_edges(self): 
        for facet in self.facets:
            self.edges[(int(min(facet.p1[0],facet.p2[0])), int(max([facet.p1[0],facet.p2[0]])))].append(facet)
            self.edges[(int(min(facet.p3[0],facet.p2[0])), int(max([facet.p3[0],facet.p2[0]])))].append(facet)
            self.edges[(int(min(facet.p1[0],facet.p3[0])), int(max([facet.p1[0],facet.p3[0]])))].append(facet)

    def get_edges(self):
        edge_list = []
        for key in self.edges.keys():
            edge_list.append(list(key))
        return np.array(edge_list,dtype=np.int64)
    
    def pt_in_hull(self, pt):
        # print("Point {} in the hull".format(pt[0]), end=' ')
        coor = pt[1:]
        for facet in self.facets:
            vec = coor - facet.p1[1:]
            if np.dot(vec, facet.norm) < 0:
                return False
        return True

    @staticmethod
    def is_visible(facet: Facet, pt):
        plane_pt_vec = pt[1:] - facet.p1[1:]
        norm = facet.norm
        return np.dot(plane_pt_vec, norm) > 0

    def increase(self, pt):
        self.points = np.concatenate([self.points, pt[np.newaxis, :]], axis=0)
        pt_in_hull = self.pt_in_hull((pt))
        # print(pt_in_hull)
        if not pt_in_hull:
            
            # self.center_coor = np.mean(self.points, axis=0)[1:]
        
            # update facet visibility
            for facet in self.facets:
                facet.visibility = self.is_visible(facet, pt)
                # print(facet.visibility)

            key_edges = []
            remove_edges = []
            for edge,facets in self.edges.items():
                assert len(facets) == 2, print("Single facet edge?")
                if not facets[0].visibility and not facets[1].visibility:
                    remove_edges.append(edge)
                elif (facets[0].visibility or facets[1].visibility) and not (facets[0].visibility and facets[1].visibility):
                    key_edges.append(edge)
            
            # print('Key edges: ',len(key_edges))
            # print('Remove edges: ',len(remove_edges))

            for edge_key in key_edges:
                new_facet = (Facet(point3=[pt,self.points[edge_key[0]],self.points[edge_key[1]]], inner=self.center_coor))
                self.facets.append(new_facet)
                self.edges[(int(min(pt[0],edge_key[0])), int(max(pt[0],edge_key[0])))].append(new_facet)
                self.edges[(int(min(pt[0],edge_key[1])), int(max(pt[0],edge_key[1])))].append(new_facet)
                self.edges[(int(min(edge_key[0],edge_key[1])), int(max(edge_key[0],edge_key[1])))].append(new_facet)
            # for key, value in self.edges.items():
            #     print(key, value)

            # remove invisible edges & facets
            self.facets = [facet for facet in self.facets if facet.visibility]
            # print('Convex Hull have {} facets'.format(len(self.facets)))
            for remove_edge in remove_edges:
                self.edges.pop(remove_edge)
            for edge, facets in self.edges.items():
                for facet in facets:
                    if facet not in self.facets:
                        facets.remove(facet)
            

    def visualize(self):
        # print('\n*******Start vis******')
        lineset = open3d.geometry.LineSet()
        points = self.points[:,1:]
        # print('Points shape: ', points.shape)
        lineset.points = open3d.utility.Vector3dVector(points)
        edges = self.get_edges()
        # print('Edges shape: ', edges.shape)
        lineset.lines = open3d.utility.Vector2iVector(edges)
        lineset.paint_uniform_color((1,0,0))
        open3d.visualization.draw_geometries([point_cloud,lineset])
        
    def get_valid_points(self):
        edges = self.get_edges()
        valid = np.unique(edges)
        return self.points[valid]

    def collide(self, other):
        for point in other.get_valid_points():
            if self.pt_in_hull(point):
                return True
        for point in self.get_valid_points():
            if other.pt_in_hull(point):
                return True
        return False


def joint_vis(C1: ConvexHull, C2: ConvexHull, collide=False):
    print('Collision: ', collide)
    LS1 = open3d.geometry.LineSet()
    pt1 = C1.points[:,1:]
    LS1.points = open3d.utility.Vector3dVector(pt1)
    e1 = C1.get_edges()
    LS1.lines = open3d.utility.Vector2iVector(e1)
    
    LS2 = open3d.geometry.LineSet()
    pt2 = C2.points[:,1:]
    LS2.points = open3d.utility.Vector3dVector(pt2)
    e2 = C2.get_edges()
    LS2.lines = open3d.utility.Vector2iVector(e2)

    if collide:
        LS1.paint_uniform_color((1,0,0))
        LS2.paint_uniform_color((1,0,0))
    else:
        LS1.paint_uniform_color((0,1,0))
        LS2.paint_uniform_color((0,1,0))
    
    open3d.visualization.draw_geometries([LS1,LS2])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--point_num', type=int, default=10000)
    parser.add_argument('--collision', action='store_true', default=False)
    parser.add_argument('--offset', type=float, default=2)
    args = parser.parse_args()

    points = np.random.randn(args.point_num,3)
    reg_points = np.concatenate([np.arange(points.shape[0],dtype=points.dtype)[:,np.newaxis],points],axis=1)
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points)
    
    begin = time.time()
    C = ConvexHull(reg_points[:4])
    hull_edges = C.get_edges()
    # C.visualize()
    for reg_point in reg_points[4:]:
        C.increase(reg_point)
    print(args.point_num, time.time()-begin)
    # C.visualize()
    

    if args.collision:
        points2 = points + args.offset
        reg_points2 = reg_points = np.concatenate([np.arange(points2.shape[0],dtype=points2.dtype)[:,np.newaxis],points2],axis=1)
        C2 = ConvexHull(reg_points2[:4])
        for reg_point in reg_points2[4:]:
            C2.increase(reg_point)
        hull_edges2 = C2.get_edges()
        point_cloud2 = open3d.geometry.PointCloud()
        point_cloud2.points = open3d.utility.Vector3dVector(points2)
        joint_vis(C, C2, C.collide(C2))

    
    
    


