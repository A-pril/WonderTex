import torch
import pytorch3d
import numpy as np
import math

from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj, IO

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
	look_at_view_transform,
	FoVPerspectiveCameras, 
	FoVOrthographicCameras,
	AmbientLights,
	PointLights, 
	DirectionalLights, 
	Materials, 
	RasterizationSettings, 
	MeshRenderer, 
	MeshRasterizer,  
	TexturesUV
)

from .geometry import HardGeometryShader
from .shader import HardNChannelFlatShader


# Pytorch3D based renderering functions, managed in a class
# Render size is recommended to be the same as your latent view size
# DO NOT USE "bilinear" sampling when you are handling latents.
# Stable Diffusion has 4 latent channels so use channels=4

class UVProjection():
	def __init__(self, texture_size=96, render_size=64, sampling_mode="nearest", channels=3, device=None):
		self.channels = channels
		self.device = device or torch.device("cpu")
		self.lights = AmbientLights(ambient_color=((1.0,)*channels,), device=self.device)
		self.target_size = (texture_size,texture_size)
		self.render_size = render_size
		self.sampling_mode = sampling_mode


	# Load obj mesh, rescale the mesh to fit into the bounding box
	def load_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=False):
		mesh = load_objs_as_meshes([mesh_path], device=self.device)
		if auto_center:
			verts = mesh.verts_packed()
			max_bb = (verts - 0).max(0)[0]
			min_bb = (verts - 0).min(0)[0]
			# TODO: Update scale to the The hypotenuse of the aabb cube
			# scale = (max_bb - min_bb).max()/2
			# dxyz = max_bb - min_bb
			# temp = dxyz[0]**2+ dxyz[1]**2+dxyz[2]**2
			scale =  (max_bb - min_bb).max() # torch.sqrt(dxyz[0]**2+ dxyz[1]**2+dxyz[2]**2)
			center = (max_bb+min_bb) /2
			mesh.offset_verts_(-center)
			mesh.scale_verts_((scale_factor / float(scale)))		
		else:
			mesh.scale_verts_((scale_factor))

		if autouv or (mesh.textures is None):
			mesh = self.uv_unwrap(mesh) # refer to TEXture : xatlas
		self.mesh = mesh.to(self.device)


	def load_glb_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=False):
		import trimesh
		mesh = trimesh.load(mesh_path, force='mesh', process=True, maintain_order=True)
		verts = torch.tensor(mesh.vertices, dtype=torch.float32)
		faces = torch.tensor(mesh.faces, dtype=torch.int64)
		mesh = Meshes(verts=[verts], faces=[faces])
		if auto_center:
			verts = mesh.verts_packed()
			max_bb = (verts - 0).max(0)[0]
			min_bb = (verts - 0).min(0)[0]
			# scale = (max_bb - min_bb).max()/2 
			scale =  (max_bb - min_bb).max() 
			center = (max_bb+min_bb) /2
			mesh.offset_verts_(-center)
			mesh.scale_verts_((scale_factor / float(scale)))
		else:
			mesh.scale_verts_((scale_factor))
		if autouv or (mesh.textures is None):
			mesh = self.uv_unwrap(mesh)
		self.mesh = mesh.to(self.device)
	
	def load_gltf_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=False):
		from .geometry import preprocess_gltf
		mesh_path = preprocess_gltf(mesh_path)
		import trimesh
		mesh = trimesh.load(mesh_path, force='mesh', process=True, maintain_order=True)
		verts = torch.tensor(mesh.vertices, dtype=torch.float32)
		faces = torch.tensor(mesh.faces, dtype=torch.int64)
		mesh = Meshes(verts=[verts], faces=[faces])
		if auto_center:
			verts = mesh.verts_packed()
			max_bb = (verts - 0).max(0)[0]
			min_bb = (verts - 0).min(0)[0]
			# scale = (max_bb - min_bb).max()/2 
			scale =  (max_bb - min_bb).max() 
			center = (max_bb+min_bb) /2
			mesh.offset_verts_(-center)
			mesh.scale_verts_((scale_factor / float(scale)))
		else:
			mesh.scale_verts_((scale_factor))
		if autouv or (mesh.textures is None):
			mesh = self.uv_unwrap(mesh)
		self.mesh = mesh.to(self.device)


	# Save obj mesh
	def save_mesh(self, mesh_path, texture):
		save_obj(mesh_path, 
				self.mesh.verts_list()[0],
				self.mesh.faces_list()[0],
				verts_uvs= self.mesh.textures.verts_uvs_list()[0],
				faces_uvs= self.mesh.textures.faces_uvs_list()[0],
				texture_map=texture)

	# Code referred to TEXTure code (https://github.com/TEXTurePaper/TEXTurePaper.git)
	def uv_unwrap(self, mesh):
		verts_list = mesh.verts_list()[0]
		faces_list = mesh.faces_list()[0]


		import xatlas
		import numpy as np
		v_np = verts_list.cpu().numpy()
		f_np = faces_list.int().cpu().numpy()
		atlas = xatlas.Atlas()
		atlas.add_mesh(v_np, f_np)
		chart_options = xatlas.ChartOptions()
		chart_options.max_iterations = 4
		atlas.generate(chart_options=chart_options)
		vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

		vt = torch.from_numpy(vt_np.astype(np.float32)).type(verts_list.dtype).to(mesh.device)
		ft = torch.from_numpy(ft_np.astype(np.int64)).type(faces_list.dtype).to(mesh.device)

		new_map = torch.zeros(self.target_size+(self.channels,), device=mesh.device)
		new_tex = TexturesUV(
			[new_map], 
			[ft], 
			[vt], 
			sampling_mode=self.sampling_mode
			)

		mesh.textures = new_tex
		return mesh


	'''
		A functions that disconnect faces in the mesh according to
		its UV seams. The number of vertices are made equal to the
		number of unique vertices its UV layout, while the faces list
		is intact.
	'''
	def disconnect_faces(self, device):
		mesh = self.mesh
		verts_list = mesh.verts_list()
		faces_list = mesh.faces_list()
		verts_uvs_list = mesh.textures.verts_uvs_list()
		faces_uvs_list = mesh.textures.faces_uvs_list()
		packed_list = [v[f] for v,f in zip(verts_list, faces_list)]
		verts_disconnect_list = [
			torch.zeros(
				(verts_uvs_list[i].shape[0], 3), 
				dtype=verts_list[0].dtype, 
				device=verts_list[0].device
			) 
			for i in range(len(verts_list))]
		for i in range(len(verts_list)):
			verts_disconnect_list[i][faces_uvs_list] = packed_list[i]
		assert not mesh.has_verts_normals(), "Not implemented for vertex normals"
		self.mesh_d = Meshes(verts_disconnect_list, faces_uvs_list, mesh.textures)
		return self.mesh_d


	'''
		A function that construct a temp mesh for back-projection.
		Take a disconnected mesh and a rasterizer, the function calculates
		the projected faces as the UV, as use its original UV with pseudo
		z value as world space geometry.
	'''
	def construct_uv_mesh(self, device):
		mesh = self.mesh_d
		verts_list = mesh.verts_list()
		verts_uvs_list = mesh.textures.verts_uvs_list()
		# faces_list = [torch.flip(faces, [-1]) for faces in mesh.faces_list()]
		new_verts_list = []
		for i, (verts, verts_uv) in enumerate(zip(verts_list, verts_uvs_list)):
			verts = verts.clone()
			verts_uv = verts_uv.clone()
			verts[...,0:2] = verts_uv[...,:]
			verts = (verts - 0.5) * 2  # [-1, 1]
			verts[...,2] *= 1
			new_verts_list.append(verts)
		textures_uv = mesh.textures.clone()
		self.mesh_uv = Meshes(new_verts_list, mesh.faces_list(), textures_uv)
		return self.mesh_uv


	# Set texture for the current mesh.
	def set_texture_map(self, texture):
		# [1536, 1536, 4]
		new_map = texture.permute(1, 2, 0)
		new_map = new_map.to(self.device)

		new_tex = TexturesUV(
			[new_map], 
			self.mesh.textures.faces_uvs_padded().to(self.device), 
			self.mesh.textures.verts_uvs_padded().to(self.device), 
			sampling_mode=self.sampling_mode
			)
		self.mesh.textures = new_tex

	# Set the initial white texture
	def set_white_texture(self, channels=None):
		if not channels:
			channels = self.channels
		# [4, 1536, 1536]
		initial_texture = torch.ones((channels,) + self.target_size, device=self.device)
		self.set_texture_map(initial_texture)
		return initial_texture

	# Set the initial normal noise texture
	# No generator here for replication of the experiment result. Add one as you wish
	def set_noise_texture(self, channels=None):
		if not channels:
			channels = self.channels
		# [4, 1536, 1536]
		noise_texture = torch.normal(0, 1, (channels,) + self.target_size, device=self.device)
		self.set_texture_map(noise_texture)
		return noise_texture
	
	def select_new_pose(self, camera_poses, all_valid_faceids, centers=None, camera_distance=4.0, scale=((1.0, 1.0, 1.0),)):
		elev = torch.FloatTensor([pose[0] for pose in camera_poses])
		azim = torch.FloatTensor([pose[1] for pose in camera_poses])
		R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim, at=centers or ((0,0,0),), device=self.device)

		cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, scale_xyz=scale)
		
		camera_id = -1
		max_faces = -1
		for i in range(len(cameras)):
			rasterize = self.renderer.rasterizer(self.mesh_d, cameras=cameras[i])
			pix2face = rasterize.pix_to_face # [1, res, res, 1]

			valid_faceid = torch.unique(pix2face)
			mask = ~torch.isin(valid_faceid, all_valid_faceids)
			
			if mask[mask].shape[0] > max_faces:
				max_faces = mask[mask].shape[0]
				camera_id = i
			
		# print("### chosen camera: ", (elev[camera_id], azim[camera_id]))
		self.camera_poses.append((elev[camera_id].item(), azim[camera_id].item()))
			
		return R[camera_id].unsqueeze(0), T[camera_id].unsqueeze(0) # , camera_id
	
	def add_camera(self, R, T, scale):
		self.R = torch.cat([self.R, R], dim=0)
		self.T = torch.cat([self.T, T], dim=0)
		self.cameras = FoVOrthographicCameras(device=self.device, R=self.R, T=self.T, scale_xyz=scale or ((1,1,1),))

# 		self.calculate_visible_triangle_mask(camera_exist=True)
# 		self.calculate_tex_gradient(camera_exist=True)
# 
# 		_,_,_,cos_maps,_, _ = self.render_geometry(camera_exist=True)
# 		cos_maps.to(self.device)
# 		self.calculate_cos_angle_weights(cos_maps, camera_exist=True)


	# Set the cameras given the camera poses and centers
	def set_cameras(self, camera_poses, centers=None, camera_distance=2.7, scale=None):
		# the angular position along the latitudinal and longitudinal axes, respectively.
		elev = torch.FloatTensor([pose[0] for pose in camera_poses])
		azim = torch.FloatTensor([pose[1] for pose in camera_poses])
		R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim, at=centers or ((0,0,0),), device=self.device)
		if not hasattr(self, "R"):
			self.R = R
			self.T = T
		else:
			self.R = torch.cat([self.R, R], dim=0)
			self.T = torch.cat([self.T, T], dim=0)
		
		self.cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, scale_xyz=scale or ((1,1,1),))

	# Set the current 4 cameras and get the other cameras for inpaint
	def get_all_cameras(self, diff_angle=10, min_elev=-30, max_elev=60, eps=0.05, samples=5, 
					 	centers=None, camera_distance=2.7, scale=None):		
		self.visible_triangles = []
		self.invalid_verts = -1
		all_valid_faceids = self.calculate_visible_triangle_mask()

		while True:
			invalid_verts = self.get_invalid_verts(all_valid_faceids)
			# print("### invalid vert number: ", invalid_verts.shape[0])
			if self.invalid_verts != -1 and self.invalid_verts - invalid_verts.shape[0] <= 5:
				break
			else:
				self.invalid_verts = invalid_verts.shape[0]
			
			new_cameras = self.get_new_cameras(self.camera_poses, invalid_verts, 
							eps=eps, min_samples=samples, 
							min_elev=min_elev, max_elev=max_elev, diff_angle=diff_angle)
			if len(new_cameras) == 0:
				break
			R, T = self.select_new_pose(camera_poses=new_cameras, all_valid_faceids=all_valid_faceids, centers=centers, camera_distance=camera_distance, scale=scale)
			self.add_camera(R, T, scale=scale)

			new_faceids = self.calculate_visible_triangle_mask(camera_exist=True)
			all_valid_faceids = torch.unique(torch.cat((new_faceids, all_valid_faceids)))
		
		self.calculate_tex_gradient()
		_,_,_,cos_maps,_, _ = self.render_geometry()
		cos_maps.to(self.device)
		self.calculate_cos_angle_weights(cos_maps)

	# Set all necessary internal data for rendering and texture baking
	# Can be used to refresh after changing camera positions
	def set_cameras_and_render_settings(self, camera_poses, centers=None, camera_distance=2.7, render_size=None, 
									 scale=None, cal_cosmap=False):
		self.camera_poses = camera_poses
		self.set_cameras(camera_poses, centers, camera_distance, scale=scale)
		if render_size is None:
			render_size = self.render_size
		if not hasattr(self, "renderer"):
			self.setup_renderer(size=render_size)
		if not hasattr(self, "mesh_d"):
			self.disconnect_faces(device=self.device)
		if not hasattr(self, "mesh_uv"):
			self.construct_uv_mesh(device=self.device)
		# the valid face_id in screen world
		# use mesh_uv to get all valid face_ids
		self.uv_pix2face = self.get_valid_faces() # [1, 1024, 1024, 1]
		
		# self.calculate_visible_triangle_mask()
		# self.calculate_tex_gradient()
		# if cal_cosmap:
		# 	_,_,_,cos_maps,_, _ = self.render_geometry()
		# 	cos_maps.to(self.device)
		# 	self.calculate_cos_angle_weights(cos_maps)


	# Setup renderers for rendering
	# max faces per bin set to 30000 to avoid overflow in many test cases.
	# You can use default value to let pytorch3d handle that for you.
	def setup_renderer(self, size=64, blur=0.0, face_per_pix=1, perspective_correct=False, channels=None):
		if not channels:
			channels = self.channels

		self.raster_settings = RasterizationSettings(
			image_size=size,  # output image size 64*64
			blur_radius=blur,  # As we are rendering images for visualization purposes only 
			faces_per_pixel=face_per_pix, # we will set faces_per_pixel=1 and blur_radius=0.0.
			perspective_correct=perspective_correct,
			cull_backfaces=True,
			max_faces_per_bin=30000, # 可微分渲染器，一个像素使用30000个面进行blend
		)


		self.renderer = MeshRenderer(
			# rasterizer
			rasterizer=MeshRasterizer(
				cameras=self.cameras, 
				raster_settings=self.raster_settings,

			),
			# shader：Customized the original pytorch3d hard flat shader to support N channel flat shading
			shader=HardNChannelFlatShader(
				device=self.device, 
				cameras=self.cameras,
				lights=self.lights,
				channels=channels
				# materials=materials
			)
		)


	# Bake screen-space cosine weights to UV space
	# May be able to reimplement using the generic "bake_texture" function, but it works so leave it here for now
	@torch.enable_grad()
	def calculate_cos_angle_weights(self, cos_angles, camera_exist=False, fill=True, channels=None):
		from .voronoi import voronoi_solve
		if not channels:
			channels = self.channels

		if camera_exist:
			cos_maps = self.cos_maps
			camera = self.cameras[-1]
		else:
			cos_maps = []
			camera = self.cameras

		tmp_mesh = self.mesh.clone()
		for i in range(len(camera)):
			zero_map = torch.zeros(self.target_size+(channels,), device=self.device, requires_grad=True)
			optimizer = torch.optim.SGD([zero_map], lr=1, momentum=0)
			optimizer.zero_grad()
			zero_tex = TexturesUV([zero_map], self.mesh.textures.faces_uvs_padded(), self.mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
			tmp_mesh.textures = zero_tex

			images_predicted = self.renderer(tmp_mesh, cameras=camera[i], lights=self.lights)

			loss = torch.sum((cos_angles[i,:,:,0:1]**1 - images_predicted)**2)
			loss.backward()
			optimizer.step()

			if fill:
				zero_map = zero_map.detach() / (self.gradient_maps[i] + 1E-8)
				zero_map = voronoi_solve(zero_map, self.gradient_maps[i][...,0], self.device)
			else:
				zero_map = zero_map.detach() / (self.gradient_maps[i]+1E-8)
			cos_maps.append(zero_map)
			
		self.cos_maps = cos_maps

		
	# Get geometric info from fragment shader
	# Can be used for generating conditioning image and cosine weights
	# Returns some information you may not need, remember to release them for memory saving
	@torch.no_grad()
	def render_geometry(self, camera_exist= False, image_size=None):
		if image_size:
			size = self.renderer.rasterizer.raster_settings.image_size
			self.renderer.rasterizer.raster_settings.image_size = image_size
		shader = self.renderer.shader
		self.renderer.shader = HardGeometryShader(device=self.device, cameras=self.cameras[0], lights=self.lights)
		tmp_mesh = self.mesh.clone()
		
		if camera_exist:
			camera = self.cameras[-1]
		else:
			camera = self.cameras

		# (N, H, W, 4)
		verts, normals, depths, cos_angles, texels, fragments = self.renderer(tmp_mesh.extend(len(camera)), cameras=camera, lights=self.lights)
		self.renderer.shader = shader

		if image_size:
			self.renderer.rasterizer.raster_settings.image_size = size

		return verts, normals, depths, cos_angles, texels, fragments


	# Project world normal to view space and normalize
	@torch.no_grad()
	def decode_view_normal(self, normals):
		w2v_mat = self.cameras.get_full_projection_transform()
		# (N, H, W, 3)
		normals_view = torch.clone(normals)[:,:,:,0:3]
		# (N, H*W, 3)
		normals_view = normals_view.reshape(normals_view.shape[0], -1, 3)
		normals_view = w2v_mat.transform_normals(normals_view)
		normals_view = normals_view.reshape(normals.shape[0:3]+(3,))
  
		normals_view[:,:,:,2] *= -1 # z轴取反
		# [(x,y,z)+1]*alpha/2 + (0.5,0.5,1)*(1-alpha)
		normals = (normals_view[...,0:3]+1) * normals[...,3:] / 2 + torch.FloatTensor(((((0.5,0.5,1))))).to(self.device) * (1 - normals[...,3:])

		normals = normals.clamp(0, 1)
		return normals


	# Normalize absolute depth to inverse depth
	@torch.no_grad()
	def decode_normalized_depth(self, depths, batched_norm=False):
		# (N, H, W, 2) --> (N, H, W) * 2
		# 移除指定维后，返回一个元组，包含了沿着指定维切片后的各个切片
		view_z, mask = depths.unbind(-1)
		view_z = view_z * mask + 100 * (1-mask)
		inv_z = 1 / view_z
		inv_z_min = inv_z * mask + 100 * (1-mask)
		if not batched_norm:
			max_ = torch.max(inv_z, 1, keepdim=True)
			max_ = torch.max(max_[0], 2, keepdim=True)[0]

			min_ = torch.min(inv_z_min, 1, keepdim=True)
			min_ = torch.min(min_[0], 2, keepdim=True)[0]
		else:
			max_ = torch.max(inv_z)
			min_ = torch.min(inv_z_min)
		inv_z = (inv_z - min_) / (max_ - min_)
		inv_z = inv_z.clamp(0,1)
		inv_z = inv_z[...,None].repeat(1,1,1,3)

		return inv_z
	
	def get_new_cameras(self, cameras, invalid_verts, eps=0.05, min_samples=5, 
					 	min_elev=-30, max_elev=60, diff_angle=5):
		from sklearn.cluster import DBSCAN

		# DBSCAN to cluster
		dbscan = DBSCAN(eps=eps, min_samples=min_samples) # (eps=0.15, min_samples=10)
		labels = dbscan.fit_predict(invalid_verts)

		unique_labels = set(labels)
		unique_labels.discard(-1)  # 排除噪声点
		new_cams = []
		all_cams = []

		for label in unique_labels:
			cur_pos = []
			if label == -1:
				continue
			cluster_points = invalid_verts[labels == label]

			# use the center of cluser and calculate the elev and azith
			# then get some random pose next to the calculated

			# center = np.mean(cluster_points, axis=0)
			# all_cams.append(center)
			# azim = math.atan2(center[0], -center[2]) * (180 / math.pi)	
			# d = math.sqrt(center[0]**2+center[2]**2)
			# elev = math.atan2(center[1], d) * (180 / math.pi)
			# cur_pos.append((elev, azim))
# 			for _ in range(5):
# 				ele_rand = elev + np.random.uniform(-5, 5)
# 				azim_rand = azim + np.random.uniform(-5, 5)
# 				cur_pos.append((ele_rand, azim_rand))
			
			# use random points of the cluser
			rand_indices = np.random.choice(cluster_points.shape[0], size=3, replace=False)
			rand_samples = cluster_points[rand_indices]
			if len(all_cams) == 0:
				all_cams=rand_samples
			else:
				all_cams = np.concatenate((all_cams, rand_samples), axis=0)
			for p in rand_samples:
				azim = math.atan2(p[0], -p[2]) * (180 / math.pi)	
				d = math.sqrt(p[0]**2+p[2]**2)
				elev = math.atan2(p[1], d) * (180 / math.pi)
				cur_pos.append((elev, azim))

			cur_pos = [
				(max(min_elev, min(elev, max_elev)), azim + 360 if azim < 0 else azim)
				for elev, azim in cur_pos
			]
			
			# filter poses which are used 
			for (elev, azim) in cur_pos:
				pose_exist = False
				for pose in cameras:
					elev_diff = abs(pose[0] - elev)
					azith_diff = abs(pose[1] - azim)
					if elev_diff + azith_diff < diff_angle:
						pose_exist = True
						break
				if not pose_exist:
					new_cams.append((elev, azim))
# 		# visualize
# 		centers = np.array(all_cams)
# 		import matplotlib.pyplot as plt
# 		fig = plt.figure()
# 
# 
# 		ax = fig.add_subplot(111, projection='3d')
# 
# 		unique_labels = np.unique(labels)
# 		colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
# 
# 		# TODO: save cluster result
# 		for label, color in zip(unique_labels, colors):
# 			if label == -1:
# 				color = [0, 0, 0, 1]  # black for noise points
# 
# 			class_member_mask = (labels == label)
# 			xyz = invalid_verts[class_member_mask]
# 			ax.scatter(xyz[:, 0], -xyz[:, 2], xyz[:, 1],  c=[color], label=f'Cluster {label}', s=3)
# 		
# 		ax.scatter(centers[:,0], -centers[:, 2], centers[:, 1], c='r', marker='o', s=50, label='Centers')
# 		ax.set_xlabel('X axis')
# 		ax.set_ylabel('Y axis')
# 		ax.set_zlabel('Z axis')
# 		ax.legend()
# 		plt.savefig(f"3d_clustering_result_{invalid_verts.shape[0]}.png", dpi=300)

		return new_cams

	# all face_ids and valid face_ids  
	# --> get invalid face_ids --> get invalid vert_ids
	def get_invalid_verts(self, all_valid_faceids):
		verts_list = self.mesh_d.verts_list()
		faces_list = self.mesh_d.faces_list()
		
		mask = ~torch.isin(self.uv_pix2face[0], all_valid_faceids)
		all_invalid_faceid = torch.unique(self.uv_pix2face[0][mask])	
		all_invalid_vertid = torch.unique(faces_list[0][all_invalid_faceid])
		all_invalid_verts = torch.unique(verts_list[0][all_invalid_vertid], dim=0)
		all_invalid_verts = all_invalid_verts.cpu().numpy()

		return all_invalid_verts

	# Use mesh_uv to get all valid face_ids
	@torch.no_grad()
	def get_valid_faces(self):
		if not hasattr(self, "mesh_uv"):
			self.construct_uv_mesh(self.device)

		raster_settings = RasterizationSettings(
			image_size=self.target_size, 
			blur_radius=0, 
			faces_per_pixel=1,
			perspective_correct=False,
			cull_backfaces=False,
			max_faces_per_bin=30000,
		)

		R, T = look_at_view_transform(dist=2, elev=0, azim=0)
		cameras = FoVOrthographicCameras(device=self.device, R=R, T=T)

		rasterizer = MeshRasterizer(
			cameras=cameras, 
			raster_settings=raster_settings
		)
		uv_pix2face = rasterizer(self.mesh_uv).pix_to_face # mesh_uv faceid == mesh_d faceid

		return uv_pix2face

	# Multiple screen pixels could pass gradient to a same texel
	# We can precalculate this gradient strength and use it to normalize gradients when we bake textures
	@torch.enable_grad()
	def calculate_tex_gradient(self, camera_exist=False, channels=None):
		if not channels:
			channels = self.channels
		tmp_mesh = self.mesh.clone()

		if camera_exist:
			gradient_maps = self.gradient_maps
			cameras = self.cameras[-1]
		else:
			gradient_maps = []
			cameras = self.cameras

		for i in range(len(cameras)):
			zero_map = torch.zeros(self.target_size+(channels,), device=self.device, requires_grad=True)
			optimizer = torch.optim.SGD([zero_map], lr=1, momentum=0)
			optimizer.zero_grad()
			zero_tex = TexturesUV([zero_map], self.mesh.textures.faces_uvs_padded(), self.mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
			tmp_mesh.textures = zero_tex
			images_predicted = self.renderer(tmp_mesh, cameras=cameras[i], lights=self.lights)
			loss = torch.sum((1 - images_predicted)**2)
			loss.backward()
			optimizer.step()

			gradient_maps.append(zero_map.detach())

		self.gradient_maps = gradient_maps


	# Get face ids from each view and the UV space masks of triangles visible in each view
	# Return the face ids
	# First get face ids from each view, then filter pixels on UV space to generate masks
	@torch.no_grad()
	def calculate_visible_triangle_mask(self, camera_exist=False, channels=None, image_size=(512,512)):
		if not channels:
			channels = self.channels

		pix2face_list = []
		if camera_exist:
			cameras = self.cameras[-1]
		else:
			cameras = self.cameras

		for i in range(len(cameras)):
			self.renderer.rasterizer.raster_settings.image_size=image_size
			pix2face = self.renderer.rasterizer(self.mesh_d, cameras=cameras[i]).pix_to_face # [1, 512, 512, 1]
			self.renderer.rasterizer.raster_settings.image_size=self.render_size
			pix2face_list.append(pix2face)

		for i in range(len(pix2face_list)):
			valid_faceid = torch.unique(pix2face_list[i])
			valid_faceid = valid_faceid[1:] if valid_faceid[0]==-1 else valid_faceid
			mask = torch.isin(self.uv_pix2face[0], valid_faceid, assume_unique=False)
			# uv_pix2face[0][~mask] = -1
			triangle_mask = torch.ones(self.target_size+(1,), device=self.device) # (res, res, 1)
			triangle_mask[~mask] = 0
			
			triangle_mask[:,1:][triangle_mask[:,:-1] > 0] = 1
			triangle_mask[:,:-1][triangle_mask[:,1:] > 0] = 1
			triangle_mask[1:,:][triangle_mask[:-1,:] > 0] = 1
			triangle_mask[:-1,:][triangle_mask[1:,:] > 0] = 1
			self.visible_triangles.append(triangle_mask)
		
		# get all face_ids 
		face_ids = torch.unique(torch.cat(pix2face_list))
		
		return face_ids


	# Render the current mesh and texture from current cameras
	def render_textured_views(self):
		meshes = self.mesh.extend(len(self.cameras))

		images_predicted = self.renderer(meshes.to(self.device), cameras=self.cameras, lights=self.lights)

		return [image.permute(2, 0, 1) for image in images_predicted]


	# Bake views into a texture
	# First bake into individual textures then combine based on cosine weight
	@torch.enable_grad()
	def bake_texture(self, views=None, main_views=[], cos_weighted=True, channels=None, exp=None, noisy=False, generator=None):
		from .voronoi import voronoi_solve
		if not exp:
			exp=1
		if not channels:
			channels = self.channels
		views = [view.permute(1, 2, 0) for view in views]

		tmp_mesh = self.mesh
		bake_maps = [torch.zeros(self.target_size+(views[0].shape[2],), device=self.device, requires_grad=True) for view in views]
		
		optimizer = torch.optim.SGD(bake_maps, lr=1, momentum=0)
		optimizer.zero_grad()
		loss = 0
		for i in range(len(views)):    
			bake_tex = TexturesUV([bake_maps[i]], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
			tmp_mesh.textures = bake_tex
			images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights, device=self.device)
			predicted_rgb = images_predicted[..., :-1]
			loss += (((predicted_rgb[...] - views[i]))**2).sum()
		loss.backward(retain_graph=False)
		optimizer.step()


		total_weights = 0
		baked = 0

		for i in range(len(bake_maps)):
			normalized_baked_map = bake_maps[i].detach() / (self.gradient_maps[i] + 1E-8)
			bake_map = voronoi_solve(normalized_baked_map, self.gradient_maps[i][...,0], self.device)

			weight = self.visible_triangles[i] * (self.cos_maps[i]) ** exp
			if noisy:
				noise = torch.rand(weight.shape[:-1]+(1,), generator=generator).type(weight.dtype).to(weight.device)
				weight *= noise

			total_weights += weight
			baked += bake_map * weight

	# 		from PIL import Image
	# 		weight_img = weight.cpu().numpy()
	# 		bake_img = bake_map.cpu().numpy()
	# 		img1 = (weight_img * 255.0).round().astype(np.uint8)
	# 		img2 = (bake_img * 255.0).round().astype(np.uint8)
	# 
	# 		img1 = Image.fromarray(img1)
	# 		img2 = Image.fromarray(img2)
	# 		img1.save(f"/data/xuqi/xuqi/code/WonderTex-1/weight_{i}.jpg")
	# 		img2.save(f"/data/xuqi/xuqi/code/WonderTex-1/map_{i}.jpg")			
		
		baked /= total_weights + 1E-8
		baked = voronoi_solve(baked, total_weights[...,0], self.device)
		bake_tex = TexturesUV([baked], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)

		tmp_mesh.textures = bake_tex
		extended_mesh = tmp_mesh.extend(len(self.cameras))
		images_predicted = self.renderer(extended_mesh, cameras=self.cameras, lights=self.lights)
		learned_views = [image.permute(2, 0, 1) for image in images_predicted] #(c, h, w)

		return learned_views, baked.permute(2, 0, 1), total_weights.permute(2, 0, 1)
		# return learned_views, bake_max.permute(2, 0, 1), max_weights.permute(2, 0, 1)

	# Bake views into a texture
	# First bake into individual textures then combine based on cosine weight
	@torch.enable_grad()
	def bake_texture_inpaint(self, views=None, main_views=[], tex=None, tex_w=None, cos_weighted=True, channels=None, exp=None, idx=0, noisy=False, generator=None):
		from .voronoi import voronoi_solve
		if not exp:
			exp=1
		if not channels:
			channels = self.channels
		views = [view.permute(1, 2, 0) for view in views]

		tmp_mesh = self.mesh
		bake_maps = [torch.zeros(self.target_size+(views[0].shape[2],), device=self.device, requires_grad=True) for view in views]
	
		optimizer = torch.optim.SGD(bake_maps, lr=1, momentum=0)
		optimizer.zero_grad()
		loss = 0
		for i in range(len(views)):    
			bake_tex = TexturesUV([bake_maps[i]], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
			tmp_mesh.textures = bake_tex
			images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[idx*4+i], lights=self.lights, device=self.device)
			predicted_rgb = images_predicted[..., :-1]
			loss += (((predicted_rgb[...] - views[i]))**2).sum()
		loss.backward(retain_graph=False)
		optimizer.step()

		total_weights = 0
		baked = 0

		for i in range(len(bake_maps)):
			normalized_baked_map = bake_maps[i].detach() / (self.gradient_maps[idx*4+i] + 1E-8)
			bake_map = voronoi_solve(normalized_baked_map, self.gradient_maps[idx*4+i][...,0], self.device)
			weight = self.visible_triangles[idx*4+i] * (self.cos_maps[idx*4+i]) ** exp
			if noisy:
				noise = torch.rand(weight.shape[:-1]+(1,), generator=generator).type(weight.dtype).to(weight.device)
				weight *= noise
			# weight[tex_idx] = 0 # use before
			total_weights += weight
			baked += bake_map * weight

		baked /= total_weights + 1E-8

		baked = voronoi_solve(baked, total_weights[...,0], self.device)

		# from torchvision import transforms
		# to_pil_image = transforms.ToPILImage()
		# image = to_pil_image(baked.permute(2, 0, 1))
		# image.save(f'tex_{idx}.jpg')

		if tex is not None:
			tex_idx = ((tex_w - total_weights) >= 0.1)  | (tex_w >= 0.5) # check
			baked[tex_idx] = tex[tex_idx]
			total_weights = torch.max(tex_w, total_weights)

		bake_tex = TexturesUV([baked], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
		tmp_mesh.textures = bake_tex
		extended_mesh = tmp_mesh.extend(len(self.cameras))
		images_predicted = self.renderer(extended_mesh, cameras=self.cameras, lights=self.lights)
		learned_views = [image.permute(2, 0, 1) for image in images_predicted] #(c, h, w)

		return learned_views, baked.permute(2, 0, 1), total_weights.permute(2, 0, 1)

	# Move the internel data to a specific device
	def to(self, device):
		for mesh_name in ["mesh", "mesh_d", "mesh_uv"]:
			if hasattr(self, mesh_name):
				mesh = getattr(self, mesh_name)
				setattr(self, mesh_name, mesh.to(device))
		for list_name in ["visible_triangles", "visibility_maps", "cos_maps"]:
			if hasattr(self, list_name):
				map_list = getattr(self, list_name)
				for i in range(len(map_list)):
					map_list[i] = map_list[i].to(device)
