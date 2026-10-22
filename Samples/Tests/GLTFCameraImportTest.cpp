#include "WickedEngine.h"
#include "ModelImporter.h"
#include "VulkanTestProcess.h"

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

// Regression test for the cameras of the editor's glTF importer: every camera node gets the parameters of the glTF camera
//	that it refers to, whatever the order of the nodes and the cameras, with orthographic cameras, cameras that no node uses
//	and cameras that several nodes use. A node with a camera and a light or a mesh keeps its camera, and the extras of a node
//	and its camera are merged. The exporter refers to the cameras the same way, so that they survive a round trip.
//	The import updates the scene, which needs a graphics device: the test runs with RunTestWithVulkanDevice(), and it is
//	skipped if there is none.

using namespace wi::ecs;
using namespace wi::scene;

namespace
{
	constexpr int TEST_FAILED = 2;
	const char* cameras_file = "GLTFCameraImportTest_cameras.gltf";
	const char* cameras_bin = "GLTFCameraImportTest_cameras.bin"; // the uri in the file below
	const char* unused_file = "GLTFCameraImportTest_unused.gltf";
	const char* output_file = "GLTFCameraImportTest_output.gltf";

	// Removes the output of the export: the file and its buffers (tinygltf names them after the file, numbered after the first one)
	void RemoveOutputFiles()
	{
		std::remove(output_file);
		std::remove("GLTFCameraImportTest_output.bin");
		for (int i = 0; i < 64; ++i)
		{
			std::remove(("GLTFCameraImportTest_output" + std::to_string(i) + ".bin").c_str());
		}
	}

	int failures = 0;

	void Check(bool condition, const std::string& message)
	{
		if (!condition)
		{
			std::cerr << "GLTFCameraImport test failed: " << message << '\n';
			failures++;
		}
	}

	bool Near(double a, double b)
	{
		return std::abs(a - b) <= 1e-4 * std::max(1.0, std::abs(b));
	}

	bool WriteFile(const char* path, const void* data, size_t size)
	{
		FILE* file = fopen(path, "wb");
		if (file == nullptr)
			return false;
		bool written = fwrite(data, size, 1, file) == 1;
		written &= fclose(file) == 0;
		return written;
	}

	// The buffer of the meshes below: a triangle, the joints and weights that skin it, and the inverse bind matrix of the joint,
	//	and the translation keys of the skinned mesh node with a camera (the first one is the translation of the node)
	struct MeshBuffer
	{
		XMFLOAT3 positions[3] = { XMFLOAT3(0, 0, 0), XMFLOAT3(1, 0, 0), XMFLOAT3(0, 1, 0) };
		uint8_t joints[3][4] = {};
		XMFLOAT4 weights[3] = { XMFLOAT4(1, 0, 0, 0), XMFLOAT4(1, 0, 0, 0), XMFLOAT4(1, 0, 0, 0) };
		XMFLOAT4X4 inverse_bind_matrix = XMFLOAT4X4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
		float times[2] = { 0, 1 };
		XMFLOAT3 translations[2] = { XMFLOAT3(4, 5, 6), XMFLOAT3(1, 2, 8) };
	};
	static_assert(sizeof(MeshBuffer) == 192 && offsetof(MeshBuffer, joints) == 36 && offsetof(MeshBuffer, weights) == 48 && offsetof(MeshBuffer, inverse_bind_matrix) == 96 &&
		offsetof(MeshBuffer, times) == 160 && offsetof(MeshBuffer, translations) == 168, "the buffer and buffer views in the file below");

	// Two perspective cameras that the nodes use in the other order than they are defined, an orthographic camera before
	//	them, a camera that no node uses, a camera that two nodes use, and a camera without zfar (an infinite projection).
	//	Nodes with a camera and a light, a mesh or a skinned mesh, and extras on a node and its camera, a skin and its node.
	const char* cameras_gltf = R"({
	"asset": { "version": "2.0" },
	"extensionsUsed": [ "KHR_lights_punctual" ],
	"extensions": { "KHR_lights_punctual": { "lights": [ { "type": "point", "name": "Lamp" } ] } },
	"scene": 0,
	"scenes": [ { "nodes": [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] } ],
	"nodes": [
		{ "name": "FarNode", "camera": 2 },
		{ "name": "NearNode", "camera": 1 },
		{ "name": "OrthoNode", "camera": 0 },
		{ "name": "SharedNode", "camera": 4, "extras": { "label": "node" } },
		{ "name": "OtherSharedNode", "camera": 4 },
		{ "name": "InfiniteNode", "camera": 5 },
		{ "name": "CameraLightNode", "camera": 6, "translation": [ 1, 2, 3 ], "extensions": { "KHR_lights_punctual": { "light": 0 } } },
		{ "name": "MeshCameraNode", "camera": 7, "mesh": 0 },
		{ "name": "SkinnedCameraNode", "camera": 8, "mesh": 1, "skin": 0, "translation": [ 4, 5, 6 ], "extras": { "label": "node", "node": 1 } },
		{ "name": "Joint" }
	],
	"cameras": [
		{ "name": "Ortho", "type": "orthographic", "orthographic": { "xmag": 3, "ymag": 2, "znear": 0.5, "zfar": 50 } },
		{ "name": "Near", "type": "perspective", "perspective": { "aspectRatio": 2, "yfov": 0.5, "znear": 0.2, "zfar": 20 } },
		{ "name": "Far", "type": "perspective", "perspective": { "aspectRatio": 1, "yfov": 1.2, "znear": 1, "zfar": 300 } },
		{ "name": "Unused", "type": "perspective", "perspective": { "aspectRatio": 0.5, "yfov": 0.9, "znear": 0.3, "zfar": 30 } },
		{ "name": "Shared", "type": "perspective", "perspective": { "aspectRatio": 1.5, "yfov": 0.7, "znear": 0.4, "zfar": 40 }, "extras": { "label": "camera", "shared": 2 } },
		{ "name": "Infinite", "type": "perspective", "perspective": { "yfov": 0.8, "znear": 0.25 } },
		{ "name": "LightCamera", "type": "perspective", "perspective": { "aspectRatio": 1.25, "yfov": 0.95, "znear": 0.35, "zfar": 35 } },
		{ "name": "MeshCamera", "type": "perspective", "perspective": { "aspectRatio": 0.75, "yfov": 0.6, "znear": 0.15, "zfar": 15 } },
		{ "name": "SkinnedCamera", "type": "orthographic", "orthographic": { "xmag": 5, "ymag": 2.5, "znear": 0.1, "zfar": 10 } }
	],
	"meshes": [
		{ "name": "Triangle", "primitives": [ { "attributes": { "POSITION": 0 } } ] },
		{ "name": "SkinnedTriangle", "primitives": [ { "attributes": { "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2 } } ] }
	],
	"skins": [ { "name": "Skin", "joints": [ 9 ], "inverseBindMatrices": 3, "extras": { "label": "skin", "skin": 1 } } ],
	"buffers": [ { "byteLength": 192, "uri": "GLTFCameraImportTest_cameras.bin" } ],
	"bufferViews": [
		{ "buffer": 0, "byteOffset": 0, "byteLength": 36 },
		{ "buffer": 0, "byteOffset": 36, "byteLength": 12 },
		{ "buffer": 0, "byteOffset": 48, "byteLength": 48 },
		{ "buffer": 0, "byteOffset": 96, "byteLength": 64 },
		{ "buffer": 0, "byteOffset": 160, "byteLength": 8 },
		{ "buffer": 0, "byteOffset": 168, "byteLength": 24 }
	],
	"accessors": [
		{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [ 0, 0, 0 ], "max": [ 1, 1, 0 ] },
		{ "bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4" },
		{ "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4" },
		{ "bufferView": 3, "componentType": 5126, "count": 1, "type": "MAT4" },
		{ "bufferView": 4, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [ 0 ], "max": [ 1 ] },
		{ "bufferView": 5, "componentType": 5126, "count": 2, "type": "VEC3" }
	],
	"animations": [ {
		"name": "SkinnedCameraMove",
		"samplers": [ { "input": 4, "output": 5 } ],
		"channels": [ { "sampler": 0, "target": { "node": 8, "path": "translation" } } ]
	} ]
})";

	// More cameras than camera nodes: the unused camera comes first
	const char* unused_gltf = R"({
	"asset": { "version": "2.0" },
	"scene": 0,
	"scenes": [ { "nodes": [ 0 ] } ],
	"nodes": [ { "name": "OnlyNode", "camera": 1 } ],
	"cameras": [
		{ "name": "Unused", "type": "perspective", "perspective": { "aspectRatio": 0.5, "yfov": 0.9, "znear": 0.3, "zfar": 30 } },
		{ "name": "Used", "type": "perspective", "perspective": { "aspectRatio": 2.5, "yfov": 1.1, "znear": 0.6, "zfar": 60 } }
	]
})";

	struct ExpectedCamera
	{
		std::string node;
		bool ortho;
		float aspect; // width / height
		float fov; // of a perspective camera
		float ortho_vertical_size; // of an orthographic camera
		float znear;
		float zfar;
	};

	// The camera, or the object, on an entity with the name (the entities of a skinned mesh node all have the node's name)
	const CameraComponent* CameraNamed(const Scene& scene, const std::string& name)
	{
		for (size_t i = 0; i < scene.cameras.GetCount(); ++i)
		{
			const NameComponent* entity_name = scene.names.GetComponent(scene.cameras.GetEntity(i));
			if (entity_name != nullptr && entity_name->name == name)
				return &scene.cameras[i];
		}
		return nullptr;
	}
	const ObjectComponent* ObjectNamed(const Scene& scene, const std::string& name)
	{
		for (size_t i = 0; i < scene.objects.GetCount(); ++i)
		{
			const NameComponent* entity_name = scene.names.GetComponent(scene.objects.GetEntity(i));
			if (entity_name != nullptr && entity_name->name == name)
				return &scene.objects[i];
		}
		return nullptr;
	}

	void CheckCameras(Scene& scene, const wi::vector<ExpectedCamera>& expected_cameras, const std::string& what)
	{
		Check(scene.cameras.GetCount() == expected_cameras.size(), what + ": not one camera component for each camera node");
		for (const ExpectedCamera& expected : expected_cameras)
		{
			const CameraComponent* camera = CameraNamed(scene, expected.node);
			if (camera == nullptr)
			{
				Check(false, what + ": the node " + expected.node + " has no camera");
				continue;
			}
			bool same = camera->IsOrtho() == expected.ortho && camera->height != 0 && Near(camera->width / camera->height, expected.aspect) &&
				Near(camera->zNearP, expected.znear) && Near(camera->zFarP, expected.zfar);
			same &= expected.ortho ? Near(camera->ortho_vertical_size, expected.ortho_vertical_size) : Near(camera->fov, expected.fov);
			Check(same, what + ": the node " + expected.node + " did not get the parameters of its camera");
		}
	}

	// The camera of a node with a light or a mesh is on the entity of the node (of a skinned mesh, on an entity with the node's transform)
	void CheckCameraNodes(Scene& scene, const std::string& what)
	{
		// The light is on a child, which does not turn the camera (the glTF camera looks to -Z, mirrored to +Z in the engine):
		const Entity camera_light = scene.Entity_FindByName("CameraLightNode");
		const CameraComponent* camera = scene.cameras.GetComponent(camera_light);
		const HierarchyComponent* light_hierarchy = scene.lights.GetCount() == 1 ? scene.hierarchy.GetComponent(scene.lights.GetEntity(0)) : nullptr;
		Check(scene.hierarchy.Contains(camera_light) && camera != nullptr && Near(camera->Eye.x, 1) && Near(camera->Eye.y, 2) && Near(camera->Eye.z, -3) &&
			Near(camera->At.x, 0) && Near(camera->At.y, 0) && Near(camera->At.z, 1), what + ": the camera of a node with a light is not on the node in the hierarchy");
		Check(light_hierarchy != nullptr && light_hierarchy->parentID == camera_light, what + ": the light of a node with a camera is not on a child of the node");

		Check(scene.objects.Contains(scene.Entity_FindByName("MeshCameraNode")), what + ": the camera of a node with a mesh is not on the object");
		const ObjectComponent* skinned = ObjectNamed(scene, "SkinnedCameraNode");
		const MeshComponent* skinned_mesh = skinned != nullptr ? scene.meshes.GetComponent(skinned->meshID) : nullptr;
		Check(skinned_mesh != nullptr && skinned_mesh->armatureID != INVALID_ENTITY, what + ": the node with a skinned mesh and a camera lost its skinned mesh");
		// Only the skinned mesh ignores the transform of its node, the camera has it:
		const CameraComponent* skinned_camera = CameraNamed(scene, "SkinnedCameraNode");
		Check(skinned_camera != nullptr && Near(skinned_camera->Eye.x, 4) && Near(skinned_camera->Eye.y, 5) && Near(skinned_camera->Eye.z, -6),
			what + ": the camera of a node with a skinned mesh does not have the node's transform");
		Check(scene.cameras.GetComponent(scene.Entity_FindByName("SkinnedCameraNode")) == skinned_camera, what + ": the name of a skinned mesh node does not find its camera");

		// The translation keys of the skinned mesh node move its camera to the keys (not by the keys from the node's translation).
		//	The timer is evaluated when it changes, also if the animation doesn't play; at the first key it is back at the node.
		AnimationComponent* move = nullptr;
		for (size_t i = 0; i < scene.animations.GetCount(); ++i)
		{
			const NameComponent* name = scene.names.GetComponent(scene.animations.GetEntity(i));
			if (name != nullptr && name->name == "SkinnedCameraMove")
			{
				move = &scene.animations[i];
			}
		}
		Check(move != nullptr, what + ": the animation of the skinned mesh node is missing");
		if (move != nullptr)
		{
			move->timer = 1;
			scene.Update(0);
			const CameraComponent* moved = CameraNamed(scene, "SkinnedCameraNode");
			Check(moved != nullptr && Near(moved->Eye.x, 1) && Near(moved->Eye.y, 2) && Near(moved->Eye.z, -8), what + ": the translation keys of a skinned mesh node did not move its camera to the keys");
			move->timer = 0;
			scene.Update(0);
		}
	}

	int RunImport()
	{
		RemoveOutputFiles(); // the files of an earlier run must not be checked

		// The camera that a node gets before the import fills it, the camera without zfar keeps its far plane
		Scene created;
		const CameraComponent& created_camera = *created.cameras.GetComponent(created.Entity_CreateCamera("", 16, 9));

		const wi::vector<ExpectedCamera> expected_cameras = {
			{ "FarNode", false, 1, 1.2f, 0, 1, 300 },
			{ "NearNode", false, 2, 0.5f, 0, 0.2f, 20 },
			{ "OrthoNode", true, 1.5f, 0, 4, 0.5f, 50 }, // the whole height of the view, glTF has half of it
			{ "SharedNode", false, 1.5f, 0.7f, 0, 0.4f, 40 },
			{ "OtherSharedNode", false, 1.5f, 0.7f, 0, 0.4f, 40 },
			{ "InfiniteNode", false, created_camera.width / created_camera.height, 0.8f, 0, 0.25f, created_camera.zFarP },
			{ "CameraLightNode", false, 1.25f, 0.95f, 0, 0.35f, 35 },
			{ "MeshCameraNode", false, 0.75f, 0.6f, 0, 0.15f, 15 },
			{ "SkinnedCameraNode", true, 2, 0, 5, 0.1f, 10 },
		};

		const MeshBuffer mesh_buffer;
		const bool written = WriteFile(cameras_file, cameras_gltf, strlen(cameras_gltf)) && WriteFile(cameras_bin, &mesh_buffer, sizeof(mesh_buffer)) &&
			WriteFile(unused_file, unused_gltf, strlen(unused_gltf));
		Check(written, "the glTF files of the test could not be written");
		if (!written)
		{
			std::remove(cameras_file);
			std::remove(cameras_bin);
			std::remove(unused_file);
			return TEST_FAILED;
		}

		Scene scene;
		ImportModel_GLTF(cameras_file, scene);
		std::remove(cameras_file);
		std::remove(cameras_bin);
		CheckCameras(scene, expected_cameras, "import");
		CheckCameraNodes(scene, "import");

		// The extras of a node and of its camera (or skin) are merged, the node's own values over its camera's (or skin's)
		auto metadata_of = [&](const std::string& name) { return scene.metadatas.GetComponent(scene.Entity_FindByName(name)); };
		const MetadataComponent* shared_metadata = metadata_of("SharedNode");
		const MetadataComponent* other_shared_metadata = metadata_of("OtherSharedNode");
		const MetadataComponent* skin_metadata = metadata_of("Skin"); // the armature, which gets the extras of the skinned mesh node too
		Check(shared_metadata != nullptr && shared_metadata->string_values.get("label") == "node" && shared_metadata->int_values.get("shared") == 2,
			"the extras of a node and its camera were not merged");
		Check(other_shared_metadata != nullptr && other_shared_metadata->string_values.get("label") == "camera" && other_shared_metadata->int_values.get("shared") == 2,
			"the extras of a shared camera are not on each of its nodes");
		Check(skin_metadata != nullptr && skin_metadata->string_values.get("label") == "node" && skin_metadata->int_values.get("skin") == 1 && skin_metadata->int_values.get("node") == 1,
			"the extras of a skin and its node were not merged (with the node's values over the skin's)");
		for (size_t i = 0; i < scene.metadatas.GetCount(); ++i)
		{
			Check(scene.metadatas.GetIndex(scene.metadatas.GetEntity(i)) == i, "an entity has more than one metadata component");
		}

		// The import used to fill the camera components in the order of the glTF cameras, writing past the last component
		//	when there were more cameras than camera nodes
		Scene unused;
		ImportModel_GLTF(unused_file, unused);
		std::remove(unused_file);
		CheckCameras(unused, { { "OnlyNode", false, 2.5f, 1.1f, 0, 0.6f, 60 } }, "import with an unused camera");

		// The exported nodes refer to the cameras with their parameters, and the cameras are imported back the same way
		ExportModel_GLTF(output_file, scene);
		tinygltf::TinyGLTF loader;
		tinygltf::Model model;
		std::string err, warn;
		if (!loader.LoadASCIIFromFile(&model, &err, &warn, output_file))
		{
			Check(false, "the exported file could not be loaded: " + err);
			RemoveOutputFiles();
			return TEST_FAILED;
		}
		for (const ExpectedCamera& expected : expected_cameras)
		{
			const tinygltf::Camera* camera = nullptr;
			for (const tinygltf::Node& node : model.nodes)
			{
				if (node.name == expected.node && node.camera >= 0 && node.camera < (int)model.cameras.size())
				{
					camera = &model.cameras[node.camera];
				}
			}
			bool same = camera != nullptr;
			if (same && expected.ortho)
			{
				same = camera->type == "orthographic" && Near(camera->orthographic.ymag, expected.ortho_vertical_size * 0.5) &&
					Near(camera->orthographic.xmag, expected.ortho_vertical_size * 0.5 * expected.aspect) &&
					Near(camera->orthographic.znear, expected.znear) && Near(camera->orthographic.zfar, expected.zfar);
			}
			else if (same)
			{
				same = camera->type == "perspective" && Near(camera->perspective.aspectRatio, expected.aspect) && Near(camera->perspective.yfov, expected.fov) &&
					Near(camera->perspective.znear, expected.znear) && Near(camera->perspective.zfar, expected.zfar);
			}
			Check(same, "export: the node " + expected.node + " does not refer to a camera with its parameters");
		}
		Scene imported;
		ImportModel_GLTF(output_file, imported);
		RemoveOutputFiles();
		CheckCameras(imported, expected_cameras, "import of the export");
		CheckCameraNodes(imported, "import of the export");

		if (failures > 0)
			return TEST_FAILED;
		std::cout << "GLTFCameraImport test passed\n";
		return 0;
	}
}

int main(int argc, char** argv)
{
	return RunTestWithVulkanDevice(argc, argv, "GLTFCameraImport", RunImport);
}
