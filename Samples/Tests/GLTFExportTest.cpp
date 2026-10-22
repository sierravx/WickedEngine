#include "WickedEngine.h"
#include "wiGraphicsDevice_Vulkan.h"
#include "ModelImporter.h"

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

// Regression test for the animation export of the editor's glTF exporter: only channels that glTF can hold are written,
//	retargeted channels are baked, the file is valid, and the scene is restored after the export.
//	It also tests the import of glTF nodes with a light (children, mesh, camera and animation of the node) and their export.
//	The exporter updates the scene, which needs a graphics device: the test runs in a child process that creates a Vulkan
//	device (like Mesa's software lavapipe), and it is skipped if there is none.

using namespace wi::ecs;
using namespace wi::scene;
using Path = AnimationComponent::AnimationChannel::Path;

namespace
{
	constexpr int TEST_SKIPPED = 77;
	constexpr int TEST_FAILED = 2;
	const char* output_name = "GLTFExportTest_output";
	const std::string output_file = std::string(output_name) + ".gltf";

	// Removes the output of an export: the file and its buffers (tinygltf names them after the file, numbered after the first one)
	void RemoveOutputFiles(const std::string& name)
	{
		std::remove((name + ".gltf").c_str());
		std::remove((name + ".bin").c_str());
		for (int i = 0; i < 64; ++i)
		{
			std::remove((name + std::to_string(i) + ".bin").c_str());
		}
	}

	int failures = 0;

	void Check(bool condition, const std::string& message)
	{
		if (!condition)
		{
			std::cerr << "GLTFExport test failed: " << message << '\n';
			failures++;
		}
	}

	bool Near(const wi::vector<float>& a, const wi::vector<float>& b, float tolerance = 1e-4f)
	{
		if (a.size() != b.size())
			return false;
		for (size_t i = 0; i < a.size(); ++i)
		{
			if (std::abs(a[i] - b[i]) > tolerance)
				return false;
		}
		return true;
	}

	Entity CreateNode(Scene& scene, const std::string& name, const XMFLOAT3& translation, const XMFLOAT3& rollPitchYaw = XMFLOAT3(0, 0, 0))
	{
		Entity entity = CreateEntity();
		scene.names.Create(entity) = name;
		TransformComponent& transform = scene.transforms.Create(entity);
		transform.RotateRollPitchYaw(rollPitchYaw);
		transform.Translate(translation);
		transform.UpdateTransform();
		return entity;
	}

	Entity CreateData(Scene& scene, const wi::vector<float>& times, const wi::vector<float>& values)
	{
		Entity entity = CreateEntity();
		AnimationDataComponent& data = scene.animation_datas.Create(entity);
		data.keyframe_times = times;
		data.keyframe_data = values;
		return entity;
	}

	Entity CreateMorphedMesh(Scene& scene, Entity material)
	{
		Entity entity = CreateEntity();
		MeshComponent& mesh = scene.meshes.Create(entity);
		mesh.vertex_positions = { XMFLOAT3(0, 0, 0), XMFLOAT3(1, 0, 0), XMFLOAT3(0, 1, 0) };
		mesh.vertex_normals = { XMFLOAT3(0, 0, 1), XMFLOAT3(0, 0, 1), XMFLOAT3(0, 0, 1) };
		mesh.indices = { 0, 1, 2 };
		MeshComponent::MeshSubset& subset = mesh.subsets.emplace_back();
		subset.materialID = material;
		subset.indexCount = 3;
		mesh.morph_targets.emplace_back().vertex_positions = { XMFLOAT3(0, 0, 1), XMFLOAT3(0, 0, 1), XMFLOAT3(0, 0, 1) };
		mesh.CreateRenderData();
		return entity;
	}

	// What the runtime evaluates for a translation key of a retargeted channel (see AnimationComponent::RetargetSourceData::RetargetValue())
	XMVECTOR RetargetTranslation(const TransformComponent& source, const XMFLOAT4X4& dst, const XMFLOAT4X4& src, XMVECTOR key)
	{
		TransformComponent transform = source;
		XMStoreFloat3(&transform.translation_local, key);
		XMVECTOR S, R, T;
		wi::math::DecomposeTransform(&S, &R, &T, XMLoadFloat4x4(&dst) * transform.GetLocalMatrix() * XMLoadFloat4x4(&src));
		return T;
	}
	wi::vector<float> RetargetTranslations(const TransformComponent& source, const XMFLOAT4X4& dst, const XMFLOAT4X4& src, const wi::vector<float>& keys)
	{
		wi::vector<float> result;
		for (size_t i = 0; i + 3 <= keys.size(); i += 3)
		{
			const XMVECTOR T = RetargetTranslation(source, dst, src, XMVectorSet(keys[i], keys[i + 1], keys[i + 2], 0));
			result.push_back(XMVectorGetX(T));
			result.push_back(XMVectorGetY(T));
			result.push_back(XMVectorGetZ(T));
		}
		return result;
	}

	// The engine's keys in the glTF space
	wi::vector<float> MirrorTranslations(wi::vector<float> keys)
	{
		for (size_t i = 2; i < keys.size(); i += 3)
		{
			keys[i] = -keys[i];
		}
		return keys;
	}
	XMVECTOR MirrorRotation(XMVECTOR Q)
	{
		return XMVectorMultiply(Q, XMVectorSet(-1, -1, 1, 1));
	}
	// The same rotation, as either of its two quaternions (the distance is about half the angle between the rotations)
	bool SameRotation(XMVECTOR q0, XMVECTOR q1, float tolerance = 1e-4f)
	{
		return std::min(XMVectorGetX(XMVector4Length(q0 - q1)), XMVectorGetX(XMVector4Length(q0 + q1))) < tolerance;
	}

	wi::vector<float> ReadAccessor(const tinygltf::Model& model, int accessor_index)
	{
		const tinygltf::Accessor& accessor = model.accessors[accessor_index];
		const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[view.buffer];
		const size_t components = accessor.type == TINYGLTF_TYPE_VEC4 ? 4 : (accessor.type == TINYGLTF_TYPE_VEC3 ? 3 : 1);
		const float* values = (const float*)(buffer.data.data() + view.byteOffset + accessor.byteOffset);
		return wi::vector<float>(values, values + accessor.count * components);
	}

	// A local matrix of the glTF space: scale, rotation, then translation (row vectors, like DirectXMath)
	XMMATRIX LocalMatrix(const XMFLOAT3& translation, XMVECTOR rotation, const XMFLOAT3& scale = XMFLOAT3(1, 1, 1))
	{
		return XMMatrixScaling(scale.x, scale.y, scale.z) * XMMatrixRotationQuaternion(rotation) * XMMatrixTranslation(translation.x, translation.y, translation.z);
	}
	// A matrix of the glTF space in the engine space: F * M * F with F = scale(1, 1, -1)
	XMMATRIX MirrorMatrix(XMMATRIX M)
	{
		const XMMATRIX F = XMMatrixScaling(1, 1, -1);
		return F * M * F;
	}
	bool NearMatrix(const XMFLOAT4X4& a, XMMATRIX B, float tolerance = 1e-4f)
	{
		XMFLOAT4X4 b;
		XMStoreFloat4x4(&b, B);
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				if (std::abs(a.m[i][j] - b.m[i][j]) > tolerance)
					return false;
			}
		}
		return true;
	}
	// The engine's light direction (which points to the light) of a glTF light on a node with this world matrix of the glTF
	//	space: glTF lights point along the -Z axis of their node
	XMVECTOR EngineLightDirection(XMMATRIX gltf_world)
	{
		const XMVECTOR to_light = XMVector3Normalize(XMVector3TransformNormal(XMVectorSet(0, 0, 1, 0), gltf_world));
		return XMVectorMultiply(to_light, XMVectorSet(1, 1, -1, 0));
	}
	// The entity with the light of a node: the node, or a child of it
	Entity LightOf(const Scene& scene, Entity node)
	{
		if (node == INVALID_ENTITY || scene.lights.Contains(node))
			return node;
		for (size_t i = 0; i < scene.lights.GetCount(); ++i)
		{
			const HierarchyComponent* hierarchy = scene.hierarchy.GetComponent(scene.lights.GetEntity(i));
			if (hierarchy != nullptr && hierarchy->parentID == node)
				return scene.lights.GetEntity(i);
		}
		return INVALID_ENTITY;
	}
	std::string JsonArray(std::initializer_list<float> values)
	{
		std::string result;
		for (float value : values)
		{
			char text[32];
			snprintf(text, sizeof(text), "%.9g", value);
			result += (result.empty() ? "[ " : ", ") + std::string(text);
		}
		return result + " ]";
	}
	std::string JsonArray(const XMFLOAT3& value)
	{
		return JsonArray({ value.x, value.y, value.z });
	}
	std::string JsonArray(XMVECTOR quaternion)
	{
		XMFLOAT4 value;
		XMStoreFloat4(&value, quaternion);
		return JsonArray({ value.x, value.y, value.z, value.w });
	}

	// Import: a glTF sampler that animates the rotation of a light node (with nothing else, so the light is the node) and of an
	//	other node. The light's keys get the rotation of the light (glTF lights point forward, engine lights down), the other
	//	node's keys must not
	void CheckSharedSamplerImport()
	{
		const XMVECTOR qa = XMQuaternionRotationRollPitchYaw(0.2f, 0.4f, 0.1f);
		const XMVECTOR qb = XMQuaternionRotationRollPitchYaw(-0.3f, 0.1f, 0.5f);
		float buffer[10] = { 0, 1 }; // two key times and two rotations
		static_assert(sizeof(buffer) == 40, "the byteLength and buffer views in the file below");
		XMStoreFloat4((XMFLOAT4*)&buffer[2], qa);
		XMStoreFloat4((XMFLOAT4*)&buffer[6], qb);
		const char* shared_file = "GLTFImportTest_shared.gltf";
		const char* shared_bin = "GLTFImportTest_shared.bin"; // the uri in the file below
		bool written = false;
		if (FILE* bin = fopen(shared_bin, "wb"))
		{
			written = fwrite(buffer, sizeof(buffer), 1, bin) == 1;
			written &= fclose(bin) == 0;
		}
		if (FILE* gltf = fopen(shared_file, "w"))
		{
			written &= fputs(R"({
	"asset": { "version": "2.0" },
	"extensionsUsed": [ "KHR_lights_punctual" ],
	"extensions": { "KHR_lights_punctual": { "lights": [ { "type": "point", "name": "SharedLight" } ] } },
	"scene": 0,
	"scenes": [ { "nodes": [ 0, 1 ] } ],
	"nodes": [ { "name": "SharedLight", "extensions": { "KHR_lights_punctual": { "light": 0 } } }, { "name": "SharedNode" } ],
	"buffers": [ { "byteLength": 40, "uri": "GLTFImportTest_shared.bin" } ],
	"bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 8 }, { "buffer": 0, "byteOffset": 8, "byteLength": 32 } ],
	"accessors": [
		{ "bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [ 0 ], "max": [ 1 ] },
		{ "bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC4" }
	],
	"animations": [ {
		"samplers": [ { "input": 0, "output": 1 } ],
		"channels": [ { "sampler": 0, "target": { "node": 0, "path": "rotation" } }, { "sampler": 0, "target": { "node": 1, "path": "rotation" } } ]
	} ]
})", gltf) >= 0;
			written &= fclose(gltf) == 0;
		}
		else
		{
			written = false;
		}
		Check(written, "the glTF file of the import test could not be written");
		if (!written)
		{
			std::remove(shared_file);
			std::remove(shared_bin);
			return;
		}

		Scene shared;
		ImportModel_GLTF(shared_file, shared);
		std::remove(shared_file);
		std::remove(shared_bin);

		const Entity shared_light = shared.Entity_FindByName("SharedLight");
		const Entity shared_node = shared.Entity_FindByName("SharedNode");
		const AnimationDataComponent* light_data = nullptr;
		const AnimationDataComponent* node_data = nullptr;
		for (size_t i = 0; i < shared.animations.GetCount(); ++i)
		{
			const AnimationComponent& animation = shared.animations[i];
			for (const AnimationComponent::AnimationChannel& channel : animation.channels)
			{
				if (channel.path != Path::ROTATION || channel.samplerIndex < 0 || channel.samplerIndex >= (int)animation.samplers.size())
					continue;
				const AnimationDataComponent* data = shared.animation_datas.GetComponent(animation.samplers[channel.samplerIndex].data);
				if (channel.target == shared_light)
					light_data = data;
				if (channel.target == shared_node)
					node_data = data;
			}
		}
		// The import rotates the keys of a light by this in the glTF space, then mirrors all keys to the engine space:
		const XMVECTOR light_rotation = XMQuaternionRotationRollPitchYaw(XM_PIDIV2, 0, 0);
		auto keys_of = [](const AnimationDataComponent* data, size_t key) { return XMLoadFloat4((const XMFLOAT4*)&data->keyframe_data[key * 4]); };
		const bool found = shared.lights.Contains(shared_light) && light_data != nullptr && node_data != nullptr && light_data != node_data &&
			light_data->keyframe_data.size() == 8 && node_data->keyframe_data.size() == 8;
		Check(found, "a sampler shared by a light and an other node did not import as two datas");
		if (found)
		{
			Check(SameRotation(keys_of(node_data, 0), MirrorRotation(qa)) && SameRotation(keys_of(node_data, 1), MirrorRotation(qb)), "the keys of a node were rotated like the keys of a light sharing their sampler");
			Check(SameRotation(keys_of(light_data, 0), MirrorRotation(XMQuaternionMultiply(light_rotation, qa))) && SameRotation(keys_of(light_data, 1), MirrorRotation(XMQuaternionMultiply(light_rotation, qb))), "the keys of a light sharing a sampler with an other node did not get the light's rotation");
		}
	}


	// Import of the names of light entities, and the lights of joints: a node with only a light is the light and keeps the
	//	node's name (the light's name is used for a node without one, "light" with a number for a light without one too), the
	//	light of a node with more is on a child named after the light (or after the node, for a light without a name). A joint
	//	with a light keeps its rotation, the light is on a child. A light node whose scale would mirror the light rotation has
	//	its light on a child too, a node with a light that doesn't exist has none. The nodes stay after an export and an import,
	//	the light of a node with only a child that is a light node is not written on that node.
	void CheckLightNames()
	{
		const std::string names_file = "GLTFImportTest_light_names.gltf";
		const std::string export_name = "GLTFImportTest_light_names_export";
		const XMVECTOR joint_rotation = XMQuaternionRotationRollPitchYaw(0.3f, 0.2f, -0.4f);
		const XMVECTOR mirrored_rotation = XMQuaternionRotationRollPitchYaw(-0.2f, 0.5f, 0.3f);
		const XMFLOAT3 mirrored_scale(1, 2, -1);
		const std::string json = R"({
	"asset": { "version": "2.0" },
	"extensionsUsed": [ "KHR_lights_punctual" ],
	"extensions": { "KHR_lights_punctual": { "lights": [
		{ "type": "point", "name": "PlainDefinition" },
		{ "type": "point", "name": "NodelessDefinition" },
		{ "type": "point" },
		{ "type": "spot", "name": "SharedDefinition", "spot": { "outerConeAngle": 0.6 } },
		{ "type": "point", "name": "HolderBulb" },
		{ "type": "point" },
		{ "type": "point", "name": "JointBulb" },
		{ "type": "spot", "name": "MirroredBulb", "spot": { "outerConeAngle": 0.5 } }
	] } },
	"scene": 0,
	"scenes": [ { "nodes": [ 0, 1, 2, 3, 4, 5, 7, 9, 10, 12, 13 ] } ],
	"nodes": [
		{ "name": "PlainNode", "extensions": { "KHR_lights_punctual": { "light": 0 } } },
		{ "extensions": { "KHR_lights_punctual": { "light": 1 } } },
		{ "extensions": { "KHR_lights_punctual": { "light": 2 } } },
		{ "name": "ShareA", "extensions": { "KHR_lights_punctual": { "light": 3 } } },
		{ "name": "ShareB", "extensions": { "KHR_lights_punctual": { "light": 3 } } },
		{ "name": "Holder", "children": [ 6 ], "extensions": { "KHR_lights_punctual": { "light": 4 } } },
		{ "name": "HolderChild" },
		{ "name": "UnnamedHolder", "children": [ 8 ], "extensions": { "KHR_lights_punctual": { "light": 5 } } },
		{ "name": "UnnamedHolderChild" },
		{ "name": "Joint", "rotation": )" + JsonArray(joint_rotation) + R"(, "extensions": { "KHR_lights_punctual": { "light": 6 } } },
		{ "name": "Pivot", "children": [ 11 ] },
		{ "name": "Lamp", "extensions": { "KHR_lights_punctual": { "light": 0 } } },
		{ "name": "Mirrored", "rotation": )" + JsonArray(mirrored_rotation) + R"(, "scale": )" + JsonArray(mirrored_scale) + R"(, "extensions": { "KHR_lights_punctual": { "light": 7 } } },
		{ "name": "MissingLight", "extensions": { "KHR_lights_punctual": { "light": 99 } } }
	],
	"skins": [ { "name": "JointSkin", "joints": [ 9 ] } ]
})";
		bool written = false;
		if (FILE* gltf = fopen(names_file.c_str(), "w"))
		{
			written = fputs(json.c_str(), gltf) >= 0;
			written &= fclose(gltf) == 0;
		}
		Check(written, "the glTF file of the light names test could not be written");
		if (!written)
		{
			std::remove(names_file.c_str());
			return;
		}
		{
			tinygltf::TinyGLTF loader;
			tinygltf::Model model;
			std::string err, warn;
			Check(loader.LoadASCIIFromFile(&model, &err, &warn, names_file), "the glTF file of the light names test is not valid: " + err);
		}
		Scene scene;
		ImportModel_GLTF(names_file, scene);
		std::remove(names_file.c_str());

		auto light_named = [](const Scene& of, const std::string& name) {
			for (size_t i = 0; i < of.lights.GetCount(); ++i)
			{
				const NameComponent* light_name = of.names.GetComponent(of.lights.GetEntity(i));
				if (light_name != nullptr && light_name->name == name)
					return of.lights.GetEntity(i);
			}
			return INVALID_ENTITY;
		};
		auto parent_of = [](const Scene& of, Entity entity) {
			const HierarchyComponent* hierarchy = of.hierarchy.GetComponent(entity);
			return hierarchy == nullptr ? INVALID_ENTITY : hierarchy->parentID;
		};
		Check(scene.lights.GetCount() == 10, "light names: expected 10 lights, got " + std::to_string(scene.lights.GetCount()));
		Check(scene.lights.Contains(scene.Entity_FindByName("Lamp")) && !scene.lights.Contains(scene.Entity_FindByName("Pivot")), "light names: the light node under a node is not the light");
		Check(!scene.lights.Contains(scene.Entity_FindByName("MissingLight")) && LightOf(scene, scene.Entity_FindByName("MissingLight")) == INVALID_ENTITY,
			"light names: a node with a light that doesn't exist got a light");
		// The light of the mirrored node points where the glTF light points (in the engine space):
		auto check_mirrored = [&](Scene& of, const std::string& what) {
			const Entity mirrored = of.Entity_FindByName("Mirrored");
			const LightComponent* light = of.lights.GetComponent(LightOf(of, mirrored));
			const XMVECTOR expected = EngineLightDirection(LocalMatrix(XMFLOAT3(0, 0, 0), mirrored_rotation, mirrored_scale));
			Check(light != nullptr && !of.lights.Contains(mirrored) && XMVectorGetX(XMVector3Length(XMLoadFloat3(&light->direction) - expected)) < 1e-4f,
				what + ": the light of a node with a mirroring scale is not on a child pointing where the glTF light points");
		};
		check_mirrored(scene, "light names");
		for (const char* node : { "PlainNode", "ShareA", "ShareB" })
		{
			Check(scene.lights.Contains(scene.Entity_FindByName(node)), std::string("light names: the light of the node ") + node + " is not the node with its name");
		}
		Check(scene.lights.Contains(scene.Entity_FindByName("NodelessDefinition")), "light names: a light node without a name did not get the light's name");
		bool numbered = false;
		for (size_t i = 0; i < scene.lights.GetCount(); ++i)
		{
			const NameComponent* name = scene.names.GetComponent(scene.lights.GetEntity(i));
			numbered |= name != nullptr && name->name.rfind("light", 0) == 0 && name->name.size() > 5 && std::isdigit((unsigned char)name->name[5]);
		}
		Check(numbered, "light names: a light node without a name and a light without a name did not get a numbered name");
		Check(scene.Entity_FindByName("SharedDefinition") == INVALID_ENTITY, "light names: the nodes sharing a light got the light's name");
		const Entity holder_bulb = light_named(scene, "HolderBulb");
		Check(holder_bulb != INVALID_ENTITY && parent_of(scene, holder_bulb) == scene.Entity_FindByName("Holder") && !scene.lights.Contains(scene.Entity_FindByName("Holder")),
			"light names: the light of a node with a child is not on a child named after the light");
		const Entity unnamed_bulb = light_named(scene, "UnnamedHolder");
		Check(unnamed_bulb != INVALID_ENTITY && parent_of(scene, unnamed_bulb) == scene.Entity_FindByName("UnnamedHolder"),
			"light names: the light without a name of a node with a child is not on a child named after the node");
		const Entity joint = scene.Entity_FindByName("Joint");
		const TransformComponent* joint_transform = scene.transforms.GetComponent(joint);
		const Entity joint_bulb = light_named(scene, "JointBulb");
		Check(joint_transform != nullptr && SameRotation(XMLoadFloat4(&joint_transform->rotation_local), MirrorRotation(joint_rotation)) && !scene.lights.Contains(joint) &&
			joint_bulb != INVALID_ENTITY && parent_of(scene, joint_bulb) == joint, "light names: a joint with a light did not keep its rotation, with the light on a child");

		RemoveOutputFiles(export_name);
		ExportModel_GLTF(export_name + ".gltf", scene);
		{
			// A light node under a node with nothing else is written as a node with its light, not as the light of its parent
			//	(the import of which would make the parent the light). (The light nodes at the top, under the root of the import
			//	with its other nodes, are written as lights of the root, which imports them back on children named after them.)
			tinygltf::TinyGLTF loader;
			tinygltf::Model model;
			std::string err, warn;
			const bool loaded = loader.LoadASCIIFromFile(&model, &err, &warn, export_name + ".gltf");
			Check(loaded, "light names: the exported file could not be loaded: " + err);
			auto light_of_node = [&](const std::string& name) {
				for (const tinygltf::Node& node : model.nodes)
				{
					if (node.name == name)
						return node.light;
				}
				return -2;
			};
			Check(!loaded || (light_of_node("Lamp") >= 0 && light_of_node("Pivot") == -1), "light names: a light node under a node with nothing else was not written as a node with its light");
		}
		Scene reimported;
		ImportModel_GLTF(export_name + ".gltf", reimported);
		RemoveOutputFiles(export_name);
		Check(reimported.lights.Contains(reimported.Entity_FindByName("Lamp")) && !reimported.lights.Contains(reimported.Entity_FindByName("Pivot")),
			"light names, export and import: the light node under a node is not the light");
		check_mirrored(reimported, "light names, export and import");
		for (const char* node : { "PlainNode", "ShareA", "ShareB", "NodelessDefinition", "Holder", "HolderChild", "UnnamedHolder", "Joint", "Pivot", "Lamp", "Mirrored" })
		{
			Check(reimported.Entity_FindByName(node) != INVALID_ENTITY, std::string("light names, export and import: the node ") + node + " lost its name");
		}
		for (const char* node : { "PlainNode", "ShareA", "ShareB" })
		{
			Check(reimported.lights.Contains(reimported.Entity_FindByName(node)), std::string("light names, export and import: the light of the node ") + node + " is not the node");
		}
		Check(light_named(reimported, "HolderBulb") != INVALID_ENTITY, "light names, export and import: the light of a node with a child lost its name");
		Check(reimported.lights.GetCount() == 10, "light names, export and import: expected 10 lights, got " + std::to_string(reimported.lights.GetCount()));
	}

	// Import and export of glTF nodes with a light (KHR_lights_punctual): (a) a light node with a child node, (b) a node with
	//	a mesh and a light (and one with a camera and a light), (c) a light node with a rotation animation, the sampler of which
	//	also animates an other node. The nodes must have the world matrices of the file (in the engine space) with their mesh,
	//	camera and children, also when the animation plays, the lights must point where the glTF lights point, and an export
	//	must write the lights on their nodes, which import back the same.
	void CheckLightNodes()
	{
		const XMFLOAT3 lamp_translation(1, 2, 3);
		const XMVECTOR lamp_rotation = XMQuaternionRotationRollPitchYaw(0.3f, -0.5f, 0.2f);
		const XMFLOAT3 lamp_scale(1, 2, 0.5f); // non-uniform, which the child must get as in glTF
		const XMFLOAT3 shade_translation(0.5f, -0.25f, -1);
		const XMVECTOR shade_rotation = XMQuaternionRotationRollPitchYaw(0.1f, 0.4f, -0.3f);
		const XMFLOAT3 mesh_light_translation(-2, 0.5f, 1);
		const XMVECTOR mesh_light_rotation = XMQuaternionRotationRollPitchYaw(-0.6f, 0.2f, 0.4f);
		const XMFLOAT3 camera_light_translation(0, 3, -2);
		const XMVECTOR camera_light_rotation = XMQuaternionRotationRollPitchYaw(0.2f, 0.9f, -0.1f);
		const XMFLOAT3 follower_translation(3, 0, 0);
		const XMVECTOR keys[2] = { XMQuaternionRotationRollPitchYaw(0.2f, 0.4f, 0.1f), XMQuaternionRotationRollPitchYaw(-0.3f, 0.1f, 0.5f) };

		struct Buffer
		{
			float times[2] = { 0, 1 };
			XMFLOAT4 keys[2];
			XMFLOAT3 positions[3] = { XMFLOAT3(0, 0, 0), XMFLOAT3(1, 0, 0), XMFLOAT3(0, 1, 0) };
			uint16_t indices[4] = { 0, 1, 2, 0 }; // (the last one pads the buffer)
		} buffer;
		static_assert(sizeof(Buffer) == 84 && offsetof(Buffer, keys) == 8 && offsetof(Buffer, positions) == 40 && offsetof(Buffer, indices) == 76, "the byteLength and buffer views in the file below");
		XMStoreFloat4(&buffer.keys[0], keys[0]);
		XMStoreFloat4(&buffer.keys[1], keys[1]);
		const std::string light_file = "GLTFImportTest_lights.gltf";
		const char* light_bin = "GLTFImportTest_lights.bin"; // the uri in the file below
		const std::string export_name = "GLTFImportTest_lights_export";
		const std::string json = R"({
	"asset": { "version": "2.0" },
	"extensionsUsed": [ "KHR_lights_punctual" ],
	"extensions": { "KHR_lights_punctual": { "lights": [
		{ "type": "spot", "name": "LampBulb", "spot": { "innerConeAngle": 0.2, "outerConeAngle": 0.6 } },
		{ "type": "spot", "name": "MeshBulb", "spot": { "innerConeAngle": 0.1, "outerConeAngle": 0.5 } },
		{ "type": "directional", "name": "CameraBulb" }
	] } },
	"scene": 0,
	"scenes": [ { "nodes": [ 0, 2, 3, 4 ] } ],
	"nodes": [
		{ "name": "Lamp", "translation": )" + JsonArray(lamp_translation) + R"(, "rotation": )" + JsonArray(lamp_rotation) + R"(, "scale": )" + JsonArray(lamp_scale) + R"(,
			"children": [ 1 ], "extensions": { "KHR_lights_punctual": { "light": 0 } } },
		{ "name": "LampShade", "translation": )" + JsonArray(shade_translation) + R"(, "rotation": )" + JsonArray(shade_rotation) + R"(, "mesh": 0 },
		{ "name": "MeshLight", "translation": )" + JsonArray(mesh_light_translation) + R"(, "rotation": )" + JsonArray(mesh_light_rotation) + R"(, "mesh": 0,
			"extensions": { "KHR_lights_punctual": { "light": 1 } } },
		{ "name": "CameraLight", "translation": )" + JsonArray(camera_light_translation) + R"(, "rotation": )" + JsonArray(camera_light_rotation) + R"(, "camera": 0,
			"extensions": { "KHR_lights_punctual": { "light": 2 } } },
		{ "name": "Follower", "translation": )" + JsonArray(follower_translation) + R"( }
	],
	"cameras": [ { "type": "perspective", "perspective": { "yfov": 0.8, "znear": 0.1, "zfar": 100 } } ],
	"meshes": [ { "primitives": [ { "attributes": { "POSITION": 2 }, "indices": 3 } ] } ],
	"buffers": [ { "byteLength": 84, "uri": "GLTFImportTest_lights.bin" } ],
	"bufferViews": [
		{ "buffer": 0, "byteOffset": 0, "byteLength": 8 },
		{ "buffer": 0, "byteOffset": 8, "byteLength": 32 },
		{ "buffer": 0, "byteOffset": 40, "byteLength": 36 },
		{ "buffer": 0, "byteOffset": 76, "byteLength": 6 }
	],
	"accessors": [
		{ "bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [ 0 ], "max": [ 1 ] },
		{ "bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC4" },
		{ "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3", "min": [ 0, 0, 0 ], "max": [ 1, 1, 0 ] },
		{ "bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR" }
	],
	"animations": [ {
		"name": "LampSwing",
		"samplers": [ { "input": 0, "output": 1 } ],
		"channels": [ { "sampler": 0, "target": { "node": 0, "path": "rotation" } }, { "sampler": 0, "target": { "node": 4, "path": "rotation" } } ]
	} ]
})";
		bool written = false;
		if (FILE* bin = fopen(light_bin, "wb"))
		{
			written = fwrite(&buffer, sizeof(buffer), 1, bin) == 1;
			written &= fclose(bin) == 0;
		}
		if (FILE* gltf = fopen(light_file.c_str(), "w"))
		{
			written &= fputs(json.c_str(), gltf) >= 0;
			written &= fclose(gltf) == 0;
		}
		else
		{
			written = false;
		}
		Check(written, "the glTF file of the light import test could not be written");
		if (!written)
		{
			std::remove(light_file.c_str());
			std::remove(light_bin);
			return;
		}

		// The world matrices of the nodes in the glTF space, with the rotation of an animation key (nullptr: without the animation):
		auto gltf_worlds = [&](const XMVECTOR* key) {
			const XMMATRIX lamp = LocalMatrix(lamp_translation, key != nullptr ? *key : lamp_rotation, lamp_scale);
			const std::pair<std::string, XMMATRIX> worlds[] = {
				{ "Lamp", lamp },
				{ "LampShade", LocalMatrix(shade_translation, shade_rotation) * lamp },
				{ "MeshLight", LocalMatrix(mesh_light_translation, mesh_light_rotation) },
				{ "CameraLight", LocalMatrix(camera_light_translation, camera_light_rotation) },
				{ "Follower", LocalMatrix(follower_translation, key != nullptr ? *key : XMQuaternionIdentity()) },
			};
			wi::vector<std::pair<std::string, XMFLOAT4X4>> result;
			for (const auto& [name, world] : worlds)
			{
				XMStoreFloat4x4(&result.emplace_back(name, XMFLOAT4X4()).second, world);
			}
			return result;
		};
		struct ExpectedLight
		{
			const char* node;
			const char* name;
			LightComponent::LightType type;
		};
		const ExpectedLight expected_lights[] = {
			{ "Lamp", "LampBulb", LightComponent::SPOT },
			{ "MeshLight", "MeshBulb", LightComponent::SPOT },
			{ "CameraLight", "CameraBulb", LightComponent::DIRECTIONAL },
		};

		auto check_scene = [&](Scene& scene, const std::string& what, const XMVECTOR* key) {
			const auto worlds = gltf_worlds(key);
			for (const auto& [name, world] : worlds)
			{
				const TransformComponent* transform = scene.transforms.GetComponent(scene.Entity_FindByName(name));
				Check(transform != nullptr && NearMatrix(transform->world, MirrorMatrix(XMLoadFloat4x4(&world))), what + ": the node " + name + " does not have the world matrix of the glTF file");
			}
			for (const char* name : { "LampShade", "MeshLight" })
			{
				const ObjectComponent* object = scene.objects.GetComponent(scene.Entity_FindByName(name));
				Check(object != nullptr && scene.meshes.Contains(object->meshID), what + ": the node " + name + " does not show its mesh");
			}
			Check(scene.objects.GetCount() == 2, what + ": expected the 2 objects of the mesh nodes, got " + std::to_string(scene.objects.GetCount()));
			Check(scene.cameras.GetCount() == 1 && scene.cameras.Contains(scene.Entity_FindByName("CameraLight")), what + ": the camera is not on its node");
			Check(scene.lights.GetCount() == std::size(expected_lights), what + ": expected 3 lights, got " + std::to_string(scene.lights.GetCount()));
			for (const ExpectedLight& expected : expected_lights)
			{
				const std::string node = expected.node;
				const Entity entity = LightOf(scene, scene.Entity_FindByName(node));
				const LightComponent* light = scene.lights.GetComponent(entity);
				const NameComponent* name = scene.names.GetComponent(entity);
				Check(light != nullptr && name != nullptr && name->name == expected.name && light->GetType() == expected.type, what + ": the node " + node + " does not have its light");
				if (light == nullptr)
					continue;
				auto world = std::find_if(worlds.begin(), worlds.end(), [&](const auto& x) { return x.first == node; });
				const XMMATRIX gltf_world = XMLoadFloat4x4(&world->second);
				Check(XMVectorGetX(XMVector3Length(XMLoadFloat3(&light->direction) - EngineLightDirection(gltf_world))) < 1e-4f, what + ": the light of " + node + " does not point where the glTF light points");
				Check(XMVectorGetX(XMVector3Length(XMLoadFloat3(&light->position) - MirrorMatrix(gltf_world).r[3])) < 1e-4f, what + ": the light of " + node + " is not at its node");
			}
		};
		// Evaluates the animation at its last key (the timer is evaluated when it changes, also if the animation doesn't play):
		auto check_animation = [&](Scene& scene, const std::string& what) {
			AnimationComponent* animation = nullptr;
			for (size_t i = 0; i < scene.animations.GetCount(); ++i)
			{
				const NameComponent* name = scene.names.GetComponent(scene.animations.GetEntity(i));
				if (name != nullptr && name->name == "LampSwing")
				{
					animation = &scene.animations[i];
				}
			}
			Check(animation != nullptr && animation->channels.size() == 2, what + ": the animation is missing");
			if (animation == nullptr)
				return;
			animation->timer = 1;
			scene.Update(0);
			check_scene(scene, what + ", at the last key of the animation", &keys[1]);
		};

		Scene imported;
		ImportModel_GLTF(light_file, imported);
		std::remove(light_file.c_str());
		std::remove(light_bin);
		check_scene(imported, "light import", nullptr);

		// Export, which writes the lights on their nodes:
		RemoveOutputFiles(export_name);
		ExportModel_GLTF(export_name + ".gltf", imported);
		tinygltf::TinyGLTF loader;
		tinygltf::Model model;
		std::string err, warn;
		if (!loader.LoadASCIIFromFile(&model, &err, &warn, export_name + ".gltf"))
		{
			Check(false, "the exported file of the light import test could not be loaded: " + err);
			RemoveOutputFiles(export_name);
			return;
		}
		Scene reimported;
		ImportModel_GLTF(export_name + ".gltf", reimported);
		RemoveOutputFiles(export_name);

		Check(model.nodes.size() == 6, "light export: expected the nodes of the file and the root of the import (6), got " + std::to_string(model.nodes.size()));
		size_t light_nodes = 0;
		for (const tinygltf::Node& node : model.nodes)
		{
			if (node.light < 0)
				continue;
			light_nodes++;
			auto expected = std::find_if(std::begin(expected_lights), std::end(expected_lights), [&](const ExpectedLight& x) { return node.name == x.node; });
			Check(expected != std::end(expected_lights) && node.light < (int)model.lights.size() && model.lights[node.light].name == expected->name, "light export: the node " + node.name + " has a light of an other node");
		}
		Check(light_nodes == std::size(expected_lights), "light export: expected 3 nodes with a light, got " + std::to_string(light_nodes));
		for (size_t i = 0; i < model.nodes.size(); ++i)
		{
			const tinygltf::Node& node = model.nodes[i];
			auto rotation = [&]() { return node.rotation.size() == 4 ? XMVectorSet(float(node.rotation[0]), float(node.rotation[1]), float(node.rotation[2]), float(node.rotation[3])) : XMQuaternionIdentity(); };
			if (node.name == "Lamp")
			{
				Check(SameRotation(rotation(), lamp_rotation), "light export: the light node was not written with its rotation");
				Check(model.animations.size() == 1 && model.animations[0].channels.size() == 2, "light export: the animation was not written");
				for (const tinygltf::Animation& animation : model.animations)
				{
					for (const tinygltf::AnimationChannel& channel : animation.channels)
					{
						if (channel.target_node != (int)i || channel.sampler < 0 || channel.sampler >= (int)animation.samplers.size())
							continue;
						const wi::vector<float> values = ReadAccessor(model, animation.samplers[channel.sampler].output);
						Check(values.size() == 8 && SameRotation(XMLoadFloat4((const XMFLOAT4*)&values[0]), keys[0]) && SameRotation(XMLoadFloat4((const XMFLOAT4*)&values[4]), keys[1]), "light export: the rotation keys of the light node were not written as they are");
					}
				}
			}
			if (node.name == "MeshLight")
			{
				Check(node.mesh >= 0 && node.light >= 0, "light export: the node with a mesh and a light lost one");
			}
			if (node.name == "CameraLight")
			{
				Check(node.camera >= 0 && node.light >= 0, "light export: the node with a camera and a light lost one");
			}
		}
		check_scene(reimported, "light export and import", nullptr);

		check_animation(imported, "light import");
		check_animation(reimported, "light export and import");
	}

	// The entity with the light of a node: the node, or a descendant of it
	Entity LightUnder(const Scene& scene, Entity node)
	{
		for (size_t i = 0; node != INVALID_ENTITY && i < scene.lights.GetCount(); ++i)
		{
			const Entity entity = scene.lights.GetEntity(i);
			if (entity == node || scene.Entity_IsDescendant(entity, node))
				return entity;
		}
		return INVALID_ENTITY;
	}

	// Export of engine lights that have more on their node: a child, a mesh, a camera. The rotation between the engine's
	//	light direction (down) and glTF's (forward) must not turn those, and the import of the export must give the world
	//	matrices and light directions of the scene, also when the rotation animation of a light plays.
	void CheckLightsWithNodes()
	{
		const std::string export_name = "GLTFExportTest_lights_with_nodes";
		Scene scene;
		const Entity material = CreateEntity();
		scene.materials.Create(material);
		const Entity mesh = CreateMorphedMesh(scene, material);

		// A spot light with a non-uniform scale, a child, a mesh and a rotation animation:
		const Entity lantern = CreateNode(scene, "Lantern", XMFLOAT3(1, 2, 3), XMFLOAT3(0.3f, -0.2f, 0.5f));
		scene.transforms.GetComponent(lantern)->scale_local = XMFLOAT3(1, 2, 0.5f);
		scene.transforms.GetComponent(lantern)->SetDirty();
		scene.lights.Create(lantern).SetType(LightComponent::SPOT);
		scene.objects.Create(lantern).meshID = mesh;
		const Entity handle = CreateNode(scene, "Handle", XMFLOAT3(0.5f, -0.25f, -1), XMFLOAT3(0.1f, 0.4f, -0.3f));
		scene.Component_Attach(handle, lantern, true);
		// A directional light with a camera:
		const Entity torch = CreateNode(scene, "Torch", XMFLOAT3(-2, 0.5f, 1), XMFLOAT3(-0.6f, 0.2f, 0.4f));
		scene.lights.Create(torch).SetType(LightComponent::DIRECTIONAL);
		scene.cameras.Create(torch);
		// Spot lights with a scale that mirrors the light rotation, with scale keys that don't, and with scale keys that do:
		const Entity mirror = CreateNode(scene, "Mirror", XMFLOAT3(3, 1, -1), XMFLOAT3(-0.2f, 0.5f, 0.3f));
		scene.transforms.GetComponent(mirror)->scale_local = XMFLOAT3(1, -1, 1);
		scene.transforms.GetComponent(mirror)->SetDirty();
		scene.lights.Create(mirror).SetType(LightComponent::SPOT);
		const Entity scaled_lamp = CreateNode(scene, "ScaledLamp", XMFLOAT3(-3, 1, 0), XMFLOAT3(0.4f, 0.1f, -0.2f));
		scene.lights.Create(scaled_lamp).SetType(LightComponent::SPOT);
		const Entity flipper = CreateNode(scene, "Flipper", XMFLOAT3(0, -2, 2), XMFLOAT3(0.3f, -0.4f, 0.1f));
		scene.lights.Create(flipper).SetType(LightComponent::SPOT);

		const XMVECTOR keys[2] = { XMQuaternionRotationRollPitchYaw(0.2f, 0.4f, 0.1f), XMQuaternionRotationRollPitchYaw(-0.3f, 0.1f, 0.5f) };
		wi::vector<float> key_data(8);
		XMStoreFloat4((XMFLOAT4*)&key_data[0], keys[0]);
		XMStoreFloat4((XMFLOAT4*)&key_data[4], keys[1]);
		const Entity animationEntity = CreateEntity();
		scene.names.Create(animationEntity) = "LanternSwing";
		{
			AnimationComponent& animation = scene.animations.Create(animationEntity);
			animation.samplers.emplace_back().data = CreateData(scene, { 0, 1 }, key_data);
			AnimationComponent::AnimationChannel& channel = animation.channels.emplace_back();
			channel.target = lantern;
			channel.samplerIndex = 0;
			channel.path = Path::ROTATION;
			animation.end = 1;
			const std::pair<Entity, wi::vector<float>> scale_keys[] = { { scaled_lamp, { 1, 1, 1, 2, 2, 2 } }, { flipper, { 1, 1, 1, 1, -1, 1 } } };
			for (const auto& [target, keys] : scale_keys)
			{
				animation.samplers.emplace_back().data = CreateData(scene, { 0, 1 }, keys);
				AnimationComponent::AnimationChannel& scale_channel = animation.channels.emplace_back();
				scale_channel.target = target;
				scale_channel.samplerIndex = int(animation.samplers.size() - 1);
				scale_channel.path = Path::SCALE;
			}
		}

		const char* nodes[] = { "Lantern", "Handle", "Torch", "Mirror", "ScaledLamp", "Flipper" };
		const char* lights[] = { "Lantern", "Torch", "Mirror", "ScaledLamp", "Flipper" };
		struct State
		{
			wi::vector<XMFLOAT4X4> worlds;
			wi::vector<XMFLOAT3> directions;
			wi::vector<XMFLOAT3> positions;
		};
		auto state_of = [&](Scene& of) {
			State state;
			for (const char* name : nodes)
			{
				const TransformComponent* transform = of.transforms.GetComponent(of.Entity_FindByName(name));
				state.worlds.push_back(transform == nullptr ? XMFLOAT4X4() : transform->world);
			}
			for (const char* name : lights)
			{
				const LightComponent* light = of.lights.GetComponent(LightUnder(of, of.Entity_FindByName(name)));
				state.directions.push_back(light == nullptr ? XMFLOAT3() : light->direction);
				state.positions.push_back(light == nullptr ? XMFLOAT3() : light->position);
			}
			return state;
		};
		auto check_state = [&](const State& state, const State& expected, const std::string& what) {
			for (size_t i = 0; i < std::size(nodes); ++i)
			{
				Check(NearMatrix(state.worlds[i], XMLoadFloat4x4(&expected.worlds[i])), what + ": the node " + nodes[i] + " does not have the world matrix of the scene");
			}
			for (size_t i = 0; i < std::size(lights); ++i)
			{
				Check(XMVectorGetX(XMVector3Length(XMLoadFloat3(&state.directions[i]) - XMLoadFloat3(&expected.directions[i]))) < 1e-4f, what + ": the light of " + lights[i] + " does not point where the light of the scene points");
				Check(XMVectorGetX(XMVector3Length(XMLoadFloat3(&state.positions[i]) - XMLoadFloat3(&expected.positions[i]))) < 1e-4f, what + ": the light of " + lights[i] + " is not where the light of the scene is");
			}
		};
		// Evaluates the animation at its last key (the timer is evaluated when it changes, also if the animation doesn't play):
		auto state_at_last_key = [&](Scene& of) {
			for (size_t i = 0; i < of.animations.GetCount(); ++i)
			{
				const NameComponent* name = of.names.GetComponent(of.animations.GetEntity(i));
				if (name != nullptr && name->name == "LanternSwing")
				{
					of.animations[i].timer = 1;
				}
			}
			of.Update(0);
			return state_of(of);
		};

		scene.Update(0);
		const State expected = state_of(scene);
		Check(LightUnder(scene, lantern) == lantern && LightUnder(scene, torch) == torch && std::abs(XMVectorGetX(XMVector3Length(XMLoadFloat3(&expected.directions[0]))) - 1) < 1e-4f, "the lights of the scene with nodes were not made");

		RemoveOutputFiles(export_name);
		ExportModel_GLTF(export_name + ".gltf", scene);
		tinygltf::TinyGLTF loader;
		tinygltf::Model model;
		std::string err, warn;
		if (!loader.LoadASCIIFromFile(&model, &err, &warn, export_name + ".gltf"))
		{
			Check(false, "the exported file of the lights with nodes could not be loaded: " + err);
			RemoveOutputFiles(export_name);
			return;
		}
		Scene imported;
		ImportModel_GLTF(export_name + ".gltf", imported);
		RemoveOutputFiles(export_name);

		// In the file, the light nodes keep their rotation (in the glTF space), mesh and camera, the lights are on child nodes:
		auto find_node = [&](const std::string& name) -> const tinygltf::Node* {
			for (const tinygltf::Node& node : model.nodes)
			{
				if (node.name == name && (node.mesh >= 0 || node.camera >= 0))
					return &node;
			}
			return nullptr;
		};
		auto rotation_of = [](const tinygltf::Node& node) {
			return node.rotation.size() == 4 ? XMVectorSet(float(node.rotation[0]), float(node.rotation[1]), float(node.rotation[2]), float(node.rotation[3])) : XMQuaternionIdentity();
		};
		auto light_children = [&](const tinygltf::Node& node) {
			size_t count = 0;
			for (int child : node.children)
			{
				count += child >= 0 && child < (int)model.nodes.size() && model.nodes[child].light >= 0 ? 1 : 0;
			}
			return count;
		};
		const tinygltf::Node* lantern_node = find_node("Lantern");
		const tinygltf::Node* torch_node = find_node("Torch");
		Check(lantern_node != nullptr && torch_node != nullptr && lantern_node->mesh >= 0 && torch_node->camera >= 0, "the nodes of lights with a mesh or a camera were not written with them");
		if (lantern_node != nullptr && torch_node != nullptr)
		{
			Check(lantern_node->light < 0 && light_children(*lantern_node) == 1 && torch_node->light < 0 && light_children(*torch_node) == 1, "the lights of nodes with more were not written on a child node");
			Check(SameRotation(rotation_of(*lantern_node), MirrorRotation(XMLoadFloat4(&scene.transforms.GetComponent(lantern)->rotation_local))), "the node of a light with more was written with the light rotation");
			for (const tinygltf::Animation& animation : model.animations)
			{
				for (const tinygltf::AnimationChannel& channel : animation.channels)
				{
					if (channel.target_node < 0 || &model.nodes[channel.target_node] != lantern_node || channel.sampler < 0 || channel.sampler >= (int)animation.samplers.size())
						continue;
					const wi::vector<float> values = ReadAccessor(model, animation.samplers[channel.sampler].output);
					Check(values.size() == 8 && SameRotation(XMLoadFloat4((const XMFLOAT4*)&values[0]), MirrorRotation(keys[0])) && SameRotation(XMLoadFloat4((const XMFLOAT4*)&values[4]), MirrorRotation(keys[1])), "the rotation keys of a light with more were written with the light rotation");
				}
			}
		}

		const ObjectComponent* object = imported.objects.GetComponent(imported.Entity_FindByName("Lantern"));
		Check(object != nullptr && imported.meshes.Contains(object->meshID), "the mesh of a light did not import back on its node");
		Check(imported.cameras.Contains(imported.Entity_FindByName("Torch")), "the camera of a light did not import back on its node");
		Check(imported.lights.GetCount() == std::size(lights), "expected the 5 lights with nodes to import back, got " + std::to_string(imported.lights.GetCount()));
		Check(imported.lights.Contains(imported.Entity_FindByName("ScaledLamp")), "a light with scale keys that don't mirror it did not import back as its node");
		Check(!imported.lights.Contains(imported.Entity_FindByName("Mirror")) && !imported.lights.Contains(imported.Entity_FindByName("Flipper")),
			"a light with a scale (or scale keys) that mirrors it was not written on a child node");
		check_state(state_of(imported), expected, "lights with nodes, export and import");

		// An other export writes the same nodes (and the root of the import), it doesn't add nodes for the lights again:
		ExportModel_GLTF(export_name + ".gltf", imported);
		tinygltf::Model again;
		const bool loaded = loader.LoadASCIIFromFile(&again, &err, &warn, export_name + ".gltf");
		RemoveOutputFiles(export_name);
		auto light_nodes = [](const tinygltf::Model& of) {
			return std::count_if(of.nodes.begin(), of.nodes.end(), [](const tinygltf::Node& node) { return node.light >= 0; });
		};
		Check(loaded && again.nodes.size() == model.nodes.size() + 1 && light_nodes(again) == light_nodes(model), "the export of the import of lights with nodes did not write the same nodes");

		check_state(state_at_last_key(imported), state_at_last_key(scene), "lights with nodes, export and import, at the last key of the animation");
	}

	int RunExport(int device_created_pipe)
	{
		wi::jobsystem::Initialize();
		wi::graphics::GetDevice() = new wi::graphics::GraphicsDevice_Vulkan(nullptr); // ends the process with EXIT_FAILURE without a device
		// Tell the parent that the device was created, so that later failures are not taken for a missing device:
		if (write(device_created_pipe, "1", 1) != 1)
			return TEST_FAILED;
		close(device_created_pipe);

		CheckSharedSamplerImport();
		CheckLightNames();
		CheckLightNodes();
		CheckLightsWithNodes();

		RemoveOutputFiles(output_name); // the files of an earlier run must not be checked

		Scene other; // the source of animations retargeted from an other scene
		Scene scene;

		Entity a = CreateNode(scene, "A", XMFLOAT3(0, 0, 1));
		Entity b = CreateNode(scene, "B", XMFLOAT3(1, 0, 0));
		Entity source = CreateNode(scene, "Source", XMFLOAT3(0, 1, 0.5f), XMFLOAT3(0.3f, 0.2f, 0.1f));
		Entity destination = CreateNode(scene, "Destination", XMFLOAT3(0, 2, 0));
		Entity cubic_destination = CreateNode(scene, "CubicDestination", XMFLOAT3(0, 2, 1));
		Entity cubic_rotation_destination = CreateNode(scene, "CubicRotationDestination", XMFLOAT3(0, 2, 2));
		Entity other_destination = CreateNode(scene, "OtherDestination", XMFLOAT3(0, 3, 0));
		Entity missing_source_destination = CreateNode(scene, "MissingSourceDestination", XMFLOAT3(0, 3, 1));
		Entity other_source = CreateNode(other, "OtherSource", XMFLOAT3(0.5f, 1, 0), XMFLOAT3(-0.2f, 0.4f, 0));
		Entity no_transform = CreateEntity();

		XMFLOAT4 light_rotation;
		XMStoreFloat4(&light_rotation, XMQuaternionRotationRollPitchYaw(0.4f, 0.6f, 0.2f));
		Entity light = CreateNode(scene, "Light", XMFLOAT3(0, 5, 0));
		scene.transforms.GetComponent(light)->rotation_local = light_rotation;
		scene.transforms.GetComponent(light)->SetDirty();
		scene.lights.Create(light).SetType(LightComponent::SPOT);
		Entity second_light = CreateNode(scene, "SecondLight", XMFLOAT3(0, 6, 0));
		scene.lights.Create(second_light).SetType(LightComponent::POINT);
		// The first light under a node that comes after the second light: the import meets the lights in the other order
		Entity light_parent = CreateNode(scene, "LightParent", XMFLOAT3(0, 0, 0));
		scene.Component_Attach(light, light_parent, true);
		Entity c = CreateNode(scene, "C", XMFLOAT3(2, 0, 0));
		Entity cubic_destination_copy = CreateNode(scene, "CubicDestinationCopy", XMFLOAT3(1, 2, 1));
		Entity linear_rotation_destination = CreateNode(scene, "LinearRotationDestination", XMFLOAT3(1, 2, 2));

		// Meshes with a morph target, the weights of which can be animated (the first one has two objects)
		Entity material = CreateEntity();
		scene.materials.Create(material);
		Entity mesh = CreateMorphedMesh(scene, material);
		Entity other_mesh = CreateMorphedMesh(scene, material);
		Entity morphed = CreateNode(scene, "Morphed", XMFLOAT3(0, 0, 0));
		Entity morphed_instance = CreateNode(scene, "MorphedInstance", XMFLOAT3(2, 0, 0));
		Entity other_morphed = CreateNode(scene, "OtherMorphed", XMFLOAT3(4, 0, 0));
		scene.objects.Create(morphed).meshID = mesh;
		scene.objects.Create(morphed_instance).meshID = mesh;
		scene.objects.Create(other_morphed).meshID = other_mesh;

		const wi::vector<float> times = { 0, 1 };
		const wi::vector<float> translation_keys = { 0, 0, 1, 0, 0, 2 };
		const wi::vector<float> last_translation_keys = { 0, 0, 3, 0, 0, 4 };
		const wi::vector<float> shared_keys = { 1, 1, 1, 2, 2, 2 };
		const wi::vector<float> retarget_keys = { 0, 0, 1, 1, 0, 0 };
		const wi::vector<float> weight_keys = { 0, 1 };
		const wi::vector<float> light_keys = { light_rotation.x, light_rotation.y, light_rotation.z, light_rotation.w, 0, 0.6f, 0, 0.8f };
		// Cubic spline keys are an in-tangent, a value and an out-tangent:
		const wi::vector<float> cubic_translation_keys = { 0.5f, 0, 0, 0, 0, 1, 1, 0.5f, 0, 0, 1, 0, 1, 0, 0, 0.2f, 0.3f, 0.4f };
		const XMVECTOR q0 = XMQuaternionRotationRollPitchYaw(0.1f, 0.2f, 0.3f);
		const XMVECTOR q1 = -XMQuaternionRotationRollPitchYaw(0.5f, -0.4f, 0.2f); // in the other hemisphere of q0
		const XMFLOAT4 m[4] = { XMFLOAT4(0.1f, 0.2f, -0.1f, 0.3f), XMFLOAT4(-0.2f, 0.1f, 0.4f, 0.05f), XMFLOAT4(0.3f, -0.1f, 0.2f, -0.2f), XMFLOAT4(0.05f, 0.2f, 0.1f, 0.1f) };
		const wi::vector<float> cubic_rotation_keys = {
			m[0].x, m[0].y, m[0].z, m[0].w, XMVectorGetX(q0), XMVectorGetY(q0), XMVectorGetZ(q0), XMVectorGetW(q0), m[1].x, m[1].y, m[1].z, m[1].w,
			m[2].x, m[2].y, m[2].z, m[2].w, XMVectorGetX(q1), XMVectorGetY(q1), XMVectorGetZ(q1), XMVectorGetW(q1), m[3].x, m[3].y, m[3].z, m[3].w,
		};
		Entity translation_data = CreateData(scene, times, translation_keys);
		Entity last_translation_data = CreateData(scene, times, last_translation_keys);
		Entity shared_data = CreateData(scene, times, shared_keys);
		Entity retarget_data = CreateData(scene, times, retarget_keys);
		Entity cubic_translation_data = CreateData(scene, times, cubic_translation_keys);
		Entity cubic_rotation_data = CreateData(scene, times, cubic_rotation_keys);
		Entity weight_data = CreateData(scene, times, weight_keys);
		Entity light_data = CreateData(scene, times, light_keys);
		Entity empty_data = CreateData(scene, {}, {});
		Entity other_data = CreateData(other, times, retarget_keys);
		Entity other_shared_data = CreateData(other, times, shared_keys);
		const wi::vector<float> linear_rotation_keys = {
			XMVectorGetX(q0), XMVectorGetY(q0), XMVectorGetZ(q0), XMVectorGetW(q0), XMVectorGetX(q1), XMVectorGetY(q1), XMVectorGetZ(q1), XMVectorGetW(q1),
		};
		Entity linear_rotation_data = CreateData(scene, times, linear_rotation_keys);
		// An other scene can have data on the same entity as this scene (when entities are not remapped):
		const wi::vector<float> other_weight_keys = { 0.3f, 0.7f };
		const Entity other_weight_data = weight_data;
		other.animation_datas.Create(other_weight_data).keyframe_times = times;
		other.animation_datas.GetComponent(other_weight_data)->keyframe_data = other_weight_keys;

		XMFLOAT4X4 retarget_dst, retarget_src;
		const XMVECTOR qD = XMQuaternionRotationRollPitchYaw(0.3f, 0.5f, 0.7f);
		const XMVECTOR qP = XMQuaternionRotationRollPitchYaw(-0.2f, 0.4f, 0.1f);
		XMStoreFloat4x4(&retarget_dst, XMMatrixRotationQuaternion(qD) * XMMatrixTranslation(1, 2, 3));
		XMStoreFloat4x4(&retarget_src, XMMatrixRotationQuaternion(qP) * XMMatrixTranslation(-1, 0, 2));

		Entity animationEntity = CreateEntity();
		scene.names.Create(animationEntity) = "anim";
		{
			AnimationComponent& animation = scene.animations.Create(animationEntity);
			auto add_sampler = [&](Entity data, const Scene* data_scene = nullptr, AnimationComponent::AnimationSampler::Mode mode = AnimationComponent::AnimationSampler::Mode::LINEAR) {
				AnimationComponent::AnimationSampler& sampler = animation.samplers.emplace_back();
				sampler.data = data;
				sampler.scene = data_scene;
				sampler.mode = mode;
				return int(animation.samplers.size() - 1);
			};
			auto add_channel = [&](Entity target, int sampler, Path path, int retarget = -1) {
				AnimationComponent::AnimationChannel& channel = animation.channels.emplace_back();
				channel.target = target;
				channel.samplerIndex = sampler;
				channel.path = path;
				channel.retargetIndex = retarget;
			};
			auto add_retarget = [&](Entity source_entity) {
				AnimationComponent::RetargetSourceData& retarget = animation.retargets.emplace_back();
				retarget.source = source_entity;
				retarget.dstRelativeMatrix = retarget_dst;
				retarget.srcRelativeParentMatrix = retarget_src;
				return int(animation.retargets.size() - 1);
			};
			const auto cubic = AnimationComponent::AnimationSampler::Mode::CUBICSPLINE;
			const int translation_sampler = add_sampler(translation_data);
			const int shared_sampler = add_sampler(shared_data);
			add_channel(a, translation_sampler, Path::TRANSLATION); // replaced by the later channel of A
			add_channel(a, translation_sampler, Path::LIGHT_COLOR); // nothing to animate in glTF
			add_channel(INVALID_ENTITY, translation_sampler, Path::TRANSLATION); // no target
			add_channel(no_transform, translation_sampler, Path::TRANSLATION); // target without a node
			add_channel(b, add_sampler(empty_data), Path::TRANSLATION); // no keys
			add_channel(b, 1000, Path::TRANSLATION); // invalid sampler
			add_channel(a, add_sampler(weight_data), Path::WEIGHTS); // no morph targets
			add_channel(morphed, add_sampler(weight_data), Path::WEIGHTS); // exported for both objects of the mesh
			add_channel(b, shared_sampler, Path::TRANSLATION); // exported, mirrored
			add_channel(b, shared_sampler, Path::SCALE); // exported, not mirrored
			add_channel(destination, add_sampler(retarget_data), Path::TRANSLATION, add_retarget(source)); // baked
			add_channel(other_destination, add_sampler(other_data, &other), Path::TRANSLATION, add_retarget(other_source)); // baked
			add_channel(a, add_sampler(last_translation_data), Path::TRANSLATION); // the last channel of A is exported
			const int cubic_translation_sampler = add_sampler(cubic_translation_data, nullptr, cubic);
			const int cubic_translation_retarget = add_retarget(source);
			add_channel(cubic_destination, cubic_translation_sampler, Path::TRANSLATION, cubic_translation_retarget); // baked with tangents
			add_channel(cubic_destination_copy, cubic_translation_sampler, Path::TRANSLATION, cubic_translation_retarget); // the same baked keys
			add_channel(linear_rotation_destination, add_sampler(linear_rotation_data), Path::ROTATION, add_retarget(source)); // baked the short way
			const int other_shared_sampler = add_sampler(other_shared_data, &other);
			add_channel(c, other_shared_sampler, Path::TRANSLATION); // keys of an other scene, mirrored
			add_channel(c, other_shared_sampler, Path::SCALE); // the same keys, not mirrored
			add_channel(cubic_rotation_destination, add_sampler(cubic_rotation_data, nullptr, cubic), Path::ROTATION, add_retarget(source)); // baked with tangents
			add_channel(missing_source_destination, add_sampler(other_data, &other), Path::TRANSLATION, add_retarget(no_transform)); // played as it is
			add_channel(light, add_sampler(light_data), Path::ROTATION); // with the light's forward flip
			add_channel(second_light, add_sampler(light_data), Path::ROTATION); // the same keys as the first light
			add_channel(other_mesh, add_sampler(other_weight_data, &other), Path::WEIGHTS); // the mesh as target, weights of an other scene
		}
		{
			Entity entity = CreateEntity();
			scene.names.Create(entity) = "nothing_to_export";
			AnimationComponent& animation = scene.animations.Create(entity);
			animation.samplers.emplace_back().data = translation_data;
			AnimationComponent::AnimationChannel& channel = animation.channels.emplace_back();
			channel.target = a;
			channel.samplerIndex = 0;
			channel.path = Path::MATERIAL_COLOR;
		}

		const TransformComponent& source_transform = *scene.transforms.GetComponent(source);
		const wi::vector<float> expected_same = MirrorTranslations(RetargetTranslations(source_transform, retarget_dst, retarget_src, retarget_keys));
		const wi::vector<float> expected_other = MirrorTranslations(RetargetTranslations(*other.transforms.GetComponent(other_source), retarget_dst, retarget_src, retarget_keys));
		wi::vector<float> expected_cubic_translation;
		for (size_t key = 0; key < 2; ++key)
		{
			auto at = [&](size_t i) { return XMVectorSet(cubic_translation_keys[key * 9 + i * 3], cubic_translation_keys[key * 9 + i * 3 + 1], cubic_translation_keys[key * 9 + i * 3 + 2], 0); };
			const XMVECTOR value = RetargetTranslation(source_transform, retarget_dst, retarget_src, at(1));
			const XMVECTOR results[3] = {
				RetargetTranslation(source_transform, retarget_dst, retarget_src, at(1) + at(0)) - value, // the translation is retargeted by an affine function
				value,
				RetargetTranslation(source_transform, retarget_dst, retarget_src, at(1) + at(2)) - value,
			};
			for (const XMVECTOR& result : results)
			{
				expected_cubic_translation.push_back(XMVectorGetX(result));
				expected_cubic_translation.push_back(XMVectorGetY(result));
				expected_cubic_translation.push_back(-XMVectorGetZ(result));
			}
		}

		const TransformComponent a_original = *scene.transforms.GetComponent(a);
		const TransformComponent light_original = *scene.transforms.GetComponent(light);
		const size_t sampler_count = scene.animations.GetComponent(animationEntity)->samplers.size();
		wi::vector<int> channel_samplers;
		for (const AnimationComponent::AnimationChannel& channel : scene.animations.GetComponent(animationEntity)->channels)
		{
			channel_samplers.push_back(channel.samplerIndex);
		}
		const size_t data_count = scene.animation_datas.GetCount();

		// The scene is restored after each export, so exporting again doesn't change it (the light node's forward flip
		//	must not be rounded into the light's rotation). The file of the last export is checked below.
		auto same_transform = [](const TransformComponent& x, const TransformComponent& y) {
			return std::memcmp(&x.scale_local, &y.scale_local, sizeof(XMFLOAT3)) == 0 && std::memcmp(&x.rotation_local, &y.rotation_local, sizeof(XMFLOAT4)) == 0 && std::memcmp(&x.translation_local, &y.translation_local, sizeof(XMFLOAT3)) == 0;
		};
		for (int i = 1; i <= 3; ++i)
		{
			ExportModel_GLTF(output_file, scene);
			const std::string after = " after export " + std::to_string(i);
			Check(same_transform(*scene.transforms.GetComponent(a), a_original), "the transforms were not restored" + after);
			Check(same_transform(*scene.transforms.GetComponent(light), light_original), "the transform of the light was not restored" + after);
		}
		Check(scene.animation_datas.GetComponent(translation_data)->keyframe_data == translation_keys && scene.animation_datas.GetComponent(shared_data)->keyframe_data == shared_keys, "the animation keys were not restored after the export");
		const AnimationComponent& restored = *scene.animations.GetComponent(animationEntity);
		Check(std::memcmp(&restored.retargets[0].dstRelativeMatrix, &retarget_dst, sizeof(XMFLOAT4X4)) == 0 && std::memcmp(&restored.retargets[1].srcRelativeParentMatrix, &retarget_src, sizeof(XMFLOAT4X4)) == 0, "the retarget matrices were not restored after the export");
		bool same_samplers = restored.samplers.size() == sampler_count && restored.channels.size() == channel_samplers.size() && scene.animation_datas.GetCount() == data_count;
		for (size_t i = 0; same_samplers && i < channel_samplers.size(); ++i)
		{
			same_samplers = restored.channels[i].samplerIndex == channel_samplers[i];
		}
		Check(same_samplers, "the samplers or animation data of the scene were changed by the export");

		// Load the file and check it
		tinygltf::TinyGLTF loader;
		tinygltf::Model model;
		std::string err, warn;
		if (!loader.LoadASCIIFromFile(&model, &err, &warn, output_file))
		{
			Check(false, "the exported file could not be loaded: " + err);
			RemoveOutputFiles(output_name);
			return TEST_FAILED;
		}
		// Importing the file back gives the keys of the scene (in the engine space)
		Scene imported;
		ImportModel_GLTF(output_file, imported);
		RemoveOutputFiles(output_name);
		auto imported_keys = [&](const std::string& name, Path path) {
			const Entity target = imported.Entity_FindByName(name);
			for (size_t i = 0; i < imported.animations.GetCount(); ++i)
			{
				const AnimationComponent& animation = imported.animations[i];
				for (const AnimationComponent::AnimationChannel& channel : animation.channels)
				{
					if (channel.target == target && channel.path == path && channel.samplerIndex >= 0 && channel.samplerIndex < (int)animation.samplers.size())
					{
						const AnimationDataComponent* data = imported.animation_datas.GetComponent(animation.samplers[channel.samplerIndex].data);
						if (data != nullptr)
							return data->keyframe_data;
					}
				}
			}
			return wi::vector<float>();
		};
		{
			const LightComponent* imported_light = imported.lights.GetComponent(LightOf(imported, imported.Entity_FindByName("Light")));
			const LightComponent* imported_second_light = imported.lights.GetComponent(LightOf(imported, imported.Entity_FindByName("SecondLight")));
			Check(imported.lights.GetCount() == 2 && imported_light != nullptr && imported_second_light != nullptr &&
				imported_light->GetType() == LightComponent::SPOT && imported_second_light->GetType() == LightComponent::POINT,
				"the lights did not import back as they were");
			Check(imported.lights.Contains(imported.Entity_FindByName("Light")) && imported.lights.Contains(imported.Entity_FindByName("SecondLight")),
				"the lights without more on their nodes did not import back as their nodes");
		}
		Check(Near(imported_keys("A", Path::TRANSLATION), last_translation_keys), "the translation keys did not import back as they were");
		{
			// The light can be on a child of the animated node, then the light turns by the rotation of the child, then by the keys:
			const Entity light_node = imported.Entity_FindByName("Light");
			const Entity light_entity = LightOf(imported, light_node);
			XMVECTOR child_rotation = XMQuaternionIdentity();
			if (light_entity != light_node && imported.transforms.Contains(light_entity))
			{
				child_rotation = XMLoadFloat4(&imported.transforms.GetComponent(light_entity)->rotation_local);
			}
			const wi::vector<float> keys = imported_keys("Light", Path::ROTATION);
			bool same = light_entity != INVALID_ENTITY && keys.size() == light_keys.size();
			for (size_t i = 0; same && i + 4 <= keys.size(); i += 4)
			{
				same = SameRotation(XMQuaternionMultiply(child_rotation, XMLoadFloat4((const XMFLOAT4*)&keys[i])), XMLoadFloat4((const XMFLOAT4*)&light_keys[i]));
			}
			Check(same, "the rotation keys of a light did not import back as they were");
		}

		auto node_index = [&](const std::string& name) {
			for (size_t i = 0; i < model.nodes.size(); ++i)
			{
				if (model.nodes[i].name == name)
					return int(i);
			}
			return -1;
		};

		Check(model.animations.size() == 1 && model.animations[0].name == "anim", "animations without anything to export were written");
		if (model.animations.size() != 1)
			return TEST_FAILED;
		const tinygltf::Animation& exported = model.animations[0];
		Check(exported.channels.size() == 17, "expected 17 exported channels, got " + std::to_string(exported.channels.size()));
		{
			wi::vector<uint8_t> used(exported.samplers.size(), 0);
			for (const tinygltf::AnimationChannel& channel : exported.channels)
			{
				if (channel.sampler >= 0 && channel.sampler < (int)used.size())
				{
					used[channel.sampler] = 1;
				}
			}
			Check(std::find(used.begin(), used.end(), 0) == used.end(), "a sampler that no channel uses was written");
		}

		for (const tinygltf::AnimationSampler& sampler : exported.samplers)
		{
			const bool valid = sampler.input >= 0 && sampler.input < (int)model.accessors.size() && sampler.output >= 0 && sampler.output < (int)model.accessors.size();
			Check(valid, "a sampler has an invalid accessor");
			if (!valid)
				return TEST_FAILED;
			const tinygltf::Accessor& input = model.accessors[sampler.input];
			Check(input.count > 0 && model.accessors[sampler.output].count > 0, "a sampler has an empty accessor");
			Check(input.minValues.size() == 1 && input.maxValues.size() == 1, "a sampler input has no min and max");
		}

		auto find_values = [&](const std::string& node, const std::string& path) {
			const int target = node_index(node);
			for (const tinygltf::AnimationChannel& channel : exported.channels)
			{
				if (channel.target_node == target && channel.target_path == path && channel.sampler >= 0 && channel.sampler < (int)exported.samplers.size())
					return ReadAccessor(model, exported.samplers[channel.sampler].output);
			}
			return wi::vector<float>();
		};
		std::set<std::pair<int, std::string>> targets;
		for (const tinygltf::AnimationChannel& channel : exported.channels)
		{
			Check(channel.target_node >= 0 && channel.target_node < (int)model.nodes.size(), "a channel has an invalid target node");
			Check(channel.sampler >= 0 && channel.sampler < (int)exported.samplers.size(), "a channel has an invalid sampler");
			Check(channel.target_path == "translation" || channel.target_path == "rotation" || channel.target_path == "scale" || channel.target_path == "weights", "a channel has an invalid path");
			if (channel.target_path == "weights")
			{
				const int node = channel.target_node;
				Check(node == node_index("Morphed") || node == node_index("MorphedInstance") || node == node_index("OtherMorphed"), "weights were exported for a node without morph targets");
			}
			Check(targets.insert({ channel.target_node, channel.target_path }).second, "a path of a node is animated by more than one channel");
		}

		Check(Near(find_values("A", "translation"), MirrorTranslations(last_translation_keys)), "the last translation channel of a node was not the exported one");
		Check(Near(find_values("Morphed", "weights"), weight_keys) && Near(find_values("MorphedInstance", "weights"), weight_keys), "the morph target weights were not exported for every object of the mesh");
		Check(Near(find_values("OtherMorphed", "weights"), other_weight_keys), "the weights of an other scene, with the mesh as target, were not exported (or mixed up with the same entity of this scene)");
		Check(Near(find_values("B", "translation"), MirrorTranslations(shared_keys)), "the translation keys shared with a scale channel were not exported mirrored");
		Check(Near(find_values("B", "scale"), shared_keys), "the scale keys shared with a translation channel were not exported as they are");
		Check(Near(find_values("Destination", "translation"), expected_same), "the channel retargeted from this scene was not baked");
		Check(Near(find_values("OtherDestination", "translation"), expected_other), "the channel retargeted from an other scene was not baked");
		Check(Near(find_values("MissingSourceDestination", "translation"), MirrorTranslations(retarget_keys)), "the keys of an other scene without a retarget source were not exported mirrored");
		Check(Near(find_values("C", "translation"), MirrorTranslations(shared_keys)) && Near(find_values("C", "scale"), shared_keys), "keys of an other scene used by a translation and a scale channel were not exported mirrored only for the translation");
		{
			auto sampler_of = [&](const std::string& name) {
				for (const tinygltf::AnimationChannel& channel : exported.channels)
				{
					if (channel.target_node == node_index(name))
						return channel.sampler;
				}
				return -1;
			};
			Check(sampler_of("CubicDestination") >= 0 && sampler_of("CubicDestination") == sampler_of("CubicDestinationCopy"), "the same sampler and retarget were baked twice");
			// With rigid retarget matrices the retargeting is qD * q * qP (in the order of XMQuaternionMultiply), as either quaternion
			const wi::vector<float> linear = find_values("LinearRotationDestination", "rotation");
			bool linear_retargeted = linear.size() == 8;
			for (size_t key = 0; linear_retargeted && key < 2; ++key)
			{
				const XMVECTOR expected = XMQuaternionMultiply(XMQuaternionMultiply(qD, XMLoadFloat4((const XMFLOAT4*)&linear_rotation_keys[key * 4])), qP);
				linear_retargeted = SameRotation(MirrorRotation(XMLoadFloat4((const XMFLOAT4*)&linear[key * 4])), expected);
			}
			Check(linear_retargeted, "baked linear rotation keys are not the retargeted rotations");
			Check(linear.size() == 8 && XMVectorGetX(XMVector4Dot(XMLoadFloat4((const XMFLOAT4*)&linear[0]), XMLoadFloat4((const XMFLOAT4*)&linear[4]))) > 0, "baked linear rotation keys are not in the same hemisphere");
		}
		Check(Near(find_values("CubicDestination", "translation"), expected_cubic_translation, 1e-3f), "the tangents of a retargeted cubic spline translation were not baked as derivatives");

		// Cubic rotation: with rigid retarget matrices the retargeting is the linear qD * q * qP (in the order of XMQuaternionMultiply),
		//	so the baked tangents are the same function of the tangents (up to the sign of the key)
		{
			const wi::vector<float> values = find_values("CubicRotationDestination", "rotation");
			bool ok = values.size() == cubic_rotation_keys.size();
			for (size_t key = 0; ok && key < 2; ++key)
			{
				auto exported_at = [&](size_t i) { return MirrorRotation(XMLoadFloat4((const XMFLOAT4*)&values[key * 12 + i * 4])); }; // back to the engine space
				auto source_at = [&](size_t i) { return XMLoadFloat4((const XMFLOAT4*)&cubic_rotation_keys[key * 12 + i * 4]); };
				auto retargeted = [&](XMVECTOR Q) { return XMQuaternionMultiply(XMQuaternionMultiply(qD, Q), qP); };
				const XMVECTOR expected_value = retargeted(source_at(1));
				const float sign = XMVectorGetX(XMVector4Dot(exported_at(1), expected_value)) < 0 ? -1.0f : 1.0f;
				for (size_t i = 0; i < 3; ++i)
				{
					ok &= XMVectorGetX(XMVector4Length(exported_at(i) - sign * retargeted(source_at(i)))) < 1e-3f;
				}
			}
			Check(ok, "the tangents of a retargeted cubic spline rotation were not baked as derivatives");
			if (values.size() == cubic_rotation_keys.size())
			{
				const float exported_relation = XMVectorGetX(XMVector4Dot(XMLoadFloat4((const XMFLOAT4*)&values[4]), XMLoadFloat4((const XMFLOAT4*)&values[16])));
				Check(exported_relation < 0, "the hemisphere relation of retargeted cubic spline rotation keys was not kept");
			}
		}

		// The first light key is the light's own rotation, which is exported as the node's rotation (with the forward flip)
		{
			const wi::vector<float> values = find_values("Light", "rotation");
			const int node = node_index("Light");
			Check(node >= 0 && model.nodes[node].rotation.size() == 4, "the light node was not exported");
			if (node >= 0 && model.nodes[node].rotation.size() == 4)
			{
				const std::vector<double>& rotation = model.nodes[node].rotation;
				const wi::vector<float> node_rotation = { float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]) };
				Check(values.size() == 8 && Near(wi::vector<float>(values.begin(), values.begin() + 4), node_rotation), "the rotation keys of a light did not get the light's forward flip");
				// The node's rotation is the light's rotation in the glTF space, rotated like TransformComponent::RotateRollPitchYaw() rotates it
				TransformComponent flipped = light_original;
				XMStoreFloat4(&flipped.rotation_local, MirrorRotation(XMLoadFloat4(&light_original.rotation_local)));
				flipped.RotateRollPitchYaw(XMFLOAT3(-XM_PIDIV2, 0, 0));
				const XMFLOAT4& expected = flipped.rotation_local;
				Check(Near(node_rotation, { expected.x, expected.y, expected.z, expected.w }, 1e-6f), "the light node was not exported with the light's rotation after the forward flip");
			}
			auto output_of = [&](const std::string& name) {
				for (const tinygltf::AnimationChannel& channel : exported.channels)
				{
					if (channel.target_node == node_index(name) && channel.target_path == "rotation")
						return exported.samplers[channel.sampler].output;
				}
				return -1;
			};
			Check(output_of("Light") >= 0 && output_of("Light") == output_of("SecondLight"), "the same changed keys of two lights were written twice");
		}

		if (failures > 0)
			return TEST_FAILED;
		std::cout << "GLTFExport test passed\n";
		return 0;
	}
}

int main(int argc, char** argv)
{
	if (argc > 2 && std::strcmp(argv[1], "--child") == 0)
	{
		// No error message box if the device can't be created:
		unsetenv("DISPLAY");
		unsetenv("WAYLAND_DISPLAY");
		unsetenv("XDG_RUNTIME_DIR");
		const int result = RunExport(std::atoi(argv[2]));
		std::cout.flush();
		std::cerr.flush();
		std::_Exit(result); // without destroying the device in the static destructors
	}

	int device_created[2];
	if (pipe(device_created) != 0)
	{
		std::cerr << "GLTFExport test failed: pipe() failed\n";
		return 1;
	}
	pid_t pid = fork();
	if (pid < 0)
	{
		std::cerr << "GLTFExport test failed: fork() failed\n";
		return 1;
	}
	if (pid == 0)
	{
		close(device_created[0]);
		const std::string fd = std::to_string(device_created[1]);
		execl("/proc/self/exe", argv[0], "--child", fd.c_str(), (char*)nullptr);
		_exit(127);
	}
	close(device_created[1]);
	int status = 0;
	pid_t waited = -1;
	do
	{
		waited = waitpid(pid, &status, 0);
	} while (waited < 0 && errno == EINTR);
	if (waited != pid)
	{
		std::cerr << "GLTFExport test failed: waitpid() failed\n";
		return 1;
	}
	char byte = 0;
	const bool created = read(device_created[0], &byte, 1) == 1;
	close(device_created[0]);

	if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
		return 0;
	if (!created && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE)
	{
		std::cout << "GLTFExport test skipped: no Vulkan device could be created\n";
		return TEST_SKIPPED;
	}
	if (WIFSIGNALED(status))
	{
		std::cerr << "GLTFExport test failed: the test process (import or export) crashed with signal " << WTERMSIG(status) << '\n';
	}
	else if (WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE)
	{
		std::cerr << "GLTFExport test failed: the test process (import or export) ended with EXIT_FAILURE\n";
	}
	return 1;
}
