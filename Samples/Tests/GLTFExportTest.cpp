#include "WickedEngine.h"
#include "wiGraphicsDevice_Vulkan.h"
#include "ModelImporter.h"

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include <algorithm>
#include <cerrno>
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
//	The exporter updates the scene, which needs a graphics device: the test runs in a child process that creates a Vulkan
//	device (like Mesa's software lavapipe), and it is skipped if there is none.

using namespace wi::ecs;
using namespace wi::scene;
using Path = AnimationComponent::AnimationChannel::Path;

namespace
{
	constexpr int TEST_SKIPPED = 77;
	constexpr int TEST_FAILED = 2;
	const char* output_file = "GLTFExportTest_output.gltf";

	// Removes the output of the export: the file and its buffers (tinygltf names them after the file, numbered after the first one)
	void RemoveOutputFiles()
	{
		std::remove(output_file);
		std::remove("GLTFExportTest_output.bin");
		for (int i = 0; i < 64; ++i)
		{
			std::remove(("GLTFExportTest_output" + std::to_string(i) + ".bin").c_str());
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

	// What the runtime evaluates for a translation key of a retargeted channel (see Scene::RunAnimationUpdateSystem())
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

	// Import: a glTF sampler that animates the rotation of a light node and of an other node. The light's keys get the
	//	rotation of the light (glTF lights point forward, engine lights down), the other node's keys must not
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

	int RunExport(int device_created_pipe)
	{
		wi::jobsystem::Initialize();
		wi::graphics::GetDevice() = new wi::graphics::GraphicsDevice_Vulkan(nullptr); // ends the process with EXIT_FAILURE without a device
		// Tell the parent that the device was created, so that later failures are not taken for a missing device:
		if (write(device_created_pipe, "1", 1) != 1)
			return TEST_FAILED;
		close(device_created_pipe);

		CheckSharedSamplerImport();

		RemoveOutputFiles(); // the files of an earlier run must not be checked

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
		const size_t sampler_count = scene.animations.GetComponent(animationEntity)->samplers.size();
		wi::vector<int> channel_samplers;
		for (const AnimationComponent::AnimationChannel& channel : scene.animations.GetComponent(animationEntity)->channels)
		{
			channel_samplers.push_back(channel.samplerIndex);
		}
		const size_t data_count = scene.animation_datas.GetCount();

		ExportModel_GLTF(output_file, scene);

		// The scene is restored after the export
		const TransformComponent& a_restored = *scene.transforms.GetComponent(a);
		Check(std::memcmp(&a_restored.translation_local, &a_original.translation_local, sizeof(XMFLOAT3)) == 0 && std::memcmp(&a_restored.rotation_local, &a_original.rotation_local, sizeof(XMFLOAT4)) == 0, "the transforms were not restored after the export");
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
			RemoveOutputFiles();
			return TEST_FAILED;
		}
		// Importing the file back gives the keys of the scene (in the engine space)
		Scene imported;
		ImportModel_GLTF(output_file, imported);
		RemoveOutputFiles();
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
			const LightComponent* imported_light = imported.lights.GetComponent(imported.Entity_FindByName("Light"));
			const LightComponent* imported_second_light = imported.lights.GetComponent(imported.Entity_FindByName("SecondLight"));
			Check(imported.lights.GetCount() == 2 && imported_light != nullptr && imported_second_light != nullptr &&
				imported_light->GetType() == LightComponent::SPOT && imported_second_light->GetType() == LightComponent::POINT,
				"the lights did not import back as they were");
		}
		Check(Near(imported_keys("A", Path::TRANSLATION), last_translation_keys), "the translation keys did not import back as they were");
		{
			const wi::vector<float> keys = imported_keys("Light", Path::ROTATION);
			bool same = keys.size() == light_keys.size();
			for (size_t i = 0; same && i + 4 <= keys.size(); i += 4)
			{
				same = SameRotation(XMLoadFloat4((const XMFLOAT4*)&keys[i]), XMLoadFloat4((const XMFLOAT4*)&light_keys[i]));
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
