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
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

// Regression test for the STEP animation sampler, which holds the value of a key until the time of the next key (like glTF STEP),
//	and the NEAREST sampler, which changes halfway between keys (this was STEP before AnimationComponent version 3):
//	- STEP samplers of older AnimationComponent versions are loaded as NEAREST, the current version keeps STEP
//	- a glTF STEP animation is imported as STEP and sampled at 0, 25, 50, 75 and 99% between its keys, also after it is
//	  exported and imported again
//	- a NEAREST animation is exported as glTF STEP keys that change halfway, which sample the same after they are imported
//	- STEP holds the keys of the light, material and camera paths too
//	Sampling updates the scene, which needs a graphics device: that part runs in a child process that creates a Vulkan device
//	(like Mesa's software lavapipe), and it is skipped if there is none.

using namespace wi::ecs;
using namespace wi::scene;
using Path = AnimationComponent::AnimationChannel::Path;
using Mode = AnimationComponent::AnimationSampler::Mode;

namespace
{
	constexpr int TEST_SKIPPED = 77;
	constexpr int TEST_FAILED = 2;
	const char* input_file = "GLTFStepAnimationTest_input.gltf";
	const char* output_file = "GLTFStepAnimationTest_output.gltf";

	// Removes the files of the test: the input file and the output with its buffers (tinygltf names them after the file, numbered after the first one)
	void RemoveFiles()
	{
		std::remove(input_file);
		std::remove(output_file);
		std::remove("GLTFStepAnimationTest_output.bin");
		for (int i = 0; i < 64; ++i)
		{
			std::remove(("GLTFStepAnimationTest_output" + std::to_string(i) + ".bin").c_str());
		}
	}

	int failures = 0;

	void Check(bool condition, const std::string& message)
	{
		if (!condition)
		{
			std::cerr << "GLTFStepAnimation test failed: " << message << '\n';
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
	// The same rotation, as either of its two quaternions (the distance is about half the angle between the rotations)
	bool SameRotation(XMVECTOR q0, XMVECTOR q1, float tolerance = 1e-4f)
	{
		return std::min(XMVectorGetX(XMVector4Length(q0 - q1)), XMVectorGetX(XMVector4Length(q0 + q1))) < tolerance;
	}
	bool SameRotations(const wi::vector<float>& a, const wi::vector<float>& b)
	{
		if (a.size() != b.size())
			return false;
		for (size_t i = 0; i + 4 <= a.size(); i += 4)
		{
			if (!SameRotation(XMLoadFloat4((const XMFLOAT4*)&a[i]), XMLoadFloat4((const XMFLOAT4*)&b[i])))
				return false;
		}
		return true;
	}

	// The keys of a value at the given indices (each key is key_size floats)
	wi::vector<float> Keys(const wi::vector<float>& values, size_t key_size, std::initializer_list<size_t> indices)
	{
		wi::vector<float> result;
		for (size_t index : indices)
		{
			result.insert(result.end(), values.begin() + index * key_size, values.begin() + (index + 1) * key_size);
		}
		return result;
	}
	wi::vector<float> Key(const wi::vector<float>& values, size_t key_size, size_t index)
	{
		return Keys(values, key_size, { index });
	}
	// Mirrors keys between the glTF and the engine space: translation Z, rotation X and Y are negated
	wi::vector<float> MirrorTranslations(wi::vector<float> keys)
	{
		for (size_t i = 2; i < keys.size(); i += 3)
		{
			keys[i] = -keys[i];
		}
		return keys;
	}
	wi::vector<float> MirrorRotations(wi::vector<float> keys)
	{
		for (size_t i = 0; i + 1 < keys.size(); i += 4)
		{
			keys[i] = -keys[i];
			keys[i + 1] = -keys[i + 1];
		}
		return keys;
	}
	wi::vector<float> Rotations(std::initializer_list<XMFLOAT3> rollPitchYaws)
	{
		wi::vector<float> result;
		for (const XMFLOAT3& rollPitchYaw : rollPitchYaws)
		{
			XMFLOAT4 rotation;
			XMStoreFloat4(&rotation, XMQuaternionRotationRollPitchYaw(rollPitchYaw.x, rollPitchYaw.y, rollPitchYaw.z));
			result.insert(result.end(), { rotation.x, rotation.y, rotation.z, rotation.w });
		}
		return result;
	}

	// The glTF STEP animation (in the glTF space), the keys of each time are held until the next time:
	const wi::vector<float> step_times = { 0, 1, 3 };
	const wi::vector<float> step_translations = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	const wi::vector<float> step_rotations = Rotations({ XMFLOAT3(0.1f, 0.2f, 0.3f), XMFLOAT3(0.5f, -0.4f, 0.2f), XMFLOAT3(-0.3f, 0.6f, -0.1f) });
	const wi::vector<float> step_scales = { 1, 1, 1, 2, 2, 2, 0.5f, 3, 1 };
	const wi::vector<float> step_weights = { 0, 1, 1, 0, 0.5f, 0.25f }; // two morph targets

	// The times at 0, 25, 50, 75 and 99% between each two keys and the time of the last key, with the key that STEP holds at each
	wi::vector<std::pair<float, size_t>> StepSamples(const wi::vector<float>& times)
	{
		wi::vector<std::pair<float, size_t>> samples;
		for (size_t key = 0; key + 1 < times.size(); ++key)
		{
			for (float fraction : { 0.0f, 0.25f, 0.5f, 0.75f, 0.99f })
			{
				samples.emplace_back(wi::math::Lerp(times[key], times[key + 1], fraction), key);
			}
		}
		samples.emplace_back(times.back(), times.size() - 1);
		return samples;
	}

	// Adds an accessor of float values to the only buffer of the model
	int AddAccessor(tinygltf::Model& model, const wi::vector<float>& values, int type)
	{
		const size_t components = type == TINYGLTF_TYPE_VEC4 ? 4 : (type == TINYGLTF_TYPE_VEC3 ? 3 : 1);
		if (model.buffers.empty())
		{
			model.buffers.emplace_back();
		}
		tinygltf::Buffer& buffer = model.buffers[0];
		tinygltf::BufferView view;
		view.buffer = 0;
		view.byteOffset = buffer.data.size();
		view.byteLength = values.size() * sizeof(float);
		const unsigned char* bytes = (const unsigned char*)values.data();
		buffer.data.insert(buffer.data.end(), bytes, bytes + view.byteLength);
		model.bufferViews.push_back(view);

		tinygltf::Accessor accessor;
		accessor.bufferView = int(model.bufferViews.size() - 1);
		accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
		accessor.type = type;
		accessor.count = values.size() / components;
		// (the key times and the vertex positions must have bounds)
		accessor.minValues.assign(components, std::numeric_limits<double>::max());
		accessor.maxValues.assign(components, std::numeric_limits<double>::lowest());
		for (size_t i = 0; i < values.size(); ++i)
		{
			accessor.minValues[i % components] = std::min(accessor.minValues[i % components], double(values[i]));
			accessor.maxValues[i % components] = std::max(accessor.maxValues[i % components], double(values[i]));
		}
		model.accessors.push_back(accessor);
		return int(model.accessors.size() - 1);
	}

	// Writes the glTF STEP animation of a node with a triangle that has two morph targets
	bool WriteStepFile()
	{
		tinygltf::Model model;
		model.asset.version = "2.0";

		tinygltf::Primitive primitive;
		primitive.attributes["POSITION"] = AddAccessor(model, { 0, 0, 0, 1, 0, 0, 0, 1, 0 }, TINYGLTF_TYPE_VEC3);
		primitive.attributes["NORMAL"] = AddAccessor(model, { 0, 0, 1, 0, 0, 1, 0, 0, 1 }, TINYGLTF_TYPE_VEC3);
		primitive.targets.push_back({ { "POSITION", AddAccessor(model, { 0, 0, 1, 0, 0, 1, 0, 0, 1 }, TINYGLTF_TYPE_VEC3) } });
		primitive.targets.push_back({ { "POSITION", AddAccessor(model, { 1, 0, 0, 1, 0, 0, 1, 0, 0 }, TINYGLTF_TYPE_VEC3) } });
		primitive.material = 0;
		model.materials.emplace_back();
		tinygltf::Mesh mesh;
		mesh.primitives.push_back(primitive);
		mesh.weights = { 0, 0 };
		model.meshes.push_back(mesh);

		tinygltf::Node node;
		node.name = "Stepped";
		node.mesh = 0;
		model.nodes.push_back(node);
		tinygltf::Scene scene;
		scene.nodes = { 0 };
		model.scenes.push_back(scene);
		model.defaultScene = 0;

		tinygltf::Animation animation;
		const int input = AddAccessor(model, step_times, TINYGLTF_TYPE_SCALAR);
		auto add_channel = [&](const wi::vector<float>& values, int type, const char* path) {
			tinygltf::AnimationSampler sampler;
			sampler.input = input;
			sampler.output = AddAccessor(model, values, type);
			sampler.interpolation = "STEP";
			animation.samplers.push_back(sampler);
			tinygltf::AnimationChannel channel;
			channel.sampler = int(animation.samplers.size() - 1);
			channel.target_node = 0;
			channel.target_path = path;
			animation.channels.push_back(channel);
		};
		add_channel(step_translations, TINYGLTF_TYPE_VEC3, "translation");
		add_channel(step_rotations, TINYGLTF_TYPE_VEC4, "rotation");
		add_channel(step_scales, TINYGLTF_TYPE_VEC3, "scale");
		add_channel(step_weights, TINYGLTF_TYPE_SCALAR, "weights");
		model.animations.push_back(animation);

		std::ofstream file(input_file);
		tinygltf::TinyGLTF writer;
		const bool written = writer.WriteGltfSceneToStream(&model, file, true, false); // (the buffer is embedded)
		file.close();
		return written && file.good();
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

	// The sampler that animates a path of a node in an exported file
	struct ExportedSampler
	{
		bool found = false;
		std::string interpolation;
		wi::vector<float> times;
		wi::vector<float> values;
	};
	ExportedSampler FindSampler(const tinygltf::Model& model, const std::string& node_name, const std::string& path)
	{
		ExportedSampler result;
		for (const tinygltf::Animation& animation : model.animations)
		{
			for (const tinygltf::AnimationChannel& channel : animation.channels)
			{
				if (channel.target_node < 0 || channel.target_node >= (int)model.nodes.size() || model.nodes[channel.target_node].name != node_name || channel.target_path != path)
					continue;
				const tinygltf::AnimationSampler& sampler = animation.samplers[channel.sampler];
				result.found = true;
				result.interpolation = sampler.interpolation;
				result.times = ReadAccessor(model, sampler.input);
				result.values = ReadAccessor(model, sampler.output);
				return result;
			}
		}
		return result;
	}
	bool LoadExportedFile(tinygltf::Model& model)
	{
		tinygltf::TinyGLTF loader;
		std::string err, warn;
		const bool loaded = loader.LoadASCIIFromFile(&model, &err, &warn, output_file);
		Check(loaded, "the exported file could not be loaded: " + err);
		return loaded;
	}

	// Plays the animation at a time (an update of the scene with no time step evaluates it there)
	void SampleAt(Scene& scene, Entity animationEntity, float time)
	{
		AnimationComponent& animation = *scene.animations.GetComponent(animationEntity);
		animation.Play();
		animation.timer = time;
		scene.Update(0);
	}

	// The samplers of all channels of an animation have this mode
	bool AllSamplers(const AnimationComponent& animation, Mode mode)
	{
		bool result = !animation.channels.empty();
		for (const AnimationComponent::AnimationChannel& channel : animation.channels)
		{
			result &= channel.samplerIndex >= 0 && channel.samplerIndex < (int)animation.samplers.size() && animation.samplers[channel.samplerIndex].mode == mode;
		}
		return result;
	}

	// The node of the imported glTF STEP animation holds the keys of the previous time (in the engine space)
	void CheckStepSampling(Scene& scene, const std::string& when)
	{
		const Entity node = scene.Entity_FindByName("Stepped");
		const ObjectComponent* object = scene.objects.GetComponent(node);
		const MeshComponent* mesh = object == nullptr ? nullptr : scene.meshes.GetComponent(object->meshID);
		const bool imported = scene.animations.GetCount() == 1 && scene.transforms.Contains(node) && mesh != nullptr && mesh->morph_targets.size() == 2;
		Check(imported, "the STEP animation or its node was not imported " + when);
		if (!imported)
			return;
		const Entity animationEntity = scene.animations.GetEntity(0);
		Check(scene.animations[0].channels.size() == 4 && AllSamplers(scene.animations[0], Mode::STEP), "the glTF STEP samplers were not imported as STEP " + when);

		for (auto [time, key] : StepSamples(step_times))
		{
			SampleAt(scene, animationEntity, time);
			const std::string at = " at time " + std::to_string(time) + " " + when;
			const TransformComponent& transform = *scene.transforms.GetComponent(node);
			const MeshComponent& morphed = *scene.meshes.GetComponent(scene.objects.GetComponent(node)->meshID);
			const XMFLOAT3& translation = transform.translation_local;
			Check(Near({ translation.x, translation.y, translation.z }, MirrorTranslations(Key(step_translations, 3, key))), "the translation was not the key of the previous time" + at);
			const wi::vector<float> rotation = { transform.rotation_local.x, transform.rotation_local.y, transform.rotation_local.z, transform.rotation_local.w };
			Check(SameRotations(rotation, MirrorRotations(Key(step_rotations, 4, key))), "the rotation was not the key of the previous time" + at);
			const XMFLOAT3& scale = transform.scale_local;
			Check(Near({ scale.x, scale.y, scale.z }, Key(step_scales, 3, key)), "the scale was not the key of the previous time" + at);
			Check(Near({ morphed.morph_targets[0].weight, morphed.morph_targets[1].weight }, Key(step_weights, 2, key)), "the morph target weights were not the key of the previous time" + at);
		}
	}

	// A glTF STEP animation is imported as STEP, it is exported as STEP with the same keys, and imported again the same
	void CheckStepImportExport()
	{
		if (!WriteStepFile())
		{
			Check(false, "the glTF file of the STEP animation could not be written");
			return;
		}
		Scene scene;
		ImportModel_GLTF(input_file, scene);
		CheckStepSampling(scene, "after the import");

		ExportModel_GLTF(output_file, scene);
		tinygltf::Model model;
		if (!LoadExportedFile(model))
			return;
		const ExportedSampler translation = FindSampler(model, "Stepped", "translation");
		const ExportedSampler rotation = FindSampler(model, "Stepped", "rotation");
		const ExportedSampler scale = FindSampler(model, "Stepped", "scale");
		const ExportedSampler weights = FindSampler(model, "Stepped", "weights");
		Check(translation.found && rotation.found && scale.found && weights.found, "the STEP animation was not exported");
		for (const ExportedSampler* sampler : { &translation, &rotation, &scale, &weights })
		{
			Check(sampler->interpolation == "STEP" && Near(sampler->times, step_times), "a STEP sampler was not exported as STEP with the same key times");
		}
		Check(Near(translation.values, step_translations) && SameRotations(rotation.values, step_rotations) && Near(scale.values, step_scales) && Near(weights.values, step_weights), "the keys of the STEP animation were changed by the export");

		Scene reimported;
		ImportModel_GLTF(output_file, reimported);
		CheckStepSampling(reimported, "after the export and import");
	}

	Entity CreateNode(Scene& scene, const std::string& name)
	{
		Entity entity = CreateEntity();
		scene.names.Create(entity) = name;
		scene.transforms.Create(entity);
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
	// Adds a channel with its own sampler to an animation
	void AddChannel(AnimationComponent& animation, Entity target, Path path, Mode mode, Entity data)
	{
		AnimationComponent::AnimationSampler& sampler = animation.samplers.emplace_back();
		sampler.mode = mode;
		sampler.data = data;
		AnimationComponent::AnimationChannel& channel = animation.channels.emplace_back();
		channel.target = target;
		channel.path = path;
		channel.samplerIndex = int(animation.samplers.size() - 1);
	}

	// A NEAREST animation is exported as glTF STEP keys that change halfway between the keys, which sample the same after the import
	void CheckNearestExport()
	{
		Scene scene;
		const Entity node = CreateNode(scene, "Nearest");
		const Entity morphed = CreateNode(scene, "NearestMorphed");
		const Entity material = CreateEntity();
		scene.materials.Create(material);
		const Entity meshEntity = CreateEntity();
		MeshComponent& mesh = scene.meshes.Create(meshEntity);
		mesh.vertex_positions = { XMFLOAT3(0, 0, 0), XMFLOAT3(1, 0, 0), XMFLOAT3(0, 1, 0) };
		mesh.vertex_normals = { XMFLOAT3(0, 0, 1), XMFLOAT3(0, 0, 1), XMFLOAT3(0, 0, 1) };
		mesh.indices = { 0, 1, 2 };
		MeshComponent::MeshSubset& subset = mesh.subsets.emplace_back();
		subset.materialID = material;
		subset.indexCount = 3;
		mesh.morph_targets.emplace_back().vertex_positions = { XMFLOAT3(0, 0, 1), XMFLOAT3(0, 0, 1), XMFLOAT3(0, 0, 1) };
		mesh.morph_targets.emplace_back().vertex_positions = { XMFLOAT3(1, 0, 0), XMFLOAT3(1, 0, 0), XMFLOAT3(1, 0, 0) };
		mesh.CreateRenderData();
		scene.objects.Create(morphed).meshID = meshEntity;

		// (in the engine space) The second translation key at time 1 is never taken: of keys at the same time, the first one is
		const wi::vector<float> translation_times = { 0, 1, 1, 3 };
		const wi::vector<float> translations = { 1, 0, 0, 2, 1, 0.5f, 9, 9, 9, 3, -1, 2 };
		const wi::vector<float> times = { 0, 1, 3 };
		const wi::vector<float> rotations = Rotations({ XMFLOAT3(0.2f, 0.1f, -0.3f), XMFLOAT3(-0.4f, 0.3f, 0.5f), XMFLOAT3(0.6f, -0.2f, 0.1f) });
		const wi::vector<float> weights = { 0, 1, 1, 0, 0.25f, 0.75f };

		const Entity animationEntity = CreateEntity();
		scene.names.Create(animationEntity) = "NearestAnimation";
		AnimationComponent& animation = scene.animations.Create(animationEntity);
		animation.start = 0;
		animation.end = 3;
		AddChannel(animation, node, Path::TRANSLATION, Mode::NEAREST, CreateData(scene, translation_times, translations));
		AddChannel(animation, node, Path::ROTATION, Mode::NEAREST, CreateData(scene, times, rotations));
		AddChannel(animation, morphed, Path::WEIGHTS, Mode::NEAREST, CreateData(scene, times, weights));

		// The samples (away from the halfway times, 0.5 and 2) with the key that NEAREST takes: 0 at time 0, 1 at time 1 and 2 at time 3
		const std::pair<float, size_t> samples[] = { { 0.0f, 0 }, { 0.25f, 0 }, { 0.4f, 0 }, { 0.6f, 1 }, { 0.75f, 1 }, { 1.0f, 1 }, { 1.5f, 1 }, { 1.9f, 1 }, { 2.1f, 2 }, { 2.5f, 2 }, { 3.0f, 2 } };
		const wi::vector<float> taken_translations = Keys(translations, 3, { 0, 1, 3 });
		auto check_samples = [&](Scene& sampled, const std::string& when) {
			const Entity sampled_node = sampled.Entity_FindByName("Nearest");
			const Entity sampled_morphed = sampled.Entity_FindByName("NearestMorphed");
			const ObjectComponent* object = sampled.objects.GetComponent(sampled_morphed);
			const bool found = sampled.animations.GetCount() == 1 && sampled.transforms.Contains(sampled_node) && object != nullptr && sampled.meshes.Contains(object->meshID);
			Check(found, "the NEAREST animation or its nodes were not found " + when);
			if (!found)
				return;
			for (auto [time, key] : samples)
			{
				SampleAt(sampled, sampled.animations.GetEntity(0), time);
				const std::string at = " at time " + std::to_string(time) + " " + when;
				const TransformComponent& transform = *sampled.transforms.GetComponent(sampled_node);
				const MeshComponent& sampled_mesh = *sampled.meshes.GetComponent(sampled.objects.GetComponent(sampled_morphed)->meshID);
				const XMFLOAT3& translation = transform.translation_local;
				Check(Near({ translation.x, translation.y, translation.z }, Key(taken_translations, 3, key)), "the translation was not the nearest key" + at);
				const wi::vector<float> rotation = { transform.rotation_local.x, transform.rotation_local.y, transform.rotation_local.z, transform.rotation_local.w };
				Check(SameRotations(rotation, Key(rotations, 4, key)), "the rotation was not the nearest key" + at);
				Check(sampled_mesh.morph_targets.size() == 2 && Near({ sampled_mesh.morph_targets[0].weight, sampled_mesh.morph_targets[1].weight }, Key(weights, 2, key)), "the morph target weights were not the nearest key" + at);
			}
		};
		check_samples(scene, "in the engine");

		ExportModel_GLTF(output_file, scene);
		tinygltf::Model model;
		if (!LoadExportedFile(model))
			return;
		// The keys change halfway between the times, the last key is repeated at the last time (in the glTF space):
		const wi::vector<float> halfway_times = { 0, 0.5f, 2, 3 };
		const ExportedSampler translation = FindSampler(model, "Nearest", "translation");
		const ExportedSampler rotation = FindSampler(model, "Nearest", "rotation");
		const ExportedSampler weight = FindSampler(model, "NearestMorphed", "weights");
		Check(translation.found && rotation.found && weight.found, "the NEAREST animation was not exported");
		for (const ExportedSampler* sampler : { &translation, &rotation, &weight })
		{
			Check(sampler->interpolation == "STEP" && Near(sampler->times, halfway_times), "a NEAREST sampler was not exported as STEP with the halfway times");
		}
		Check(Near(translation.values, MirrorTranslations(Keys(translations, 3, { 0, 1, 3, 3 }))), "the translation keys of a NEAREST sampler were not exported as the STEP keys");
		Check(SameRotations(rotation.values, MirrorRotations(Keys(rotations, 4, { 0, 1, 2, 2 }))), "the rotation keys of a NEAREST sampler were not exported as the STEP keys");
		Check(Near(weight.values, Keys(weights, 2, { 0, 1, 2, 2 })), "the weight keys of a NEAREST sampler were not exported as the STEP keys (all weights of a time are one key)");

		Scene imported;
		ImportModel_GLTF(output_file, imported);
		Check(imported.animations.GetCount() == 1 && AllSamplers(imported.animations[0], Mode::STEP), "the exported NEAREST samplers were not imported as STEP");
		check_samples(imported, "after the export and import");
	}

	// STEP holds the keys of the other path data types too (a float, float2, float3 and float4 path)
	void CheckOtherPaths()
	{
		Scene scene;
		const Entity light = CreateNode(scene, "Light");
		scene.lights.Create(light);
		const Entity material = CreateEntity();
		scene.materials.Create(material);
		const Entity camera = CreateNode(scene, "Camera");
		scene.cameras.Create(camera);

		const wi::vector<float> intensities = { 1, 5, 2 };
		const wi::vector<float> aperture_shapes = { 1, 1, 2, 1, 1, 3 };
		const wi::vector<float> colors = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
		const wi::vector<float> base_colors = { 1, 0, 0, 1, 0, 1, 0, 0.5f, 0, 0, 1, 0.25f };

		const Entity animationEntity = CreateEntity();
		AnimationComponent& animation = scene.animations.Create(animationEntity);
		animation.start = 0;
		animation.end = 3;
		AddChannel(animation, light, Path::LIGHT_INTENSITY, Mode::STEP, CreateData(scene, step_times, intensities));
		AddChannel(animation, camera, Path::CAMERA_APERTURE_SHAPE, Mode::STEP, CreateData(scene, step_times, aperture_shapes));
		AddChannel(animation, light, Path::LIGHT_COLOR, Mode::STEP, CreateData(scene, step_times, colors));
		AddChannel(animation, material, Path::MATERIAL_COLOR, Mode::STEP, CreateData(scene, step_times, base_colors));

		for (auto [time, key] : StepSamples(step_times))
		{
			SampleAt(scene, animationEntity, time);
			const std::string at = " at time " + std::to_string(time);
			const LightComponent& sampled_light = *scene.lights.GetComponent(light);
			const XMFLOAT2& aperture_shape = scene.cameras.GetComponent(camera)->aperture_shape;
			const XMFLOAT4& base_color = scene.materials.GetComponent(material)->baseColor;
			Check(Near({ sampled_light.intensity }, Key(intensities, 1, key)), "the light intensity (float) was not the key of the previous time" + at);
			Check(Near({ aperture_shape.x, aperture_shape.y }, Key(aperture_shapes, 2, key)), "the camera aperture shape (float2) was not the key of the previous time" + at);
			Check(Near({ sampled_light.color.x, sampled_light.color.y, sampled_light.color.z }, Key(colors, 3, key)), "the light color (float3) was not the key of the previous time" + at);
			Check(Near({ base_color.x, base_color.y, base_color.z, base_color.w }, Key(base_colors, 4, key)), "the material color (float4) was not the key of the previous time" + at);
		}
	}

	// The current version of the scene serialization keeps STEP and NEAREST
	void CheckSceneSerialization()
	{
		Scene scene;
		AnimationComponent& animation = scene.animations.Create(CreateEntity());
		animation.samplers.emplace_back().mode = Mode::STEP;
		animation.samplers.emplace_back().mode = Mode::NEAREST;
		wi::Archive archive;
		scene.Serialize(archive);
		archive.SetReadModeAndResetPos(true);
		Scene loaded;
		loaded.Serialize(archive);
		Check(loaded.animations.GetCount() == 1 && loaded.animations[0].samplers.size() == 2 &&
			loaded.animations[0].samplers[0].mode == Mode::STEP && loaded.animations[0].samplers[1].mode == Mode::NEAREST,
			"the STEP and NEAREST samplers were changed by the serialization of the scene");
	}

	int RunSampling(int device_created_pipe)
	{
		wi::jobsystem::Initialize();
		wi::graphics::GetDevice() = new wi::graphics::GraphicsDevice_Vulkan(nullptr); // ends the process with EXIT_FAILURE without a device
		// Tell the parent that the device was created, so that later failures are not taken for a missing device:
		if (write(device_created_pipe, "1", 1) != 1)
			return TEST_FAILED;
		close(device_created_pipe);

		RemoveFiles(); // the files of an earlier run must not be checked
		CheckSceneSerialization();
		CheckStepImportExport();
		RemoveFiles();
		CheckNearestExport();
		RemoveFiles();
		CheckOtherPaths();

		if (failures > 0)
			return TEST_FAILED;
		std::cout << "GLTFStepAnimation test passed\n";
		return 0;
	}

	// STEP samplers of AnimationComponent versions before 3 took the nearest key, they are loaded as NEAREST. This needs no device.
	void CheckSerializedVersions()
	{
		const Mode modes[] = { Mode::LINEAR, Mode::STEP, Mode::CUBICSPLINE, Mode::NEAREST };
		for (uint64_t version = 0; version <= 3; ++version)
		{
			AnimationComponent animation = {};
			for (Mode mode : modes)
			{
				animation.samplers.emplace_back().mode = mode;
			}
			wi::Archive archive;
			EntitySerializer write_seri;
			write_seri.version = version;
			animation.Serialize(archive, write_seri);
			archive.SetReadModeAndResetPos(true);
			AnimationComponent loaded = {};
			EntitySerializer read_seri;
			read_seri.version = version;
			loaded.Serialize(archive, read_seri);
			const Mode step = version < 3 ? Mode::NEAREST : Mode::STEP;
			Check(loaded.samplers.size() == 4 && loaded.samplers[0].mode == Mode::LINEAR && loaded.samplers[1].mode == step &&
				loaded.samplers[2].mode == Mode::CUBICSPLINE && loaded.samplers[3].mode == Mode::NEAREST,
				"the sampler modes of AnimationComponent version " + std::to_string(version) + " were not loaded as expected");
		}
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
		const int result = RunSampling(std::atoi(argv[2]));
		std::cout.flush();
		std::cerr.flush();
		std::_Exit(result); // without destroying the device in the static destructors
	}

	CheckSerializedVersions();
	if (failures > 0)
		return 1;

	int device_created[2];
	if (pipe(device_created) != 0)
	{
		std::cerr << "GLTFStepAnimation test failed: pipe() failed\n";
		return 1;
	}
	pid_t pid = fork();
	if (pid < 0)
	{
		std::cerr << "GLTFStepAnimation test failed: fork() failed\n";
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
		std::cerr << "GLTFStepAnimation test failed: waitpid() failed\n";
		return 1;
	}
	char byte = 0;
	const bool created = read(device_created[0], &byte, 1) == 1;
	close(device_created[0]);

	if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
		return 0;
	if (!created && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE)
	{
		std::cout << "GLTFStepAnimation test: the serialized versions passed, the sampling is skipped: no Vulkan device could be created\n";
		return TEST_SKIPPED;
	}
	if (WIFSIGNALED(status))
	{
		std::cerr << "GLTFStepAnimation test failed: the test process (sampling, import or export) crashed with signal " << WTERMSIG(status) << '\n';
	}
	else if (WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE)
	{
		std::cerr << "GLTFStepAnimation test failed: the test process (sampling, import or export) ended with EXIT_FAILURE\n";
	}
	return 1;
}
