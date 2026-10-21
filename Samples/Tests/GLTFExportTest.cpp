#include "WickedEngine.h"
#include "wiGraphicsDevice_Vulkan.h"
#include "ModelImporter.h"

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
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

	int failures = 0;

	void Check(bool condition, const std::string& message)
	{
		if (!condition)
		{
			std::cerr << "GLTFExport test failed: " << message << '\n';
			failures++;
		}
	}

	bool Near(const wi::vector<float>& a, const wi::vector<float>& b)
	{
		if (a.size() != b.size())
			return false;
		for (size_t i = 0; i < a.size(); ++i)
		{
			if (std::abs(a[i] - b[i]) > 1e-4f)
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

	// What the runtime evaluates for a translation key of a retargeted channel (see Scene::RunAnimationUpdateSystem())
	wi::vector<float> RetargetTranslations(const TransformComponent& source, const XMFLOAT4X4& dst, const XMFLOAT4X4& src, const wi::vector<float>& keys)
	{
		wi::vector<float> result;
		for (size_t i = 0; i + 3 <= keys.size(); i += 3)
		{
			TransformComponent transform = source;
			transform.translation_local = XMFLOAT3(keys[i], keys[i + 1], keys[i + 2]);
			XMVECTOR S, R, T;
			wi::math::DecomposeTransform(&S, &R, &T, XMLoadFloat4x4(&dst) * transform.GetLocalMatrix() * XMLoadFloat4x4(&src));
			result.push_back(XMVectorGetX(T));
			result.push_back(XMVectorGetY(T));
			result.push_back(XMVectorGetZ(T));
		}
		return result;
	}

	// The engine's translation keys in the glTF space
	wi::vector<float> MirrorTranslations(wi::vector<float> keys)
	{
		for (size_t i = 2; i < keys.size(); i += 3)
		{
			keys[i] = -keys[i];
		}
		return keys;
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

	int RunExport()
	{
		wi::jobsystem::Initialize();
		wi::graphics::GetDevice() = new wi::graphics::GraphicsDevice_Vulkan(nullptr); // ends the process with EXIT_FAILURE without a device

		Scene other; // the source of an animation retargeted from an other scene
		Scene scene;

		Entity a = CreateNode(scene, "A", XMFLOAT3(0, 0, 1));
		Entity b = CreateNode(scene, "B", XMFLOAT3(1, 0, 0));
		Entity source = CreateNode(scene, "Source", XMFLOAT3(0, 1, 0.5f), XMFLOAT3(0.3f, 0.2f, 0.1f));
		Entity destination = CreateNode(scene, "Destination", XMFLOAT3(0, 2, 0));
		Entity other_destination = CreateNode(scene, "OtherDestination", XMFLOAT3(0, 3, 0));
		Entity other_source = CreateNode(other, "OtherSource", XMFLOAT3(0.5f, 1, 0), XMFLOAT3(-0.2f, 0.4f, 0));
		Entity no_transform = CreateEntity();

		// A mesh with a morph target, the weights of which can be animated
		Entity morphed = CreateNode(scene, "Morphed", XMFLOAT3(0, 0, 0));
		Entity meshEntity = CreateEntity();
		Entity materialEntity = CreateEntity();
		scene.materials.Create(materialEntity);
		{
			MeshComponent& mesh = scene.meshes.Create(meshEntity);
			mesh.vertex_positions = { XMFLOAT3(0, 0, 0), XMFLOAT3(1, 0, 0), XMFLOAT3(0, 1, 0) };
			mesh.vertex_normals = { XMFLOAT3(0, 0, 1), XMFLOAT3(0, 0, 1), XMFLOAT3(0, 0, 1) };
			mesh.indices = { 0, 1, 2 };
			MeshComponent::MeshSubset& subset = mesh.subsets.emplace_back();
			subset.materialID = materialEntity;
			subset.indexCount = 3;
			mesh.morph_targets.emplace_back().vertex_positions = { XMFLOAT3(0, 0, 1), XMFLOAT3(0, 0, 1), XMFLOAT3(0, 0, 1) };
			mesh.CreateRenderData();
		}
		scene.objects.Create(morphed).meshID = meshEntity;

		const wi::vector<float> times = { 0, 1 };
		const wi::vector<float> translation_keys = { 0, 0, 1, 0, 0, 2 };
		const wi::vector<float> shared_keys = { 1, 1, 1, 2, 2, 2 };
		const wi::vector<float> retarget_keys = { 0, 0, 1, 1, 0, 0 };
		const wi::vector<float> weight_keys = { 0, 1 };
		Entity translation_data = CreateData(scene, times, translation_keys);
		Entity shared_data = CreateData(scene, times, shared_keys);
		Entity retarget_data = CreateData(scene, times, retarget_keys);
		Entity weight_data = CreateData(scene, times, weight_keys);
		Entity empty_data = CreateData(scene, {}, {});
		Entity other_data = CreateData(other, times, retarget_keys);

		XMFLOAT4X4 retarget_dst, retarget_src;
		XMStoreFloat4x4(&retarget_dst, XMMatrixRotationRollPitchYaw(0.3f, 0.5f, 0.7f) * XMMatrixTranslation(1, 2, 3));
		XMStoreFloat4x4(&retarget_src, XMMatrixRotationRollPitchYaw(-0.2f, 0.4f, 0.1f) * XMMatrixTranslation(-1, 0, 2));

		Entity animationEntity = CreateEntity();
		scene.names.Create(animationEntity) = "anim";
		{
			AnimationComponent& animation = scene.animations.Create(animationEntity);
			auto add_sampler = [&](Entity data, const Scene* data_scene = nullptr) {
				AnimationComponent::AnimationSampler& sampler = animation.samplers.emplace_back();
				sampler.data = data;
				sampler.scene = data_scene;
				return int(animation.samplers.size() - 1);
			};
			auto add_channel = [&](Entity target, int sampler, Path path, int retarget = -1) {
				AnimationComponent::AnimationChannel& channel = animation.channels.emplace_back();
				channel.target = target;
				channel.samplerIndex = sampler;
				channel.path = path;
				channel.retargetIndex = retarget;
			};
			const int translation_sampler = add_sampler(translation_data);
			const int shared_sampler = add_sampler(shared_data);
			add_channel(a, translation_sampler, Path::TRANSLATION); // exported, mirrored
			add_channel(a, translation_sampler, Path::LIGHT_COLOR); // nothing to animate in glTF
			add_channel(INVALID_ENTITY, translation_sampler, Path::TRANSLATION); // no target
			add_channel(no_transform, translation_sampler, Path::TRANSLATION); // target without a node
			add_channel(b, add_sampler(empty_data), Path::TRANSLATION); // no keys
			add_channel(b, 1000, Path::TRANSLATION); // invalid sampler
			add_channel(a, add_sampler(weight_data), Path::WEIGHTS); // no morph targets
			add_channel(morphed, add_sampler(weight_data), Path::WEIGHTS); // exported
			add_channel(b, shared_sampler, Path::TRANSLATION); // exported, mirrored
			add_channel(b, shared_sampler, Path::SCALE); // exported, not mirrored

			AnimationComponent::RetargetSourceData& same = animation.retargets.emplace_back();
			same.source = source;
			same.dstRelativeMatrix = retarget_dst;
			same.srcRelativeParentMatrix = retarget_src;
			add_channel(destination, add_sampler(retarget_data), Path::TRANSLATION, 0); // baked
			AnimationComponent::RetargetSourceData& from_other = animation.retargets.emplace_back();
			from_other.source = other_source;
			from_other.dstRelativeMatrix = retarget_dst;
			from_other.srcRelativeParentMatrix = retarget_src;
			add_channel(other_destination, add_sampler(other_data, &other), Path::TRANSLATION, 1); // baked
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

		const wi::vector<float> expected_same = MirrorTranslations(RetargetTranslations(*scene.transforms.GetComponent(source), retarget_dst, retarget_src, retarget_keys));
		const wi::vector<float> expected_other = MirrorTranslations(RetargetTranslations(*other.transforms.GetComponent(other_source), retarget_dst, retarget_src, retarget_keys));
		const TransformComponent a_original = *scene.transforms.GetComponent(a);

		ExportModel_GLTF(output_file, scene);

		// The scene is restored after the export
		const TransformComponent& a_restored = *scene.transforms.GetComponent(a);
		Check(std::memcmp(&a_restored.translation_local, &a_original.translation_local, sizeof(XMFLOAT3)) == 0 && std::memcmp(&a_restored.rotation_local, &a_original.rotation_local, sizeof(XMFLOAT4)) == 0, "the transforms were not restored after the export");
		Check(scene.animation_datas.GetComponent(translation_data)->keyframe_data == translation_keys && scene.animation_datas.GetComponent(shared_data)->keyframe_data == shared_keys, "the animation keys were not restored after the export");
		const AnimationComponent& animation = *scene.animations.GetComponent(animationEntity);
		Check(std::memcmp(&animation.retargets[0].dstRelativeMatrix, &retarget_dst, sizeof(XMFLOAT4X4)) == 0 && std::memcmp(&animation.retargets[1].srcRelativeParentMatrix, &retarget_src, sizeof(XMFLOAT4X4)) == 0, "the retarget matrices were not restored after the export");

		// Load the file and check it
		tinygltf::TinyGLTF loader;
		tinygltf::Model model;
		std::string err, warn;
		if (!loader.LoadASCIIFromFile(&model, &err, &warn, output_file))
		{
			Check(false, "the exported file could not be loaded: " + err);
			return TEST_FAILED;
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
		Check(exported.channels.size() == 6, "expected 6 exported channels, got " + std::to_string(exported.channels.size()));

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
		for (const tinygltf::AnimationChannel& channel : exported.channels)
		{
			Check(channel.target_node >= 0 && channel.target_node < (int)model.nodes.size(), "a channel has an invalid target node");
			Check(channel.sampler >= 0 && channel.sampler < (int)exported.samplers.size(), "a channel has an invalid sampler");
			Check(channel.target_path == "translation" || channel.target_path == "rotation" || channel.target_path == "scale" || channel.target_path == "weights", "a channel has an invalid path");
			if (channel.target_path == "weights")
			{
				Check(channel.target_node == node_index("Morphed"), "weights were exported for a node without morph targets");
			}
		}

		Check(Near(find_values("A", "translation"), MirrorTranslations(translation_keys)), "the translation keys were not exported mirrored");
		Check(Near(find_values("Morphed", "weights"), weight_keys), "the morph target weights were not exported");
		Check(Near(find_values("B", "translation"), MirrorTranslations(shared_keys)), "the translation keys shared with a scale channel were not exported mirrored");
		Check(Near(find_values("B", "scale"), shared_keys), "the scale keys shared with a translation channel were not exported as they are");
		Check(Near(find_values("Destination", "translation"), expected_same), "the channel retargeted from this scene was not baked");
		Check(Near(find_values("OtherDestination", "translation"), expected_other), "the channel retargeted from an other scene was not baked");

		if (failures > 0)
			return TEST_FAILED;
		std::cout << "GLTFExport test passed\n";
		return 0;
	}
}

int main(int argc, char** argv)
{
	if (argc > 1 && std::strcmp(argv[1], "--child") == 0)
	{
		// No error message box if the device can't be created:
		unsetenv("DISPLAY");
		unsetenv("WAYLAND_DISPLAY");
		unsetenv("XDG_RUNTIME_DIR");
		const int result = RunExport();
		std::cout.flush();
		std::cerr.flush();
		std::_Exit(result); // without destroying the device in the static destructors
	}

	pid_t pid = fork();
	if (pid < 0)
	{
		std::cerr << "GLTFExport test failed: fork() failed\n";
		return 1;
	}
	if (pid == 0)
	{
		execl("/proc/self/exe", argv[0], "--child", (char*)nullptr);
		_exit(127);
	}
	int status = 0;
	waitpid(pid, &status, 0);
	if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
		return 0;
	if (WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE)
	{
		std::cout << "GLTFExport test skipped: no Vulkan device could be created\n";
		return TEST_SKIPPED;
	}
	if (WIFSIGNALED(status))
	{
		std::cerr << "GLTFExport test failed: the export crashed with signal " << WTERMSIG(status) << '\n';
	}
	return 1;
}
