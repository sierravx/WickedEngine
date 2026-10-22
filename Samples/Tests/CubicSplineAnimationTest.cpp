#include "WickedEngine.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

// Regression test for the cubic spline sampling of Scene::RunAnimationUpdateSystem(): it must play the curve of glTF, in which
//	the tangents of the keys are scaled by the time between the keys (not by the time of the frame), and in which a key of
//	n components is n in-tangents, n values and n out-tangents (for morph target weights too), for every kind of path data.

using namespace wi::ecs;
using namespace wi::scene;
using Path = AnimationComponent::AnimationChannel::Path;

namespace
{
	int failures = 0;

	void Check(bool condition, const std::string& message)
	{
		if (!condition)
		{
			std::cerr << "CubicSplineAnimation test failed: " << message << '\n';
			failures++;
		}
	}

	const wi::vector<float> times = { 0, 0.5f, 2 }; // the times between the keys are not 1

	// The cubic spline of glTF at the time, for keys of the given number of components
	wi::vector<float> Expected(const wi::vector<float>& keys, size_t components, float time)
	{
		size_t key = 0;
		while (key + 2 < times.size() && time > times[key + 1])
		{
			key++;
		}
		const float key_dt = times[key + 1] - times[key];
		const float t = std::clamp((time - times[key]) / key_dt, 0.0f, 1.0f);
		const float t2 = t * t;
		const float t3 = t2 * t;
		auto at = [&](size_t k, size_t part, size_t j) { return keys[(k * 3 + part) * components + j]; }; // part 0: in-tangent, 1: value, 2: out-tangent
		wi::vector<float> result(components);
		for (size_t j = 0; j < components; ++j)
		{
			result[j] =
				(2 * t3 - 3 * t2 + 1) * at(key, 1, j) +
				key_dt * (t3 - 2 * t2 + t) * at(key, 2, j) +
				(-2 * t3 + 3 * t2) * at(key + 1, 1, j) +
				key_dt * (t3 - t2) * at(key + 1, 0, j);
		}
		return result;
	}

	float Difference(const wi::vector<float>& a, const wi::vector<float>& b)
	{
		float difference = 0;
		for (size_t i = 0; i < a.size(); ++i)
		{
			difference = std::max(difference, std::abs(a[i] - b[i]));
		}
		return difference;
	}

	// Plays the (paused) animation at the time: it is updated because its timer moved
	void Play(Scene& scene, Entity animation_entity, float time)
	{
		AnimationComponent& animation = *scene.animations.GetComponent(animation_entity);
		animation.timer = time;
		animation.last_update_time = time - 1;
		scene.ScanAnimationDependencies();
		wi::jobsystem::context ctx;
		scene.RunAnimationUpdateSystem(ctx);
		wi::jobsystem::Wait(ctx);
	}

	void TestCubicSplines()
	{
		Scene scene;
		scene.dt = 1.0f / 60.0f; // the time of a frame, which the curve doesn't depend on

		Entity node = CreateEntity();
		scene.transforms.Create(node);
		Entity light = CreateEntity();
		scene.lights.Create(light);
		Entity camera = CreateEntity();
		scene.cameras.Create(camera);
		Entity mesh = CreateEntity();
		scene.meshes.Create(mesh).morph_targets.resize(2);

		struct ChannelDesc
		{
			std::string name;
			Entity target;
			Path path;
			size_t components;
			wi::vector<float> keys;
		};
		const wi::vector<ChannelDesc> channels = {
			{ "translation", node, Path::TRANSLATION, 3, {
				0.5f, 0, 0.1f, 0, 1, 0, 0.3f, -0.2f, 0.4f,
				-0.2f, 0.4f, 0, 0.1f, 1.2f, 0, 0.6f, 0.1f, -0.3f,
				0, 0, 0.5f, -0.5f, 0.8f, 0.3f, 0.2f, 0.2f, 0.2f,
			} },
			{ "rotation", node, Path::ROTATION, 4, {
				0.1f, 0.2f, -0.1f, 0.3f, 0, 0, 0, 1, -0.2f, 0.1f, 0.4f, 0.05f,
				0.3f, -0.1f, 0.2f, -0.2f, 0.5f, 0.5f, 0.5f, 0.5f, 0.05f, 0.2f, 0.1f, 0.1f,
				-0.1f, 0.3f, 0, 0.2f, 0, 0.6f, 0, 0.8f, 0.2f, -0.2f, 0.1f, -0.1f,
			} },
			{ "scale", node, Path::SCALE, 3, {
				-0.3f, 0.2f, 0.1f, 1, 1, 1, 0.4f, -0.5f, 0.2f,
				0.1f, -0.2f, -0.3f, 1.3f, 0.7f, 1.1f, -0.4f, 0.3f, 0.2f,
				0.2f, 0.2f, -0.6f, 0.8f, 1.2f, 0.9f, 0.1f, 0.1f, 0.1f,
			} },
			{ "light intensity", light, Path::LIGHT_INTENSITY, 1, { 1, 2, -3, 0.5f, 4, 2, -1, 3, 0 } },
			{ "camera aperture shape", camera, Path::CAMERA_APERTURE_SHAPE, 2, {
				0.2f, -0.1f, 1, 1, 0.5f, 0.3f,
				-0.4f, 0.2f, 0.5f, 2, 0.1f, -0.6f,
				0.3f, 0.3f, 1.5f, 0.5f, 0, 0,
			} },
			{ "morph target weights", mesh, Path::WEIGHTS, 2, {
				0.5f, -0.5f, 0, 1, 1, 0.2f,
				-1, 0.3f, 1, 0.5f, 0.4f, -0.8f,
				0.2f, 0.6f, 0.2f, 0, 0, 0,
			} },
		};

		Entity animation_entity = CreateEntity();
		AnimationComponent& animation = scene.animations.Create(animation_entity);
		animation.start = times.front();
		animation.end = times.back();
		for (const ChannelDesc& desc : channels)
		{
			Entity data = CreateEntity();
			AnimationDataComponent& animation_data = scene.animation_datas.Create(data);
			animation_data.keyframe_times = times;
			animation_data.keyframe_data = desc.keys;
			AnimationComponent::AnimationSampler& sampler = animation.samplers.emplace_back();
			sampler.data = data;
			sampler.mode = AnimationComponent::AnimationSampler::Mode::CUBICSPLINE;
			AnimationComponent::AnimationChannel& channel = animation.channels.emplace_back();
			channel.target = desc.target;
			channel.samplerIndex = int(animation.samplers.size() - 1);
			channel.path = desc.path;
		}

		// The played value of a channel, the rotation as the unit quaternion in the hemisphere of the expected one:
		auto played = [&](const ChannelDesc& desc, const wi::vector<float>& expected) -> wi::vector<float> {
			switch (desc.path)
			{
			case Path::TRANSLATION:
			{
				const XMFLOAT3& T = scene.transforms.GetComponent(node)->translation_local;
				return { T.x, T.y, T.z };
			}
			case Path::ROTATION:
			{
				XMVECTOR Q = XMQuaternionNormalize(XMLoadFloat4(&scene.transforms.GetComponent(node)->rotation_local));
				if (XMVectorGetX(XMVector4Dot(Q, XMVectorSet(expected[0], expected[1], expected[2], expected[3]))) < 0)
				{
					Q = -Q;
				}
				return { XMVectorGetX(Q), XMVectorGetY(Q), XMVectorGetZ(Q), XMVectorGetW(Q) };
			}
			case Path::SCALE:
			{
				const XMFLOAT3& S = scene.transforms.GetComponent(node)->scale_local;
				return { S.x, S.y, S.z };
			}
			case Path::LIGHT_INTENSITY:
				return { scene.lights.GetComponent(light)->intensity };
			case Path::CAMERA_APERTURE_SHAPE:
			{
				const XMFLOAT2& shape = scene.cameras.GetComponent(camera)->aperture_shape;
				return { shape.x, shape.y };
			}
			default:
			{
				const MeshComponent& morphed = *scene.meshes.GetComponent(mesh);
				return { morphed.morph_targets[0].weight, morphed.morph_targets[1].weight };
			}
			}
		};

		wi::vector<float> max_difference(channels.size(), 0.0f);
		for (int i = 0; i <= 40; ++i)
		{
			const float time = i / 20.0f;
			Play(scene, animation_entity, time);
			for (size_t c = 0; c < channels.size(); ++c)
			{
				wi::vector<float> expected = Expected(channels[c].keys, channels[c].components, time);
				if (channels[c].path == Path::ROTATION)
				{
					XMVECTOR Q = XMQuaternionNormalize(XMVectorSet(expected[0], expected[1], expected[2], expected[3]));
					expected = { XMVectorGetX(Q), XMVectorGetY(Q), XMVectorGetZ(Q), XMVectorGetW(Q) };
				}
				max_difference[c] = std::max(max_difference[c], Difference(played(channels[c], expected), expected));
			}
		}
		for (size_t c = 0; c < channels.size(); ++c)
		{
			Check(max_difference[c] < 1e-4f, channels[c].name + ": the cubic spline is not played like in glTF, difference: " + std::to_string(max_difference[c]));
		}
	}
}

int main()
{
	wi::jobsystem::Initialize();

	TestCubicSplines();

	wi::jobsystem::ShutDown();

	if (failures > 0)
	{
		std::cerr << failures << " check(s) failed\n";
		return 1;
	}
	std::cout << "CubicSplineAnimation test passed\n";
	return 0;
}
