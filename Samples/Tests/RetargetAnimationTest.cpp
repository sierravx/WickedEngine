#include "WickedEngine.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

// Regression test for Scene::RetargetAnimation(): the keys that are baked for the destination (bake_data = true) must play
//	the same as the source keys that are retargeted while playing (bake_data = false), for linear and cubic spline translation,
//	rotation and scale channels. The tangents of cubic spline keys are derivatives, they must be retargeted as derivatives,
//	and the baked rotation keys must keep the hemispheres that the interpolation relies on.

using namespace wi::ecs;
using namespace wi::scene;
using Path = AnimationComponent::AnimationChannel::Path;
using Mode = AnimationComponent::AnimationSampler::Mode;
using Bone = HumanoidComponent::HumanoidBone;

namespace
{
	int failures = 0;

	void Check(bool condition, const std::string& message)
	{
		if (!condition)
		{
			std::cerr << "RetargetAnimation test failed: " << message << '\n';
			failures++;
		}
	}

	XMFLOAT4 Rotation(float roll, float pitch, float yaw, float sign = 1)
	{
		XMFLOAT4 q;
		XMStoreFloat4(&q, sign * XMQuaternionRotationRollPitchYaw(roll, pitch, yaw));
		return q;
	}
	wi::vector<float> Flatten(const wi::vector<XMFLOAT4>& values)
	{
		wi::vector<float> result;
		for (const XMFLOAT4& value : values)
		{
			result.insert(result.end(), { value.x, value.y, value.z, value.w });
		}
		return result;
	}

	// A bone of a skeleton in its rest pose. A turned bone gets the world rotation of the same bone of the source skeleton, turned
	//	by a quarter turn about Z (its rotation is not used): then the retargeting of scale keys is a linear function of them.
	struct BoneDesc
	{
		Bone bone;
		int parent; // index of the parent bone, -1 for the root
		XMFLOAT3 translation;
		XMFLOAT3 rotation; // roll, pitch, yaw
		XMFLOAT3 scale;
		bool turned = false;
	};
	struct Skeleton
	{
		Entity humanoid = INVALID_ENTITY;
		wi::vector<Entity> bones; // in the order of the bone descriptions
	};

	XMMATRIX RestWorldMatrix(const Scene& scene, Entity bone)
	{
		return scene.transforms.GetComponent(bone)->GetLocalMatrix() * scene.ComputeParentMatrixRecursive(bone);
	}

	Skeleton CreateSkeleton(Scene& scene, const wi::vector<BoneDesc>& descs, const Scene* source_scene = nullptr, const Skeleton* source = nullptr)
	{
		Skeleton skeleton;
		for (size_t i = 0; i < descs.size(); ++i)
		{
			const BoneDesc& desc = descs[i];
			Entity entity = CreateEntity();
			skeleton.bones.push_back(entity);
			TransformComponent& transform = scene.transforms.Create(entity);
			transform.translation_local = desc.translation;
			transform.rotation_local = Rotation(desc.rotation.x, desc.rotation.y, desc.rotation.z);
			transform.scale_local = desc.scale;
			if (desc.parent >= 0)
			{
				scene.Component_Attach(entity, skeleton.bones[desc.parent], true);
			}
			if (desc.turned)
			{
				XMVECTOR S, T, source_world, parent_world;
				wi::math::DecomposeTransform(&S, &source_world, &T, RestWorldMatrix(*source_scene, source->bones[i]));
				wi::math::DecomposeTransform(&S, &parent_world, &T, scene.ComputeParentMatrixRecursive(entity));
				const XMVECTOR world = XMQuaternionMultiply(XMQuaternionRotationRollPitchYaw(0, 0, XM_PIDIV2), source_world);
				XMStoreFloat4(&scene.transforms.GetComponent(entity)->rotation_local, XMQuaternionMultiply(world, XMQuaternionInverse(parent_world)));
			}
		}
		skeleton.humanoid = CreateEntity();
		HumanoidComponent& humanoid = scene.humanoids.Create(skeleton.humanoid);
		for (size_t i = 0; i < descs.size(); ++i)
		{
			humanoid.bones[size_t(descs[i].bone)] = skeleton.bones[i];
		}
		return skeleton;
	}

	// The times of the keys, not a second apart (the tangents of cubic spline keys are scaled by the time between the keys):
	const float key_times[] = { 0, 0.5f, 2 };

	// A channel of the source animation on one of the bones, with keys at the key times (cubic spline keys are an
	//	in-tangent, a value and an out-tangent). linear: the retargeting of the keys is a linear (for translations affine) function
	//	of them for this bone, so the baked keys play exactly the same at all times. Otherwise they play the same at the keys, and
	//	cubic splines also with the same slopes there.
	struct ChannelDesc
	{
		std::string name;
		int bone;
		Path path;
		Mode mode;
		wi::vector<float> keys;
		bool linear;
	};

	Entity CreateAnimation(Scene& scene, const Skeleton& skeleton, const wi::vector<ChannelDesc>& channels)
	{
		Entity entity = CreateEntity();
		AnimationComponent& animation = scene.animations.Create(entity);
		animation.start = 0;
		animation.end = 2;
		for (const ChannelDesc& desc : channels)
		{
			Entity data = CreateEntity();
			AnimationDataComponent& animation_data = scene.animation_datas.Create(data);
			animation_data.keyframe_times = { key_times[0], key_times[1], key_times[2] };
			animation_data.keyframe_data = desc.keys;
			AnimationComponent::AnimationSampler& sampler = animation.samplers.emplace_back();
			sampler.data = data;
			sampler.mode = desc.mode;
			AnimationComponent::AnimationChannel& channel = animation.channels.emplace_back();
			channel.target = skeleton.bones[desc.bone];
			channel.samplerIndex = int(animation.samplers.size() - 1);
			channel.path = desc.path;
		}
		return entity;
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

	// The animated value of a bone: the translation, the scale, or the normalized rotation (as either of its two quaternions)
	XMVECTOR Value(const Scene& scene, Entity bone, Path path)
	{
		const TransformComponent& transform = *scene.transforms.GetComponent(bone);
		switch (path)
		{
		case Path::TRANSLATION:
			return XMLoadFloat3(&transform.translation_local);
		case Path::SCALE:
			return XMLoadFloat3(&transform.scale_local);
		default:
			return XMQuaternionNormalize(XMLoadFloat4(&transform.rotation_local));
		}
	}
	// A rotation as the quaternion in the hemisphere of the reference
	XMVECTOR Aligned(XMVECTOR V, XMVECTOR reference, Path path)
	{
		return path == Path::ROTATION && XMVectorGetX(XMVector4Dot(V, reference)) < 0 ? -V : V;
	}
	float Distance(XMVECTOR A, XMVECTOR B)
	{
		return XMVectorGetX(XMVector4Length(A - B));
	}

	XMVECTOR Key(const AnimationDataComponent& data, size_t key, bool cubic)
	{
		return XMLoadFloat4((const XMFLOAT4*)&data.keyframe_data[key * (cubic ? 12 : 4) + (cubic ? 4 : 0)]);
	}

	// The source animation is retargeted with and without baking, and both are played:
	void TestRetargetAnimation(bool same_scene)
	{
		const std::string prefix = same_scene ? "(within the scene) " : "(from an other scene) ";

		// The source skeleton has uniform scales. The destination skeleton has uniform scales too, except the left upper leg:
		//	its own rotation is still retargeted by a linear function, but the bones under it are retargeted by nonlinear
		//	functions (their matrices are sheared).
		const wi::vector<BoneDesc> source_bones = {
			{ Bone::Hips, -1, XMFLOAT3(0, 1, 0), XMFLOAT3(0.1f, 0.2f, 0.3f), XMFLOAT3(1.1f, 1.1f, 1.1f) },
			{ Bone::Spine, 0, XMFLOAT3(0, 0.2f, 0.05f), XMFLOAT3(-0.2f, 0.1f, 0.4f), XMFLOAT3(1, 1, 1) },
			{ Bone::Chest, 1, XMFLOAT3(0, 0.25f, 0), XMFLOAT3(0.3f, -0.1f, 0.2f), XMFLOAT3(0.9f, 0.9f, 0.9f) },
			{ Bone::UpperChest, 2, XMFLOAT3(0.01f, 0.2f, 0), XMFLOAT3(0.1f, 0.3f, -0.2f), XMFLOAT3(1, 1, 1) },
			{ Bone::LeftUpperLeg, 0, XMFLOAT3(0.1f, -0.05f, 0), XMFLOAT3(0.2f, 0.1f, 0.05f), XMFLOAT3(1, 1, 1) },
			{ Bone::LeftLowerLeg, 4, XMFLOAT3(0, -0.45f, 0), XMFLOAT3(0.2f, 0, 0), XMFLOAT3(1, 1, 1) },
			{ Bone::LeftFoot, 5, XMFLOAT3(0, -0.4f, 0), XMFLOAT3(-0.3f, 0, 0.1f), XMFLOAT3(1, 1, 1) },
			{ Bone::Neck, 3, XMFLOAT3(0, 0.15f, 0), XMFLOAT3(0.1f, 0, 0.1f), XMFLOAT3(1, 1, 1) },
		};
		const wi::vector<BoneDesc> destination_bones = {
			{ Bone::Hips, -1, XMFLOAT3(0.2f, 0.9f, -0.1f), XMFLOAT3(-0.3f, 0.5f, 0.2f), XMFLOAT3(0.8f, 0.8f, 0.8f) },
			{ Bone::Spine, 0, XMFLOAT3(0, 0.25f, 0), XMFLOAT3(0.4f, -0.2f, 0.1f), XMFLOAT3(1.2f, 1.2f, 1.2f) },
			{ Bone::Chest, 1, XMFLOAT3(0, 0.3f, 0.02f), XMFLOAT3(0, 0, 0), XMFLOAT3(1, 1, 1), true },
			{ Bone::UpperChest, 2, XMFLOAT3(0, 0.2f, 0), XMFLOAT3(0, 0, 0), XMFLOAT3(1.1f, 1.1f, 1.1f), true },
			{ Bone::LeftUpperLeg, 0, XMFLOAT3(0.12f, -0.02f, 0.01f), XMFLOAT3(0.1f, -0.1f, 0.2f), XMFLOAT3(1, 1.1f, 0.95f) },
			{ Bone::LeftLowerLeg, 4, XMFLOAT3(0, -0.5f, 0), XMFLOAT3(0.1f, 0.2f, 0), XMFLOAT3(1, 1, 1) },
			{ Bone::LeftFoot, 5, XMFLOAT3(0, -0.45f, 0), XMFLOAT3(-0.2f, 0.1f, 0), XMFLOAT3(1, 1, 1) },
			{ Bone::Neck, 3, XMFLOAT3(0, 0.18f, 0), XMFLOAT3(0, 0, 0), XMFLOAT3(1, 1, 1), true },
		};

		// Rotation keys go to the other hemisphere and back, rotation tangents have parts along the rotations too:
		const wi::vector<ChannelDesc> channels = {
			{ "linear translation", 0, Path::TRANSLATION, Mode::LINEAR, { 0, 1, 0, 0.2f, 1.1f, -0.1f, 0.1f, 0.95f, 0.3f }, true },
			{ "linear rotation", 0, Path::ROTATION, Mode::LINEAR, Flatten({ Rotation(0.1f, 0.2f, 0.3f), Rotation(0.5f, -0.3f, 0.4f, -1), Rotation(0.9f, 0.1f, -0.2f) }), true },
			{ "cubic translation", 1, Path::TRANSLATION, Mode::CUBICSPLINE, {
				0.5f, 0, 0.1f, 0, 0.2f, 0.05f, 0.3f, -0.2f, 0.4f,
				-0.2f, 0.4f, 0, 0.1f, 0.25f, 0, 0.6f, 0.1f, -0.3f,
				0, 0, 0.5f, -0.05f, 0.2f, 0.1f, 0.2f, 0.2f, 0.2f,
			}, true },
			{ "cubic rotation", 1, Path::ROTATION, Mode::CUBICSPLINE, Flatten({
				XMFLOAT4(0.1f, 0.2f, -0.1f, 0.3f), Rotation(-0.2f, 0.1f, 0.4f), XMFLOAT4(-0.2f, 0.1f, 0.4f, 0.05f),
				XMFLOAT4(0.3f, -0.1f, 0.2f, -0.2f), Rotation(0.3f, 0.4f, 0.1f, -1), XMFLOAT4(0.05f, 0.2f, 0.1f, 0.1f),
				XMFLOAT4(-0.1f, 0.3f, 0, 0.2f), Rotation(-0.5f, 0.2f, 0.6f), XMFLOAT4(0.2f, -0.2f, 0.1f, -0.1f),
			}), true },
			{ "linear scale", 2, Path::SCALE, Mode::LINEAR, { 0.9f, 0.9f, 0.9f, 1.2f, 0.8f, 1, 0.7f, 1.1f, 1.3f }, true },
			{ "cubic scale", 3, Path::SCALE, Mode::CUBICSPLINE, {
				-0.3f, 0.2f, 0.1f, 1, 1, 1, 0.4f, -0.5f, 0.2f,
				0.1f, -0.2f, -0.3f, 1.3f, 0.7f, 1.1f, -0.4f, 0.3f, 0.2f,
				0.2f, 0.2f, -0.6f, 0.8f, 1.2f, 0.9f, 0.1f, 0.1f, 0.1f,
			}, true },
			{ "cubic rotation with non-uniform scale", 4, Path::ROTATION, Mode::CUBICSPLINE, Flatten({
				XMFLOAT4(0.2f, 0, 0.1f, -0.1f), Rotation(0.3f, 0.1f, 0), XMFLOAT4(-0.1f, 0.2f, 0, 0.1f),
				XMFLOAT4(0, 0.1f, -0.2f, 0.1f), Rotation(0.1f, -0.3f, 0.2f, -1), XMFLOAT4(0.2f, 0.1f, 0.1f, 0),
				XMFLOAT4(0.1f, 0, 0.1f, 0.1f), Rotation(0.4f, 0.2f, -0.1f), XMFLOAT4(-0.1f, -0.1f, 0.2f, 0),
			}), true },
			{ "cubic translation under non-uniform scale", 5, Path::TRANSLATION, Mode::CUBICSPLINE, {
				0.1f, -0.2f, 0, 0, -0.45f, 0, 0.2f, 0.1f, 0.3f,
				0.3f, 0, -0.1f, 0.05f, -0.5f, 0.1f, -0.2f, 0.2f, 0,
				0, 0.1f, 0.1f, 0.02f, -0.4f, -0.05f, 0.1f, 0, -0.2f,
			}, true },
			{ "cubic rotation under non-uniform scale", 5, Path::ROTATION, Mode::CUBICSPLINE, Flatten({
				XMFLOAT4(0.1f, 0.1f, 0, -0.05f), Rotation(0.35f, 0.1f, 0), XMFLOAT4(-0.1f, 0.1f, 0.1f, 0.05f),
				XMFLOAT4(0.05f, -0.1f, 0.1f, 0.1f), Rotation(0.6f, -0.1f, 0.1f, -1), XMFLOAT4(0.1f, 0.05f, -0.1f, 0),
				XMFLOAT4(-0.05f, 0.1f, 0.05f, 0.1f), Rotation(0.15f, 0.2f, -0.1f), XMFLOAT4(0.1f, 0, 0.05f, -0.1f),
			}), false },
			{ "cubic scale under non-uniform scale", 5, Path::SCALE, Mode::CUBICSPLINE, {
				0.2f, -0.3f, 0.1f, 1, 1, 1, -0.1f, 0.4f, 0.2f,
				-0.3f, 0.1f, 0.2f, 1.2f, 0.9f, 1.1f, 0.2f, -0.2f, 0.3f,
				0.1f, 0.2f, -0.1f, 0.9f, 1.1f, 0.8f, 0, 0.3f, -0.2f,
			}, false },
			{ "linear rotation under non-uniform scale", 6, Path::ROTATION, Mode::LINEAR, Flatten({ Rotation(-0.2f, 0.1f, 0.1f), Rotation(-0.5f, 0, 0.2f, -1), Rotation(0.1f, -0.1f, 0) }), false },
			{ "linear scale under non-uniform scale", 6, Path::SCALE, Mode::LINEAR, { 1, 1, 1, 1.3f, 0.8f, 1.1f, 0.9f, 1.2f, 0.7f }, false },
			// A scale that grows from zero: the retargeted scale has a kink at zero, which the baked tangent must not span
			{ "cubic scale from zero", 7, Path::SCALE, Mode::CUBICSPLINE, {
				0, 0, 0, 0, 0, 0, 3, 3, 3,
				1, 1, 1, 1, 1, 1, 0.5f, 0.5f, 0.5f,
				0.2f, 0.2f, 0.2f, 1.2f, 1.2f, 1.2f, 0, 0, 0,
			}, true },
		};

		Scene other_scene;
		Scene scene;
		Scene& source_scene = same_scene ? scene : other_scene;
		const Skeleton source = CreateSkeleton(source_scene, source_bones);
		const Skeleton destination = CreateSkeleton(scene, destination_bones, &source_scene, &source);
		const Entity source_animation = CreateAnimation(source_scene, source, channels);

		const Scene* src_scene = same_scene ? nullptr : &source_scene;
		const Entity baked = scene.RetargetAnimation(destination.humanoid, source_animation, true, src_scene);
		const Entity runtime = scene.RetargetAnimation(destination.humanoid, source_animation, false, src_scene);
		Check(baked != INVALID_ENTITY && runtime != INVALID_ENTITY, prefix + "the animation was not retargeted");
		if (baked == INVALID_ENTITY || runtime == INVALID_ENTITY)
			return;

		// The values of the channels' bones, played at the time (stored as XMFLOAT4, the rotations as unit quaternions):
		auto playback = [&](Entity animation, float time) {
			Play(scene, animation, time);
			wi::vector<XMFLOAT4> values(channels.size());
			for (size_t c = 0; c < channels.size(); ++c)
			{
				XMStoreFloat4(&values[c], Value(scene, destination.bones[channels[c].bone], channels[c].path));
			}
			return values;
		};

		// The same values at many times (only at the keys for nonlinear retargeting):
		wi::vector<float> max_difference(channels.size(), 0.0f);
		for (int i = 0; i <= 40; ++i)
		{
			const float time = i / 20.0f;
			const wi::vector<XMFLOAT4> baked_values = playback(baked, time);
			const wi::vector<XMFLOAT4> runtime_values = playback(runtime, time);
			for (size_t c = 0; c < channels.size(); ++c)
			{
				if (channels[c].linear || time == key_times[0] || time == key_times[1] || time == key_times[2])
				{
					const XMVECTOR baked_value = XMLoadFloat4(&baked_values[c]);
					const XMVECTOR runtime_value = Aligned(XMLoadFloat4(&runtime_values[c]), baked_value, channels[c].path);
					max_difference[c] = std::max(max_difference[c], Distance(baked_value, runtime_value));
				}
			}
		}
		for (size_t c = 0; c < channels.size(); ++c)
		{
			Check(max_difference[c] < 1e-4f, prefix + channels[c].name + ": the baked keys don't play like the keys retargeted while playing, difference: " + std::to_string(max_difference[c]));
		}

		// The same slopes at the keys for cubic splines, from both sides, where the tangents of the keys are used. The slopes
		//	are extrapolated from two steps (Richardson), the rotations are compared in the hemisphere of the baked key:
		auto slopes = [&](Entity animation, float time, float side, const wi::vector<XMFLOAT4>* hemispheres) {
			const float step = 0.02f;
			const wi::vector<XMFLOAT4> at_key = playback(animation, time);
			const wi::vector<XMFLOAT4> half = playback(animation, time + side * step * 0.5f);
			const wi::vector<XMFLOAT4> full = playback(animation, time + side * step);
			wi::vector<XMFLOAT4> result(channels.size());
			for (size_t c = 0; c < channels.size(); ++c)
			{
				const Path path = channels[c].path;
				const XMVECTOR key = Aligned(XMLoadFloat4(&at_key[c]), XMLoadFloat4(hemispheres == nullptr ? &at_key[c] : &(*hemispheres)[c]), path);
				const XMVECTOR slope_half = (Aligned(XMLoadFloat4(&half[c]), key, path) - key) / (step * 0.5f);
				const XMVECTOR slope_full = (Aligned(XMLoadFloat4(&full[c]), key, path) - key) / step;
				XMStoreFloat4(&result[c], side * (2 * slope_half - slope_full));
			}
			return result;
		};
		wi::vector<float> max_slope_difference(channels.size(), 0.0f);
		const float sides[][2] = { { key_times[0], 1 }, { key_times[1], -1 }, { key_times[1], 1 }, { key_times[2], -1 } }; // time, side
		for (const auto& side : sides)
		{
			const wi::vector<XMFLOAT4> baked_at_key = playback(baked, side[0]);
			const wi::vector<XMFLOAT4> baked_slopes = slopes(baked, side[0], side[1], nullptr);
			const wi::vector<XMFLOAT4> runtime_slopes = slopes(runtime, side[0], side[1], &baked_at_key);
			for (size_t c = 0; c < channels.size(); ++c)
			{
				if (channels[c].mode == Mode::CUBICSPLINE)
				{
					max_slope_difference[c] = std::max(max_slope_difference[c], Distance(XMLoadFloat4(&baked_slopes[c]), XMLoadFloat4(&runtime_slopes[c])));
				}
			}
		}
		for (size_t c = 0; c < channels.size(); ++c)
		{
			if (channels[c].mode == Mode::CUBICSPLINE)
			{
				Check(max_slope_difference[c] < 1e-3f, prefix + channels[c].name + ": the baked tangents don't play like the tangents retargeted while playing, slope difference: " + std::to_string(max_slope_difference[c]));
			}
		}

		// The hemispheres of the baked rotation keys: cubic splines keep the relation of the source keys (the interpolation
		//	goes to the other hemisphere where the source goes there), the others go the short way (like slerp does):
		const AnimationComponent& baked_animation = *scene.animations.GetComponent(baked);
		const AnimationComponent& source_component = *source_scene.animations.GetComponent(source_animation);
		for (size_t c = 0; c < channels.size(); ++c)
		{
			if (channels[c].path != Path::ROTATION)
				continue;
			const bool cubic = channels[c].mode == Mode::CUBICSPLINE;
			const AnimationDataComponent* baked_data = nullptr;
			for (const AnimationComponent::AnimationChannel& channel : baked_animation.channels)
			{
				if (channel.target == destination.bones[channels[c].bone] && channel.path == Path::ROTATION)
				{
					baked_data = scene.animation_datas.GetComponent(baked_animation.samplers[channel.samplerIndex].data);
				}
			}
			const AnimationDataComponent& source_data = *source_scene.animation_datas.GetComponent(source_component.samplers[source_component.channels[c].samplerIndex].data);
			Check(baked_data != nullptr && baked_data->keyframe_data.size() == source_data.keyframe_data.size(), prefix + channels[c].name + ": the baked keys were not found");
			if (baked_data == nullptr || baked_data->keyframe_data.size() != source_data.keyframe_data.size())
				continue;
			bool kept = true;
			for (size_t key = 1; key < 3; ++key)
			{
				const bool baked_flips = XMVectorGetX(XMVector4Dot(Key(*baked_data, key, cubic), Key(*baked_data, key - 1, cubic))) < 0;
				const bool source_flips = cubic && XMVectorGetX(XMVector4Dot(Key(source_data, key, cubic), Key(source_data, key - 1, cubic))) < 0;
				kept &= baked_flips == source_flips;
			}
			Check(kept, prefix + channels[c].name + (cubic ? ": the baked keys don't keep the hemispheres of the source keys" : ": the baked keys don't go the short way"));
		}
	}
}

int main()
{
	wi::jobsystem::Initialize();

	TestRetargetAnimation(false); // like the editor's humanoid window
	TestRetargetAnimation(true); // like the editor's animation window

	wi::jobsystem::ShutDown();

	if (failures > 0)
	{
		std::cerr << failures << " check(s) failed\n";
		return 1;
	}
	std::cout << "RetargetAnimation test passed\n";
	return 0;
}
