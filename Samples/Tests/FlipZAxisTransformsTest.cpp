#include "WickedEngine.h"
#include "../../Editor/ModelImporter_FlipZAxis.h"

#include <cstring>
#include <iostream>
#include <string>

// Regression test for the Z axis flip of the model importers (glTF, FBX) on a skinned, animated hierarchy with
//	non-uniform scaling, where the world matrices have shear and can't be decomposed.

using namespace wi::ecs;
using namespace wi::scene;

namespace
{
	int failures = 0;

	void Check(bool condition, const std::string& message)
	{
		if (!condition)
		{
			std::cerr << "FlipZAxisTransforms test failed: " << message << '\n';
			failures++;
		}
	}

	float MaxDifference(const XMMATRIX& A, const XMMATRIX& B)
	{
		float diff = 0;
		for (int i = 0; i < 4; ++i)
		{
			diff = std::max(diff, XMVectorGetX(XMVector4Length(A.r[i] - B.r[i])));
		}
		return diff;
	}
	XMMATRIX Mirror(const XMMATRIX& M)
	{
		const XMMATRIX F = XMMatrixScaling(1, 1, -1);
		return F * M * F;
	}

	void UpdateWorldMatrices(Scene& scene)
	{
		wi::jobsystem::context ctx;
		scene.RunTransformUpdateSystem(ctx);
		wi::jobsystem::Wait(ctx);
		scene.RunHierarchyUpdateSystem(ctx);
		wi::jobsystem::Wait(ctx);
	}

	Entity CreateNode(Scene& scene, const XMFLOAT3& scale, const XMFLOAT3& rollPitchYawDegrees, const XMFLOAT3& translation)
	{
		Entity entity = CreateEntity();
		TransformComponent& transform = scene.transforms.Create(entity);
		transform.Scale(scale);
		transform.RotateRollPitchYaw(XMFLOAT3(
			XMConvertToRadians(rollPitchYawDegrees.x),
			XMConvertToRadians(rollPitchYawDegrees.y),
			XMConvertToRadians(rollPitchYawDegrees.z)
		));
		transform.Translate(translation);
		transform.UpdateTransform();
		return entity;
	}

	Entity CreateAnimationData(Scene& scene, Entity parent, const wi::vector<float>& keyframe_data)
	{
		Entity entity = CreateEntity();
		AnimationDataComponent& data = scene.animation_datas.Create(entity);
		data.keyframe_times = { 0, 1 };
		data.keyframe_data = keyframe_data;
		scene.Component_Attach(entity, parent);
		return entity;
	}

	void Test(bool halfTurnRoot)
	{
		const std::string variant = halfTurnRoot ? " (root turned half about Y)" : "";

		Scene other_scene; // the source of an animation retargeted from an other scene
		Scene scene;
		Entity root = CreateNode(scene, XMFLOAT3(1, 1, 1), XMFLOAT3(0, halfTurnRoot ? 180.0f : 0.0f, 0), XMFLOAT3(0, 0, 0));
		Entity bones[3] = {
			CreateNode(scene, XMFLOAT3(2, 1, 1), XMFLOAT3(0, 10, 0), XMFLOAT3(1, 0, 0.5f)),
			CreateNode(scene, XMFLOAT3(1, 1, 1), XMFLOAT3(20, 45, 0), XMFLOAT3(0, 0, 1)),
			CreateNode(scene, XMFLOAT3(1, 0.5f, 1), XMFLOAT3(0, 30, 15), XMFLOAT3(0, 1, 0.25f)),
		};
		scene.Component_Attach(bones[0], root, true);
		scene.Component_Attach(bones[1], bones[0], true);
		scene.Component_Attach(bones[2], bones[1], true);

		Entity armatureEntity = CreateNode(scene, XMFLOAT3(1, 1, 1), XMFLOAT3(0, 0, 0), XMFLOAT3(0, 0, 0));
		scene.Component_Attach(armatureEntity, root, true);
		ArmatureComponent& armature = scene.armatures.Create(armatureEntity);
		UpdateWorldMatrices(scene);
		const XMMATRIX armature_world = XMLoadFloat4x4(&scene.transforms.GetComponent(armatureEntity)->world);
		for (Entity bone : bones)
		{
			// The bind pose is the current pose, relative to the armature:
			armature.boneCollection.push_back(bone);
			const XMMATRIX bone_world = XMLoadFloat4x4(&scene.transforms.GetComponent(bone)->world);
			XMFLOAT4X4 bind;
			XMStoreFloat4x4(&bind, XMMatrixInverse(nullptr, bone_world * XMMatrixInverse(nullptr, armature_world)));
			armature.inverseBindMatrices.push_back(bind);
		}

		Entity animationEntity = CreateEntity();
		AnimationComponent& animation = scene.animations.Create(animationEntity);
		scene.Component_Attach(animationEntity, root);
		const wi::vector<float> translation_keys = { 1, 2, 3, 4, 5, 6 };
		const wi::vector<float> rotation_keys = { 0.1f, 0.2f, 0.3f, 0.9f, 0.4f, 0.5f, 0.6f, 0.5f };
		const wi::vector<float> scale_keys = { 1, 2, 3, 4, 5, 6 };
		const wi::vector<float> shared_keys = { 1, 1, 1, 2, 2, 2 };
		XMFLOAT4X4 retarget_dst, retarget_src;
		XMStoreFloat4x4(&retarget_dst, XMMatrixRotationRollPitchYaw(0.3f, 0.5f, 0.7f) * XMMatrixTranslation(1, 2, 3));
		XMStoreFloat4x4(&retarget_src, XMMatrixScaling(1, 2, 1) * XMMatrixRotationRollPitchYaw(-0.2f, 0.4f, 0.1f) * XMMatrixTranslation(-1, 0, 2));
		const AnimationComponent::AnimationChannel::Path paths[] = {
			AnimationComponent::AnimationChannel::Path::TRANSLATION,
			AnimationComponent::AnimationChannel::Path::ROTATION,
			AnimationComponent::AnimationChannel::Path::SCALE,
		};
		const wi::vector<float>* keys[] = { &translation_keys, &rotation_keys, &scale_keys };
		Entity dataEntities[3];
		for (int i = 0; i < 3; ++i)
		{
			dataEntities[i] = CreateAnimationData(scene, animationEntity, *keys[i]);
			AnimationComponent::AnimationSampler& sampler = animation.samplers.emplace_back();
			sampler.data = dataEntities[i];
			AnimationComponent::AnimationChannel& channel = animation.channels.emplace_back();
			channel.target = bones[1];
			channel.samplerIndex = i;
			channel.path = paths[i];
		}
		{
			// Two channels using the same sampler (allowed in glTF), the data must be flipped once:
			AnimationComponent::AnimationChannel& shared = animation.channels.emplace_back();
			shared.target = bones[2];
			shared.samplerIndex = 0;
			shared.path = AnimationComponent::AnimationChannel::Path::TRANSLATION;
		}
		{
			// Channels without data in this scene (like an animation retargeted without baking) must be skipped:
			AnimationComponent::AnimationSampler& sampler = animation.samplers.emplace_back();
			sampler.data = CreateEntity();
			AnimationComponent::AnimationChannel& channel = animation.channels.emplace_back();
			channel.target = bones[1];
			channel.samplerIndex = int(animation.samplers.size() - 1);
			channel.path = AnimationComponent::AnimationChannel::Path::TRANSLATION;
			AnimationComponent::AnimationChannel& invalid = animation.channels.emplace_back();
			invalid.target = bones[1];
			invalid.samplerIndex = 1000; // (not the sampler count, more samplers are added below)
			invalid.path = AnimationComponent::AnimationChannel::Path::ROTATION;
		}
		Entity sharedDataEntity = INVALID_ENTITY;
		const size_t shared_channel_index = animation.channels.size(); // then the retargeted channels follow
		{
			// A sampler used by a translation and a scale channel (allowed in glTF): only the translation may be mirrored
			sharedDataEntity = CreateAnimationData(scene, animationEntity, shared_keys);
			AnimationComponent::AnimationSampler& sampler = animation.samplers.emplace_back();
			sampler.data = sharedDataEntity;
			const int samplerIndex = int(animation.samplers.size() - 1);
			AnimationComponent::AnimationChannel& translation = animation.channels.emplace_back();
			translation.target = bones[0];
			translation.samplerIndex = samplerIndex;
			translation.path = AnimationComponent::AnimationChannel::Path::TRANSLATION;
			AnimationComponent::AnimationChannel& scale = animation.channels.emplace_back();
			scale.target = bones[0];
			scale.samplerIndex = samplerIndex;
			scale.path = AnimationComponent::AnimationChannel::Path::SCALE;
		}
		{
			// Channels retargeted without baking, from a source in this scene and from a source in an other scene
			AnimationComponent::RetargetSourceData& same = animation.retargets.emplace_back();
			same.source = bones[0];
			same.dstRelativeMatrix = retarget_dst;
			same.srcRelativeParentMatrix = retarget_src;
			AnimationComponent::AnimationChannel& same_channel = animation.channels.emplace_back();
			same_channel.target = bones[2];
			same_channel.samplerIndex = 0;
			same_channel.retargetIndex = 0;
			same_channel.path = AnimationComponent::AnimationChannel::Path::TRANSLATION;

			AnimationComponent::RetargetSourceData& from_other = animation.retargets.emplace_back();
			from_other.source = CreateNode(other_scene, XMFLOAT3(1, 1, 1), XMFLOAT3(0, 0, 0), XMFLOAT3(0, 0, 0));
			from_other.dstRelativeMatrix = retarget_dst;
			from_other.srcRelativeParentMatrix = retarget_src;
			AnimationComponent::AnimationSampler& other_sampler = animation.samplers.emplace_back();
			other_sampler.data = CreateEntity();
			other_scene.animation_datas.Create(other_sampler.data).keyframe_data = translation_keys;
			other_sampler.scene = &other_scene;
			AnimationComponent::AnimationChannel& other_channel = animation.channels.emplace_back();
			other_channel.target = bones[1];
			other_channel.samplerIndex = int(animation.samplers.size() - 1);
			other_channel.retargetIndex = 1;
			other_channel.path = AnimationComponent::AnimationChannel::Path::TRANSLATION;

			// The retarget from this scene also used by a channel from the other scene: each must be mirrored for its own source
			const int other_sampler_index = other_channel.samplerIndex; // (before adding a channel, which can move the one referenced)
			AnimationComponent::AnimationChannel& shared_retarget_channel = animation.channels.emplace_back();
			shared_retarget_channel.target = bones[0];
			shared_retarget_channel.samplerIndex = other_sampler_index;
			shared_retarget_channel.retargetIndex = 0;
			shared_retarget_channel.path = AnimationComponent::AnimationChannel::Path::TRANSLATION;

			// A translation sampler of the other scene on an entity that has (scale) data in this scene too: this scene's data is not flipped by it
			AnimationComponent::AnimationSampler& colliding_sampler = animation.samplers.emplace_back();
			colliding_sampler.data = dataEntities[2];
			colliding_sampler.scene = &other_scene;
			other_scene.animation_datas.Create(dataEntities[2]).keyframe_data = translation_keys;
			AnimationComponent::AnimationChannel& colliding_channel = animation.channels.emplace_back();
			colliding_channel.target = bones[2];
			colliding_channel.samplerIndex = int(animation.samplers.size() - 1);
			colliding_channel.path = AnimationComponent::AnimationChannel::Path::TRANSLATION;
		}
		{
			// (created last, because creating an other animation can move the one referenced above)
			// An other animation using the same data (like an animation retargeted without baking in the same scene), the data must be flipped once:
			Entity otherAnimationEntity = CreateEntity();
			AnimationComponent& other = scene.animations.Create(otherAnimationEntity);
			scene.Component_Attach(otherAnimationEntity, root);
			AnimationComponent::AnimationSampler& sampler = other.samplers.emplace_back();
			sampler.data = dataEntities[1];
			AnimationComponent::AnimationChannel& channel = other.channels.emplace_back();
			channel.target = bones[2];
			channel.samplerIndex = 0;
			channel.path = AnimationComponent::AnimationChannel::Path::ROTATION;
		}

		XMMATRIX world_original[3];
		for (int i = 0; i < 3; ++i)
		{
			world_original[i] = XMLoadFloat4x4(&scene.transforms.GetComponent(bones[i])->world);
		}
		const XMMATRIX root_world_original = XMLoadFloat4x4(&scene.transforms.GetComponent(root)->world);
		const wi::vector<XMFLOAT4X4> binds_original = armature.inverseBindMatrices;
		wi::vector<TransformComponent> transforms_original;
		for (size_t i = 0; i < scene.transforms.GetCount(); ++i)
		{
			transforms_original.push_back(scene.transforms[i]);
		}

		XMVECTOR S, R, T;
		Check(!XMMatrixDecompose(&S, &R, &T, world_original[1]), "precondition: a world matrix of the test hierarchy should have shear" + variant);

		const size_t sampler_count = scene.animations.GetComponent(animationEntity)->samplers.size();
		const size_t retarget_count = scene.animations.GetComponent(animationEntity)->retargets.size();
		const size_t data_count = scene.animation_datas.GetCount();
		wi::vector<std::pair<int, int>> channel_indices; // sampler and retarget of each channel
		for (const auto& channel : scene.animations.GetComponent(animationEntity)->channels)
		{
			channel_indices.push_back({ channel.samplerIndex, channel.retargetIndex });
		}

		FlipZAxisSplits splits;
		FlipZAxisTransforms(scene, root, &splits);
		UpdateWorldMatrices(scene);

		const char* names[3] = { "first", "second", "third" };
		for (int i = 0; i < 3; ++i)
		{
			const XMMATRIX world = XMLoadFloat4x4(&scene.transforms.GetComponent(bones[i])->world);
			Check(MaxDifference(world, Mirror(world_original[i])) < 1e-5f, std::string("world matrix of the ") + names[i] + " node is not the mirror of the original" + variant);
		}
		Check(MaxDifference(XMLoadFloat4x4(&scene.transforms.GetComponent(root)->world), root_world_original) == 0, "the root was changed" + variant);

		const ArmatureComponent& flipped = *scene.armatures.GetComponent(armatureEntity);
		const XMMATRIX armature_world_inverse = XMMatrixInverse(nullptr, XMLoadFloat4x4(&scene.transforms.GetComponent(armatureEntity)->world));
		for (int i = 0; i < 3; ++i)
		{
			const XMMATRIX bind = XMLoadFloat4x4(&flipped.inverseBindMatrices[i]);
			Check(MaxDifference(bind, Mirror(XMLoadFloat4x4(&binds_original[i]))) == 0, std::string("inverse bind matrix of the ") + names[i] + " bone is not the mirror of the original" + variant);
			const XMMATRIX skinning = bind * XMLoadFloat4x4(&scene.transforms.GetComponent(bones[i])->world) * armature_world_inverse;
			Check(MaxDifference(skinning, XMMatrixIdentity()) < 1e-5f, std::string("skinning matrix of the ") + names[i] + " bone in bind pose is not the identity" + variant);
		}

		const wi::vector<float>& translation_flipped = scene.animation_datas.GetComponent(dataEntities[0])->keyframe_data;
		const wi::vector<float>& rotation_flipped = scene.animation_datas.GetComponent(dataEntities[1])->keyframe_data;
		const wi::vector<float>& scale_flipped = scene.animation_datas.GetComponent(dataEntities[2])->keyframe_data;
		Check(translation_flipped == wi::vector<float>({ 1, 2, -3, 4, 5, -6 }), "translation keys were not mirrored" + variant);
		Check(rotation_flipped == wi::vector<float>({ -0.1f, -0.2f, 0.3f, 0.9f, -0.4f, -0.5f, 0.6f, 0.5f }), "rotation keys were not mirrored" + variant);
		Check(scale_flipped == scale_keys, "scale keys were changed" + variant);

		const AnimationComponent& flipped_animation = *scene.animations.GetComponent(animationEntity);
		{
			// The shared data stays with the translation and is mirrored, the scale channel has an unchanged copy
			const AnimationComponent::AnimationChannel& translation = flipped_animation.channels[shared_channel_index];
			const AnimationComponent::AnimationChannel& scale = flipped_animation.channels[shared_channel_index + 1];
			const Entity translationData = flipped_animation.samplers[translation.samplerIndex].data;
			const Entity scaleData = flipped_animation.samplers[scale.samplerIndex].data;
			Check(translationData == sharedDataEntity && scene.animation_datas.GetComponent(translationData)->keyframe_data == wi::vector<float>({ 1, 1, -1, 2, 2, -2 }), "data shared by a translation and a scale channel was not mirrored for the translation" + variant);
			Check(scaleData != sharedDataEntity && scene.animation_datas.Contains(scaleData) && scene.animation_datas.GetComponent(scaleData)->keyframe_data == shared_keys, "data shared by a translation and a scale channel was mirrored for the scale" + variant);
		}
		{
			const XMMATRIX F = XMMatrixScaling(1, 1, -1);
			const XMMATRIX dst = XMLoadFloat4x4(&retarget_dst);
			const XMMATRIX src = XMLoadFloat4x4(&retarget_src);
			const AnimationComponent::RetargetSourceData& same = flipped_animation.retargets[0];
			const AnimationComponent::RetargetSourceData& from_other = flipped_animation.retargets[1];
			Check(MaxDifference(XMLoadFloat4x4(&same.dstRelativeMatrix), F * dst * F) == 0 && MaxDifference(XMLoadFloat4x4(&same.srcRelativeParentMatrix), F * src * F) == 0, "the matrices of a retarget from this scene were not mirrored" + variant);
			Check(MaxDifference(XMLoadFloat4x4(&from_other.dstRelativeMatrix), F * dst) == 0 && MaxDifference(XMLoadFloat4x4(&from_other.srcRelativeParentMatrix), src * F) == 0, "the matrices of a retarget from an other scene were not mirrored" + variant);
			Check(other_scene.animation_datas.GetComponent(flipped_animation.samplers[flipped_animation.channels[shared_channel_index + 3].samplerIndex].data)->keyframe_data == translation_keys, "the keys of an other scene were changed" + variant);

			const AnimationComponent::AnimationChannel& shared_retarget_channel = flipped_animation.channels[shared_channel_index + 4];
			Check(shared_retarget_channel.retargetIndex >= 2 && shared_retarget_channel.retargetIndex < (int)flipped_animation.retargets.size(), "a retarget shared by sources of this and an other scene was not split" + variant);
			if (shared_retarget_channel.retargetIndex >= 2 && shared_retarget_channel.retargetIndex < (int)flipped_animation.retargets.size())
			{
				const AnimationComponent::RetargetSourceData& copy = flipped_animation.retargets[shared_retarget_channel.retargetIndex];
				Check(MaxDifference(XMLoadFloat4x4(&copy.dstRelativeMatrix), F * dst) == 0 && MaxDifference(XMLoadFloat4x4(&copy.srcRelativeParentMatrix), src * F) == 0, "the split retarget was not mirrored for its source in an other scene" + variant);
			}
		}

		// Flipping back restores the original state exactly (this is how the glTF export restores the scene)
		FlipZAxisTransforms(scene, root, &splits);
		bool restored = scene.transforms.GetCount() == transforms_original.size();
		for (size_t i = 0; restored && i < scene.transforms.GetCount(); ++i)
		{
			const TransformComponent& a = scene.transforms[i];
			const TransformComponent& b = transforms_original[i];
			restored =
				std::memcmp(&a.scale_local, &b.scale_local, sizeof(a.scale_local)) == 0 &&
				std::memcmp(&a.rotation_local, &b.rotation_local, sizeof(a.rotation_local)) == 0 &&
				std::memcmp(&a.translation_local, &b.translation_local, sizeof(a.translation_local)) == 0;
		}
		Check(restored, "flipping back did not restore the transforms" + variant);
		Check(std::memcmp(flipped.inverseBindMatrices.data(), binds_original.data(), binds_original.size() * sizeof(XMFLOAT4X4)) == 0, "flipping back did not restore the inverse bind matrices" + variant);
		Check(translation_flipped == translation_keys && rotation_flipped == rotation_keys && scale_flipped == scale_keys, "flipping back did not restore the animation keys" + variant);
		Check(scene.animation_datas.GetComponent(sharedDataEntity)->keyframe_data == shared_keys, "flipping back did not restore shared data" + variant);
		const AnimationComponent& restored_animation = *scene.animations.GetComponent(animationEntity);
		Check(std::memcmp(&restored_animation.retargets[0].dstRelativeMatrix, &retarget_dst, sizeof(XMFLOAT4X4)) == 0 && std::memcmp(&restored_animation.retargets[0].srcRelativeParentMatrix, &retarget_src, sizeof(XMFLOAT4X4)) == 0 &&
			std::memcmp(&restored_animation.retargets[1].dstRelativeMatrix, &retarget_dst, sizeof(XMFLOAT4X4)) == 0 && std::memcmp(&restored_animation.retargets[1].srcRelativeParentMatrix, &retarget_src, sizeof(XMFLOAT4X4)) == 0,
			"flipping back did not restore the retarget matrices" + variant);

		// The splits are undone
		UndoFlipZAxisSplits(scene, splits);
		const AnimationComponent& undone = *scene.animations.GetComponent(animationEntity);
		bool undone_ok = undone.samplers.size() == sampler_count && undone.retargets.size() == retarget_count && scene.animation_datas.GetCount() == data_count && undone.channels.size() == channel_indices.size();
		for (size_t i = 0; undone_ok && i < channel_indices.size(); ++i)
		{
			undone_ok = undone.channels[i].samplerIndex == channel_indices[i].first && undone.channels[i].retargetIndex == channel_indices[i].second;
		}
		Check(undone_ok, "UndoFlipZAxisSplits() did not restore the samplers, retargets and animation data" + variant);
	}
}

int main()
{
	wi::jobsystem::Initialize();

	Test(false);
	Test(true);

	wi::jobsystem::ShutDown();

	if (failures > 0)
	{
		std::cerr << failures << " check(s) failed\n";
		return 1;
	}
	std::cout << "FlipZAxisTransforms test passed\n";
	return 0;
}
