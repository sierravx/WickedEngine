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
			// Channels without data in this scene (like an animation retargeted without baking) must be skipped:
			AnimationComponent::AnimationSampler& sampler = animation.samplers.emplace_back();
			sampler.data = CreateEntity();
			AnimationComponent::AnimationChannel& channel = animation.channels.emplace_back();
			channel.target = bones[1];
			channel.samplerIndex = int(animation.samplers.size() - 1);
			channel.path = AnimationComponent::AnimationChannel::Path::TRANSLATION;
			AnimationComponent::AnimationChannel& invalid = animation.channels.emplace_back();
			invalid.target = bones[1];
			invalid.samplerIndex = int(animation.samplers.size());
			invalid.path = AnimationComponent::AnimationChannel::Path::ROTATION;
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

		FlipZAxisTransforms(scene, root);
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

		// Flipping back restores the original state exactly (this is how the glTF export restores the scene)
		FlipZAxisTransforms(scene, root);
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
