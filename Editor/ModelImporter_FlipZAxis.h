#pragma once
#include "wiScene.h"

// Mirrors the transforms, inverse bind matrices and animation keys of a scene along the Z axis, shared by the model importers
//	to convert between the right handed space of the model files and the left handed space of the engine (the mesh data is flipped by the importers).
//	The transforms are mirrored in local space: translation Z, rotation X and rotation Y are negated and the scale is unchanged.
//	This needs no matrix decomposition, so it is exact for any hierarchy, even with non-uniform scaling and shear.
//	The world matrices follow from the mirrored local transforms on the next scene update.
//	The inverse bind matrices are mirrored as F * M * F, with F = scale(1, 1, -1), which is exact for any matrix.
//	rootEntity is not mirrored (it can be INVALID_ENTITY to mirror every transform). The world matrices of the other
//	transforms are the mirror of the original ones if the root's world matrix commutes with the mirror, like the identity
//	or a half turn about the Y axis.
//	Calling it a second time restores the original state.
inline void FlipZAxisTransforms(wi::scene::Scene& scene, wi::ecs::Entity rootEntity)
{
	for (size_t i = 0; i < scene.transforms.GetCount(); ++i)
	{
		if (scene.transforms.GetEntity(i) == rootEntity)
			continue;

		wi::scene::TransformComponent& transform = scene.transforms[i];
		transform.translation_local.z *= -1.f;
		transform.rotation_local.x *= -1.f;
		transform.rotation_local.y *= -1.f;
		transform.SetDirty();
	}

	// Flip armature's bind pose
	for (size_t i = 0; i < scene.armatures.GetCount(); ++i)
	{
		auto& armature = scene.armatures[i];
		for (XMFLOAT4X4& bind : armature.inverseBindMatrices)
		{
			// F * M * F negates the elements where exactly one of the row and the column is Z:
			bind._13 *= -1.f;
			bind._23 *= -1.f;
			bind._43 *= -1.f;
			bind._31 *= -1.f;
			bind._32 *= -1.f;
			bind._34 *= -1.f;
		}
	}

	// Flip animation data for translation and rotation
	for (size_t i = 0; i < scene.animations.GetCount(); ++i)
	{
		auto& animation = scene.animations[i];

		for (auto& channel : animation.channels)
		{
			auto data = scene.animation_datas.GetComponent(animation.samplers[channel.samplerIndex].data);

			if (channel.path == wi::scene::AnimationComponent::AnimationChannel::Path::TRANSLATION)
			{
				for (size_t k = 0; k < data->keyframe_data.size() / 3; ++k)
				{
					data->keyframe_data[k * 3 + 2] *= -1.f;
				}
			}
			if (channel.path == wi::scene::AnimationComponent::AnimationChannel::Path::ROTATION)
			{
				for (size_t k = 0; k < data->keyframe_data.size() / 4; ++k)
				{
					data->keyframe_data[k * 4] *= -1.f;
					data->keyframe_data[k * 4 + 1] *= -1.f;
				}
			}
		}
	}
}
