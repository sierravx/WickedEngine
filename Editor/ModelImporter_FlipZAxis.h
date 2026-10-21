#pragma once
#include "wiScene.h"
#include "wiUnorderedSet.h"

// Mirrors the transforms, inverse bind matrices and animation keys of a scene along the Z axis, shared by the model importers
//	to convert between the right handed space of the model files and the left handed space of the engine (the mesh data is flipped by the importers).
//	The transforms are mirrored in local space: translation Z, rotation X and rotation Y are negated and the scale is unchanged.
//	This needs no matrix decomposition, so it is exact for any hierarchy, even with non-uniform scaling and shear.
//	The world matrices follow from the mirrored local transforms on the next scene update.
//	The inverse bind matrices are mirrored as F * M * F, with F = scale(1, 1, -1), which is exact for any matrix.
//	rootEntity is not mirrored (it can be INVALID_ENTITY to mirror every transform). The world matrices of the other
//	transforms are the mirror of the original ones if the root's world matrix commutes with the mirror, like the identity
//	or a half turn about the Y axis (this is asserted for the root's local matrix in debug builds).
//	Calling it a second time restores the original state.
inline void FlipZAxisTransforms(wi::scene::Scene& scene, wi::ecs::Entity rootEntity)
{
	// F * M * F with F = scale(1, 1, -1) negates the elements where exactly one of the row and the column is Z:
	auto mirror = [](XMFLOAT4X4& m) {
		m._13 *= -1.f;
		m._23 *= -1.f;
		m._43 *= -1.f;
		m._31 *= -1.f;
		m._32 *= -1.f;
		m._34 *= -1.f;
	};

#ifndef NDEBUG
	const wi::scene::TransformComponent* root = scene.transforms.GetComponent(rootEntity);
	if (root != nullptr)
	{
		// The root is not mirrored, the world matrices are only mirrored correctly if it commutes with the mirror.
		//	Its local matrix is checked, which is its world matrix when it has no parent (the world matrices can be out of date here):
		assert(!scene.hierarchy.Contains(rootEntity));
		XMFLOAT4X4 root_matrix;
		XMStoreFloat4x4(&root_matrix, root->GetLocalMatrix());
		XMFLOAT4X4 root_matrix_mirrored = root_matrix;
		mirror(root_matrix_mirrored);
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				assert(std::abs(root_matrix.m[i][j] - root_matrix_mirrored.m[i][j]) < 0.0001f);
			}
		}
	}
#endif // NDEBUG

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
			mirror(bind);
		}
	}

	// Flip animation data for translation and rotation, once for each data (several channels can use the same data)
	wi::unordered_set<wi::ecs::Entity> flipped_datas;
	for (size_t i = 0; i < scene.animations.GetCount(); ++i)
	{
		auto& animation = scene.animations[i];

		for (auto& channel : animation.channels)
		{
			const bool translation = channel.path == wi::scene::AnimationComponent::AnimationChannel::Path::TRANSLATION;
			const bool rotation = channel.path == wi::scene::AnimationComponent::AnimationChannel::Path::ROTATION;
			if (!translation && !rotation)
				continue;
			if (channel.samplerIndex < 0 || channel.samplerIndex >= (int)animation.samplers.size())
				continue;
			// The data can be missing, or be in an other scene (animation retargeted without baking), which is not flipped here:
			const wi::ecs::Entity dataEntity = animation.samplers[channel.samplerIndex].data;
			auto data = scene.animation_datas.GetComponent(dataEntity);
			if (data == nullptr || !flipped_datas.insert(dataEntity).second)
				continue;

			if (translation)
			{
				for (size_t k = 0; k < data->keyframe_data.size() / 3; ++k)
				{
					data->keyframe_data[k * 3 + 2] *= -1.f;
				}
			}
			if (rotation)
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
