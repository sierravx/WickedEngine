#pragma once
#include "wiScene.h"
#include "wiUnorderedSet.h"

// The animation data that FlipZAxisTransforms() split, which UndoFlipZAxisSplits() puts back
struct FlipZAxisSplits
{
	struct Channel
	{
		wi::ecs::Entity animation = wi::ecs::INVALID_ENTITY;
		size_t channel = 0;
		int samplerIndex = -1; // the original sampler of the channel
	};
	wi::vector<Channel> channels;
	struct Samplers
	{
		wi::ecs::Entity animation = wi::ecs::INVALID_ENTITY;
		size_t count = 0; // the original sampler count of the animation
	};
	wi::vector<Samplers> samplers;
	wi::vector<wi::ecs::Entity> datas; // the copied animation datas
};

// Mirrors the transforms, inverse bind matrices and animation keys of a scene along the Z axis, shared by the model importers
//	to convert between the right handed space of the model files and the left handed space of the engine (the mesh data is flipped by the importers).
//	The transforms are mirrored in local space: translation Z, rotation X and rotation Y are negated and the scale is unchanged.
//	This needs no matrix decomposition, so it is exact for any hierarchy, even with non-uniform scaling and shear.
//	The world matrices follow from the mirrored local transforms on the next scene update.
//	The inverse bind matrices are mirrored as F * M * F, with F = scale(1, 1, -1), which is exact for any matrix.
//	rootEntity is not mirrored (it can be INVALID_ENTITY to mirror every transform). The world matrices of the other
//	transforms are the mirror of the original ones if the root's world matrix commutes with the mirror, like the identity
//	or a half turn about the Y axis (this is asserted for the root's local matrix in debug builds).
//	The matrices of retargeted animations are mirrored so that they are evaluated in the mirrored space.
//	Animation data used by channels that are mirrored differently is split first, so that each one can be mirrored.
//	Calling it a second time restores the original state, except that such data stays split (the animation plays the same).
//	The splitting is recorded in splits (if given), and UndoFlipZAxisSplits() can undo it after flipping back.
inline void FlipZAxisTransforms(wi::scene::Scene& scene, wi::ecs::Entity rootEntity, FlipZAxisSplits* splits = nullptr)
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

	using Path = wi::scene::AnimationComponent::AnimationChannel::Path;

	// Retargeted channels are evaluated as dstRelativeMatrix * source local transform * srcRelativeParentMatrix (see Scene::RunAnimationUpdateSystem()).
	//	With the source in this scene, its local transform and keys are mirrored too, so the matrices are mirrored like the others (F * M * F).
	//	With the source in an other scene, which is not mirrored here, the result is mirrored by F * dstRelativeMatrix and srcRelativeParentMatrix * F.
	for (size_t i = 0; i < scene.animations.GetCount(); ++i)
	{
		auto& animation = scene.animations[i];
		wi::vector<uint8_t> mirrored_retargets(animation.retargets.size(), 0); // 1: source in this scene, 2: in an other scene
		for (const auto& channel : animation.channels)
		{
			if (channel.retargetIndex < 0 || channel.retargetIndex >= (int)animation.retargets.size())
				continue;
			if (channel.samplerIndex < 0 || channel.samplerIndex >= (int)animation.samplers.size())
				continue;
			const void* source_scene = animation.samplers[channel.samplerIndex].scene;
			const uint8_t source_kind = (source_scene == nullptr || source_scene == &scene) ? 1 : 2;
			if (mirrored_retargets[channel.retargetIndex] != 0)
			{
				// Scene::RetargetAnimation() gives each channel its own retarget, a shared one must have one source scene:
				assert(mirrored_retargets[channel.retargetIndex] == source_kind);
				continue;
			}
			mirrored_retargets[channel.retargetIndex] = source_kind;
			auto& retarget = animation.retargets[channel.retargetIndex];
			if (source_kind == 1)
			{
				mirror(retarget.dstRelativeMatrix);
				mirror(retarget.srcRelativeParentMatrix);
			}
			else
			{
				// F * M negates the Z row, M * F negates the Z column:
				retarget.dstRelativeMatrix._31 *= -1.f;
				retarget.dstRelativeMatrix._32 *= -1.f;
				retarget.dstRelativeMatrix._33 *= -1.f;
				retarget.dstRelativeMatrix._34 *= -1.f;
				retarget.srcRelativeParentMatrix._13 *= -1.f;
				retarget.srcRelativeParentMatrix._23 *= -1.f;
				retarget.srcRelativeParentMatrix._33 *= -1.f;
				retarget.srcRelativeParentMatrix._43 *= -1.f;
			}
		}
	}

	// How the keys of a path are mirrored: 0: not at all, 1: like a translation, 2: like a rotation
	auto mirroring = [](Path path) {
		return path == Path::TRANSLATION ? 1 : (path == Path::ROTATION ? 2 : 0);
	};
	// The data of a channel if it is in this scene (it can be missing, or be in an other scene for an animation retargeted without baking):
	auto channel_data = [&](const wi::scene::AnimationComponent& animation, const wi::scene::AnimationComponent::AnimationChannel& channel) {
		if (channel.samplerIndex < 0 || channel.samplerIndex >= (int)animation.samplers.size())
			return wi::ecs::INVALID_ENTITY;
		const wi::ecs::Entity dataEntity = animation.samplers[channel.samplerIndex].data;
		return scene.animation_datas.Contains(dataEntity) ? dataEntity : wi::ecs::INVALID_ENTITY;
	};

	// Data used by channels that are mirrored differently (like a glTF sampler used by a translation and a scale channel)
	//	can't be mirrored in place for all of them. The data stays with its translation channels (or rotation channels if there
	//	are none), and the other channels get a copy of it with a sampler of their own. They stay split after flipping back,
	//	the animation plays the same.
	wi::unordered_map<wi::ecs::Entity, uint32_t> data_mirrorings; // data -> bit of each mirroring that uses it
	for (size_t i = 0; i < scene.animations.GetCount(); ++i)
	{
		const auto& animation = scene.animations[i];
		for (const auto& channel : animation.channels)
		{
			const wi::ecs::Entity dataEntity = channel_data(animation, channel);
			if (dataEntity != wi::ecs::INVALID_ENTITY)
			{
				data_mirrorings[dataEntity] |= 1u << mirroring(channel.path);
			}
		}
	}
	for (size_t i = 0; i < scene.animations.GetCount(); ++i)
	{
		auto& animation = scene.animations[i];
		const wi::ecs::Entity animationEntity = scene.animations.GetEntity(i);
		const size_t sampler_count = animation.samplers.size();
		wi::unordered_map<uint64_t, int> split_samplers; // sampler index and mirroring -> sampler of the copied data
		for (size_t channel_index = 0; channel_index < animation.channels.size(); ++channel_index)
		{
			auto& channel = animation.channels[channel_index];
			const wi::ecs::Entity dataEntity = channel_data(animation, channel);
			if (dataEntity == wi::ecs::INVALID_ENTITY)
				continue;
			const uint32_t used = data_mirrorings[dataEntity];
			if ((used & (used - 1)) == 0)
				continue; // used by one kind of mirroring
			const int kind = mirroring(channel.path);
			const int owner = (used & (1u << 1)) != 0 ? 1 : 2;
			if (kind == owner)
				continue;

			const uint64_t key = (uint64_t(channel.samplerIndex) << 8) | uint64_t(kind);
			auto split = split_samplers.find(key);
			if (split == split_samplers.end())
			{
				const wi::scene::AnimationDataComponent copy = *scene.animation_datas.GetComponent(dataEntity);
				const wi::ecs::Entity copyEntity = wi::ecs::CreateEntity();
				scene.animation_datas.Create(copyEntity) = copy;
				scene.Component_Attach(copyEntity, animationEntity);

				if (splits != nullptr)
				{
					if (split_samplers.empty())
					{
						splits->samplers.push_back({ animationEntity, sampler_count });
					}
					splits->datas.push_back(copyEntity);
				}
				wi::scene::AnimationComponent::AnimationSampler sampler = animation.samplers[channel.samplerIndex];
				sampler.data = copyEntity;
				animation.samplers.push_back(sampler);
				split = split_samplers.insert({ key, int(animation.samplers.size() - 1) }).first;
			}
			if (splits != nullptr)
			{
				splits->channels.push_back({ animationEntity, channel_index, channel.samplerIndex });
			}
			channel.samplerIndex = split->second;
		}
	}

	// Flip animation data for translation and rotation, once for each data (several channels can use the same data)
	wi::unordered_set<wi::ecs::Entity> flipped_datas;
	for (size_t i = 0; i < scene.animations.GetCount(); ++i)
	{
		auto& animation = scene.animations[i];

		for (auto& channel : animation.channels)
		{
			const bool translation = channel.path == Path::TRANSLATION;
			const bool rotation = channel.path == Path::ROTATION;
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

// Undoes the splitting of animation data recorded by FlipZAxisTransforms(), after flipping back.
inline void UndoFlipZAxisSplits(wi::scene::Scene& scene, FlipZAxisSplits& splits)
{
	for (const FlipZAxisSplits::Channel& split : splits.channels)
	{
		wi::scene::AnimationComponent* animation = scene.animations.GetComponent(split.animation);
		if (animation != nullptr && split.channel < animation->channels.size())
		{
			animation->channels[split.channel].samplerIndex = split.samplerIndex;
		}
	}
	for (const FlipZAxisSplits::Samplers& split : splits.samplers)
	{
		wi::scene::AnimationComponent* animation = scene.animations.GetComponent(split.animation);
		if (animation != nullptr && split.count <= animation->samplers.size())
		{
			animation->samplers.resize(split.count);
		}
	}
	for (wi::ecs::Entity entity : splits.datas)
	{
		scene.Entity_Remove(entity);
	}
	splits = {};
}
