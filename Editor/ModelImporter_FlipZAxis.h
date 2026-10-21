#pragma once
#include "wiScene.h"
#include "wiUnorderedSet.h"

// Mirrors animation keys along the Z axis like FlipZAxisTransforms() mirrors the transforms: translation Z, rotation X and
//	rotation Y are negated, the other paths are unchanged. The tangents of cubic spline keys are mirrored the same way.
inline void MirrorAnimationKeys(wi::scene::AnimationComponent::AnimationChannel::Path path, wi::vector<float>& keys)
{
	using Path = wi::scene::AnimationComponent::AnimationChannel::Path;
	if (path == Path::TRANSLATION)
	{
		for (size_t i = 2; i < keys.size(); i += 3)
		{
			keys[i] *= -1.f;
		}
	}
	else if (path == Path::ROTATION)
	{
		for (size_t i = 0; i + 1 < keys.size(); i += 4)
		{
			keys[i] *= -1.f;
			keys[i + 1] *= -1.f;
		}
	}
}

// The animation data and retargets that FlipZAxisTransforms() split, which UndoFlipZAxisSplits() puts back
struct FlipZAxisSplits
{
	struct Channel
	{
		wi::ecs::Entity animation = wi::ecs::INVALID_ENTITY;
		size_t channel = 0;
		int samplerIndex = -1; // the sampler and retarget of the channel before it was changed
		int retargetIndex = -1;
	};
	wi::vector<Channel> channels; // in the order of the changes
	struct Count
	{
		wi::ecs::Entity animation = wi::ecs::INVALID_ENTITY;
		size_t count = 0;
	};
	wi::vector<Count> samplers; // the original sampler counts of the animations
	wi::vector<Count> retargets; // the original retarget counts of the animations
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
//	Animation data (and retargets) used by channels that are mirrored differently are split first, so that each can be mirrored.
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
	//	A retarget used by channels with sources in both kinds of scenes (Scene::RetargetAnimation() doesn't make such) is copied
	//	for the channels of the other kind, so that each can be mirrored.
	for (size_t i = 0; i < scene.animations.GetCount(); ++i)
	{
		auto& animation = scene.animations[i];
		const wi::ecs::Entity animationEntity = scene.animations.GetEntity(i);
		const size_t retarget_count = animation.retargets.size();
		wi::vector<uint8_t> retarget_sources(retarget_count, 0); // 1: source in this scene, 2: in an other scene
		wi::vector<int> retarget_copies(retarget_count, -1);
		for (size_t channel_index = 0; channel_index < animation.channels.size(); ++channel_index)
		{
			auto& channel = animation.channels[channel_index];
			if (channel.retargetIndex < 0 || channel.retargetIndex >= (int)retarget_count)
				continue;
			if (channel.samplerIndex < 0 || channel.samplerIndex >= (int)animation.samplers.size())
				continue;
			const void* source_scene = animation.samplers[channel.samplerIndex].scene;
			const uint8_t source = (source_scene == nullptr || source_scene == &scene) ? 1 : 2;
			if (retarget_sources[channel.retargetIndex] == 0)
			{
				retarget_sources[channel.retargetIndex] = source;
			}
			if (retarget_sources[channel.retargetIndex] == source)
				continue;

			if (retarget_copies[channel.retargetIndex] < 0)
			{
				if (splits != nullptr && animation.retargets.size() == retarget_count)
				{
					splits->retargets.push_back({ animationEntity, retarget_count });
				}
				const wi::scene::AnimationComponent::RetargetSourceData copy = animation.retargets[channel.retargetIndex];
				animation.retargets.push_back(copy);
				retarget_sources.push_back(source);
				retarget_copies[channel.retargetIndex] = int(animation.retargets.size() - 1);
			}
			if (splits != nullptr)
			{
				splits->channels.push_back({ animationEntity, channel_index, channel.samplerIndex, channel.retargetIndex });
			}
			channel.retargetIndex = retarget_copies[channel.retargetIndex];
		}

		for (size_t r = 0; r < animation.retargets.size(); ++r)
		{
			auto& retarget = animation.retargets[r];
			if (retarget_sources[r] == 1)
			{
				mirror(retarget.dstRelativeMatrix);
				mirror(retarget.srcRelativeParentMatrix);
			}
			else if (retarget_sources[r] == 2)
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
	// The data of a channel if it is in this scene (it can be missing, or be in an other scene for an animation retargeted without baking,
	//	where the same entity can also have data in this scene):
	auto channel_data = [&](const wi::scene::AnimationComponent& animation, const wi::scene::AnimationComponent::AnimationChannel& channel) {
		if (channel.samplerIndex < 0 || channel.samplerIndex >= (int)animation.samplers.size())
			return wi::ecs::INVALID_ENTITY;
		const wi::scene::AnimationComponent::AnimationSampler& sampler = animation.samplers[channel.samplerIndex];
		if (sampler.scene != nullptr && sampler.scene != &scene)
			return wi::ecs::INVALID_ENTITY;
		return scene.animation_datas.Contains(sampler.data) ? sampler.data : wi::ecs::INVALID_ENTITY;
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
				splits->channels.push_back({ animationEntity, channel_index, channel.samplerIndex, channel.retargetIndex });
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
			const wi::ecs::Entity dataEntity = channel_data(animation, channel);
			if (dataEntity == wi::ecs::INVALID_ENTITY || !flipped_datas.insert(dataEntity).second)
				continue;
			auto data = scene.animation_datas.GetComponent(dataEntity);

			MirrorAnimationKeys(channel.path, data->keyframe_data);
		}
	}
}

// Undoes the splitting of animation data and retargets recorded by FlipZAxisTransforms(), after flipping back.
inline void UndoFlipZAxisSplits(wi::scene::Scene& scene, FlipZAxisSplits& splits)
{
	for (auto split = splits.channels.rbegin(); split != splits.channels.rend(); ++split)
	{
		wi::scene::AnimationComponent* animation = scene.animations.GetComponent(split->animation);
		if (animation != nullptr && split->channel < animation->channels.size())
		{
			animation->channels[split->channel].samplerIndex = split->samplerIndex;
			animation->channels[split->channel].retargetIndex = split->retargetIndex;
		}
	}
	for (const FlipZAxisSplits::Count& split : splits.samplers)
	{
		wi::scene::AnimationComponent* animation = scene.animations.GetComponent(split.animation);
		if (animation != nullptr && split.count <= animation->samplers.size())
		{
			animation->samplers.resize(split.count);
		}
	}
	for (const FlipZAxisSplits::Count& split : splits.retargets)
	{
		wi::scene::AnimationComponent* animation = scene.animations.GetComponent(split.animation);
		if (animation != nullptr && split.count <= animation->retargets.size())
		{
			animation->retargets.resize(split.count);
		}
	}
	for (wi::ecs::Entity entity : splits.datas)
	{
		// The copies only have these components:
		scene.animation_datas.Remove(entity);
		scene.hierarchy.Remove(entity);
	}
	splits = {};
}
