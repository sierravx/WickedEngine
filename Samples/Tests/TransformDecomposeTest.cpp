#include "WickedEngine.h"

#include <cmath>
#include <iostream>
#include <string>

// Regression tests for decomposing matrices that XMMatrixDecompose() refuses (sheared world matrices under non-uniform scaling),
//	and for a non-finite bone in the armature skinning rebase.

using namespace wi::ecs;
using namespace wi::scene;

namespace
{
	int failures = 0;

	void Check(bool condition, const std::string& message)
	{
		if (!condition)
		{
			std::cerr << "TransformDecompose test failed: " << message << '\n';
			failures++;
		}
	}

	bool IsFinite(XMVECTOR V)
	{
		return !XMVector4IsNaN(V) && !XMVector4IsInfinite(V);
	}
	bool IsUnitQuaternion(XMVECTOR Q)
	{
		return IsFinite(Q) && std::abs(XMVectorGetX(XMVector4Length(Q)) - 1) < 1e-4f;
	}
	bool IsUnitQuaternion(const XMFLOAT4& q)
	{
		return IsUnitQuaternion(XMLoadFloat4(&q));
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
	XMMATRIX Compose(XMVECTOR S, XMVECTOR R, XMVECTOR T)
	{
		return XMMatrixScalingFromVector(S) * XMMatrixRotationQuaternion(R) * XMMatrixTranslationFromVector(T);
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

	// The rotated child of a non-uniformly scaled parent: its world matrix has shear (the first row is the longest axis)
	const XMMATRIX sheared = XMMatrixRotationY(XMConvertToRadians(30)) * XMMatrixScaling(2, 1, 1) * XMMatrixRotationY(XMConvertToRadians(10)) * XMMatrixTranslation(1, 2, 3);

	void TestDecomposableMatrices()
	{
		wi::random::RNG rng(42);
		for (int i = 0; i < 1000; ++i)
		{
			XMVECTOR S = XMVectorSet(rng.next_float(0.1f, 3), rng.next_float(0.1f, 3), rng.next_float(0.1f, 3), 0);
			if (i % 4 == 0)
			{
				S = XMVectorMultiply(S, XMVectorSet(-1, 1, 1, 0)); // mirrored
			}
			XMVECTOR R = XMQuaternionNormalize(XMVectorSet(rng.next_float(-1, 1), rng.next_float(-1, 1), rng.next_float(-1, 1), rng.next_float(-1, 1)));
			XMVECTOR T = XMVectorSet(rng.next_float(-10, 10), rng.next_float(-10, 10), rng.next_float(-10, 10), 1);
			XMMATRIX M = Compose(S, R, T);

			XMVECTOR expectedS, expectedR, expectedT;
			if (!XMMatrixDecompose(&expectedS, &expectedR, &expectedT, M))
				continue;
			XMVECTOR outS, outR, outT;
			wi::math::DecomposeTransform(&outS, &outR, &outT, M);
			if (!XMVector4Equal(outS, expectedS) || !XMVector4Equal(outR, expectedR) || !XMVector4Equal(outT, expectedT))
			{
				Check(false, "a matrix that XMMatrixDecompose() accepts did not decompose to the same result");
				return;
			}
		}
	}

	void TestShearedMatrices()
	{
		XMVECTOR S, R, T;
		Check(!XMMatrixDecompose(&S, &R, &T, sheared), "precondition: XMMatrixDecompose() should refuse the sheared matrix");

		for (bool mirrored : { false, true })
		{
			const XMMATRIX M = mirrored ? XMMatrixScaling(1, 1, -1) * sheared : sheared;
			wi::math::DecomposeTransform(&S, &R, &T, M);
			Check(IsUnitQuaternion(R), std::string("rotation of a sheared ") + (mirrored ? "mirrored " : "") + "matrix is not a unit quaternion");
			Check(IsFinite(S), "scale of a sheared matrix is not finite");
			Check(XMVector4Equal(T, M.r[3]), "translation of a sheared matrix is not exact");

			// Only the shear is dropped: the longest axis is kept exactly, the length of every axis and the handedness are kept,
			//	and the other axes still point to the same side as the original ones
			const XMMATRIX rebuilt = Compose(S, R, T);
			Check(XMVectorGetX(XMVector3Length(rebuilt.r[0] - M.r[0])) < 1e-4f, "the longest axis of a sheared matrix changed");
			Check(XMVectorGetX(XMMatrixDeterminant(rebuilt)) * XMVectorGetX(XMMatrixDeterminant(M)) > 0, "the handedness of a sheared matrix changed");
			for (int i = 0; i < 3; ++i)
			{
				Check(std::abs(XMVectorGetX(XMVector3Length(rebuilt.r[i]) - XMVector3Length(M.r[i]))) < 1e-4f, "the length of an axis of a sheared matrix changed");
				Check(XMVectorGetX(XMVector3Dot(rebuilt.r[i], M.r[i])) > 0, "an axis of a sheared matrix was turned away from the original");
			}
		}

		// Random sheared matrices, including mirrored ones
		wi::random::RNG rng(7);
		for (int i = 0; i < 1000; ++i)
		{
			XMMATRIX M = XMMatrixIdentity();
			for (int k = 0; k < 3; ++k)
			{
				M.r[k] = XMVectorSet(rng.next_float(-2, 2), rng.next_float(-2, 2), rng.next_float(-2, 2), 0);
			}
			if (std::abs(XMVectorGetX(XMMatrixDeterminant(M))) < 0.01f)
				continue;
			wi::math::DecomposeTransform(&S, &R, &T, M);
			if (!IsUnitQuaternion(R) || XMVectorGetX(XMMatrixDeterminant(Compose(S, R, T))) * XMVectorGetX(XMMatrixDeterminant(M)) <= 0)
			{
				Check(false, "a random sheared matrix was not decomposed to a unit rotation with the same handedness");
				return;
			}
		}
	}

	void TestDegenerateMatrices()
	{
		XMVECTOR S, R, T;
		XMMATRIX parallel = XMMatrixIdentity();
		parallel.r[0] = XMVectorSet(1, 1, 0, 0);
		parallel.r[1] = XMVectorSet(2, 2, 0, 0);
		parallel.r[2] = XMVectorSet(0, 0, 1, 0);
		wi::math::DecomposeTransform(&S, &R, &T, parallel);
		Check(IsUnitQuaternion(R), "rotation of a matrix with parallel axes is not a unit quaternion");

		XMMATRIX flat = XMMatrixRotationY(XMConvertToRadians(30)) * XMMatrixScaling(2, 1, 0);
		wi::math::DecomposeTransform(&S, &R, &T, flat);
		Check(IsUnitQuaternion(R), "rotation of a flattened matrix is not a unit quaternion");

		wi::math::DecomposeTransform(&S, &R, &T, XMMatrixScaling(0, 0, 0));
		Check(IsUnitQuaternion(R), "rotation of a zero matrix is not a unit quaternion");
	}

	void TestTransformComponent()
	{
		// Detach (ApplyTransform of the sheared world matrix)
		{
			Scene scene;
			Entity parent = CreateNode(scene, XMFLOAT3(2, 1, 1), XMFLOAT3(0, 10, 0), XMFLOAT3(0, 0, 0));
			Entity child = CreateNode(scene, XMFLOAT3(1, 1, 1), XMFLOAT3(0, 45, 0), XMFLOAT3(0, 0, 1));
			scene.Component_Attach(child, parent, true);
			UpdateWorldMatrices(scene);
			TransformComponent& transform = *scene.transforms.GetComponent(child);
			const XMFLOAT3 position = transform.GetPosition();
			Check(IsUnitQuaternion(transform.GetRotationV()), "GetRotationV() of a sheared world matrix is not a unit quaternion");

			scene.Component_Detach(child);
			Check(IsUnitQuaternion(transform.rotation_local), "Component_Detach() across non-uniform scale left a non-unit rotation");
			transform.UpdateTransform();
			const XMFLOAT3 position_after = transform.GetPosition();
			Check(position.x == position_after.x && position.y == position_after.y && position.z == position_after.z, "Component_Detach() across non-uniform scale moved the child");
		}

		// Attach of a world space child (MatrixTransform with the inverse of the parent's world matrix)
		{
			Scene scene;
			Entity parent = CreateNode(scene, XMFLOAT3(2, 1, 1), XMFLOAT3(0, 10, 0), XMFLOAT3(0, 0, 0));
			Entity child = CreateNode(scene, XMFLOAT3(1, 1, 1), XMFLOAT3(0, 45, 0), XMFLOAT3(0, 0, 1));
			scene.Component_Attach(child, parent);
			Check(IsUnitQuaternion(scene.transforms.GetComponent(child)->rotation_local), "Component_Attach() across non-uniform scale left a non-unit rotation");
		}

		// With uniform scale nothing has shear, a detach and attach round trip keeps the world matrix
		{
			Scene scene;
			Entity parent = CreateNode(scene, XMFLOAT3(2, 2, 2), XMFLOAT3(0, 10, 0), XMFLOAT3(1, 0, 0));
			Entity child = CreateNode(scene, XMFLOAT3(1, 1, 1), XMFLOAT3(20, 45, 0), XMFLOAT3(0, 0, 1));
			scene.Component_Attach(child, parent, true);
			UpdateWorldMatrices(scene);
			const XMMATRIX world = XMLoadFloat4x4(&scene.transforms.GetComponent(child)->world);
			scene.Component_Detach(child);
			scene.Component_Attach(child, parent);
			UpdateWorldMatrices(scene);
			Check(MaxDifference(world, XMLoadFloat4x4(&scene.transforms.GetComponent(child)->world)) < 1e-4f, "detach and attach under uniform scale changed the world matrix");
		}

		// Lerp between sheared transforms
		{
			TransformComponent a, b, result;
			XMStoreFloat4x4(&a.world, sheared);
			XMStoreFloat4x4(&b.world, XMMatrixRotationX(XMConvertToRadians(30)) * sheared);
			result.Lerp(a, b, 0.5f);
			Check(IsUnitQuaternion(result.rotation_local), "Lerp() of sheared transforms is not a unit quaternion");
		}
	}

	void TestKeepAxis()
	{
		const XMVECTOR canonical[3] = { XMVectorSet(1, 0, 0, 0), XMVectorSet(0, 1, 0, 0), XMVectorSet(0, 0, 1, 0) };
		XMVECTOR S, R, T;

		// Random sheared and mirrored matrices: the kept axis points exactly along the matrix's axis
		wi::random::RNG rng(11);
		for (int i = 0; i < 1000; ++i)
		{
			XMMATRIX M = XMMatrixIdentity();
			for (int k = 0; k < 3; ++k)
			{
				M.r[k] = XMVectorSet(rng.next_float(-2, 2), rng.next_float(-2, 2), rng.next_float(-2, 2), 0);
			}
			M.r[3] = XMVectorSet(rng.next_float(-10, 10), rng.next_float(-10, 10), rng.next_float(-10, 10), 1);
			if (std::abs(XMVectorGetX(XMMatrixDeterminant(M))) < 0.01f)
				continue;
			for (int keepAxis = 0; keepAxis < 3; ++keepAxis)
			{
				wi::math::DecomposeTransform(&S, &R, &T, M, keepAxis);
				const XMMATRIX rebuilt = Compose(S, R, T);
				bool lengths_kept = true;
				for (int k = 0; k < 3; ++k)
				{
					lengths_kept &= std::abs(XMVectorGetX(XMVector3Length(rebuilt.r[k]) - XMVector3Length(M.r[k]))) < 1e-4f;
				}
				const bool ok =
					IsUnitQuaternion(R) &&
					XMVectorGetX(XMVector3Length(XMVector3Rotate(canonical[keepAxis], R) - XMVector3Normalize(M.r[keepAxis]))) < 1e-4f &&
					XMVectorGetX(XMMatrixDeterminant(rebuilt)) * XMVectorGetX(XMMatrixDeterminant(M)) > 0 &&
					lengths_kept &&
					XMVector4Equal(T, M.r[3]);
				if (!ok)
				{
					Check(false, "DecomposeTransform() with a kept axis did not keep the axis of a sheared or mirrored matrix");
					return;
				}
			}
		}

		// Without shear and mirroring, the result is the one of XMMatrixDecompose()
		const XMMATRIX M = XMMatrixScaling(2, 3, 4) * XMMatrixRotationRollPitchYaw(0.3f, 1.2f, -0.7f) * XMMatrixTranslation(1, 2, 3);
		XMVECTOR expectedS, expectedR, expectedT;
		XMMatrixDecompose(&expectedS, &expectedR, &expectedT, M);
		for (int keepAxis = 0; keepAxis < 3; ++keepAxis)
		{
			wi::math::DecomposeTransform(&S, &R, &T, M, keepAxis);
			Check(XMVector4Equal(S, expectedS) && XMVector4Equal(R, expectedR) && XMVector4Equal(T, expectedT), "DecomposeTransform() with a kept axis changed the result of a matrix without shear");
		}

		// A slight shear is accepted by XMMatrixDecompose(), but the kept axis must still be exact
		const XMMATRIX slightly_sheared = XMMatrixRotationRollPitchYaw(0.3f, 0.2f, XM_PIDIV4) * XMMatrixScaling(1.002f, 1, 1) * XMMatrixRotationY(0.17f);
		Check(XMMatrixDecompose(&S, &R, &T, slightly_sheared), "precondition: XMMatrixDecompose() should accept the slightly sheared matrix");
		for (int keepAxis = 0; keepAxis < 3; ++keepAxis)
		{
			wi::math::DecomposeTransform(&S, &R, &T, slightly_sheared, keepAxis);
			Check(XMVectorGetX(XMVector3Length(XMVector3Rotate(canonical[keepAxis], R) - XMVector3Normalize(slightly_sheared.r[keepAxis]))) < 1e-5f, "DecomposeTransform() with a kept axis did not keep the axis of a slightly sheared matrix");
		}

		// A kept axis of zero length has no direction to keep, the other axes still give the rotation
		const XMMATRIX flat = XMMatrixScaling(1, 1, 0) * XMMatrixRotationRollPitchYaw(0.4f, 0.7f, 0.2f);
		wi::math::DecomposeTransform(&S, &R, &T, flat, 2);
		Check(IsUnitQuaternion(R) && MaxDifference(Compose(S, R, T), flat) < 1e-4f, "DecomposeTransform() with a kept axis of zero length did not decompose the matrix");

		// A mirrored matrix turning slowly: the rotation must not jump (the mirroring must stay on the same axis)
		for (int keepAxis = 0; keepAxis < 3; ++keepAxis)
		{
			XMVECTOR previous = XMQuaternionIdentity();
			bool continuous = true;
			for (int frame = 0; frame < 2000 && continuous; ++frame)
			{
				const XMMATRIX mirrored = XMMatrixRotationRollPitchYaw(0.001f * frame, 0.0007f * frame, 0.0003f * frame) * XMMatrixScaling(-1, 1, 1);
				wi::math::DecomposeTransform(&S, &R, &T, mirrored, keepAxis);
				continuous = frame == 0 || std::abs(XMVectorGetX(XMQuaternionDot(R, previous))) > 0.999f;
				previous = R;
			}
			Check(continuous, "DecomposeTransform() with a kept axis jumped on a slowly turning mirrored matrix");
		}
	}

	void TestLight()
	{
		struct LightType
		{
			LightComponent::LightType type;
			const char* name;
			XMVECTOR axis; // the light direction in the light's local space
		};
		const LightType types[] = {
			{ LightComponent::DIRECTIONAL, "directional", XMVectorSet(0, 1, 0, 0) },
			{ LightComponent::SPOT, "spot", XMVectorSet(0, 1, 0, 0) },
			{ LightComponent::POINT, "point", XMVectorSet(1, 0, 0, 0) },
			{ LightComponent::RECTANGLE, "rectangle", XMVectorSet(0, 0, -1, 0) },
		};
		// Non-uniform scales stretching each axis, and mirrored ones
		const XMFLOAT3 parent_scales[] = { XMFLOAT3(2, 1, 1), XMFLOAT3(1, 2, 1), XMFLOAT3(1, 1, 2), XMFLOAT3(1.002f, 1, 1), XMFLOAT3(-1, 1, 1), XMFLOAT3(-1, 2, 1) };

		for (const LightType& type : types)
		{
			for (const XMFLOAT3& parent_scale : parent_scales)
			{
				bool ok = true;
				for (int pitch = 0; pitch < 360 && ok; pitch += 30)
				{
					for (int roll = 0; roll < 360 && ok; roll += 30)
					{
						Scene scene;
						Entity parent = CreateNode(scene, parent_scale, XMFLOAT3(0, 10, 0), XMFLOAT3(0, 0, 0));
						Entity entity = CreateNode(scene, XMFLOAT3(1, 1, 1), XMFLOAT3(float(pitch), 20, float(roll)), XMFLOAT3(0, 3, 0));
						scene.lights.Create(entity).SetType(type.type);
						scene.Component_Attach(entity, parent, true);
						UpdateWorldMatrices(scene);

						wi::jobsystem::context ctx;
						scene.RunLightUpdateSystem(ctx);
						wi::jobsystem::Wait(ctx);
						const LightComponent& light = *scene.lights.GetComponent(entity);

						// The rotation orients the shadow cameras and the light in the shaders, the direction is where it shines, they must agree:
						const XMVECTOR rotated = XMVector3Rotate(type.axis, XMLoadFloat4(&light.rotation));
						ok = IsUnitQuaternion(light.rotation) && XMVectorGetX(XMVector3Length(rotated - XMLoadFloat3(&light.direction))) < 1e-4f;
					}
				}
				Check(ok, std::string("the rotation of a ") + type.name + " light under a non-uniformly scaled or mirrored parent does not point to the light direction");
			}

			// A light without shear and mirroring gets the rotation of XMMatrixDecompose()
			Scene scene;
			Entity parent = CreateNode(scene, XMFLOAT3(2, 2, 2), XMFLOAT3(0, 10, 0), XMFLOAT3(0, 0, 0));
			Entity entity = CreateNode(scene, XMFLOAT3(1, 1, 1), XMFLOAT3(30, 20, 40), XMFLOAT3(0, 3, 0));
			scene.lights.Create(entity).SetType(type.type);
			scene.Component_Attach(entity, parent, true);
			UpdateWorldMatrices(scene);
			wi::jobsystem::context ctx;
			scene.RunLightUpdateSystem(ctx);
			wi::jobsystem::Wait(ctx);
			XMVECTOR S, R, T;
			XMMatrixDecompose(&S, &R, &T, XMLoadFloat4x4(&scene.transforms.GetComponent(entity)->world));
			Check(XMVector4Equal(XMLoadFloat4(&scene.lights.GetComponent(entity)->rotation), R), std::string("the rotation of a ") + type.name + " light without shear changed");
		}
	}

	void TestNonFiniteBone()
	{
		Scene scene;
		Entity armatureEntity = CreateEntity();
		scene.transforms.Create(armatureEntity);
		ArmatureComponent& armature = scene.armatures.Create(armatureEntity);
		const int boneCount = 4;
		const int badBone = 2;
		for (int i = 0; i < boneCount; ++i)
		{
			armature.boneCollection.push_back(CreateNode(scene, XMFLOAT3(1, 1, 1), XMFLOAT3(0, 0, 0), XMFLOAT3(0, float(i), 0)));
			XMFLOAT4X4 bind;
			XMStoreFloat4x4(&bind, XMMatrixTranslation(0, -float(i), 0.5f)); // the translation carries a NaN rotation into the skinning translation
			armature.inverseBindMatrices.push_back(bind);
		}
		TransformComponent& bone = *scene.transforms.GetComponent(armature.boneCollection[badBone]);
		bone.rotation_local = XMFLOAT4(NAN, NAN, NAN, NAN);
		bone.SetDirty();
		UpdateWorldMatrices(scene);

		wi::jobsystem::context ctx;
		scene.RunArmatureUpdateSystem(ctx);
		wi::jobsystem::Wait(ctx);

		const ArmatureComponent& result = *scene.armatures.GetComponent(armatureEntity);
		const XMFLOAT3 rebase = result.skinningRebase;
		Check(std::isfinite(rebase.x) && std::isfinite(rebase.y) && std::isfinite(rebase.z), "one non-finite bone made the skinning rebase non-finite");
		Check(std::abs(rebase.z - 0.5f) < 1e-5f, "the skinning rebase is not the mean of the finite bones");
		for (int i = 0; i < boneCount; ++i)
		{
			if (i == badBone)
				continue;
			const ShaderTransform& m = result.boneData[i];
			const bool finite =
				std::isfinite(m.mat0.x) && std::isfinite(m.mat0.y) && std::isfinite(m.mat0.z) && std::isfinite(m.mat0.w) &&
				std::isfinite(m.mat1.x) && std::isfinite(m.mat1.y) && std::isfinite(m.mat1.z) && std::isfinite(m.mat1.w) &&
				std::isfinite(m.mat2.x) && std::isfinite(m.mat2.y) && std::isfinite(m.mat2.z) && std::isfinite(m.mat2.w);
			Check(finite, "one non-finite bone made the skinning matrix of an other bone non-finite");
		}
	}
}

int main()
{
	wi::jobsystem::Initialize();

	TestDecomposableMatrices();
	TestShearedMatrices();
	TestDegenerateMatrices();
	TestKeepAxis();
	TestTransformComponent();
	TestLight();
	TestNonFiniteBone();

	wi::jobsystem::ShutDown();

	if (failures > 0)
	{
		std::cerr << failures << " check(s) failed\n";
		return 1;
	}
	std::cout << "TransformDecompose test passed\n";
	return 0;
}
