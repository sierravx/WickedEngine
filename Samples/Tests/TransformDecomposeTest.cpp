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

	void TestLight()
	{
		Scene scene;
		Entity parent = CreateNode(scene, XMFLOAT3(2, 1, 1), XMFLOAT3(0, 10, 0), XMFLOAT3(0, 0, 0));
		Entity entity = CreateNode(scene, XMFLOAT3(1, 1, 1), XMFLOAT3(0, 45, 0), XMFLOAT3(0, 3, 0));
		scene.lights.Create(entity).SetType(LightComponent::SPOT);
		scene.Component_Attach(entity, parent, true);
		UpdateWorldMatrices(scene);

		wi::jobsystem::context ctx;
		scene.RunLightUpdateSystem(ctx);
		wi::jobsystem::Wait(ctx);
		Check(IsUnitQuaternion(scene.lights.GetComponent(entity)->rotation), "the rotation of a light under non-uniform scale is not a unit quaternion");
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
