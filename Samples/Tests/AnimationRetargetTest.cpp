#include "WickedEngine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

// Regression test for animation retargeting: Scene::RetargetAnimation() with and without baking, and the retargeted channels
//	of Scene::RunAnimationUpdateSystem(), which both retarget with AnimationComponent::RetargetSourceData::RetargetValue().
//	The skeletons have non-uniform scaling and mirroring, so the retargeted matrices have shear and XMMatrixDecompose() refuses them.
//	The local transforms of the destination bones at several times, and the baked keys, are compared to the ones recorded before
//	the retargeting was shared, when the translation and scale were decomposed with XMMatrixDecompose() and the rotation with
//	wi::math::DecomposeTransform(). Run with --record to print the values of the current engine in the form of the tables below.

using namespace wi::ecs;
using namespace wi::scene;
using Path = AnimationComponent::AnimationChannel::Path;
using HumanoidBone = HumanoidComponent::HumanoidBone;

namespace
{
	// Recorded before the retargeting was shared. For each case, at each sample time, for each bone (hips, spine, chest):
	//	translation_local (3), rotation_local (4), scale_local (3)
	const float recorded_transforms[] = {
		0.0f, 0.899999976f, -1.1920929e-07f, -3.81573795e-09f, 0.0f, 0.216439605f, 0.976295948f, 1.86339343f, 1.19100308f, 0.985087931f,
		1.1920929e-07f, 0.600000322f, -5.96046448e-08f, 0.165611207f, 0.296137422f, -0.0522169396f, 0.939227879f, 1.27221549f, -1.6870203f, 0.919569671f,
		0.100000083f, 0.500000179f, 2.98023224e-08f, 2.89252835e-08f, -7.52057439e-09f, 0.258819044f, 0.965925872f, -1.25f, 0.999999881f, 0.800000072f,
		0.0833998621f, 0.960000038f, -0.105092764f, -0.101902254f, 0.176042393f, 0.290190339f, 0.935101509f, 1.89855206f, 1.13223636f, 0.988847375f,
		0.0407951474f, 0.620732069f, -0.0200316906f, 0.172600284f, 0.199154332f, -0.291218638f, 0.919640183f, 1.21187735f, -2.00264692f, 0.947215974f,
		0.0121637285f, 0.462976873f, 0.108285904f, -0.0517728217f, 0.286219686f, -0.870189011f, 0.397704393f, -1.25487161f, 0.890780807f, 1.0042156f,
		0.13899982f, 1.0f, -0.175154567f, -0.167388543f, 0.28917402f, 0.331731021f, 0.882220984f, 1.92486048f, 1.09369552f, 0.991696239f,
		0.067991823f, 0.634553194f, -0.0333861709f, 0.158991426f, 0.140955597f, -0.436933696f, 0.874037862f, 1.17364049f, -2.21559882f, 0.96733582f,
		-0.0463938117f, 0.43829459f, 0.180476367f, -0.0984513909f, 0.393668145f, -0.805031836f, 0.432731748f, -1.29295373f, 0.823579669f, 1.14049006f,
		0.0802312493f, 0.885714293f, 0.104450345f, -0.124301389f, -0.08028882f, 0.534615159f, 0.832039356f, 1.84046459f, 1.26741946f, 1.02802682f,
		0.00242346525f, 0.612136722f, -0.125163674f, 0.177363947f, 0.377218187f, -0.0145361535f, 0.908865809f, 1.29218256f, -1.90485823f, 0.815895736f,
		0.0725260973f, 0.510472178f, -0.000344723463f, -0.0246686544f, -0.0276763476f, 0.327433378f, 0.944146514f, -1.45331919f, 0.987011611f, 0.892353177f,
		0.0361547172f, 0.799999952f, 0.314154267f, -0.0765342042f, -0.351489097f, 0.62431103f, 0.693421721f, 1.80838227f, 1.40174675f, 1.05923998f,
		-0.0467528403f, 0.595324337f, -0.193996847f, 0.160156965f, 0.490330398f, 0.208473668f, 0.830941916f, 1.39133704f, -1.67933273f, 0.702607453f,
		0.161715984f, 0.564605355f, -0.135960594f, -0.415164858f, -0.106030442f, 0.179913133f, 0.885453165f, -1.66735518f, 1.11712384f, 0.706705809f,
		0.0f, 0.899999976f, -1.1920929e-07f, -3.81573795e-09f, 0.0f, 0.216439605f, 0.976295948f, 1.86339343f, 1.19100308f, 0.985087931f,
		1.1920929e-07f, 0.600000322f, -5.96046448e-08f, 0.165611207f, 0.296137422f, -0.0522169396f, 0.939227879f, 1.27221549f, -1.6870203f, 0.919569671f,
		0.100000083f, 0.500000179f, 2.98023224e-08f, 2.89252835e-08f, -7.52057439e-09f, 0.258819044f, 0.965925872f, -1.25f, 0.999999881f, 0.800000072f,
		0.0833998621f, 0.960000038f, -0.105092764f, -0.101902254f, 0.176042393f, 0.290190339f, 0.935101509f, 1.89855206f, 1.13223636f, 0.988847375f,
		0.0407951474f, 0.620732069f, -0.0200316906f, 0.172600284f, 0.199154332f, -0.291218638f, 0.919640183f, 1.21187735f, -2.00264692f, 0.947215974f,
		0.0121637285f, 0.462976873f, 0.108285904f, -0.0517728217f, 0.286219686f, -0.870189011f, 0.397704393f, -1.25487161f, 0.890780807f, 1.0042156f,
		0.13899982f, 1.0f, -0.175154567f, -0.167388543f, 0.28917402f, 0.331731021f, 0.882220984f, 1.92486048f, 1.09369552f, 0.991696239f,
		0.067991823f, 0.634553194f, -0.0333861709f, 0.158991426f, 0.140955597f, -0.436933696f, 0.874037862f, 1.17364049f, -2.21559882f, 0.96733582f,
		-0.0463938117f, 0.43829459f, 0.180476367f, -0.0984513909f, 0.393668145f, -0.805031836f, 0.432731748f, -1.29295373f, 0.823579669f, 1.14049006f,
		0.0802312493f, 0.885714293f, 0.104450345f, -0.124301389f, -0.08028882f, 0.534615159f, 0.832039356f, 1.84046459f, 1.26741946f, 1.02802682f,
		0.00242346525f, 0.612136722f, -0.125163674f, 0.177363947f, 0.377218187f, -0.0145361535f, 0.908865809f, 1.29218256f, -1.90485823f, 0.815895736f,
		0.0725260973f, 0.510472178f, -0.000344723463f, -0.0246686544f, -0.0276763476f, 0.327433378f, 0.944146514f, -1.45331919f, 0.987011611f, 0.892353177f,
		0.0361547172f, 0.799999952f, 0.314154267f, -0.0765342042f, -0.351489097f, 0.62431103f, 0.693421721f, 1.80838227f, 1.40174675f, 1.05923998f,
		-0.0467528403f, 0.595324337f, -0.193996847f, 0.160156965f, 0.490330398f, 0.208473668f, 0.830941916f, 1.39133704f, -1.67933273f, 0.702607453f,
		0.161715984f, 0.564605355f, -0.135960594f, -0.415164858f, -0.106030442f, 0.179913133f, 0.885453165f, -1.66735518f, 1.11712384f, 0.706705809f,
		0.0f, 0.899999976f, -1.1920929e-07f, -3.81573839e-09f, 0.0f, 0.21643962f, 0.976296008f, 1.86339343f, 1.19100308f, 0.985087931f,
		1.1920929e-07f, 0.600000322f, -5.96046448e-08f, 0.165611193f, 0.296137393f, -0.0522169881f, 0.939227879f, 1.27221549f, -1.6870203f, 0.919569671f,
		0.100000083f, 0.500000179f, 2.98023224e-08f, 2.89252835e-08f, -7.52057439e-09f, 0.258819044f, 0.965925872f, -1.25f, 0.999999881f, 0.800000072f,
		0.0833998919f, 0.959999979f, -0.105092794f, -0.101902217f, 0.176042333f, 0.290190279f, 0.93510139f, 1.90027368f, 1.13261855f, 0.989052892f,
		0.0407951437f, 0.620732069f, -0.0200317279f, 0.165187225f, 0.207700402f, -0.288677484f, 0.919912398f, 1.21307051f, -2.00416732f, 0.948229373f,
		0.0121637434f, 0.462976813f, 0.108285837f, -0.0733261332f, 0.293202311f, -0.462621242f, 0.833449066f, -1.27577221f, 0.894147754f, 1.00429404f,
		0.13899982f, 1.0f, -0.175154567f, -0.167388558f, 0.28917402f, 0.331731051f, 0.882221162f, 1.92486048f, 1.09369552f, 0.991696239f,
		0.067991823f, 0.634553194f, -0.0333861709f, 0.158991441f, 0.140955597f, -0.436933666f, 0.874037921f, 1.17364049f, -2.21559882f, 0.96733582f,
		-0.0463938117f, 0.43829459f, 0.180476367f, -0.0984513685f, 0.393668145f, -0.805031776f, 0.432731628f, -1.29295373f, 0.823579669f, 1.14049006f,
		0.0802312046f, 0.885714293f, 0.104450434f, -0.124301389f, -0.0802888572f, 0.534615219f, 0.832039535f, 1.85830152f, 1.26972473f, 1.03029263f,
		0.0024234578f, 0.612136722f, -0.125163674f, 0.171424121f, 0.364456177f, -0.075467065f, 0.912189662f, 1.29803848f, -1.90916109f, 0.81606251f,
		0.0725260526f, 0.510472178f, -0.000344708562f, -0.349352658f, 0.143599227f, -0.319849342f, 0.868923604f, -1.50689733f, 0.99131918f, 0.892613411f,
		0.0361547172f, 0.799999952f, 0.314154267f, -0.0765341818f, -0.351489037f, 0.624310911f, 0.693421721f, 1.80838227f, 1.40174675f, 1.05923998f,
		-0.0467528403f, 0.595324337f, -0.193996847f, 0.16015698f, 0.490330428f, 0.208473668f, 0.830941975f, 1.39133704f, -1.67933273f, 0.702607453f,
		0.161715984f, 0.564605355f, -0.135960594f, -0.415164739f, -0.106030479f, 0.179913118f, 0.885453105f, -1.66735518f, 1.11712384f, 0.706705809f,
		0.0f, 0.899999976f, -1.1920929e-07f, -3.81573839e-09f, 0.0f, 0.21643962f, 0.976296008f, 1.86339343f, 1.19100308f, 0.985087931f,
		1.1920929e-07f, 0.600000322f, -5.96046448e-08f, 0.165611193f, 0.296137393f, -0.0522169881f, 0.939227879f, 1.27221549f, -1.6870203f, 0.919569671f,
		0.100000083f, 0.500000179f, 2.98023224e-08f, 2.89252835e-08f, -7.52057439e-09f, 0.258819044f, 0.965925872f, -1.25f, 0.999999881f, 0.800000072f,
		0.0833998919f, 0.959999979f, -0.105092794f, -0.101902217f, 0.176042333f, 0.290190279f, 0.93510139f, 1.90027368f, 1.13261855f, 0.989052892f,
		0.0407951437f, 0.620732069f, -0.0200317279f, 0.165187225f, 0.207700402f, -0.288677484f, 0.919912398f, 1.21307051f, -2.00416732f, 0.948229373f,
		0.0121637434f, 0.462976813f, 0.108285837f, -0.0733261332f, 0.293202311f, -0.462621242f, 0.833449066f, -1.27577221f, 0.894147754f, 1.00429404f,
		0.13899982f, 1.0f, -0.175154567f, -0.167388558f, 0.28917402f, 0.331731051f, 0.882221162f, 1.92486048f, 1.09369552f, 0.991696239f,
		0.067991823f, 0.634553194f, -0.0333861709f, 0.158991441f, 0.140955597f, -0.436933666f, 0.874037921f, 1.17364049f, -2.21559882f, 0.96733582f,
		-0.0463938117f, 0.43829459f, 0.180476367f, -0.0984513685f, 0.393668145f, -0.805031776f, 0.432731628f, -1.29295373f, 0.823579669f, 1.14049006f,
		0.0802312046f, 0.885714293f, 0.104450434f, -0.124301389f, -0.0802888572f, 0.534615219f, 0.832039535f, 1.85830152f, 1.26972473f, 1.03029263f,
		0.0024234578f, 0.612136722f, -0.125163674f, 0.171424121f, 0.364456177f, -0.075467065f, 0.912189662f, 1.29803848f, -1.90916109f, 0.81606251f,
		0.0725260526f, 0.510472178f, -0.000344708562f, -0.349352658f, 0.143599227f, -0.319849342f, 0.868923604f, -1.50689733f, 0.99131918f, 0.892613411f,
		0.0361547172f, 0.799999952f, 0.314154267f, -0.0765341818f, -0.351489037f, 0.624310911f, 0.693421721f, 1.80838227f, 1.40174675f, 1.05923998f,
		-0.0467528403f, 0.595324337f, -0.193996847f, 0.16015698f, 0.490330428f, 0.208473668f, 0.830941975f, 1.39133704f, -1.67933273f, 0.702607453f,
		0.161715984f, 0.564605355f, -0.135960594f, -0.415164739f, -0.106030479f, 0.179913118f, 0.885453105f, -1.66735518f, 1.11712384f, 0.706705809f,
		0.0f, 1.0f, 0.0f, 0.0870727897f, -0.00380168017f, 0.0434534028f, 0.995246589f, 1.0f, 1.0f, 1.0f,
		0.0f, 0.5f, 0.0f, 0.0449434556f, 0.254887015f, 0.167731255f, 0.951251268f, 1.0f, 1.0f, 1.0f,
		0.0f, 0.400000006f, 0.100000001f, 0.130526185f, 0.0f, 0.0f, 0.991444767f, 1.0f, 1.0f, 1.0f,
		0.120000005f, 1.06000006f, -0.0600000024f, -0.0666121617f, 0.153619334f, 0.045076035f, 0.98485148f, 1.06000006f, 0.939999998f, 1.0f,
		0.0600000024f, 0.560000002f, 0.0f, 0.141858608f, 0.172624812f, -0.0202848911f, 0.974507749f, 0.939999998f, 1.17999995f, 1.0f,
		-0.120000005f, 0.340000004f, 0.0399999991f, 0.0137159117f, -0.126181051f, 0.179457307f, 0.975543618f, 1.0f, 0.879999995f, 1.24000001f,
		0.200000003f, 1.10000002f, -0.100000001f, -0.167731255f, 0.254887015f, 0.0449434556f, 0.951251268f, 1.10000002f, 0.899999976f, 1.0f,
		0.100000001f, 0.600000024f, 0.0f, 0.202664986f, 0.112505414f, -0.14564988f, 0.961798251f, 0.899999976f, 1.29999995f, 1.0f,
		-0.200000003f, 0.300000012f, 0.0f, -0.0650846511f, -0.20642215f, 0.29357788f, 0.931110084f, 1.0f, 0.800000012f, 1.39999998f,
		0.0285714567f, 0.985714316f, 0.128571391f, -0.160333782f, -0.0747929141f, 0.315021902f, 0.932448685f, 0.928571463f, 1.07142854f, 1.05714285f,
		0.0428571515f, 0.485714316f, -0.1142857f, 0.0446535796f, 0.334854364f, 0.177669153f, 0.92429018f, 1.07142854f, 1.12857139f, 0.828571439f,
		-0.0285714567f, 0.41428569f, 0.1142857f, 0.164898008f, 0.0213212166f, 0.0504100993f, 0.984790802f, 1.17142856f, 0.971428573f, 1.11428571f,
		-0.099999994f, 0.899999976f, 0.300000012f, -0.135349944f, -0.316410601f, 0.482319087f, 0.805563688f, 0.800000012f, 1.20000005f, 1.10000002f,
		0.0f, 0.400000006f, -0.200000003f, -0.0808046982f, 0.463826925f, 0.402198493f, 0.785220742f, 1.20000005f, 1.0f, 0.699999988f,
		0.099999994f, 0.5f, 0.200000003f, 0.321320653f, 0.191911131f, -0.139820546f, 0.916718841f, 1.29999995f, 1.10000002f, 0.899999976f,
		0.0f, 1.0f, 0.0f, -3.81573795e-09f, 0.0f, 0.216439605f, 0.976295948f, 1.86339343f, 1.19100308f, 0.985087931f,
		1.1920929e-07f, 0.600000322f, -5.96046448e-08f, 0.0449434556f, 0.254887015f, 0.167731255f, 0.951251268f, 1.27221549f, -1.6870203f, 0.919569671f,
		0.100000083f, 0.500000179f, 2.98023224e-08f, 2.89252835e-08f, -7.52057439e-09f, 0.258819044f, 0.965925872f, -1.25f, 0.999999881f, 0.800000072f,
		0.120000005f, 1.06000006f, -0.0600000024f, -0.101902254f, 0.176042393f, 0.290190339f, 0.935101509f, 1.89855206f, 1.13223636f, 0.988847375f,
		0.0407951474f, 0.620732069f, -0.0200316906f, 0.141858608f, 0.172624812f, -0.0202848911f, 0.974507749f, 1.21187735f, -2.00264692f, 0.947215974f,
		0.0121637285f, 0.462976873f, 0.108285904f, -0.0517728217f, 0.286219686f, -0.870189011f, 0.397704393f, -1.25487161f, 0.890780807f, 1.0042156f,
		0.200000003f, 1.10000002f, -0.100000001f, -0.167388543f, 0.28917402f, 0.331731021f, 0.882220984f, 1.92486048f, 1.09369552f, 0.991696239f,
		0.067991823f, 0.634553194f, -0.0333861709f, 0.202664986f, 0.112505414f, -0.14564988f, 0.961798251f, 1.17364049f, -2.21559882f, 0.96733582f,
		-0.0463938117f, 0.43829459f, 0.180476367f, -0.0984513909f, 0.393668145f, -0.805031836f, 0.432731748f, -1.29295373f, 0.823579669f, 1.14049006f,
		0.0285714567f, 0.985714316f, 0.128571391f, -0.124301389f, -0.08028882f, 0.534615159f, 0.832039356f, 1.84046459f, 1.26741946f, 1.02802682f,
		0.00242346525f, 0.612136722f, -0.125163674f, 0.0446535796f, 0.334854364f, 0.177669153f, 0.92429018f, 1.29218256f, -1.90485823f, 0.815895736f,
		0.0725260973f, 0.510472178f, -0.000344723463f, -0.0246686544f, -0.0276763476f, 0.327433378f, 0.944146514f, -1.45331919f, 0.987011611f, 0.892353177f,
		-0.099999994f, 0.899999976f, 0.300000012f, -0.0765342042f, -0.351489097f, 0.62431103f, 0.693421721f, 1.80838227f, 1.40174675f, 1.05923998f,
		-0.0467528403f, 0.595324337f, -0.193996847f, -0.0808046982f, 0.463826925f, 0.402198493f, 0.785220742f, 1.39133704f, -1.67933273f, 0.702607453f,
		0.161715984f, 0.564605355f, -0.135960594f, -0.415164858f, -0.106030442f, 0.179913133f, 0.885453165f, -1.66735518f, 1.11712384f, 0.706705809f,
	};
	// Recorded before the retargeting was shared. The keyframe_data of each channel of the baked animations, in channel order:
	const float recorded_baked_keys[] = {
		0.0f, 0.899999976f, -1.1920929e-07f, 0.13899982f, 1.0f, -0.175154567f, 0.0361547172f, 0.799999952f, 0.314154267f, -3.81573795e-09f,
		0.0f, 0.216439605f, 0.976295948f, -0.167388543f, 0.28917399f, 0.331731021f, 0.882221043f, -0.0765341893f, -0.351489067f, 0.62431097f,
		0.693421781f, 1.86339343f, 1.19100308f, 0.985087931f, 1.92486048f, 1.09369552f, 0.991696239f, 1.80838227f, 1.40174675f, 1.05923998f,
		1.1920929e-07f, 0.600000322f, -5.96046448e-08f, 0.067991823f, 0.634553194f, -0.0333861709f, -0.0467528403f, 0.595324337f, -0.193996847f, 0.165611193f,
		0.296137393f, -0.0522169881f, 0.939227879f, 0.158991426f, 0.140955582f, -0.436933637f, 0.874037862f, 0.16015698f, 0.490330428f, 0.208473668f,
		0.830941975f, 1.27221549f, -1.68702018f, 0.919569671f, 1.17364049f, -2.21559882f, 0.96733582f, 1.39133704f, -1.67933273f, 0.702607453f,
		0.100000083f, 0.500000179f, 2.98023224e-08f, -0.0463938117f, 0.43829459f, 0.180476367f, 0.161715984f, 0.564605355f, -0.135960594f, 2.89252835e-08f,
		-7.52057439e-09f, 0.258819044f, 0.965925872f, 0.0984513611f, -0.393668115f, 0.805031717f, -0.432731599f, -0.415164739f, -0.106030479f, 0.179913118f,
		0.885453105f, -1.25f, 0.999999881f, 0.800000072f, -1.29295373f, 0.823579669f, 1.14049006f, -1.66735518f, 1.11712384f, 0.706705809f,
		0.0f, 0.899999976f, -1.1920929e-07f, 0.13899982f, 1.0f, -0.175154567f, 0.0361547172f, 0.799999952f, 0.314154267f, -3.81573795e-09f,
		0.0f, 0.216439605f, 0.976295948f, -0.167388543f, 0.28917399f, 0.331731021f, 0.882221043f, -0.0765341893f, -0.351489067f, 0.62431097f,
		0.693421781f, 1.86339343f, 1.19100308f, 0.985087931f, 1.92486048f, 1.09369552f, 0.991696239f, 1.80838227f, 1.40174675f, 1.05923998f,
		1.1920929e-07f, 0.600000322f, -5.96046448e-08f, 0.067991823f, 0.634553194f, -0.0333861709f, -0.0467528403f, 0.595324337f, -0.193996847f, 0.165611193f,
		0.296137393f, -0.0522169881f, 0.939227879f, 0.158991426f, 0.140955582f, -0.436933637f, 0.874037862f, 0.16015698f, 0.490330428f, 0.208473668f,
		0.830941975f, 1.27221549f, -1.68702018f, 0.919569671f, 1.17364049f, -2.21559882f, 0.96733582f, 1.39133704f, -1.67933273f, 0.702607453f,
		0.100000083f, 0.500000179f, 2.98023224e-08f, -0.0463938117f, 0.43829459f, 0.180476367f, 0.161715984f, 0.564605355f, -0.135960594f, 2.89252835e-08f,
		-7.52057439e-09f, 0.258819044f, 0.965925872f, 0.0984513611f, -0.393668115f, 0.805031717f, -0.432731599f, -0.415164739f, -0.106030479f, 0.179913118f,
		0.885453105f, -1.25f, 0.999999881f, 0.800000072f, -1.29295373f, 0.823579669f, 1.14049006f, -1.66735518f, 1.11712384f, 0.706705809f,
	};
	// Recorded before the retargeting was shared, see TestFlatScale():
	const float recorded_flat_scale[] = {
		-2.06673241f, 0.0f, 1.32287586f,
	};

	int failures = 0;
	bool record = false;

	void Check(bool condition, const std::string& message)
	{
		if (!condition)
		{
			std::cerr << "AnimationRetarget test failed: " << message << '\n';
			failures++;
		}
	}

	bool Near(float a, float b, float tolerance)
	{
		return std::abs(a - b) <= tolerance * std::max(1.0f, std::abs(b));
	}
	// The same rotation, either of its two quaternions:
	bool SameRotation(const XMFLOAT4& a, const XMFLOAT4& b, float tolerance)
	{
		const XMVECTOR A = XMLoadFloat4(&a);
		const XMVECTOR B = XMLoadFloat4(&b);
		return std::min(XMVectorGetX(XMVector4Length(A - B)), XMVectorGetX(XMVector4Length(A + B))) <= tolerance;
	}

	// Prints values as the initializer of a recorded table:
	void Print(const char* name, const wi::vector<float>& values)
	{
		std::printf("\tconst float %s[] = {", name);
		for (size_t i = 0; i < values.size(); ++i)
		{
			char text[64];
			std::snprintf(text, sizeof(text), "%.9g", values[i]);
			std::string literal = text;
			if (literal.find_first_of(".e") == std::string::npos)
			{
				literal += ".0";
			}
			std::printf("%s%sf,", i % 10 == 0 ? "\n\t\t" : " ", literal.c_str());
		}
		std::printf("\n\t};\n");
	}

	// Compares values to a recorded table, or prints them with --record:
	void Compare(const char* name, const wi::vector<float>& values, const float* recorded, size_t recorded_count, std::string (*describe)(size_t))
	{
		if (record)
		{
			Print(name, values);
			return;
		}
		if (values.size() != recorded_count)
		{
			Check(false, std::string(name) + ": " + std::to_string(values.size()) + " values, " + std::to_string(recorded_count) + " were recorded");
			return;
		}
		for (size_t i = 0; i < values.size(); ++i)
		{
			if (!Near(values[i], recorded[i], 1e-4f))
			{
				Check(false, std::string(name) + ": " + describe(i) + " is " + std::to_string(values[i]) + ", " + std::to_string(recorded[i]) + " was recorded");
			}
		}
	}

	struct NodeDesc
	{
		XMFLOAT3 scale;
		XMFLOAT3 rotation; // XMQuaternionRotationRollPitchYaw() angles in degrees
		XMFLOAT3 translation;
	};

	XMVECTOR Rotation(const XMFLOAT3& degrees)
	{
		return XMQuaternionRotationRollPitchYaw(XMConvertToRadians(degrees.x), XMConvertToRadians(degrees.y), XMConvertToRadians(degrees.z));
	}

	Entity CreateNode(Scene& scene, const NodeDesc& desc)
	{
		Entity entity = CreateEntity();
		TransformComponent& transform = scene.transforms.Create(entity);
		transform.scale_local = desc.scale;
		XMStoreFloat4(&transform.rotation_local, Rotation(desc.rotation));
		transform.translation_local = desc.translation;
		transform.UpdateTransform();
		return entity;
	}

	// A humanoid skeleton of a root and three bones (hips, spine, chest), the humanoid component is on the root
	constexpr int bone_count = 3;
	struct Skeleton
	{
		Entity root = INVALID_ENTITY;
		Entity bones[bone_count] = {};
	};
	const char* bone_names[bone_count] = { "hips", "spine", "chest" };
	const HumanoidBone humanoid_bones[bone_count] = { HumanoidBone::Hips, HumanoidBone::Spine, HumanoidBone::Chest };

	Skeleton CreateSkeleton(Scene& scene, const NodeDesc (&nodes)[bone_count + 1])
	{
		Skeleton skeleton;
		skeleton.root = CreateNode(scene, nodes[0]);
		scene.humanoids.Create(skeleton.root);
		Entity parent = skeleton.root;
		for (int i = 0; i < bone_count; ++i)
		{
			skeleton.bones[i] = CreateNode(scene, nodes[i + 1]);
			scene.Component_Attach(skeleton.bones[i], parent, true);
			scene.humanoids.GetComponent(skeleton.root)->bones[size_t(humanoid_bones[i])] = skeleton.bones[i];
			parent = skeleton.bones[i];
		}
		return skeleton;
	}

	// The source has non-uniform scaling and a mirrored spine. (The mirrored nodes are scaled non-uniformly too: the decomposition
	//	of a mirrored matrix with axes of the same length negates the axis that comes out longest by rounding.)
	const NodeDesc source_nodes[bone_count + 1] = {
		{ XMFLOAT3(1, 1, 1), XMFLOAT3(0, 15, 0), XMFLOAT3(0.5f, 0, 0) },
		{ XMFLOAT3(1.2f, 0.8f, 1), XMFLOAT3(10, 0, 5), XMFLOAT3(0, 1, 0) },
		{ XMFLOAT3(1.1f, 0.9f, -1.3f), XMFLOAT3(0, 30, 20), XMFLOAT3(0, 0.5f, 0) },
		{ XMFLOAT3(1, 1, 1), XMFLOAT3(15, 0, 0), XMFLOAT3(0, 0.4f, 0.1f) },
	};
	// The destination has other non-uniform scaling and a mirrored chest:
	const NodeDesc destination_nodes[bone_count + 1] = {
		{ XMFLOAT3(1, 1, 1), XMFLOAT3(0, -10, 0), XMFLOAT3(0, 0, 1) },
		{ XMFLOAT3(2, 1, 1), XMFLOAT3(0, 0, 25), XMFLOAT3(0, 0.9f, 0) },
		{ XMFLOAT3(1, 1.5f, 1), XMFLOAT3(20, 35, 0), XMFLOAT3(0, 0.6f, 0) },
		{ XMFLOAT3(-1.25f, 1, 0.8f), XMFLOAT3(0, 0, 30), XMFLOAT3(0.1f, 0.5f, 0) },
	};

	// The keys of the source animation, linearly interpolated:
	constexpr int key_count = 3;
	const float key_times[key_count] = { 0, 0.5f, 1.2f };
	struct BoneKeys
	{
		XMFLOAT3 translation[key_count];
		XMFLOAT3 rotation[key_count]; // XMQuaternionRotationRollPitchYaw() angles in degrees
		XMFLOAT3 scale[key_count];
	};
	const BoneKeys source_keys[bone_count] = {
		{
			{ XMFLOAT3(0, 1, 0), XMFLOAT3(0.2f, 1.1f, -0.1f), XMFLOAT3(-0.1f, 0.9f, 0.3f) },
			{ XMFLOAT3(10, 0, 5), XMFLOAT3(-20, 30, 0), XMFLOAT3(5, -40, 60) },
			{ XMFLOAT3(1, 1, 1), XMFLOAT3(1.1f, 0.9f, 1), XMFLOAT3(0.8f, 1.2f, 1.1f) },
		},
		{
			{ XMFLOAT3(0, 0.5f, 0), XMFLOAT3(0.1f, 0.6f, 0), XMFLOAT3(0, 0.4f, -0.2f) },
			{ XMFLOAT3(0, 30, 20), XMFLOAT3(25, 10, -15), XMFLOAT3(-30, 50, 40) },
			{ XMFLOAT3(1, 1, 1), XMFLOAT3(0.9f, 1.3f, 1), XMFLOAT3(1.2f, 1, 0.7f) },
		},
		{
			{ XMFLOAT3(0, 0.4f, 0.1f), XMFLOAT3(-0.2f, 0.3f, 0), XMFLOAT3(0.1f, 0.5f, 0.2f) },
			{ XMFLOAT3(15, 0, 0), XMFLOAT3(0, -25, 35), XMFLOAT3(40, 20, -10) },
			{ XMFLOAT3(1, 1, 1), XMFLOAT3(1, 0.8f, 1.4f), XMFLOAT3(1.3f, 1.1f, 0.9f) },
		},
	};
	// The channels of the source animation, for each bone (in the order of the bones), retargeted animations keep this order:
	constexpr int path_count = 3;
	const Path channel_paths[path_count] = { Path::TRANSLATION, Path::ROTATION, Path::SCALE };
	constexpr int channel_count = bone_count * path_count;

	// The value of a source key as a vector:
	XMVECTOR KeyValue(int bone, Path path, int key)
	{
		switch (path)
		{
		case Path::TRANSLATION:
			return XMLoadFloat3(&source_keys[bone].translation[key]);
		case Path::ROTATION:
			return Rotation(source_keys[bone].rotation[key]);
		default:
			return XMLoadFloat3(&source_keys[bone].scale[key]);
		}
	}

	// The source animation of a skeleton, paused (only the animations retargeted from it are played):
	Entity CreateAnimation(Scene& scene, const Skeleton& skeleton)
	{
		Entity entity = CreateEntity();
		AnimationComponent& animation = scene.animations.Create(entity);
		animation.start = key_times[0];
		animation.end = key_times[key_count - 1];
		for (int bone = 0; bone < bone_count; ++bone)
		{
			for (Path path : channel_paths)
			{
				Entity data_entity = CreateEntity();
				AnimationDataComponent& data = scene.animation_datas.Create(data_entity);
				for (int key = 0; key < key_count; ++key)
				{
					data.keyframe_times.push_back(key_times[key]);
					const XMVECTOR V = KeyValue(bone, path, key);
					const int components = path == Path::ROTATION ? 4 : 3;
					for (int i = 0; i < components; ++i)
					{
						data.keyframe_data.push_back(XMVectorGetByIndex(V, i));
					}
				}

				AnimationComponent::AnimationSampler& sampler = animation.samplers.emplace_back();
				sampler.data = data_entity;
				AnimationComponent::AnimationChannel& channel = animation.channels.emplace_back();
				channel.target = skeleton.bones[bone];
				channel.samplerIndex = int(animation.samplers.size() - 1);
				channel.path = path;
			}
		}
		return entity;
	}

	// What Scene::Update() does for animations, without a graphics device:
	void UpdateAnimations(Scene& scene)
	{
		wi::jobsystem::context ctx;
		scene.ScanAnimationDependencies();
		scene.RunAnimationUpdateSystem(ctx);
		scene.RunTransformUpdateSystem(ctx);
		wi::jobsystem::Wait(ctx);
		scene.RunHierarchyUpdateSystem(ctx);
		wi::jobsystem::Wait(ctx);
	}

	// The retargeted cases, each plays an animation retargeted to its own destination skeleton in the same scene:
	enum Case
	{
		CASE_THIS_SCENE, // retargeted without baking, the source is in the same scene
		CASE_OTHER_SCENE, // retargeted without baking, the source is in an other scene
		CASE_BAKED, // retargeted with baking, from the same scene
		CASE_BAKED_OTHER_SCENE, // retargeted with baking, from an other scene
		CASE_NO_SOURCE_TRANSFORM, // retargeted without baking, the source transforms are removed after retargeting
		CASE_INVALID_RETARGET, // retargeted without baking, some channels have no valid retargetIndex
		CASE_COUNT,
	};
	const char* case_names[CASE_COUNT] = {
		"retargeted from this scene",
		"retargeted from an other scene",
		"baked from this scene",
		"baked from an other scene",
		"retargeted without the source transforms",
		"retargeted with invalid retarget indices",
	};
	const float sample_times[] = { 0, 0.3f, 0.5f, 0.9f, 1.2f };
	constexpr int sample_count = arraysize(sample_times);
	constexpr int values_per_bone = 3 + 4 + 3;
	constexpr int values_per_case = sample_count * bone_count * values_per_bone;
	// The channels made invalid in CASE_INVALID_RETARGET:
	constexpr int no_retarget_channel = 0; // hips translation, retargetIndex = -1
	constexpr int out_of_range_retarget_channel = 4; // spine rotation, retargetIndex = retargets.size()

	std::string DescribeTransformValue(size_t index)
	{
		const char* components[values_per_bone] = { "translation.x", "translation.y", "translation.z", "rotation.x", "rotation.y", "rotation.z", "rotation.w", "scale.x", "scale.y", "scale.z" };
		const size_t c = index / values_per_case;
		const size_t sample = index % values_per_case / (bone_count * values_per_bone);
		const size_t bone = index % (bone_count * values_per_bone) / values_per_bone;
		return std::string(c < CASE_COUNT ? case_names[c] : "?") + ", " + bone_names[bone] + " " + components[index % values_per_bone] + " at " + std::to_string(sample_times[sample]);
	}
	std::string DescribeBakedKey(size_t index)
	{
		return "baked key value " + std::to_string(index);
	}

	// The local transform of a bone, read from values of a case:
	struct BoneTransform
	{
		XMFLOAT3 translation;
		XMFLOAT4 rotation;
		XMFLOAT3 scale;
	};
	BoneTransform GetBoneTransform(const wi::vector<float>& values, int sample, int bone)
	{
		const float* v = &values[(sample * bone_count + bone) * values_per_bone];
		return { XMFLOAT3(v[0], v[1], v[2]), XMFLOAT4(v[3], v[4], v[5], v[6]), XMFLOAT3(v[7], v[8], v[9]) };
	}

	void TestRetargetedAnimations()
	{
		Scene scene; // the destinations, and the source retargeted from this scene
		Scene other; // the source retargeted from an other scene
		Scene vanished; // the source whose transforms are removed after retargeting

		const Skeleton source = CreateSkeleton(scene, source_nodes);
		const Entity source_animation = CreateAnimation(scene, source);
		const Skeleton other_source = CreateSkeleton(other, source_nodes);
		const Entity other_animation = CreateAnimation(other, other_source);
		const Skeleton vanished_source = CreateSkeleton(vanished, source_nodes);
		const Entity vanished_animation = CreateAnimation(vanished, vanished_source);

		Skeleton destinations[CASE_COUNT];
		Entity animations[CASE_COUNT];
		for (int c = 0; c < CASE_COUNT; ++c)
		{
			destinations[c] = CreateSkeleton(scene, destination_nodes);
		}
		animations[CASE_THIS_SCENE] = scene.RetargetAnimation(destinations[CASE_THIS_SCENE].root, source_animation, false);
		animations[CASE_OTHER_SCENE] = scene.RetargetAnimation(destinations[CASE_OTHER_SCENE].root, other_animation, false, &other);
		animations[CASE_BAKED] = scene.RetargetAnimation(destinations[CASE_BAKED].root, source_animation, true);
		animations[CASE_BAKED_OTHER_SCENE] = scene.RetargetAnimation(destinations[CASE_BAKED_OTHER_SCENE].root, other_animation, true, &other);
		animations[CASE_NO_SOURCE_TRANSFORM] = scene.RetargetAnimation(destinations[CASE_NO_SOURCE_TRANSFORM].root, vanished_animation, false, &vanished);
		animations[CASE_INVALID_RETARGET] = scene.RetargetAnimation(destinations[CASE_INVALID_RETARGET].root, source_animation, false);
		for (Entity bone : vanished_source.bones)
		{
			vanished.transforms.Remove(bone);
		}
		{
			AnimationComponent& animation = *scene.animations.GetComponent(animations[CASE_INVALID_RETARGET]);
			animation.channels[no_retarget_channel].retargetIndex = -1;
			animation.channels[out_of_range_retarget_channel].retargetIndex = (int)animation.retargets.size();
		}

		// Preconditions:
		for (int c = 0; c < CASE_COUNT; ++c)
		{
			const AnimationComponent* animation = scene.animations.GetComponent(animations[c]);
			Check(animation != nullptr && animation->channels.size() == size_t(channel_count), std::string("precondition: the animation was not retargeted: ") + case_names[c]);
			if (animation == nullptr || animation->channels.size() != size_t(channel_count))
				return;
			const bool baked = c == CASE_BAKED || c == CASE_BAKED_OTHER_SCENE;
			const void* source_scene = c == CASE_OTHER_SCENE ? (const void*)&other : (c == CASE_NO_SOURCE_TRANSFORM ? (const void*)&vanished : nullptr);
			for (int i = 0; i < channel_count; ++i)
			{
				const AnimationComponent::AnimationChannel& channel = animation->channels[i];
				const bool retargeted = channel.retargetIndex >= 0 && channel.retargetIndex < (int)animation->retargets.size();
				const bool invalid = c == CASE_INVALID_RETARGET && (i == no_retarget_channel || i == out_of_range_retarget_channel);
				Check(channel.target == destinations[c].bones[i / path_count] && channel.path == channel_paths[i % path_count], std::string("precondition: a retargeted channel has an other target or path: ") + case_names[c]);
				Check(retargeted == (!baked && !invalid), std::string("precondition: a channel has an unexpected retargetIndex: ") + case_names[c]);
				Check(animation->samplers[channel.samplerIndex].scene == source_scene, std::string("precondition: a sampler has data in an unexpected scene: ") + case_names[c]);
			}
		}
		{
			// The retargeted matrices of the keys must have shear (refused by XMMatrixDecompose()) and mirroring:
			const AnimationComponent& animation = *scene.animations.GetComponent(animations[CASE_THIS_SCENE]);
			bool shear = false;
			bool mirror = false;
			for (int i = 0; i < channel_count; ++i)
			{
				const int retargetIndex = animation.channels[i].retargetIndex;
				if (retargetIndex < 0 || retargetIndex >= (int)animation.retargets.size())
					continue; // (failed above)
				const AnimationComponent::RetargetSourceData& retarget = animation.retargets[retargetIndex];
				for (int key = 0; key < key_count; ++key)
				{
					TransformComponent transform = *scene.transforms.GetComponent(retarget.source);
					const XMVECTOR V = KeyValue(i / path_count, channel_paths[i % path_count], key);
					switch (channel_paths[i % path_count])
					{
					case Path::TRANSLATION: XMStoreFloat3(&transform.translation_local, V); break;
					case Path::ROTATION: XMStoreFloat4(&transform.rotation_local, V); break;
					default: XMStoreFloat3(&transform.scale_local, V); break;
					}
					const XMMATRIX M = XMLoadFloat4x4(&retarget.dstRelativeMatrix) * transform.GetLocalMatrix() * XMLoadFloat4x4(&retarget.srcRelativeParentMatrix);
					XMVECTOR S, R, T;
					shear |= !XMMatrixDecompose(&S, &R, &T, M);
					mirror |= XMVectorGetX(XMMatrixDeterminant(M)) < 0;
				}
			}
			Check(shear, "precondition: a retargeted matrix should have shear");
			Check(mirror, "precondition: a retargeted matrix should be mirrored");
		}

		// Play the retargeted animations at the sample times:
		wi::vector<float> values[CASE_COUNT];
		for (float time : sample_times)
		{
			for (Entity entity : animations)
			{
				AnimationComponent& animation = *scene.animations.GetComponent(entity);
				animation.Play();
				animation.timer = time;
			}
			scene.dt = 0; // (the timers are set)
			UpdateAnimations(scene);
			for (int c = 0; c < CASE_COUNT; ++c)
			{
				for (Entity bone : destinations[c].bones)
				{
					const TransformComponent& transform = *scene.transforms.GetComponent(bone);
					values[c].push_back(transform.translation_local.x);
					values[c].push_back(transform.translation_local.y);
					values[c].push_back(transform.translation_local.z);
					values[c].push_back(transform.rotation_local.x);
					values[c].push_back(transform.rotation_local.y);
					values[c].push_back(transform.rotation_local.z);
					values[c].push_back(transform.rotation_local.w);
					values[c].push_back(transform.scale_local.x);
					values[c].push_back(transform.scale_local.y);
					values[c].push_back(transform.scale_local.z);
				}
			}
		}
		wi::vector<float> transform_values;
		for (const auto& case_values : values)
		{
			transform_values.insert(transform_values.end(), case_values.begin(), case_values.end());
		}
		Compare("recorded_transforms", transform_values, recorded_transforms, arraysize(recorded_transforms), DescribeTransformValue);

		wi::vector<float> baked_keys[CASE_COUNT];
		for (int c : { CASE_BAKED, CASE_BAKED_OTHER_SCENE })
		{
			const AnimationComponent& animation = *scene.animations.GetComponent(animations[c]);
			for (const AnimationComponent::AnimationChannel& channel : animation.channels)
			{
				const AnimationDataComponent* data = scene.animation_datas.GetComponent(animation.samplers[channel.samplerIndex].data);
				Check(data != nullptr && data->keyframe_times.size() == size_t(key_count), std::string("the baked data is missing: ") + case_names[c]);
				if (data != nullptr)
				{
					baked_keys[c].insert(baked_keys[c].end(), data->keyframe_data.begin(), data->keyframe_data.end());
				}
			}
		}
		wi::vector<float> baked_key_values = baked_keys[CASE_BAKED];
		baked_key_values.insert(baked_key_values.end(), baked_keys[CASE_BAKED_OTHER_SCENE].begin(), baked_keys[CASE_BAKED_OTHER_SCENE].end());
		Compare("recorded_baked_keys", baked_key_values, recorded_baked_keys, arraysize(recorded_baked_keys), DescribeBakedKey);

		if (record)
			return;

		// The relations between the cases, which the recorded values must have:
		Check(values[CASE_OTHER_SCENE] == values[CASE_THIS_SCENE], "the animation retargeted from an other scene differs from the one retargeted from this scene with the same source");
		Check(values[CASE_BAKED_OTHER_SCENE] == values[CASE_BAKED] && baked_keys[CASE_BAKED_OTHER_SCENE] == baked_keys[CASE_BAKED], "the animation baked from an other scene differs from the one baked from this scene with the same source");
		for (int sample = 0; sample < sample_count; ++sample)
		{
			int key = -1;
			for (int k = 0; k < key_count; ++k)
			{
				if (sample_times[sample] == key_times[k])
					key = k;
			}
			for (int bone = 0; bone < bone_count; ++bone)
			{
				const std::string where = std::string(bone_names[bone]) + " at " + std::to_string(sample_times[sample]);
				const BoneTransform retargeted = GetBoneTransform(values[CASE_THIS_SCENE], sample, bone);
				const BoneTransform invalid = GetBoneTransform(values[CASE_INVALID_RETARGET], sample, bone);
				if (key >= 0)
				{
					// At the keys, the baked keys are what the runtime retargets:
					const BoneTransform baked = GetBoneTransform(values[CASE_BAKED], sample, bone);
					Check(Near(baked.translation.x, retargeted.translation.x, 1e-5f) && Near(baked.translation.y, retargeted.translation.y, 1e-5f) && Near(baked.translation.z, retargeted.translation.z, 1e-5f) &&
						SameRotation(baked.rotation, retargeted.rotation, 1e-5f) &&
						Near(baked.scale.x, retargeted.scale.x, 1e-5f) && Near(baked.scale.y, retargeted.scale.y, 1e-5f) && Near(baked.scale.z, retargeted.scale.z, 1e-5f),
						"the baked animation differs from the retargeted one at a key: " + where);

					// Without the source transforms, the keys are played as they are:
					const BoneTransform keys = GetBoneTransform(values[CASE_NO_SOURCE_TRANSFORM], sample, bone);
					XMFLOAT4 key_rotation;
					XMStoreFloat4(&key_rotation, KeyValue(bone, Path::ROTATION, key));
					const XMFLOAT3& key_translation = source_keys[bone].translation[key];
					const XMFLOAT3& key_scale = source_keys[bone].scale[key];
					Check(Near(keys.translation.x, key_translation.x, 1e-6f) && Near(keys.translation.y, key_translation.y, 1e-6f) && Near(keys.translation.z, key_translation.z, 1e-6f) &&
						SameRotation(keys.rotation, key_rotation, 1e-5f) &&
						Near(keys.scale.x, key_scale.x, 1e-6f) && Near(keys.scale.y, key_scale.y, 1e-6f) && Near(keys.scale.z, key_scale.z, 1e-6f),
						"a channel without its source transform was retargeted: " + where);

					// The channels without a valid retarget play their keys as they are:
					if (bone == no_retarget_channel / path_count)
					{
						Check(Near(invalid.translation.x, key_translation.x, 1e-6f) && Near(invalid.translation.y, key_translation.y, 1e-6f) && Near(invalid.translation.z, key_translation.z, 1e-6f),
							"a channel with retargetIndex -1 was retargeted: " + where);
					}
					if (bone == out_of_range_retarget_channel / path_count)
					{
						Check(SameRotation(invalid.rotation, key_rotation, 1e-5f), "a channel with an out of range retargetIndex was retargeted: " + where);
					}
				}
				// The other channels are retargeted:
				if (bone != no_retarget_channel / path_count)
				{
					Check(std::memcmp(&invalid.translation, &retargeted.translation, sizeof(XMFLOAT3)) == 0, "a translation next to an invalid retarget differs from the retargeted one: " + where);
				}
				if (bone != out_of_range_retarget_channel / path_count)
				{
					Check(std::memcmp(&invalid.rotation, &retargeted.rotation, sizeof(XMFLOAT4)) == 0, "a rotation next to an invalid retarget differs from the retargeted one: " + where);
				}
				Check(std::memcmp(&invalid.scale, &retargeted.scale, sizeof(XMFLOAT3)) == 0, "a scale next to an invalid retarget differs from the retargeted one: " + where);
			}
		}
	}

	// A scale key with a zero component, retargeted with shear: the retargeted matrix has a zero length axis and no handedness.
	//	The scale and the rotation of a retargeted channel are now decomposed alike, with wi::math::DecomposeTransform(), which
	//	doesn't mirror such a matrix. Before, the scale was decomposed with XMMatrixDecompose(), which mirrored it depending on the
	//	order of the axis it puts in place of the zero length one (here it negates the longest axis), so the recorded scale differs
	//	from the current one in the sign of its x, and the retargeted longest axis pointed opposite to the retargeted matrix's.
	void TestFlatScale()
	{
		Scene scene;
		const Entity source = CreateNode(scene, { XMFLOAT3(1, 1, 1), XMFLOAT3(30, 40, 10), XMFLOAT3(0, 1, 0) });
		const Entity target = CreateNode(scene, { XMFLOAT3(1, 1, 1), XMFLOAT3(0, 0, 0), XMFLOAT3(0, 0, 0) });
		const XMFLOAT3 key(2, 0, 1);

		Entity animation_entity = CreateEntity();
		AnimationComponent& animation = scene.animations.Create(animation_entity);
		animation.end = 1;
		AnimationComponent::RetargetSourceData& retarget = animation.retargets.emplace_back();
		retarget.source = source;
		// (dstRelativeMatrix is the identity, which keeps the zero length axis of the key)
		XMStoreFloat4x4(&retarget.srcRelativeParentMatrix, XMMatrixScaling(1, 2, 1) * XMMatrixRotationRollPitchYaw(0.3f, 0.5f, 0.2f));
		Entity data_entity = CreateEntity();
		AnimationDataComponent& data = scene.animation_datas.Create(data_entity);
		data.keyframe_times = { 0, 1 };
		data.keyframe_data = { key.x, key.y, key.z, key.x, key.y, key.z };
		AnimationComponent::AnimationSampler& sampler = animation.samplers.emplace_back();
		sampler.data = data_entity;
		AnimationComponent::AnimationChannel& channel = animation.channels.emplace_back();
		channel.target = target;
		channel.samplerIndex = 0;
		channel.retargetIndex = 0;
		channel.path = Path::SCALE;

		TransformComponent transform = *scene.transforms.GetComponent(source);
		transform.scale_local = key;
		const XMMATRIX M = XMLoadFloat4x4(&retarget.dstRelativeMatrix) * transform.GetLocalMatrix() * XMLoadFloat4x4(&retarget.srcRelativeParentMatrix);
		XMVECTOR S, R, T;
		Check(!XMMatrixDecompose(&S, &R, &T, M) && XMVectorGetX(S) < 0, "precondition: XMMatrixDecompose() should refuse and mirror the retargeted matrix of the flat scale");
		XMVECTOR expected_scale, expected_rotation;
		wi::math::DecomposeTransform(&expected_scale, &expected_rotation, &T, M);

		animation.Play();
		animation.timer = 0.5f;
		UpdateAnimations(scene);
		const XMFLOAT3 scale = scene.transforms.GetComponent(target)->scale_local;
		if (record)
		{
			Print("recorded_flat_scale", { scale.x, scale.y, scale.z });
			return;
		}

		// The expected differences from the recorded scale:
		Check(Near(scale.x, -recorded_flat_scale[0], 1e-4f) && Near(scale.y, recorded_flat_scale[1], 1e-4f) && Near(scale.z, recorded_flat_scale[2], 1e-4f), "the retargeted flat scale should be the recorded one, not mirrored");
		Check(scale.x > 0 && recorded_flat_scale[0] < 0, "the retargeted flat scale should not be mirrored, the recorded one was");
		Check(XMVectorGetX(XMVector3Length(XMLoadFloat3(&scale) - expected_scale)) < 1e-6f, "the retargeted flat scale is not decomposed with DecomposeTransform()");
		// The retargeted longest axis, with the rotation that the retargeting decomposes from the same matrix, points along the matrix's:
		const XMVECTOR axis = scale.x * XMVector3Rotate(XMVectorSet(1, 0, 0, 0), expected_rotation);
		Check(XMVectorGetX(XMVector3Length(axis - M.r[0])) < 1e-5f, "the retargeted flat scale and rotation don't make the retargeted matrix's longest axis");
	}
}

int main(int argc, char** argv)
{
	record = argc > 1 && std::strcmp(argv[1], "--record") == 0;
	wi::jobsystem::Initialize();

	TestRetargetedAnimations();
	TestFlatScale();

	if (record)
		return 0;
	if (failures > 0)
	{
		std::cerr << "AnimationRetarget test: " << failures << " check(s) failed\n";
		return 1;
	}
	std::cout << "AnimationRetarget test passed\n";
	return 0;
}
