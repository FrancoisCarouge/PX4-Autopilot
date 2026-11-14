/****************************************************************************
 *
 *   Copyright (C) 2026 PX4 Development Team. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/**
 * Test the wind estimator's state and covariance handling.
 */

#include <gtest/gtest.h>

#include "../WindEstimator.hpp"

TEST(TestWindEstimator, ScaleResetKeepsWindCovariance)
{
	WindEstimator wind_estimator;

	// initialised without airspeed: the wind variances are the initial ones
	wind_estimator.fuse_beta(hrt_absolute_time(), matrix::Vector3f{15.f, 0.f, 0.f}, 0.1f, matrix::Quatf{});
	ASSERT_TRUE(wind_estimator.is_estimate_valid());
	const matrix::Vector2f wind_var = wind_estimator.get_wind_var();
	ASSERT_GT(wind_var(0), 0.f);

	wind_estimator.set_scale_init(1.2f);
	wind_estimator.reset_scale_to_init();

	EXPECT_FLOAT_EQ(wind_estimator.get_wind_var()(0), wind_var(0));
	EXPECT_FLOAT_EQ(wind_estimator.get_wind_var()(1), wind_var(1));
	EXPECT_FLOAT_EQ(wind_estimator.get_tas_scale(), 1.2f);
	EXPECT_FLOAT_EQ(wind_estimator.get_tas_scale_var(), 0.f);
}
