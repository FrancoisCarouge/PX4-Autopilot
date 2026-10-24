/****************************************************************************
 *
 *   Copyright (c) 2020-2022 PX4 Development Team. All rights reserved.
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
 * @file zero_order_hover_thrust_ekf.cpp
 *
 * @author Mathieu Bresciani 	<brescianimathieu@gmail.com>
 */

#include "zero_order_hover_thrust_ekf.hpp"

using matrix::sign;

void ZeroOrderHoverThrustEkf::predict(const float dt)
{
	// State is constant
	// Predict state covariance only
	_filter.q(Thrust{_process_var *dt *dt * mp_units::one});
	_filter.predict();
	_dt = dt;
}

void ZeroOrderHoverThrustEkf::fuseAccZ(const float acc_z_value, const float thrust_value)
{
	const Acceleration acc_z{acc_z_value * metre_per_second_squared};
	const Thrust thrust{thrust_value *mp_units::one};

	const AccelerationPerThrust H = zero_order_hover_thrust_ekf::observationJacobian(_filter.x(), thrust);
	const AccelerationVariance innov_var = computeInnovVar(H);
	const Acceleration innov = computeInnov(acc_z, thrust);
	const float innov_test_ratio = computeInnovTestRatio(innov, innov_var);

	Acceleration residual = innov;

	if (isTestRatioPassing(innov_test_ratio)) {
		_filter.r(_acc_var * _acc_var_scale);
		_filter.update(thrust, acc_z);
		constrainState();

		residual = computeInnov(acc_z, thrust); // residual != innovation since the hover thrust changed

	} else if (isLargeOffsetDetected()) {
		// Rejecting all the measurements for some time,
		// it means that the hover thrust suddenly changed or that the EKF
		// is diverging. To recover, we bump the state variance
		bumpStateVariance();
	}

	const float signed_innov_test_ratio = sign(innov.numerical_value_in(metre_per_second_squared)) * innov_test_ratio;
	updateLpf(residual, signed_innov_test_ratio);
	updateMeasurementNoise(residual, H);

	// save for logging
	_innov = innov;
	_innov_var = innov_var;
	_innov_test_ratio = innov_test_ratio;
}

inline ZeroOrderHoverThrustEkf::AccelerationVariance ZeroOrderHoverThrustEkf::computeInnovVar(const AccelerationPerThrust H) const
{
	const AccelerationVariance R = _acc_var * _acc_var_scale;
	const Thrust P = _filter.p();
	return math::max(H * P * H + R, R);
}

ZeroOrderHoverThrustEkf::Acceleration ZeroOrderHoverThrustEkf::computeInnov(const Acceleration acc_z, const Thrust thrust) const
{
	return acc_z - zero_order_hover_thrust_ekf::predictedAccZ(_filter.x(), thrust);
}

inline float ZeroOrderHoverThrustEkf::computeInnovTestRatio(const Acceleration innov, const AccelerationVariance innov_var) const
{
	return (innov * innov / innov_var).numerical_value_in(mp_units::one) / (_gate_size * _gate_size);
}

inline bool ZeroOrderHoverThrustEkf::isTestRatioPassing(const float innov_test_ratio) const
{
	return innov_test_ratio < 1.f;
}

inline void ZeroOrderHoverThrustEkf::constrainState()
{
	_filter.x(Thrust{math::constrain(_filter.x().numerical_value_in(mp_units::one), _hover_thr_min, _hover_thr_max) * mp_units::one});
	_filter.p(Thrust{math::constrain(_filter.p().numerical_value_in(mp_units::one), 1e-10f, 1.f) * mp_units::one});
}

inline bool ZeroOrderHoverThrustEkf::isLargeOffsetDetected() const
{
	return fabsf(_signed_innov_test_ratio_lpf) > 0.2f;
}

inline void ZeroOrderHoverThrustEkf::bumpStateVariance()
{
	_filter.p(_filter.p() + Thrust{1e3f * _process_var *_dt *_dt * mp_units::one});
}

inline void ZeroOrderHoverThrustEkf::updateLpf(const Acceleration residual, const float signed_innov_test_ratio)
{
	const float alpha = _dt / (_lpf_time_constant + _dt);
	_residual_lpf = (1.f - alpha) * _residual_lpf + alpha * residual;
	_signed_innov_test_ratio_lpf = (1.f - alpha) * _signed_innov_test_ratio_lpf + alpha * math::constrain(
					       signed_innov_test_ratio, -1.f, 1.f);
}

inline void ZeroOrderHoverThrustEkf::updateMeasurementNoise(const Acceleration residual, const AccelerationPerThrust H)
{
	const float alpha = _dt / (_noise_learning_time_constant + _dt);
	const Acceleration res_no_bias = residual - _residual_lpf;
	const Thrust P = _filter.p();
	const AccelerationVariance learned = (1.f - alpha) * _acc_var + alpha * (res_no_bias * res_no_bias + H * P * H);
	_acc_var = math::constrain(learned.numerical_value_in(AccelerationVariance::unit), 1.f, 400.f) * AccelerationVariance::reference;
}
