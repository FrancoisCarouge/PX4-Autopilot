/****************************************************************************
 *
 *   Copyright (c) 2020-2021 PX4 Development Team. All rights reserved.
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
 * @file bias_estimator.cpp
 *
 * @author Mathieu Bresciani 	<mathieu@auterion.com>
 */

#include "bias_estimator.hpp"

#include <px4_typed_linear_algebra.hpp>

#include <new>
#include <type_traits>

namespace
{
using px4_typed_linear_algebra::Length;
using Area = px4_typed_linear_algebra::Quantity<mp_units::si::metre *mp_units::si::metre>;

constexpr auto metre = mp_units::si::metre;
constexpr auto square_metre = metre * metre;

// The bias is noise driven, A = 1, and observed directly, H = 1: the
// prediction adds the process noise to the variance, set before each
// prediction, and the update corrects the variance as P - K P: the simple
// form of a scalar filter is (1 - K) P, which rounds differently, so the
// variance is corrected again from the filter's gain.
auto makeFilter(float state_init, float state_var_init)
{
	using namespace fcarouge;
	return kalman{
		state{Length{state_init * metre}},
		output<Length>,
		estimate_uncertainty{Area{state_var_init * square_metre}},
		process_uncertainty{Area{0.f * square_metre}},
		output_uncertainty{Area{0.f * square_metre}},
		simple_form};
}
} // namespace

struct BiasEstimator::Filter {
	decltype(makeFilter(0.f, 0.f)) kalman;
};

BiasEstimator::Filter &BiasEstimator::filter()
{
	return *std::launder(static_cast<Filter *>(static_cast<void *>(_filter_storage.bytes)));
}

const BiasEstimator::Filter &BiasEstimator::filter() const
{
	return *std::launder(static_cast<const Filter *>(static_cast<const void *>(_filter_storage.bytes)));
}

BiasEstimator::BiasEstimator() : BiasEstimator(0.f, 0.1f) {}

BiasEstimator::BiasEstimator(float state_init, float state_var_init)
{
	static_assert(sizeof(Filter) <= kFilterSize, "BiasEstimator::kFilterSize is too small for the filter");
	static_assert(alignof(Filter) <= alignof(FilterStorage), "BiasEstimator::FilterStorage is under-aligned for the filter");
	static_assert(std::is_trivially_copy_constructible_v<Filter> &&std::is_trivially_copy_assignable_v<Filter>
		      &&std::is_trivially_destructible_v<Filter>, "BiasEstimator is copied with its filter's storage");
	new (_filter_storage.bytes) Filter{makeFilter(state_init, state_var_init)};
}

BiasEstimator::BiasEstimator(const BiasEstimator &other) = default;
BiasEstimator &BiasEstimator::operator=(const BiasEstimator &other) = default;

void BiasEstimator::setBias(float bias)
{
	filter().kalman.x(bias * metre);
}

void BiasEstimator::setBiasVar(float bias_var)
{
	filter().kalman.p(bias_var * square_metre);
}

float BiasEstimator::getBias() const
{
	return filter().kalman.x().numerical_value_in(metre);
}

float BiasEstimator::getBiasVar() const
{
	return filter().kalman.p().numerical_value_in(square_metre);
}

void BiasEstimator::predict(const float dt)
{
	// State is constant
	// Predict state covariance only
	float delta_state_var = _process_psd * dt;

	if (isOffsetDetected()) {
		// A bias in the state has been detected by the innovation sequence check
		// Boost the process noise until the offset is removed
		delta_state_var *= _process_var_boost_gain;
	}

	filter().kalman.q(delta_state_var * square_metre);
	filter().kalman.predict();
	constrainStateVar();

	if (dt > FLT_EPSILON && fabsf(_dt - dt) > 0.001f) {
		_signed_innov_test_ratio_lpf.setParameters(dt, _innov_sequence_monitnoring_time_constant);
		_dt = dt;
	}

	_status.bias_var = getBiasVar();
}

void BiasEstimator::constrainStateVar()
{
	setBiasVar(math::constrain(getBiasVar(), 1e-8f, _state_var_max));
}

void BiasEstimator::fuseBias(const float measurement, const float measurement_var)
{
	const float R = math::max(sq(0.01f), measurement_var);
	const float innov_var = getBiasVar() + R;
	const float innov = measurement - getBias();
	const float innov_test_ratio = computeInnovTestRatio(innov, innov_var);

	if (isTestRatioPassing(innov_test_ratio)) {
		// K = P / (P + R), x += K innov, P -= K P
		const float P = getBiasVar();
		filter().kalman.r(R * square_metre);
		filter().kalman.update(measurement * metre);
		const float K = filter().kalman.k().numerical_value_in(mp_units::one);
		setBiasVar(P - K * P);
		constrainStateVar();
	}

	updateOffsetDetection(innov, innov_test_ratio);

	_status = packStatus(innov, innov_var, innov_test_ratio);
}

inline float BiasEstimator::computeInnovTestRatio(const float innov, const float innov_var) const
{
	return innov * innov / (_gate_size * _gate_size * innov_var);
}

inline bool BiasEstimator::isTestRatioPassing(const float innov_test_ratio) const
{
	return innov_test_ratio < 1.f;
}

inline void BiasEstimator::updateOffsetDetection(const float innov, const float innov_test_ratio)
{
	const float signed_innov_test_ratio = matrix::sign(innov) * innov_test_ratio;
	_signed_innov_test_ratio_lpf.update(math::constrain(signed_innov_test_ratio, -1.f, 1.f));

	if (innov > 0.f) {
		_time_since_last_positive_innov = 0.f;
		_time_since_last_negative_innov += _dt;

	} else {
		_time_since_last_negative_innov = 0.f;
		_time_since_last_positive_innov += _dt;
	}
}

inline bool BiasEstimator::isOffsetDetected() const
{
	// There is an offset in the estimate if the average of innovation is statistically too large
	// or if the sign of the innovation is constantly the same
	return fabsf(_signed_innov_test_ratio_lpf.getState()) > 0.2f
	       || (_time_since_last_positive_innov > _innov_sequence_monitnoring_time_constant)
	       || (_time_since_last_negative_innov > _innov_sequence_monitnoring_time_constant);
}

inline BiasEstimator::status BiasEstimator::packStatus(const float innov, const float innov_var,
		const float innov_test_ratio) const
{
	// Send back status for logging
	status ret{};
	ret.bias = getBias();
	ret.bias_var = getBiasVar();
	ret.innov = innov;
	ret.innov_var = innov_var;
	ret.innov_test_ratio = innov_test_ratio;

	return ret;
}
