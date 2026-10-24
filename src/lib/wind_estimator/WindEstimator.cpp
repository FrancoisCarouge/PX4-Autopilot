/****************************************************************************
 *
 *   Copyright (c) 2018-2023 PX4 Development Team. All rights reserved.
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
 * @file WindEstimator.cpp
 * A wind and airspeed scale estimator.
 */

#include "WindEstimator.hpp"

#include <algorithm>
#include "python/generated/init_wind_using_airspeed.h"

using namespace wind_estimator;

namespace
{
// the typed values of the generated functions' matrices, and back
template <typename Typed, typename T, size_t R, size_t C>
Typed typed(const matrix::Matrix<T, R, C> &m)
{
	return Typed{fcarouge_px4::Matrix<T, R, C>{m}};
}

template <typename Typed>
auto untyped(const Typed &typed)
{
	return typed.data().toMatrix();
}
} // namespace

bool
WindEstimator::initialise(const matrix::Vector3f &velI, const float hor_vel_variance, const float heading_rad,
			  const float tas_meas, const float tas_variance)
{
	if (PX4_ISFINITE(tas_meas) && PX4_ISFINITE(tas_variance)) {

		constexpr float initial_heading_var = sq(math::radians(kInitialHeadingErrorDeg));
		constexpr float initial_sideslip_var = sq(math::radians(kInitialBetaErrorDeg));

		matrix::SquareMatrix<float, 2> P_wind_init;
		matrix::Vector2f wind_init;

		sym::InitWindUsingAirspeed(velI, heading_rad, tas_meas, hor_vel_variance, initial_heading_var, initial_sideslip_var,
					   tas_variance,
					   &wind_init, &P_wind_init);

		_state = State{wind_init(0) *metre_per_second, wind_init(1) *metre_per_second, _scale_init};
		// the wind covariance, the scale's variance and covariances kept
		_P.at<INDEX_W_N, INDEX_W_N>(P_wind_init(0, 0) * SpeedVariance::reference);
		_P.at<INDEX_W_N, INDEX_W_E>(P_wind_init(0, 1) * SpeedVariance::reference);
		_P.at<INDEX_W_E, INDEX_W_N>(P_wind_init(1, 0) * SpeedVariance::reference);
		_P.at<INDEX_W_E, INDEX_W_E>(P_wind_init(1, 1) * SpeedVariance::reference);

	} else {
		// no airspeed available
		_state = State{0.f * metre_per_second, 0.f * metre_per_second, 1.f *mp_units::one};
		_P = Covariance{};
		_P.at<INDEX_W_N, INDEX_W_N>(sq(kInitialWindError) * SpeedVariance::reference);
		_P.at<INDEX_W_E, INDEX_W_E>(sq(kInitialWindError) * SpeedVariance::reference);
	}

	_wind_estimator_reset = true;
	_time_initialised = hrt_absolute_time();

	return true;
}

void
WindEstimator::reset_scale_to_init()
{
	_state.at<INDEX_TAS_SCALE>(_scale_init);
	Covariance P{};
	P.at<INDEX_W_N, INDEX_W_N>(_P.at<INDEX_W_N, INDEX_W_N>());
	P.at<INDEX_W_N, INDEX_W_E>(_P.at<INDEX_W_N, INDEX_W_E>());
	P.at<INDEX_W_E, INDEX_W_N>(_P.at<INDEX_W_E, INDEX_W_N>());
	P.at<INDEX_W_E, INDEX_W_E>(_P.at<INDEX_W_E, INDEX_W_E>());
	_P = P;
}

void
WindEstimator::update(uint64_t time_now)
{
	if (!_initialised) {
		return;
	}

	// set reset state to false (is set to true when initialise fuction is called later)
	_wind_estimator_reset = false;

	// run covariance prediction at 1Hz
	if (time_now - _time_last_update < 1_s || _time_last_update == 0) {
		if (_time_last_update == 0) {
			_time_last_update = time_now;
		}

		return;
	}

	const Seconds dt{(float)(time_now - _time_last_update) * 1e-6f * mp_units::si::second};
	_time_last_update = time_now;

	// if airspeed scale is at default (1.0) and we are in the first 5 minutes of flight time, multiply kTasScalePsd by 100 for faster learning
	const float tas_psd_multiplier = (fabsf(_scale_init.numerical_value_in(mp_units::one) - 1.0f) < FLT_EPSILON
					  && (time_now - _time_initialised < kTASScaleFastLearnTime)) ? kTASScalePSDMultiplier : 1.f;

	Covariance Qk{};
	Qk.at<INDEX_W_N, INDEX_W_N>(_wind_psd * dt);
	Qk.at<INDEX_W_E, INDEX_W_E>(_wind_psd * dt);
	Qk.at<INDEX_TAS_SCALE, INDEX_TAS_SCALE>(kTasScalePsd * tas_psd_multiplier * dt);
	_P = Covariance{_P + Qk};
}

template <typename Observation>
void
WindEstimator::correct(const matrix::Matrix<float, 1, 3> &H, const matrix::Matrix<float, 3, 1> &K, Observation innov)
{
	const auto h = typed<Jacobian<Observation>>(H);
	const auto k = typed<Gain<Observation>>(K);

	// apply correction to state
	_state = State{_state + k *fcarouge_px4::ColumnVector<float, Observation> {innov}};

	// update covariance matrix
	_P = Covariance{_P - k *h * _P};
}

void
WindEstimator::fuse_airspeed(uint64_t time_now, const float true_airspeed, const matrix::Vector3f &velI,
			     const float hor_vel_variance, const matrix::Quatf &q_att)
{

	if (!_initialised) {
		// try to initialise
		_initialised = initialise(velI, hor_vel_variance, matrix::Eulerf(q_att).psi(), true_airspeed,
					  _tas_var.numerical_value_in(SpeedVariance::unit));
		return;
	}

	// don't fuse faster than 10Hz
	if (time_now - _time_last_airspeed_fuse < 100_ms) {
		return;
	}

	_time_last_airspeed_fuse = time_now;

	matrix::Matrix<float, 1, 3> H_tas;
	matrix::Matrix<float, 3, 1> K;
	float innov_var;
	float innov;

	sym::FuseAirspeed(velI, untyped(_state), untyped(_P), true_airspeed, _tas_var.numerical_value_in(SpeedVariance::unit),
			  FLT_EPSILON, &H_tas, &K, &innov_var, &innov);

	_tas_innov = innov * metre_per_second;
	_tas_innov_var = innov_var * SpeedVariance::reference;

	const bool meas_is_rejected = check_if_meas_is_rejected(innov, innov_var, kTasGate);

	if (innov_var < FLT_EPSILON) {
		// re init filter in case of a negative variance, and trigger early return to not fuse measurement
		_initialised = initialise(velI, hor_vel_variance, matrix::Eulerf(q_att).psi(), true_airspeed,
					  _tas_var.numerical_value_in(SpeedVariance::unit));
		return;

	} else if (meas_is_rejected) {
		return;
	}

	correct(H_tas, K, _tas_innov);

	run_sanity_checks();
}

void
WindEstimator::fuse_beta(uint64_t time_now, const matrix::Vector3f &velI, const float hor_vel_variance,
			 const matrix::Quatf &q_att)
{
	if (!_initialised) {
		_initialised = initialise(velI, hor_vel_variance, matrix::Eulerf(q_att).psi());
		return;
	}

	// don't fuse faster than 10Hz
	if (time_now - _time_last_beta_fuse < 100_ms) {
		return;
	}

	_time_last_beta_fuse = time_now;


	matrix::Matrix<float, 1, 3> H_beta;
	matrix::Matrix<float, 3, 1> K;
	float innov_var;
	float innov;

	sym::FuseBeta(velI, untyped(_state), untyped(_P), q_att, kBetaVar.numerical_value_in(SideslipVariance::unit),
		      FLT_EPSILON, &H_beta, &K, &innov_var, &innov);

	_beta_innov = innov * mp_units::si::radian;
	_beta_innov_var = innov_var * SideslipVariance::reference;

	const bool meas_is_rejected = check_if_meas_is_rejected(innov, innov_var, kBetaGate);

	if (innov_var < FLT_EPSILON) {
		// re init filter in case of a negative variance, and trigger early return to not fuse measurement
		_initialised = initialise(velI, hor_vel_variance, matrix::Eulerf(q_att).psi());
		return;

	} else if (meas_is_rejected) {
		return;
	}

	correct(H_beta, K, _beta_innov);

	run_sanity_checks();
}

void
WindEstimator::run_sanity_checks()
{
	const auto wind_var_limit = 25.f * SpeedVariance::reference;
	const auto scale_var_limit = 0.1f * mp_units::one;

	if (_P.at<INDEX_W_N, INDEX_W_N>() < 0.f * SpeedVariance::reference
	    || _P.at<INDEX_W_E, INDEX_W_E>() < 0.f * SpeedVariance::reference
	    || _P.at<INDEX_TAS_SCALE, INDEX_TAS_SCALE>() < 0.f * mp_units::one) {
		// ill-conditioned covariance matrix, reset filter
		_initialised = false;
		return;
	}

	// limit covariance diagonals if they grow too large
	_P.at<INDEX_W_N, INDEX_W_N>(std::min(_P.at<INDEX_W_N, INDEX_W_N>(), wind_var_limit));
	_P.at<INDEX_W_E, INDEX_W_E>(std::min(_P.at<INDEX_W_E, INDEX_W_E>(), wind_var_limit));
	_P.at<INDEX_TAS_SCALE, INDEX_TAS_SCALE>(std::min(_P.at<INDEX_TAS_SCALE, INDEX_TAS_SCALE>(), scale_var_limit));

	if (!untyped(_state).isAllFinite()) {
		_initialised = false;
		return;
	}

	// attain symmetry
	_P = Covariance{(_P + fcarouge::transposed(_P)) * 0.5f};
}

bool
WindEstimator::check_if_meas_is_rejected(float innov, float innov_var, uint8_t gate_size)
{
	return (innov * innov > gate_size * gate_size * innov_var);
}
