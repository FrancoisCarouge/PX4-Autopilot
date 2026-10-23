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
 * @file WindEstimator.hpp
 * A wind and airspeed scale estimator.
 */

#pragma once

#include <drivers/drv_hrt.h>

#include <mathlib/mathlib.h>
#include <matrix/math.hpp>

#include <fcarouge/px4_typed_linear_algebra.hpp>
#include <mp-units/systems/si.h>

#include "python/generated/fuse_airspeed.h"
#include "python/generated/fuse_beta.h"
#include "python/generated/init_wind_using_airspeed.h"

using namespace time_literals;

namespace wind_estimator
{
template <auto Reference>
using Quantity = mp_units::quantity<Reference, float>;

inline constexpr auto metre_per_second = mp_units::si::metre / mp_units::si::second;

using Unitless = Quantity<mp_units::one>;
using Seconds = Quantity<mp_units::si::second>;
using Speed = Quantity<metre_per_second>;
using SpeedVariance = Quantity<metre_per_second *metre_per_second>;
/// the wind process noise power spectral density, (m/s^2)^2/Hz
using WindPsd = Quantity < metre_per_second * metre_per_second / mp_units::si::second >;
/// the true airspeed scale process noise power spectral density, 1/s^2/Hz
using ScalePsd = Quantity < mp_units::one / mp_units::si::second >;
/// the sideslip, the ratio of the lateral to the longitudinal relative wind
using Sideslip = Quantity<mp_units::si::radian>;

/// the wind north and east, and the true airspeed scale
using State = fcarouge_px4::ColumnVector<float, Speed, Speed, Unitless>;
using Covariance = decltype(State{} * fcarouge::transposed(State{}));
/// a scalar observation, its sensitivity to the state, and the gain of its correction
template <typename Observation>
using Jacobian = decltype(fcarouge_px4::ColumnVector<float, Observation> {} / State{});
template <typename Observation>
using Gain = decltype(State{} / fcarouge_px4::ColumnVector<float, Observation> {});
} // namespace wind_estimator

class WindEstimator
{
public:
	WindEstimator() = default;
	~WindEstimator() = default;

	// no copy, assignment, move, move assignment
	WindEstimator(const WindEstimator &) = delete;
	WindEstimator &operator=(const WindEstimator &) = delete;
	WindEstimator(WindEstimator &&) = delete;
	WindEstimator &operator=(WindEstimator &&) = delete;

	void update(uint64_t time_now);

	void fuse_airspeed(uint64_t time_now, float true_airspeed, const matrix::Vector3f &velI,
			   const float hor_vel_variance, const matrix::Quatf &q_att);
	void fuse_beta(uint64_t time_now, const matrix::Vector3f &velI, const float hor_vel_variance,
		       const matrix::Quatf &q_att);

	bool is_estimate_valid() { return _initialised; }

	bool check_if_meas_is_rejected(float innov, float innov_var, uint8_t gate_size);

	matrix::Vector2f get_wind()
	{
		return matrix::Vector2f{_state.at<INDEX_W_N>().numerical_value_in(metre_per_second), _state.at<INDEX_W_E>().numerical_value_in(metre_per_second)};
	}

	// invert scale (CAS = IAS * scale), protect agains division by 0, constrain to [0.1, 10]
	float get_tas_scale() { return 1.f / math::constrain(_state.at<INDEX_TAS_SCALE>().numerical_value_in(mp_units::one), 0.1f, 10.0f); }
	float get_tas_scale_var() { return _P.at<INDEX_TAS_SCALE, INDEX_TAS_SCALE>().numerical_value_in(mp_units::one); }
	float get_tas_innov() { return _tas_innov.numerical_value_in(metre_per_second); }
	float get_tas_innov_var() { return _tas_innov_var.numerical_value_in(SpeedVariance::unit); }
	float get_beta_innov() { return _beta_innov.numerical_value_in(mp_units::si::radian); }
	float get_beta_innov_var() { return _beta_innov_var.numerical_value_in(SideslipVariance::unit); }
	matrix::Vector2f get_wind_var()
	{
		return matrix::Vector2f{_P.at<INDEX_W_N, INDEX_W_N>().numerical_value_in(SpeedVariance::unit), _P.at<INDEX_W_E, INDEX_W_E>().numerical_value_in(SpeedVariance::unit)};
	}
	bool get_wind_estimator_reset() { return _wind_estimator_reset; }

	// unaided, the state uncertainty (diagonal of sqrt(P)) grows by the process noise spectral density every second
	void set_wind_process_noise_spectral_density(float wind_nsd) { _wind_psd = wind_nsd * wind_nsd * WindPsd::reference; }

	void set_tas_noise(float tas_sigma) { _tas_var = tas_sigma * tas_sigma * SpeedVariance::reference; }

	void set_scale_init(float scale_init) {_scale_init = 1.f / math::constrain(scale_init, 0.1f, 10.f) * mp_units::one; }

	void reset_scale_to_init();

private:
	using Unitless = wind_estimator::Unitless;
	using Seconds = wind_estimator::Seconds;
	using Speed = wind_estimator::Speed;
	using SpeedVariance = wind_estimator::SpeedVariance;
	using WindPsd = wind_estimator::WindPsd;
	using ScalePsd = wind_estimator::ScalePsd;
	using Sideslip = wind_estimator::Sideslip;
	using SideslipVariance = wind_estimator::Quantity<mp_units::si::radian *mp_units::si::radian>;
	using State = wind_estimator::State;
	using Covariance = wind_estimator::Covariance;
	static constexpr auto metre_per_second = wind_estimator::metre_per_second;

	enum {
		INDEX_W_N = 0,
		INDEX_W_E,
		INDEX_TAS_SCALE
	};	///< enum which can be used to access state.

	static constexpr float kInitialWindError =
		2.5f; // initial uncertainty of each wind state when filter is initialised without airspeed [m/s]
	static constexpr float kInitialBetaErrorDeg =
		15.0f;	// sidelip uncertainty used to initialise the filter with a true airspeed measurement [deg]
	static constexpr float kInitialHeadingErrorDeg =
		10.0f; // heading uncertainty used to initialise the filter with a true airspeed measurement [deg]

	State _state{0.f * metre_per_second, 0.f * metre_per_second, 1.f *mp_units::one};
	Covariance _P{};		///< state covariance matrix

	Speed _tas_innov{};	///< true airspeed innovation
	SpeedVariance _tas_innov_var{};	///< true airspeed innovation variance

	Sideslip _beta_innov{};	///< sideslip innovation
	SideslipVariance _beta_innov_var{};	///< sideslip innovation variance

	bool _initialised{false};	///< True: filter has been initialised

	WindPsd _wind_psd{0.01f * WindPsd::reference};	///< wind process noise power spectral density, (0.1 m/s^2/sqrt(Hz))^2
	static constexpr ScalePsd kTasScalePsd{1e-8f * ScalePsd::reference};	///< true airspeed scale process noise power spectral density, (0.0001 1/s/sqrt(Hz))^2
	SpeedVariance _tas_var{1.96f * SpeedVariance::reference};		///< true airspeed measurement noise variance, (1.4 m/s)^2
	static constexpr SideslipVariance kBetaVar{0.0225f * SideslipVariance::reference};	///< sideslip measurement noise variance, (0.15 rad)^2
	static constexpr uint8_t kTasGate{4};	///< airspeed fusion gate size (SD)
	static constexpr uint8_t kBetaGate{1};	///< sideslip fusion gate size (SD)

	Unitless _scale_init{1.f *mp_units::one};

	uint64_t _time_last_airspeed_fuse = 0;	///< timestamp of last airspeed fusion
	uint64_t _time_last_beta_fuse = 0;	///< timestamp of last sideslip fusion
	uint64_t _time_last_update = 0;		///< timestamp of last covariance prediction
	uint64_t _time_initialised = 0;         ///< timestamp when estimator is initialised

	static constexpr float kTASScalePSDMultiplier = 100;
	static constexpr hrt_abstime kTASScaleFastLearnTime = 300_s;

	bool _wind_estimator_reset = false; ///< wind estimator was reset in this cycle

	// initialise state and state covariance matrix
	bool initialise(const matrix::Vector3f &velI, const float hor_vel_variance, const float heading_rad,
			const float tas_meas = NAN, const float tas_variance = NAN);

	// x += K innov, P -= K H P: the correction of a scalar fusion, from the generated H and K
	template <typename Observation>
	void correct(const matrix::Matrix<float, 1, 3> &H, const matrix::Matrix<float, 3, 1> &K, Observation innov);

	void run_sanity_checks();

	// return the square of two floating point numbers
	static constexpr float sq(float var) { return var * var; }
};
