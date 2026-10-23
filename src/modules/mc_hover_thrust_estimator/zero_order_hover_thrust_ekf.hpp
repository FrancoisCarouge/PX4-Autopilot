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
 * @file zero_order_hover_thrust_ekf.hpp
 * @brief Implementation of a single-state hover thrust estimator
 *
 * state: hover thrust (Th)
 * Vertical acceleration is used as a measurement and the current
 * thrust (T[k]) is used in the measurement model.
 *
 * The state is noise driven: Transition matrix A = 1
 * x[k+1] = Ax[k] + v with v ~ N(0, Q)
 * y[k] = h(u, x) + w with w ~ N(0, R)
 *
 * Where the measurement model and corresponding partial derivative (w.r.t. Th) are:
 * h(u, x)[k] = g * T[k] / Th[k] - g
 * H[k] = -g * T[k] / Th[k]**2
 *
 * The measurmement noise variance R is continuously estimated using the residual after each
 * measurement update with the following equation:
 * R = (1 - alpha) * R + alpha * (z^2 + P * H^2)
 * Where z is the residual and alpha a forgetting factor between 0 and 1
 * (see S.Akhlaghi et al., Adaptive adjustment of noise covariance in Kalman filter for dynamic state estimation, 2017)
 *
 * During the measurment update step, the Normalized Innovation Squared (NIS) is checked
 * and the measurement is rejected if larger than the gate size.
 * A recovery logic is triggered when too many measurements are continuously rejected and
 * consists of a measurement variance reset and a state variance boost.
 *
 * @author Mathieu Bresciani 	<brescianimathieu@gmail.com>
 */

#pragma once

#include <matrix/matrix/math.hpp>
#include <geo/geo.h>
#include <mathlib/mathlib.h>

#include <fcarouge/px4_kalman.hpp>
#include <mp_units/fcarouge/kalman_mp_units.hpp>
#include <mp-units/framework.h>
#include <mp-units/systems/si.h>

namespace zero_order_hover_thrust_ekf
{

template <auto Reference>
using Quantity = mp_units::quantity<Reference, float>;

inline constexpr auto metre_per_second_squared = mp_units::si::metre / (mp_units::si::second * mp_units::si::second);

/// a thrust normalized to the vehicle's maximum, and its variance
using Thrust = Quantity<mp_units::one>;
/// the vertical acceleration, the observation, and its variance
using Acceleration = Quantity<metre_per_second_squared>;
using AccelerationVariance = Quantity<metre_per_second_squared *metre_per_second_squared>;
/// the derivative of the observation with respect to the hover thrust
using AccelerationPerThrust = decltype(Acceleration{} / Thrust{});

inline constexpr Acceleration gravity{CONSTANTS_ONE_G * metre_per_second_squared};

/// h(u, x) = g * T / Th - g, with the thrust T as the update argument
inline Acceleration predictedAccZ(const Thrust &hover_thrust, const Thrust &thrust)
{
	return gravity * thrust / hover_thrust - gravity;
}

/// H = -g * T / Th^2
inline AccelerationPerThrust observationJacobian(const Thrust &hover_thrust, const Thrust &thrust)
{
	return -gravity * thrust / (hover_thrust * hover_thrust);
}

inline auto makeFilter()
{
	using namespace fcarouge;
	return kalman{
		state{Thrust{0.5f * mp_units::one}},
		output<Acceleration>,
		// the hover thrust uncertainty variance (thrust^2)
		estimate_uncertainty{Thrust{0.01f * mp_units::one}},
		// set before each prediction
		process_uncertainty{Thrust{0.f *mp_units::one}},
		// set before each update
		output_uncertainty{AccelerationVariance{5.f *AccelerationVariance::reference}},
		output_model{
			[](const Thrust & hover_thrust, const Thrust & thrust) -> AccelerationPerThrust {
				return observationJacobian(hover_thrust, thrust);
			}},
		// the state is noise driven: the transition is the identity
		transition{[](const Thrust & hover_thrust) -> Thrust { return hover_thrust; }},
		observation{
			[](const Thrust & hover_thrust, const Thrust & thrust) -> Acceleration {
				return predictedAccZ(hover_thrust, thrust);
			}},
		update_types<Thrust>,
		prediction_types<>};
}

} // namespace zero_order_hover_thrust_ekf

class ZeroOrderHoverThrustEkf
{
public:
	ZeroOrderHoverThrustEkf() = default;
	~ZeroOrderHoverThrustEkf() = default;

	void resetAccelNoise() { _acc_var = 5.f * AccelerationVariance::reference; };

	void predict(float _dt);
	void fuseAccZ(float acc_z, float thrust);

	void setHoverThrust(float hover_thrust) { _filter.x(Thrust{math::constrain(hover_thrust, _hover_thr_min, _hover_thr_max) * mp_units::one}); }
	void setProcessNoiseStdDev(float process_noise) { _process_var = process_noise * process_noise; }
	void setMeasurementNoiseScale(float scale) { _acc_var_scale = scale * scale; }
	void setHoverThrustStdDev(float hover_thrust_noise) { _filter.p(Thrust{hover_thrust_noise *hover_thrust_noise * mp_units::one}); }
	void setAccelInnovGate(float gate_size) { _gate_size = gate_size; }
	void setMinHoverThrust(float hover_thrust_min) { _hover_thr_min = math::constrain(hover_thrust_min, 0.1f, 0.8f); }
	void setMaxHoverThrust(float hover_thrust_max) { _hover_thr_max = math::constrain(hover_thrust_max, 0.2f, 0.9f); }

	float getHoverThrustEstimate() const { return _filter.x().numerical_value_in(mp_units::one); }
	float getHoverThrustEstimateVar() const { return _filter.p().numerical_value_in(mp_units::one); }
	float getInnovation() const { return _innov.numerical_value_in(metre_per_second_squared); }
	float getInnovationVar() const { return _innov_var.numerical_value_in(AccelerationVariance::unit); }
	float getInnovationTestRatio() const { return _innov_test_ratio; }
	float getAccelNoiseVar() const { return _acc_var.numerical_value_in(AccelerationVariance::unit); }

private:
	using Thrust = zero_order_hover_thrust_ekf::Thrust;
	using Acceleration = zero_order_hover_thrust_ekf::Acceleration;
	using AccelerationVariance = zero_order_hover_thrust_ekf::AccelerationVariance;
	using AccelerationPerThrust = zero_order_hover_thrust_ekf::AccelerationPerThrust;
	static constexpr auto metre_per_second_squared = zero_order_hover_thrust_ekf::metre_per_second_squared;

	/// the hover thrust and its variance
	decltype(zero_order_hover_thrust_ekf::makeFilter()) _filter{zero_order_hover_thrust_ekf::makeFilter()};
	float _hover_thr_min{0.1f};
	float _hover_thr_max{0.9f};

	float _gate_size{3.f};
	float _process_var{12.5e-6f}; ///< Hover thrust process noise variance (thrust^2/s^2)
	AccelerationVariance _acc_var{5.f *AccelerationVariance::reference};  ///< Acceleration variance
	float _acc_var_scale{1.f}; ///< Multiplicator of the measurement variance, used to decrease sensivity
	float _dt{0.02f};

	Acceleration _innov{}; ///< Measurement innovation
	AccelerationVariance _innov_var{}; ///< Measurement innovation variance
	float _innov_test_ratio{0.f}; ///< Noramlized Innovation Squared test ratio

	Acceleration _residual_lpf{}; ///< used to remove the constant bias of the residual
	float _signed_innov_test_ratio_lpf{}; ///< used as a delay to trigger the recovery logic

	AccelerationVariance computeInnovVar(AccelerationPerThrust H) const;
	Acceleration computeInnov(Acceleration acc_z, Thrust thrust) const;

	/*
	 * Compute the ratio between the Normalized Innovation Squared (NIS)
	 * and its maximum gate size. Use isTestRatioPassing to know if the
	 * measurement should be fused or not.
	 */
	float computeInnovTestRatio(Acceleration innov, AccelerationVariance innov_var) const;
	bool isTestRatioPassing(float innov_test_ratio) const;

	void constrainState();
	bool isLargeOffsetDetected() const;

	void bumpStateVariance();
	void updateLpf(Acceleration residual, float signed_innov_test_ratio);
	void updateMeasurementNoise(Acceleration residual, AccelerationPerThrust H);

	static constexpr float _noise_learning_time_constant = 2.f; ///< in seconds
	static constexpr float _lpf_time_constant = 1.f; ///< in seconds
};
