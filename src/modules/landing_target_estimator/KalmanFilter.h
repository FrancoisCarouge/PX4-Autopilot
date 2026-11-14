/****************************************************************************
 *
 *   Copyright (c) 2013-2018 PX4 Development Team. All rights reserved.
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
/*
 * @file KalmanFilter.h
 * Simple Kalman Filter for variable gain low-passing
 *
 * Constant velocity model. Prediction according to
 * x_{k+1} = A * x_{k} + G * acc
 * with A = [1 dt; 0 1], G = [dt^2/2; dt]
 *
 * Update with a direct measurement of the first state:
 * H = [1 0]
 *
 * A typed Kalman filter: the position in metres, the velocity in metres per
 * second, every covariance element in its own unit.
 *
 * @author Nicolas de Palezieux (Sunflower Labs) <ndepal@gmail.com>
 *
 */

#pragma once

#include <px4_typed_linear_algebra.hpp>

namespace landing_target_estimator
{

class KalmanFilter
{
public:
	using Time = px4_typed_linear_algebra::Time;
	using Length = px4_typed_linear_algebra::Length;
	using Speed = px4_typed_linear_algebra::Speed;
	using Acceleration = px4_typed_linear_algebra::Quantity<mp_units::si::metre / mp_units::si::second / mp_units::si::second>;

	// The target: its position and velocity relative to the vehicle.
	using State = px4_typed_linear_algebra::Vector<Length, Speed>;
	using Covariance = px4_typed_linear_algebra::Matrix<State, State>;
	// The relative acceleration.
	using Input = px4_typed_linear_algebra::Vector<Acceleration>;
	// The measured relative position.
	using Output = px4_typed_linear_algebra::Vector<Length>;
	using OutputVariance = px4_typed_linear_algebra::Matrix<Output, Output>;

	// The variances of the position [m²], of the velocity [(m/s)²], and of the
	// acceleration [(m/s²)²].
	using PositionVariance = px4_typed_linear_algebra::Element<Covariance, 0, 0>;
	using VelocityVariance = px4_typed_linear_algebra::Element<Covariance, 1, 1>;
	using AccelerationVariance = decltype(Acceleration{} * Acceleration{});

	using Transition = fcarouge::kalman_filter::internal::evaluate <
			   fcarouge::kalman_filter::internal::quotient<State, State >>;
	using InputControl = fcarouge::kalman_filter::internal::evaluate <
			     fcarouge::kalman_filter::internal::quotient<State, Input >>;

	// A = [1 dt; 0 1]
	struct propagation {
		Transition operator()(const Input &acc, const Time &dt, const AccelerationVariance &acc_unc) const;
	};

	// G = [dt^2/2; dt]
	struct noise_model {
		InputControl operator()(const Time &dt, const AccelerationVariance &acc_unc) const;
	};

	// Q = G * G^T * acc_unc
	struct process_noise {
		Covariance operator()(const State &x, const Time &dt, const AccelerationVariance &acc_unc) const;
	};

	// H = [1 0]: the filter's default output model, the first row of the
	// identity.
	using Filter = decltype(fcarouge::kalman{
		fcarouge::state{State{}}, fcarouge::output<Output>, fcarouge::input<Input>,
		fcarouge::estimate_uncertainty{Covariance{}}, fcarouge::process_uncertainty{process_noise{}},
		fcarouge::output_uncertainty{OutputVariance{}},
		fcarouge::state_transition{propagation{}}, fcarouge::input_control{noise_model{}},
		fcarouge::prediction_types<Time, AccelerationVariance>, fcarouge::simple_form});

	/**
	 * Initialize filter state, only specifying diagonal covariance elements
	 * @param position     initial position
	 * @param velocity     initial velocity
	 * @param position_var initial variance of the position
	 * @param velocity_var initial variance of the velocity
	 */
	void init(Length position, Speed velocity, PositionVariance position_var, VelocityVariance velocity_var);

	/**
	 * Predict the state with an external acceleration estimate
	 * @param dt            Time delta since last state change
	 * @param acc           Acceleration estimate
	 * @param acc_unc       Variance of acceleration estimate
	 */
	void predict(Time dt, Acceleration acc, AccelerationVariance acc_unc);

	/**
	 * Update the state estimate with a measurement
	 * @param meas    position measurement
	 * @param measUnc measurement variance
	 * @return update success (measurement not rejected)
	 */
	bool update(Length meas, PositionVariance measUnc);

	Length position() const { return _filter.x().at<0>(); }
	Speed velocity() const { return _filter.x().at<1>(); }
	PositionVariance position_variance() const { return _filter.p().at<0, 0>(); }
	VelocityVariance velocity_variance() const { return _filter.p().at<1, 1>(); }

	// The measurement innovation and its variance of the last update call.
	Length innovation() const { return _residual; }
	PositionVariance innovation_variance() const { return _innovCov; }

private:
	Filter _filter{
		fcarouge::state{State{}}, fcarouge::output<Output>, fcarouge::input<Input>,
		fcarouge::estimate_uncertainty{Covariance{}}, fcarouge::process_uncertainty{process_noise{}},
		fcarouge::output_uncertainty{OutputVariance{}},
		fcarouge::state_transition{propagation{}}, fcarouge::input_control{noise_model{}},
		fcarouge::prediction_types<Time, AccelerationVariance>, fcarouge::simple_form};
	Length _residual{}; // residual of last measurement update
	PositionVariance _innovCov{}; // innovation covariance of last measurement update
};

} // namespace landing_target_estimator
