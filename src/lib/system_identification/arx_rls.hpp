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
 * @file arx_rls.hpp
 * @brief Efficient recursive weighted least-squares algorithm without matrix inversion
 *
 * Assumes an ARX (autoregressive) model:
 * A(q^-1)y(k) = q^-d * B(q^-1)u(k) + A(q^-1)e(k)
 *
 * with:
 * q^-i  backward shift operator
 * A(q^-1) = 1 + a_1*q^-1 +...+ a_n*q^-n
 * B(q^-1) = b_0 + b_1*q^-1...+ b_m*q^-m
 * n  order of A(q^-1)
 * m  order of B(q^-1)
 * d  delay
 * u  input of the system
 * y  output of the system
 * e  white noise input
 *
 * References:
 * - Identification de systemes dynamiques, D.Bonvin and A.Karimi, epfl, 2011
 *
 * @author Mathieu Bresciani <mathieu@auterion.com>
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <matrix/matrix/math.hpp>
#include <px4_typed_linear_algebra.hpp>

namespace arx_rls
{

template<auto Reference>
using Quantity = px4_typed_linear_algebra::Quantity<Reference>;
using Ratio = Quantity<mp_units::one>;

template<std::size_t, typename Type>
using Always = Type;

// The typed column vector of N elements of A followed by K elements of B.
template<typename A, typename B, std::size_t N, std::size_t K,
	 typename = std::make_index_sequence<N>, typename = std::make_index_sequence<K>>
struct split_vector;

template<typename A, typename B, std::size_t N, std::size_t K, std::size_t... I, std::size_t... J>
struct split_vector<A, B, N, K, std::index_sequence<I...>, std::index_sequence<J...>> {
	using type = px4_typed_linear_algebra::Vector<Always<I, A>..., Always<J, B>...>;
};

template<typename A, typename B, std::size_t N, std::size_t K>
using SplitVector = typename split_vector<A, B, N, K>::type;

// Calls `function` with each index of [0, Count) as a compile-time constant.
template<std::size_t Count, typename Function>
constexpr void forEachIndex(Function &&function)
{
	[&]<std::size_t... Index>(std::index_sequence<Index...>) {
		(function(std::integral_constant<std::size_t, Index> {}), ...);
	}(std::make_index_sequence<Count> {});
}

} // namespace arx_rls

// The output y and the input u are of the quantity types Output and Input:
// the coefficients a_i are ratios, the b_i in Output per Input, and the
// covariance of the coefficients, normalized by the variance of the output
// noise, in their products per squared Output.
template<size_t N, size_t M, size_t D, typename Output = arx_rls::Ratio, typename Input = arx_rls::Ratio>
class ArxRls final
{
	using Ratio = arx_rls::Ratio;
	using Gain = decltype(Output{} / Input{});
	// [a_1 .. a_n b_0 .. b_m]'
	using Coefficients = arx_rls::SplitVector < Ratio, Gain, N, M + 1 >;
	// [-y(k-1) .. -y(k-n) u(k-d) .. u(k-d-m)]'
	using Design = arx_rls::SplitVector < Output, Input, N, M + 1 >;
	// The coefficients per unit of output: the update's gain.
	using CoefficientsPerOutput = arx_rls::SplitVector < decltype(1.f / Output{}), decltype(1.f / Input{}), N, M + 1 >;
	using Covariance = px4_typed_linear_algebra::Matrix<CoefficientsPerOutput, CoefficientsPerOutput>;
	static constexpr size_t kSize{N + M + 1};

public:
	ArxRls()
	{
		static_assert(N >= M, "The transfer function needs to be proper");

		reset();
	}

	~ArxRls() = default;

	void setForgettingFactor(float time_constant, float dt) { _lambda = 1.f - dt / time_constant; }
	void setForgettingFactor(float lambda) { _lambda = lambda; }

	/*
	 * return the vector of estimated parameters
	 * [a_1 .. a_n b_0 .. b_m]', the b_i in Output per Input
	 */
	matrix::Vector<float, kSize> getCoefficients() const
	{
		matrix::Vector<float, kSize> value;
		arx_rls::forEachIndex<kSize>([&](auto i) {
			value(i) = _theta_hat.template at<i>().numerical_value_in(std::remove_cvref_t<decltype(_theta_hat.template at<i>())>::unit);
		});
		return value;
	}

	// The variances of the coefficients normalized by the output noise
	// variance: of the a_i in 1/Output^2, of the b_i in 1/Input^2.
	matrix::Vector<float, kSize> getVariances() const
	{
		matrix::Vector<float, kSize> value;
		arx_rls::forEachIndex<kSize>([&](auto i) {
			value(i) = _P.template at<i, i>().numerical_value_in(std::remove_cvref_t<decltype(_P.template at<i, i>())>::unit);
		});
		return value;
	}

	float getInnovation() const { return _innovation.numerical_value_in(Output::unit); }
	const matrix::Vector<float, kSize> &getDiffEstimate() const { return _diff_theta_hat; }

	void reset(const matrix::Vector<float, kSize> &theta_init = {})
	{
		_P = Covariance{};
		_theta_hat = Coefficients{};

		arx_rls::forEachIndex<kSize>([&](auto i) {
			using Variance = std::remove_cvref_t<decltype(_P.template at<i, i>())>;
			using Coefficient = std::remove_cvref_t<decltype(_theta_hat.template at<i>())>;
			_P.template at<i, i>(10e3f * Variance::reference);
			_theta_hat.template at<i>(theta_init(i) * Coefficient::reference);
		});

		_diff_theta_hat.setZero();

		for (size_t i = 0; i < M + D + 1; i++) {
			_u[i] = Input{};
		}

		for (size_t i = 0; i < N + 1; i++) {
			_y[i] = Output{};
		}

		_nb_samples = 0;
		_innovation = Output{};
	}

	void update(float u, float y)
	{
		const Coefficients theta_prev = _theta_hat;

		addInputOutput(u * Input::reference, y * Output::reference);

		if (!isBufferFull()) {
			// Do not start to update the RLS algorithm when the
			// buffer still contains zeros
			return;
		}

		const Design phi = constructDesignVector();
		const auto phi_t = fcarouge::transposed(phi);

		// Divisions by a scalar multiply by its reciprocal, as the matrix library does.
		const Ratio scale{_lambda *mp_units::one + (phi_t *_P * phi).template at<>()};
		_P = Covariance{(_P - _P *phi *phi_t *_P * (1.f / scale)) *(1.f / _lambda)};
		_innovation = Output{_y[N] - (phi_t * _theta_hat).template at<>()};
		_theta_hat = Coefficients{_theta_hat + _P *phi * _innovation};

		arx_rls::forEachIndex<kSize>([&](auto i) {
			using Coefficient = std::remove_cvref_t<decltype(_theta_hat.template at<i>())>;
			_diff_theta_hat(i) = fabsf((_theta_hat.template at<i>() - theta_prev.template at<i>()).numerical_value_in(
							   Coefficient::unit));
		});
	}

private:
	void addInputOutput(Input u, Output y)
	{
		shiftRegisters();
		_u[M + D] = u;
		_y[N] = y;

		if (!isBufferFull()) {
			_nb_samples++;
		}
	}

	void shiftRegisters()
	{
		for (size_t i = 0; i < N; i++) {
			_y[i] = _y[i + 1];
		}

		for (size_t i = 0; i < (M + D); i++) {
			_u[i] = _u[i + 1];
		}
	}

	bool isBufferFull() const { return _nb_samples > (M + N + D); }

	Design constructDesignVector() const
	{
		Design phi;

		arx_rls::forEachIndex<N>([&](auto i) {
			phi.template at<i>(-_y[N - i - 1]);
		});

		arx_rls::forEachIndex < M + 1 > ([&](auto j) {
			phi.template at < N + j > (_u[M - j]);
		});

		return phi;
	}

	Covariance _P;
	Coefficients _theta_hat;
	matrix::Vector<float, kSize> _diff_theta_hat;
	Output _innovation{};
	Input _u[M + D + 1] {};
	Output _y[N + 1] {};
	unsigned _nb_samples{0};
	float _lambda{1.f};
};
