/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
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
 * @file px4_typed_linear_algebra.hpp
 *
 * The FrancoisCarouge TypedLinearAlgebra library for PX4, with its mp-units
 * support, over a fixed size matrix backend: include this header, not
 * <fcarouge/typed_linear_algebra.hpp> directly when the backend is not needed.
 * Each element of a typed matrix has its own type, an mp-units quantity for
 * example, so that unit mismatches fail to compile.
 */

#pragma once

#include <cstddef>
#include <tuple>

#include <matrix/matrix/math.hpp>

#include <fcarouge/typed_linear_algebra.hpp>
#include <fcarouge/mp_units.hpp>

namespace fcarouge_px4
{

/**
 * The storage backend of the typed matrices: fixed size, row major, owning,
 * allocation free and usable in constant expressions, convertible to and from
 * PX4's matrix::Matrix.
 */
template <typename T, std::size_t R, std::size_t C>
struct Matrix {
	T v[R * C] {};

	constexpr Matrix() = default;

	explicit Matrix(const matrix::Matrix<T, R, C> &m)
	{
		for (std::size_t i = 0; i < R; i++) {
			for (std::size_t j = 0; j < C; j++) {
				(*this)(i, j) = m(i, j);
			}
		}
	}

	matrix::Matrix<T, R, C> toMatrix() const { return matrix::Matrix<T, R, C> {v}; }

	static constexpr Matrix Zero() { return Matrix{}; }

	static constexpr Matrix Identity()
	{
		Matrix m{};

		for (std::size_t i = 0; i < R && i < C; i++) {
			m(i, i) = T(1);
		}

		return m;
	}

	// element access, as TypedLinearAlgebra expects it
	constexpr T &operator()(std::size_t i, std::size_t j) { return v[i * C + j]; }
	constexpr const T &operator()(std::size_t i, std::size_t j) const { return v[i * C + j]; }
	constexpr T &operator()(std::size_t i) { return v[i]; }
	constexpr const T &operator()(std::size_t i) const { return v[i]; }
	constexpr T &operator[](std::size_t i, std::size_t j) { return v[i * C + j]; }
	constexpr const T &operator[](std::size_t i, std::size_t j) const { return v[i * C + j]; }
	constexpr T &operator[](std::size_t i) { return v[i]; }
	constexpr const T &operator[](std::size_t i) const { return v[i]; }

	constexpr Matrix<T, C, R> transpose() const
	{
		Matrix<T, C, R> m{};

		for (std::size_t i = 0; i < R; i++) {
			for (std::size_t j = 0; j < C; j++) {
				m(j, i) = (*this)(i, j);
			}
		}

		return m;
	}

	friend constexpr Matrix operator-(Matrix a)
	{
		for (T &e : a.v) { e = -e; }

		return a;
	}

	friend constexpr Matrix operator+(Matrix a, const Matrix &b)
	{
		for (std::size_t k = 0; k < R * C; k++) { a.v[k] += b.v[k]; }

		return a;
	}

	friend constexpr Matrix operator-(Matrix a, const Matrix &b)
	{
		for (std::size_t k = 0; k < R * C; k++) { a.v[k] -= b.v[k]; }

		return a;
	}

	friend constexpr Matrix operator*(Matrix a, const T &s)
	{
		for (T &e : a.v) { e *= s; }

		return a;
	}

	friend constexpr Matrix operator*(const T &s, Matrix a) { return a * s; }

	friend constexpr Matrix operator/(Matrix a, const T &s)
	{
		for (T &e : a.v) { e /= s; }

		return a;
	}

	template <std::size_t N>
	friend constexpr Matrix<T, R, N> operator*(const Matrix &a, const Matrix<T, C, N> &b)
	{
		Matrix<T, R, N> m{};

		for (std::size_t i = 0; i < R; i++) {
			for (std::size_t j = 0; j < N; j++) {
				for (std::size_t k = 0; k < C; k++) {
					m(i, j) += a(i, k) * b(k, j);
				}
			}
		}

		return m;
	}

	/**
	 * Right division x = a / b, the x solving x * b = a: a * inverse(b) for a
	 * square b, otherwise the minimum norm or least squares solution. The
	 * typed library divides, for example, a state by a state to form the type
	 * of a transition matrix, and the Kalman filter forms its gain by it.
	 */
	template <std::size_t N>
	friend constexpr Matrix<T, R, N> operator/(const Matrix &a, const Matrix<T, N, C> &b)
	{
		if constexpr(N == C) {
			return solveRight(a, b);

		} else if constexpr(N > C) {
			return solveRight(a, b.transpose() * b) * b.transpose();

		} else {
			return solveRight(a * b.transpose(), b * b.transpose());
		}
	}

	// s / b, the row vector x solving x * b = s, for a column vector b
	friend constexpr Matrix<T, 1, R> operator/(const T &s, const Matrix &b) requires(C == 1)
	{
		Matrix<T, 1, 1> a{};
		a(0, 0) = s;
		return a / b;
	}

private:
	// x * b = a for a square b, by Gaussian elimination with partial pivoting
	// of transpose(b) * transpose(x) = transpose(a)
	template <std::size_t A, std::size_t K>
	static constexpr Matrix<T, A, K> solveRight(const Matrix<T, A, K> &a, const Matrix<T, K, K> &b)
	{
		Matrix<T, K, K> m = b.transpose();
		Matrix<T, K, A> x = a.transpose();
		const auto magnitude = [](const T & e) { return e < T(0) ? -e : e; };

		for (std::size_t c = 0; c < K; c++) {
			std::size_t pivot = c;

			for (std::size_t i = c + 1; i < K; i++) {
				if (magnitude(m(i, c)) > magnitude(m(pivot, c))) {
					pivot = i;
				}
			}

			for (std::size_t j = 0; pivot != c && j < K; j++) {
				const T t = m(c, j);
				m(c, j) = m(pivot, j);
				m(pivot, j) = t;
			}

			for (std::size_t j = 0; pivot != c && j < A; j++) {
				const T t = x(c, j);
				x(c, j) = x(pivot, j);
				x(pivot, j) = t;
			}

			for (std::size_t i = c + 1; i < K; i++) {
				const T factor = m(i, c) / m(c, c);

				for (std::size_t j = c; j < K; j++) {
					m(i, j) -= factor * m(c, j);
				}

				for (std::size_t j = 0; j < A; j++) {
					x(i, j) -= factor * x(c, j);
				}
			}
		}

		for (std::size_t c = K; c-- > 0;) {
			for (std::size_t j = 0; j < A; j++) {
				T sum = x(c, j);

				for (std::size_t k = c + 1; k < K; k++) {
					sum -= m(c, k) * x(k, j);
				}

				x(c, j) = sum / m(c, c);
			}
		}

		return x.transpose();
	}
};

/// a column vector whose i-th element has the i-th type
template <typename T, typename... Types>
using ColumnVector = fcarouge::typed_column_vector<Matrix<T, sizeof...(Types), 1>, Types...>;

/// a row vector whose j-th element has the j-th type
template <typename T, typename... Types>
using RowVector = fcarouge::typed_row_vector<Matrix<T, 1, sizeof...(Types)>, Types...>;

} // namespace fcarouge_px4
