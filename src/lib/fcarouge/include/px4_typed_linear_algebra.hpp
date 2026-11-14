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
 * PX4's adapter between the Kalman, TypedLinearAlgebra, and
 * SparseLinearAlgebra libraries and mp-units: typed vectors and matrices of
 * float quantities over sparse storage, for the filters that use them.
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include <fcarouge/kalman.hpp>
#include <fcarouge/sparse_linear_algebra.hpp>
#include <fcarouge/typed_linear_algebra.hpp>
#include <mp-units/framework/quantity.h>
#include <mp-units/systems/si.h>

namespace fcarouge
{

// A quantity element is stored as its numerical value in its own unit, and a
// stored value is read back as a quantity of that unit.
template <typename To, mp_units::Quantity From>
struct element_caster<To, From> {
	[[nodiscard]] static constexpr auto operator()(From value) -> To
	{
		static_assert(std::same_as<typename From::rep, std::remove_cvref_t<To>>);
		return value.numerical_value_in(From::unit);
	}
};

template <mp_units::Quantity To, typename From>
struct element_caster<To, From> {
	[[nodiscard]] static constexpr auto operator()(From value) -> To
	{
		static_assert(std::same_as<typename To::rep, std::remove_cvref_t<From>>);
		return value * To::reference;
	}
};

// A quantity reference cannot be materialized out of a stored number: a
// reference read gives a constant value, so that a lost write does not compile.
template <mp_units::Quantity To, typename From>
struct element_caster<To &, From &> {
	[[nodiscard]] static constexpr auto operator()(From value) -> const To
	{
		static_assert(std::same_as<typename To::rep, std::remove_cvref_t<From>>);
		return value * To::reference;
	}
};

// A unit magnitude, which a 1x1 typed matrix can carry through mp-units
// arithmetic: read as its value, and scaled by.
template <typename To, mp_units::UnitMagnitude Magnitude>
struct element_caster<To, Magnitude> {
	[[nodiscard]] static constexpr auto operator()(Magnitude value) -> To
	{
		return get_value<To>(value);
	}
};

template <typename Type, mp_units::UnitMagnitude Magnitude>
struct multiplies<Type, Magnitude> {
	[[nodiscard]] static constexpr auto operator()(const Type &lhs, const Magnitude &rhs) -> Type;
};

// Kalman names the matrices of a filter by dividing its column vectors, which
// SparseLinearAlgebra rejects for a non-square denominator. Declared only: it
// names the type, and a run-time use fails to link.
template <typename Type, auto Lhs, auto Rhs>
requires(Lhs.columns == 1 && Rhs.columns == 1 && Rhs.rows > 1)
auto operator/(const sparse_matrix<Type, Lhs> &, const sparse_matrix<Type, Rhs> &)
-> sparse_matrix<Type, sparse::dense<Lhs.rows, Rhs.rows>>;

} // namespace fcarouge

namespace fcarouge::kalman_filter::internal
{

// The transposes: of a sparse storage, through its free transposed(); of a
// typed matrix, its storage's with the indexes exchanged; of a single quantity,
// that quantity.
template <typename Type, auto Pattern>
struct transposes<sparse_matrix<Type, Pattern>> {
	[[nodiscard]] static constexpr auto operator()(const sparse_matrix<Type, Pattern> &value)
	{
		return transposed(value);
	}
};

template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
struct transposes<typed_matrix<Matrix, RowIndexes, ColumnIndexes>> {
	[[nodiscard]] static constexpr auto operator()(const typed_matrix<Matrix, RowIndexes, ColumnIndexes> &value)
	{
		return typed_matrix<evaluate<transpose<Matrix>>, ColumnIndexes, RowIndexes> {t(value.data())};
	}
};

template <auto Reference, typename Representation>
struct transposes<mp_units::quantity<Reference, Representation>> {
	[[nodiscard]] static constexpr auto operator()(const mp_units::quantity<Reference, Representation> &value)
	{
		return value;
	}
};

// The zero and one: of a sparse storage, zero initialized and ones on its
// stored diagonal; of a typed matrix, of its storage; of a single quantity.
template <typename Type, auto Pattern>
inline constexpr sparse_matrix<Type, Pattern> zero<sparse_matrix<Type, Pattern>> {};

template <typename Type, auto Pattern>
inline constexpr sparse_matrix<Type, Pattern> one<sparse_matrix<Type, Pattern>> {[] {
		sparse_matrix<Type, Pattern> value;

		for (std::size_t i{0}; i < Pattern.rows && i < Pattern.columns; ++i)
		{
			if (Pattern.stores(i, i)) {
				value.values[Pattern.index_of(i, i)] = Type{1};
			}
		}

		return value;
	}()};

template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline constexpr typed_matrix<Matrix, RowIndexes, ColumnIndexes>
zero<typed_matrix<Matrix, RowIndexes, ColumnIndexes>> {zero<Matrix>};

template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline constexpr typed_matrix<Matrix, RowIndexes, ColumnIndexes>
one<typed_matrix<Matrix, RowIndexes, ColumnIndexes>> {one<Matrix>};

template <typename Representation, auto Reference>
inline constexpr mp_units::quantity<Reference, Representation>
one<mp_units::quantity<Reference, Representation>> {Representation{1}, Reference};

} // namespace fcarouge::kalman_filter::internal

namespace px4_typed_linear_algebra
{

template <auto Reference>
using Quantity = mp_units::quantity<Reference, float>;

using Length = Quantity<mp_units::si::metre>;
using Speed = Quantity<mp_units::si::metre / mp_units::si::second>;
using Time = Quantity<mp_units::si::second>;

// A column vector of the given element types.
template <typename... Types>
using Vector = fcarouge::typed_column_vector <
	       fcarouge::sparse_matrix<float, fcarouge::sparse::dense<sizeof...(Types), 1>>, Types... >;

// A dense matrix indexed by the rows of a column vector and the columns of another.
template <typename Rows, typename Columns>
using Matrix = fcarouge::typed_matrix <
	       fcarouge::sparse_matrix<float, fcarouge::sparse::dense<Rows::rows, Columns::rows>>,
	       typename Rows::row_indexes, typename Columns::row_indexes >;

// The type of an element of a typed matrix.
template <typename Typed, std::size_t Row, std::size_t Column = 0>
using Element = fcarouge::tla::element_at<Typed, Row, Column>;

} // namespace px4_typed_linear_algebra
