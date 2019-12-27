#ifndef __tuple_helper_utilities_hpp__
#define __tuple_helper_utilities_hpp__

#include <tuple>

#include "variadic_arg_helpers.hpp"

namespace aks
{
	namespace tuple_utils
	{
		// index_sequence implementation since VS2013 doesn't have it yet
		template <size_t... Ints> class index_sequence {
		public:
			static size_t size() { return sizeof...(Ints); }
		};

		template <size_t Start, typename Indices, size_t End>
		struct make_index_sequence_impl;

		template <size_t Start, size_t... Indices, size_t End>
		struct make_index_sequence_impl<Start, index_sequence<Indices...>, End> {
			typedef typename make_index_sequence_impl<
				Start + 1, index_sequence<Indices..., Start>, End>::type type;
		};

		template <size_t End, size_t... Indices>
		struct make_index_sequence_impl<End, index_sequence<Indices...>, End> {
			typedef index_sequence<Indices...> type;
		};

		template <size_t N>
		using make_index_sequence =
			typename make_index_sequence_impl<0, index_sequence<>, N>::type;

		/////////////////////////////////////////////////////////

		template<typename T, typename... Ts>
		T head(std::tuple<T, Ts...>const& t)
		{
			return std::get<0>(t);
		}

		template<typename T, size_t... Is>
		auto tail_impl(T const& t, index_sequence<Is...>) -> decltype(std::make_tuple(std::get<Is + 1>(t)...))
		{
			return std::make_tuple(std::get<Is + 1>(t)...);
		}

		template<typename T, typename... Ts>
		std::tuple<Ts...> tail(std::tuple<T, Ts...>const& t)
		{
			return tail_impl(t, make_index_sequence<std::tuple_size<std::tuple<T, Ts...>>::value - 1>());
		}

		/////////////////////////////////////////////////////////

		template<typename... Ts, typename F, size_t... Is>
		auto map_impl(std::tuple<Ts...> const& t, F func, index_sequence<Is...>) -> decltype(std::make_tuple(func(std::get<Is>(t))...))
		{
			return std::make_tuple(func(std::get<Is>(t))...);
		}

		template<typename... Ts, typename F>
		auto map(std::tuple<Ts...> const& t, F func) -> decltype(map_impl(t, func, make_index_sequence<sizeof...(Ts)>()))
		{
			return map_impl(t, func, make_index_sequence<sizeof...(Ts)>());
		}

		/////////////////////////////////////////////////////////

		template<typename T, typename... Ts, typename F>
		F for_each(std::tuple<T, Ts...> const& t, F func)
		{
			func(head(t));
			return for_each(tail(t), func);
		}

		template<typename F>
		F for_each(std::tuple<> const& t, F func)
		{
			return func;
		}

		template<typename T, typename... Ts, size_t... Is, typename Op>
		T reduce_impl(std::tuple<T, Ts...>const& t, index_sequence<Is...>, Op)
		{
			return variadic_arg_helpers::reduce<Op>::apply(std::get<Is>(t)...);
		}

		template<typename T, typename... Ts, typename Op>
		T reduce(std::tuple<T, Ts...>const& t, Op op)
		{
			return reduce_impl(t, make_index_sequence<std::tuple_size<std::tuple<T, Ts...>>::value>(), op);
		}

		//////////////////////////////////////////////////////////////

		namespace detail {
			template<typename T, size_t N>
			struct tuple_of_length
			{
				typedef std::tuple<T> current_type;
				typedef tuple_of_length<T, N - 1> internal_type;
				typedef decltype(std::tuple_cat(std::declval<current_type>(), std::declval<typename internal_type::type>())) type;
			};

			template<typename T>
			struct tuple_of_length<T, 1>
			{
				typedef std::tuple<T> type;
			};
		}

		template<typename T, size_t N>
		using tuple_of_length = typename detail::tuple_of_length<T, N>::type;
	}
}

#endif // !__tuple_helper_utilities_hpp__