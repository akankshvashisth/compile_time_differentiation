#ifndef __compile_time_differentiation_hpp__
#define __compile_time_differentiation_hpp__

#pragma warning( disable:4503 )

#include <tuple>
#include <type_traits>

namespace aks
{
	template<size_t N>
	struct variable
	{
		static const size_t dim = N + 1;
		template<typename... Ts> auto operator()(Ts... ts) const
		{
			static_assert(dim <= sizeof...(Ts), "dimension insufficient"); return std::get<N>(std::make_tuple(ts...));
		}
	};

	template<typename T> struct is_variable { enum { value = false }; };
	template<size_t N>   struct is_variable< variable<N> > { enum { value = true }; };

	template<typename T, typename U, typename Op>
	struct binary
	{
		static const size_t dim = T::dim > U::dim ? T::dim : U::dim;
		binary(T a, U b) :t(a), u(b) {}
		template<typename... Args> auto operator()(Args... args) const { static_assert(dim <= sizeof...(Args), "dimension insufficient"); return op(t(args...), u(args...)); }
		template<typename... Args> auto operator()(Args... args) { static_assert(dim <= sizeof...(Args), "dimension insufficient"); return op(t(args...), u(args...)); }
		T t; U u; Op op;
	};

	template<typename T>                          struct is_binary { enum { value = false }; };
	template<typename T, typename U, typename Op> struct is_binary<binary<T, U, Op>> { enum { value = true }; };

	template<typename T, typename Op>
	struct unary
	{
		static const size_t dim = T::dim;
		unary(T a) :t(a) {}
		template<typename... Args> auto operator()(Args... args) const { static_assert(dim <= sizeof...(Args), "dimension insufficient"); return op(t(args...)); }
		template<typename... Args> auto operator()(Args... args) { static_assert(dim <= sizeof...(Args), "dimension insufficient"); return op(t(args...)); }
		T t; Op op;
	};

	template<typename T>              struct is_unary { enum { value = false }; };
	template<typename T, typename Op> struct is_unary<unary<T, Op>> { enum { value = true }; };

	template<typename T>
	struct constant
	{
		static const size_t dim = 0;
		typedef T value_type;
		constant(T a) :t(a) {}
		template<typename... Ts> auto operator()(Ts...) const { return t; }

		//operator T() { return t; }

		T t;
	};

	template<typename T>
	auto make_constant(T t) { return constant<T>(t); }

	template<typename T> struct is_constant_type { enum { value = false }; };
	template<typename T> struct is_constant_type<constant<T>> { enum { value = true }; };

	template<typename T> struct is_constant : std::integral_constant<bool, !is_variable<T>::value && !is_binary<T>::value && !is_unary<T>::value> {};
	template<typename T> struct is_not_constant : std::integral_constant<bool, !is_constant<T>::value> {};

	template<typename T> struct is_independent : std::integral_constant<bool, !is_variable<T>::value && !is_binary<T>::value && !is_unary<T>::value && !is_constant_type<T>::value> {};
	template<typename T> struct is_not_independent : std::integral_constant<bool, !is_independent<T>::value> {};

#define DEFINE_BINARY_OP_INFIX(NAME, OP)                                                                                                                                                                             \
    struct NAME { template<typename T, typename U> auto operator()( T t, U u ) const { return t OP u; } };                                                                                                           \
    template<typename T, typename U> typename std::enable_if<is_not_independent<T>::value && is_not_independent<U>::value && is_not_constant<T>::value && is_not_constant<U>::value, binary<T,U, NAME >>::type operator OP ( T t, U u ) { return binary<T,U, NAME >(t,u); }                 \
    template<typename T, typename U> typename std::enable_if<is_not_constant<T>::value && is_independent<U>::value, binary<T,constant<U>, NAME >>::type operator OP ( T t, U u ) { return binary<T,constant<U>, NAME >(t,u); } \
    template<typename T, typename U> typename std::enable_if<is_independent<T>::value && is_not_constant<U>::value, binary<constant<T>,U, NAME >>::type operator OP ( T t, U u ) { return binary<constant<T>,U, NAME >(t,u); } \
    template<typename T, typename U> auto operator OP ( constant<T> t, constant<U> u ) { return make_constant(t.t OP u.t); } \
    template<typename T, typename U> auto operator OP ( T t, constant<U> u ) -> typename std::enable_if<is_not_independent<T>::value, decltype(( t OP u.t  ))>::type { return (t OP u.t  ); } \
    template<typename T, typename U> auto operator OP ( constant<T> t, U u ) -> typename std::enable_if<is_not_independent<U>::value, decltype(( t.t OP u  ))>::type { return (t.t OP u  ); } \
    template<typename T, typename U> auto operator OP ( T t, constant<U> u ) -> typename std::enable_if<is_independent<T>::value, decltype(make_constant( t OP u.t  ))>::type { return make_constant(t OP u.t  ); } \
    template<typename T, typename U> auto operator OP ( constant<T> t, U u ) -> typename std::enable_if<is_independent<U>::value, decltype(make_constant( t.t OP u  ))>::type { return make_constant(t.t OP u  ); } \

	DEFINE_BINARY_OP_INFIX(add, +);
	DEFINE_BINARY_OP_INFIX(sub, -);
	DEFINE_BINARY_OP_INFIX(mul, *);
	DEFINE_BINARY_OP_INFIX(dvd, / );
	DEFINE_BINARY_OP_INFIX(mod, %);

	struct negation { template<typename T> auto operator()(T t) { return -t; } };
	template<typename T> typename std::enable_if<is_not_constant<T>::value, unary<T, negation >>::type operator - (T t) { return unary<T, negation >(t); }
}

#include <cmath>
namespace aks
{
#define DEFINE_BINARY_OP_PREFIX(NAME, PREFIX_NAME, OP)                                                                                                                                                                  \
    struct NAME { template<typename T, typename U> auto operator()( T t, U u ) const { using namespace std; return PREFIX_NAME (t, u); } };                                                                                 \
    template<typename T, typename U> typename std::enable_if<is_not_constant<T>::value && is_not_constant<U>::value, binary<T,U, NAME >>::type operator OP ( T t, U u ) { return binary<T,U, NAME >(t,u); }                 \
    template<typename T, typename U> typename std::enable_if<is_not_constant<T>::value && is_constant<U>::value, binary<T,constant<U>, NAME >>::type operator OP ( T t, U u ) { return binary<T,constant<U>, NAME >(t,u); } \
    template<typename T, typename U> typename std::enable_if<is_constant<T>::value && is_not_constant<U>::value, binary<constant<T>,U, NAME >>::type operator OP ( T t, U u ) { return binary<constant<T>,U, NAME >(t,u); } \
    template<typename T, typename U> auto operator OP ( constant<T> t, constant<U> u ) { return make_constant(PREFIX_NAME(t.t,u.t)); } \
    template<typename T, typename U> auto operator OP ( T t, constant<U> u )           { return (PREFIX_NAME(t, u.t) ); } \
    template<typename T, typename U> auto operator OP ( constant<T> t, U u )           { return (PREFIX_NAME(t.t, u) ); }

	DEFINE_BINARY_OP_PREFIX(power, pow, ^);

#define DEFINE_BINARY_FUNC_PREFIX(NAME, PREFIX_NAME, OP)                                                                                                                                                                  \
    struct NAME { template<typename T, typename U> auto operator()( T t, U u ) const { using namespace std; return PREFIX_NAME (t, u); } };                                                                                 \
    template<typename T, typename U> typename std::enable_if<is_not_constant<T>::value && is_not_constant<U>::value, binary<T,U, NAME >>::type OP ( T t, U u ) { return binary<T,U, NAME >(t,u); }                 \
    template<typename T, typename U> typename std::enable_if<is_not_constant<T>::value && is_constant<U>::value, binary<T,constant<U>, NAME >>::type OP ( T t, U u ) { return binary<T,constant<U>, NAME >(t,u); } \
    template<typename T, typename U> typename std::enable_if<is_constant<T>::value && is_not_constant<U>::value, binary<constant<T>,U, NAME >>::type OP ( T t, U u ) { return binary<constant<T>,U, NAME >(t,u); } \
    template<typename T, typename U> auto OP ( constant<T> t, constant<U> u ) { return make_constant(PREFIX_NAME(t.t,u.t)); } \
    template<typename T, typename U> auto OP ( T t, constant<U> u )           { return (PREFIX_NAME(t, u.t) ); } \
    template<typename T, typename U> auto OP ( constant<T> t, U u )           { return (PREFIX_NAME(t.t, u) ); }

	DEFINE_BINARY_FUNC_PREFIX(arc_tangent_2, std::atan2, atan2);

#define DEFINE_UNARY_FUNC(NAME, PREFIX_NAME, CALL_NAME)                                                                                       \
    struct NAME { template<typename T> auto operator()( T t ) { using namespace std; return PREFIX_NAME(t); } };                                \
    template<typename T> typename std::enable_if<is_not_constant<T>::value, unary<T, NAME >>::type CALL_NAME (T t){ return unary<T, NAME >(t); }\
    template<typename T> auto CALL_NAME( constant<T> t ){ using namespace std; return make_constant(PREFIX_NAME(t.t)); }

	DEFINE_UNARY_FUNC(cosine, std::cos, cos); //
	DEFINE_UNARY_FUNC(sine, std::sin, sin); //
	DEFINE_UNARY_FUNC(tangent, std::tan, tan); //
	DEFINE_UNARY_FUNC(arc_cosine, std::acos, acos); //
	DEFINE_UNARY_FUNC(arc_sine, std::asin, asin); //
	DEFINE_UNARY_FUNC(arc_tangent, std::atan, atan); //
	DEFINE_UNARY_FUNC(hyperbolic_cosine, std::cosh, cosh); //
	DEFINE_UNARY_FUNC(hyperbolic_sine, std::sinh, sinh); //
	DEFINE_UNARY_FUNC(hyperbolic_tangent, std::tanh, tanh);
	DEFINE_UNARY_FUNC(hyperbolic_arc_cosine, std::acosh, acosh); //
	DEFINE_UNARY_FUNC(hyperbolic_arc_sine, std::asinh, asinh); //
	DEFINE_UNARY_FUNC(hyperbolic_arc_tangent, std::atanh, atanh); //
	DEFINE_UNARY_FUNC(exponential, std::exp, exp); //
	DEFINE_UNARY_FUNC(logarithm_natural, std::log, log);
	DEFINE_UNARY_FUNC(logarithm_common, std::log10, log10);
	DEFINE_UNARY_FUNC(sq_root, std::sqrt, sqrt); //
	DEFINE_UNARY_FUNC(cube_root, std::cbrt, cbrt); //
	DEFINE_UNARY_FUNC(absolute_value, std::abs, abs); //
}

namespace aks
{
	template<typename T>
	auto derivative(T);

	template<typename T>
	auto derivative(constant<T>) { return constant<int>(0); }

	template<size_t N>
	auto derivative(variable<N>) { return constant<int>(1); }

	template<typename T, typename U>
	auto derivative(binary<T, U, add> b) { return derivative(b.t) + derivative(b.u); }

	template<typename T, typename U>
	auto derivative(binary<T, constant<U>, add> b) { return derivative(b.t); }

	template<typename T, typename U>
	auto derivative(binary<constant<T>, U, add> b) { return derivative(b.u); }

	template<typename T, typename U>
	auto derivative(binary<T, U, sub> b) { return derivative(b.t) - derivative(b.u); }

	template<typename T, typename U>
	auto derivative(binary<T, constant<U>, sub> b) { return derivative(b.t); }

	template<typename T, typename U>
	auto derivative(binary<constant<T>, U, sub> b) { return make_constant(-1) * derivative(b.u); }

	template<typename T, typename U>
	auto derivative(binary<T, U, mul> b) { return (derivative(b.t)*b.u) + (derivative(b.u)*b.t); }

	template<size_t N, typename U>
	auto derivative(binary<variable<N>, U, mul> b) { return (b.u) + (derivative(b.u)*b.t); }

	template<typename T, size_t N>
	auto derivative(binary<T, variable<N>, mul> b) { return (derivative(b.t)*b.u) + (b.t); }

	template<size_t T, size_t N>
	auto derivative(binary<variable<T>, variable<N>, mul> b) { return (b.u) + (b.t); }

	template<typename T, typename U>
	auto derivative(binary<T, U, dvd> b) { return (derivative(b.t)*b.u - derivative(b.u)*b.t) / (b.u*b.u); }

	template<typename T>
	auto derivative(unary<T, negation> u) { return -1 * derivative(u.t); }

	template<size_t N>
	auto derivative(unary<variable<N>, negation> u) { return constant<int>(-1); }
}

namespace aks
{
	template<typename T>
	auto derivative(unary<T, cosine> u) { return -sin(u.t)*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, sine> u) { return cos(u.t)*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, tangent> u) { return (1 + tan(u.t)*tan(u.t))*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, arc_cosine> u) { return (-(1 / sqrt(1 - u.t*u.t)))*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, arc_sine> u) { return ((1 / sqrt(1 - u.t*u.t)))*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, arc_tangent> u) { return (1 / (1 + u.t*u.t))*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, hyperbolic_cosine> u) { return sinh(u.t)*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, hyperbolic_sine> u) { return cosh(u.t)*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, hyperbolic_arc_cosine> u) { return (1 / sqrt(u.t*u.t + 1))*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, hyperbolic_arc_sine> u) { return (1 / sqrt(u.t*u.t - 1))*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, hyperbolic_arc_tangent> u) { return (1 / (1 - u.t*u.t))*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, exponential> u) { return u * derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, sq_root> u) { return (0.5 * (1.0 / u))*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, cube_root> u) { return (1.0 / (3 * cbrt(u.t*u.t)))*derivative(u.t); }

	template<typename T>
	auto derivative(unary<T, absolute_value> u) { return (u.t / u)*derivative(u.t); }

	//

	template<size_t N>
	auto derivative(unary<variable<N>, cosine> u) { return -sin(u.t); }

	template<size_t N>
	auto derivative(unary<variable<N>, sine> u) { return cos(u.t); }

	template<size_t N>
	auto derivative(unary<variable<N>, tangent> u) { return (1 + tan(u.t)*tan(u.t)); }

	template<size_t N>
	auto derivative(unary<variable<N>, arc_cosine> u) { return (-(1 / sqrt(1 - u.t*u.t))); }

	template<size_t N>
	auto derivative(unary<variable<N>, arc_sine> u) { return ((1 / sqrt(1 - u.t*u.t))); }

	template<size_t N>
	auto derivative(unary<variable<N>, arc_tangent> u) { return (1 / (1 + u.t*u.t)); }

	template<size_t N>
	auto derivative(unary<variable<N>, hyperbolic_cosine> u) { return sinh(u.t); }

	template<size_t N>
	auto derivative(unary<variable<N>, hyperbolic_sine> u) { return cosh(u.t); }

	template<size_t N>
	auto derivative(unary<variable<N>, hyperbolic_arc_cosine> u) { return (1 / sqrt(u.t*u.t + 1)); }

	template<size_t N>
	auto derivative(unary<variable<N>, hyperbolic_arc_sine> u) { return (1 / sqrt(u.t*u.t - 1)); }

	template<size_t N>
	auto derivative(unary<variable<N>, hyperbolic_arc_tangent> u) { return (1 / (1 - u.t*u.t)); }

	template<size_t N>
	auto derivative(unary<variable<N>, exponential> u) { return u; }

	template<size_t N>
	auto derivative(unary<variable<N>, sq_root> u) { return (0.5 * (1.0 / u)); }

	template<size_t N>
	auto derivative(unary<variable<N>, cube_root> u) { return (1.0 / (3 * cbrt(u.t*u.t))); }

	template<size_t N>
	auto derivative(unary<variable<N>, absolute_value> u) { return (u.t / u); }
}

namespace aks
{
	template<typename T, size_t M>
	struct depends_on_dim;

	template<size_t N, size_t M>
	struct depends_on_dim<variable<N>, M>
	{
		enum { value = M == N };
	};

	template<typename T, size_t M>
	struct depends_on_dim<constant<T>, M>
	{
		enum { value = false };
	};

	template<typename T, typename U, typename Op, size_t M>
	struct depends_on_dim<binary<T, U, Op>, M>
	{
		enum { value = depends_on_dim<T, M>::value || depends_on_dim<U, M>::value };
	};

	template<typename T, typename Op, size_t M>
	struct depends_on_dim<unary<T, Op>, M>
	{
		enum { value = depends_on_dim<T, M>::value };
	};

	template<typename T, size_t N>
	constexpr bool depends_on(T t, aks::variable<N>)
	{
		return aks::depends_on_dim<T, N>::value;
	}

	template<bool B>
	struct get_return
	{
		template<typename T, typename U>
		static auto apply(T t, U) { return t; }
	};

	template<>
	struct get_return<false>
	{
		template<typename T, typename U>
		static auto apply(T, U u) { return u; }
	};
}

namespace aks
{
	template<typename T, size_t D>
	auto differentiate(T, variable<D>);

	template<typename T, size_t N>
	auto differentiate(constant<T>, variable<N>) { return constant<int>(0); }

	template<size_t M, size_t N>
	auto differentiate(variable<M>, variable<N>) { return constant<int>(M == N ? 1 : 0); }

	template<typename T, typename U, size_t N>
	auto differentiate(binary<T, U, add> b, variable<N> v) { return get_return<!depends_on_dim<decltype(b), N>::value>::apply(constant<int>(0), differentiate(b.t, v) + differentiate(b.u, v)); }

	template<typename T, typename U, size_t N>
	auto differentiate(binary<T, constant<U>, add> b, variable<N> v) { return get_return<!depends_on_dim<decltype(b), N>::value>::apply(constant<int>(0), differentiate(b.t, v)); }

	template<typename T, typename U, size_t N>
	auto differentiate(binary<constant<T>, U, add> b, variable<N> v) { return get_return<!depends_on_dim<decltype(b), N>::value>::apply(constant<int>(0), differentiate(b.u, v)); }

	template<typename T, typename U, size_t N>
	auto differentiate(binary<T, U, sub> b, variable<N> v) { return get_return<!depends_on_dim<decltype(b), N>::value>::apply(constant<int>(0), differentiate(b.t, v) - differentiate(b.u, v)); }

	template<typename T, typename U, size_t N>
	auto differentiate(binary<T, constant<U>, sub> b, variable<N> v) { return get_return<!depends_on_dim<decltype(b), N>::value>::apply(constant<int>(0), differentiate(b.t, v)); }

	template<typename T, typename U, size_t N>
	auto differentiate(binary<constant<T>, U, sub> b, variable<N> v) { return get_return<!depends_on_dim<decltype(b), N>::value>::apply(constant<int>(0), make_constant(-1) * differentiate(b.u, v)); }

	template<typename T, typename U, size_t N>
	auto differentiate(binary<T, U, mul> b, variable<N> v) { return get_return<!depends_on_dim<decltype(b), N>::value>::apply(constant<int>(0), (differentiate(b.t, v)*b.u) + (differentiate(b.u, v)*b.t)); }

	//template<size_t N, typename U, size_t D>
	//auto differentiate( binary<variable<N>,U,mul> b, variable<D> v) { return get_return<!depends_on_dim<decltype(b), D>::value>::apply(constant<int>(0),  (b.u) + (differentiate(b.u)*b.t));  }
	//
	//template<typename T, size_t N, size_t D>
	//auto differentiate( binary<T,variable<N>,mul> b, variable<D> v) { return get_return<!depends_on_dim<decltype(b), D>::value>::apply(constant<int>(0),  (differentiate(b.t)*b.u) + (b.t));  }
	//
	//template<size_t T, size_t N, size_t D>
	//auto differentiate( binary<variable<T>,variable<N>,mul> b, variable<D> v) { return get_return<!depends_on_dim<decltype(b), D>::value>::apply(constant<int>(0),  (b.u) + (b.t));  }

	template<typename T, typename U, size_t N>
	auto differentiate(binary<T, U, dvd> b, variable<N> v) { return get_return<!depends_on_dim<decltype(b), N>::value>::apply(constant<int>(0), (differentiate(b.t, v)*b.u - differentiate(b.u, v)*b.t) / (b.u*b.u)); }

	template<typename T, size_t N>
	auto differentiate(unary<T, negation> u, variable<N> v) { return get_return<!depends_on_dim<decltype(u), N>::value>::apply(constant<int>(0), -1 * differentiate(u.t, v)); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, negation> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), constant<int>(-1)); }
}

namespace aks
{
	template<typename T, size_t D>
	auto differentiate(unary<T, cosine> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), -sin(u.t)*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, sine> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), cos(u.t)*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, tangent> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), (1 + tan(u.t)*tan(u.t))*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, arc_cosine> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), (-(1 / sqrt(1 - u.t*u.t)))*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, arc_sine> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), ((1 / sqrt(1 - u.t*u.t)))*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, arc_tangent> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), (1 / (1 + u.t*u.t))*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, hyperbolic_cosine> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), sinh(u.t)*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, hyperbolic_sine> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), cosh(u.t)*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, hyperbolic_arc_cosine> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), (1 / sqrt(u.t*u.t + 1))*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, hyperbolic_arc_sine> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), (1 / sqrt(u.t*u.t - 1))*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, hyperbolic_arc_tangent> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), (1 / (1 - u.t*u.t))*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, exponential> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), u*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, sq_root> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), (0.5 * (1.0 / u))*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, cube_root> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), (1.0 / (3 * cbrt(u.t*u.t)))*differentiate(u.t, v)); }

	template<typename T, size_t D>
	auto differentiate(unary<T, absolute_value> u, variable<D> v) { return get_return<!depends_on_dim<decltype(u), D>::value>::apply(constant<int>(0), (u.t / u)*differentiate(u.t, v)); }

	//

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, cosine> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), -sin(u.t)); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, sine> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), cos(u.t)); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, tangent> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), (1 + tan(u.t)*tan(u.t))); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, arc_cosine> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), (-(1 / sqrt(1 - u.t*u.t)))); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, arc_sine> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), ((1 / sqrt(1 - u.t*u.t)))); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, arc_tangent> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), (1 / (1 + u.t*u.t))); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, hyperbolic_cosine> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), sinh(u.t)); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, hyperbolic_sine> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), cosh(u.t)); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, hyperbolic_arc_cosine> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), (1 / sqrt(u.t*u.t + 1))); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, hyperbolic_arc_sine> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), (1 / sqrt(u.t*u.t - 1))); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, hyperbolic_arc_tangent> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), (1 / (1 - u.t*u.t))); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, exponential> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), u); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, sq_root> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), (0.5 * (1.0 / u))); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, cube_root> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), (1.0 / (3 * cbrt(u.t*u.t)))); }

	template<size_t N, size_t D>
	auto differentiate(unary<variable<N>, absolute_value> u, variable<D> v) { return get_return<!(N == D)>::apply(constant<int>(0), (u.t / u)); }
}

#endif // !__compile_time_differentiation_hpp__