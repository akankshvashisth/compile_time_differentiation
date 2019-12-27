#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <type_traits>
#include <typeinfo>
#include <algorithm>

#include <cstdlib>
#include <memory>
//#include <cxxabi.h>

#include "compile_time_differentiation.hpp"
#include "tuple_helper_utilities.hpp"

std::string& replace(
	std::string& s,
	std::string const & o,
	std::string const & n)
{
	std::string::size_type p = 0u;
	auto const ol = o.length(), nl = n.length();
	while ((p = s.find(o, p)) != std::string::npos) {
		s.replace(p, ol, n);
		p += nl;
	}
	return s;
}
std::string demangle(const char* name) {
	//int status = -4; // some arbitrary value to eliminate the compiler warning
	//
	//				 // enable c++11 by passing the flag -std=c++11 to g++
	//std::unique_ptr<char, void(*)(void*)> res{
	//	abi::__cxa_demangle(name, NULL, NULL, &status),
	//	std::free
	//};
	//
	//return (status == 0) ? res.get() : name;
	std::string ret(name);
	replace(ret, "aks::", "");
	replace(ret, "struct ", "");
	replace(ret, "binary", "B");
	replace(ret, "unary", "U");
	replace(ret, "variable", "V");
	replace(ret, "constant", "C");
	replace(ret, "power", "^");
	replace(ret, "add", "+");
	replace(ret, "mul", "*");
	replace(ret, "dvd", "/");
	replace(ret, "sub", "-");
	replace(ret, "exponential", "exp");
	replace(ret, "sq_root", "sqrt");
	replace(ret, "double", "dbl");
	replace(ret, "tangent", "tan");
	replace(ret, "cosine", "cos");
	replace(ret, "sine", "sin");
	replace(ret, "cube_root", "cbrt");
	replace(ret, "negation", "neg");
	replace(ret, "V<0>", "x");
	replace(ret, "V<1>", "y");
	replace(ret, "V<2>", "z");
	replace(ret, "V<3>", "w");
	replace(ret, "C<int>", "K");
	replace(ret, "C<dbl>", "K");
	return ret;
}

template<typename T>
constexpr size_t get_dim(T)
{
	return T::dim;
}

#include <tuple>

template<typename... Ts>
auto functions(Ts... ts)
{
	return std::make_tuple(ts...);
}

namespace detail
{
	template<int... Is>
	struct seq { };

	template<int N, int... Is>
	struct gen_seq : gen_seq<N - 1, N - 1, Is...> { };

	template<int... Is>
	struct gen_seq<0, Is...> : seq<Is...> { };

	template<typename T, typename F, int... Is>
	void for_each(T&& t, F f, seq<Is...>)
	{
		auto l = { (f(std::get<Is>(t)), 0)... };
	}
}

template<typename... Ts, typename F>
void for_each_in_tuple(std::tuple<Ts...> const& t, F f)
{
	detail::for_each(t, f, detail::gen_seq<sizeof...(Ts)>());
}

template<typename T, int... Is, typename... As >
auto eval(T t, detail::seq<Is...>, As... as)
{
	return std::make_tuple(std::get<Is>(t)(as...)...);
}

template<typename... Ts, typename... As>
auto evaluate(std::tuple<Ts...> t, As... ts)
{
	return eval(t, detail::gen_seq<sizeof...(Ts)>(), ts...);
}

template<size_t N, size_t... Ns>
struct Max
{
	enum { value = N > Max<Ns...>::value ? N : Max<Ns...>::value };
};

template<size_t N>
struct Max<N>
{
	enum { value = N };
};

template<size_t... Is>
struct sz_seq { };

template<int N, size_t... Is>
struct sz_gen_seq : sz_gen_seq<N - 1, N - 1, Is...> { };

template<size_t... Is>
struct sz_gen_seq<0, Is...> : sz_seq<Is...> { };

template<typename F, size_t... Ns>
auto jacobian_detail2(F f, sz_seq<Ns...>)
{
	return std::make_tuple(aks::differentiate(f, aks::variable<Ns>())...);
}

template<typename... Fs, size_t... Ns>
auto jacobian_detail(sz_seq<Ns...>, Fs... fs)
{
	return std::make_tuple(jacobian_detail2(fs, sz_seq<Ns...>())...);
}

template<typename... Fs>
auto jacobian(Fs... fs)
{
	return jacobian_detail(sz_gen_seq<Max<decltype(fs)::dim...>::value>(), fs...);
}

template<size_t FN, size_t VN, typename... Ts>
auto get_jacobian_element(std::tuple<Ts...> const& t)
{
	return std::get<VN>(std::get<FN>(t));
}
//
//template<typename... Fs, size_t... Ns>
//auto hessian_detail(sz_seq<Ns...>, Fs... fs)
//{
//    return std::make_tuple(hessian_detail2(fs, sz_seq<Ns...>())...);
//}
//
//template<typename... Fs>
//auto hessian(Fs... fs)
//{
//
//}

#include <vector>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

template<size_t N>
struct Apply {
	template<typename F, typename T, typename... A>
	static inline auto apply(F && f, T && t, A &&... a) {
		return Apply<N - 1>::apply(::std::forward<F>(f), ::std::forward<T>(t),
			::std::get<N - 1>(::std::forward<T>(t)), ::std::forward<A>(a)...
		);
	}
};

template<>
struct Apply<0> {
	template<typename F, typename T, typename... A>
	static inline auto apply(F && f, T &&, A &&... a) {
		return ::std::forward<F>(f)(::std::forward<A>(a)...);
	}
};

template<typename F, typename T>
inline auto apply(F && f, T && t) {
	return Apply< ::std::tuple_size< typename ::std::decay<T>::type
	>::value>::apply(::std::forward<F>(f), ::std::forward<T>(t));
}

template<size_t I>
struct size_t_holder
{
	static const size_t value = I;
};

template<size_t Fn, size_t Vn, typename... Ts, typename R, typename... As>
auto set_value(std::tuple<Ts...> const& t, R& res, std::tuple<As...> const& a, size_t_holder<Vn>)
{
	//std::cout << "Function = " << Fn << " " << "Variable = " << Vn << " " << "Value = " << apply(get_jacobian_element<Fn, Vn>(t), a) << std::endl;
	res(Fn, Vn) = apply(get_jacobian_element<Fn, Vn>(t), a);
	return true;
}

template<typename... As>
void eat_up(As...) {}

template<typename... Ts, size_t... Vs, size_t Fn, typename R, typename... As>
auto value_jacobian_at_internal(sz_seq<Vs...>, size_t_holder<Fn>, std::tuple<Ts...> const& t, R& res, std::tuple<As...> const& a)
{
	eat_up(set_value<Fn>(t, res, a, size_t_holder<Vs>())...);
	return true;
}

template<typename... Ts, size_t... Fs, typename R, typename... As>
auto value_jacobian_at(sz_seq<Fs...>, std::tuple<Ts...> const& t, R& res, std::tuple<As...> const& a)
{
	typedef typename std::tuple_element<0, std::tuple<Ts...>>::type type;
	eat_up(value_jacobian_at_internal(sz_gen_seq< std::tuple_size<type>::value >(), size_t_holder<Fs>(), t, res, a)...);
}

template<typename... Ts, typename R, typename... As>
auto value_jacobian_at(std::tuple<Ts...> const& t, R& res, As... as)
{
	value_jacobian_at(sz_gen_seq<sizeof...(Ts)>(), t, res, std::make_tuple(as...));
}

template<typename... Ts>
constexpr size_t get_number_of_function(std::tuple<Ts...>)
{
	typedef std::tuple<Ts...> type;
	return std::tuple_size<type>::value;
}

template<typename... Ts>
constexpr size_t get_number_of_variables(std::tuple<Ts...>)
{
	typedef typename std::tuple_element<0, std::tuple<Ts...>>::type type;
	return std::tuple_size< type >::value;
}

template<typename F>
constexpr auto gradient(F f)
{
	return std::get<0>(jacobian(f));
}

template<typename... Ts, typename... As>
auto value_gradient_at(std::tuple<Ts...> const& t, As... as)
{
	return aks::tuple_utils::map(t, [=](auto f) { return f(as...); });
}

template<typename... Ts, size_t... Is, typename R, typename... As>
auto value_gradient_at_in_res_impl(std::tuple<Ts...> const& t, sz_seq<Is...>, R& res, As... as)
{
	auto assign = [&res](size_t i, auto t) {res(i) = t; return 0; };
	eat_up(assign(Is, std::get<Is>(t)(as...))...);
}

template<typename... Ts, typename R, typename... As>
auto value_gradient_at_in_res(std::tuple<Ts...> const& t, R& res, As... as)
{
	value_gradient_at_in_res_impl(t, sz_gen_seq<sizeof...(Ts)>(), res, as...);
}

template<typename T>
bool is_abs_greater_than(T t, T x)
{
	return t > x || t < -x;
}

template<typename F, typename T>
auto newton_raphson(F f, T guess, T tolerance, int max_counter)
{
	auto val = f(guess);
	while (is_abs_greater_than(val, tolerance) && max_counter-- > 0)
	{
		guess = guess - val / aks::derivative(f)(guess);
		val = f(guess);
		//std::cout << max_counter << " " << guess << " " << val << std::endl;
	}

	return std::make_tuple(guess, !is_abs_greater_than(val, tolerance));
}

#include <string>
#include <sstream>
#include <iomanip>

template<typename _Container>
std::string join(std::string const& delimiter, _Container const& c)
{
	if (std::empty(c)) {
		return std::string();
	}

	auto begin = std::begin(c);
	std::stringstream ss;
	ss << *begin++;

	for (; begin != std::end(c);) {
		ss << delimiter << *begin++;
	}

	return ss.str();
}

template<typename _Container, typename _Func>
std::string join(std::string const& delimiter, _Container const& c, _Func f)
{
	if (std::empty(c)) {
		return std::string();
	}

	auto begin = std::begin(c);
	std::stringstream ss;
	ss << f(*begin++);

	for (; begin != std::end(c);) {
		ss << delimiter << f(*begin++);
	}

	return ss.str();
}

std::string to_string(bool b)
{
	return b ? "true" : "false";
};

template<typename T>
std::string to_string(T const& v)
{
	std::stringstream ss;
	ss << std::setfill('0') << std::setw(6) << v;
	return ss.str();
}

template<typename T>
std::string to_string(std::vector<T> const& v) {
	std::stringstream ss;
	ss << "{";
	ss << join(", ", v, [](T const& t) {return to_string(t); });
	ss << "}";
	return ss.str();
};

template<typename T>
std::string to_string(std::vector<std::vector<T>> const& v) {
	std::stringstream ss;
	ss << "{\n  ";
	if (!v.empty())
	{
		ss << join(",\n  ", v, [](std::vector<T> const& t) {return to_string(t); });
		/*auto begin = v.cbegin();
		ss << "\n  " << to_string(*begin++);
		std::for_each(begin, v.cend(), [&](auto const& d) {
			ss << "\n, " << to_string(d);
		});*/
		ss << "\n";
	}
	ss << "}";
	return ss.str();
};

int compile_time_differentiation_tests()
{
	aks::variable<0> x;
	aks::variable<1> y;
	aks::variable<2> z;
	aks::variable<3> w;
	auto const pi = 3.1415926535897932384626433;

	/////////////////////////////
	//auto ode = sin(x*y)/y;
	//
	//double h = 0.01;
	//double t = 0;
	//auto Y = std::make_tuple(0.2);
	//for_each_in_tuple(Y, [](double d){ std::cout << d << std::endl; });
	//for(size_t i=0; i<10000; ++i)
	//{
	//    Y = runge_kutta_step(ode, h, t, Y);
	//    //if(i%10 == 0)
	//    {
	//        for_each_in_tuple(Y, [](double d){ std::cout << d << std::endl; });
	//    }
	//    t+=h;
	//}
	//std::cout << t << std::endl;
	/////////////////////////////
	{
		std::cout << ((x)(2) == 2) << std::endl;
		std::cout << ((y)(2, 3.5) == 3.5) << std::endl;
		std::cout << ((z)(2, 3, 4) == 4) << std::endl;
		std::cout << ((w)(2, 3, 4, 5) == 5) << std::endl;

		std::cout << ((x + x)(2) == 4) << std::endl;
		std::cout << ((y + y)(2, 3.5) == 7.0) << std::endl;
		std::cout << ((z + z)(2, 3, 4) == 8) << std::endl;

		std::cout << ((x + x)(2) == 4) << std::endl;
		std::cout << ((x + y)(2, 3.5) == 5.5) << std::endl;
		std::cout << ((x + z)(2, 3, 4) == 6) << std::endl;

		std::cout << ((x + 2)(2) == 4) << std::endl;
		std::cout << ((2 + x)(2) == 4) << std::endl;

		std::cout << ((x + y + 3 + z)(1, 2, 4) == 10) << std::endl;

		std::cout << ((x*x)(2) == 4) << std::endl;
		std::cout << ((y*y)(2, 3.5) == 12.25) << std::endl;
		std::cout << ((z*z)(2, 3, 4) == 16) << std::endl;

		std::cout << ((x*x)(2) == 4) << std::endl;
		std::cout << ((x*y)(2, 3.5) == 7.0) << std::endl;
		std::cout << ((x*z)(2, 3, 4) == 8) << std::endl;

		std::cout << ((x * 2)(2) == 4) << std::endl;
		std::cout << ((2 * x)(2) == 4) << std::endl;

		std::cout << ((x*y * 3 * z)(1, 2, 4) == 24) << std::endl;

		std::cout << ((x*y*z + x * x*x + y * z / (x*2.5) - z + z * z)(2.0, 3.0, 4.0) == 46.4) << std::endl;

		std::cout << ((x ^ 5)(2) == 32) << std::endl;
		std::cout << ((5 ^ x)(2) == 25) << std::endl;
	}
	{
		std::string const a("hello ");
		std::string const b("hello1 ");
		std::string const c("hello2 ");

		std::cout << (x + " world " + y + y + " you")(a, b) << std::endl;
		std::cout << (x + z + y)(a, b, c) << std::endl;
		std::cout << cos(x)(3.14159265 / 3.0) << std::endl;
		std::cout << atan2(x, y)(1.0, 2.0) << std::endl;
	}

	std::cout << ((sin(x) ^ 2) + (cos(x) ^ 2))(0.23) << std::endl;
	std::cout << ((sin(x) ^ 2) + (cos(x) ^ 2))(0.53) << std::endl;
	std::cout << ((sin(x) ^ 2) + (cos(x) ^ 2))(3.14159265) << std::endl;

	/////////////////////////////
	std::cout << (sin(x) / cos(x))(0.23) << " " << (tan(x))(0.23) << std::endl;

	std::cout << sin(2 * x)(0.23) << " " << (2 * sin(x)*cos(x))(0.23) << std::endl;

	std::cout << sin(-x)(0.23) << " " << (-sin(x))(0.23) << std::endl;

	std::cout << (exp(sin(x) ^ cos(y)*z) / w)(0.25, 0.50, 0.33, 0.75) << std::endl;

	std::cout << demangle(typeid((2 * (x^y) + sqrt(w*z*x)*4.5 / tan(x / y)*exp(sin(x) ^ cos(y)*z) / w)).name()) << std::endl;

	std::cout << get_dim(exp(sin(x) ^ cos(y)*z)) << std::endl;
	{
		auto eq = sin(x) - 1;
		double root;
		bool converged;
		std::tie(root, converged) = newton_raphson(eq, 1.2/*guess*/, 1e-13/*tolerance*/, 20/*max iterations*/);
		auto pi_by_what = [pi](auto x) { return pi / x; };
		std::cout << root << " " << to_string(converged) << " " << eq(root) << " pi/" << pi_by_what(root) << std::endl;
	}
	{
		std::cout << depends_on(x*x, x) << std::endl;
		std::cout << depends_on(x*x, y) << std::endl;
		std::cout << depends_on(x*y, y) << std::endl;
		std::cout << depends_on((2 * x) ^ w + sqrt(w*x)*4.5 / tan(x / w)*exp(sin(x) ^ cos(w)*z) / w, x) << std::endl;
		std::cout << depends_on((2 * x) ^ w + sqrt(w*x)*4.5 / tan(x / w)*exp(sin(x) ^ cos(w)*z) / w, y) << std::endl;
		std::cout << depends_on((2 * x) ^ w + sqrt(w*x)*4.5 / tan(x / w)*exp(sin(x) ^ cos(w)*z) / w, z) << std::endl;
		std::cout << depends_on((2 * x) ^ w + sqrt(w*x)*4.5 / tan(x / w)*exp(sin(x) ^ cos(w)*z) / w, w) << std::endl;
	}
	{
		auto dd = differentiate(x*y*y, y);
		std::cout << dd(2, 3) << std::endl;
		std::cout << demangle(typeid(dd).name()) << std::endl;
	}
	{
		auto dd2 = differentiate(sin(y*x), x);
		std::cout << dd2(0.25, 0.63) << std::endl;
	}
	{
		std::cout << Max<3, 2, 4, 2, 5, 3, 6, 3, 2, 4, 2, 2>::value << std::endl;
	}
	{
		auto j = jacobian(x*w, y*z, y*y, z*z, x*x, y*x*z + x * w);
		std::vector<std::vector<double>> res(get_number_of_function(j), std::vector<double>(get_number_of_variables(j), 0.0));
		auto res_func = [&res](size_t fn, size_t v)->double& {return res[fn][v]; };
		value_jacobian_at(j, res_func, 1.5, 3.3, 2.7, 3.4);
		std::cout << to_string(res) << std::endl;
	}
	{
		auto j = jacobian(2 * x + 3 * y + 4 * z);
		auto k = gradient(2 * x + 3 * y + 4 * z);
		std::vector<int> res2(3);
		auto m = value_gradient_at(k, 1, 1, 1);
		std::tie(res2[0], res2[1], res2[2]) = m;
		std::cout << to_string(res2) << std::endl;
		auto res2_func = [&res2](size_t v)->auto& {return res2[v]; };
		value_gradient_at_in_res(k, res2_func, 1, 1, 1);
		std::cout << to_string(res2) << std::endl;
		std::vector<std::vector<double>> res(get_number_of_function(j), std::vector<double>(get_number_of_variables(j), 0.0));
		auto res_func = [&res](size_t fn, size_t v)->auto& {return res[fn][v]; };
		value_jacobian_at(j, res_func, 1, 1, 1);
		std::cout << to_string(res2) << std::endl;
		std::cout << std::endl;
	}
	/////////////////////////////
	{
		auto ds = (x*y*z);
		std::cout << ds(2, 3, 4) << std::endl;
		std::cout << differentiate((x*y*z), x)(2, 3, 4) << std::endl;
		std::cout << demangle(typeid(ds).name()) << std::endl;
		auto dds = differentiate(ds, x);
		std::cout << demangle(typeid(dds).name()) << std::endl;
		auto ddds = differentiate(dds, y);
		std::cout << demangle(typeid(ddds).name()) << std::endl;
		auto dddds = differentiate(ddds, z);
		std::cout << demangle(typeid(dddds).name()) << std::endl;
		auto ddddds = differentiate(dddds, x);
		std::cout << demangle(typeid(ddddds).name()) << std::endl;
	}

	//std::cout << std::to_string(ds) << "\n"
	//	<< std::to_string(dds) << "\n"
	//  << std::to_string(ddds) << "\n"
	//	<< std::to_string(ddds) << "\n"
	//	<< std::to_string(dddds) << "\n"
	//	<< std::to_string(ddddds) << "\n"
	//	<< std::endl;

	{
		auto j = jacobian(x*w, y*z, y*y, z*z, x*x, y*x*z + x * w);
		std::vector<std::vector<double>> res(get_number_of_function(j), std::vector<double>(get_number_of_variables(j), 0.0));
		auto res_func = [&res](size_t fn, size_t v)->double& {return res[fn][v]; };
		value_jacobian_at(j, res_func, 1.5, 3.3, 2.7, 3.4);
		std::cout << to_string(res) << std::endl;
		std::cout << std::endl;
		//std::cout << demangle(typeid(j).name()) << std::endl;
		std::cout << get_jacobian_element<0, 0>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<0, 1>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<0, 2>(j)(1.5, 3.3, 2.7, 3.4) << std::endl;
		std::cout << get_jacobian_element<1, 0>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<1, 1>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<1, 2>(j)(1.5, 3.3, 2.7, 3.4) << std::endl;
		std::cout << get_jacobian_element<2, 0>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<2, 1>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<2, 2>(j)(1.5, 3.3, 2.7, 3.4) << std::endl;
		std::cout << get_jacobian_element<3, 0>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<3, 1>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<3, 2>(j)(1.5, 3.3, 2.7, 3.4) << std::endl;
	}
	{
		auto j = jacobian(x*w, y*z, y*y, z*z, x*x, y*x*z + x * w);
		std::vector<double> res(get_number_of_function(j) * get_number_of_variables(j));
		auto res_func = [&res, j](size_t fn, size_t v)->double& {return res[fn * get_number_of_variables(j) + v]; };
		value_jacobian_at(j, res_func, 1.5, 3.3, 2.7, 3.4);
		std::cout << to_string(res) << std::endl;
		std::cout << std::endl;
		//std::cout << demangle(typeid(j).name()) << std::endl;
		std::cout << get_jacobian_element<0, 0>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<0, 1>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<0, 2>(j)(1.5, 3.3, 2.7, 3.4) << std::endl;
		std::cout << get_jacobian_element<1, 0>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<1, 1>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<1, 2>(j)(1.5, 3.3, 2.7, 3.4) << std::endl;
		std::cout << get_jacobian_element<2, 0>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<2, 1>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<2, 2>(j)(1.5, 3.3, 2.7, 3.4) << std::endl;
		std::cout << get_jacobian_element<3, 0>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<3, 1>(j)(1.5, 3.3, 2.7, 3.4) << ", ";
		std::cout << get_jacobian_element<3, 2>(j)(1.5, 3.3, 2.7, 3.4) << std::endl;
	}
	{
		auto fs_sum = x + y + z + (x^2.0) + x * y + x * z + (y^2.0) + y * z + (z^2.0) + (x^3.0);
		std::cout << "-------" << fs_sum(1.2, 3.2, 1.3) << "-------" << std::endl;

		auto fs = functions(x, y, z, x ^ 2, x*y, x*z, y ^ 2, y*z, z ^ 2, x ^ 3);
		auto eval = evaluate(fs, 1.2, 3.2, 1.3);

		std::cout << demangle(typeid(fs).name()) << std::endl;
		std::cout << demangle(typeid(eval).name()) << std::endl;
		std::cout << demangle(typeid(fs_sum).name()) << std::endl;

		for_each_in_tuple(eval, [](double d) { std::cout << d << ", "; }); std::cout << std::endl;
		double sum = 0;
		for_each_in_tuple(eval, [&sum](double d) { sum += d; });
		std::cout << "-------" << sum << "-------" << std::endl;
	}
	{
		std::cout << tan(sin(cos(x)))(0.23) << std::endl;
	}
	{
		auto g = x / 3.0;
		auto dg = derivative(g);

		std::cout << demangle(typeid(g).name()) << std::endl;
		std::cout << demangle(typeid(dg).name()) << std::endl;
		std::cout << g(2.0) << " " << dg(2.0) << std::endl;
	}
	{
		auto h = sin(cos(x)*cos(x));
		auto dh = derivative(h);

		std::cout << demangle(typeid(h).name()) << std::endl;
		std::cout << demangle(typeid(dh).name()) << std::endl;
		std::cout << h(0.5) << " " << dh(0.5) << std::endl;
	}
	{
		auto j2 = sqrt(x*x*x*x - 2);
		auto dj2 = derivative(j2);

		std::cout << demangle(typeid(j2).name()) << std::endl;
		std::cout << demangle(typeid(dj2).name()) << std::endl;
		std::cout << j2(2) << " " << dj2(2) << std::endl;
	}
	{
		auto k = exp(x*x);
		auto dk = derivative(k);

		std::cout << demangle(typeid(k).name()) << std::endl;
		std::cout << demangle(typeid(dk).name()) << std::endl;
		std::cout << k(2.0) << " " << dk(2.0) << std::endl;
	}
	{
		auto m = cbrt(sin(x));
		auto dm = derivative(m);
		auto dm2 = derivative(dm);
		auto dm3 = derivative(dm2);
		auto dm4 = derivative(dm3);
		//auto dm5 = derivative(dm4);

		std::cout << demangle(typeid(m).name()) << std::endl;
		std::cout << demangle(typeid(dm).name()) << std::endl;
		std::cout << demangle(typeid(dm2).name()) << std::endl;
		std::cout << demangle(typeid(dm3).name()) << std::endl;
		std::cout << m(0.5) << " " << dm(0.5) << " " << dm2(0.5) << " " << dm3(0.5) << " " << dm4(0.5) << std::endl;
	}
	{
		auto n = -x * x;
		auto dn = derivative(n);
		auto dn2 = derivative(dn);
		auto dn3 = derivative(dn2);
		auto dn4 = derivative(dn3);
		auto dn5 = derivative(dn4);

		auto const vn = n(0.5);
		auto const vdn = dn(0.5);
		auto const vdn2 = dn2(0.5);
		auto const vdn3 = dn3(0.5);
		auto const vdn4 = dn4(0.5);
		auto const vdn5 = dn5(0.5);

		std::cout << vn << " ";
		std::cout << vdn << " ";
		std::cout << vdn2 << " ";
		std::cout << vdn3 << " ";
		std::cout << vdn4 << " ";
		std::cout << vdn5 << " " << std::endl;

		std::cout << demangle(typeid(n).name()) << std::endl;
		std::cout << demangle(typeid(dn).name()) << std::endl;
		std::cout << demangle(typeid(dn2).name()) << std::endl;
		std::cout << demangle(typeid(dn3).name()) << std::endl;
	}
	{
		auto p = x + x;
		auto dp = aks::derivative(p);
		auto dp2 = aks::derivative(dp);
		auto dp3 = aks::derivative(dp2);

		auto vp = p(2);
		auto vdp = dp(2);
		auto vdp2 = dp2(2);
		auto vdp3 = dp3(2);

		std::cout << vp << " ";
		std::cout << vdp << " ";
		std::cout << vdp2 << " ";
		std::cout << vdp3 << " " << std::endl;

		std::cout << demangle(typeid(p).name()) << std::endl;
		std::cout << demangle(typeid(dp).name()) << std::endl;
		std::cout << demangle(typeid(dp2).name()) << std::endl;
		std::cout << demangle(typeid(dp3).name()) << std::endl;
	}
	{
		auto a0 = sin(x);
		auto a1 = differentiate(a0, x);
		auto a2 = differentiate(a1, x);
		auto a3 = differentiate(a2, x);
		auto a4 = differentiate(a3, x);
		auto a5 = differentiate(a4, x);
		auto a6 = differentiate(a5, x);
		auto a7 = differentiate(a6, x);
		auto a8 = differentiate(a7, x);
		auto a9 = differentiate(a8, x);
		auto a10 = differentiate(a9, x);
		auto a11 = differentiate(a10, x);
		//auto a12 = derivative(a11);

		std::cout << demangle(typeid(a0).name()) << std::endl;
		std::cout << demangle(typeid(a1).name()) << std::endl;
		std::cout << demangle(typeid(a2).name()) << std::endl;
		std::cout << demangle(typeid(a3).name()) << std::endl;
		std::cout << demangle(typeid(a4).name()) << std::endl;
		std::cout << demangle(typeid(a5).name()) << std::endl;
		std::cout << demangle(typeid(a6).name()) << std::endl;
		std::cout << demangle(typeid(a7).name()) << std::endl;
		std::cout << demangle(typeid(a8).name()) << std::endl;
		std::cout << demangle(typeid(a9).name()) << std::endl;
		std::cout << demangle(typeid(a10).name()) << std::endl;
		std::cout << demangle(typeid(a11).name()) << std::endl;
		//std::cout << demangle(typeid(a12).name()) << std::endl;

		std::cout << a0(pi / 2.0) << std::endl;
		std::cout << a1(pi / 2.0) << std::endl;
		std::cout << a2(pi / 2.0) << std::endl;
		std::cout << a3(pi / 2.0) << std::endl;
		std::cout << a4(pi / 2.0) << std::endl;
		std::cout << a5(pi / 2.0) << std::endl;
		std::cout << a6(pi / 2.0) << std::endl;
		std::cout << a7(pi / 2.0) << std::endl;
		std::cout << a8(pi / 2.0) << std::endl;
		std::cout << a9(pi / 2.0) << std::endl;
		std::cout << a10(pi / 2.0) << std::endl;
		std::cout << a11(pi / 2.0) << std::endl;
	}
	{
		auto b0 = sqrt(x*y);
		auto b1 = differentiate(b0, y);
		std::cout << b0(20, 20) << " " << b1(20, 20) << std::endl;
	}
	return 0;
}