
#ifndef AKS_COMPILE_TIME_DIFFERENTIATION_HPP__
#define AKS_COMPILE_TIME_DIFFERENTIATION_HPP__

#include <tuple>

namespace aks {
namespace compile_time_diff {

namespace detail {
using real = double;

template <std::size_t N, typename T, typename... Xs>
constexpr auto nth(T x, Xs... xs) {
  if constexpr (N == 0) {
    return x;
  } else {
    return nth<N - 1>(xs...);
  }
}

template <std::size_t N>
struct var {
  static constexpr std::size_t id = N;
  static constexpr std::size_t arity = N + 1;
  using type = var<N>;
  template <typename... Ts>
  real value(Ts... xs) const {
    static_assert(sizeof...(Ts) >= arity,
                  "trying to get index, but not enough arguments");
    // return xs...[N]; --> C++26
    return nth<N>(xs...);
  }

  template <typename... Ts>
  real operator()(Ts... xs) const {
    return value(xs...);
  }
};

template <typename T>
struct is_var_s : std::bool_constant<false> {};
template <std::size_t N>
struct is_var_s<var<N>> : std::bool_constant<true> {};
template <typename T>
constexpr bool is_var_v = is_var_s<T>::value;
template <typename T>
concept is_var = is_var_v<T>;

struct constant {
  static constexpr std::size_t arity = 0;
  using type = constant;
  constant(real v) : v_(v) {}
  real v_;
  template <typename... Ts>
  real value(Ts...) const {
    return v_;
  }

  template <typename... Ts>
  real operator()(Ts...) const {
    return value();
  }
};

template <typename T>
struct is_constant_s : std::bool_constant<false> {};
template <>
struct is_constant_s<constant> : std::bool_constant<true> {};
template <typename T>
constexpr bool is_constant_v = is_constant_s<T>::value;
template <typename T>
concept is_constant = is_constant_v<T>;

struct op_neg {
  template <typename... Ts, typename... Xs>
  real value(std::tuple<Ts...> ts, Xs... xs) const {
    return -std::get<0>(ts).value(xs...);
  }
};
struct op_add {
  template <typename... Ts, typename... Xs>
  real value(std::tuple<Ts...> ts, Xs... xs) const {
    return std::get<0>(ts).value(xs...) + std::get<1>(ts).value(xs...);
  }
};
struct op_sub {
  template <typename... Ts, typename... Xs>
  real value(std::tuple<Ts...> ts, Xs... xs) const {
    return std::get<0>(ts).value(xs...) - std::get<1>(ts).value(xs...);
  }
};
struct op_mul {
  template <typename... Ts, typename... Xs>
  real value(std::tuple<Ts...> ts, Xs... xs) const {
    return std::get<0>(ts).value(xs...) * std::get<1>(ts).value(xs...);
  }
};
struct op_div {
  template <typename... Ts, typename... Xs>
  real value(std::tuple<Ts...> ts, Xs... xs) const {
    return std::get<0>(ts).value(xs...) / std::get<1>(ts).value(xs...);
  }
};
struct op_lt {
  // bool value(real l, real r) const { return l < r; }

  template <typename... Ts, typename... Xs>
  bool value(std::tuple<Ts...> ts, Xs... xs) const {
    return std::get<0>(ts).value(xs...) < std::get<1>(ts).value(xs...);
  }
};
struct op_gt {
  // bool value(real l, real r) const { return l > r; }

  template <typename... Ts, typename... Xs>
  bool value(std::tuple<Ts...> ts, Xs... xs) const {
    return std::get<0>(ts).value(xs...) > std::get<1>(ts).value(xs...);
  }
};

template <typename... Xs, typename T>
real compose(std::tuple<Xs...> xs, T t) {
  auto apply = [&](Xs... vs) { return t.value(vs...); };
  return std::apply(apply, xs);
}

template <typename... Xs, typename... Ts, typename T>
real compose(std::tuple<Xs...> xs, T t, Ts... ts) {
  return t.value(compose(xs, ts...));
}

template <typename... Xs, typename T>
real compose2(real r, T t) {
  return t.value(r);
}

template <typename... Xs, typename... Ts, typename T>
real compose2(real r, T t, Ts... ts) {
  return compose2(t.value(r), ts...);
}

template <typename... Xs, typename T, typename... Ts>
real compose2(std::tuple<Xs...> xs, T t, Ts... ts) {
  auto apply = [&](Xs... vs) { return t.value(vs...); };
  return compose2(std::apply(apply, xs), ts...);
}

struct op_compose {
  template <typename... Ts, typename... Xs>
  real value(std::tuple<Ts...> ts, Xs... xs) const {
    auto apply = [&](Ts... vs) {
      return compose2(std::make_tuple(xs...), vs...);
    };
    return std::apply(apply, ts);
  }
};

template <typename T, typename... Ts>
struct max_arity {
  static constexpr std::size_t arity =
      T::arity > max_arity<Ts...>::arity ? T::arity : max_arity<Ts...>::arity;
};

template <typename T>
struct max_arity<T> {
  static constexpr std::size_t arity = T::arity;
};

template <typename OP, typename... Ts>
struct expr {
  static constexpr std::size_t arity = max_arity<Ts...>::arity;
  expr(OP op, Ts... ts) : ts_(ts...), op_(op) {}
  std::tuple<Ts...> ts_;
  using type = std::tuple<Ts...>;
  using op_type = OP;
  OP op_;

  template <typename... Xs>
  real value(Xs... xs) const {
    static_assert(sizeof...(Xs) >= arity,
                  "trying to get index, but not enough arguments");
    // auto apply = [&](Ts... vs) { return op_.value(vs.value(xs...)...); };
    // return std::apply(apply, ts_);
    return op_.value(ts_, xs...);
  }

  template <typename... Xs>
  real operator()(Xs... xs) const {
    return value(xs...);
  }
};

template <typename T>
struct is_expr_s : std::bool_constant<false> {};
template <typename OP, typename... Ts>
struct is_expr_s<expr<OP, Ts...>> : std::bool_constant<true> {};
template <typename T>
constexpr bool is_expr_v = is_expr_s<T>::value;
template <typename T>
concept is_expr = is_expr_v<T>;

template <typename T>
concept is_value = is_expr<T> || is_constant<T> || is_var<T>;

template <typename OP, typename... Ts>
struct cexpr {
  static constexpr std::size_t arity = max_arity<Ts...>::arity;
  cexpr(OP op, Ts... ts) : ts_(ts...), op_(op) {}
  std::tuple<Ts...> ts_;
  using type = std::tuple<Ts...>;
  using op_type = OP;
  OP op_;

  template <typename... Xs>
  bool value(Xs... xs) const {
    static_assert(sizeof...(Xs) >= arity,
                  "trying to get index, but not enough arguments");
    // auto apply = [&](Ts... vs) { return op_.value(vs.value(xs...)...); };
    // return std::apply(apply, ts_);
    return op_.value(ts_, xs...);
  }
};

template <typename T>
struct is_cexpr_s : std::bool_constant<false> {};
template <typename OP, typename... Ts>
struct is_cexpr_s<cexpr<OP, Ts...>> : std::bool_constant<true> {};
template <typename T>
constexpr bool is_cexpr_v = is_cexpr_s<T>::value;
template <typename T>
concept is_cexpr = is_cexpr_v<T>;

template <typename C, typename E0, typename E1>
struct if_ {
  C c_;
  E0 e0_;
  E1 e1_;
  static constexpr std::size_t arity = max_arity<C, E0, E1>::arity;

  template <typename... Xs>
  real value(Xs... xs) const {
    static_assert(sizeof...(Xs) >= arity,
                  "trying to get index, but not enough arguments");
    return c_.value(xs...) ? e0_.value(xs...) : e1_.value(xs...);
  }
};

template <typename T>
struct is_cond_s : std::bool_constant<false> {};
template <typename C, typename E0, typename E1>
struct is_cond_s<if_<C, E0, E1>> : std::bool_constant<true> {};
template <typename T>
constexpr bool is_cond_v = is_cond_s<T>::value;
template <typename T>
concept is_cond = is_cond_v<T>;

template <typename T>
concept is_conditional = is_cond<T> || is_cexpr<T>;

auto if_else(is_conditional auto c, auto e0, auto e1) {
  return if_{c, e0, e1};
}

template <std::size_t N, typename OP, typename... Ts>
auto get_op(expr<OP, Ts...> ex) {
  return std::get<N>(ex.ts_);
}

template <typename T>
concept is_any_type = is_conditional<T> || is_value<T>;

auto operator-(is_value auto x) {
  return expr{op_neg(), x};
}
auto operator+(is_value auto x, is_value auto y) {
  return expr{op_add(), x, y};
}
auto operator-(is_value auto x, is_value auto y) {
  return expr{op_sub(), x, y};
}
auto operator*(is_value auto x, is_value auto y) {
  return expr{op_mul(), x, y};
}
auto operator/(is_value auto x, is_value auto y) {
  return expr{op_div(), x, y};
}
auto operator<(is_value auto x, is_value auto y) {
  return cexpr{op_lt(), x, y};
}
auto operator>(is_value auto x, is_value auto y) {
  return cexpr{op_gt(), x, y};
}
auto operator>>(is_value auto x, is_value auto y) {
  static_assert(decltype(y)::arity == 1,
                "cannot compose function of multiple variables");
  return expr{op_compose(), x, y};
}

auto operator-(is_cond auto x) {
  return if_{x.c_, -x.e0_, -x.e1_};
}
auto operator+(is_cond auto x, is_value auto y) {
  return if_{x.c_, x.e0_ + y, x.e1_ + y};
}
auto operator-(is_cond auto x, is_value auto y) {
  return if_{x.c_, x.e0_ - y, x.e1_ - y};
}
auto operator*(is_cond auto x, is_value auto y) {
  return if_{x.c_, x.e0_ * y, x.e1_ * y};
}
auto operator/(is_cond auto x, is_value auto y) {
  return if_{x.c_, x.e0_ / y, x.e1_ / y};
}
auto operator<(is_cond auto x, is_value auto y) {
  return if_{x.c_, x.e0_ < y, x.e1_ < y};
}
auto operator>(is_cond auto x, is_value auto y) {
  return if_{x.c_, x.e0_ > y, x.e1_ > y};
}
auto operator+(is_value auto y, is_cond auto x) {
  return if_{x.c_, y + x.e0_, y + x.e1_};
}
auto operator-(is_value auto y, is_cond auto x) {
  return if_{x.c_, y - x.e0_, y - x.e1_};
}
auto operator*(is_value auto y, is_cond auto x) {
  return if_{x.c_, y * x.e0_, y * x.e1_};
}
auto operator/(is_value auto y, is_cond auto x) {
  return if_{x.c_, y / x.e0_, y / x.e1_};
}
auto operator<(is_value auto y, is_cond auto x) {
  return if_{x.c_, y < x.e0_, y < x.e1_};
}
auto operator>(is_value auto y, is_cond auto x) {
  return if_{x.c_, y > x.e0_, y > x.e1_};
}

auto operator+(is_cond auto x, is_cond auto y) {
  return if_{x.c_, if_{y.c_, x.e0_ + y.e0_, x.e0_ + y.e1_},
             if_{y.c_, x.e1_ + y.e0_, x.e1_ + y.e1_}};
}
auto operator-(is_cond auto x, is_cond auto y) {
  return if_{x.c_, if_{y.c_, x.e0_ - y.e0_, x.e0_ - y.e1_},
             if_{y.c_, x.e1_ - y.e0_, x.e1_ - y.e1_}};
}
auto operator*(is_cond auto x, is_cond auto y) {
  return if_{x.c_, if_{y.c_, x.e0_ * y.e0_, x.e0_ * y.e1_},
             if_{y.c_, x.e1_ * y.e0_, x.e1_ * y.e1_}};
}
auto operator/(is_cond auto x, is_cond auto y) {
  return if_{x.c_, if_{y.c_, x.e0_ / y.e0_, x.e0_ / y.e1_},
             if_{y.c_, x.e1_ / y.e0_, x.e1_ / y.e1_}};
}
auto operator<(is_cond auto x, is_cond auto y) {
  return if_{x.c_, if_{y.c_, x.e0_ < y.e0_, x.e0_ < y.e1_},
             if_{y.c_, x.e1_ < y.e0_, x.e1_ < y.e1_}};
}
auto operator>(is_cond auto x, is_cond auto y) {
  return if_{x.c_, if_{y.c_, x.e0_ > y.e0_, x.e0_ > y.e1_},
             if_{y.c_, x.e1_ > y.e0_, x.e1_ > y.e1_}};
}

auto operator+(is_value auto x, real y) {
  return expr{op_add(), x, constant(y)};
}
auto operator-(is_value auto x, real y) {
  return expr{op_sub(), x, constant(y)};
}
auto operator*(is_value auto x, real y) {
  return expr{op_mul(), x, constant(y)};
}
auto operator/(is_value auto x, real y) {
  return expr{op_div(), x, constant(y)};
}
auto operator<(is_value auto x, real y) {
  return cexpr{op_lt(), x, constant(y)};
}
auto operator>(is_value auto x, real y) {
  return cexpr{op_gt(), x, constant(y)};
}

auto operator+(real x, is_value auto y) {
  return expr{op_add(), constant(x), y};
}
auto operator-(real x, is_value auto y) {
  return expr{op_sub(), constant(x), y};
}
auto operator*(real x, is_value auto y) {
  return expr{op_mul(), constant(x), y};
}
auto operator/(real x, is_value auto y) {
  return expr{op_div(), constant(x), y};
}
auto operator<(real x, is_value auto y) {
  return cexpr{op_lt(), constant(x), y};
}
auto operator>(real x, is_value auto y) {
  return cexpr{op_gt(), constant(x), y};
}
//
// template <typename X, typename A, typename B>
// struct repl_helper {
//  using type = std::conditional_t<std::is_same_v<X, A>, B, A>;
//};
//
// template <typename X, typename A, typename B>
// using repl_helper_t = typename repl_helper<X, A, B>::type;
//
//// replace X in A with B
// template <typename X, typename A, typename B>
// struct replace;
//
// template <typename X, typename A, typename B>
// using replace_t = typename replace<X, A, B>::type;
//
// template <is_var X, is_var A, is_value B>
// struct replace<X, A, B> {
//   using type = repl_helper_t<X, A, B>;
// };
//
// template <is_var X, typename OP, typename E0, is_value B>
// struct replace<X, expr<OP, E0>, B> {
//   using type = expr<OP, replace_t<X, E0, B>>;
// };
//
// template <is_var X, typename OP, typename E0, typename E1, is_value B>
// struct replace<X, expr<OP, E0, E1>, B> {
//   using type = expr<OP, replace_t<X, E0, B>, replace_t<X, E1, B>>;
// };

template <is_var X, is_value A, is_value B>
auto substitute(X x, A a, B b) {
  if constexpr (std::is_same_v<X, A>) {
    return b;
  } else if constexpr (is_expr_v<A>) {
    auto apply = [&](auto... vs) {
      return expr{a.op_, substitute(x, vs, b)...};
    };
    return std::apply(apply, a.ts_);
  } else {
    return a;
  }
}

template <std::size_t N>
auto d_wrt(var<N> e, var<N> v) {
  return constant{1.0};
}

template <std::size_t N>
auto d_wrt(constant e, var<N> v) {
  return constant{0.0};
}

template <std::size_t N, std::size_t M>
auto d_wrt(var<N> e, var<M> v) {
  return constant{0.0};
}

template <typename T0, std::size_t N>
auto d_wrt(expr<op_neg, T0> ex, var<N> v) {
  return -d_wrt(get_op<0>(ex), v);
}

template <typename T0, typename T1, std::size_t N>
auto d_wrt(expr<op_add, T0, T1> ex, var<N> v) {
  return d_wrt(get_op<0>(ex), v) + d_wrt(get_op<1>(ex), v);
}

template <typename T0, typename T1, std::size_t N>
auto d_wrt(expr<op_sub, T0, T1> ex, var<N> v) {
  return d_wrt(get_op<0>(ex), v) - d_wrt(get_op<1>(ex), v);
}

template <typename T0, typename T1, std::size_t N>
auto d_wrt(expr<op_mul, T0, T1> ex, var<N> v) {
  return (d_wrt(get_op<0>(ex), v) * get_op<1>(ex)) +
         (get_op<0>(ex) * d_wrt(get_op<1>(ex), v));
}

template <typename T0, typename T1, std::size_t N>
auto d_wrt(expr<op_div, T0, T1> ex, var<N> v) {
  auto f = get_op<0>(ex);
  auto g = get_op<1>(ex);
  auto f_p = d_wrt(f, v);
  auto g_p = d_wrt(g, v);
  return ((f_p * g) - (g_p * f)) / (g * g);
}

auto d_wrt(is_cond auto c, is_var auto v) {
  return if_{c.c_, d_wrt(c.e0_, v), d_wrt(c.e1_, v)};
}

constexpr std::size_t arity(is_any_type auto x) {
  return decltype(x)::arity;
}
}  // namespace detail

using detail::constant;
// using detail::d_wrt;
using detail::real;
using detail::var;

}  // namespace compile_time_diff
}  // namespace aks

#endif
