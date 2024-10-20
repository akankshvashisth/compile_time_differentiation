# compile_time_differentiation
Proof of concept using C++20 metaprogramming to do compile time differentiation of functions

For now only support simple operations like ( -,+,-,*,/,<,> )

## Setup

```
using X = aks::compile_time_diff::var<0>;
using Y = aks::compile_time_diff::var<1>;
using Z = aks::compile_time_diff::var<2>;
using W = aks::compile_time_diff::var<3>;
using C = aks::compile_time_diff::constant;
using real = aks::compile_time_diff::real;  // this is a double

auto x = X{};
auto y = Y{};
auto z = Z{};
auto w = W{};
```

## General usage

> To get the value of the function given the inputs, call value explicitly
```
  pp(x.value(2.0), "\n");                 // prints 2.0
  pp(y.value(2.0, 3.0), "\n");            // prints 3.0
  pp(z.value(2.0, 3.0, 4.0), "\n");       // prints 4.0
  pp(w.value(2.0, 3.0, 4.0, 5.0), "\n");  // prints 5.0
```

  > or use the function interface
```
  pp(x(2.0), "\n");                 // prints 2.0
  pp(y(2.0, 3.0), "\n");            // prints 3.0
  pp(z(2.0, 3.0, 4.0), "\n");       // prints 4.0
  pp(w(2.0, 3.0, 4.0, 5.0), "\n");  // prints 5.0
```

  > create functions from variables
```
  auto f0 = x * x;
  pp(f0.value(2.0), "\n");  // prints 4.0 using value explicitly
  pp(f0(2.0), "\n");        // prints 4.0 using function interface
```

  > use constants
```
  auto f0 = 2.0 * x;
  pp(f0.value(3.0), "\n");  // prints 6.0
```

  > the value can be called in place
```
  pp((x * x).value(2.0), "\n");  // prints 4.0
  pp((x * x)(2.0), "\n");        // prints 4.0

  auto f0 = x * y;               // y is var<1>, so it requires 2 arguments.
  pp(f0.value(2.0, 3.0), "\n");  // prints 6.0

  auto f0 = 3.0 * x / y;
  pp(f0.value(2.0, 3.0), "\n");  // prints 2.0

  auto f = (x * y) / (z + w);
  pp(f.value(3.0, 5.0, 1.0, 2.0), "\n");  // prints 5.0
  ```

  > functions can be built from functions;
  ```
  auto f0 = x * y;
  auto f1 = z + w;
  auto f = f0 / f1;
  pp(f.value(3.0, 5.0, 1.0, 2.0), "\n");  // prints 5.0
  ``` 
  
  > conditions can be added
  ```
  auto f = if_else(x > y, x, y);
  pp(f.value(3.0, 5.0), "\n");  // prints 5.0
  pp(f.value(5.0, 2.0), "\n");  // prints 5.0
  ```

> conditions can be used as part of other functions
  ```
  auto f = (x * y) + if_else(z < w, z * z, w * w);
  pp(f.value(3.0, 5.0, 1.0, 2.0), "\n");  // prints 16.0
  pp(f.value(3.0, 5.0, 3.0, 2.0), "\n");  // prints 19.0
  ```
  > the condition can be used to implement other sub functions
  ```
  auto zw_min = if_else(z < w, z, w);
  auto f = (x * y) + zw_min * zw_min;
  pp(f.value(3.0, 5.0, 1.0, 2.0), "\n");  // prints 16.0
  pp(f.value(3.0, 5.0, 3.0, 2.0), "\n");  // prints 19.0
  ```
> create multiple branches
```
  auto xyz_min = if_else(z < if_else(x < y, x, y), z, if_else(x < y, x, y));
  pp(xyz_min.value(3.0, 5.0, 1.0), "\n");  // prints 1.0
  pp(xyz_min.value(1.0, 5.0, 3.0), "\n");  // prints 1.0
  pp(xyz_min.value(5.0, 1.0, 3.0), "\n");  // prints 1.0
  ```
> functions of only `var<0>` or `x` can be composed using `>>`
```
  auto f = 2.0 + x;
  auto g = 3.0 * x;

  auto h = f >> g;  // calculate f and then pass value to g
  // so h = g ( f ( x ) )
  pp(h.value(2.0), "\n");  // prints 12.0
  ```

> substitute a variable in a function with constant
```
  auto f = x * y;
  auto g = substitute(y, f, C{8.0});  // substitute y in f with constant 8

  pp(f.value(2.0, 3.0), "\n");  // prints 6.0 - takes 2 args as y present
  pp(g.value(2.0), "\n");  // prints 16.0, takes only 1 arg as y not present
  ```
> substitute a variable in a function with another variable
```
  auto f = x * y;
  auto g = substitute(x, f, z);  // substitute x in f with z

  pp(f.value(2.0, 3.0),
	 "\n");  // prints 6.0 - takes only 2 args as no z yet
  pp(g.value(2.0, 3.0, 5.0), "\n");  // prints 15.0 - 3 args as z introduced
  ```
> substitute a variable in a function with another function
```
  auto f = x * y;
  auto g = substitute(x, f, y * y);  // substitute x in f with (y*y)
  pp(f.value(2.0, 3.0), "\n");       // prints 6.0
  pp(g.value(2.0, 3.0), "\n");       // prints 27.0
```

## Compile Time Derivatives

> take derivatives
  ```
  auto f = x * x * x;
  auto df_dx = d_wrt(f, x);    // derivate or f with respect to x
  pp(f.value(2.0), "\n");      // prints 8.0
  pp(df_dx.value(2.0), "\n");  // prints 12.0
```
  > multi variable derivatives
  ```
  auto f = x * y * z;
  auto df_dx = d_wrt(f, x);
  auto df_dy = d_wrt(f, y);
  auto df_dz = d_wrt(f, z);
  pp(f.value(2.0, 3.0, 4.0), "\n");      // prints 24.0
  pp(df_dx.value(2.0, 3.0, 4.0), "\n");  // prints 12.0
  pp(df_dy.value(2.0, 3.0, 4.0), "\n");  // prints 8.0
  pp(df_dz.value(2.0, 3.0, 4.0), "\n");  // print 6.0
```
  > take higher order derivatives
  ```
  auto f = x * x * x;
  auto df_dx = d_wrt(f, x);
  auto d2f_dx2 = d_wrt(df_dx, x);
  pp(f.value(5.0), "\n");        // prints 125.0
  pp(df_dx.value(5.0), "\n");    // prints 75.0
  pp(d2f_dx2.value(5.0), "\n");  // prints 30.0
```
  > take higher order derivatives with multiple variables
  ```
  auto f = x * x * y * y * z * z;
  auto df_dx = d_wrt(f, x);
  auto d2f_dxdy = d_wrt(df_dx, y);
  auto d3f_dxdydz = d_wrt(d2f_dxdy, z);
  pp(f.value(5.0, 3.0, 4.0), "\n");           // prints 3600.0
  pp(df_dx.value(5.0, 3.0, 4.0), "\n");       // prints 1440.0
  pp(d2f_dxdy.value(5.0, 3.0, 4.0), "\n");    // prints 960.0
  pp(d3f_dxdydz.value(5.0, 3.0, 4.0), "\n");  // prints 480.0
  ```

## Example

> a simple newton solver

```
auto solve = [x](auto f, real guess) {
  auto newton_step = -f / d_wrt(f, x);

  for (int i = 0; i < 10; i++) {
	auto dx = newton_step(guess);
	guess = guess + dx;
	if (std::abs(dx) < 1e-8)
	  break;
  }
  return guess;
};

{
  pp(solve(x * x - 9.0, 2.0), "\n");       // prints 3.0
  pp(solve(x * x * x - 27.0, 2.0), "\n");  // prints 3.0
}
```

**See `compile_time_differentiation.cpp` for more details**
