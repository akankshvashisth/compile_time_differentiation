#ifndef __variadic_arg_helpers_hpp__
#define __variadic_arg_helpers_hpp__

namespace aks
{
	namespace variadic_arg_helpers
	{
		template<typename binary_op>
		struct reduce
		{
			template<typename T, typename... Ts>
			static T apply(T x, Ts... xs)
			{
				return binary_op::apply(x, reduce<binary_op>::apply(xs...));
			}

			template<typename T>
			static T apply(T x)
			{
				return x;
			}
		};

		struct product
		{
			template<typename T>
			static T apply(T x, T y)
			{
				return x * y;
			}
		};

		struct add
		{
			template<typename T>
			static T apply(T x, T y)
			{
				return x + y;
			}
		};
	}
}

#endif // !__variadic_arg_helpers_hpp__