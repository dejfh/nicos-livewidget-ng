#ifndef PYFC_SHARED_PTR_H
#define PYFC_SHARED_PTR_H

/*
 * SIP does not support converting a std::shared_ptr of a type to a std::shared_ptr of a base class of that type.
 * In SIPs point of view the shared_ptrs have no relationship to each other.
 * This is overcome by introducing wrapper classes with explicit inheritance.
 *
 * // Example:
 *
 * // Assuming:
 * class A;
 * class B1;
 * class B2 : B1;
 * class C : A, B2;
 *
 * // Usage:
 *
 * // Type definitions:
 * using PtrA = InheritPtr<A>;
 * using PtrB1 = InheritPtr<B1>;
 * using PtrB2 = InheritPtr<B1, PtrB1>;
 * using PtrC = InheritPtr<C, PtrA, PtrB2>;
 *
 * // Creating:
 * PtrA* a = PtrA::create(...);
 * PtrB1* b1 = PtrB1::create(...);
 * PtrB2* b2 = PtrB2::create(...);
 * PtrC* c = PtrC::create(...);
 *
 * Allowed conversions:
 * a = c;
 * b1 = b2;
 * b1 = c;
 * b2 = c;
 *
  * */

namespace pyfc
{

template <typename Type, typename... Bases>
struct InheritPtr : Bases... {
	virtual ~InheritPtr() = default;

	virtual Type *get() const = 0;

	virtual std::shared_ptr<volatile const void> getOwnership() const = 0;

	operator std::shared_ptr<Type>() const
	{
		return std::shared_ptr<Type>(getOwnership(), get());
	}

	Type *operator->() const
	{
		return get();
	}
	Type &operator*() const
	{
		return *get();
	}

	static InheritPtr<Type, Bases...> *create(std::shared_ptr<Type>);
};

template <typename Type, typename... Bases>
struct FinalPtr : InheritPtr<Type, Bases...> {
	using type = Type;
	virtual ~FinalPtr() = default;

	FinalPtr() = default;
	FinalPtr(const FinalPtr<Type, Bases...> &) = default;
	FinalPtr(FinalPtr<Type, Bases...> &&) = default;

	FinalPtr<Type, Bases...> &operator=(const FinalPtr<Type, Bases...> &) = default;
	FinalPtr<Type, Bases...> &operator=(FinalPtr<Type, Bases...> &&) = default;

	std::shared_ptr<Type> ptr;

	FinalPtr(std::shared_ptr<Type> ptr)
		: ptr(std::move(ptr))
	{
	}

	virtual Type *get() const override
	{
		return ptr.get();
	}

	virtual std::shared_ptr<volatile const void> getOwnership() const
	{
		return ptr;
	}
};

template <typename Type, typename... Bases>
InheritPtr<Type, Bases...> *InheritPtr<Type, Bases...>::create(std::shared_ptr<Type> ptr)
{
	return new FinalPtr<Type, Bases...>(std::move(ptr));
}

template <typename TFinalPtr, typename... TArgs>
TFinalPtr *make_final_ptr(TArgs &&... args)
{
	using Type = typename TFinalPtr::type;
	std::shared_ptr<Type> ptr = std::make_shared<Type>(std::forward<TArgs>(args)...);
	return new TFinalPtr(std::move(ptr));
}

} // namespace pyfc

#endif // PYFC_SHARED_PTR_H
