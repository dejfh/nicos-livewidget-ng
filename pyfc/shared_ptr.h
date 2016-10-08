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
 * class Base;
 * class DerivedA : Base;
 * class DerivedB : Base;
 * class DerivedB2 : DerivedB;
 *
 * // Usage:
 *
 * // Type definitions:
 * using PtrBase = InheritPtr<Base>;
 * using PtrA = InheritPtr<A, PtrBase>;
 * using PtrB = InheritPtr<B, PtrBase>;
 * using PtrB2 = InheritPtr<B2, PtrB>;
 *
 * // Creating:
 * PtrBase* base = PtrBase::create(...);
 * PtrA* a = PtrB2::create(...);
 * PtrB* b = PtrB2::create(...);
 * PtrB2* b2 = PtrB2::create(...);
 *
 * Allowed conversions:
 * base = a;
 * base = b;
 * base = b2;
 * b = b2;
 *
  * */

namespace pyfc
{

template <typename Type, typename... Bases>
struct InheritPtr;

template <typename Type>
struct InheritPtr<Type> {
	using type = Type;

	virtual ~InheritPtr() = default;

	virtual type *get() const = 0;

	virtual operator std::shared_ptr<type>() const = 0;

	type *operator->() const
	{
		return get();
	}
	type &operator*() const
	{
		return *get();
	}

	static InheritPtr<Type> *create(std::shared_ptr<Type>);
};

template <typename Type, typename Base, typename... MoreBases>
struct InheritPtr<Type, Base, MoreBases...> : Base, InheritPtr<Type, MoreBases...> {
	using type = Type;
	using base_type = typename Base::type;

	virtual ~InheritPtr() = default;

	virtual type *get() const = 0;

	virtual operator std::shared_ptr<type>() const = 0;

	type *operator->() const
	{
		return get();
	}
	type &operator*() const
	{
		return *get();
	}

	virtual operator std::shared_ptr<base_type>() const override
	{
		return this->operator std::shared_ptr<type>();
	}

	static InheritPtr<Type, Base, MoreBases...> *create(std::shared_ptr<Type>);
};

template <typename Type, typename... Bases>
struct FinalPtr : InheritPtr<Type, Bases...> {
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

	virtual operator std::shared_ptr<Type>() const override
	{
		return ptr;
	}
};

template <typename Type>
InheritPtr<Type> *InheritPtr<Type>::create(std::shared_ptr<Type> ptr)
{
	return new FinalPtr<Type>(std::move(ptr));
}

template <typename Type, typename Base, typename... MoreBases>
InheritPtr<Type, Base, MoreBases...> *InheritPtr<Type, Base, MoreBases...>::create(std::shared_ptr<Type> ptr)
{
	return new FinalPtr<Type, Base, MoreBases...>(std::move(ptr));
}

} // namespace pyfc

#endif // PYFC_SHARED_PTR_H
