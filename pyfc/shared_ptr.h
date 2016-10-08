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

template <typename Base>
struct _FinalPtr : Base {
	using type = typename Base::type;

	std::shared_ptr<type> ptr;

	virtual type *get() const override
	{
		return ptr.get();
	}

	virtual operator std::shared_ptr<type>() const override
	{
		return ptr;
	}

	_FinalPtr(std::shared_ptr<type> ptr)
		: ptr(std::move(ptr))
	{
	}
	_FinalPtr(const _FinalPtr<Base> &) = default;
	_FinalPtr(_FinalPtr<Base> &&) = default;
	virtual ~_FinalPtr() = default;
};

template <typename Type, typename Base = void>
struct InheritPtr : Base {
	using type = Type;
	using base_type = typename Base::type;

	virtual ~InheritPtr() = default;

	virtual type *get() const override = 0;

	virtual operator std::shared_ptr<type>() const = 0;

	virtual operator std::shared_ptr<base_type>() const override
	{
		return this->operator std::shared_ptr<type>();
	}

	type *operator->() const
	{
		return get();
	}
	type &operator*() const
	{
		return *get();
	}

	static InheritPtr<Type, Base> *create(std::shared_ptr<type> ptr)
	{
		return new _FinalPtr<InheritPtr<Type, Base>>(std::move(ptr));
	}
};

template <typename Type>
struct InheritPtr<Type, void> {
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

	static InheritPtr<Type, void> *create(std::shared_ptr<type> ptr)
	{
		return new _FinalPtr<InheritPtr<Type, void>>(std::move(ptr));
	}
};

} // namespace pyfc

#endif // PYFC_SHARED_PTR_H
