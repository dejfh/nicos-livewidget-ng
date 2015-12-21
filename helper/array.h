#ifndef HELPER_ARRAY_H
#define HELPER_ARRAY_H

#include <array>
#include <algorithm>

namespace hlp
{

namespace array
{

template <size_t Dimensionality>
using Indices = std::array<size_t, Dimensionality>;

template <size_t SizeTotal, size_t SizeIn>
Indices<SizeTotal - SizeIn> invertSelection(Indices<SizeIn> selection)
{
	Indices<SizeTotal - SizeIn> result;
	std::sort(selection.begin(), selection.end());

	size_t nextSelection = 0;
	auto itIn = selection.cbegin();
	auto itOut = result.begin();

	for (; itIn != selection.cend(); ++nextSelection)
		if (*itIn != nextSelection)
			*itOut++ = nextSelection;
		else
			++itIn;

	for (; itOut != result.end(); ++itOut)
		*itOut = nextSelection++;

	return result;
}

template <size_t Size>
std::array<size_t, Size> invertReorder(Indices<Size> indices)
{
	Indices<Size> result;
	for (size_t i = 0; i < Size; ++i)
		result[indices[i]] = i;
	return result;
}

template <typename T, size_t SizeIn, size_t SizeOut>
std::array<T, SizeOut> select(const std::array<T, SizeIn> &array, Indices<SizeOut> selected)
{
	std::array<T, SizeOut> result;

	auto itOut = result.begin();
	auto itIn = selected.cbegin(), endIn = selected.cend();
	for (; itIn != endIn; ++itIn, ++itOut)
		*itOut = array[*itIn];

	return result;
}

template <typename T, size_t SizeLeft, size_t SizeRight>
std::array<T, SizeLeft + SizeRight> cat(const std::array<T, SizeLeft> &left, const std::array<T, SizeRight> &right)
{
	std::array<T, SizeLeft + SizeRight> result;
	auto itOut = result.begin();

	for (auto itIn = left.cbegin(), endIn = left.cend(); itIn != endIn; ++itIn, ++itOut)
		*itOut = *itIn;
	for (auto itIn = right.cbegin(), endIn = right.cend(); itIn != endIn; ++itIn, ++itOut)
		*itOut = *itIn;

	return result;
}

template <typename T, size_t SizeLeft, size_t SizeRight>
std::array<T, SizeLeft + SizeRight> merge(
	const std::array<T, SizeLeft> &left, Indices<SizeLeft> indicesLeft, const std::array<T, SizeRight> &right, Indices<SizeRight> indicesRight)
{
	std::array<T, SizeLeft + SizeRight> result;

	{
		auto itIn = left.cbegin();
		auto itIndex = indicesLeft.cbegin(), endIndex = indicesLeft.cend();
		for (; itIndex != endIndex; ++itIndex, ++itIn)
			result[*itIndex] = *itIn;
	}
	{
		auto itIn = right.cbegin();
		auto itIndex = indicesRight.cbegin(), endIndex = indicesRight.cend();
		for (; itIndex != endIndex; ++itIndex, ++itIn)
			result[*itIndex] = *itIn;
	}

	return result;
}

template <typename T, size_t Size, size_t SizeSet>
void set(std::array<T, Size> &array, Indices<SizeSet> indices, const T &value = T())
{
	for (size_t index : indices)
		array[index] = value;
}

template <typename T, size_t SizeTarget, size_t SizeSource>
void set(std::array<T, SizeTarget> &target, const std::array<T, SizeSource> &source, Indices<SizeSource> indices)
{
	auto itIn = source.cbegin();
	auto itIndex = indices.cbegin(), endIndex = indices.cend();
	for (; itIndex != endIndex; ++itIndex, ++itIn) {
		target[*itIndex] = *itIn;
	}
}

template <size_t SizeAdd, typename T, size_t SizeIn>
std::array<T, SizeIn + SizeAdd> append(const std::array<T, SizeIn> &array, const T &value = T())
{
	std::array<T, SizeIn + SizeAdd> result;

	auto itOut = result.begin(), a = itOut + SizeIn, b = result.end();

	for (auto itIn = array.cbegin(); itOut != a; ++itOut, ++itIn)
		*itOut = *itIn;
	for (; itOut != b; ++itOut)
		*itOut = value;

	return result;
}

template <typename T, size_t SizeIn, size_t SizeRemove>
std::array<T, SizeIn - SizeRemove> remove(const std::array<T, SizeIn> &array, Indices<SizeRemove> toRemove)
{
	return select(array, invertSelection<SizeIn>(toRemove));
}

template <typename T, size_t SizeIn>
std::array<T, SizeIn - 1> remove(const std::array<T, SizeIn> &array, size_t toRemove)
{
	return select(array, invertSelection<SizeIn>(Indices<1>{toRemove}));
}

template <typename T, size_t SizeLeft, size_t SizeRight>
std::array<T, SizeLeft + SizeRight> insert(
	const std::array<T, SizeLeft> &left, const std::array<T, SizeRight> &right, Indices<SizeRight> indicesRight)
{
	return merge(left, invertSelection<SizeLeft + SizeRight>(indicesRight), right, indicesRight);
}

template <typename T, size_t SizeIn, size_t SizeInsert>
std::array<T, SizeIn + SizeInsert> insert(const std::array<T, SizeIn> &array, Indices<SizeInsert> indices, const T &value = T())
{
	std::array<T, SizeIn + SizeInsert> result;
	set(result, array, invertSelection<SizeIn + SizeInsert>(indices));
	set(result, indices, value);
	return result;
}

template <typename T, size_t SizeIn>
std::array<T, SizeIn + 1> insert(const std::array<T, SizeIn> &array, size_t index, const T &value = T())
{
	return insert(array, std::array<T, 1>{value}, Indices<1>{index});
}

} // namespace array

} // namespace hlp

#endif // HELPER_ARRAY_H
