#ifndef FILTER_FS_MAPMERGE_H
#define FILTER_FS_MAPMERGE_H

#include <QVector>
#include <QMap>
#include <QSet>

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "fc/gethelper.h"

#include "helper/helper.h"
#include "helper/threadsafe.h"

namespace fc
{
namespace filter
{

template <typename _KeyType, typename _ValueType>
class MapMerge : public FilterBase, public virtual DataFilter<QMap<_KeyType, _ValueType>>
{
public:
	using KeyType = _KeyType;
	using ValueType = _ValueType;

	hlp::Threadsafe<QVector<std::shared_ptr<const DataFilter<QMap<KeyType, ValueType>>>>> m_predecessors;

public:
	QVector<std::shared_ptr<const DataFilter<QMap<KeyType, ValueType>>>> predecessors() const
	{
		return m_predecessors.get();
	}

	void setPredecessors(const QVector<std::shared_ptr<const DataFilter<QMap<KeyType, ValueType>>>> &predecessors)
	{
		this->invalidate();
		this->unregisterSuccessor(m_predecessors.unguarded());
		this->registerSuccessor(predecessors);
		m_predecessors = predecessors;
	}

	// DataFilter interface
public:
	virtual ndim::sizes<0> prepare(PreparationProgress &progress) const override
	{
		auto predecessors = this->predecessors();
		progress.throwIfCancelled();
		for (const auto &predecessor : predecessors)
			hlp::notNull(predecessor.get())->prepare(progress);
		return ndim::makeSizes();
	}
	virtual Container<QMap<KeyType, ValueType>> getData(ValidationProgress &progress, Container<QMap<KeyType, ValueType>> *recycle) const override
	{
		auto predecessors = this->predecessors();
		progress.throwIfCancelled();
		Container<QMap<KeyType, ValueType>> result = fc::makeMutableContainer(recycle);
		QMap<KeyType, ValueType> &resultMap = result.mutableData().first();
		if (predecessors.isEmpty()) {
			resultMap.clear();
			return result;
		}
		QSet<KeyType> ambiguousKeys;
		fc::getData(progress, predecessors.first(), resultMap);
		for (auto iterator = predecessors.cbegin() + 1, end = predecessors.cend(); iterator != end; ++iterator) {
			const auto &predecessor = *iterator;
			QMap<KeyType, ValueType> mergeMap;
			fc::getData(progress, predecessor, mergeMap);
			for (auto iterator = mergeMap.cbegin(), end = mergeMap.cend(); iterator != end; ++iterator) {
				if (ambiguousKeys.contains(iterator.key()))
					continue;
				else if (resultMap.value(iterator.key()) == iterator.value())
					continue;
				ambiguousKeys.insert(iterator.key());
				resultMap.remove(iterator.key());
			}
		}
		return result;
	}
};

template <typename KeyType, typename ValueType>
std::shared_ptr<MapMerge<KeyType, ValueType>> makeMapMerge(const QVector<std::shared_ptr<const DataFilter<QMap<KeyType, ValueType>>>> &predecessors)
{
	auto filter = std::make_shared<MapMerge<KeyType, ValueType>>();
	filter->setPredecessors(predecessors);
	return filter;
}

} // namespace filter
} // namespace fc

#endif // FILTER_FS_MAPMERGE_H
