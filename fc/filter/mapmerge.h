#ifndef FC_FILTER_MAPMERGE_H
#define FC_FILTER_MAPMERGE_H

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
		this->unregisterAsSuccessor(m_predecessors.unguarded());
		this->registerAsSuccessor(predecessors);
		m_predecessors = predecessors;
	}

	// DataFilter interface
public:
	virtual ndim::sizes<0> prepare(PreparationProgress &progress) const override
	{
		auto predecessors = this->predecessors();
		progress.throwIfCancelled();
		for (const auto &predecessor : predecessors)
			hlp::throwIfNull(predecessor.get())->prepare(progress);
		return ndim::makeSizes();
	}
	virtual ndim::Container<QMap<KeyType, ValueType>> getData(ValidationProgress &progress, ndim::Container<QMap<KeyType, ValueType>> *recycle) const override
	{
		auto predecessors = this->predecessors();
		progress.throwIfCancelled();
		ndim::Container<QMap<KeyType, ValueType>> result = ndim::makeMutableContainer(recycle);
		QMap<KeyType, ValueType> &resultMap = result.mutableData().first();
		if (predecessors.isEmpty()) {
			resultMap.clear();
			return result;
		}
		QSet<KeyType> ambiguousKeys;
		fc::getData(progress, predecessors.first(), resultMap);
		for (auto iterator = predecessors.constBegin() + 1, end = predecessors.constEnd(); iterator != end; ++iterator) {
			const auto &predecessor = *iterator;
			QMap<KeyType, ValueType> mergeMap;
			fc::getData(progress, predecessor, mergeMap);
			for (auto iterator = mergeMap.constBegin(), end = mergeMap.constEnd(); iterator != end; ++iterator) {
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

#endif // FC_FILTER_MAPMERGE_H
