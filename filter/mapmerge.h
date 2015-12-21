#ifndef FILTER_MAPMERGE_H
#define FILTER_MAPMERGE_H

#include "filter/filter.h"
#include "filter/filterbase.h"

#include "helper/helper.h"
#include "helper/threadsafe.h"

#include <QVector>
#include <QMap>
#include <QSet>

namespace filter
{

template <typename _KeyType, typename _ValueType>
class MapMerge : public PredecessorVectorFilterBase<DefaultFilterTypeTraits<QMap<_KeyType, _ValueType>>>,
				 public virtual NoConstDataFilter<DefaultFilterTypeTraits<QMap<_KeyType, _ValueType>>>
{
public:
	using KeyType = _KeyType;
	using ValueType = _ValueType;

	// DataFilter interface
public:
	virtual void prepare(AsyncProgress &progress, DurationCounter &counter, MetaDummy<QMap<KeyType, ValueType>> &meta) const override
	{
		auto predecessors = this->predecessors();
		progress.throwIfCancelled();
		for (const auto &predecessor : predecessors)
			hlp::notNull(predecessor.get())->prepare(progress, counter, meta);
	}
	virtual void getData(ValidationProgress &progress, QMap<KeyType, ValueType> data) const override
	{
		auto predecessors = this->predecessors();
		progress.throwIfCancelled();
		data.clear();
		if (predecessors.isEmpty())
			return;
		QSet<KeyType> ambiguousKeys;
		QMap<KeyType, ValueType> resultMap;
		hlp::notNull(predecessors.at(0).get())->getData(progress, resultMap);
		for (auto iterator = predecessors.cbegin() + 1, end = predecessors.cend(); iterator != end; ++iterator) {
			auto predecessor = jfh::notNull(iterator->get());
			QMap<KeyType, ValueType> mergeMap;
			predecessor->getData(progress, mergeMap);
			for (const auto &item : mergeMap) {
				if (ambiguousKeys.contains(item.key))
					continue;
				else if (resultMap.value(item.key) == item.value)
					continue;
				ambiguousKeys.insert(item.key);
				resultMap.remove(item.key);
			}
		}
		data = resultMap;
	}
};

} // namespace filter

#endif // FILTER_MAPMERGE_H
