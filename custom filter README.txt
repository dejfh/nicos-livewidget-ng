Custom filters should inherit fc::FilterBase, to make them valid Predecessors and Successors.

Normally filters provide data by implementing the fc::DataFilter<ElementType, Dimensionality> interface.

Filters are NOT completly thread-safe. Modifications are only allowed from the main thread.
All const member-functions must be thread-safe. (That is good practice in general.)

The filter must guarantee, that a modification invalidates its successors and that the modification is visible in all succeeding calculations.

If parameters are changed after a validation has been triggered, the filter should garantee that the new values are not used in the ongoing validation.

- Call fc::FilterBase::invalidate() before a modification of parameter
- In fc::DataFilter::prepare(…) and  fc::DataFilter::getData(…): First read the parameter, then check if the progress has been canceled (call progress.throwIfCancelled() ).
