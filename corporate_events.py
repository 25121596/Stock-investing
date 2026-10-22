"""Canonicalize duplicate versions of one delivered corporate action.

Raw provider files remain immutable. Metadata such as plan notices or tax
descriptions must not turn a single cash/share delivery into two entitlements.
"""
from collections import defaultdict
from decimal import Decimal, InvalidOperation


def numeric(value):
    value = str(value or '').strip()
    try:
        return format(Decimal(value or '0').normalize(), 'f')
    except InvalidOperation as exc:
        raise ValueError(f'Unresolved corporate-action amount: {value}') from exc


def canonicalize_events(events):
    groups = defaultdict(list)
    for i, event in enumerate(events):
        key = (str(event.get('code', '')), str(event.get('dividRegistDate', '') or ''),
            str(event.get('dividOperateDate', '') or ''), numeric(event.get('dividCashPsBeforeTax')),
            numeric(event.get('dividStocksPs')), numeric(event.get('dividReserveToStockPs')))
        groups[key].append((i, dict(event)))
    result, duplicates, incomplete = [], [], []
    for key, versions in groups.items():
        if len(versions) > 1 and (not key[1] or not key[2]):
            result.extend(event for _, event in versions)
            incomplete.append({'identity': list(key), 'rows': len(versions), 'issue': 'missing_identity_dates_not_deduplicated'})
            continue
        for field in ('dividPayDate', 'dividStockMarketDate'):
            values = {str(event.get(field, '') or '') for _, event in versions} - {''}
            if len(values) > 1:
                raise ValueError(f'Conflicting delivery dates for {key}: {field}={values}')
        best = max(versions, key=lambda item: sum(bool(v) for v in item[1].values()))[1].copy()
        for _, version in versions:
            for field, value in version.items():
                if not best.get(field) and value:
                    best[field] = value
        result.append(best)
        if len(versions) > 1:
            duplicates.append({'identity': list(key), 'raw_rows': len(versions),
                'removed_rows': len(versions) - 1, 'source_indices': [i for i, _ in versions],
                'retained_event': best})
    result.sort(key=lambda event: (str(event.get('dividOperateDate', '')), str(event.get('code', ''))))
    return result, {'input_rows': len(events), 'output_rows': len(result),
        'removed_rows': len(events) - len(result), 'duplicate_event_groups': duplicates,
        'unresolved_incomplete_groups': incomplete,
        'definition': 'one code/record-date/ex-date/cash-before-tax/share-ratio delivery; conflicts not silently merged'}
