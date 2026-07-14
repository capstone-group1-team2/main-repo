from agent.tools import cancel_order, track_order


def test_cancel_within_one_hour_is_free_no_fee():
    result = cancel_order("ORD-1001")
    assert result.outcome == "free_no_fee"
    assert result.fee_amount is None


def test_cancel_after_one_hour_before_shipped_is_free_pending():
    result = cancel_order("ORD-1002")
    assert result.outcome == "free_pending"
    assert result.fee_amount is None


def test_cancel_after_shipped_applies_10_percent_restocking_fee():
    result = cancel_order("ORD-1003")
    assert result.outcome == "fee_applied"
    assert result.fee_amount == round(120.0 * 0.10, 2)


def test_cancel_after_delivery_is_unavailable():
    result = cancel_order("ORD-1004")
    assert result.outcome == "unavailable_after_delivery"
    assert result.fee_amount is None


def test_cancel_already_cancelled_order():
    result = cancel_order("ORD-1005")
    assert result.outcome == "already_cancelled"


def test_cancel_order_not_found():
    result = cancel_order("ORD-9999")
    assert result.outcome == "not_found"


def test_track_order_found_with_standard_eta():
    result = track_order("ORD-1002")
    assert result.outcome == "found"
    assert result.eta == "3-5 business days"


def test_track_order_expedited_eta():
    result = track_order("ORD-1003")
    assert result.outcome == "found"
    assert result.eta == "1-2 business days"


def test_track_order_in_store_pickup_has_no_shipping_eta():
    result = track_order("ORD-1006")
    assert result.outcome == "found"
    assert result.eta is None


def test_track_order_delivered_status():
    result = track_order("ORD-1004")
    assert result.outcome == "found"
    assert result.status == "delivered"


def test_track_order_not_found():
    result = track_order("ORD-9999")
    assert result.outcome == "not_found"
