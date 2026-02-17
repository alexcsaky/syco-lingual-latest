from src.cost import estimate_cost


class TestEstimateCost:
    def test_basic_cost(self):
        cost = estimate_cost(
            input_tokens=1000,
            output_tokens=500,
            input_rate=3.0,   # per million tokens
            output_rate=15.0,
        )
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert abs(cost - expected) < 1e-10

    def test_zero_tokens(self):
        cost = estimate_cost(0, 0, 3.0, 15.0)
        assert cost == 0.0
