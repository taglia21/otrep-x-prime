from src.core import doctor as doctor_mod


def test_doctor_broker_state_skips_when_env_missing(monkeypatch, capsys):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)

    rc = doctor_mod.main(["--check-broker-state"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "broker_state=SKIP" in out
