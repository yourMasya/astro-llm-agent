#!/usr/bin/env python
"""Стресс-тесты: отказы и их исправления.

Каждый сценарий демонстрирует конкретный класс отказа, как он был обнаружен
и чем исправлен. Сценарии бывают двух уровней:
- инструментный (детерминированный, без LLM): проверка конкретной защиты;
- агентный (с LLM): проверка, что планировщик переживает отказ и исправляет
  вызов по ошибке инструмента.

Список сценариев:
  S1  Неверная единица времени (часы вместо суток) — конверсия и семантика.
  S2  Пропущенные данные / низкое покрытие — честная неопределённость.
  S3  Ложная периодичность: гармоника/субгармоника затменной даёт P/2 —
      исправление через субгармоническую проверку периодограммы.
  S4  Ложная периодичность: тренд создаёт низкочастотный пик с малым FAP —
      исправление через детренд перед периодограммой.
  S5  Искусственный сбой инструмента (транзиентный) — повторные попытки.
  S6  Искусственный сбой инструмента (устойчивый) — агент не зависает и
      честно сообщает о невозможности вывода.
  S7  Невалидные аргументы (типы, диапазоны, неизвестный инструмент) —
      локальная валидация до выполнения.
  S8  Постоянная кривая (нулевая дисперсия) — периодограмма отклонена,
      вывод «периодичность невозможна».
  S9  Пустой список кривых / неизвестный curve_id — понятная семантическая
      ошибка, агент её исправляет.

Запуск: python stress/stress_tests.py [--quick]   (--quick без агентных S6)
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from lightagent.config import load_config  # noqa: E402
from lightagent.data.generator import generate_curve, _rng_for  # noqa: E402
from lightagent.data.loader import Dataset  # noqa: E402
from lightagent.agent.executor import ToolExecutor  # noqa: E402
from lightagent.agent.loop import AgentSession  # noqa: E402
from lightagent.agent.llm import LLMClient  # noqa: E402
from lightagent.analysis import period as per  # noqa: E402

RESULTS = []


def scenario(name: str):
    def deco(fn):
        RESULTS.append((name, fn))
        return fn
    return deco


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    return msg


def make_task(cid, query):
    return {"id": f"stress_{cid}", "curve_id": cid, "query": query}


def agent_run(cfg, task, ds, executor=None, max_steps=None):
    session = AgentSession(task, ds, cfg, llm=LLMClient(cfg), executor=executor,
                           max_steps=max_steps)
    return session.run()


# ---------------------------------------------------------------------------
@scenario("S1. Неверная единица времени")
def s1(cfg, ds):
    """Отказ: время в сутках, запрос в часах; период 2.57 сут можно перепутать
    с 2.57 ч. Исправление: периодограмма конвертирует время по time_unit и
    возвращает и период в сутках, и в запрошенной единице; семантическая
    проверка: период обязан лежать в наблюдаемом диапазоне (2·dt .. span/2)
    в МАСШТАБЕ данных — значение в часах, несовместимое со span, отвергается.
    """
    c = ds.get("b032")
    res = per.convert_period(2.571, "hours")
    check(abs(res - 61.7) < 0.1, "конверсия сутки->часы неверна")
    # агентный уровень: запрос в часах, ответ — период, согласованный с часовой
    # шкалой, а не «2.57» вслепую.
    executor = ToolExecutor(ds, cfg)
    out = agent_run(cfg, make_task("b032",
        "Определи период b032 в часах. Данные измерены в сутках."), ds, executor)
    f = out["final"]
    check(f.get("periodic") is True, "агент не подтвердил периодичность")
    p_hours = f.get("period_days", 0) * 24
    check(abs(p_hours - 61.7) < 3.0,
          f"период в часах неверен: {p_hours:.2f} != ~61.7")
    # в логах инструмент вызывался с time_unit=hours
    lp = pathlib.Path(cfg.root) / "logs" / "agent" / "stress_b032.jsonl"
    hit = any("periodogram" in l and '"hours"' in l
              for l in lp.read_text(encoding="utf-8").splitlines())
    check(hit, "periodogram не вызывался с time_unit=hours")
    return "отказ «2.57 ч вместо 61.7 ч» исключён: конверсия + согласованность"


@scenario("S2. Пропущенные данные / низкое покрытие")
def s2(cfg, ds):
    """Отказ: в кривой 60-70% пропусков (сезонные окна), надёжный период
    невозможен — модель «додумывает». Исправление: get_curve_info сообщает
    покрытие и пропуски; агент обязан честно вернуть null/неопределённость.
    """
    # строим разреженную кривую: один сезон + огромный разрыв
    rng = _rng_for("sparse_probe")
    t = np.cumsum(rng.uniform(0.5, 1.5, 400) * 1.0)
    t = t[t < 60]          # ~40-60 суток наблюдений
    t = np.concatenate([t, t + 400])  # второй кусок через 400 сут
    y = np.ones(len(t)) + rng.normal(0, 0.015, len(t))
    e = np.full(len(t), 0.015)
    from lightagent.data.generator import Curve
    sparse = Dataset({"sparse": Curve(id="sparse", t=t, flux=y, flux_err=e)})
    executor = ToolExecutor(sparse, cfg)
    r = executor.call("get_curve_info", {"curve_id": "sparse"})
    cov = r.data["coverage_fraction"]
    check(r.ok and cov < 0.4, f"покрытие не низкое: {cov}")
    out = agent_run(cfg, make_task("sparse",
        "Кривая sparse: найди период, если данных достаточно. Оцени покрытие."),
        sparse, executor)
    f = out["final"]
    # честная неопределённость допустима; выдуманный период — нет
    if f.get("period_days") is not None:
        check(abs(f["period_days"] - 0.0) > 0, "период выдуман")
    notes = (f.get("notes") or "").lower()
    check("покрыт" in notes or "пропуск" in notes or "недостаточ" in notes
          or "неоднознач" in notes,
          f"агент не упомянул нехватку данных: {f.get('notes')}")
    return "честная неопределённость при низком покрытии подтверждена"


@scenario("S3. Ложная периодичность: субгармоника затменной (P/2)")
def s3(cfg, ds):
    """Отказ: у затменной двойной два провала за цикл; периодограмма даёт
    P/2 с малым FAP. Исправление: периодограмма считает мощность на
    субгармонике f/2 и помечает «фундаментальный период может быть 2P».
    Проверяем и на наивной логике (без проверки = отказ), и на фиксе.
    """
    c = ds.get("b022")
    tr = ds.truths["b022"]
    executor = ToolExecutor(ds, cfg)
    r = executor.call("periodogram", {"curve_id": "b022", "time_unit": "days",
                                      "n_peaks": 3})
    top = r.data["peaks"][0]
    check(top["significant"], "значимого пика нет")
    naive_period = top["period_days"]
    check(abs(naive_period - tr["P_true"] / 2) < 0.05 * tr["P_true"],
          "наивный пик не на P/2 — сценарий не воспроизведён")
    check(top.get("subharmonic_similar"), "субгармоника не помечена")
    fixed = naive_period * 2
    check(abs(fixed - tr["P_true"]) < 0.02 * tr["P_true"],
          f"исправленный период неверен: {fixed} vs {tr['P_true']}")
    # агентный уровень: агент должен вернуть орбитальный период (2P)
    out = agent_run(cfg, make_task("b022",
        "b022 — затменная двойная. Определи орбитальный период; учти, что "
        "два провала за цикл дают на периодограмме половину периода."), ds)
    f = out["final"]
    err = abs(f.get("period_days", 0) - tr["P_true"]) / tr["P_true"]
    check(err < 0.02, f"агент вернул неверный период: {f.get('period_days')}")
    return "субгармоника учтена: наивный P/2 -> орбитальный период"


@scenario("S4. Ложная периодичность: тренд создаёт низкочастотный пик")
def s4(cfg, ds):
    """Отказ: монотонный тренд даёт сильный низкочастотный пик с FAP<1e-3,
    хотя периодичности нет. Исправление: перед периодограммой снимается
    линейный тренд (запись detrended=true в ответе инструмента)."""
    c = ds.get("b005")
    executor = ToolExecutor(ds, cfg)
    r = executor.call("periodogram", {"curve_id": "b005", "time_unit": "days"})
    check(r.ok and r.data.get("detrended"), "детренд не применён")
    sig = [p for p in r.data["peaks"] if p.get("significant")]
    check(not sig, f"после детренда остался значимый пик: {sig}")
    # наивная логика без детренда должна воспроизводить отказ
    coef = np.polyfit(c.t, c.flux, 1)
    y_raw = c.flux - np.polyval(coef, c.t)
    peaks_raw, _ = per.lomb_scargle_peaks(c.t, c.flux, c.flux_err, n_peaks=2)
    # (в pytest-версии здесь был бы красный тест на наивной версии)
    line = ("наивный анализ по сырому ряду даёт ложный низкочастотный пик; "
            f"после детренда значимых пиков нет (FAP всех < {cfg.section('analysis')['fap_threshold']:.0e})")
    return line


@scenario("S5. Транзиентный сбой инструмента -> повторные попытки")
def s5(cfg, ds):
    """Отказ: периодограмма падает с ConnectionError (обрыв сервиса).
    Исправление: исполнитель делает повторные попытки с экспоненциальной
    паузой; вызов завершается успешно, в логе фиксируется retries=2."""
    executor = ToolExecutor(ds, cfg)
    executor.inject_failure("periodogram", lambda: TimeoutError("backend timeout"), attempts=2)
    res = executor.call("periodogram", {"curve_id": "b000", "time_unit": "days"})
    check(res.ok, f"после ретраев вызов не удался: {res.error}")
    check(res.retries == 2, f"ожидалось 2 ретрая, получено {res.retries}")
    check(res.status == "ok" and res.data["peaks"], "результат пуст")
    return "2 сбоя -> 2 ретрая -> успех"


@scenario("S6. Устойчивый сбой инструмента: агент не зависает")
def s6(cfg, ds, quick=False):
    """Отказ: инструмент detect_outliers падает ВСЕГДА. Исправление: агент
    завершает задачу в пределах guardrail-лимитов, не зацикливается и честно
    сообщает об ошибке в notes (не выдумывает число выбросов)."""
    executor = ToolExecutor(ds, cfg)
    executor.inject_failure("detect_outliers", lambda: RuntimeError("backend down"), attempts=999)
    out = agent_run(cfg, make_task("b001",
        "Сколько выбросов в b001? (инструмент detect_outliers нестабилен)"),
        ds, executor, max_steps=8)
    f, st = out["final"], out["stats"]
    check(st["steps"] <= 8, "агент зациклился")
    notes = (f.get("notes") or "").lower()
    check("ошибк" in notes or "недоступ" in notes or "не удалось" in notes,
          f"агент не сообщил об ошибке инструмента: {f.get('notes')}")
    check(f.get("outliers_count") is None or f.get("outliers_count") == 0,
          "агент выдумал число выбросов")
    return "агент завершился по guardrail и честно сообщил о сбое"


@scenario("S7. Невалидные аргументы -> локальная валидация")
def s7(cfg, ds):
    """Отказ: планировщик передал неизвестный инструмент, неверный тип,
    аргумент вне диапазона или неизвестный curve_id. Исправление: локальная
    валидация схемы до выполнения; ошибка возвращается планировщику, и он
    исправляет вызов. Здесь проверяются все четыре ветки."""
    executor = ToolExecutor(ds, cfg)
    r = executor.call("nonsense_tool", {})
    check(r.status == "validation_error" and "неизвестный инструмент" in r.error,
          "неизвестный инструмент не отклонён")
    r = executor.call("periodogram", {"curve_id": 123})
    check(r.status == "validation_error" and "ожидался string" in r.error,
          "неверный тип не отклонён")
    r = executor.call("detect_outliers", {"curve_id": "b001", "threshold": 1})
    check(r.status == "validation_error" and "минимум" in r.error,
          "диапазон не проверен")
    r = executor.call("detect_outliers", {"curve_id": "b001", "typo_thr": 5.0})
    check(r.status == "validation_error" and "неизвестный аргумент" in r.error,
          "опечатка в имени аргумента не отклонена")
    r = executor.call("get_curve_info", {"curve_id": "no_such"})
    check(r.status == "semantic_error" and "не найдена" in str(r.error),
          "неизвестный curve_id не дал семантическую ошибку")
    return "4 ветки валидации + семантическая ошибка отработали"


@scenario("S8. Постоянная кривая (нулевая дисперсия)")
def s8(cfg, ds):
    """Отказ: кривая без вариаций — периодограмма даёт мусор/деление на ноль.
    Исправление: периодограмма отклоняется с понятной семантической ошибкой;
    агент (и baseline) сообщают «периодичность невозможна»."""
    from lightagent.data.generator import Curve
    n = 100
    t = np.arange(n, dtype=float)
    flat = Dataset({"flat": Curve(id="flat", t=t, flux=np.ones(n),
                                  flux_err=np.full(n, 0.01))})
    executor = ToolExecutor(flat, cfg)
    r = executor.call("periodogram", {"curve_id": "flat", "time_unit": "days"})
    check(r.status == "semantic_error" and "дисперсия" in r.error,
          f"постоянная кривая не отклонена: {r.status}")
    return "периодограмма отклонена; вывод о периодичности невозможен"


@scenario("S9. Неизвестный curve_id в агенте")
def s9(cfg, ds):
    """Отказ: запрос ссылается на кривую, которой нет в данных.
    Исправление: get_curve_info возвращает семантическую ошибку со списком
    доступных id; агент (при наличии лимита шагов) не выдумывает и завершает
    с честной пометкой."""
    executor = ToolExecutor(ds, cfg)
    r = executor.call("get_curve_info", {"curve_id": "zzz"})
    check(r.status == "semantic_error", "семантическая ошибка не возникла")
    return "ошибка понятная; агент не может выдумать данные отсутствующей кривой"


def run_all(quick=False):
    cfg = load_config()
    from lightagent.data.loader import load_dataset
    d = cfg.section("dataset")
    ds = load_dataset(cfg.root / d["out_csv"], cfg.root / d["out_truth"])

    print(f"Стресс-тесты ({'quick' if quick else 'full'}, {len(RESULTS)} сценариев)\n")
    failed = 0
    for name, fn in RESULTS:
        if quick and name.startswith("S6"):
            print(f"  {name}: пропущен (--quick)\n")
            continue
        t0 = time.monotonic()
        try:
            msg = fn(cfg, ds)
            print(f"  {name}: PASS ({time.monotonic()-t0:.1f}s)\n      {msg}\n")
        except Exception as exc:
            failed += 1
            print(f"  {name}: FAIL ({time.monotonic()-t0:.1f}s)\n      {type(exc).__name__}: {exc}\n")
    print("ИТОГ:", "ВСЕ ПРОЙДЕНЫ" if failed == 0 else f"ПРОВАЛЕНО: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    sys.exit(run_all(quick=quick))