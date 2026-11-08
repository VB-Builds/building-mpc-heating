# Point list (generic MQTT / BACnet-style)

A controller like this needs a small, boring set of points from a building automation system. This list is generic: map
the names to whatever your MQTT broker topics or BACnet objects are called. Nothing here is tied to a vendor.

The planner itself is stateless (`POST /plan` or `bmpc plan request.json`): the integration layer reads the points,
calls it once per hour, and writes the returned set-points back.

## Inputs (read)

| Point | Unit | MQTT topic (example) | BACnet object (example) | Used for |
|---|---|---|---|---|
| Room air temperature, per zone | degC | `site/bldg/zone/<z>/temp_c` | Analog Input, present-value | state estimate, comfort check |
| Outdoor air temperature | degC | `site/bldg/weather/temp_c` | Analog Input | forecast correction |
| Heat meter, per zone loop (or total) | kW | `site/bldg/zone/<z>/heat_kw` | Analog Input | estimator update |
| Heat pump electric power | kW | `site/bldg/hp/power_kw` | Analog Input | cost and peak reporting |
| Occupancy / schedule flag, per zone | 0/1 | `site/bldg/zone/<z>/occupied` | Binary Input or Schedule | comfort band selection |
| Weather forecast: temperature, 24 values | degC | from a forecast service | n/a | plan |
| Weather forecast: global irradiance, 24 values | W/m2 | from a forecast service | n/a | plan |
| Tariff, next 24 h | CHF/kWh | from the utility or a static schedule | n/a | plan |

## Outputs (write)

| Point | Unit | MQTT topic (example) | BACnet object (example) | Notes |
|---|---|---|---|---|
| Heat demand set-point, per zone | kW | `site/bldg/zone/<z>/cmd/heat_kw` | Analog Value, commandable | held for the hour |
| Supply temperature set-point | degC | `site/bldg/hp/cmd/supply_temp_c` | Analog Value, commandable | the `supply_temp_c` field of the plan |
| Controller heartbeat | ISO time | `site/bldg/mpc/heartbeat` | Analog Value | alarm if older than 2 h |
| Controller status | text | `site/bldg/mpc/status` | Character String | `optimal`, or the error text |

## Fallback

If the planner is unreachable, the heartbeat goes stale, or a room sensor reads outside 5..35 degC, the BMS must fall
back to its own heating curve and thermostat. The baseline in this repo is exactly that logic.

## REST request (`POST /plan`)

```json
{
  "room_temp_c": [20.1, 18.4, 21.0],
  "envelope_temp_c": [19.0, 18.0, 19.5],
  "outdoor_temp_c": [1.0, 0.5, 0.2],
  "solar_w_m2": [0, 0, 0],
  "internal_gains_kw": [[0.05, 0.05, 0.05], [0.15, 0.15, 0.15], [0.05, 0.05, 0.05]],
  "comfort_low_c": [[20.5, 20.5, 20.5], [18, 18, 18], [16, 16, 21]],
  "comfort_high_c": 27,
  "price_chf_kwh": [0.18, 0.18, 0.30]
}
```

* `outdoor_temp_c` sets the horizon (2 to 72 hourly values). All other series use the same length.
* `comfort_low_c` / `comfort_high_c` accept a scalar, one value per zone, or a zones x hours matrix. They bound the room
  temperature at the **end** of each hour step. Pass the stricter of the current and the next hour's band, as the
  simulator does.
* `envelope_temp_c` is optional (defaults to the room temperature). Unknown in practice: carry the estimate between
  calls, or accept a few hours of mis-estimate after a restart.
* Response: `heat_kw` per zone for the coming hour, `supply_temp_c`, the predicted room temperatures, and `solve_ms`.
