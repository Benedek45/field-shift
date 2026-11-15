# What the field simulator is, and where it is weak

The simulator carries a field's state from year to year (soil water, mineral nitrogen, organic matter, pest pressure by plant family, erosion exposure) and runs each proposed rotation over the historical weather years until it settles into a steady cycle. Results are medians and 10th to 90th percentiles over those weather years, per hectare per year.

How far to trust each output:

- **Yield response to weather.** Checked against county yields of corn and soybean in 300 counties: the simulated year-to-year changes correlate about 0.5 with the observed ones. Losses in two of three major drought years were reproduced within a few points; one was missed almost entirely. For other crops the response is not checked.
- **Rotation effect on yield.** Not learned from data. Published penalties for repeating corn or soybean are applied; for other crop families a generic same-family penalty is used with low confidence. Benefits of longer, more diverse rotations enter only as low-confidence values from single experiments.
- **Water use.** A daily soil-water bucket with standard crop coefficients. Season totals are plausible, but year-to-year changes in water use barely match the satellite estimate. Rank rotations by water use only when the difference is large. No water table, capillary rise or runoff; irrigation is assumed to refill the soil whenever needed, and pumping cost and water limits are not priced.
- **Nitrate loss and nitrogen.** A monthly budget that is not validated against drainage measurements. The direction of a difference between rotations is more reliable than its size.
- **Winter covers.** Growth depends on the heat available between harvest and winter and in spring. The effect of a cover on the yield of the following crop is set to zero, which may be optimistic or pessimistic depending on termination timing and water.
- **Erosion.** Standard RUSLE with county-level slope and soil; a real field can differ several-fold from its county.
- **Organic matter.** One pool with a fixed yearly decay; trends are indicative only.
- **Income.** Recent reference prices and budget costs. The confidence flag in each result says which input drives the uncertainty. New equipment, labour peaks, storage, contracts and market access are not included.
- **Soil and slope** are county averages for cropland, not this field's own values.

Crops exist in the simulator only as trait records (water need, heat need, season length, nitrogen fixed, residue, plant family and so on). A crop that is not offered by `./farmsim options` cannot be simulated; you may still recommend it from the evidence, and say that it was not simulated.
