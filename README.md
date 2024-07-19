# WinGO - A Wingsail Optimisation Model

This project is ispired from the hull form optimisation model [ShipD/ShipGen](https://github.com/noahbagz/ShipGen) series, but for the modern vessels' wingsail design application. Due to time constraints, the dataset only considers wingsail with the NACA 4-digits series symmetrical airfoil profile.


## Wingsail Parameterisation Overview

*The table lists the definition of parameters for illustrating a NACA00XX-based wingsail geometry in the WinGeo module.*

| Parameter Sections | Serial | Description | Variable Name | Range | Unit / Scale | Type |
|:-----------------:|:----------------------:|:-----------:|:-------------:|:-----:|:----------------------:|:----:|
| Generally Sail Body | 0 | Overall Span | span | 5 - 80 | meter | float |
| | 1 | Clearance over Water | clearance_ow | 1 - 30 | meter | float |
| Bottom Key Section (BS) | 2 | Chord of BS | chord_bot | 0.25 - 25 | meter | float |
| | 3 | BS NACA Profile Code (the last two digits) | naca_bot | 06 - 25 | meter | int |
| Lower Key Section (LS) | 4 | Distance to Bottom for LS | height_low | 0.05 - 0.65 | Fraction of span | float |
| | 5 | Chord of LS | chord_low | 0.7 - 1.2 | Fraction of chord_bot | float |
| | 6 | LS NACA Profile Code (the last two digits) | naca_low | 06 - 25 | [-] | int |
| | 7 | Lengthways Offset of LS | offset_low | -0.2 - 0.2 | Fraction of chord_bot | float |
| Middle Key Section (MS) | 8 | Distance to Bottom for MS | height_mid | 0.35 - 0.85 | Fraction of span | float |
| | 9 | Chord of MS | chord_mid | 0.5 - 1.2 | Fraction of chord_bot | float |
| | 10 | MS NACA Profile Code (the last two digits) | naca_mid | 06 - 25 | [-] | int |
| | 11 | Lengthways Offset of MS | offset_mid | -0.2 - 0.4 | Fraction of chord_bot | float |
| Upper Key Section (US) | 12 | Distance to Bottom for US | height_upp | 0.80 - 0.97 | Fraction of span | float |
| | 13 | Chord of US | chord_upp | 0.3 - 1 | Fraction of chord_bot | float |
| | 14 | US NACA Profile Code (the last two digits) | naca_upp | 06 - 25 | [-] | int |
| | 15 | Lengthways Offset of US | offset_upp | 0 - 0.4 | Fraction of chord_bot | float |
| Tip Key Section (TS) | 16 | Chord of TS | chord_tip | 0.2 - 1 | Fraction of chord_bot | float |
| | 17 | TS NACA Profile Code (the last two digits) | naca_tip | 06 - 25 | [-] | int |
| | 18 | Lengthways Offset of TS | offset_tip | 0 - 0.5 | Fraction of chord_bot | float |



## Current functionalities:
- Generate wingsail .stl mesh with 19 control parameters.
- Calculate the generated thrust under given navigation conditions, with Xfoil.